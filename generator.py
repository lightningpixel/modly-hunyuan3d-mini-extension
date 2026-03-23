"""
Reference : https://huggingface.co/tencent/Hunyuan3D-2mini
"""
import io
import os
import sys
import tempfile
import time
import threading
import uuid
import zipfile
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from services.generators.base import BaseGenerator, smooth_progress, GenerationCancelled

_HF_REPO_ID       = "tencent/Hunyuan3D-2mini"
_SUBFOLDER        = "hunyuan3d-dit-v2-mini"
_GITHUB_ZIP       = "https://github.com/Tencent/Hunyuan3D-2/archive/refs/heads/main.zip"
_PAINT_HF_REPO    = "tencent/Hunyuan3D-2"
_PAINT_SUBFOLDER  = "hunyuan3d-paint-v2-0-turbo"


class Hunyuan3DMiniGenerator(BaseGenerator):
    MODEL_ID     = "hunyuan3d-mini"
    DISPLAY_NAME = "Hunyuan3D 2 Mini"
    VRAM_GB      = 6

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def is_downloaded(self) -> bool:
        subfolder = self.download_check if self.download_check else _SUBFOLDER
        model_dir = self.model_dir / subfolder
        return model_dir.exists() and (model_dir / "model.fp16.safetensors").exists()

    def load(self) -> None:
        if self._model is not None:
            return

        if not self.is_downloaded():
            self._download_weights()

        self._ensure_hy3dgen()

        import torch
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype  = torch.float16 if device == "cuda" else torch.float32

        subfolder = self.download_check if self.download_check else _SUBFOLDER
        print(f"[Hunyuan3DMiniGenerator] Loading pipeline from {self.model_dir} (subfolder={subfolder})…")
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            str(self.model_dir),
            subfolder=subfolder,
            use_safetensors=True,
            device=device,
            dtype=dtype,
        )
        self._model = pipeline
        print(f"[Hunyuan3DMiniGenerator] Loaded on {device}.")

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def generate(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        import torch

        num_steps        = int(params.get("num_inference_steps", 30))
        vert_count       = int(params.get("vertex_count", 0))
        enable_texture   = bool(params.get("enable_texture", False))
        octree_res       = int(params.get("octree_resolution", 380))
        guidance_scale   = float(params.get("guidance_scale", 5.5))
        seed             = int(params.get("seed", -1))

        self._report(progress_cb, 5, "Removing background…")
        image = self._preprocess(image_bytes)
        self._check_cancelled(cancel_event)

        shape_end = 70 if enable_texture else 82
        self._report(progress_cb, 12, "Generating 3D shape…")
        stop_evt = threading.Event()
        if progress_cb:
            t = threading.Thread(
                target=smooth_progress,
                args=(progress_cb, 12, shape_end, "Generating 3D shape…", stop_evt),
                daemon=True,
            )
            t.start()

        try:
            with torch.no_grad():
                import torch
                generator = torch.Generator().manual_seed(seed) if seed >= 0 else None
                outputs = self._model(
                    image=image,
                    num_inference_steps=num_steps,
                    octree_resolution=octree_res,
                    guidance_scale=guidance_scale,
                    num_chunks=4000,
                    generator=generator,
                    output_type="trimesh",
                )
            mesh = outputs[0]
        finally:
            stop_evt.set()

        self._check_cancelled(cancel_event)

        if enable_texture:
            self._report(progress_cb, 72, "Freeing VRAM for texture model…")
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self._check_cancelled(cancel_event)
            mesh = self._run_texture(mesh, image, progress_cb)
        else:
            if vert_count > 0 and hasattr(mesh, "vertices") and len(mesh.vertices) > vert_count:
                self._report(progress_cb, 85, "Optimizing mesh…")
                mesh = self._decimate(mesh, vert_count)

        self._report(progress_cb, 96, "Exporting GLB…")
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.glb"
        path = self.outputs_dir / name
        mesh.export(str(path))

        self._report(progress_cb, 100, "Done")
        return path

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _preprocess(self, image_bytes: bytes) -> Image.Image:
        import rembg
        return rembg.remove(Image.open(io.BytesIO(image_bytes))).convert("RGBA")

    def _run_texture(self, mesh, image: "Image.Image", progress_cb=None):
        import torch

        self._check_texgen_extensions()

        self._report(progress_cb, 73, "Preparing texture model…")
        self._ensure_paint_weights()

        self._report(progress_cb, 78, "Loading texture model…")
        from hy3dgen.texgen import Hunyuan3DPaintPipeline

        paint_dir = self.model_dir / "_paint_weights"
        paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
            str(paint_dir), subfolder=_PAINT_SUBFOLDER
        )

        from hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender
        paint_pipeline.config.render_size  = 1024
        paint_pipeline.config.texture_size = 1024
        paint_pipeline.render = MeshRender(default_resolution=1024, texture_size=1024)

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        try:
            image.save(tmp.name)
            tmp.close()

            self._report(progress_cb, 83, "Generating textures…")
            with torch.no_grad():
                result = paint_pipeline(mesh, image=tmp.name)
        finally:
            os.unlink(tmp.name)
            del paint_pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return result[0] if isinstance(result, (list, tuple)) else result

    def _check_texgen_extensions(self) -> None:
        try:
            import custom_rasterizer  # noqa: F401
            return
        except (ImportError, ModuleNotFoundError):
            pass

        # 1. Try prebuilt binaries bundled in the extension repo
        self._inject_prebuilt_extensions()
        try:
            import custom_rasterizer  # noqa: F401
            return
        except (ImportError, ModuleNotFoundError):
            pass

        # 2. Fall back to auto-compile from source
        print("[Hunyuan3DMiniGenerator] No prebuilt binary found, attempting auto-compile…")
        try:
            self._compile_texgen_extensions()
        except Exception as exc:
            base = self.model_dir / "_hy3dgen" / "hy3dgen" / "texgen"
            raise RuntimeError(
                "C++ extensions for texture generation are missing and could not be compiled.\n"
                "Try building them manually:\n\n"
                f"  cd \"{base / 'custom_rasterizer'}\"\n"
                f"  pip install . --no-build-isolation\n\n"
                f"  cd \"{base / 'differentiable_renderer'}\"\n"
                f"  pip install . --no-build-isolation\n\n"
                f"Error: {exc}"
            ) from exc

    @staticmethod
    def _platform_tag() -> str:
        import platform
        os_tag = {"win32": "win", "linux": "linux", "darwin": "macos"}[sys.platform]
        arch = platform.machine().lower().replace("x86_64", "amd64").replace("aarch64", "arm64")
        return f"{os_tag}-{arch}-cp{sys.version_info.major}{sys.version_info.minor}"

    def _inject_prebuilt_extensions(self) -> None:
        """Add the platform-specific prebuilt/ folder to sys.path if it exists."""
        tag = self._platform_tag()
        prebuilt = Path(__file__).parent / "prebuilt" / tag
        if prebuilt.exists() and str(prebuilt) not in sys.path:
            sys.path.insert(0, str(prebuilt))
            print(f"[Hunyuan3DMiniGenerator] Injected prebuilt extensions from {prebuilt}")

    def _compile_texgen_extensions(self) -> None:
        import subprocess
        texgen_base = self.model_dir / "_hy3dgen" / "hy3dgen" / "texgen"

        for ext_name in ("custom_rasterizer", "differentiable_renderer"):
            ext_dir = texgen_base / ext_name
            if not ext_dir.exists():
                raise RuntimeError(f"Extension source not found: {ext_dir}")

            print(f"[Hunyuan3DMiniGenerator] Compiling {ext_name}…")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", ".", "--no-build-isolation"],
                cwd=str(ext_dir),
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Compilation of {ext_name} failed:\n{result.stderr[-3000:]}"
                )
            print(f"[Hunyuan3DMiniGenerator] {ext_name} compiled and installed.")

    def _ensure_paint_weights(self) -> None:
        paint_dir = self.model_dir / "_paint_weights"
        if (paint_dir / _PAINT_SUBFOLDER).exists() and (paint_dir / "hunyuan3d-delight-v2-0").exists():
            return

        from huggingface_hub import snapshot_download
        print(f"[Hunyuan3DMiniGenerator] Downloading paint model ({_PAINT_HF_REPO})…")
        snapshot_download(
            repo_id=_PAINT_HF_REPO,
            local_dir=str(paint_dir),
            ignore_patterns=[
                "hunyuan3d-dit-v2-0/**",
                "hunyuan3d-dit-v2-0-fast/**",
                "hunyuan3d-dit-v2-0-turbo/**",
                "hunyuan3d-vae-v2-0/**",
                "hunyuan3d-vae-v2-0-turbo/**",
                "hunyuan3d-vae-v2-0-withencoder/**",
                "hunyuan3d-paint-v2-0/**",
                "assets/**",
                "*.md", "LICENSE", "NOTICE", ".gitattributes",
            ],
        )
        print("[Hunyuan3DMiniGenerator] Paint model downloaded.")

    def _decimate(self, mesh, target_vertices: int):
        target_faces = max(4, target_vertices * 2)
        try:
            return mesh.simplify_quadric_decimation(target_faces)
        except Exception as exc:
            print(f"[Hunyuan3DMiniGenerator] Decimation skipped: {exc}")
            return mesh

    def _download_weights(self) -> None:
        from huggingface_hub import snapshot_download
        print(f"[Hunyuan3DMiniGenerator] Downloading {_HF_REPO_ID} (base variant)…")
        snapshot_download(
            repo_id=_HF_REPO_ID,
            local_dir=str(self.model_dir),
            ignore_patterns=[
                "hunyuan3d-dit-v2-mini-fast/**",
                "hunyuan3d-dit-v2-mini-turbo/**",
                "hunyuan3d-vae-v2-mini-turbo/**",
                "hunyuan3d-vae-v2-mini-withencoder/**",
                "*.md", "LICENSE", "NOTICE", ".gitattributes",
            ],
        )
        print("[Hunyuan3DMiniGenerator] Download complete.")

    def _ensure_hy3dgen(self) -> None:
        try:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # noqa: F401
            return
        except ImportError:
            pass

        src_dir = self.model_dir / "_hy3dgen"
        if not (src_dir / "hy3dgen").exists():
            self._download_hy3dgen(src_dir)

        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        try:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                f"hy3dgen still not importable after extraction to {src_dir}.\n"
                f"Check the folder contents.\n{exc}"
            ) from exc

    def _download_hy3dgen(self, dest: Path) -> None:
        import urllib.request

        dest.mkdir(parents=True, exist_ok=True)
        print("[Hunyuan3DMiniGenerator] Downloading hy3dgen source from GitHub…")
        with urllib.request.urlopen(_GITHUB_ZIP, timeout=180) as resp:
            data = resp.read()
        print("[Hunyuan3DMiniGenerator] Extracting hy3dgen…")

        prefix = "Hunyuan3D-2-main/hy3dgen/"
        strip  = "Hunyuan3D-2-main/"

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for member in zf.namelist():
                if not member.startswith(prefix):
                    continue
                rel    = member[len(strip):]
                target = dest / rel
                if member.endswith("/"):
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zf.read(member))

        print(f"[Hunyuan3DMiniGenerator] hy3dgen extracted to {dest}.")

    @classmethod
    def params_schema(cls) -> list:
        return [
            {
                "id":      "num_inference_steps",
                "label":   "Quality",
                "type":    "select",
                "default": 30,
                "options": [
                    {"value": 10, "label": "Fast (10 steps)"},
                    {"value": 30, "label": "Balanced (30 steps)"},
                    {"value": 50, "label": "High (50 steps)"},
                ],
            },
            {
                "id":      "octree_resolution",
                "label":   "Mesh Resolution",
                "type":    "select",
                "default": 380,
                "options": [
                    {"value": 256, "label": "Low (256)"},
                    {"value": 380, "label": "Medium (380)"},
                    {"value": 512, "label": "High (512)"},
                ],
            },
            {
                "id":      "guidance_scale",
                "label":   "Guidance Scale",
                "type":    "float",
                "default": 5.5,
                "min":     1.0,
                "max":     10.0,
            },
            {
                "id":      "seed",
                "label":   "Seed",
                "type":    "int",
                "default": -1,
                "min":     -1,
                "max":     2147483647,
            },
        ]

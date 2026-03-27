"""
Hunyuan3D 2 Mini — extension setup script.

Creates an isolated venv and installs all required dependencies.
Called by Modly at extension install time with:

    python setup.py <json_args>

where json_args contains:
    python_exe  — path to Modly's embedded Python (used to create the venv)
    ext_dir     — absolute path to this extension directory
    gpu_sm      — GPU compute capability as integer (e.g. 61 for Pascal, 86 for Ampere)

Example (manual test):
    python setup.py '{"python_exe":"C:/…/python.exe","ext_dir":"C:/…/hunyuan3d-2-mini","gpu_sm":86}'
"""
import json
import platform
import subprocess
import sys
from pathlib import Path


def pip(venv: Path, *args: str) -> None:
    is_win = platform.system() == "Windows"
    pip_exe = venv / ("Scripts/pip.exe" if is_win else "bin/pip")
    subprocess.run([str(pip_exe), *args], check=True)


def setup(python_exe: str, ext_dir: Path, gpu_sm: int) -> None:
    venv = ext_dir / "venv"
    is_win = platform.system() == "Windows"

    print(f"[setup] Creating venv at {venv} …")
    subprocess.run([python_exe, "-m", "venv", str(venv)], check=True)

    # ------------------------------------------------------------------ #
    # PyTorch — choose version based on GPU architecture
    # ------------------------------------------------------------------ #
    if gpu_sm >= 70:
        # Volta and newer — PyTorch 2.6 + CUDA 12.4
        torch_index = "https://download.pytorch.org/whl/cu124"
        torch_pkgs  = ["torch==2.6.0", "torchvision==0.21.0"]
        print(f"[setup] GPU SM {gpu_sm} -> PyTorch 2.6 + CUDA 12.4")
    else:
        # Pascal (SM 6.x) — PyTorch 2.5 + CUDA 11.8 (last version with SM 6.1)
        torch_index = "https://download.pytorch.org/whl/cu118"
        torch_pkgs  = ["torch==2.5.1", "torchvision==0.20.1"]
        print(f"[setup] GPU SM {gpu_sm} (legacy) -> PyTorch 2.5 + CUDA 11.8")

    print("[setup] Installing PyTorch …")
    pip(venv, "install", *torch_pkgs, "--index-url", torch_index)

    # ------------------------------------------------------------------ #
    # Core dependencies
    # ------------------------------------------------------------------ #
    print("[setup] Installing core dependencies …")
    pip(venv, "install",
        "Pillow",
        "numpy",
        "trimesh",
        "pymeshlab",
        "opencv-python-headless",
        "huggingface_hub",
        "diffusers>=0.31.0",
        "transformers>=4.46.0",
        "accelerate",
        "einops",
        "scipy",
        "scikit-image",
    )

    # ------------------------------------------------------------------ #
    # rembg (background removal)
    # ------------------------------------------------------------------ #
    print("[setup] Installing rembg …")
    if gpu_sm >= 70:
        pip(venv, "install", "rembg[gpu]")
    else:
        # onnxruntime-gpu has cuDNN FE issues on Pascal — use CPU provider
        pip(venv, "install", "rembg")
        pip(venv, "install", "onnxruntime")

    # ------------------------------------------------------------------ #
    # Texture generation dependencies (optional — heavy)
    # Skipped here; will be installed on first texture request if needed.
    # Requires custom C++ extensions (custom_rasterizer, differentiable_renderer)
    # built via separate wheel distribution.
    # ------------------------------------------------------------------ #

    print("[setup] Done. Venv ready at:", venv)


if __name__ == "__main__":
    # Accepts either JSON (from Electron) or positional args (for manual testing)
    # Positional: python setup.py <python_exe> <ext_dir> <gpu_sm>
    # JSON:       python setup.py '{"python_exe":"...","ext_dir":"...","gpu_sm":86}'
    if len(sys.argv) >= 4:
        setup(
            python_exe = sys.argv[1],
            ext_dir    = Path(sys.argv[2]),
            gpu_sm     = int(sys.argv[3]),
        )
    elif len(sys.argv) == 2:
        args = json.loads(sys.argv[1])
        setup(
            python_exe = args["python_exe"],
            ext_dir    = Path(args["ext_dir"]),
            gpu_sm     = int(args["gpu_sm"]),
        )
    else:
        print("Usage: python setup.py <python_exe> <ext_dir> <gpu_sm>")
        print('   or: python setup.py \'{"python_exe":"...","ext_dir":"...","gpu_sm":86}\'')
        sys.exit(1)

# modly-hunyuan3d-mini-extension

Modly extension for **Hunyuan3D 2 Mini**, Tencent's lightweight image-to-3D pipeline.

## What this extension does

- installs an isolated Python environment for the extension
- loads the Hunyuan3D 2 Mini shape pipeline inside Modly
- optionally uses the texture pipeline when the required native extensions are available

## Installation flow

At install time, `setup.py` creates a virtual environment and selects the PyTorch stack from the platform information that Modly passes in:

- `gpu_sm`
- `cuda_version`
- operating system / CPU architecture

## Platform notes

### Linux ARM64

Linux ARM64 support is handled explicitly in `setup.py`.

The installer now:

- uses ARM64-specific PyTorch/TorchVision wheel selection
- switches to the CUDA 12.8 path when Modly reports `cuda_version >= 128`
- installs pinned direct wheel URLs with published `sha256` hashes
- keeps the matching PyTorch index available for transitive dependencies such as `triton`
- increases pip timeout/retry settings for large wheel downloads
- uses `rembg` + `onnxruntime` on ARM64 instead of `rembg[gpu]`

This path was validated successfully on Linux ARM64 in Modly for both:

- extension installation
- mesh generation

### Windows

The Windows path remains unchanged.

The ARM64-specific logic is gated to:

- `platform.system() == "Linux"`
- `platform.machine() in {"aarch64", "arm64"}`

So Windows still uses the original virtualenv executable layout (`Scripts/python.exe`, `Scripts/pip.exe`) and the existing non-ARM64 dependency resolution path.

## Troubleshooting

- If installation fails after changing installer logic, run **Repair** in Modly so the extension venv is recreated.
- If texture generation fails, verify the optional native texture-generation extensions are available for your environment.

## Upstream model sources

- Model weights: `tencent/Hunyuan3D-2mini`
- Project source: `Tencent/Hunyuan3D-2`

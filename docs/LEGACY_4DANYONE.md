# Legacy: the 4DAnyone / Docker approach (abandoned)

> This was the project README for the **first** approach, which never finished on the target PC
> (see [STATUS.md](STATUS.md) for why). The working pipeline is described in the main
> [README](../README.md) and [pipeline_breakdown.md](../pipeline_breakdown.md). The Docker files,
> `scripts\build.ps1`, `test_6views.ps1` and friends belong to this legacy path. Its `patches\`
> folder (copies of 4DAnyone and GVHMR sources) is not published in the repository.

**Goal:** one front-facing full-body PNG → **static** Gaussian splat (`output/<name>/final/character.ply`). Not 4D. Not animation.

```
character.png
    → 121 identical frames (no motion)
    → 4DAnyone Turbo (6-view quality gate, then 24-view)
    → one synchronized frame across views
    → official 4DAnyone Nerfstudio exporter (known poses, no COLMAP)
    → Splatfacto
    → character.ply
```

4DAnyone is only a multiview synthesizer for a completely stationary person. Splatfacto fits a static 3D Gaussian.

## Status (read this first)

**This pipeline is blocked on the Windows 11 + Docker Desktop + WSL2 + RTX 4090 (WDDM) machine it was built on (2026-09-18).** Six-view 4DAnyone never finished. There is no contact sheet and no `character.ply`.

Agents and future sessions: read **[`docs/STATUS.md`](docs/STATUS.md)** (what we tried, what works, exact crash sites) and **[`AGENTS.md`](AGENTS.md)** before changing GPU inference.

| Works on this PC | Does not work on this PC |
|---|---|
| Docker GPU: Torch 2.8.0+cu128, CUDA 12.8, `cuda=True`, RTX 4090 23.99 GiB | Official 6-view: pose-encoder `conv3d` → `CUDA driver error: device not ready` |
| Noble Python 3.11.15, pin `61e0ccd`, weights + SMPL-X on `D:\` bind mounts | After a custom pose workaround: Turbo denoise still dies at ~24 GiB nvidia-smi / ~97 W |
| 121-frame static video; GVHMR (after turtle/tkinter overlay) | FlashAttention-3 (Hopper-only; not a 4090 path) |
| Official `low_memory=True`, `auto`→**SDPA**, Turbo, expandable CUDA allocator | SageAttention as a 24 GB workaround (also `device not ready`) |
| Scripts/Dockerfiles for 24-view, export, Splatfacto (untested) | Contact sheet, 24-view, Nerfstudio train, `character.ply` |

Do **not** rerun unmodified official 6-view in Docker Desktop on this same WDDM 4090 expecting a different result. That path was selected correctly and still failed. A Linux 4090 with exclusive VRAM is the environment 4DAnyone’s ~22 GiB numbers assume.

Code lives in this folder on `D:\Image-to-Gaussian`. Weights, caches, and outputs stay in bind-mounted `models/`, `cache/`, `output/` so Docker rebuilds do not bake checkpoints into image layers.

## Architecture

Two GPU containers, two Python environments:

| Service | Image | Role |
|---|---|---|
| `4danyone` | CUDA 12.8 devel on Ubuntu 24.04 + deadsnakes Python 3.11 + PyTorch 2.8 + official 4DAnyone | static video, inference, contact sheet, Nerfstudio frameset export |
| `nerfstudio` | official `nerfstudio:1.1.5` | Splatfacto train + `ns-export gaussian-splat` |

Pinned upstream (see `pins.json`):

- 4DAnyone `61e0ccd5c85c01c612b4c4aecb3de148e6f5bae5`
- GVHMR `6ec3ca39336c50492c0fae65fba2fb831fc7d866`
- Nerfstudio `v1.1.5` / `6b60855003011b2ca23c2fe3f8e2ca6314c69924`

Current 4DAnyone facts verified from source, not old blog posts:

- Input is sampled to **exactly 121 frames**
- Turbo is **on by default**; Base is `--enable_turbo=False`
- `views_per_layer` is a **single integer** (6 or 24). Total views must be divisible by 6
- GPUs with ≤24 GiB: official `inference.py` sets `low_memory`, expandable CUDA segments, and `attention_backend auto` → **SDPA**. Same Turbo checkpoint; peak ~22.18 GiB **allocated on exclusive Linux**. That selection works here; 6-view still does not finish under Windows WDDM + Docker. Pin: `61e0ccd5c85c01c612b4c4aecb3de148e6f5bae5`.
- Exporter writes `images/`, `masks/`, `transforms.json`, `sparse_pcd.ply` for **one** `frame_index`

## Prerequisites

- Windows 11, Docker Desktop with WSL2 backend, NVIDIA driver (already present: 596.21, CUDA 13.2 capability)
- RTX 4090 24 GB (GPU passthrough is verified; 6-view inference is **not** — see Status)
- ~30 GB free for Docker images, ~25 GB for model weights, plus output space
- **SMPL-X license**: `models_smplx_v1_1.zip` from https://smpl-x.is.tue.mpg.de/ (not redistributed)

Do not install a second Windows CUDA toolkit. Do not install a Linux NVIDIA kernel driver. Userspace CUDA lives in the containers.

## First-time setup

```powershell
cd D:\Image-to-Gaussian
.\scripts\verify_gpu.ps1
.\scripts\build.ps1
```

`verify_gpu.ps1` currently requires the 4DAnyone image. For a host-only check before the first build:

```powershell
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

That command was run on this machine and showed **NVIDIA GeForce RTX 4090**.

Install SMPL-X, then download the published 4DAnyone weights (~22 GB) into `models/`:

```powershell
.\scripts\install_smplx.ps1 -ArchivePath PATH\to\models_smplx_v1_1.zip
.\scripts\legacy_4danyone_download_models.ps1
```

## GPU verification

```powershell
.\scripts\verify_gpu.ps1
```

Inside the 4DAnyone container this prints `torch.__version__`, `torch.version.cuda`, `torch.cuda.is_available()`, GPU name, VRAM, and CUDA capability. It **fails** if CUDA is missing. There is no CPU fallback.

## Where to place character.png

```
D:\Image-to-Gaussian\input\character.png
```

Front-facing full-body still. The helper letterboxes to **1080×1920 (9:16)** without stretching, then encodes **121** frames of that same image. 4DAnyone later center/mask-crops to its trained **704×1280** canvas.

## Six-view test (quality gate)

On this Windows Docker 4090 host the command below **fails** during 4DAnyone (pose encode or first denoise step). See `docs/STATUS.md`. Do not start `-Stage generate` / train until dense `00.mp4`…`05.mp4` and `contact_sheet.png` exist.

```powershell
.\scripts\run_pipeline.ps1 -Input .\input\character.png -Stage test
```

or:

```powershell
.\scripts\test_6views.ps1 -Input .\input\character.png
```

This is the default stage. It does **not** start 24-view generation or Nerfstudio. On this host the expected outputs below were **never written** (GVHMR json only).

Intended outputs:

```
output\<name>\4danyone_6views\
    cameras.json
    videos/dense/00.mp4 … 05.mp4
    contact_sheet.png
    contact_sheet.temporal.json
    run_report.json
```

Logs: `logs\`.

## Contact-sheet review

Open `contact_sheet.png` and check identity, clothing, hair, pose, feet, side/back plausibility, and lack of animation. `contact_sheet.temporal.json` reports how much each view changes versus frame 0.

**Stop here until that sheet looks usable.** Do not “fix” the views automatically.

## 24-view generation

After you approve the six-view result:

```powershell
.\scripts\run_pipeline.ps1 -Input .\input\character.png -Stage generate
```

Writes a **separate** directory: `output\<name>\4danyone_24views\`.

## Static-frame export

```powershell
.\scripts\run_pipeline.ps1 -Input .\input\character.png -Stage export -FrameIndex 0
```

Uses official `scripts/export_nerfstudio.py`. No COLMAP. Expected frameset:

```
output\<name>\static_frameset_frame_000\
    images/
    masks/
    transforms.json
    sparse_pcd.ply
```

## Splatfacto training

```powershell
.\scripts\run_pipeline.ps1 -Input .\input\character.png -Stage train
```

Viewer: http://localhost:7007

This trains **static** `splatfacto` only (not FreeTimeGS, not a deformation field).

## Final PLY

```
output\<name>\final\character.ply
```

The Nerfstudio `config.yml` remains under `output\<name>\nerfstudio\` for inspection.

## Rebuild

```powershell
.\scripts\build.ps1
.\scripts\build.ps1 -NoCache -Service 4danyone
```

Rebuilds do not delete `models/` or `output/`.

## Clear caches safely

| Path | Safe to delete? |
|---|---|
| `cache/` | Yes. Hugging Face / Torch cache. Models already in `models/` stay. |
| `models/` | Only if you want to re-download ~22 GB (keep `body_models/smplx/` unless you still have the ZIP). |
| `output/` | Deletes experiments. |
| `logs/` | Yes. |
| Docker images | `docker image rm image-to-gaussian-4danyone:61e0ccd` — does not remove bind-mounted weights. |

## Disk budget (approximate)

- 4DAnyone published weights + BiRefNet: **~22 GB**
- SMPL-X NPZ: small relative to the rest
- CUDA 12.8 devel image + PyTorch 2.8: **tens of GB** in Docker Desktop’s disk (often on C:)
- 6-view / 24-view MP4s: small compared with weights
- Splatfacto run: checkpoint + PLY, typically well under a few GB

## One-command stages

```powershell
.\scripts\run_pipeline.ps1 -Input .\input\character.png -Stage prepare
.\scripts\run_pipeline.ps1 -Input .\input\character.png -Stage test
.\scripts\run_pipeline.ps1 -Input .\input\character.png -Stage generate
.\scripts\run_pipeline.ps1 -Input .\input\character.png -Stage export
.\scripts\run_pipeline.ps1 -Input .\input\character.png -Stage train
```

`-Force` reruns a stage. `-Rebuild` rebuilds images first. `-Stage all` still **stops after the six-view test**.

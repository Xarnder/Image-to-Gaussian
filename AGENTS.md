# Agent briefing

> **2026-09-18 update: the v2 pipeline works. Read [`plan.md`](plan.md) first.** v2 runs natively on
> Windows: ComfyUI (Qwen-Image-Edit-2511 views + SAM 3D Body) → `i2g/` registration → gsplat. It has
> already produced `output/test-image/v2/final/character.ply`. Everything below describes the
> **legacy** Docker/4DAnyone path, which is still blocked; keep it only as history.

**Goal:** one front-facing full-body PNG → **static** Gaussian splat `output/<name>/final/character.ply`. Not 4D, not animation.

**Status (2026-09-18):** blocked on this Windows 11 + Docker Desktop + WSL2 + RTX 4090 (WDDM) host. 4DAnyone never finished 6-view generation. There is no contact sheet and no PLY.

Read **`docs/STATUS.md`** before touching GPU inference, attention backends, Docker CUDA, or “low VRAM” patches.

## Working

- Docker GPU: Torch 2.8.0+cu128, CUDA 12.8, `cuda=True`, RTX 4090, 23.99 GiB.
- Noble + deadsnakes Python 3.11.15; pin 4DAnyone `61e0ccd`; weights and SMPL-X on `models/`.
- Static 121-frame video encode; GVHMR turtle overlay; GVHMR/YOLO/ViTPose run.
- Official `auto` → SDPA + Turbo + expandable allocator **does** select on this 4090 (`low_memory=True`).

## Not working

- Official 6-view dies in pose-encoder `F.conv3d` (`CUDA driver error: device not ready`, ThreadPool, batch 6).
- After a custom pose workaround (main thread, batch 3), Turbo denoise still dies at ~24 GiB nvidia-smi / ~97 W (SDPA FFN `Linear`, or Sage `torch.empty`).
- FlashAttention-3 is Hopper-only; do not install it for a 4090.
- SageAttention is not the official ≤24 GiB path and also failed here.

## Do not

- Rerun unmodified official 6-view in Docker Desktop on this same WDDM 4090 expecting a different result.
- Start 24-view, Nerfstudio, or splat export until `4danyone_6views/videos/dense/*.mp4` and `contact_sheet.png` exist.
- Bake weights into images, use Jammy Python 3.11, poll `nvidia-smi` during the CUDA child, or name PowerShell params `$Input` / `$Video`.

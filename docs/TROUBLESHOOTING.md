# Troubleshooting (legacy 4DAnyone / Docker path)

> **Legacy.** For the working pipeline, see Troubleshooting in the main [README](../README.md).

Only issues that were actually hit or directly verified on this machine are listed with commands. Planned-but-untested notes are marked.

**Pipeline status:** six-view 4DAnyone does not finish on this Windows + Docker Desktop 4090. What was tried, what still works, and the original PNG → static splat goal are in [`STATUS.md`](STATUS.md).

## Docker sees the RTX 4090

Verified:

```powershell
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

Showed `NVIDIA GeForce RTX 4090` with host driver **596.21**. Container CUDA userspace is independent of a Windows CUDA toolkit.

If this stops working:

- Docker Desktop → Settings → Resources → WSL integration enabled
- Docker Desktop → Settings → use the WSL2 engine
- `nvidia-smi` on Windows still lists the 4090
- Do **not** install a Linux NVIDIA kernel driver inside WSL

## CUDA unavailable inside the 4DAnyone image

`tools/gpu_check.py` exits non-zero if `torch.cuda.is_available()` is false. Scripts do not fall back to CPU.

## SMPL-X missing

4DAnyone will not run without `models/body_models/smplx/SMPLX_NEUTRAL.npz`. That file is **not** on Hugging Face. Install from the official ZIP:

```powershell
.\scripts\install_smplx.ps1 -ArchivePath PATH\to\models_smplx_v1_1.zip
```

A filesystem search of `D:\Colmap`, `D:\Docker_Projects`, `D:\gpu-project`, and `D:\Render` did not find an existing `SMPLX_NEUTRAL.npz`.

## Hugging Face download into Docker layers

Weights are written to the bind mount `/workspace/models`. Rebuilding the image should not re-download them if the files are already present. `HF_HOME` is `/workspace/cache/huggingface`.

## CUDA `device not ready` / 24 GB (observed; this is the blocker)

Official `inference.py` treats ≤24 GiB as low memory: `auto` → SDPA and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:4096` (set **inside** `inference.py`, not by compose). That selection **was logged** on this 4090 (`low_memory=True`). It still failed.

This host never raised a clean PyTorch OOM. It raised `RuntimeError: CUDA driver error: device not ready` while nvidia-smi sat at ~24.0/24.5 GiB, ~97 W, ~44 °C (WDDM paging). Official Linux 4090 Turbo+SDPA 6-view is ~22.18 GiB allocated / ~2.7 min plus GVHMR.

See `docs/STATUS.md` for the pose-encoder vs DiT crash sites and the experiments that already failed (FA3, Sage, official ThreadPool batch-6 pose).

## Static video rejected

The helper encodes 121 frames at 1080×1920, preferring lossless `libx264rgb` / `rgb24`, then verifies decoded frame count, size, and that frame 0 ≈ last frame.

## OneDrive

The original Cursor folder under `Documents\Code` is on OneDrive and was empty. This working tree is on `D:\Image-to-Gaussian` so Docker bind mounts and multi-GB weights are not synced by OneDrive.

## Host RAM 32 GB

Compose uses `shm_size` 4 GB (4DAnyone) and 8 GB (Nerfstudio), not a 64 GB `/dev/shm`. Docker Desktop on this PC is allocated ~30 GB RAM; do not raise `shm_size` to the full allocation.

## Nerfstudio / COLMAP

Do not run `ns-process-data`. The 4DAnyone exporter already writes `transforms.json`. Train with `ns-train splatfacto --data <frameset>`.

## Python 3.11 must be deadsnakes Noble, not Jammy

The 4DAnyone image is **Ubuntu 24.04 Noble** (`nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04`). Jammy's archive `python3.11` is `3.11.0~rc1` and must not be mixed into this container.

Verified in the torch-stage image:

```text
python3.11 --version
Python 3.11.15

releaselevel final

import torch ...
2.8.0+cu128
12.8
True
NVIDIA GeForce RTX 4090
```

The apt source is:

```text
deb ... http://ppa.launchpad.net/deadsnakes/ppa/ubuntu noble main
```

`software-properties-common` was avoided: it pulled systemd and hung BuildKit. A previous Jammy-based build was stopped after ~44 minutes of slow apt unpack.

`libsm6t64` does not exist on this CUDA Noble image; use `libsm6`.

## Docker Desktop apt unpack is slow

On this Windows + WSL2 Docker Desktop host, unpacking `.deb` files inside BuildKit can take many seconds each. That is I/O on the Docker virtual disk, not a hung process, as long as the log keeps advancing.

## GVHMR `from turtle import forward`

Six-view inference died in GVHMR with `ModuleNotFoundError: No module named 'tkinter'`. Upstream `third_party/GVHMR/hmr4d/utils/body_model/body_model.py` starts with `from turtle import forward`, which pulls in Tk. The name is unused; `BodyModel` defines its own `forward`. We overlay `patches/gvhmr_body_model.py` and delete that import in the image.

## CUDA `device not ready` during pose encoding

After GVHMR, six-view inference died in `build_pose_feature_cache` with `RuntimeError: CUDA driver error: device not ready` on the pose-encoder `conv3d`. Two things showed up on this machine:

1. Upstream always submits pose encoding to a `ThreadPoolExecutor`, even for one GPU.
2. A 6-video BF16 batch at 1280×704 peaked at **23686 MiB** (almost the whole 4090), then the first 3D conv failed. Official Linux 4090 runs peak around 22 GiB; Windows desktop + WSL2 Docker leave less headroom.

The DiT then died in the first FFN `Linear` (`device not ready` again) after ~20 minutes at 24.0/24.5 GiB and ~97 W on Windows WDDM. Official 4DAnyone does **not** ship a separate 22 GB weight: pin `61e0ccd5c85c01c612b4c4aecb3de148e6f5bae5` already contains `has_low_memory_gpu` / `configure_inference_cuda_allocator` / auto→SDPA. Remounting official files and passing `--attention_backend auto` still dies at pose `conv3d` on this host. Full chronology: `docs/STATUS.md`.

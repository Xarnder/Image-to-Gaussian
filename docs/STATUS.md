# Pipeline status for future agents

> **2026-09-18 update:** this file documents the **legacy** Docker + 4DAnyone path, which is still
> blocked. The v2 pipeline in [`plan.md`](../plan.md) (native Windows, ComfyUI + gsplat) works and
> has produced `output/test-image/v2/final/character.ply`. See plan.md's status table.

**Date:** 2026-09-18  
**Machine:** Windows 11, Docker Desktop + WSL2, RTX 4090 24 GB (WDDM, display on), Ryzen 9 7900X, ~32 GB RAM, NVIDIA driver 596.21 / host CUDA 13.2  
**Verdict:** This repo’s Docker pipeline is **blocked**. It does **not** produce a Gaussian splat on this host. Do not treat the README scripts as a working end-to-end path until 6-view videos exist.

Read this file before changing inference, Docker GPU settings, attention backends, or “low VRAM” code.

---

## Goal (unchanged)

**Input:** one front-facing full-body character still (PNG).  
**Output:** one **static** 3D Gaussian splat (`output/<name>/final/character.ply`).

Not animated. Not 4D. Not FreeTimeGS. 4DAnyone is only a **multiview synthesizer** for a completely stationary person, with known cameras, so Nerfstudio Splatfacto can train without COLMAP/Postshot.

Intended graph:

```
character.png
  → 121 identical frames (no motion), 1080×1920, 24 fps, h264
  → 4DAnyone Turbo, 6 views first (quality gate + contact sheet)
  → human review of contact_sheet.png
  → 4DAnyone 24 views
  → one synchronized frame via official export_nerfstudio.py (known poses)
  → Nerfstudio splatfacto (static)
  → ns-export gaussian-splat → character.ply
```

Quality gate: **do not** start 24-view or Splatfacto until `output/<name>/4danyone_6views/contact_sheet.png` exists and looks usable.

---

## Input used on this machine

| Item | Path |
|---|---|
| Still | `test-assets/test-image.png` (also copied under `input/`) |
| Static video | `test-assets/test-image_static_4danyone.mp4` and `input/test-image_static_4danyone.mp4` |
| Video spec | 121 frames, 1080×1920, 24 fps, h264; 4DAnyone later uses a 704×1280 canvas |
| SMPL-X ZIP | `models/_incoming/models_smplx_v1_1.zip` (licensed, not redistributed) |
| Installed body model | `models/body_models/smplx/SMPLX_NEUTRAL.npz` |
| 4DAnyone / GVHMR / BiRefNet weights | bind-mounted under `models/` (~22 GB, Hugging Face `AntResearch/4DAnyone` rev `4c80e87…`) |

Default scripts also accept `input/character.png`. The real run used the test-image assets above.

---

## What is actually working

These pieces were verified on this PC. They are safe to reuse.

### Host / Docker / PyTorch

- Docker GPU passthrough: `nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi` sees **NVIDIA GeForce RTX 4090**, 24564 MiB.
- Image `image-to-gaussian-4danyone:full` / `:61e0ccd` builds on **Ubuntu 24.04 Noble**.
- Python **3.11.15** from deadsnakes **noble** (not Jammy `3.11.0~rc1`).
- Torch **2.8.0+cu128**, `torch.version.cuda` **12.8**, `torch.cuda.is_available()` **True**, `vram_gib=23.99`, capability **8.9**, `matmul_ok=True`.
- Weights stay on `D:\Image-to-Gaussian\models` and `cache/`; they are **not** baked into image layers.
- Compose `shm_size` 4g (4DAnyone) / 8g (Nerfstudio). Host RAM is 32 GB; do not raise shm to the full allocation.

### Data prep

- `scripts/install_smplx.ps1` extracts `SMPLX_NEUTRAL.npz`.
- `scripts/legacy_4danyone_download_models.ps1` (then named `download_models.ps1`) pulled published 4DAnyone + GVHMR + BiRefNet + turbo LoRA into `models/`.
- `tools/make_static_video.py` / `scripts/make_static_video.ps1` encode 121 identical frames. PowerShell parameters must **not** be named `$Input` or `$Video` (reserved; they emptied paths). Use `$InputImage` / `-StaticVideo`.

### 4DAnyone **up to** pose encoding / denoising

- GVHMR YOLO + ViTPose + HMR2 + skeleton render **does** run. Adaptive framing reported `input=full_body confidence=0.980`.
- Partial outputs exist, for example:
  - `output/test-image/4danyone_6views/gvhmr/motion.json`
  - `output/test-image/4danyone_6views/.4danyone-request.json` (`enable_turbo: true`, `attention_backend: sdpa`, `views_per_layer: 6`)
  - scratch cameras under `output/test-image/.test-image_static_4danyone.scratch-*/conditioning/cameras.json`
- Official ≤24 GiB selection **does** activate. Logged on the last official run (`logs/test-image_6views_20260918-173425.log`):

```
low_memory=True
attention_backend_requested=auto
attention_backend_resolved=sdpa
enable_turbo=True
cuda_alloc_conf=expandable_segments:True,max_split_size_mb:4096
fdanyone_commit=61e0ccd5c85c01c612b4c4aecb3de148e6f5bae5
Using attention backend: sdpa
```

`has_low_memory_gpu` treats nvidia-smi total ≤ `24 * 1024**3` as low memory. 24564 MiB still counts.

- GVHMR unused `from turtle import forward` (pulls Tk) is fixed by `patches/gvhmr_body_model.py` plus a Dockerfile `sed`. Without that, inference dies with `ModuleNotFoundError: tkinter`.
- **Custom** pose-encoder execution (not official): main thread when there is one GPU, `POSE_ENCODER_BATCH_LIMIT = 3`, `empty_cache` after batches. That path printed `Pose cache complete allocated_mib=0 reserved_mib=0` and reached `Generate 6 target views 0/4`. Official unmodified pose encode never completed here.

### Scaffolding that exists but was never reached

Scripts and Dockerfiles for 24-view, `export_nerfstudio.py`, and Nerfstudio `1.1.5` / Splatfacto are in the tree. **None of them have been run successfully.** There is no contact sheet, no 24-view set, no frameset, no `character.ply`.

---

## What is not working (blocker)

**6-view 4DAnyone generation does not finish** in Docker Desktop GPU on this Windows 4090.

The CUDA error is almost always:

```text
RuntimeError: CUDA driver error: device not ready
```

That is **not** a clean `torch.cuda.OutOfMemoryError`. On this host it showed up while nvidia-smi sat at **~24.0 / 24.5 GiB**, GPU util high, **power ~97 W**, **temps ~44 °C** — paging / WDDM “device not ready”, not a busy exclusive Linux 4090. Official Linux 4090 Turbo + SDPA 6-view is documented at ~**22.18 GiB allocated** and ~27 s per generated view.

Two failure sites:

1. **Official pose encoder** (`fdanyone/model/conditioning.py`): `ThreadPoolExecutor` even for one GPU, `POSE_ENCODER_BATCH_LIMIT = 6`, first `F.conv3d` on BF16 roughly `(6, 3, 121, 1280, 704)` (T later padded toward 124). Last official run died in `future.result()` → `_encode_pose_batch` → `wan_video_pose_encoder` conv3d after loading skeleton `00.mp4`…`04.mp4`. Log: `logs/test-image_6views_20260918-173425.log`. `run_report.json` `peak_gpu_memory_mib=387` is a **post-crash** sample, not the real peak.
2. **DiT Turbo denoising** (only after the custom pose workaround): first step of `Generate 6 target views 0/4`.
   - SDPA: `Linear` FFN `3072 → 14336` in `wan_video_dit.py` (`logs/test-image_6views_20260918-162548.log`). One attempt sat ~20 minutes at a full nvidia-smi bar.
   - SageAttention 2.2.0: `sageattn` → `torch.empty` for fp8 V (`logs/test-image_6views_20260918-171740.log`).

**Conclusion for this environment:** the official 4DAnyone 4090 path is real on **exclusive Linux**. It does not have enough WDDM + WSL2 Docker GPU headroom here, even when `auto→SDPA` and Turbo are selected correctly. Changing attention backends did not produce 6 view MP4s.

---

## Experiments tried (do not repeat blindly)

| Attempt | Result |
|---|---|
| Jammy `python3.11` | Wrong: `3.11.0~rc1`. Stopped. Use Noble + deadsnakes. |
| `libsm6t64` | Package missing on this CUDA Noble image. Use `libsm6`. |
| `software-properties-common` for the PPA | Pulled systemd; BuildKit hung. Install deadsnakes via keyring + sources.list. |
| Bake weights into the image / download during build | Slow, overlaps GPU work. Bind-mount `models/` instead. |
| Concurrent `nvidia-smi` polling while the child holds CUDA | Avoid on WSL2. Scripts stopped doing this. |
| Official GVHMR `from turtle import forward` | `tkinter` missing. Overlay `patches/gvhmr_body_model.py`. |
| Official pose encode, batch 6, ThreadPool | `device not ready` in `conv3d`. |
| Custom pose: main thread, batch 3, `empty_cache` | Pose cache **completed**. Denoise still failed. |
| Forced PyTorch fused SDPA kernels (still SDPA API) | Denoise still `device not ready` in FFN `Linear`. |
| FlashAttention-3 | **Do not install.** Hopper-only (H100/H200). Official 4090 tables mark FA3 as `—`. `FLASH_ATTN_3_AVAILABLE` is False. |
| SageAttention 2.2.0 from GitHub (`d1a57a54`; PyPI is 1.0.x) | Imports. Official `auto` on ≤24 GiB still forces **SDPA first**, so Sage is not the 4090 default. Forced Sage still died in `sageattn` at ~24 GiB. |
| Official files remounted (`patches/official/{conditioning,inference,wan_video_dit}.py`) + `--attention-backend auto` | Logger proved `low_memory=True` → SDPA. Died again at official pose `conv3d`. |
| Quantization / CPU offload / ComfyUI / custom DiT rewrite | Explicitly rejected. Not implemented. |
| 24-view, Nerfstudio, PLY | Never started. Gate is 6-view videos + contact sheet. |

Pinned 4DAnyone commit **is** current `main` as of 2026-09-18 and **does** contain `has_low_memory_gpu`, `configure_inference_cuda_allocator`, and `auto` → SDPA. There is **no** separate “22 GB checkpoint”. Same Turbo weights; peak allocated ~22 GiB on Linux.

---

## Current repo wiring (as left)

- `pins.json`: 4DAnyone `61e0ccd5c85c01c612b4c4aecb3de148e6f5bae5`, GVHMR `6ec3ca39336c50492c0fae65fba2fb831fc7d866`, Nerfstudio `v1.1.5`, torch 2.8 cu128, Python 3.11.
- `scripts/test_6views.ps1` passes `--enable-turbo True --attention-backend auto`.
- `scripts/common.ps1` `Invoke-4danyone` mounts **official** `patches/official/*.py` over the image copies, plus the GVHMR turtle overlay. It does **not** inject `PYTORCH_CUDA_ALLOC_CONF`; official `inference.py` sets it when `low_memory`.
- Custom (non-official) pose/SDPA patches still exist at `patches/fdanyone_*.py` but are **not** mounted.
- Existing `:full` image may still contain SageAttention 2.2.0 in layers; `auto` on this 4090 must not select it.
- `tools/log_official_path.py` must `sys.path.insert(0, "/opt/4danyone")` or `fdanyone` is missing.

---

## Constraints that still apply

- Project, models, cache, output stay on `D:\Image-to-Gaussian` (not OneDrive `C:`).
- Two GPU containers: 4DAnyone vs Nerfstudio. No Conda. No Windows NVIDIA driver reinstall from this repo. Userspace CUDA lives in the container.
- Do not poll `nvidia-smi` in a second process while inference holds the GPU on WSL2.
- `views_per_layer` is a **single integer** (6 or 24). Total views divisible by 6. Input sampled to **exactly 121** frames.
- Do not run `ns-process-data` / COLMAP. Exporter writes `images/`, `masks/`, `transforms.json`, `sparse_pcd.ply` for **one** `frame_index`.

---

## What a future agent should do

1. **Do not** spend another long Docker Desktop session expecting unmodified official 6-view to complete on this Windows WDDM 4090. That path was selected correctly and still died.
2. If the **goal** is still PNG → static PLY, the missing capability is **6 reconstruction views with known cameras**. Options that were **not** proven here:
   - Native Linux on this same 4090 (matches 4DAnyone’s 22 GiB numbers).
   - WSL2 Ubuntu + NVIDIA CUDA **without** Docker Desktop (not tried).
   - A different multiview method that fits ~24 GB WDDM, if the user accepts leaving 4DAnyone.
3. If retrying 4DAnyone at all, require a host where nvidia-smi stays near **~22 GiB allocated** and GPU **power is hundreds of watts**, not ~97 W at 24.0/24.5 GiB.
4. Pose-encoder ThreadPool + batch 6 is a **known** WSL2 footgun; the custom main-thread / batch-3 overlay got past pose but **not** past DiT.
5. Do not install FlashAttention-3. Do not treat Sage as the official ≤24 GiB path.
6. Do not start `scripts/generate_24views.ps1`, `export_static_frame.ps1`, or `train_splat.ps1` until `videos/dense/00.mp4`…`05.mp4` and `contact_sheet.png` exist.

### Proof artifacts

| Artifact | Meaning |
|---|---|
| `logs/test-image_6views_20260918-173425.log` | Official auto→SDPA; crash at pose `conv3d` |
| `logs/test-image_6views_20260918-162548.log` | Custom pose OK; SDPA denoise crash in FFN `Linear` |
| `logs/test-image_6views_20260918-171740.log` | Custom pose OK; Sage denoise crash in `torch.empty` |
| `output/test-image/4danyone_6views/` | Incomplete; GVHMR json only, **no** dense view MP4s |

---

## Success criteria (still unmet)

Pipeline is only done when all of these exist:

1. `output/<name>/4danyone_6views/videos/dense/00.mp4` … `05.mp4`
2. `output/<name>/4danyone_6views/contact_sheet.png` reviewed
3. `output/<name>/4danyone_24views/` generated after that review
4. `output/<name>/static_frameset_frame_000/` with `transforms.json`
5. `output/<name>/final/character.ply` from **static** Splatfacto

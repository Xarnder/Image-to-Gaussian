# Setup (legacy 4DAnyone / Docker path)

> **Legacy.** This is the setup for the abandoned first approach. For the working pipeline, follow
> the main [README](../README.md).

Project root: `D:\Image-to-Gaussian` (storage drive, not the OneDrive `C:` copy).

**Blocked on this PC (2026-09-18):** Docker GPU, weights, SMPL-X, and the static 121-frame video all work. Six-view 4DAnyone does **not** complete under Windows WDDM + Docker Desktop. Full write-up: [`STATUS.md`](STATUS.md). Do not treat step 6 as a working quality gate until a 6-view contact sheet exists.

## 1. Confirm Docker GPU passthrough

Already verified on this PC:

```text
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
→ NVIDIA GeForce RTX 4090, 24564 MiB, host driver 596.21, container CUDA 13.2
```

If this ever fails, see `docs/TROUBLESHOOTING.md`. Do not reinstall the Windows NVIDIA driver from these scripts.

## 2. Build images

```powershell
cd D:\Image-to-Gaussian
.\scripts\build.ps1
```

The 4DAnyone build clones commit `61e0ccd` and installs Python 3.11 + PyTorch 2.8 cu128. Weights are **not** copied into the image.

## 3. SMPL-X (manual, licensed)

1. Create a free account at https://smpl-x.is.tue.mpg.de/
2. Accept the license and download `models_smplx_v1_1.zip`
3. Run:

```powershell
.\scripts\install_smplx.ps1 -ArchivePath C:\path\to\models_smplx_v1_1.zip
```

Inference cannot start without `models\body_models\smplx\SMPLX_NEUTRAL.npz`.

## 4. Hugging Face weights

```powershell
.\scripts\legacy_4danyone_download_models.ps1
```

Downloads persist under `models\` and `cache\huggingface\`. If a repo is gated, set `HF_TOKEN` in the environment (do not put tokens in the repo).

## 5. Input image

Copy the character still to `input\character.png`.

## 6. Six-view quality gate

```powershell
.\scripts\run_pipeline.ps1 -Input .\input\character.png -Stage test
```

Inspect `output\<name>\4danyone_6views\contact_sheet.png` before any 24-view or Splatfacto work.

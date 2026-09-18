#!/usr/bin/env python3
r"""
Download every model file the pipeline needs (skips files you already have).

EXAMPLE (from the project folder, environment activated):
    python3 scripts\download_models.py                   # show what is missing, ask, then download
    python3 scripts\download_models.py --yes             # no question
    python3 scripts\download_models.py --local_only      # just the two small models in .\models
    python3 scripts\download_models.py --comfy_root "C:\ComfyUI_windows_portable"

Local models go to .\models; the big ones go to <ComfyUI>\ComfyUI\models\<folder>.
About 33 GB in total if you have none of them. Hugging Face sometimes asks you to log in
(`hf auth login`) or to accept a model licence on its web page first; the error says so.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = REPO_ROOT / ".venv-i2g"


def _relaunch_in_venv() -> None:
    if Path(sys.prefix).resolve() == VENV_DIR.resolve():
        return
    for exe in ("python3.exe", "python.exe"):
        python = VENV_DIR / "Scripts" / exe
        if python.exists():
            sys.exit(subprocess.call([str(python), str(Path(__file__).resolve()), *sys.argv[1:]]))
    print("ERROR: run scripts\\v2_setup_env.ps1 first (it creates the .venv-i2g environment).")
    sys.exit(2)


if __name__ == "__main__":
    _relaunch_in_venv()

import argparse  # noqa: E402
import shutil  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))
from i2g.paths import COMFY_ROOT, MODELS_DIR  # noqa: E402

# (repo, file in repo, destination folder, approx GB, revision)
LOCAL_MODELS = [
    ("ZhengPeng7/BiRefNet", "model.safetensors", "birefnet", 0.44, None),
    ("ZhengPeng7/BiRefNet", "config.json", "birefnet", 0.0, None),
    ("ZhengPeng7/BiRefNet", "birefnet.py", "birefnet", 0.0, None),
    ("ZhengPeng7/BiRefNet", "BiRefNet_config.py", "birefnet", 0.0, None),
    ("AntResearch/4DAnyone", "perceptual/imagenet-vgg-verydeep-19-conv.safetensors", "perceptual", 0.08,
     "7850985888b56aabf09e69480b73248f1a76bcbe"),
]
COMFY_MODELS = [
    ("Comfy-Org/Qwen-Image-Edit_ComfyUI", "split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors",
     "diffusion_models", 20.5, None),
    ("Comfy-Org/Qwen-Image_ComfyUI", "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
     "text_encoders", 9.4, None),
    ("Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae/qwen_image_vae.safetensors", "vae", 0.25, None),
    ("fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA", "qwen-image-edit-2511-multiple-angles-lora.safetensors",
     "loras", 0.29, None),
    ("lightx2v/Qwen-Image-Edit-2511-Lightning", "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
     "loras", 0.85, None),
    ("Comfy-Org/sam-3d-body", "detection/sam_3d_body_dinov3_bf16.safetensors", "detection", 2.8, None),
]
ALSO_ACCEPT = {"text_encoders": ["clip"]}   # older ComfyUI layouts keep text encoders in models\clip


def _download(repo: str, filename: str, dest: Path, revision: str | None) -> None:
    from huggingface_hub import hf_hub_download
    staging = dest.parent / ".i2g_download"
    got = Path(hf_hub_download(repo_id=repo, filename=filename, revision=revision, local_dir=str(staging)))
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(got), str(dest))
    shutil.rmtree(staging, ignore_errors=True)   # only our own staging folder


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the model files the pipeline needs.")
    parser.add_argument("--comfy_root", type=Path, default=COMFY_ROOT,
                        help=f"portable ComfyUI folder (default from config.local.json: {COMFY_ROOT})")
    parser.add_argument("--local_only", action="store_true", help="only the small models in .\\models")
    parser.add_argument("--comfy_only", action="store_true", help="only the ComfyUI models")
    parser.add_argument("--yes", "-y", action="store_true", help="download without asking")
    args = parser.parse_args()

    comfy_models_dir = args.comfy_root / "ComfyUI" / "models"
    jobs = []   # (repo, file in repo, destination folder, folders that also count as present, GB, revision)
    if not args.comfy_only:
        jobs += [(r, f, MODELS_DIR / d, [], gb, rev) for r, f, d, gb, rev in LOCAL_MODELS]
    if not args.local_only:
        if not comfy_models_dir.exists():
            print(f"ERROR: {comfy_models_dir} does not exist.\n"
                  f"FIX:   pass --comfy_root, or set comfy_root in config.local.json (copy config.example.json).")
            return 2
        jobs += [(r, f, comfy_models_dir / d, [comfy_models_dir / a for a in ALSO_ACCEPT.get(d, [])], gb, rev)
                 for r, f, d, gb, rev in COMFY_MODELS]

    missing = []
    for repo, filename, folder, also, gb, rev in jobs:
        name = Path(filename).name
        present = any((d / name).exists() for d in [folder] + also)
        print(f"  {'have   ' if present else 'MISSING'}  {name:<62} -> {folder}")
        if not present:
            missing.append((repo, filename, folder / name, gb, rev))

    if not missing:
        print("\nAll model files are present.")
        return 0
    total = sum(m[3] for m in missing)
    print(f"\n{len(missing)} file(s) to download, about {total:.1f} GB.")
    if not args.yes and input("Download now? [y/N] ").strip().lower() not in ("y", "yes"):
        print("Nothing downloaded.")
        return 1

    failed = 0
    for repo, filename, dest, gb, rev in missing:
        print(f"\n-> {repo} / {filename}  ({gb:.1f} GB)")
        try:
            _download(repo, filename, dest, rev)
            print(f"   saved {dest}")
        except Exception as exc:  # noqa: BLE001 - report every failure, keep going
            failed += 1
            print(f"   FAILED: {type(exc).__name__}: {exc}")
            print("   FIX:    check your internet connection; if Hugging Face says 401/403, open "
                  f"https://huggingface.co/{repo}, accept its terms if asked, then run `hf auth login`.")
    print(f"\nDone: {len(missing) - failed} downloaded, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

"""Filesystem locations shared by the v2 pipeline.

Machine-specific settings come from, in order: environment variables (I2G_COMFY_ROOT, I2G_COMFY_URL),
then config.local.json in the repository root (git-ignored; copy config.example.json), then defaults.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMFY_ROOT = r"C:\ComfyUI_windows_portable"
DEFAULT_COMFY_URL = "http://127.0.0.1:8188"


def local_config() -> dict:
    path = REPO_ROOT / "config.local.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError):
        return {}


_CONFIG = local_config()
COMFY_ROOT = Path(os.environ.get("I2G_COMFY_ROOT") or _CONFIG.get("comfy_root") or DEFAULT_COMFY_ROOT)
COMFY_URL = os.environ.get("I2G_COMFY_URL") or _CONFIG.get("comfy_url") or DEFAULT_COMFY_URL
COMFY_OUTPUT_DIR = COMFY_ROOT / "ComfyUI" / "output"

MODELS_DIR = REPO_ROOT / "models"
OUTPUT_DIR = REPO_ROOT / "output"
LOGS_DIR = REPO_ROOT / "logs"

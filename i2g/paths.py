"""Filesystem locations shared by the v2 pipeline.

Machine-specific settings come from, in order: environment variables (I2G_COMFY_ROOT, I2G_COMFY_URL),
then config.local.json in the repository root (git-ignored; copy config.example.json), then defaults.
"""
from __future__ import annotations

import json
import os
import re
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

# Final-folder files are written as <photo>-and-<experiment>-<kind>, e.g. alice-and-exp3-character.ply.
OUTPUT_KINDS = (
    "character.ply",
    "turntable.mp4",
    "preview_8_angles.png",
    "photo_vs_render.png",
    "report.json",
    "contact_sheet.png",
    "run_summary.json",
)


def safe_run_name(name: str) -> str:
    """Photo stem, --name, or --experiment as a Windows-safe folder/file piece."""
    cleaned = re.sub(r'[<>:"/\\|?*]', "_", name).strip(". ")
    return cleaned or "photo"


def deliverable_stem(run_name: str, experiment: str = "") -> str:
    """alice + exp3 -> alice-and-exp3; alice alone -> alice."""
    photo = safe_run_name(run_name)
    exp = safe_run_name(experiment) if experiment else ""
    if exp:
        return f"{photo}-and-{exp}"
    return photo


def deliverable_filename(run_name: str, kind: str, experiment: str = "") -> str:
    """alice + character.ply + exp3 -> alice-and-exp3-character.ply."""
    return f"{deliverable_stem(run_name, experiment)}-{kind}"


def deliverable_path(output_dir: Path, run_name: str, kind: str, experiment: str = "") -> Path:
    return output_dir / deliverable_filename(run_name, kind, experiment)


def find_deliverable(output_dir: Path, run_name: str, kind: str, experiment: str = "") -> Path | None:
    """Prefixed file if it exists, else older names from earlier runs."""
    metadata_path = output_dir.parent / "output_naming.json"
    if metadata_path.is_file():
        import json
        try:
            naming = json.loads(metadata_path.read_text(encoding="utf-8"))
            named = deliverable_path(output_dir, naming["name"], kind, naming.get("experiment", experiment))
            if named.is_file():
                return named
        except (OSError, ValueError, KeyError):
            pass
    for candidate in (
        deliverable_path(output_dir, run_name, kind, experiment),
        output_dir / f"{safe_run_name(run_name)}-{kind}",
        output_dir / kind,
    ):
        if candidate.exists():
            return candidate
    if kind == "character.ply":
        from .batch_outputs import linked_ply
        return linked_ply(output_dir)
    return None


def existing_experiments(root: Path | None = None) -> list[str]:
    """Experiment folders under output/ (skips old per-character folders that contain v2/)."""
    folder = root or OUTPUT_DIR
    if not folder.is_dir():
        return []
    names = []
    for p in folder.iterdir():
        if not p.is_dir() or p.name.startswith("."):
            continue
        if (p / "v2").is_dir():
            continue
        names.append(p.name)
    return sorted(names)

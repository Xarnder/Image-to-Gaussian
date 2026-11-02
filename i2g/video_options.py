"""Shared video launch validation for the GUI and Windows launchers (no GPU imports)."""
from pathlib import Path
import hashlib
import json
import time
import uuid
from .paths import OUTPUT_DIR, REPO_ROOT, safe_run_name

VIDEO_PRESETS = {
    "automatic": {"coverage": "auto", "initialization": "auto", "iterations": 15000},
    "standard": {"coverage": "full", "initialization": "hull", "iterations": 15000},
    "closeup": {"coverage": "partial", "initialization": "sparse", "iterations": 7500},
    "partial-hull": {"coverage": "partial", "initialization": "hull", "iterations": 7500},
}

VIDEO_STAGES = (
    ("prepare", "Prepare video frames", "Extract alternate frames and isolate the person", 180),
    ("calibrate", "Recover cameras", "Match foreground features and validate camera coverage", 60),
    ("train", "Train the Gaussian splat", "Fit the splat using the selected video settings", 800),
    ("export", "Export and evaluate", "Write the PLY, comparison and rendered turntable", 30),
)


def video_output(experiment, source, output_name=None):
    return OUTPUT_DIR / safe_run_name(experiment) / safe_run_name(output_name or Path(source).stem) / "video"


def archive_replaced_video(body):
    """Keep earlier results when rebuilding or changing the source video/settings."""
    work = video_output(body["experiment"], body["input"], body.get("output_name")).resolve()
    work.relative_to(OUTPUT_DIR.resolve())
    manifest = work / "input.json"
    if not work.exists():
        return None
    replace = bool(body.get("redo"))
    if manifest.exists():
        old = json.loads(manifest.read_text(encoding="utf-8"))
        replace |= old.get("sha256") != hashlib.sha256(Path(body["input"]).read_bytes()).hexdigest()
        replace |= old.get("stride") != int(body.get("stride", 2)) or old.get("max_height") != int(body.get("max_height", 1280))
    if not replace:
        return None
    archive = work.with_name("video_previous_" + time.strftime("%Y%m%d-%H%M%S") + "_" + uuid.uuid4().hex[:8])
    archive.resolve().relative_to(OUTPUT_DIR.resolve())
    work.rename(archive)
    return archive


def video_command(python, body):
    settings = video_settings(body)
    source = Path(str(body.get("input") or "")).expanduser()
    if not source.is_file() or source.suffix.lower() != ".mp4":
        raise ValueError("Choose a pre-generated 360° MP4 video.")
    if body.get("approved_video") is not True:
        raise ValueError("Review the video and confirm that its 360° orbit is suitable before training.")
    experiment = str(body.get("experiment") or "").strip()
    if not experiment:
        raise ValueError("Enter an experiment name or number.")
    basis = body.get("approval_basis", "manual-review")
    if basis not in ("manual-review", "automatic-first"):
        raise ValueError("Unknown video selection policy.")
    naming = ["--name", safe_run_name(body.get("output_name") or source.stem), "--experiment", safe_run_name(experiment)]
    if body.get("source_image"):
        naming += ["--source-image", str(body["source_image"])]
    return [str(python), "-u", str(REPO_ROOT / "video_to_splat.py"), *naming, "--input", str(source.resolve()),
            "--output", str(video_output(experiment, source, body.get("output_name"))), "--approved-video", "--approval-basis", basis, *settings]


def video_settings(body):
    """Validate reconstruction options before spending time generating a video."""
    preset = body.get("video_preset") or "standard"
    if preset not in VIDEO_PRESETS:
        raise ValueError("Unknown video preset.")
    body = VIDEO_PRESETS[preset] | {k: v for k, v in body.items() if v is not None}
    cmd = []
    for field, flag, default, lo, hi in (
        ("iterations", "--iterations", 15000, 100, 100000),
        ("stride", "--stride", 2, 1, 30),
        ("max_height", "--max-height", 1280, 256, 2160),
    ):
        try:
            value = int(body.get(field, default))
        except (ValueError, TypeError):
            raise ValueError(f"{field} must be an integer.") from None
        if not lo <= value <= hi:
            raise ValueError(f"{field} must be between {lo} and {hi}.")
        cmd.extend([flag, str(value)])
    coverage = body.get("coverage", "full")
    if coverage not in ("auto", "full", "partial"):
        raise ValueError("Coverage must be auto, full or partial.")
    cmd.extend(["--coverage", coverage])
    initialization = body.get("initialization", "auto")
    if initialization not in ("auto", "sparse", "hull"):
        raise ValueError("Initialization must be auto, sparse or hull.")
    cmd.extend(["--initialization", initialization])
    return cmd

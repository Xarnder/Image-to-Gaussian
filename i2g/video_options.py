"""Shared video launch validation for the GUI and Windows launchers (no GPU imports)."""
from pathlib import Path
from .paths import OUTPUT_DIR, REPO_ROOT, safe_run_name

VIDEO_PRESETS = {
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


def video_output(experiment, source):
    return OUTPUT_DIR / safe_run_name(experiment) / safe_run_name(Path(source).stem) / "video"


def video_command(python, body):
    preset = body.get("video_preset") or "standard"
    if preset not in VIDEO_PRESETS:
        raise ValueError("Unknown video preset.")
    body = VIDEO_PRESETS[preset] | {k: v for k, v in body.items() if v is not None}
    source = Path(str(body.get("input") or "")).expanduser()
    if not source.is_file() or source.suffix.lower() != ".mp4":
        raise ValueError("Choose a pre-generated 360Â° MP4 video.")
    if body.get("approved_video") is not True:
        raise ValueError("Review the video and confirm that its 360Â° orbit is suitable before training.")
    experiment = str(body.get("experiment") or "").strip()
    if not experiment:
        raise ValueError("Enter an experiment name or number.")
    cmd = [str(python), "-u", str(REPO_ROOT / "video_to_splat.py"), "--input", str(source.resolve()),
           "--output", str(video_output(experiment, source)), "--approved-video"]
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
    if coverage not in ("full", "partial"):
        raise ValueError("Coverage must be full or partial.")
    cmd.extend(["--coverage", coverage])
    initialization = body.get("initialization", "auto")
    if initialization not in ("auto", "sparse", "hull"):
        raise ValueError("Initialization must be auto, sparse or hull.")
    cmd.extend(["--initialization", initialization])
    return cmd

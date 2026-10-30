"""Shared video launch validation for the GUI and Windows launchers (no GPU imports)."""
from pathlib import Path
from .paths import OUTPUT_DIR, REPO_ROOT, safe_run_name

VIDEO_STAGES = (
    ("prepare", "Prepare video frames", "Extract alternate frames and isolate the person", 180),
    ("calibrate", "Recover cameras", "Match foreground features and check the full orbit", 60),
    ("train", "Train the Gaussian splat", "15,000 steps with the successful video recipe", 800),
    ("export", "Export and evaluate", "Write the PLY, comparison and rendered turntable", 30),
)


def video_output(experiment, source):
    return OUTPUT_DIR / safe_run_name(experiment) / safe_run_name(Path(source).stem) / "video"


def video_command(python, body):
    source = Path(str(body.get("input") or "")).expanduser()
    if not source.is_file() or source.suffix.lower() != ".mp4":
        raise ValueError("Choose a pre-generated 360° MP4 video.")
    if body.get("approved_video") is not True:
        raise ValueError("Review the video and confirm that its 360° orbit is suitable before training.")
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
    return cmd

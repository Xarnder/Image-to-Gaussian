"""Windows launcher dispatch: preserve image CLI options and add approved MP4 reconstruction."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
from i2g.paths import REPO_ROOT
from i2g.video_options import video_command


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--method", choices=("image", "video"))
    parser.add_argument("--batch-launcher", action="store_true")
    opts, rest = parser.parse_known_args(argv)
    if not rest:
        print("Choose a method:\n  1. Image edit pipeline (photo or folder)\n  2. Pre-generated 360 video (MP4)")
        choice = opts.method or input("Method [1/2]: ").strip()
        if choice not in ("1", "2", "image", "video"):
            print("Choose 1 or 2."); return 2
        opts.method = "video" if choice in ("2", "video") else "image"
        label = "360 MP4" if opts.method == "video" else ("Photo folder" if opts.batch_launcher else "Photo")
        source = input(f"{label} path (or drag it here): ").strip().strip('"')
        rest = ["--input", source]
    source = rest[rest.index("--input")+1] if "--input" in rest and rest.index("--input")+1 < len(rest) else ""
    method = opts.method or ("video" if Path(source).suffix.lower() == ".mp4" else "image")
    if method == "image":
        script = "batch_to_splat.py" if opts.batch_launcher else "image_to_splat.py"
        return subprocess.call([sys.executable, str(REPO_ROOT/script), *rest], cwd=REPO_ROOT)
    vp = argparse.ArgumentParser(description="Train a splat from a manually approved 360 MP4.")
    vp.add_argument("--input", required=True)
    vp.add_argument("--experiment")
    vp.add_argument("--approved-video", action="store_true")
    vp.add_argument("--iterations", type=int, default=15000)
    vp.add_argument("--stride", type=int, default=2)
    vp.add_argument("--max-height", type=int, default=1280)
    args = vp.parse_args(rest)
    if not args.experiment:
        if not sys.stdin.isatty(): vp.error("--experiment is required")
        args.experiment = input("Experiment name or number: ").strip()
    if not args.approved_video:
        if sys.stdin.isatty():
            print("Review the complete video: the person should stay still while the camera makes a full turn.")
            args.approved_video = input("Have you reviewed and approved this video? [y/N]: ").strip().lower() == "y"
    try:
        cmd = video_command(sys.executable, vars(args))
    except ValueError as exc:
        vp.error(str(exc))
    return subprocess.call(cmd, cwd=REPO_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())

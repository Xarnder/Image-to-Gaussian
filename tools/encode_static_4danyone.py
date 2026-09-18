#!/usr/bin/env python3
"""Encode one still image as a static 4DAnyone input video.

Runs the cover-crop ffmpeg recipe:
scale to fill 1080x1920, center-crop, 121 frames at 24 fps, libx264 CRF 0.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

TARGET_W = 1080
TARGET_H = 1920
NUM_FRAMES = 121
FPS = 24


def default_output(source: Path) -> Path:
    return source.with_name(f"{source.stem}_static_4danyone.mp4")


def encode(
    source: Path,
    output: Path,
    *,
    width: int = TARGET_W,
    height: int = TARGET_H,
    frames: int = NUM_FRAMES,
    fps: int = FPS,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg was not found on PATH")
    if not source.is_file():
        raise SystemExit(f"source image does not exist: {source}")

    output.parent.mkdir(parents=True, exist_ok=True)
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height}"
    )
    cmd = [
        ffmpeg,
        "-y",
        "-loop",
        "1",
        "-i",
        str(source),
        "-vf",
        vf,
        "-r",
        str(fps),
        "-frames:v",
        str(frames),
        "-c:v",
        "libx264",
        "-crf",
        "0",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    print("+", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg failed with exit code {result.returncode}")
    if not output.is_file() or output.stat().st_size == 0:
        raise SystemExit(f"ffmpeg produced no output: {output}")
    print(f"wrote {output} ({output.stat().st_size} bytes)", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Source still image (PNG/JPEG)")
    parser.add_argument(
        "-o",
        "--output",
        help="Output mp4 path (default: <stem>_static_4danyone.mp4 next to the source)",
    )
    parser.add_argument("--width", type=int, default=TARGET_W)
    parser.add_argument("--height", type=int, default=TARGET_H)
    parser.add_argument("--frames", type=int, default=NUM_FRAMES)
    parser.add_argument("--fps", type=int, default=FPS)
    args = parser.parse_args()

    source = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else default_output(source)
    encode(source, output, width=args.width, height=args.height, frames=args.frames, fps=args.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

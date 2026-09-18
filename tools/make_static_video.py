#!/usr/bin/env python3
"""Build a completely static N-frame video from one PNG (no interpolation)."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageChops, ImageStat


TARGET_W = 1080
TARGET_H = 1920
NUM_FRAMES = 121
FPS = 24


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, check=False, text=True, **kwargs)
    if result.returncode != 0:
        raise SystemExit(f"Command failed ({result.returncode}): {' '.join(cmd)}")
    return result


def letterbox(src: Path, dest: Path, width: int, height: int) -> tuple[int, int]:
    image = Image.open(src).convert("RGBA")
    # Composite onto opaque black so video encoders do not see a checkerboard.
    canvas = Image.new("RGB", image.size, (0, 0, 0))
    canvas.paste(image, mask=image.split()[-1] if image.mode == "RGBA" else None)
    src_w, src_h = canvas.size
    scale = min(width / src_w, height / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = canvas.resize((new_w, new_h), Image.Resampling.LANCZOS)
    padded = Image.new("RGB", (width, height), (0, 0, 0))
    padded.paste(resized, ((width - new_w) // 2, (height - new_h) // 2))
    dest.parent.mkdir(parents=True, exist_ok=True)
    padded.save(dest, format="PNG")
    return padded.size


def encode(frame_png: Path, output: Path, frames: int, fps: int) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    encoders = run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True).stdout
    attempts: list[list[str]] = []
    common = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-loop",
        "1",
        "-framerate",
        str(fps),
        "-i",
        str(frame_png),
        "-frames:v",
        str(frames),
        "-an",
        "-movflags",
        "+faststart",
    ]
    if "libx264rgb" in encoders:
        attempts.append(
            common
            + ["-c:v", "libx264rgb", "-pix_fmt", "rgb24", "-crf", "0", "-preset", "medium", str(output)]
        )
    attempts.append(
        common
        + ["-c:v", "libx264", "-pix_fmt", "yuv444p", "-crf", "0", "-preset", "medium", str(output)]
    )
    attempts.append(
        common
        + ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "0", "-preset", "veryslow", str(output)]
    )
    last_error = "no encoder attempt ran"
    for cmd in attempts:
        result = subprocess.run(cmd, check=False, text=True, capture_output=True)
        print("+", " ".join(cmd), flush=True)
        if result.returncode == 0 and output.is_file() and output.stat().st_size > 0:
            return cmd[cmd.index("-c:v") + 1]
        last_error = result.stderr or result.stdout or f"exit {result.returncode}"
        print(last_error, file=sys.stderr, flush=True)
    raise SystemExit(f"Could not encode static video: {last_error}")


def probe(path: Path) -> dict[str, object]:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames,width,height,codec_name,pix_fmt,avg_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    return stream


def extract_frame(video: Path, index: int, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vf",
            f"select=eq(n\\,{index})",
            "-frames:v",
            "1",
            str(dest),
        ]
    )


def compare_frames(a: Path, b: Path) -> dict[str, float]:
    im_a = Image.open(a).convert("RGB")
    im_b = Image.open(b).convert("RGB")
    if im_a.size != im_b.size:
        raise SystemExit(f"Decoded frames differ in size: {im_a.size} vs {im_b.size}")
    diff = ImageChops.difference(im_a, im_b)
    stats = ImageStat.Stat(diff)
    mean = sum(stats.mean) / 3.0
    extrema = diff.getextrema()
    max_channel = max(channel[1] for channel in extrema)
    return {"mean_abs_diff": mean, "max_channel_diff": float(max_channel)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--width", type=int, default=TARGET_W)
    parser.add_argument("--height", type=int, default=TARGET_H)
    parser.add_argument("--frames", type=int, default=NUM_FRAMES)
    parser.add_argument("--fps", type=int, default=FPS)
    args = parser.parse_args()

    source = Path(args.input)
    output = Path(args.output)
    if not source.is_file():
        print(f"ERROR: source image does not exist: {source}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="static-video-") as tmp:
        tmp_dir = Path(tmp)
        framed = tmp_dir / "framed.png"
        size = letterbox(source, framed, args.width, args.height)
        encoder = encode(framed, output, args.frames, args.fps)
        first = tmp_dir / "frame_000.png"
        last = tmp_dir / f"frame_{args.frames - 1:03d}.png"
        extract_frame(output, 0, first)
        extract_frame(output, args.frames - 1, last)
        comparison = compare_frames(first, last)
        shutil.copyfile(framed, output.with_suffix(".png"))

    info = probe(output)
    read_frames = int(info.get("nb_read_frames") or 0)
    width = int(info["width"])
    height = int(info["height"])
    codec = str(info.get("codec_name"))
    report = {
        "source": str(source),
        "output": str(output),
        "encoder": encoder,
        "requested_frames": args.frames,
        "decoded_frames": read_frames,
        "width": width,
        "height": height,
        "codec": codec,
        "pix_fmt": info.get("pix_fmt"),
        "framed_size": list(size),
        "frame0_vs_last": comparison,
    }
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)

    if read_frames != args.frames:
        print(f"ERROR: expected {args.frames} frames, decoded {read_frames}", file=sys.stderr)
        return 1
    if width != args.width or height != args.height:
        print(f"ERROR: expected {args.width}x{args.height}, got {width}x{height}", file=sys.stderr)
        return 1
    if comparison["max_channel_diff"] > 4:
        print(
            f"ERROR: first and last frames differ too much: {comparison}",
            file=sys.stderr,
        )
        return 1
    print("static video verified", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

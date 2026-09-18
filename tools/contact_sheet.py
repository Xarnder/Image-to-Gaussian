#!/usr/bin/env python3
"""Extract one synchronized frame from every dense view and build a contact sheet."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def find_dense_videos(root: Path) -> list[Path]:
    candidates = sorted(root.glob("videos/dense/*.mp4"))
    if candidates:
        return candidates
    candidates = sorted(root.glob("**/videos/dense/*.mp4"))
    if candidates:
        return candidates
    raise SystemExit(f"No dense view videos found under {root}")


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
            "-vsync",
            "vfr",
            "-frames:v",
            "1",
            str(dest),
        ]
    )


def temporal_stats(video: Path, sample_count: int = 8) -> dict[str, float]:
    """Mean absolute difference vs frame 0 on a few sampled later frames."""
    import av

    container = av.open(str(video))
    stream = container.streams.video[0]
    frames: list[np.ndarray] = []
    for index, frame in enumerate(container.decode(stream)):
        if index == 0 or index in {30, 60, 90, 120} or index % max(1, 120 // sample_count) == 0:
            frames.append(frame.to_ndarray(format="rgb24"))
        if index >= 120:
            break
    container.close()
    if len(frames) < 2:
        return {"samples": float(len(frames)), "mean_abs_diff_vs_frame0": 0.0}
    first = frames[0].astype(np.float32)
    diffs = [float(np.mean(np.abs(frame.astype(np.float32) - first))) for frame in frames[1:]]
    return {
        "samples": float(len(frames)),
        "mean_abs_diff_vs_frame0": float(sum(diffs) / len(diffs)),
        "max_mean_abs_diff_vs_frame0": float(max(diffs)),
    }


def make_contact_sheet(images: list[tuple[str, Path]], dest: Path, title: str) -> None:
    opened = [(label, Image.open(path).convert("RGB")) for label, path in images]
    width, height = opened[0][1].size
    cols = min(3, len(opened))
    rows = (len(opened) + cols - 1) // cols
    pad = 16
    label_h = 36
    sheet = Image.new(
        "RGB",
        (cols * width + (cols + 1) * pad, rows * (height + label_h) + (rows + 1) * pad + 40),
        (18, 18, 18),
    )
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((pad, 12), title, fill=(240, 240, 240), font=font)
    for index, (label, image) in enumerate(opened):
        row, col = divmod(index, cols)
        x = pad + col * (width + pad)
        y = 40 + pad + row * (height + label_h + pad)
        sheet.paste(image, (x, y))
        draw.text((x, y + height + 8), label, fill=(220, 220, 220), font=font)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dest)
    print(f"wrote {dest}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--temporal-stats", action="store_true")
    args = parser.parse_args()

    root = Path(args.input_dir)
    videos = find_dense_videos(root)
    frame_dir = root / f"review_frame_{args.frame_index:03d}"
    labeled: list[tuple[str, Path]] = []
    stats: dict[str, object] = {}
    for video in videos:
        dest = frame_dir / f"{video.stem}.png"
        extract_frame(video, args.frame_index, dest)
        labeled.append((video.stem, dest))
        if args.temporal_stats:
            stats[video.stem] = temporal_stats(video)

    make_contact_sheet(
        labeled,
        Path(args.output),
        f"{root.name}  frame={args.frame_index}  views={len(labeled)}",
    )
    if stats:
        stats_path = Path(args.output).with_suffix(".temporal.json")
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(json.dumps(stats, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

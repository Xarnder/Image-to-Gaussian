"""Aspect-preserving MiniMax input sizing, independently computed for each image."""
import math
import json
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np

SIZING_VERSION = "fit-pad-per-image-v1"


def image_layout(source):
    with Image.open(source) as raw:
        image = ImageOps.exif_transpose(raw)
        width, height = image.size
    # Keep within the previous square workflow's pixel budget. No forced upscaling.
    scale = min(1., math.sqrt(1248 * 1248 / (width * height)), 1920 / max(width, height))
    while True:
        resized = (max(1, round(width * scale)), max(1, round(height * scale)))
        canvas = tuple(max(64, math.ceil(n / 32) * 32) for n in resized)
        if canvas[0] * canvas[1] <= 1248 * 1248:
            break
        scale *= .99
    left, top = (canvas[0] - resized[0]) // 2, (canvas[1] - resized[1]) // 2
    return {"version": SIZING_VERSION, "source_size": [width, height], "resized_size": list(resized),
            "video_size": list(canvas), "padding": [left, top, canvas[0]-resized[0]-left, canvas[1]-resized[1]-top],
            "crop": False}


def prepare_video_image(source, dest, layout):
    with Image.open(source) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
        image = image.resize(tuple(layout["resized_size"]), Image.Resampling.LANCZOS)
        left, top, right, bottom = layout["padding"]
        padded = np.pad(np.asarray(image), ((top, bottom), (left, right), (0, 0)), mode="edge")
        Image.fromarray(padded).save(dest)
    return dest


def legacy_aspect_mismatch(source, work):
    """Old square-cropped exports must not be skipped when restarting a mixed batch."""
    frame = work / "frames" / "frame_0000.png"
    manifest = work / "input.json"
    if manifest.is_file():
        try:
            clip = Path(json.loads(manifest.read_text())["path"])
            sizing = clip.parent / "image_layout.json"
            if sizing.is_file() and json.loads(sizing.read_text()).get("version") == SIZING_VERSION:
                return False
        except (OSError, ValueError, KeyError):
            pass
    if not frame.is_file():
        return False
    with Image.open(source) as raw, Image.open(frame) as extracted:
        width, height = ImageOps.exif_transpose(raw).size
        fw, fh = extracted.size
    return abs((fw / fh) / (width / height) - 1) > .04

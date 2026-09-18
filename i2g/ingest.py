"""S0: matte the photo and place the person on a fixed portrait canvas with a neutral grey background.

The canvas (832x1248, 2:3) is the size Qwen-Image-Edit works at (a FLUX Kontext preferred bucket), so
every generated view shares one set of intrinsics. A native-resolution crop of the same framing is
kept for training the photo view at full detail.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

CANVAS_W, CANVAS_H = 832, 1248
PERSON_FRAC = 0.86   # person height / canvas height
GREY = 0.5


def _crop_padded(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, fill) -> np.ndarray:
    """Crop [x0, x1) x [y0, y1), padding with `fill` where the box leaves the image."""
    h, w = img.shape[:2]
    out = np.empty((y1 - y0, x1 - x0) + img.shape[2:], dtype=img.dtype)
    out[...] = fill
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return out


def composite_on_grey(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = alpha[..., None].astype(np.float32)
    return np.clip(rgb.astype(np.float32) * a + 255.0 * GREY * (1 - a), 0, 255).round().astype(np.uint8)


def ingest(image_path: Path, out_dir: Path, matter) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb = np.array(Image.open(image_path).convert("RGB"))
    alpha = matter(rgb)

    ys, xs = np.nonzero(alpha > 0.5)
    if len(ys) == 0:
        raise RuntimeError(f"no person found in {image_path}")
    bx0, bx1, by0, by1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1

    # crop box in photo pixels with the canvas aspect, person centred at PERSON_FRAC of the height
    crop_h = (by1 - by0) / PERSON_FRAC
    crop_w = crop_h * CANVAS_W / CANVAS_H
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    x0, y0 = int(round(cx - crop_w / 2)), int(round(cy - crop_h / 2))
    x1, y1 = x0 + int(round(crop_w)), y0 + int(round(crop_h))

    hr_rgb = _crop_padded(rgb, x0, y0, x1, y1, 0)
    hr_alpha = _crop_padded(alpha, x0, y0, x1, y1, 0.0)
    canvas_rgb = cv2.resize(hr_rgb, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_AREA)
    canvas_alpha = np.clip(cv2.resize(hr_alpha, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_AREA), 0, 1)

    Image.fromarray(composite_on_grey(canvas_rgb, canvas_alpha)).save(out_dir / "canvas.png")
    Image.fromarray((canvas_alpha * 255).round().astype(np.uint8)).save(out_dir / "alpha.png")
    hr_rgba = np.dstack([hr_rgb, (hr_alpha * 255).round().astype(np.uint8)])
    Image.fromarray(hr_rgba).save(out_dir / "source_hr_rgba.png")

    meta = {
        "source": str(image_path),
        "source_size": [int(rgb.shape[1]), int(rgb.shape[0])],
        "person_bbox_source": [bx0, by0, bx1, by1],
        "crop_box_source": [x0, y0, x1, y1],
        "canvas_size": [CANVAS_W, CANVAS_H],
        "canvas_px_per_source_px": CANVAS_H / (y1 - y0),
        "hr_size": [x1 - x0, y1 - y0],
    }
    (out_dir / "ingest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta

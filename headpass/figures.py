"""Comparison figures for the head-pass arms of one subject (written into <exp>/report/)."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def arms_of(exp: Path) -> dict[str, Path]:
    cfg = json.loads((exp / "config.json").read_text())
    arms = {"B0 product": Path(cfg["product_final"])}
    for name, label in (("e1_coh2", "E1 coherent standin"), ("e2_merged", "E2 +head pass"),
                        ("e3_merged", "E3 head pass on old clips")):
        p = exp / "arms" / name / "character.ply"
        if p.exists():
            arms[label] = p
    return arms


def neck_sheet(plys: dict[str, Path], world: dict, maps: dict, out: Path, az=(0, 45, 90, 180, -90, -45), size: int = 384):
    """Close-ups centred on the neck band, to see the head/body seam."""
    from i2g.cameras import look_at
    from i2g.evaluate import load_ply, render
    from .headtrain import neck_band
    from .views import head_K
    h0, h1 = neck_band(maps)
    c = np.array(maps["center"], float)
    target = np.array([c[0], (h0 + h1) / 2, c[2]])
    K = head_K(size, dist=0.7, fov_m=0.30)
    rows = []
    for name, ply in plys.items():
        params = load_ply(ply)
        tiles = []
        for a in az:
            t = np.radians(a)
            ref_dir = np.asarray(world["photo_w2c"])[2, :3]            # camera forward in world
            fwd = -ref_dir
            base = np.arctan2(fwd[0], fwd[2])
            eye = target + 0.7 * np.array([np.sin(base + t), 0.0, np.cos(base + t)])
            img, _ = render(params, look_at(eye, target), K, size, size)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.putText(img, f"{a}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
            tiles.append(img)
        strip = np.full((size, 170, 3), 35, np.uint8)
        for j, part in enumerate(name.split(" ")):
            cv2.putText(strip, part[:20], (6, 26 + 22 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1)
        rows.append(np.hstack([strip] + tiles))
        del params
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])


def make(exp: Path) -> Path:
    from .experiment import load
    from .views import depth_sheet, head_sheet
    _, world = load(exp)
    maps = dict(np.load(exp / "standins" / "coherent2_maps.npz"))
    arms = arms_of(exp)
    rep = exp / "report"
    head_sheet(arms, world, rep / "head_eye.jpg", az=(0, 30, 60, 90, -30, -60, -90, 180))
    head_sheet(arms, world, rep / "head_above.jpg", az=(0, 45, 90, -45, -90, 180), el=25)
    head_sheet(arms, world, rep / "head_below.jpg", az=(0, 45, -45), el=-15)
    depth_sheet(arms, world, rep / "head_geometry.jpg", az=(0, 30, 60, 90, -45, -90))
    neck_sheet(arms, world, maps, rep / "neck.jpg")
    return rep

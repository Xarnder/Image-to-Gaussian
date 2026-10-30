"""Score arms of one subject on the head, the same way for every arm.

  kink_mm     forehead kink (headpass.metrics.forehead_step): the largest change of depth slope walking up the
              forehead, max over views at 0, +-30, +-45 deg around the head (landmarks found on each render).
              A face pasted on as a plate shows its top edge here. A smooth bald head scores ~2 mm; a real
              hairline or hat brim also scores, so compare arms of the same subject only.
  id_mean/worst  ArcFace similarity to the photo over 21 head cameras (az 0, +-20, +-40, +-60; el -15, 0, +20).
              Independent of the generated clips.
  photo_db    masked PSNR against the photo, at the native-resolution photo camera, head region only.
  sharp       mean |Laplacian| inside the head over the 21 cameras (softness proxy; compare within a subject).
  holdout_*   against held-out frames of the head clips (psnr / sharp / speckle as pacgs.facecheck), only for
              arms scored on the same generated set.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from i2g.log import get_logger

log = get_logger("headpass.evaluate")
GRID_AZ = (0, -20, 20, -40, 40, -60, 60)
GRID_EL = (-15, 0, 20)


class _LmRef:
    """What metrics.forehead_step needs, for an arbitrary camera."""

    def __init__(self, lm, K, w2c, h, w):
        self.lm, self.K, self.w2c, self.h, self.w = lm, K, w2c, h, w


def kink(params, world, geo, size: int = 768) -> dict:
    from i2g.evaluate import face_camera, render
    from .metrics import forehead_step, render_depth
    from .views import head_K
    K = head_K(size, fov_m=0.30)
    out = {}
    for az in (0, -30, 30, -45, 45):
        w2c = face_camera(world, az, 0)
        img, _ = render(params, w2c, K, size, size)
        lm = geo.landmarks(img)
        if lm is None:
            out[az] = None
            continue
        d, a = render_depth(params, w2c, K, size, size)
        out[az] = forehead_step(_LmRef(lm, K, w2c, size, size), d, a)["forehead_jump_mm"]
    vals = [v for v in out.values() if v is not None]
    return {"kink_mm": round(max(vals), 1) if vals else None, "kink_by_az": out}


def profile_dent(params, maps, half_width_bins: int = 10) -> dict:
    """The dip in the side profile between the brow and the crown. On the model's radius map around the
    head centre, take the meridians within +-half_width_bins of the face's azimuth; each gives a 2D profile
    curve (forward, up) from the upper forehead (40 deg) to the crown (88 deg); below 40 is the brow ridge,
    whose natural concavity (~6 mm on the body mesh) is not a defect. That stretch of a skull is convex, so the
    depth of the curve below its own convex hull is the dent (mm). Median over those meridians, and worst."""
    from scipy.spatial import ConvexHull
    from . import headsurf as hs
    frame = {"center": maps["center"], "cut_y": float(maps["cut_y"])}
    r = hs.splat_radius_map(params, frame).cpu().numpy()
    ia0 = int(round(np.nonzero(maps["core"])[1].mean()))
    lo, hi = int(40 + 89.5), int(88 + 89.5)
    el = np.radians(np.arange(lo, hi) + 0.5 - 90)
    dents, where = [], []
    for ia in range(ia0 - half_width_bins, ia0 + half_width_bins + 1):
        col = r[lo:hi, ia % hs.N_AZ]
        ok = np.isfinite(col)
        if ok.sum() < 10:
            continue
        pts = np.stack([col[ok] * np.cos(el[ok]), col[ok] * np.sin(el[ok])], 1)
        pts = np.vstack([pts, [[0.0, 0.0]]])                   # the centre closes the fan
        hull = ConvexHull(pts)
        # distance of each profile point inside the hull to the nearest hull edge
        eq = hull.equations                                    # a x + b y + c <= 0 inside, unit normals
        d = -(pts[:-1] @ eq[:, :2].T + eq[:, 2])               # >= 0 inside
        dd = d.min(1)
        dents.append(float(dd.max()) * 1000)
        where.append(float(np.degrees(el[ok][int(dd.argmax())])))
    return {"dent_mm": round(float(np.median(dents)), 1) if dents else None,
            "dent_worst_mm": round(float(np.max(dents)), 1) if dents else None,
            "dent_at_el": round(float(np.median(where)), 0) if where else None}


def identity_and_sharp(params, world, face, e_ref, size: int = 512) -> dict:
    from i2g.evaluate import face_camera, render
    from .views import head_K
    K = head_K(size)
    ids, sharp = [], []
    for el in GRID_EL:
        for az in GRID_AZ:
            img, a = render(params, face_camera(world, az, el), K, size, size)
            det = face.detect(img)
            ids.append(float(face.embed(img, det[1]) @ e_ref) if det else 0.0)
            g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
            lap = np.abs(cv2.Laplacian(g, cv2.CV_32F))
            m = cv2.erode((a > 0.9).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            sharp.append(float(lap[m].mean()) if m.any() else 0.0)
    return {"id_mean": round(float(np.mean(ids)), 3), "id_worst": round(float(np.min(ids)), 3),
            "id_side60": round(float(np.mean([ids[i] for i in range(len(ids)) if abs(GRID_AZ[i % len(GRID_AZ)]) == 60])), 3),
            "sharp": round(float(np.mean(sharp)), 2)}


def photo_db(params, exp: Path, world, maps) -> float:
    from i2g.evaluate import _psnr, render
    from pacgs.proxy import photo_source
    from .headtrain import head_zone, neck_band
    from .metrics import render_depth
    src, K = photo_source(exp, world, 2.0)
    from PIL import Image
    rgba = np.asarray(Image.open(src).convert("RGBA"))
    h, w = rgba.shape[:2]
    img, _ = render(params, world["photo_w2c"], K, w, h)
    d, a = render_depth(params, world["photo_w2c"], K, w, h)
    vs, us = np.mgrid[:h, :w].astype(np.float32)
    cam = np.stack([(us + 0.5 - K[0, 2]) / K[0, 0] * d, (vs + 0.5 - K[1, 2]) / K[1, 1] * d, d], -1).reshape(-1, 3)
    R, t = world["photo_w2c"][:3, :3], world["photo_w2c"][:3, 3]
    pts = torch.as_tensor((cam - t) @ R, device="cuda").float()
    h0, _ = neck_band(maps)
    m = head_zone(pts, maps["center"], h0).view(h, w).cpu().numpy() & (rgba[..., 3] > 127) & (a > 0.5)
    gt = (rgba[..., :3].astype(np.float32) * (rgba[..., 3:] / 255.0) + 127.5 * (1 - rgba[..., 3:] / 255.0))
    return round(_psnr(img, gt.round().astype(np.uint8), m), 2)


def holdout(params, exp: Path, gset: str, world, rigs=("face", "face_up", "face_down", "face_high", "face_top")) -> dict:
    from i2g.evaluate import render
    from pacgs.facecheck import _metrics
    from pacgs.orbit import layout_cameras, rig_tag
    from PIL import Image
    root = exp / "gen" / gset
    meta = json.loads((root / "gen.json").read_text())
    width, height = meta["size"]
    rows = []
    for rig in rigs:
        if rig not in meta["rigs"]:
            continue
        cams = layout_cameras(world, rig, meta["frames"], width, height)
        hold = set(json.loads((root / "data" / rig / "qa.json").read_text())["summary"]["holdout"])
        tag = rig_tag(rig)
        for cam in cams:
            name = f"{tag}{cam['arc']}f_s{cam['index']}"
            if name not in hold:
                continue
            frame = np.asarray(Image.open(root / "frames" / rig / f"arc{cam['arc']}" / f"frame_{cam['index']:04d}.png").convert("RGB"))
            matte = np.asarray(Image.open(root / "data" / rig / "mattes" / f"{tag}{cam['arc']}f{cam['index']:03d}.png")) > 127
            img, _ = render(params, cam["w2c"], cam["K"], width, height)
            # score the head: the upper part of the matte around the head centre projection
            from i2g.cameras import project
            uv, z = project(np.asarray(json.loads((exp / "standins" / "coherent2.json").read_text())["center"])[None],
                            cam["w2c"], cam["K"])
            r = int(0.16 * cam["K"][1, 1] / max(float(z[0]), 0.1))       # a 32 cm box around the head
            box = np.zeros_like(matte)
            cx, cy = int(uv[0, 0]), int(uv[0, 1])
            box[max(0, cy - r):cy + r, max(0, cx - r):cx + r] = True
            m = matte & box
            if m.sum() > 500:
                rows.append(_metrics(img, frame, m))
    if not rows:
        return {}
    return {f"holdout_{k}": round(float(np.mean([r[k] for r in rows])), 3) for k in ("psnr", "sharp", "speckle")} | \
        {"holdout_n": len(rows)}


def score(exp: Path, arms: dict[str, Path], gset: str | None = None) -> dict:
    from i2g.evaluate import FaceID, load_ply
    from .facegeo import FaceGeo
    from .experiment import load
    from PIL import Image
    cfg, world = load(exp)
    maps = dict(np.load(exp / "standins" / "coherent2_maps.npz"))
    face, geo = FaceID(), FaceGeo()
    src = np.asarray(Image.open(exp / "00_ingest" / "source_hr_rgba.png").convert("RGB"))
    e_ref = face.embed(src, face.detect(src)[1])
    out = {}
    for name, ply in arms.items():
        params = load_ply(ply)
        r = kink(params, world, geo) | profile_dent(params, maps) | identity_and_sharp(params, world, face, e_ref)
        r["photo_db"] = photo_db(params, exp, world, maps)
        if gset:
            r |= holdout(params, exp, gset, world)
        out[name] = r
        log.info("%-22s %s", name, {k: v for k, v in r.items() if k != "kink_by_az"})
        del params
        torch.cuda.empty_cache()
    return out

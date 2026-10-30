"""H0: "the face is a mask" as numbers.

face relief (photo view)
    From the photo's camera, read the splat's depth at the 68 landmark pixels the 3D landmark network finds in
    the photo, and compare the depth *pattern* with the network's own estimate (headpass.facegeo). Both are
    expressed relative to the nose tip, in millimetres (the network's pixel units are converted with the
    photo camera's focal length and the face's distance).
      sides_mm   (nose tip -> sides of the face at the tragus) splat minus reference. > 0: the nose stands
                 too far in front of the sides, i.e. the face sits forward of the head (a mask floating out);
                 < 0: the face is sunk into the head.
      inner_mm   median |error| over brows, eyes, nose and mouth: is the face's own relief right, or flat?
hidden layer
    A pasted face is a separate surface with the head behind it. Drop every Gaussian within `gap_m` of the
    front surface along each face pixel's ray, render what is left: the alpha still covering the face is
    a second surface hiding behind the first (share of face pixels with alpha > 0.5).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .facegeo import EYES, JAW_UPPER, NOSE_TIP, FaceGeo

INNER = list(range(17, 68))


def hr_camera(subject, world: dict):
    from pacgs.proxy import photo_source
    src, K = photo_source(subject, world, 2.0)
    from PIL import Image
    img = Image.open(src)
    rgb = np.asarray(img.convert("RGB")).copy()
    return rgb, K, np.asarray(world["photo_w2c"], float)


@torch.no_grad()
def render_depth(params: dict, w2c: np.ndarray, K: np.ndarray, width: int, height: int):
    """(expected depth HxW, alpha HxW) of a splat."""
    from gsplat import rasterization
    dev = params["means"].device
    colors = torch.zeros(len(params["means"]), 3, device=dev)
    out, alpha, _ = rasterization(params["means"], torch.nn.functional.normalize(params["quats"], dim=-1),
                                  torch.exp(params["scales"]), torch.sigmoid(params["opacities"]), colors,
                                  torch.as_tensor(w2c, device=dev).float()[None],
                                  torch.as_tensor(K, device=dev).float()[None], width, height,
                                  render_mode="RGB+ED", packed=False, near_plane=0.01, far_plane=100.0)
    a = alpha[0, ..., 0]
    return out[0, ..., 3].cpu().numpy(), a.cpu().numpy()


def _sample(depth: np.ndarray, alpha: np.ndarray, uv: np.ndarray, r: int = 2) -> np.ndarray:
    h, w = depth.shape
    vals = []
    for u, v in uv:
        u, v = int(round(u)), int(round(v))
        win_d = depth[max(0, v - r):v + r + 1, max(0, u - r):u + r + 1]
        win_a = alpha[max(0, v - r):v + r + 1, max(0, u - r):u + r + 1]
        ok = win_a > 0.9
        vals.append(float(np.median(win_d[ok])) if ok.any() else np.nan)
    return np.array(vals)


def _inward(lm: np.ndarray, idx, frac: float = 0.12) -> np.ndarray:
    """Contour landmarks moved a little towards the nose, off the face's silhouette edge."""
    return lm[idx, :2] + frac * (lm[NOSE_TIP, :2] - lm[idx, :2])


class Reference:
    """The photo's landmarks and the network's depth pattern, computed once per subject."""

    def __init__(self, subject, world: dict, geo: FaceGeo | None = None):
        self.rgb, self.K, self.w2c = hr_camera(subject, world)
        geo = geo or FaceGeo()
        self.lm = geo.landmarks(self.rgb)
        if self.lm is None:
            raise RuntimeError("no face found in the photo")
        self.h, self.w = self.rgb.shape[:2]
        self.uv_inner = self.lm[INNER, :2]
        self.uv_sides = _inward(self.lm, list(JAW_UPPER))

    def mm_per_px(self, z_face_m: float) -> float:
        return 1000.0 * z_face_m / float(self.K[0, 0])

    def reference_pattern(self, z_face_m: float) -> dict:
        s = self.mm_per_px(z_face_m)
        z = self.lm[:, 2]
        return {"inner": (z[NOSE_TIP] - z[INNER]) * s, "sides": (z[NOSE_TIP] - z[list(JAW_UPPER)]) * s}


def face_relief(ref: Reference, depth: np.ndarray, alpha: np.ndarray) -> dict:
    """Signed errors of a depth map against the reference pattern (see module docstring)."""
    nose = _sample(depth, alpha, ref.lm[[NOSE_TIP], :2])[0]
    if not np.isfinite(nose):
        return {"sides_mm": None, "inner_mm": None}
    pat = ref.reference_pattern(nose)
    inner = (_sample(depth, alpha, ref.uv_inner) - nose) * 1000.0      # mm behind the nose tip
    sides = (_sample(depth, alpha, ref.uv_sides) - nose) * 1000.0
    # the reference's sides are at the true contour; ours are 12% of the way in towards the nose, where the
    # depth is interpolated linearly (a cheek curves, so this slightly understates the true depth there)
    ref_sides = pat["sides"] * (1.0 - 0.12)
    return {"sides_mm": round(float(np.nanmean(sides - ref_sides)), 1),
            "sides_depth_mm": round(float(np.nanmean(sides)), 1),
            "ref_sides_mm": round(float(np.mean(ref_sides)), 1),
            "inner_mm": round(float(np.nanmedian(np.abs(inner - pat["inner"]))), 1),
            "nose_depth_m": round(float(nose), 4)}


@torch.no_grad()
def hidden_layer(params: dict, ref: Reference, depth: np.ndarray, alpha: np.ndarray, gap_m: float = 0.015,
                 slab_m: float = 0.05) -> float:
    """Share of face pixels covered (alpha > 0.5) by Gaussians between gap_m and slab_m behind the front surface:
    a second face/skull layer. (The back of the skull is ~18 cm behind the face, well outside the slab.)"""
    dev = params["means"].device
    w2c = torch.as_tensor(ref.w2c, device=dev).float()
    K = torch.as_tensor(ref.K, device=dev).float()
    cam = params["means"] @ w2c[:3, :3].T + w2c[:3, 3]
    u = (K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2]).long().clamp(0, ref.w - 1)
    v = (K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]).long().clamp(0, ref.h - 1)
    front = torch.as_tensor(depth, device=dev)[v, u]
    keep = (cam[:, 2] > front + gap_m) & (cam[:, 2] < front + slab_m)
    sub = {k: t[keep] for k, t in params.items()}
    _, a2 = render_depth(sub, ref.w2c, ref.K, ref.w, ref.h)
    face = _face_mask(ref) & (alpha > 0.9)
    return round(float((a2[face] > 0.5).mean()), 3) if face.any() else float("nan")


def forehead_step(ref: Reference, depth: np.ndarray, alpha: np.ndarray) -> dict:
    """Walk up the forehead in the photo view, from the brows to 1.3x the brow-to-nose-tip height above them
    (about the hairline; the crown beyond turns away from the camera and would dominate): the largest
    change in depth *slope* (mm, second difference over 3 px steps) on three columns (brow centre and above
    each outer brow). A face pasted on as a plate shows its top edge as a kink of centimetres; a continuous
    forehead curves back smoothly. Measured on the body mesh too, as the smooth-head baseline."""
    jumps = []
    span = 1.3 * (ref.lm[NOSE_TIP, 1] - ref.lm[[21, 22], 1].mean())
    for i0, i1 in ((21, 22), (17, 18), (25, 26)):
        u = int(round((ref.lm[i0, 0] + ref.lm[i1, 0]) / 2))
        v0 = int(round(min(ref.lm[i0, 1], ref.lm[i1, 1])))
        v1 = max(0, int(round(v0 - span)))
        d = depth[v1:v0, u][::-1] * 1000.0                 # from the brow upwards, mm
        a = alpha[v1:v0, u][::-1]
        d = d[a > 0.9]
        if len(d) > 9:
            slope = d[3:] - d[:-3]
            jumps.append(float(np.max(np.abs(slope[3:] - slope[:-3]))))
    return {"forehead_jump_mm": round(float(np.max(jumps)), 1) if jumps else None,
            "forehead_jump_mean_mm": round(float(np.mean(jumps)), 1) if jumps else None}


def _face_mask(ref: Reference) -> np.ndarray:
    import cv2
    hull = cv2.convexHull(np.concatenate([ref.lm[:27, :2]]).astype(np.int32))
    m = np.zeros((ref.h, ref.w), np.uint8)
    cv2.fillConvexPoly(m, hull, 1)
    return m.astype(bool)


def mesh_depth(world: dict, ref: Reference):
    """The body mesh's own depth from the photo camera (for the same numbers on SAM 3D Body's head)."""
    from i2g.priors import MeshPrior
    prior = MeshPrior(world, "cuda")
    _, depth, alpha = prior.render(torch.as_tensor(ref.w2c, device="cuda").float(),
                                   torch.as_tensor(ref.K, device="cuda").float(), ref.w, ref.h)
    del prior
    return depth.cpu().numpy(), alpha.cpu().numpy()


def measure_ply(ply, ref: Reference) -> dict:
    from i2g.evaluate import load_ply
    params = load_ply(ply)
    depth, alpha = render_depth(params, ref.w2c, ref.K, ref.w, ref.h)
    out = face_relief(ref, depth, alpha)
    out["hidden_layer"] = hidden_layer(params, ref, depth, alpha)
    out |= forehead_step(ref, depth, alpha)
    del params
    torch.cuda.empty_cache()
    return out

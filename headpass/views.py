"""Head close-up renders of a splat: the sheet used to look at the face from the side."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SHEET_AZ = (0, 30, 60, 90, -60, -90)


def head_K(size: int = 512, dist: float = 0.9, fov_m: float = 0.36) -> np.ndarray:
    """A square lens that frames about fov_m metres at the head's distance."""
    f = size * dist / fov_m
    return np.array([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1.0]])


def head_sheet(plys: dict[str, Path], world: dict, out: Path, az=SHEET_AZ, el: float = 0.0, size: int = 384):
    """One row per splat, one column per azimuth around the head (0 = the photo's view of the head)."""
    from i2g.evaluate import face_camera, load_ply, render
    K = head_K(size)
    rows = []
    for name, ply in plys.items():
        params = load_ply(ply)
        tiles = []
        for a in az:
            img, _ = render(params, face_camera(world, a, el), K, size, size)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.putText(img, f"{a}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
            tiles.append(img)
        strip = np.full((size, 170, 3), 35, np.uint8)
        for j, part in enumerate(name.split("/")):
            cv2.putText(strip, part[:20], (6, 26 + 22 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1)
        rows.append(np.hstack([strip] + tiles))
        del params
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return out


def mesh_overlay_sheet(plys: dict[str, Path], world: dict, out: Path, az=(0, 45, 90, -45, -90), el: float = 0.0,
                       size: int = 384):
    """Head renders with the body mesh's outline (red) and its face region outline (green) on top: shows
    where the splat's face and skull sit relative to one coherent head surface."""
    import torch
    from i2g.evaluate import face_camera, load_ply, render
    from i2g.priors import MeshPrior
    K = head_K(size)
    prior = MeshPrior(world, "cuda")
    outlines = []
    for a in az:
        w2c = torch.as_tensor(face_camera(world, a, el), device="cuda").float()
        face, depth, alpha = prior.render(w2c, torch.as_tensor(K, device="cuda").float(), size, size)
        outlines.append(((alpha > 0.5).cpu().numpy().astype(np.uint8), (face > 0.5).cpu().numpy().astype(np.uint8)))
    del prior
    rows = []
    for name, ply in plys.items():
        params = load_ply(ply)
        tiles = []
        for a, (m, f) in zip(az, outlines):
            img, _ = render(params, face_camera(world, a, el), K, size, size)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            for mask, col in ((m, (0, 0, 255)), (f, (0, 255, 0))):
                cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(img, cs, -1, col, 1)
            cv2.putText(img, f"{a}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
            tiles.append(img)
        strip = np.full((size, 170, 3), 35, np.uint8)
        for j, part in enumerate(name.split("/")):
            cv2.putText(strip, part[:20], (6, 26 + 22 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1)
        rows.append(np.hstack([strip] + tiles))
        del params
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return out


def shaded_depth(depth: np.ndarray, alpha: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Grey Lambert shading of a depth map from its normals (light at the camera): shows geometry only."""
    h, w = depth.shape
    vs, us = np.mgrid[:h, :w].astype(np.float32)
    x = (us + 0.5 - K[0, 2]) / K[0, 0] * depth
    y = (vs + 0.5 - K[1, 2]) / K[1, 1] * depth
    P = np.stack([x, y, depth], -1)
    dx = np.zeros_like(P); dy = np.zeros_like(P)
    dx[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dy[1:-1] = P[2:] - P[:-2]
    n = np.cross(dx, dy)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    v = -P / (np.linalg.norm(P, axis=-1, keepdims=True) + 1e-9)
    shade = np.clip(np.abs((n * v).sum(-1)), 0, 1) ** 1.5
    g = np.where(alpha > 0.5, 40 + 215 * shade, 128).astype(np.uint8)
    return np.repeat(g[..., None], 3, -1)


def depth_sheet(plys: dict[str, Path], world: dict, out: Path, az=SHEET_AZ, el: float = 0.0, size: int = 384):
    """Like head_sheet, but the depth the video model is given (sharpened, median-filtered), shaded."""
    from i2g.evaluate import face_camera
    from pacgs.proxy import SplatProxy
    K = head_K(size)
    rows = []
    for name, ply in plys.items():
        proxy = SplatProxy(Path(ply))
        tiles = []
        for a in az:
            d, al = proxy.depth(face_camera(world, a, el), K, size, size)
            img = shaded_depth(d, al, K)
            cv2.putText(img, f"{a}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 160, 255), 2)
            tiles.append(img)
        strip = np.full((size, 170, 3), 35, np.uint8)
        for j, part in enumerate(name.split("/")):
            cv2.putText(strip, part[:20], (6, 26 + 22 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1)
        rows.append(np.hstack([strip] + tiles))
        del proxy
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return out

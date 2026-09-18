"""Camera math. Cameras are OpenCV-convention (x right, y down, z forward) world-to-camera 4x4
matrices; the world is Y-up, in metres, with the person facing +Z (see body.build_world)."""
from __future__ import annotations

import math

import cv2
import numpy as np


def intrinsics_from_focal(focal: float, width: int, height: int) -> np.ndarray:
    return np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]])


def focal_from_vfov(vfov_deg: float, height: int) -> float:
    return (height / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)


def look_at(center: np.ndarray, target: np.ndarray, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    fwd = target - center
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, np.asarray(up, dtype=float))
    right = right / np.linalg.norm(right)
    down = np.cross(fwd, right)
    w2c = np.eye(4)
    w2c[:3, :3] = np.stack([right, down, fwd])
    w2c[:3, 3] = -w2c[:3, :3] @ center
    return w2c


def orbit_camera(target: np.ndarray, radius: float, az_deg: float, el_deg: float) -> np.ndarray:
    """Camera on a sphere around `target`; az=0 is in front of the person (+Z), az=90 on their left (+X)."""
    az, el = math.radians(az_deg), math.radians(el_deg)
    offset = radius * np.array([math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el)])
    return look_at(target + offset, target)


def camera_center(w2c: np.ndarray) -> np.ndarray:
    return -w2c[:3, :3].T @ w2c[:3, 3]


def azimuth_elevation(w2c: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    v = camera_center(w2c) - target
    r = float(np.linalg.norm(v))
    return math.degrees(math.atan2(v[0], v[2])) % 360.0, math.degrees(math.asin(v[1] / r)), r


def project(points: np.ndarray, w2c: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cam = points @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam[:, 2]
    uv = (cam[:, :2] / np.maximum(z[:, None], 1e-6)) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    return uv, z


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity (s, R, t) minimising |dst - (s R src + t)|^2."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / (xs ** 2).sum(1).mean())
    t = mu_d - s * R @ mu_s
    return s, R, t


def refine_pnp(points_w: np.ndarray, uv: np.ndarray, K: np.ndarray, w2c0: np.ndarray) -> np.ndarray:
    """Levenberg-Marquardt reprojection refinement of a camera, starting from w2c0."""
    rvec, _ = cv2.Rodrigues(w2c0[:3, :3])
    tvec = w2c0[:3, 3].reshape(3, 1).copy()
    rvec, tvec = cv2.solvePnPRefineLM(points_w.astype(np.float64), uv.astype(np.float64),
                                      K.astype(np.float64), None, rvec, tvec)
    w2c = np.eye(4)
    w2c[:3, :3] = cv2.Rodrigues(rvec)[0]
    w2c[:3, 3] = tvec.ravel()
    return w2c


def render_silhouette(vertices: np.ndarray, faces: np.ndarray, w2c: np.ndarray, K: np.ndarray,
                      width: int, height: int) -> np.ndarray:
    uv, z = project(vertices, w2c, K)
    mask = np.zeros((height, width), np.uint8)
    tris = np.round(uv[faces] * 16).astype(np.int32)  # 4 fractional bits for sub-pixel accuracy
    for tri in tris[(z[faces] > 0).all(1)]:
        cv2.fillConvexPoly(mask, tri, 255, lineType=cv2.LINE_8, shift=4)
    return mask

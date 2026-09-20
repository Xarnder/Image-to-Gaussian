"""Depth control videos from a 3D proxy of the person, rendered with gsplat: the trainer's own
renderer, so control frames and training cameras share one projection model.

  mesh:  the SAM 3D Body mesh (research plan P2). Exact body, but no hair or clothing volume.
  splat: an existing splat of the person, e.g. the v2 result (P4). Carries hair, clothing and shoe
         volume, at the cost of softer edges.

The depth is encoded like a monocular depth estimate, which is what Fun-Control was trained on:
inverse depth, near = bright, background black. One normalisation covers the whole orbit, so a surface
keeps its grey level as the camera moves.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from i2g.priors import sample_surface

SUBJECT_LO = 0.2       # grey level of the farthest subject pixel (background is 0)


class MeshProxy:
    """The body mesh as ~1.5M tiny opaque Gaussians (no mesh rasteriser needed; like priors.MeshPrior)."""

    def __init__(self, world: dict, device: str = "cuda", n: int = 1_500_000, splat_m: float = 0.002):
        pts = sample_surface(world, n, seed=1)
        self.means = torch.from_numpy(pts).float().to(device)
        self.quats = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(n, 1)
        self.scales = torch.full((n, 3), splat_m, device=device)
        self.opacities = torch.full((n,), 0.99, device=device)
        self.colors = torch.zeros(n, 3, device=device)

    @torch.no_grad()
    def depth(self, w2c: np.ndarray, K: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        from gsplat import rasterization
        dev = self.means.device
        out, alpha, _ = rasterization(self.means, self.quats, self.scales, self.opacities, self.colors,
                                      viewmats=torch.as_tensor(w2c, device=dev).float()[None],
                                      Ks=torch.as_tensor(K, device=dev).float()[None], width=width, height=height,
                                      render_mode="RGB+ED", packed=False, near_plane=0.01, far_plane=100.0)
        return out[0, ..., 3].cpu().numpy(), alpha[0, ..., 0].cpu().numpy()


class SplatProxy:
    """An existing splat (INRIA-layout PLY, e.g. a v2 character.ply).

    Depth is rendered with sharpened opacities, o' = 1 - (1 - o)^sharpen, then median-filtered. Plain
    expected depth averages semi-transparent hair with the surfaces behind it, which shows up as dark
    blotches (dents) on the hair and upper back; sharpening makes the first surface dominate."""

    def __init__(self, ply: Path, device: str = "cuda", sharpen: float = 8.0, median_px: int = 5):
        from i2g.evaluate import load_ply
        self.params = load_ply(ply, device)
        self.sh_degree = int(round(math.sqrt(self.params["shN"].shape[1] + 1))) - 1
        o = torch.sigmoid(self.params["opacities"])
        o = 1 - (1 - o).clamp_min(1e-6) ** sharpen
        self.depth_params = dict(self.params, opacities=torch.logit(o.clamp(1e-6, 1 - 1e-6)))
        self.median_px = median_px

    @torch.no_grad()
    def depth(self, w2c: np.ndarray, K: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        from i2g.splat import render
        dev = self.params["means"].device
        _, alpha, info = render(self.depth_params, torch.as_tensor(w2c, device=dev).float(),
                                torch.as_tensor(K, device=dev).float(), width, height, self.sh_degree,
                                with_depth=True)
        depth = info["depth"][..., 0].cpu().numpy().astype(np.float32)
        if self.median_px > 1:
            depth = cv2.medianBlur(depth, self.median_px)
        return depth, alpha[..., 0].cpu().numpy()

    @torch.no_grad()
    def rgb(self, w2c: np.ndarray, K: np.ndarray, width: int, height: int) -> np.ndarray:
        from i2g.splat import render_rgb
        dev = self.params["means"].device
        return render_rgb(self.params, torch.as_tensor(w2c, device=dev).float(),
                          torch.as_tensor(K, device=dev).float(), width, height)


def make_proxy(kind: str, world: dict, splat_ply: Path | None = None):
    if kind == "mesh":
        return MeshProxy(world)
    if kind == "splat":
        if splat_ply is None or not Path(splat_ply).exists():
            raise FileNotFoundError(f"splat proxy needs an existing PLY, got {splat_ply}")
        return SplatProxy(Path(splat_ply))
    raise ValueError(f"unknown proxy kind {kind!r} (mesh or splat)")


def inverse_depth_range(proxy, cams: list[dict], width: int, height: int) -> tuple[float, float]:
    """(far, near) inverse depth over the subject pixels of the whole orbit (0.5 / 99.5 percentiles)."""
    samples = []
    for cam in cams:
        depth, alpha = proxy.depth(cam["w2c"], cam["K"], width, height)
        d = depth[alpha > 0.5]
        if len(d):
            samples.append(1.0 / np.maximum(d[:: max(1, len(d) // 4000)], 1e-3))
    inv = np.concatenate(samples)
    return float(np.percentile(inv, 0.5)), float(np.percentile(inv, 99.5))


def encode_depth(depth: np.ndarray, alpha: np.ndarray, inv_range: tuple[float, float],
                 blur_sigma: float = 0.8) -> np.ndarray:
    """HxWx3 uint8 control frame: inverse depth, near white, subject in [SUBJECT_LO, 1], background 0."""
    inv_far, inv_near = inv_range
    t = np.clip((1.0 / np.maximum(depth, 1e-3) - inv_far) / (inv_near - inv_far), 0.0, 1.0)
    v = np.where(alpha > 0.5, SUBJECT_LO + (1.0 - SUBJECT_LO) * t, 0.0).astype(np.float32)
    if blur_sigma > 0:      # estimator depth has soft edges; a hard 1-px staircase is out of distribution
        v = cv2.GaussianBlur(v, (0, 0), blur_sigma)
    g = (v * 255.0).round().astype(np.uint8)
    return np.repeat(g[..., None], 3, axis=2)


def reframe_photo(source: Path, source_K: np.ndarray, K_video: np.ndarray, width: int, height: int,
                  background: float = 0.5) -> Image.Image:
    """The photo as the video camera at orbit 0 sees it. Same camera centre and orientation, so any
    change of focal length or principal point is an exact crop and scale (PIL box resampling).
    Zoomed rigs (the head close-up) should pass the native-resolution photo, which has the detail."""
    img = Image.open(source)
    if img.mode == "RGBA":
        flat = Image.new("RGB", img.size, tuple([int(background * 255)] * 3))
        flat.paste(img, mask=img.getchannel("A"))
        img = flat
    img = img.convert("RGB")
    s = K_video[1, 1] / source_K[1, 1]                    # video pixels per source pixel
    x0 = source_K[0, 2] - K_video[0, 2] / s
    y0 = source_K[1, 2] - K_video[1, 2] / s
    box = [x0, y0, x0 + width / s, y0 + height / s]
    pad = [max(0, math.ceil(-box[0])), max(0, math.ceil(-box[1])),
           max(0, math.ceil(box[2] - img.width)), max(0, math.ceil(box[3] - img.height))]
    if any(pad):                                          # the frame reaches past the photo: pad, don't crop it
        from PIL import ImageOps
        img = ImageOps.expand(img, tuple(pad), fill=tuple([int(background * 255)] * 3))
        box = [box[0] + pad[0], box[1] + pad[1], box[2] + pad[0], box[3] + pad[1]]
    return img.resize((width, height), Image.Resampling.LANCZOS, box=tuple(box))


def photo_source(subject: Path, world: dict, zoom: float) -> tuple[Path, np.ndarray]:
    """(image, its K) for the start image: the canvas normally, the native-resolution photo when zoomed in."""
    canvas_K = np.asarray(world["K"], float)
    if zoom <= 1.0:
        return subject / "00_ingest" / "canvas.png", canvas_K
    hr = subject / "00_ingest" / "source_hr_rgba.png"
    if not hr.exists():
        return subject / "00_ingest" / "canvas.png", canvas_K
    with Image.open(hr) as img:
        scale = img.size[1] / (2 * canvas_K[1, 2])
    K = canvas_K.copy()
    K[0] *= scale
    K[1] *= scale
    return hr, K

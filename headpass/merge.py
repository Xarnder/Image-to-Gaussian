"""H4: put the head pass onto the body pass, joined at the neck instead of around the face.

Ownership (world y against the neck band [h0, h1] from headtrain.neck_band, head zone from headtrain.head_zone):
  * inside the head zone above h1: the head pass only (the body's head is deleted)
  * below h0, or outside the head zone (shoulders, torso, long hair on the shoulders): the body pass only
  * inside the band: a dither. Each head Gaussian is kept with probability w = smoothstep((y - h0) / (h1 - h0)),
    each body Gaussian with 1 - w. Multiplying opacities instead (two layers at 50 %) would cover only 75 %
    and leave a translucent ring; random thinning of two dense sets is a true cross-dissolve.
Then an optional short fine-tune of only the band's Gaussians on all frames (fine_tune), to close what the
dither leaves (colour or small geometric mismatch between the two passes).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from i2g.log import get_logger

from .headtrain import head_zone, neck_band

log = get_logger("headpass.merge")


@torch.no_grad()
def merge_params(body: dict, head: dict, maps: dict, seed: int = 0) -> tuple[dict, dict]:
    """(merged params, stats) - see the module docstring."""
    h0, h1 = neck_band(maps)
    dev = body["means"].device
    g = torch.Generator(device=dev).manual_seed(seed)

    def w_of(y):
        t = ((y - h0) / (h1 - h0)).clamp(0, 1)
        return t * t * (3 - 2 * t)

    yb = body["means"][:, 1]
    zb = head_zone(body["means"], maps["center"], h0)
    rb = torch.rand(len(yb), generator=g, device=dev)
    keep_b = ~zb | (yb < h0) | ((yb <= h1) & (rb >= w_of(yb)))
    yh = head["means"][:, 1]
    zh = head_zone(head["means"], maps["center"], h0)
    rh = torch.rand(len(yh), generator=g, device=dev)
    keep_h = zh & ((yh > h1) | ((yh >= h0) & (rh < w_of(yh))))
    b = {k: v[keep_b].detach() for k, v in body.items()}
    h = {k: v[keep_h].detach() for k, v in head.items()}
    if b["shN"].shape[1] != h["shN"].shape[1]:           # match SH degree (pad the lower one with zeros)
        n = max(b["shN"].shape[1], h["shN"].shape[1])
        for d in (b, h):
            if d["shN"].shape[1] < n:
                d["shN"] = torch.cat([d["shN"], torch.zeros(len(d["shN"]), n - d["shN"].shape[1], 3, device=dev)], 1)
    out = {k: torch.cat([b[k], h[k]], 0) for k in b}
    stats = {"h0": round(h0, 4), "h1": round(h1, 4), "body_kept": int(keep_b.sum()), "body_total": len(yb),
             "head_kept": int(keep_h.sum()), "head_total": len(yh)}
    return out, stats


@torch.no_grad()
def merge(body_ply: Path, head_ply: Path, maps_npz: Path, out_ply: Path, seed: int = 0) -> dict:
    from i2g.evaluate import load_ply
    from i2g.splat import export_ply
    out, stats = merge_params(load_ply(body_ply), load_ply(head_ply), dict(np.load(maps_npz)), seed)
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    export_ply(out, out_ply)
    log.info("Merged: %s", stats)
    return stats

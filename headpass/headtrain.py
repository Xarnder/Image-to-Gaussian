"""H3: the head pass. A splat of the head and hair only, trained on the head close-up clips and the photo, with
its own Gaussian budget and loss schedule, anchored to the coherent head surface (H2) by a soft leash.

From CAP4D / GaussianAvatars:
  * the leash: a Gaussian may sit anywhere within tau of its surface and is pulled back beyond that
    (relu(|d| - tau)), instead of our old FaceShell, which projected face Gaussians onto a 2 mm shell every
    step and was measured to hurt. Skin directions (the face core and the blend band) are leashed both
    ways; hair directions only against sinking into the head.
  * the loss is restricted to the head: pixels whose stand-in surface is not head (arms, shoulders, the
    torso the head clips also show) are excluded, not taught as background.
  * a perceptual weight that ramps up late in training (CAP4D ramps LPIPS to 0.9 of its loss).

The neck: the head pass owns everything above h1 (just under the chin) and fades out over a band down to h0;
the body pass owns everything below. The head's loss zone reaches 2 cm below h0 so it has content across the
whole band.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from i2g.log import get_logger

from . import headsurf as hs

log = get_logger("headpass.headtrain")

HEAD_RIGS = ("face", "face_up", "face_down", "face_high", "face_top")


def neck_band(maps: dict, band_m: float = 0.05) -> tuple[float, float]:
    """(h0, h1): the neck blend band's bottom and top (world y). h1 is 1 cm under the chin."""
    h1 = float(maps["chin"][1]) - 0.01
    return h1 - band_m, h1


def head_zone(p: torch.Tensor, center, h0: float, reach_m: float = 0.28, neck_r_m: float = 0.10) -> torch.Tensor:
    """World points that belong to the head pass: above h0 - 2 cm, near the head, and below the face only
    inside a neck column (shoulders, collar and long hair on the shoulders stay with the body pass)."""
    c = torch.as_tensor(center, device=p.device).float()
    d = p - c
    horiz = torch.sqrt(d[:, 0] ** 2 + d[:, 2] ** 2)
    return (p[:, 1] > h0 - 0.02) & (d.norm(dim=-1) < reach_m) & ((d[:, 1] > -0.08) | (horiz < neck_r_m))


@torch.no_grad()
def loss_mask(proxy, w2c, K, width: int, height: int, center, h0: float, frame_alpha: np.ndarray,
              grow_px: int = 6) -> np.ndarray:
    """0 only on the body: pixels whose stand-in surface is not head (torso, shoulders, arms in front of the
    face) and that the frame's matte calls foreground. Everything else - the head and ALL background - is in
    the loss, so the head pass learns to be transparent everywhere but the head. (A first version kept only
    the head plus a 12 px margin of background: Gaussians above the head were never told to vanish and grew
    into coloured floaters.)"""
    depth, alpha = proxy.depth(w2c, K, width, height)
    vs, us = np.mgrid[:height, :width].astype(np.float32)
    x = (us + 0.5 - K[0, 2]) / K[0, 0] * depth
    y = (vs + 0.5 - K[1, 2]) / K[1, 1] * depth
    cam = np.stack([x, y, depth], -1).reshape(-1, 3)
    R, t = np.asarray(w2c)[:3, :3], np.asarray(w2c)[:3, 3]
    pts = torch.as_tensor((cam - t) @ R, device="cuda").float()
    zone = head_zone(pts, center, h0).view(height, width).cpu().numpy() & (alpha > 0.5)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow_px + 1, 2 * grow_px + 1))
    zone = cv2.dilate(zone.astype(np.uint8), k) > 0
    body = ((alpha > 0.5) | (frame_alpha > 0.5)) & ~zone
    return cv2.GaussianBlur((~body).astype(np.float32), (0, 0), 2.0)


class Leash:
    """relu(|rho - r| - tau) in skin directions, relu(r - rho - tau_in) (no sinking) elsewhere; metres."""

    def __init__(self, maps: dict, weight: float = 10.0, tau_m: float = 0.004, tau_in_m: float = 0.015,
                 start: int = 0):
        dev = "cuda"
        self.c = torch.as_tensor(maps["center"], device=dev).float()
        self.r = hs._fill_nan(torch.as_tensor(maps["r"], device=dev).float())
        self.skin = torch.as_tensor(maps["core"] | maps["band"], device=dev)
        self.weight, self.tau, self.tau_in, self.start = weight, tau_m, tau_in_m, start

    def __call__(self, params, step: int):
        if step < self.start or self.weight <= 0:
            return torch.zeros((), device=self.c.device)
        ie, ia, rho = hs._to_bins(params["means"], self.c)
        d = rho - self.r[ie, ia]
        skin = self.skin[ie, ia]
        pen = torch.where(skin, F.relu(d.abs() - self.tau), F.relu(-d - self.tau_in))
        return self.weight * pen.mean()


def head_records(data_root: Path, rigs) -> list[dict]:
    """The photo plus every head-rig frame that is not held out, from prepared clip folders
    (<data_root>/<rig>/records.json and qa.json, as pacgs.dataset.prepare writes them)."""
    records, holdout = [], set()
    for rig in rigs:
        if not (data_root / rig / "records.json").exists():
            continue
        rows = json.loads((data_root / rig / "records.json").read_text())
        records += rows if not records else rows[1:]
        holdout |= set(json.loads((data_root / rig / "qa.json").read_text())["summary"]["holdout"])
    return [r for r in records if r["tag"] == "photo" or f"{r['tag']}_s{r['seed']}" not in holdout]


def train_head_core(records: list[dict], world: dict, cfg, maps: dict, standin_ply: Path, hr_photo: Path | None,
                    leash_weight: float = 0.0, out_dir: Path | None = None, bar=None) -> tuple[dict, dict]:
    """Train the head pass. `cfg` is the orbit pass's SplatConfig with the head's iterations and perceptual ramp
    already set. Returns (params restricted to the head zone, report).

    The leash is off by default: pulling face Gaussians towards the stand-in's surface fought the geometry Wan
    actually painted and left blotches (measured, headpass_results.md 3.1)."""
    from i2g.splat import clean, config_dict, load_views, train
    from pacgs.proxy import SplatProxy
    h0, h1 = neck_band(maps)
    cfg.face_mesh_init = False          # else the trainer swaps the start points on the face for body-mesh points
    views = load_views(records, world["K"], cfg, "cuda", hr_photo=hr_photo, world=world)
    proxy = SplatProxy(Path(standin_ply))
    masks = []
    for i in range(len(views["names"])):
        H, W = views["images"][i].shape[:2]
        m = loss_mask(proxy, views["w2c"][i].cpu().numpy(), views["K"][i].cpu().numpy(), W, H, maps["center"], h0,
                      views["alphas"][i][..., 0].cpu().numpy())
        masks.append(torch.as_tensor(m, device="cuda")[..., None])
    views["loss_masks"] = masks
    # start from the stand-in's Gaussians in the head zone (the coherent surface + the hair kept on it)
    sp = proxy.params
    zone = head_zone(sp["means"], maps["center"], h0)
    SH_C0 = 0.28209479177387814
    init = (sp["means"][zone].clone(), (sp["sh0"][zone, 0] * SH_C0 + 0.5).clamp(0, 1).clone())
    del proxy
    torch.cuda.empty_cache()
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / "mask_photo.png"), (masks[0].cpu().numpy()[..., 0] * 255).astype(np.uint8))
    log.info("Head pass: %d views, %d start points, %d iterations, leash %.1f, perceptual -> %.2f",
             len(views["names"]), len(init[0]), cfg.iterations, leash_weight, cfg.perceptual_ramp_to)
    t0 = time.perf_counter()
    extra = Leash(maps, weight=leash_weight) if leash_weight > 0 else None
    params, stats = train(views, world, cfg, bar, init=init, extra_loss=extra)
    refined = torch.tensor(stats.pop("refined_w2c"), device="cuda")
    report = {"h0": h0, "h1": h1, "config": config_dict(cfg), "leash_weight": leash_weight,
              "views": len(views["names"]), "train_minutes": round((time.perf_counter() - t0) / 60, 1)}
    report["clean"] = clean(params, views, refined, cfg)
    keep = head_zone(params["means"], maps["center"], h0)
    params = {k: v[keep].detach() for k, v in params.items()}
    report["gaussians"] = int(keep.sum())
    del views
    torch.cuda.empty_cache()
    log.info("Head pass done: %d Gaussians, %.1f min", report["gaussians"], report["train_minutes"])
    return params, report


def train_head(exp: Path, arm: str, gset: str, rigs=HEAD_RIGS, iterations: int = 30000,
               leash_weight: float = 0.0, perceptual_ramp_to: float = 0.6, overrides: dict | None = None,
               standin: str = "coherent") -> dict:
    """Research runner: the head pass on an experiment folder's prepared set (see headpass/experiment.py)."""
    from i2g.splat import export_ply
    from .experiment import load, product_config
    cfg, world = load(exp)
    maps = dict(np.load(exp / "standins" / f"{standin}_maps.npz"))
    records = head_records(exp / "gen" / gset / "data", rigs)
    c = product_config(cfg, list(rigs), {"iterations": iterations, "perceptual_ramp_to": perceptual_ramp_to,
                                         **(overrides or {})})
    out = exp / "arms" / arm
    params, report = train_head_core(records, world, c, maps, exp / "standins" / f"{standin}.ply",
                                     exp / "00_ingest" / "source_hr_rgba.png" if cfg["options"]["photo_hr"] else None,
                                     leash_weight, out)
    report |= {"arm": arm, "gset": gset, "rigs": list(rigs)}
    export_ply(params, out / "head.ply")
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    return report

"""Render-and-compare camera fitting: align generated views to a splat trained on the other views.

Keypoint registration (register.py) relies on SAM 3D Body reading the pose from each generated image,
which is unreliable for pure profiles (the far arm and leg are hidden) and loose for some back views.
With a first splat in hand the camera can be fitted directly instead: minimise the difference between
the splat's silhouette and colours and the view's matte and image, coarse to fine, from a few starting
cameras. This rescues views the keypoints placed badly (the true side views) and tightens the rest.

    python3 -m i2g.reregister --run output/test-image/v2 --ply <splat.ply> \\
        --registration output/test-image/v2/03_qa/registration_full.json --out <fitted.json>
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .cameras import azimuth_elevation, orbit_camera
from .log import get_logger, setup_logging

log = get_logger("reregister")
MIN_FIT_IOU = 0.85          # silhouette agreement with the splat needed to accept a fitted view


def _shrink(t: torch.Tensor, s: int) -> torch.Tensor:
    return F.interpolate(t.permute(2, 0, 1)[None], scale_factor=1 / s, mode="area")[0].permute(1, 2, 0)


def fit_camera(params: dict, w2c0: np.ndarray, K: np.ndarray, rgb: torch.Tensor, matte: torch.Tensor,
               color_weight: float = 0.3, schedule=((8, 150), (4, 100), (2, 60))) -> tuple[np.ndarray, float]:
    """(fitted w2c, final loss). rgb/matte: (H, W, 3)/(H, W, 1) float tensors on the GPU."""
    from .splat import apply_delta, render
    device = rgb.device
    sh_degree = int(round(math.sqrt(params["shN"].shape[1] + 1))) - 1
    w2c = torch.as_tensor(w2c0, device=device).float()
    delta = torch.zeros(6, device=device, requires_grad=True)
    opt = torch.optim.Adam([{"params": [delta], "lr": 2e-3}])
    loss = torch.tensor(0.0)
    for s, n in schedule:
        h, w = rgb.shape[0] // s, rgb.shape[1] // s
        k = torch.as_tensor(K, device=device).float().clone()
        k[0] *= w / rgb.shape[1]
        k[1] *= h / rgb.shape[0]
        t_rgb, t_mat = _shrink(rgb, s), _shrink(matte, s)
        for _ in range(n):
            out, alpha, _ = render(params, apply_delta(w2c, delta), k, w, h, sh_degree)
            loss = (alpha - t_mat).abs().mean() + color_weight * (out - t_rgb * t_mat).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return apply_delta(w2c, delta).cpu().numpy().astype(np.float64), float(loss)


@torch.no_grad()
def silhouette_iou(params: dict, w2c: np.ndarray, K: np.ndarray, matte: np.ndarray) -> float:
    from .evaluate import render
    _, alpha = render(params, w2c, K, matte.shape[1], matte.shape[0])
    a, m = alpha > 0.5, matte > 0.5
    return float((a & m).sum() / max((a | m).sum(), 1))


def refit_records(params: dict, world: dict, records: list[dict], skip: set[str] = frozenset(),
                  bar=None) -> list[dict]:
    """Fit every generated view's camera; returns updated copies (old camera kept as w2c_keypoints)."""
    from .log import NullBar
    from .register import flipped
    bar = bar or NullBar()
    K = world["K"]
    target = world["target"]
    photo_az, _, radius = azimuth_elevation(world["photo_w2c"], target)
    out = []
    for rec in records:
        rec = dict(rec)
        name = f"{rec['tag']}_s{rec['seed']}"
        if rec["tag"] == "photo" or name in skip or not np.isfinite(rec.get("azimuth_deg", float("nan"))):
            out.append(rec)
            if rec["tag"] != "photo":
                bar.update(1)
            continue
        rgb = torch.from_numpy(np.asarray(Image.open(rec["path"]).convert("RGB"), np.float32) / 255).cuda()
        matte_np = np.asarray(Image.open(rec["mask"]).convert("L"), np.float32) / 255
        matte = torch.from_numpy(matte_np)[..., None].cuda()
        starts = {"keypoints": np.asarray(rec["w2c"]),
                  "orbit": orbit_camera(target, radius, rec["azimuth_deg"], rec["elevation_deg"])}
        if rec.get("requested_deg"):
            photo_el = azimuth_elevation(world["photo_w2c"], target)[1]
            req = rec["requested_deg"]
            starts["requested"] = orbit_camera(target, radius, req[0], req[1] + photo_el)
        fits = {k: fit_camera(params, w, K, rgb, matte) for k, w in starts.items()}
        # a back view fits a front camera almost as well (mirrored silhouette), so never take a flipped fit
        flips = {k: bool(rec.get("requested_deg")) and
                 flipped(azimuth_elevation(v[0], target)[0], rec["requested_deg"][0], photo_az)
                 for k, v in fits.items()}
        best = min([k for k in fits if not flips[k]] or fits, key=lambda k: fits[k][1])
        w2c = fits[best][0]
        iou_before = silhouette_iou(params, np.asarray(rec["w2c"]), K, matte_np)
        iou = silhouette_iou(params, w2c, K, matte_np)
        az, el, r = azimuth_elevation(w2c, target)
        moved = float(np.linalg.norm(-w2c[:3, :3].T @ w2c[:3, 3] + np.asarray(rec["w2c"])[:3, :3].T @ np.asarray(rec["w2c"])[:3, 3]))
        rec.update({"w2c_keypoints": rec["w2c"], "w2c": w2c.tolist(), "azimuth_deg": az, "elevation_deg": el,
                    "radius_m": r, "fit_start": best, "fit_iou": iou, "fit_iou_before": iou_before,
                    "fit_moved_m": moved, "accepted_keypoints": rec.get("accepted")})
        rec["accepted"] = bool(iou >= MIN_FIT_IOU) and not flips[best]
        rec["reasons"] = ([] if iou >= MIN_FIT_IOU else [f"fit IoU {iou:.2f}"]) + \
            (["front/back flipped"] if flips[best] else [])
        log.info("%-10s %-9s az %6.1f el %5.1f  IoU %.3f -> %.3f  moved %.1f cm  %s", name, best, az, el,
                 iou_before, iou, moved * 100, "OK" if rec["accepted"] else "REJECT")
        out.append(rec)
        bar.update(1)
    return out


def main() -> None:
    from . import body
    from .evaluate import load_ply
    from .experiment import HOLDOUT
    from .register import select_views
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--ply", type=Path, required=True)
    ap.add_argument("--registration", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--holdout", default=HOLDOUT, help="views left out of selection (still fitted)")
    args = ap.parse_args()
    setup_logging(args.out.with_suffix(".log"))
    world = body.load_world(args.run / "01_body")
    params = load_ply(args.ply)
    records = refit_records(params, world, json.loads(args.registration.read_text()))
    hold = {s for s in args.holdout.split(",") if s}
    for r in records:
        r["_acc"] = r["accepted"]
        if f"{r['tag']}_s{r['seed']}" in hold:
            r["accepted"] = False
    select_views(records)
    for r in records:
        r["accepted"] = r.pop("_acc")
    args.out.write_text(json.dumps(records, indent=1), encoding="utf-8")
    sel = [(r["tag"], r["seed"], round(r["azimuth_deg"]), round(r["elevation_deg"])) for r in records if r.get("selected")]
    log.info("%d selected: %s", len(sel), sel)


if __name__ == "__main__":
    main()

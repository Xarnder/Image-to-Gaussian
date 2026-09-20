"""Scores for a splat trained on a video dataset.

  * leave-frames-out (LFO): held-out generated frames rendered at their orbit camera (exact), and
    again after fitting the camera to the splat. A gap between the two means the frame's content
    sits "as if" seen from a slightly different camera.
  * the v2 harness (i2g.evaluate) without its Qwen hold-outs: photo fidelity, face identity over a
    face grid, face surface depth, needles, floaters, sharpness, and a sheet row for compare sheets.
  * a turntable video and an 8-angle preview (same cameras as the v2 pipeline's).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from i2g.log import get_logger

log = get_logger("pacgs.evaluate")


def lfo(params: dict, records: list[dict], holdout: list[str], vgg) -> dict:
    from i2g.evaluate import _psnr, _ssim, render
    from i2g.reregister import fit_camera
    by_name = {f"{r['tag']}_s{r['seed']}": r for r in records}
    rows = {}
    for name in holdout:
        r = by_name[name]
        gt = np.asarray(Image.open(r["path"]).convert("RGB"))
        mat = np.asarray(Image.open(r["mask"]).convert("L"), np.float32) / 255.0
        K, w2c = np.asarray(r["K"]), np.asarray(r["w2c"])
        h, w = gt.shape[:2]
        gt_grey = (gt * mat[..., None] + 127.5 * (1 - mat[..., None])).round().astype(np.uint8)
        m = mat > 0.5
        row = {}
        fitted, _ = fit_camera(params, w2c, K, torch.from_numpy(gt.astype(np.float32) / 255).cuda(),
                               torch.from_numpy(mat)[..., None].cuda(), schedule=((4, 100), (2, 60)))
        for tag, cam in (("orbit", w2c), ("fitted", fitted)):
            pred, alpha = render(params, cam, K, w, h)
            a = alpha > 0.5
            row[tag] = {"psnr": round(_psnr(pred, gt_grey, m), 2), "ssim": round(_ssim(pred, gt_grey, m), 4),
                        "vgg": round(vgg(pred, gt_grey), 4), "iou": round(float((a & m).sum() / max((a | m).sum(), 1)), 4)}
        R = fitted[:3, :3] @ w2c[:3, :3].T
        row["fit_rot_deg"] = round(float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))), 3)
        c0, c1 = -w2c[:3, :3].T @ w2c[:3, 3], -fitted[:3, :3].T @ fitted[:3, 3]
        row["fit_move_cm"] = round(float(np.linalg.norm(c1 - c0) * 100), 2)
        rows[name] = row
    out = {"frames": rows}
    for tag in ("orbit", "fitted"):
        for k in ("psnr", "ssim", "vgg", "iou"):
            out[f"lfo_{tag}_{k}"] = round(float(np.mean([v[tag][k] for v in rows.values()])), 4)
    out["lfo_fit_rot_deg_median"] = round(float(np.median([v["fit_rot_deg"] for v in rows.values()])), 3)
    out["lfo_fit_move_cm_median"] = round(float(np.median([v["fit_move_cm"] for v in rows.values()])), 2)
    return out


def turntable(params: dict, world: dict, out_dir: Path) -> None:
    import imageio.v2 as imageio

    from i2g.cameras import azimuth_elevation, orbit_camera
    from i2g.evaluate import render
    K, (w, h) = world["K"], world["size"]
    _, _, radius = azimuth_elevation(world["photo_w2c"], world["target"])
    frames = [render(params, orbit_camera(world["target"], radius, az, 5.0), K, w, h)[0]
              for az in np.linspace(0, 360, 73)[:-1]]
    imageio.mimsave(out_dir / "turntable.mp4", frames, fps=24, quality=8, macro_block_size=8)
    tiles = [frames[j] for j in range(0, 72, 9)]
    grid = np.concatenate([np.concatenate(tiles[:4], 1), np.concatenate(tiles[4:], 1)], 0)
    grid = cv2.resize(grid, (grid.shape[1] // 2, grid.shape[0] // 2), interpolation=cv2.INTER_AREA)
    Image.fromarray(grid).save(out_dir / "preview_8_angles.png")


def evaluate_splat(ply: Path, data_dir: Path, subject_run: Path, out_dir: Path) -> dict:
    from i2g.body import load_world
    from i2g.evaluate import _VGG, evaluate, load_ply
    out_dir.mkdir(parents=True, exist_ok=True)
    records = json.loads((data_dir / "records.json").read_text())
    holdout = json.loads((data_dir / "qa.json").read_text())["summary"]["holdout"]
    vgg = _VGG()
    result = evaluate(subject_run, ply, out_dir, holdout=[], vgg=vgg)
    for key in ("holdout", "holdout_views", "holdout_psnr", "holdout_ssim", "holdout_vgg", "holdout_iou"):
        result.pop(key, None)                  # the Qwen hold-outs are not part of this dataset
    params = load_ply(ply)
    result |= lfo(params, records, holdout, vgg)
    turntable(params, load_world(subject_run / "01_body"), out_dir)
    (out_dir / "eval.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    log.info("%s: LFO PSNR %.2f dB (fitted %.2f), photo %.2f dB, face id %.3f, needles %d, floaters %.3f%%",
             ply.parent.name, result["lfo_orbit_psnr"], result["lfo_fitted_psnr"], result["photo_psnr"],
             result.get("face_mean") or 0.0, result["needles_8x_3cm"], result["floater_pct"])
    del params
    torch.cuda.empty_cache()
    return result

"""Does the face hold up away from the front?

The head close-up clip is the target: it shows the face from every angle, at a size where detail is
visible. So the test is to render a splat from that clip's own cameras and compare, angle by angle:

  psnr       agreement with the generated frame, inside the face region
  sharp      how much of the frame's fine detail the render reproduces (1.0 = the same)
  speckle    fine-grained error the frame does not have: the mottling a pasted-on photo layer causes
  arcface    identity against the photo, where a face is detected

A front-only score hides exactly the failure we are chasing, so every number is reported per angle as
well as averaged, and the sheet puts each render directly under the frame it should match.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from i2g.log import get_logger

log = get_logger("pacgs.facecheck")


def _highpass(grey: np.ndarray) -> np.ndarray:
    return grey - cv2.GaussianBlur(grey, (0, 0), 1.5)


def _metrics(render_rgb: np.ndarray, frame_rgb: np.ndarray, mask: np.ndarray) -> dict:
    from i2g.evaluate import _psnr
    r = cv2.cvtColor(render_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    f = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hr, hf = _highpass(r), _highpass(f)
    detail_f = float(np.abs(hf)[mask].mean())
    return {
        "psnr": round(_psnr(render_rgb, frame_rgb, mask), 2),
        "sharp": round(float(np.abs(hr)[mask].mean()) / max(detail_f, 1e-6), 3),
        # fine-grained error: what the render adds that the frame has no counterpart for
        "speckle": round(float(np.abs(_highpass(r - f))[mask].mean()) / max(detail_f, 1e-6), 3),
    }


def _clip(run: Path, cfg: dict, world: dict, variant: str, every: int):
    """(rig, cameras, frame images, face masks, frame indices) for one generated close-up clip."""
    from i2g.priors import MeshPrior

    from .orbit import layout_cameras, rig_tag
    meta = json.loads((run / "gen" / variant / "variant.json").read_text())
    rig = meta.get("rig") or meta["layout"]
    cams = layout_cameras(world, rig, cfg["frames"], cfg["width"], cfg["height"])
    idx = list(range(0, len(cams) - 1, every))
    mattes, tag = run / "data" / variant / "mattes", rig_tag(rig)
    prior = MeshPrior(world, "cuda")
    frames, masks = {}, {}
    for i in idx:
        frames[i] = np.asarray(Image.open(run / "gen" / variant / f"arc{cams[i]['arc']}" /
                                          f"frame_{i:04d}.png").convert("RGB"))
        m = prior.face_targets(torch.as_tensor(cams[i]["w2c"], device="cuda").float(),
                               torch.as_tensor(cams[i]["K"], device="cuda").float(), cfg["width"], cfg["height"])
        if m is not None:                                   # the face is visible from this camera
            masks[i] = m[2][..., 0].cpu().numpy() > 0.5
        else:                                               # behind the head: score the whole person
            mp = mattes / f"{tag}0f{i:03d}.png"
            masks[i] = (np.asarray(Image.open(mp).convert("L"), np.float32) / 255 > 0.5) if mp.exists() else None
    del prior
    torch.cuda.empty_cache()
    return rig, cams, frames, masks, idx


def check(run: Path, names: list[str], variants: list[str] | str, out_dir: Path, every: int = 8,
          tile: int = 300) -> dict:
    from i2g.body import load_world
    from i2g.evaluate import FaceID, load_ply, render

    variants = variants.split(",") if isinstance(variants, str) else variants
    cfg = json.loads((run / "config.json").read_text())
    world = load_world(Path(cfg["subject"]) / "01_body")
    out_dir.mkdir(parents=True, exist_ok=True)
    face = FaceID()
    src = np.asarray(Image.open(Path(cfg["subject"]) / "00_ingest" / "source_hr_rgba.png").convert("RGB"))
    det = face.detect(src)
    e_ref = face.embed(src, det[1]) if det else None

    clips = {v: _clip(run, cfg, world, v, every) for v in variants}
    result = {"clips": {v: {"rig": c[0], "angles": [round(c[1][i]["orbit_deg"], 1) for i in c[4]]}
                        for v, c in clips.items()}, "splats": {}}
    sheet_rows = []
    for v, (rig, cams, frames, masks, idx) in clips.items():
        sheet_rows.append((f"frames {rig}", [frames[i] for i in idx], cams, idx))

    for name in names:
        ply = run / "splats" / name / "character.ply"
        if not ply.exists():
            log.warning("no splat at %s", ply)
            continue
        params = load_ply(ply)
        by_clip, all_rows = {}, []
        for v, (rig, cams, frames, masks, idx) in clips.items():
            per_angle, images = [], []
            for i in idx:
                img, _ = render(params, cams[i]["w2c"], cams[i]["K"], cfg["width"], cfg["height"])
                images.append(img)
                mask = masks[i]
                row = {"angle": round(cams[i]["orbit_deg"], 1)}
                if mask is not None and mask.any():
                    row |= _metrics(img, frames[i], mask)
                    if e_ref is not None:
                        d = face.detect(img)
                        row["arcface"] = round(float(face.embed(img, d[1]) @ e_ref), 3) if d else None
                per_angle.append(row)
            by_clip[v] = {"rig": rig, "mean": _summarise(per_angle), "per_angle": per_angle}
            all_rows += per_angle
            sheet_rows.append((f"{name} {rig}", images, cams, idx))
            m = by_clip[v]["mean"]
            log.info("%-18s %-10s psnr %.2f  sharp %.2f  speckle %.2f  arcface %s", name, rig, m["psnr"],
                     m["sharp"], m["speckle"], m["arcface"])
        result["splats"][name] = {"mean": _summarise(all_rows), "clips": by_clip}
        del params
        torch.cuda.empty_cache()

    h = int(round(tile * cfg["height"] / cfg["width"]))
    sheet = []
    for label, images, cams, idx in sheet_rows:
        strip = np.full((h, 190, 3), 35, np.uint8)
        for j, part in enumerate(label.split()):
            cv2.putText(strip, part, (6, 26 + 22 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1)
        tiles = [cv2.resize(cv2.cvtColor(im, cv2.COLOR_RGB2BGR), (tile, h), interpolation=cv2.INTER_AREA)
                 for im in images]
        for t, i in zip(tiles, idx):
            cv2.putText(t, f"{cams[i]['orbit_deg']:.0f}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
        sheet.append(np.hstack([strip] + tiles))
    cv2.imwrite(str(out_dir / "facecheck.jpg"), np.vstack(sheet), [cv2.IMWRITE_JPEG_QUALITY, 94])
    (out_dir / "facecheck.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def _summarise(rows: list[dict]) -> dict:
    out = {k: round(float(np.mean([r[k] for r in rows if k in r])), 3) for k in ("psnr", "sharp", "speckle")}
    af = [r["arcface"] for r in rows if r.get("arcface") is not None]
    out["arcface"] = round(float(np.mean(af)), 3) if af else None
    out["arcface_worst"] = round(float(np.min(af)), 3) if af else None
    return out

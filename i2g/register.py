"""S3: register every generated view to the body scaffold, score it, and draw the contact sheet.

Per candidate: matte it, fit SAM 3D Body to it, then find the camera that puts the scaffold's
keypoints onto the view's 2D keypoints. We keep the camera the image was actually rendered from,
not the one we asked for.

Why several hypotheses: a near-symmetric pose (like hands-on-head) looks the same from the front and
from behind once left and right are swapped, and the body fitter sometimes reads a back view as a
front view. So each view gets camera hypotheses from {both left/right labellings} x {the 3D alignment,
its mirror, the requested angle}, each refined by 2D PnP. The winner has the lowest reprojection error
plus a *weak* pull toward the requested angle: strong enough to break front/back ties, too weak to
overrule clear 2D evidence (e.g. when the generator rendered the other side than we asked).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import body
from .cameras import azimuth_elevation, orbit_camera, project, refine_pnp, umeyama
from .comfy_client import ComfyClient
from .log import NullBar, get_logger
from .views import expected_angles

log = get_logger("register")

LR_SWAP = {body.L_EYE: body.R_EYE, body.L_EAR: body.R_EAR, body.L_SHOULDER: body.R_SHOULDER,
           body.L_ELBOW: body.R_ELBOW, body.L_HIP: body.R_HIP, body.L_KNEE: body.R_KNEE,
           body.L_ANKLE: body.R_ANKLE, body.L_WRIST: body.R_WRIST}
LR_SWAP |= {v: k for k, v in LR_SWAP.items()}

# starting thresholds (plan.md S3), tuned on the test image
MAX_DRIFT_M = 0.10          # 3D keypoint residual after similarity alignment (best labelling)
MAX_REPROJ_FRAC = 0.025     # keypoint reprojection RMSE / image height
MIN_COVERAGE = 0.80         # share of the scaffold silhouette inside the view's matte
PRIOR_PER_DEG = 0.005       # score units (% of image height) per degree away from the requested angle
MIN_SEPARATION_DEG = 20.0   # selected views closer than this to each other are redundant


def _direction(az_deg: float, el_deg: float) -> np.ndarray:
    az, el = math.radians(az_deg), math.radians(el_deg)
    return np.array([math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el)])


def _angle_between(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.degrees(math.acos(float(np.clip(_direction(*a) @ _direction(*b), -1.0, 1.0))))


def register_view(world: dict, view: dict, alpha: np.ndarray, expected: tuple[float, float] | None,
                  photo_el: float) -> dict:
    K, target = world["K"], world["target"]
    h = world["size"][1]
    _, _, radius = azimuth_elevation(world["photo_w2c"], target)
    idx = np.array(body.BODY_KP)
    exp = (expected[0], expected[1] + photo_el) if expected else None

    hyps, drifts = [], []
    for swap in (False, True):
        vidx = np.array([LR_SWAP.get(i, i) for i in body.BODY_KP]) if swap else idx
        src, uv = world["kp3d"][idx], view["kp2d"][vidx]
        s, R, t = umeyama(src, view["kp3d"][vidx])
        drifts.append(float(np.linalg.norm(s * src @ R.T + t - view["kp3d"][vidx], axis=1).mean() / s))
        aligned = np.eye(4)
        aligned[:3, :3], aligned[:3, 3] = R, t / s
        a_az, a_el, _ = azimuth_elevation(aligned, target)
        inits = {"aligned": aligned, "mirrored": orbit_camera(target, radius, a_az + 180.0, a_el)}
        if exp:
            inits["requested"] = orbit_camera(target, radius, *exp)
        for origin, w2c0 in inits.items():
            w2c = refine_pnp(src, uv, K, w2c0)
            proj, z = project(src, w2c, K)
            if (z <= 0).any():
                continue
            rmse = float(np.sqrt(((proj - uv) ** 2).sum(1).mean()))
            az, el, r = azimuth_elevation(w2c, target)
            dev = _angle_between((az, el), exp) if exp else 0.0
            hyps.append({"w2c": w2c, "reproj_rmse_px": rmse, "azimuth_deg": az, "elevation_deg": el,
                         "radius_m": r, "prior_dev_deg": dev, "lr_swapped": swap, "init": origin,
                         "score": rmse / h * 100.0 + PRIOR_PER_DEG * dev})
    if not hyps:
        raise ValueError("no camera hypothesis kept the body in front of the camera")
    best = min(hyps, key=lambda hyp: hyp["score"])
    alternatives = sorted({(round(x["azimuth_deg"]), round(x["reproj_rmse_px"], 1), x["init"], x["lr_swapped"])
                           for x in hyps})
    return best | {"drift_m": min(drifts), "hypotheses": [list(a) for a in alternatives]} | \
        body.silhouette_stats(world, best["w2c"], alpha)


def sharpness(rgb: np.ndarray, alpha: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F)[alpha > 0.5].var())


def verdict(rec: dict, height: int) -> list[str]:
    reasons = []
    if rec["drift_m"] > MAX_DRIFT_M:
        reasons.append(f"pose drift {rec['drift_m'] * 1000:.0f} mm")
    if rec["reproj_rmse_px"] > MAX_REPROJ_FRAC * height:
        reasons.append(f"reprojection {rec['reproj_rmse_px']:.1f} px")
    if rec["coverage"] < MIN_COVERAGE:
        reasons.append(f"coverage {rec['coverage']:.2f}")
    return reasons


def select_views(records: list[dict]) -> None:
    """Photo first, then accepted views best-first, skipping ones within MIN_SEPARATION_DEG of a kept view."""
    kept = [records[0]]
    for rec in sorted((r for r in records[1:] if r["accepted"]), key=lambda r: r["score"]):
        angle = (rec["azimuth_deg"], rec["elevation_deg"])
        if all(_angle_between(angle, (k["azimuth_deg"], k["elevation_deg"])) >= MIN_SEPARATION_DEG for k in kept):
            kept.append(rec)
    for rec in records:
        rec["selected"] = any(rec is k for k in kept)


def _cached_sam3d(client: ComfyClient, cache: dict, key: str, path: Path, name: str, vfov_deg: float,
                  npz_dir: Path) -> Path:
    if key in cache and Path(cache[key]).exists():
        return Path(cache[key])
    local = npz_dir / f"{key.split('@')[0]}.npz"
    if not local.exists():
        body.run_sam3d(client, path, f"i2g/{name}/body_views/{key.split('@')[0]}", vfov_deg, local)
    cache[key] = str(local)
    return local


def _find_existing_npz(name: str, key: str) -> Path | None:
    from .paths import COMFY_OUTPUT_DIR
    found = sorted((COMFY_OUTPUT_DIR / "i2g" / name / "body_views").glob(f"{key}_*.npz"))
    return found[-1] if found else None


def register_all(client: ComfyClient, matter, world: dict, dirs: dict, name: str, vfov_deg: float,
                 pairs: set[tuple[str, int]] | None = None, bar=None) -> list[dict]:
    """Register every generated view in views.json (only `pairs` if given); writes registration.json."""
    bar = bar or NullBar()
    manifest = json.loads((dirs["views"] / "views.json").read_text())
    if pairs is not None:
        manifest = [m for m in manifest if (m["tag"], m["seed"]) in pairs]
    masks_dir = dirs["qa"] / "masks"
    # body fits depend on the assumed field of view; 40 degrees is the default the first runs used
    default_fov = abs(vfov_deg - 40.0) < 1e-6
    npz_dir = dirs["qa"] / ("sam3d" if default_fov else f"sam3d_fov{vfov_deg:g}")
    masks_dir.mkdir(parents=True, exist_ok=True)
    npz_dir.mkdir(parents=True, exist_ok=True)
    cache_path = dirs["qa"] / "sam3d_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    w, h = world["size"]

    canvas = np.array(Image.open(dirs["ingest"] / "canvas.png").convert("RGB"))
    canvas_alpha = np.array(Image.open(dirs["ingest"] / "alpha.png")) / 255.0
    photo_sharp = sharpness(canvas, canvas_alpha)
    az, photo_el, radius = azimuth_elevation(world["photo_w2c"], world["target"])
    photo_rec = {"tag": "photo", "seed": 0, "path": str(dirs["ingest"] / "canvas.png"),
                 "mask": str(dirs["ingest"] / "alpha.png"), "w2c": world["photo_w2c"].tolist(),
                 "azimuth_deg": az, "elevation_deg": photo_el, "radius_m": radius, "drift_m": 0.0,
                 "reproj_rmse_px": 0.0, "score": 0.0, "prior_dev_deg": 0.0, "lr_swapped": False,
                 "sharpness_rel": 1.0, "accepted": True, "reasons": []}
    photo_rec |= body.silhouette_stats(world, world["photo_w2c"], canvas_alpha)
    records = [photo_rec]

    for item in manifest:
        key = f"{item['tag']}_s{item['seed']}"
        path = Path(item["path"])
        rgb = np.array(Image.open(path).convert("RGB"))
        if rgb.shape[:2] != (h, w):
            raise ValueError(f"{path} is {rgb.shape[1]}x{rgb.shape[0]}, expected {w}x{h}")
        mask_path = masks_dir / f"{key}.png"
        if mask_path.exists():
            alpha = np.array(Image.open(mask_path)) / 255.0
        else:
            alpha = matter(rgb)
            Image.fromarray((alpha * 255).round().astype(np.uint8)).save(mask_path)
        cache_key = key if default_fov else f"{key}@fov{vfov_deg:g}"
        if default_fov and cache_key not in cache and (existing := _find_existing_npz(name, key)):
            cache[cache_key] = str(existing)

        expected = expected_angles(item["prompt"])
        base = {"tag": item["tag"], "seed": item["seed"], "prompt": item["prompt"], "path": str(path),
                "mask": str(mask_path), "requested_deg": list(expected),
                "sharpness_rel": sharpness(rgb, alpha) / photo_sharp}
        bar.set_postfix_str(key)
        try:
            npz = _cached_sam3d(client, cache, cache_key, path, name, vfov_deg, npz_dir)
            rec = register_view(world, body.load_person(npz), alpha, expected, photo_el)
        except (RuntimeError, KeyError, ValueError, np.linalg.LinAlgError) as exc:
            records.append(base | {"accepted": False, "reasons": [f"registration failed: {exc}"],
                                   "w2c": np.eye(4).tolist(), "azimuth_deg": float("nan"),
                                   "elevation_deg": float("nan"), "drift_m": float("nan"), "score": 1e9,
                                   "reproj_rmse_px": float("nan"), "coverage": 0.0, "iou": 0.0})
            log.warning("view %s could not be registered and is skipped: %s", key, exc)
            bar.update(1)
            continue
        rec = base | rec | {"w2c": rec["w2c"].tolist()}
        rec["reasons"] = verdict(rec, h)
        rec["accepted"] = not rec["reasons"]
        records.append(rec)
        log.debug("%s: asked az %.0f el %.0f -> az %.1f el %.1f (%s%s) drift %.0f mm reproj %.1f px cov %.2f %s",
                  key, expected[0], expected[1], rec["azimuth_deg"], rec["elevation_deg"], rec["init"],
                  ", LR swapped" if rec["lr_swapped"] else "", rec["drift_m"] * 1000, rec["reproj_rmse_px"],
                  rec["coverage"], "OK" if rec["accepted"] else "REJECT " + "; ".join(rec["reasons"]))
        if not rec["accepted"]:
            log.info("view %s rejected: %s", key, "; ".join(rec["reasons"]))
        bar.update(1)

    cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    select_views(records)
    (dirs["qa"] / "registration.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    contact_sheet(world, records, dirs["qa"] / "contact_sheet.png")
    return records


def contact_sheet(world: dict, records: list[dict], out: Path, tile_w: int = 312, cols: int = 6) -> None:
    """Selected views (green) by angle, then accepted-but-redundant (amber), then rejected (red)."""
    w, h = world["size"]
    tile_h = int(round(tile_w * h / w))
    rank = lambda r: 0 if r.get("selected") else (1 if r["accepted"] else 2)  # noqa: E731
    ordered = [records[0]] + sorted(records[1:], key=lambda r: (rank(r), r["elevation_deg"] > 15,
                                                                 r["azimuth_deg"] if r["azimuth_deg"] == r["azimuth_deg"] else 999))
    rows = (len(ordered) + cols - 1) // cols
    sheet = np.full((rows * (tile_h + 44), cols * tile_w, 3), 40, np.uint8)
    for i, rec in enumerate(ordered):
        img = cv2.imread(rec["path"])
        if rec["accepted"]:
            uv, _ = project(world["kp3d"], np.array(rec["w2c"]), world["K"])
            img = body.draw_skeleton(img, uv, thickness=4)
        tile = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        color = [(80, 200, 80), (0, 200, 255), (60, 60, 230)][rank(rec)]
        cv2.rectangle(tile, (0, 0), (tile_w - 1, tile_h - 1), color, 6)
        r, c = divmod(i, cols)
        y0, x0 = r * (tile_h + 44), c * tile_w
        sheet[y0:y0 + tile_h, x0:x0 + tile_w] = tile
        line1 = f"{rec['tag']} s{rec['seed']}  az {rec['azimuth_deg']:.0f} el {rec['elevation_deg']:.0f}"
        if rec["tag"] == "photo":
            line2 = "input photo"
        elif rec["accepted"]:
            line2 = f"drift {rec['drift_m'] * 1000:.0f}mm rp {rec['reproj_rmse_px']:.1f}px cov {rec['coverage']:.2f}"
        else:
            line2 = "; ".join(rec["reasons"])[:44]
        cv2.putText(sheet, line1, (x0 + 6, y0 + tile_h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(sheet, line2, (x0 + 6, y0 + tile_h + 37), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(out), sheet)

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
FLIP_COS = 0.3              # front/back flip: requested and registered on opposite halves, both > ~17 deg off the side
RESCUE_MAX_DEV_DEG = 25.0   # silhouette rescue: keypoints and silhouette must both land this close to the request
RESCUE_MIN_COVERAGE = 0.72  # silhouette rescue: share of the scaffold silhouette inside the matte after the fit
RESCUE_MIN_COVERAGE_DRIFT = 0.80    # ... for a view the keypoints also read in another pose (pose drift)
RESCUE_MAX_DRIFT_M = 0.15           # beyond this the pose itself differs; no rescue


def _direction(az_deg: float, el_deg: float) -> np.ndarray:
    az, el = math.radians(az_deg), math.radians(el_deg)
    return np.array([math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el)])


def _angle_between(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.degrees(math.acos(float(np.clip(_direction(*a) @ _direction(*b), -1.0, 1.0))))


def flipped(az_deg: float, requested_az_deg: float, photo_az_deg: float = 0.0) -> bool:
    """True when a view registered on the other front/back half than it was asked for (a back view read as
    a front view). Near-symmetric bodies make the keypoints and the silhouette fit fall for this; the
    generator itself does not turn a person round, so the camera is wrong. Left/right mirrors (asked for
    front-left, got front-right) stay on the same half and are allowed."""
    a = math.cos(math.radians(az_deg - photo_az_deg))
    b = math.cos(math.radians(requested_az_deg))
    return a * b < 0 and min(abs(a), abs(b)) > FLIP_COS


def register_view(world: dict, view: dict, alpha: np.ndarray, expected: tuple[float, float] | None,
                  photo_el: float) -> dict:
    K, target = world["K"], world["target"]
    h = world["size"][1]
    photo_az, _, radius = azimuth_elevation(world["photo_w2c"], target)
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
    for hyp in hyps:
        hyp["flipped"] = bool(exp) and flipped(hyp["azimuth_deg"], exp[0], photo_az)
    best = min([x for x in hyps if not x["flipped"]] or hyps, key=lambda hyp: hyp["score"])
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
    if rec.get("flipped"):
        reasons.append("front/back flipped")
    return reasons


def mesh_gaussians(world: dict, n: int = 250_000, size_m: float = 0.004, device: str = "cuda") -> dict:
    """The scaffold mesh as opaque Gaussians on its surface, so the splat renderer can fit cameras to it."""
    import torch
    v, f = world["vertices"], world["faces"]
    rng = np.random.default_rng(0)
    tri = v[f]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    idx = rng.choice(len(f), n, p=area / area.sum())
    a, b = rng.random((2, n, 1))
    fold = (a + b) > 1
    a, b = np.where(fold, 1 - a, a), np.where(fold, 1 - b, b)
    p = tri[idx, 0] + a * (tri[idx, 1] - tri[idx, 0]) + b * (tri[idx, 2] - tri[idx, 0])
    return {"means": torch.tensor(p, dtype=torch.float32, device=device),
            "scales": torch.full((n, 3), math.log(size_m), device=device),
            "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(n, 1),
            "opacities": torch.full((n,), 6.0, device=device),
            "sh0": torch.zeros(n, 1, 3, device=device), "shN": torch.zeros(n, 0, 3, device=device)}


def rescue_by_silhouette(world: dict, records: list[dict]) -> list[str]:
    """Second chance for views the keypoints could not place well enough, mostly true side views.

    The scaffold comes from the front photo, so its depth (posture seen from the side: how far the head
    leans forward, the hips back) is SAM 3D Body's guess. A side view's 2D keypoints measure exactly
    that depth, so a slim, curvy or leaning subject gets every side view rejected (reprojection and
    coverage just over the limits), and nothing then pins the body's depth: the splat grows into a
    diamond twice as deep as the body, whose far faces show from the side as a ghost second body.

    Back-left quarter views have a keypoint problem of their own: SAM 3D Body reads them 10-12 cm off
    the scaffold's pose (back-right views: ~5 cm), so 8 of 12 V15 subjects lost them to "pose drift"
    although their silhouettes fit the scaffold as well as the accepted back-right views do. Those get
    the same second chance with a stricter coverage bar.

    Here the camera is fitted to the matte by the scaffold's silhouette instead (render and compare,
    from the keypoint camera and from the requested one). The view is kept only if the keypoints and
    the silhouette agree it is the side that was asked for; the second pass then fits it to the splat.
    Updates `records` in place; returns the rescued view names."""
    import torch
    from PIL import Image

    from .reregister import fit_camera

    target, K = world["target"], world["K"]
    w, h = world["size"]
    photo_az, photo_el, radius = azimuth_elevation(world["photo_w2c"], target)
    todo = [r for r in records[1:] if not r["accepted"] and r.get("requested_deg")
            and np.isfinite(r.get("azimuth_deg", float("nan")))
            and r.get("drift_m", 1.0) <= RESCUE_MAX_DRIFT_M
            and all(s.startswith(("reprojection", "coverage", "pose drift")) for s in r["reasons"])]
    if not todo:
        return []
    params = mesh_gaussians(world)
    rescued = []
    for rec in todo:
        req = (rec["requested_deg"][0] + photo_az, rec["requested_deg"][1] + photo_el)
        if _angle_between((rec["azimuth_deg"], rec["elevation_deg"]), req) > RESCUE_MAX_DEV_DEG:
            continue
        matte_np = np.asarray(Image.open(rec["mask"]).convert("L"), np.float32) / 255.0
        matte = torch.from_numpy(matte_np)[..., None].cuda()
        black = torch.zeros(h, w, 3, device="cuda")
        fits = [fit_camera(params, w2c0, K, black, matte, color_weight=0.0)
                for w2c0 in (np.asarray(rec["w2c"]), orbit_camera(target, radius, *req))]
        w2c = min(fits, key=lambda fit: fit[1])[0]
        az, el, r = azimuth_elevation(w2c, target)
        stats = body.silhouette_stats(world, w2c, matte_np)
        dev = _angle_between((az, el), req)
        min_cov = RESCUE_MIN_COVERAGE_DRIFT if rec["drift_m"] > MAX_DRIFT_M else RESCUE_MIN_COVERAGE
        if dev > RESCUE_MAX_DEV_DEG or stats["coverage"] < min_cov:
            log.debug("%s_s%s: silhouette fit az %.1f el %.1f cov %.2f: not rescued", rec["tag"], rec["seed"],
                      az, el, stats["coverage"])
            continue
        rec.update({"w2c_keypoints": rec["w2c"], "w2c": w2c.tolist(), "azimuth_deg": az, "elevation_deg": el,
                    "radius_m": r, "prior_dev_deg": dev, "coverage_keypoints": rec["coverage"],
                    "reasons_keypoints": rec["reasons"], "rescued": "silhouette", "accepted": True,
                    "reasons": []} | stats)
        rescued.append(f"{rec['tag']}_s{rec['seed']}")
        log.info("view %s_s%s rescued by its silhouette: az %.1f el %.1f, coverage %.2f (keypoints: %s)",
                 rec["tag"], rec["seed"], az, el, stats["coverage"], "; ".join(rec["reasons_keypoints"]))
    del params
    torch.cuda.empty_cache()
    return rescued


# The eye-level ring every splat needs a usable view on (the photo covers the front). Without the two
# sides nothing pins the body's depth and the splat grows a ghost second body (docs: ghost_body_fix.md);
# without the back or a diagonal, that side is smeared. Tag -> azimuth relative to the photo.
CORE_VIEWS = {"r": 90.0, "l": 270.0, "b": 180.0, "fr": 45.0, "fl": 315.0, "br": 135.0, "bl": 225.0}
CORE_TOLERANCE_DEG = 25.0   # a usable view this close in azimuth (below 40 deg elevation) covers the angle


def core_gaps(records: list[dict]) -> list[str]:
    """Core view tags (CORE_VIEWS) with no accepted view near their angle; the photo is records[0]."""
    photo_az = records[0]["azimuth_deg"]
    usable = [r for r in records[1:] if r.get("accepted") and np.isfinite(r.get("azimuth_deg", float("nan")))
              and r["elevation_deg"] < 40.0]
    return [tag for tag, target in CORE_VIEWS.items()
            if not any(abs((r["azimuth_deg"] - photo_az - target + 180.0) % 360.0 - 180.0) <= CORE_TOLERANCE_DEG
                       for r in usable)]


def select_views(records: list[dict], min_separation_deg: float = MIN_SEPARATION_DEG) -> None:
    """Photo first, then accepted views best-first, skipping ones within MIN_SEPARATION_DEG of a kept view.
    Best = highest silhouette agreement with the first splat once cameras were fitted (reregister.py),
    else the lowest keypoint score."""
    def rank(r: dict) -> float:
        return -r["fit_iou"] if "fit_iou" in r else r["score"]

    kept = [records[0]]
    for rec in sorted((r for r in records[1:] if r["accepted"]), key=rank):
        angle = (rec["azimuth_deg"], rec["elevation_deg"])
        if all(_angle_between(angle, (k["azimuth_deg"], k["elevation_deg"])) >= min_separation_deg for k in kept):
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

        expected = tuple(item["expected_deg"]) if item.get("expected_deg") else expected_angles(item["prompt"])
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
    client.free()                   # the silhouette rescue renders on the GPU
    rescue_by_silhouette(world, records)
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
        elif rec.get("rescued"):
            line2 = f"silhouette fit cov {rec['coverage']:.2f} (kp rp {rec['reproj_rmse_px']:.0f}px)"
        elif rec["accepted"]:
            line2 = f"drift {rec['drift_m'] * 1000:.0f}mm rp {rec['reproj_rmse_px']:.1f}px cov {rec['coverage']:.2f}"
        else:
            line2 = "; ".join(rec["reasons"])[:44]
        cv2.putText(sheet, line1, (x0 + 6, y0 + tile_h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(sheet, line2, (x0 + 6, y0 + tile_h + 37), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(out), sheet)

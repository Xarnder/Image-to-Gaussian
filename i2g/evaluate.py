"""Measure a finished splat so quality changes are compared by numbers, not by eye alone.

Metrics (all from fixed cameras, so two splats of the same run are directly comparable):
  * holdout  - masked PSNR / SSIM / VGG distance / alpha IoU against generated views that were
               registered but never trained on (an independent sample of the same angles)
  * face     - ArcFace similarity between the photo's face and the splat's face rendered from
               15 head close-up cameras (az -30..30, 0..30 deg above the photo's view of the head)
  * needles  - Gaussians much longer than wide (the streaks that bridge gaps between views)
  * floaters - rendered alpha outside the dilated visual hull, over an orbit
  * sharpness- Laplacian energy inside the person over an orbit (softness proxy)
The ArcFace/SCRFD models are the InsightFace buffalo_l ONNX files ComfyUI already has.

    python3 -m i2g.evaluate --run output/exp3/test-image/v2 --ply output/exp3/test-image/v2/final/test-image-and-exp3-character.ply
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .cameras import azimuth_elevation, camera_center, look_at, orbit_camera
from .log import get_logger
from .paths import COMFY_ROOT

log = get_logger("evaluate")
INSIGHTFACE_DIR = COMFY_ROOT / "ComfyUI" / "models" / "insightface" / "models" / "buffalo_l"
ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)
ORBIT_AZ = list(range(0, 360, 30))
ORBIT_EL = (-10.0, 10.0, 30.0)
FACE_AZ = (-30.0, -15.0, 0.0, 15.0, 30.0)
FACE_EL = (0.0, 15.0, 30.0)          # degrees above the photo's view of the head
SHEET_CAMS = [("front", 0, 5), ("face +15up", "face", (0, 15)), ("face 30 side", "face", (30, 5)),
              ("45", 45, 5), ("90", 90, 5), ("110", 110, 5), ("180", 180, 5), ("250", 250, 5),
              ("270", 270, 5), ("top 40", 20, 40)]


# --- splat I/O -------------------------------------------------------------------------------------

def load_ply(path: Path, device: str = "cuda") -> dict:
    """INRIA-layout PLY (what export_ply writes) back into the trainer's parameter layout."""
    with open(path, "rb") as fh:
        header = []
        while True:
            line = fh.readline().decode("ascii").strip()
            header.append(line)
            if line == "end_header":
                break
        body = fh.read()
    n = next(int(h.split()[2]) for h in header if h.startswith("element vertex"))
    props = [h.split()[2] for h in header if h.startswith("property")]
    arr = np.frombuffer(body, dtype="<f4", count=n * len(props)).reshape(n, len(props))
    col = {p: j for j, p in enumerate(props)}

    def take(names):
        return torch.from_numpy(np.ascontiguousarray(arr[:, [col[c] for c in names]])).float().to(device)

    rest = sorted((p for p in props if p.startswith("f_rest_")), key=lambda p: int(p.split("_")[-1]))
    shN = take(rest).reshape(n, 3, -1).transpose(1, 2).contiguous() if rest else torch.zeros(n, 0, 3, device=device)
    return {"means": take(["x", "y", "z"]), "scales": take(["scale_0", "scale_1", "scale_2"]),
            "quats": take(["rot_0", "rot_1", "rot_2", "rot_3"]), "opacities": take(["opacity"])[:, 0],
            "sh0": take(["f_dc_0", "f_dc_1", "f_dc_2"])[:, None], "shN": shN}


@torch.no_grad()
def render(params: dict, w2c: np.ndarray, K: np.ndarray, width: int, height: int, bg: float = 0.5):
    """(rgb uint8 HxWx3 over a grey background, alpha float HxW)."""
    from .splat import render as splat_render
    device = params["means"].device
    sh_degree = int(round(math.sqrt(params["shN"].shape[1] + 1))) - 1
    rgb, alpha, _ = splat_render(params, torch.as_tensor(w2c, device=device).float(),
                                 torch.as_tensor(K, device=device).float(), width, height, sh_degree)
    out = (rgb + (1 - alpha) * bg).clamp(0, 1)
    return (out.cpu().numpy() * 255).round().astype(np.uint8), alpha[..., 0].cpu().numpy()


# --- face identity (InsightFace SCRFD + ArcFace, onnxruntime) -------------------------------------

class FaceID:
    def __init__(self, model_dir: Path = INSIGHTFACE_DIR):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        providers = ["CPUExecutionProvider"]
        self.det = ort.InferenceSession(str(model_dir / "det_10g.onnx"), opts, providers=providers)
        self.rec = ort.InferenceSession(str(model_dir / "w600k_r50.onnx"), opts, providers=providers)
        self.rec_mean, self.rec_std = 127.5, 127.5     # insightface's ArcFaceONNX values for w600k_r50

    def detect(self, rgb: np.ndarray, thresh: float = 0.4) -> tuple[float, np.ndarray] | None:
        """Best face as (score, 5x2 keypoints) in image pixels, or None."""
        size = 640
        h, w = rgb.shape[:2]
        scale = size / max(h, w)
        img = np.zeros((size, size, 3), np.float32)
        img[:round(h * scale), :round(w * scale)] = cv2.resize(rgb, (round(w * scale), round(h * scale)))
        blob = ((img - 127.5) / 128.0).transpose(2, 0, 1)[None].astype(np.float32)
        outs = self.det.run(None, {self.det.get_inputs()[0].name: blob})
        best = None
        for idx, stride in enumerate((8, 16, 32)):
            scores = outs[idx].reshape(-1)
            boxes = outs[idx + 3].reshape(-1, 4) * stride
            kps = outs[idx + 6].reshape(-1, 10) * stride
            g = size // stride
            centers = np.stack(np.mgrid[:g, :g][::-1], -1).reshape(-1, 2).astype(np.float32) * stride
            centers = np.repeat(centers, 2, axis=0)
            j = int(scores.argmax())
            if scores[j] >= thresh and (best is None or scores[j] > best[0]):
                pts = centers[j] + kps[j].reshape(5, 2)
                best = (float(scores[j]), pts / scale)
        return best

    def embed(self, rgb: np.ndarray, kps: np.ndarray) -> np.ndarray:
        M = _similarity_2d(kps.astype(np.float64), ARCFACE_DST.astype(np.float64))
        crop = cv2.warpAffine(rgb, M, (112, 112), borderValue=0)
        blob = ((crop.astype(np.float32) - self.rec_mean) / self.rec_std).transpose(2, 0, 1)[None]
        e = self.rec.run(None, {self.rec.get_inputs()[0].name: blob.astype(np.float32)})[0][0]
        return e / np.linalg.norm(e)


def _similarity_2d(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """2x3 least-squares similarity src -> dst (Umeyama), as insightface's norm_crop."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.diag([1.0, -1.0 if np.linalg.det(U) * np.linalg.det(Vt) < 0 else 1.0])
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (xs ** 2).sum(1).mean()
    return np.hstack([s * R, (mu_d - s * R @ mu_s)[:, None]])


def _head(world: dict) -> np.ndarray:
    from .body import L_EAR, L_EYE, NOSE, R_EAR, R_EYE
    return world["kp3d"][[NOSE, L_EYE, R_EYE, L_EAR, R_EAR]].mean(0)


def face_camera(world: dict, az_offset: float, el_offset: float, dist: float = 0.9) -> np.ndarray:
    """Close-up on the head, rotated from the direction the photo saw the head from."""
    head = _head(world)
    v = camera_center(world["photo_w2c"]) - head
    az0, el0 = math.atan2(v[0], v[2]), math.asin(v[1] / np.linalg.norm(v))
    az, e = az0 + math.radians(az_offset), el0 + math.radians(el_offset)
    return look_at(head + dist * np.array([math.sin(az) * math.cos(e), math.sin(e), math.cos(az) * math.cos(e)]),
                   head)


# --- metrics ---------------------------------------------------------------------------------------

def _psnr(a, b, mask):
    mse = (((a.astype(np.float64) - b.astype(np.float64)) / 255.0) ** 2).mean(-1)[mask].mean()
    return 10 * math.log10(1.0 / max(float(mse), 1e-12))


def _ssim(a, b, mask):
    from pytorch_msssim import ssim
    ta = torch.from_numpy(a).permute(2, 0, 1)[None].float() / 255
    tb = torch.from_numpy(b).permute(2, 0, 1)[None].float() / 255
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return float(ssim(ta[..., y0:y1, x0:x1], tb[..., y0:y1, x0:x1], data_range=1.0))


class _VGG:
    def __init__(self):
        from .splat import VGG_WEIGHTS
        from .splatkit_vendor import PerceptualLoss
        self.loss = PerceptualLoss(VGG_WEIGHTS).cuda().eval()

    @torch.no_grad()
    def __call__(self, a, b):
        ta = torch.from_numpy(a).permute(2, 0, 1)[None].float().cuda() / 255
        tb = torch.from_numpy(b).permute(2, 0, 1)[None].float().cuda() / 255
        return float(self.loss(ta, tb))


def _laplacian_energy(rgb, mask):
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap = cv2.Laplacian(grey, cv2.CV_32F, ksize=3)
    inner = cv2.erode(mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    return float(np.abs(lap)[inner].mean()) if inner.any() else 0.0


def evaluate(run_dir: Path, ply: Path, out_dir: Path, holdout: list[str] | None = None,
             face: FaceID | None = None, vgg: _VGG | None = None) -> dict:
    from .body import load_world
    from .splat import needle_mask
    out_dir.mkdir(parents=True, exist_ok=True)
    world = load_world(run_dir / "01_body")
    from .priors import align_face
    world, _ = align_face(world, json.loads((run_dir / "03_qa" / "registration.json").read_text()), "cuda")
    K = world["K"]
    W, H = world["size"]
    params = load_ply(ply)
    records = json.loads((run_dir / "03_qa" / "registration.json").read_text())
    by_name = {f"{r['tag']}_s{r['seed']}": r for r in records}
    if holdout is None:
        holdout = default_holdout(records)
    result: dict = {"ply": str(ply), "gaussians": len(params["means"]), "holdout_views": holdout}

    # hold-out views. Their keypoint cameras are off by several cm (reregister.py), so each camera is
    # first fitted to this splat: the scores then measure appearance, not the hold-out's camera error.
    from .reregister import fit_camera
    vgg = vgg or _VGG()
    rows = {}
    for name in holdout:
        r = by_name[name]
        gt = np.asarray(Image.open(r["path"]).convert("RGB"))
        mat = np.asarray(Image.open(r["mask"]).convert("L"), np.float32) / 255.0
        gt_t = torch.from_numpy(gt.astype(np.float32) / 255).cuda()
        mat_t = torch.from_numpy(mat)[..., None].cuda()
        w2c, _ = fit_camera(params, np.asarray(r["w2c"]), K, gt_t, mat_t, schedule=((4, 100), (2, 60)))
        pred, alpha = render(params, w2c, K, W, H)
        gt_grey = (gt * mat[..., None] + 127.5 * (1 - mat[..., None])).round().astype(np.uint8)
        m = mat > 0.5
        inter, union = ((alpha > 0.5) & m).sum(), ((alpha > 0.5) | m).sum()
        rows[name] = {"psnr": round(_psnr(pred, gt_grey, m), 2), "ssim": round(_ssim(pred, gt_grey, m), 4),
                      "vgg": round(vgg(pred, gt_grey), 4), "iou": round(float(inter / max(union, 1)), 4)}
    result["holdout"] = rows
    for k in ("psnr", "ssim", "vgg", "iou"):
        result[f"holdout_{k}"] = round(float(np.mean([v[k] for v in rows.values()])), 4) if rows else None

    # photo view (training view; fidelity to the only real image)
    photo = by_name["photo_s0"]
    gt = np.asarray(Image.open(photo["path"]).convert("RGB"))
    mat = np.asarray(Image.open(photo["mask"]).convert("L"), np.float32) / 255.0
    pred, _ = render(params, np.asarray(photo["w2c"]), K, W, H)
    gt_grey = (gt * mat[..., None] + 127.5 * (1 - mat[..., None])).round().astype(np.uint8)
    result["photo_psnr"] = round(_psnr(pred, gt_grey, mat > 0.5), 2)
    hr = np.asarray(Image.open(run_dir / "00_ingest" / "source_hr_rgba.png").convert("RGBA"), np.float32) / 255.0
    K_hr = np.asarray(K, np.float64).copy()
    K_hr[0] *= hr.shape[1] / W
    K_hr[1] *= hr.shape[0] / H
    pred_hr, _ = render(params, np.asarray(photo["w2c"]), K_hr, hr.shape[1], hr.shape[0])
    gt_hr = ((hr[..., :3] * hr[..., 3:] + 0.5 * (1 - hr[..., 3:])) * 255).round().astype(np.uint8)
    result["photo_psnr_hr"] = round(_psnr(pred_hr, gt_hr, hr[..., 3] > 0.5), 2)

    # face identity
    face = face or FaceID()
    src_hr = np.asarray(Image.open(run_dir / "00_ingest" / "source_hr_rgba.png").convert("RGB"))
    ref = face.detect(src_hr)
    faces = {}
    if ref is not None:
        e_ref = face.embed(src_hr, ref[1])
        for az in FACE_AZ:
            for el in FACE_EL:
                img, _ = render(params, face_camera(world, az, el), K, W, H)
                det = face.detect(img)
                faces[f"{az:+.0f}_{el:+.0f}"] = round(float(face.embed(img, det[1]) @ e_ref), 4) if det else 0.0
        result["face"] = faces
        result["face_mean"] = round(float(np.mean(list(faces.values()))), 4)
        result["face_min"] = round(float(np.min(list(faces.values()))), 4)
        result["face_missed"] = sum(v == 0.0 for v in faces.values())
    else:
        log.warning("No face found in the photo; face metrics skipped")

    # face surface: splat depth against the body mesh's face (photo camera and two turned cameras)
    from .priors import MeshPrior
    from .splat import render as splat_render
    prior = MeshPrior(world, "cuda")
    K_t = torch.as_tensor(K, device="cuda").float()
    sh_degree = int(round(math.sqrt(params["shN"].shape[1] + 1))) - 1
    errs = []
    for w2c in (world["photo_w2c"], face_camera(world, -30, 0), face_camera(world, 30, 0)):
        w2c_t = torch.as_tensor(w2c, device="cuda").float()
        target = prior.face_targets(w2c_t, K_t, W, H)
        if target is None:
            continue
        with torch.no_grad():
            _, a, info = splat_render(params, w2c_t, K_t, W, H, sh_degree, with_depth=True)
        m = (target[1][..., 0] > 0.5) & (a[..., 0] > 0.5)
        errs.append(float((info["depth"][..., 0] - target[0][..., 0]).abs()[m].median()) * 1000)
    result["face_depth_mm"] = round(float(np.mean(errs)), 1) if errs else None
    del prior

    # needles (streaks)
    s = torch.exp(params["scales"])
    result["needles_8x_3cm"] = int(needle_mask(params, 8.0, 0.03).sum())
    result["needles_6x_2cm"] = int(needle_mask(params, 6.0, 0.02).sum())
    result["max_scale_p999_cm"] = round(float(torch.quantile(s.max(1).values, 0.999)) * 100, 2)

    # floaters + sharpness over an orbit
    hull = _hull_points(run_dir, world)
    _, _, radius = azimuth_elevation(world["photo_w2c"], world["target"])
    float_frac, sharp = [], []
    for el in ORBIT_EL:
        for az in ORBIT_AZ:
            w2c = orbit_camera(world["target"], radius, az, el)
            img, alpha = render(params, w2c, K, W, H)
            support = _splat_mask(hull, w2c, K, W, H, dilate=10)
            total = alpha.sum()
            float_frac.append(float(alpha[~support].sum() / max(total, 1e-6)))
            sharp.append(_laplacian_energy(img, alpha > 0.9))
    result["floater_pct"] = round(100 * float(np.mean(float_frac)), 3)
    result["sharpness"] = round(float(np.mean(sharp)), 3)

    (out_dir / "eval.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    sheet_row(params, world, out_dir / "sheet_row.png")
    return result


def default_holdout(records: list[dict]) -> list[str]:
    """Accepted-but-unselected generated views, skipping near-copies of the photo."""
    names = []
    for r in records:
        if r["tag"] == "photo" or r.get("selected") or not r.get("accepted"):
            continue
        az = abs((r["azimuth_deg"] + 180.0) % 360.0 - 180.0)
        if az < 15.0 and abs(r["elevation_deg"]) < 10.0:
            continue          # the model returned (almost) the input photo
        names.append(f"{r['tag']}_s{r['seed']}")
    return names


_HULL_CACHE: dict = {}


def _hull_points(run_dir: Path, world: dict) -> np.ndarray:
    """Visual hull (all accepted views) as points: the region where the person can be."""
    key = str(run_dir)
    if key not in _HULL_CACHE:
        from .splat import SplatConfig, load_views, visual_hull
        records = json.loads((run_dir / "03_qa" / "registration.json").read_text())
        accepted = [dict(r, selected=bool(r.get("accepted") or r["tag"] == "photo")) for r in records]
        views = load_views(accepted, world["K"], SplatConfig(), "cuda")
        points, _ = visual_hull(views, world, SplatConfig(hull_tolerance=2))[:2]
        _HULL_CACHE[key] = points.cpu().numpy()
    return _HULL_CACHE[key]


def _splat_mask(points: np.ndarray, w2c: np.ndarray, K: np.ndarray, w: int, h: int, dilate: int) -> np.ndarray:
    from .cameras import project
    uv, z = project(points, w2c, K)
    u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    ok = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    mask = np.zeros((h, w), np.uint8)
    mask[v[ok], u[ok]] = 1
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1)))
    return mask > 0


def _sheet_cam(world: dict, spec, radius: float):
    if spec[1] == "face":
        return face_camera(world, spec[2][0], spec[2][1])
    return orbit_camera(world["target"], radius, spec[1], spec[2])


def sheet_row(params: dict, world: dict, path: Path, height: int = 480) -> None:
    K, (W, H) = world["K"], world["size"]
    _, _, radius = azimuth_elevation(world["photo_w2c"], world["target"])
    tiles = []
    for spec in SHEET_CAMS:
        img, _ = render(params, _sheet_cam(world, spec, radius), K, W, H)
        img = cv2.resize(img, (round(W * height / H), height), interpolation=cv2.INTER_AREA)
        cv2.putText(img, spec[0], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(img)
    Image.fromarray(np.concatenate(tiles, 1)).save(path)


def compare_sheet(rows: list[tuple[str, Path]], path: Path) -> None:
    """Stack several sheet_row.png files with a label strip on the left."""
    imgs = []
    for label, p in rows:
        img = np.asarray(Image.open(p).convert("RGB")).copy()
        strip = np.full((img.shape[0], 220, 3), 40, np.uint8)
        for k, part in enumerate(label.split("\n")):
            cv2.putText(strip, part, (8, 30 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        imgs.append(np.concatenate([strip, img], 1))
    Image.fromarray(np.concatenate(imgs, 0)).save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="run folder with 00_ingest ... 03_qa")
    ap.add_argument("--ply", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, help="default: next to each PLY, in eval/")
    args = ap.parse_args()
    face, vgg = FaceID(), _VGG()
    for ply in args.ply:
        out = args.out or ply.parent / "eval"
        res = evaluate(args.run, ply, out, face=face, vgg=vgg)
        print(json.dumps({k: v for k, v in res.items() if k not in ("holdout", "face")}, indent=1))


if __name__ == "__main__":
    main()

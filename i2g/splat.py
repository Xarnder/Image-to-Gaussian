"""S5/S6: fit one static 3D Gaussian splat to the registered views, clean it and export it.

Model and densification follow nerfstudio's splatfacto through gsplat (as SplatKit's trainer does).
Additions for *generated* views, which are never perfectly consistent:
  * matte loss on the rendered alpha (the mattes are reliable even when colours drift)
  * per-view camera refinement and per-view colour correction (the photo is frozen: it is the reference)
  * per-view loss weights (photo highest) and weighted view sampling
Initialisation is a visual hull carved from the registered mattes (4DAnyone's FreeTimeGS init).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .log import NullBar, get_logger
from .paths import MODELS_DIR

log = get_logger("splat")
SH_C0 = 0.28209479177387814
VGG_WEIGHTS = MODELS_DIR / "perceptual" / "imagenet-vgg-verydeep-19-conv.safetensors"


@dataclass
class SplatConfig:
    iterations: int = 15000
    sh_degree: int = 1                 # generated views carry no real view dependence; higher bands overfit
    sh_degree_interval: int = 1000
    ssim_lambda: float = 0.2
    perceptual_weight: float = 0.2
    alpha_weight: float = 0.5
    photo_weight: float = 1.0
    view_weight: float = 0.6
    photo_sample_share: float = 0.25   # fraction of iterations spent on the photo view
    lr_means: float = 1.6e-4
    lr_means_final: float = 1.6e-6
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacities: float = 5e-2
    lr_sh0: float = 2.5e-3
    lr_shN: float = 2.5e-3 / 20
    pose_lr: float = 1e-4
    pose_start: int = 1000
    pose_reg: float = 10.0
    color_lr: float = 1e-3
    color_reg: float = 1.0
    refine_start: int = 500
    refine_stop: int = 12000
    refine_every: int = 100
    reset_every: int = 3000
    grow_grad2d: float = 0.0002
    prune_opa: float = 0.05
    half_res_until: int = 2000
    perceptual_max_px: int = 1024      # larger views get the perceptual loss on a random crop this size
    voxel_m: float = 0.012
    hull_tolerance: int = 1
    hull_dilate_px: int = 6
    clean_min_opacity: float = 0.05
    clean_matte_support: float = 0.6
    seed: int = 0


# --- data -----------------------------------------------------------------------------------------

def load_views(registration: Path, K: np.ndarray, cfg: SplatConfig, device,
               hr_photo: Path | None = None) -> dict:
    """Selected views as per-view lists (sizes may differ: the photo can be the native-resolution crop)."""
    records = [r for r in json.loads(registration.read_text()) if r.get("selected")]
    if records[0]["tag"] != "photo":
        raise ValueError("registration.json must list the photo first")
    images, alphas, Ks, w2cs, weights, names = [], [], [], [], [], []
    for r in records:
        if r["tag"] == "photo" and hr_photo is not None and hr_photo.exists():
            rgba = np.asarray(Image.open(hr_photo).convert("RGBA"), np.float32) / 255.0
            rgb, alpha = rgba[..., :3], rgba[..., 3]
            canvas_h, canvas_w = np.asarray(Image.open(r["path"])).shape[:2]
            k = K.copy()
            k[0] *= rgb.shape[1] / canvas_w
            k[1] *= rgb.shape[0] / canvas_h
        else:
            rgb = np.asarray(Image.open(r["path"]).convert("RGB"), np.float32) / 255.0
            alpha = np.asarray(Image.open(r["mask"]).convert("L"), np.float32) / 255.0
            k = K
        images.append(torch.from_numpy(rgb).to(device))
        alphas.append(torch.from_numpy(alpha)[..., None].to(device))
        Ks.append(k)
        w2cs.append(np.asarray(r["w2c"], np.float64))
        weights.append(cfg.photo_weight if r["tag"] == "photo" else cfg.view_weight)
        names.append(f"{r['tag']}_s{r['seed']}")
    return {
        "names": names,
        "images": images,                                                    # V x (H,W,3)
        "alphas": alphas,                                                    # V x (H,W,1)
        "K": torch.from_numpy(np.stack(Ks)).float().to(device),              # V,3,3
        "w2c": torch.from_numpy(np.stack(w2cs)).float().to(device),          # V,4,4
        "weights": torch.tensor(weights, device=device),
    }


def _project(points: torch.Tensor, w2c: torch.Tensor, K: torch.Tensor):
    cam = points @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam[:, 2]
    uv = cam[:, :2] / z.clamp_min(1e-6)[:, None] * torch.stack([K[0, 0], K[1, 1]]) + torch.stack([K[0, 2], K[1, 2]])
    return uv, z


def _sample_mask(mask: torch.Tensor, uv: torch.Tensor, z: torch.Tensor):
    """Returns (inside image & in front, mask value at uv)."""
    h, w = mask.shape
    u, v = uv[:, 0].round().long(), uv[:, 1].round().long()
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (z > 0)
    val = torch.zeros_like(z, dtype=torch.bool)
    val[inside] = mask[v[inside], u[inside]]
    return inside, val


def visual_hull(views: dict, world: dict, cfg: SplatConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Surface voxels of the space carved from the mattes, coloured from the best-facing view."""
    device = views["w2c"].device
    verts = world["vertices"]
    lo = verts.min(0) - np.array([0.25, 0.03, 0.25])
    hi = verts.max(0) + np.array([0.25, 0.15, 0.25])
    axes = [torch.arange(lo[i], hi[i], cfg.voxel_m, device=device) for i in range(3)]
    shape = tuple(len(a) for a in axes)
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3).float()

    votes = torch.zeros(len(grid), device=device)
    seen = torch.zeros(len(grid), device=device)
    k = 2 * cfg.hull_dilate_px + 1
    for i in range(len(views["names"])):
        alpha = views["alphas"][i][..., 0]
        pad = max(1, round(cfg.hull_dilate_px * alpha.shape[0] / 1248))  # dilation in canvas pixels
        mask = F.max_pool2d((alpha > 0.5).float()[None, None], 2 * pad + 1, 1, pad)[0, 0] > 0
        uv, z = _project(grid, views["w2c"][i], views["K"][i])
        inside, val = _sample_mask(mask, uv, z)
        seen += inside.float()
        votes += val.float()
    occ = ((votes >= seen - cfg.hull_tolerance) & (seen >= 2)).reshape(shape).float()
    eroded = -F.max_pool3d(-occ[None, None], 3, 1, 1)[0, 0]
    surface = (occ > 0) & (eroded < 1)
    points = grid[surface.reshape(-1)]

    # colour: the view whose camera best faces each point's horizontal outward direction
    centers = torch.stack([-w[:3, :3].T @ w[:3, 3] for w in views["w2c"]])     # V,3
    target = torch.tensor(world["target"], device=device).float()
    out = F.normalize((points - target) * torch.tensor([1.0, 0.0, 1.0], device=device), dim=1)
    cam_dir = F.normalize((centers - target) * torch.tensor([1.0, 0.0, 1.0], device=device), dim=1)
    best = (out @ cam_dir.T).argmax(1)
    colors = torch.full_like(points, 0.5)
    for i in range(len(views["names"])):
        sel = best == i
        uv, z = _project(points[sel], views["w2c"][i], views["K"][i])
        h, w = views["alphas"][i].shape[:2]
        u, v = uv[:, 0].round().long().clamp(0, w - 1), uv[:, 1].round().long().clamp(0, h - 1)
        colors[sel] = views["images"][i][v, u]
    return points, colors


# --- model ----------------------------------------------------------------------------------------

def so3_exp(w: torch.Tensor) -> torch.Tensor:
    theta = w.norm().clamp_min(1e-12)
    k = w / theta
    Kx = torch.zeros(3, 3, device=w.device)
    Kx[0, 1], Kx[0, 2], Kx[1, 0], Kx[1, 2], Kx[2, 0], Kx[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]
    return torch.eye(3, device=w.device) + torch.sin(theta) * Kx + (1 - torch.cos(theta)) * Kx @ Kx


def apply_delta(w2c: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    T = torch.eye(4, device=w2c.device)
    T[:3, :3] = so3_exp(delta[:3])
    T[:3, 3] = delta[3:]
    return T @ w2c


def init_params(points: torch.Tensor, colors: torch.Tensor, cfg: SplatConfig) -> torch.nn.ParameterDict:
    n = len(points)
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
    quats = F.normalize(torch.randn(n, 4, generator=gen), dim=1).to(points.device)
    sh = torch.zeros(n, (cfg.sh_degree + 1) ** 2, 3, device=points.device)
    sh[:, 0] = (colors - 0.5) / SH_C0
    return torch.nn.ParameterDict({
        "means": torch.nn.Parameter(points.clone()),
        "scales": torch.nn.Parameter(torch.full((n, 3), math.log(cfg.voxel_m * 0.7), device=points.device)),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), 0.1, device=points.device))),
        "sh0": torch.nn.Parameter(sh[:, :1].contiguous()),
        "shN": torch.nn.Parameter(sh[:, 1:].contiguous()),
    })


def render(params, viewmat, K, width, height, sh_degree, absgrad=False):
    from gsplat import rasterization
    colors = torch.cat([params["sh0"], params["shN"]], 1)
    rgb, alpha, info = rasterization(
        means=params["means"], quats=params["quats"], scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]), colors=colors, viewmats=viewmat[None], Ks=K[None],
        width=width, height=height, sh_degree=sh_degree, packed=False, absgrad=absgrad,
        near_plane=0.01, far_plane=100.0, rasterize_mode="classic")
    return rgb[0], alpha[0], info   # premultiplied colour over black, alpha


# --- training -------------------------------------------------------------------------------------

def _perceptual_crop(pred: torch.Tensor, target: torch.Tensor, mat: torch.Tensor, size: int,
                     gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Full image if it is small, else a size x size crop centred on a random person pixel."""
    _, _, h, w = pred.shape
    if h * w <= size * size * 1.3:
        return pred, target
    ys, xs = torch.nonzero(mat[..., 0] > 0.5, as_tuple=True)
    j = int(torch.randint(len(ys), (1,), generator=gen, device=ys.device)) if len(ys) else 0
    cy, cx = (int(ys[j]), int(xs[j])) if len(ys) else (h // 2, w // 2)
    ch, cw = min(size, h), min(size, w)
    y0 = min(max(cy - ch // 2, 0), h - ch)
    x0 = min(max(cx - cw // 2, 0), w - cw)
    return pred[..., y0:y0 + ch, x0:x0 + cw], target[..., y0:y0 + ch, x0:x0 + cw]

def train(views: dict, world: dict, cfg: SplatConfig, bar=None) -> tuple[torch.nn.ParameterDict, dict]:
    from gsplat.strategy import DefaultStrategy
    from pytorch_msssim import SSIM

    from .splatkit_vendor import PerceptualLoss

    bar = bar or NullBar()
    torch.manual_seed(cfg.seed)
    device = views["w2c"].device
    n_views = len(views["names"])

    points, colors = visual_hull(views, world, cfg)
    log.info("Visual hull: %d surface points to start from", len(points))
    if len(points) < 1000:
        raise RuntimeError(f"visual hull has only {len(points)} points: the view mattes or cameras disagree")
    params = init_params(points, colors, cfg)

    lrs = {"means": cfg.lr_means, "scales": cfg.lr_scales, "quats": cfg.lr_quats,
           "opacities": cfg.lr_opacities, "sh0": cfg.lr_sh0, "shN": cfg.lr_shN}
    optimizers = {k: torch.optim.Adam([params[k]], lr=lr, eps=1e-15) for k, lr in lrs.items()}
    means_decay = (cfg.lr_means_final / cfg.lr_means) ** (1.0 / cfg.iterations)

    strategy = DefaultStrategy(prune_opa=cfg.prune_opa, grow_grad2d=cfg.grow_grad2d, grow_scale3d=0.01,
                               grow_scale2d=0.05, prune_scale3d=0.1, prune_scale2d=0.15,
                               refine_scale2d_stop_iter=4000, refine_start_iter=cfg.refine_start,
                               refine_stop_iter=cfg.refine_stop, reset_every=cfg.reset_every,
                               refine_every=cfg.refine_every, pause_refine_after_reset=n_views + cfg.refine_every,
                               absgrad=True, revised_opacity=False, verbose=False)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=1.0)

    pose_delta = torch.zeros(n_views, 6, device=device, requires_grad=True)
    color_A = torch.eye(3, device=device).repeat(n_views, 1, 1).requires_grad_(True)
    color_b = torch.zeros(n_views, 3, device=device, requires_grad=True)
    extra_opt = torch.optim.Adam([{"params": [pose_delta], "lr": cfg.pose_lr},
                                  {"params": [color_A, color_b], "lr": cfg.color_lr}])
    frozen = torch.zeros(n_views, 1, device=device)
    frozen[0] = 1.0  # the photo

    ssim = SSIM(data_range=1.0, size_average=True, channel=3).to(device)
    perceptual = PerceptualLoss(VGG_WEIGHTS).to(device) if cfg.perceptual_weight > 0 else None

    probs = torch.full((n_views,), (1 - cfg.photo_sample_share) / max(n_views - 1, 1), device=device)
    probs[0] = cfg.photo_sample_share if n_views > 1 else 1.0
    gen = torch.Generator(device=device).manual_seed(cfg.seed)

    start = time.perf_counter()
    reported = 0
    for step in range(cfg.iterations):
        i = int(torch.multinomial(probs, 1, generator=gen))
        d = 2 if step < cfg.half_res_until else 1
        img, mat = views["images"][i], views["alphas"][i]
        h, w = img.shape[0] // d, img.shape[1] // d
        K = views["K"][i].clone()
        K[:2] /= d
        if d > 1:
            img = F.interpolate(img.permute(2, 0, 1)[None], (h, w), mode="area")[0].permute(1, 2, 0)
            mat = F.interpolate(mat.permute(2, 0, 1)[None], (h, w), mode="area")[0].permute(1, 2, 0)

        delta = pose_delta[i] * (1 - frozen[i]) if step >= cfg.pose_start else pose_delta[i].detach() * 0
        viewmat = apply_delta(views["w2c"][i], delta)
        sh_degree = min(step // cfg.sh_degree_interval, cfg.sh_degree)
        rgb, alpha, info = render(params, viewmat, K, w, h, sh_degree, absgrad=True)
        strategy.step_pre_backward(params, optimizers, state, step, info)

        if i != 0:
            rgb = rgb @ color_A[i].T + color_b[i] * alpha
        bg = torch.rand(3, device=device)
        pred = rgb + (1 - alpha) * bg
        target = img * mat + (1 - mat) * bg

        l1 = (pred - target).abs().mean()
        pred_n, target_n = pred.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None]
        loss = (1 - cfg.ssim_lambda) * l1 + cfg.ssim_lambda * (1 - ssim(pred_n, target_n))
        loss = loss + cfg.alpha_weight * (alpha - mat).abs().mean()
        if perceptual is not None:
            p_pred, p_target = _perceptual_crop(pred_n, target_n, mat, cfg.perceptual_max_px, gen)
            loss = loss + cfg.perceptual_weight * perceptual(p_pred, p_target)
        loss = views["weights"][i] * loss
        loss = loss + cfg.pose_reg * (pose_delta[i] ** 2).sum()
        loss = loss + cfg.color_reg * ((color_A[i] - torch.eye(3, device=device)) ** 2).sum() + \
            cfg.color_reg * (color_b[i] ** 2).sum()
        loss.backward()

        # same order as gsplat's simple_trainer: optimise, then densify/prune
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        extra_opt.step()
        extra_opt.zero_grad(set_to_none=True)
        strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
        for group in optimizers["means"].param_groups:
            group["lr"] *= means_decay

        if step % 50 == 0 or step == cfg.iterations - 1:
            bar.update(step + 1 - reported)
            reported = step + 1
            bar.set_postfix_str(f"loss {loss.item():.4f}  gaussians {len(params['means']):,}")
        if step % 1000 == 0 or step == cfg.iterations - 1:
            log.debug("step %5d loss %.4f l1 %.4f gaussians %d %.1fs", step, loss.item(), l1.item(),
                      len(params["means"]), time.perf_counter() - start)

    with torch.no_grad():
        deltas = pose_delta.detach().clone()
        deltas[0] = 0
        refined = torch.stack([apply_delta(views["w2c"][i], deltas[i]) for i in range(n_views)])
    stats = {"train_seconds": round(time.perf_counter() - start, 1), "gaussians": len(params["means"]),
             "pose_refinement": {views["names"][i]: {
                 "rot_deg": round(float(deltas[i, :3].norm()) * 180 / math.pi, 3),
                 "trans_cm": round(float(deltas[i, 3:].norm()) * 100, 2)} for i in range(n_views)},
             "refined_w2c": refined.cpu().numpy().tolist()}
    return params, stats


@torch.no_grad()
def clean(params, views: dict, refined_w2c: torch.Tensor, cfg: SplatConfig) -> dict:
    """Drop near-transparent gaussians and floaters outside the mattes of most views that see them."""
    keep = torch.sigmoid(params["opacities"]) >= cfg.clean_min_opacity
    means = params["means"]
    support = torch.zeros(len(means), device=means.device)
    seen = torch.zeros(len(means), device=means.device)
    for i in range(len(views["names"])):
        alpha = views["alphas"][i][..., 0]
        pad = max(1, round(4 * alpha.shape[0] / 1248))
        mask = F.max_pool2d((alpha > 0.5).float()[None, None], 2 * pad + 1, 1, pad)[0, 0] > 0
        uv, z = _project(means, refined_w2c[i], views["K"][i])
        inside, val = _sample_mask(mask, uv, z)
        seen += inside.float()
        support += val.float()
    keep &= support >= cfg.clean_matte_support * seen.clamp_min(1)
    removed = int((~keep).sum())
    for k in list(params.keys()):
        params[k] = torch.nn.Parameter(params[k][keep])
    return {"removed": removed, "remaining": int(keep.sum())}


@torch.no_grad()
def export_ply(params, path: Path) -> None:
    from .splatkit_vendor import write_ply
    n = len(params["means"])
    f = lambda t: t.detach().float().cpu().numpy()  # noqa: E731
    # pad to degree 3 (45 f_rest values): the layout every 3DGS viewer accepts; zero bands change nothing
    shN = torch.zeros(n, 15, 3, device=params["shN"].device)
    shN[:, :params["shN"].shape[1]] = params["shN"]
    write_ply(path, {
        "x": f(params["means"][:, 0]), "y": f(params["means"][:, 1]), "z": f(params["means"][:, 2]),
        "nx": np.zeros(n, np.float32), "ny": np.zeros(n, np.float32), "nz": np.zeros(n, np.float32),
        "f_dc": f(params["sh0"][:, 0]),
        "f_rest": f(shN.transpose(1, 2).reshape(n, -1)),
        "opacity": f(params["opacities"]),
        "scale": f(params["scales"]),
        "rot": f(F.normalize(params["quats"], dim=1)),
    }, comments=("image-to-gaussian v2", "Y-up, metres, feet at y=0, person faces +Z"))


@torch.no_grad()
def render_rgb(params, w2c: torch.Tensor, K: torch.Tensor, width: int, height: int, bg: float = 0.5) -> np.ndarray:
    sh_degree = int(round(math.sqrt(params["shN"].shape[1] + 1))) - 1
    rgb, alpha, _ = render(params, w2c, K, width, height, sh_degree=sh_degree)
    out = (rgb + (1 - alpha) * bg).clamp(0, 1)
    return (out.cpu().numpy() * 255).round().astype(np.uint8)


def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    diff = (a.astype(np.float64) - b.astype(np.float64)) / 255.0
    mse = (diff ** 2).mean(-1)
    mse = float(mse[mask].mean()) if mask is not None else float(mse.mean())
    return 10 * math.log10(1.0 / max(mse, 1e-12))


def config_dict(cfg: SplatConfig) -> dict:
    return asdict(cfg)

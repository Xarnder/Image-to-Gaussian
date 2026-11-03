"""S5/S6: fit one static 3D Gaussian splat to the registered views, clean it and export it.

Model and densification follow nerfstudio's splatfacto through gsplat (as SplatKit's trainer does).
Additions for *generated* views, which are never perfectly consistent:
  * matte loss on the rendered alpha (the mattes are reliable even when colours drift)
  * per-view camera refinement and per-view colour correction (the photo is frozen: it is the reference)
  * per-view loss weights (photo highest) and weighted view sampling
Quality options measured in quality_plan.md (i2g/evaluate.py, i2g/experiment.py):
  * sh_degree 0: view-dependent colour extrapolates to unseen angles as teal/green streaks
  * multiscale: random zoomed-out steps, so a splat fitted to the high-res photo does not alias smaller
  * aniso_weight: needle-shaped Gaussians hide in the gaps between views and streak from the side
  * face_*: a face prior from the body mesh (priors.py) -- depth targets, a thin shell, starting points,
    and the photo owns the face's colours
  * interior_margin: Gaussians inside the body mesh are cleared (they tint surfaces and show through)
  * restored views (restore.py) with their own weight and sampling share
  * video frames (kind "video", pacgs): generated along a known camera orbit, each with its own K
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
    sh_degree: int = 0                 # flat colour: with any SH band, unseen angles extrapolate to teal streaks
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
    pose_lr: float = 5e-4              # camera refinement of generated views (keypoint cameras are cm off)
    pose_start: int = 1500
    pose_reg: float = 0.1
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
    # --- measured in quality_plan.md ("Results of the 2026-09-19 quality sprint") ---
    aniso_weight: float = 0.1          # penalty on needle-shaped Gaussians (longest / 2nd axis > aniso_ratio)
    aniso_ratio: float = 6.0
    prune_scale3d: float = 0.1         # metres; densification prunes Gaussians larger than this
    needle_ratio: float = 10.0         # post-train cleanup: drop longest/2nd axis > ratio ... (0 = off)
    needle_min_m: float = 0.03         # ... when the longest axis is also longer than this
    face_weight: float = 0.0           # colour-loss weight on the face core in generated views that see it
    face_core_m: float = 0.06          # radius of the face "core" the photo owns (face_weight applies there)
    face_bake: bool = True             # after training, replace the core with the photo on the mesh's face
    face_bake_m: float = 0.085         # radius of the baked patch (hair inside it stays 3D)
    face_align: bool = True            # move the mesh's head to where the views put it first (pipeline._aligned_world)
    face_align_method: str = "keypoints"   # keypoints: the views' head keypoints; silhouette: the old face_push
    face_depth_weight: float = 1.0     # keep face Gaussians on the body mesh's face surface (per metre)
    face_mesh_init: bool = True        # start the face from the mesh surface instead of the visual hull
    face_shell: bool = True            # keep Gaussians near the face within face_shell_m of the mesh's face
    face_shell_m: float = 0.002
    interior_margin: float = 0.0       # clear Gaussians this deep inside the body mesh (0 = off; 0.03 hurt)
    interior_face_margin: float = 0.005  # ... and deeper than this behind the face
    multiscale: float = 0.5            # share of steps rendered at a random lower resolution (anti-aliasing)
    ms_min_height: int = 560           # lowest image height used by the multiscale steps
    multiscale_video: float = 0.1      # ... for video frames: they are already small, and the zoom-outs
                                       # blur the detail the head close-ups exist to provide
    pose_stop: int = 10000             # camera refinement runs in [pose_start, pose_stop), then freezes
    strategy: str = "default"          # "default" (clone/split) or "mcmc" (fixed Gaussian budget)
    mcmc_cap: int = 300_000
    opacity_reg: float = 0.0
    scale_reg: float = 0.0
    init_opacity: float = 0.1
    exclude: str = ""                  # comma-separated view names kept out of training (hold-out views)
    hull_carve: bool = False           # keep Gaussians inside the visual hull: kills spikes between views
    restored_weight: float = 0.3       # loss weight of restored orbit frames (restore.py)
    restored_share: float = 0.35       # share of iterations spent on restored frames, when there are any
    restored_alpha: float = 0.0        # alpha-loss factor for restored frames (their matte is the splat's own)
    hull_carve_dilate: int = 2         # hull dilation in voxels before carving (tolerance for camera error)
    # --- calibrated video frames (pacgs: proxy-controlled video with known cameras) ---
    video_weight: float = 0.6          # loss weight of video frames
    video_share: float = 1.0           # share of the non-photo, non-restored steps given to video frames
                                       # when generated views are also present (alone they get all of them)
    video_pose_refine: bool = False    # refine video-frame cameras like generated views (else exact)
    # CAP4D-style perceptual schedule (headpass): the perceptual weight ramps from perceptual_weight to
    # perceptual_ramp_to between these fractions of training (0 = off, the default)
    perceptual_ramp_to: float = 0.0
    perceptual_ramp_start: float = 0.25
    perceptual_ramp_end: float = 0.75


# --- data -----------------------------------------------------------------------------------------

def load_views(registration: Path | list, K: np.ndarray, cfg: SplatConfig, device,
               hr_photo: Path | None = None, world: dict | None = None) -> dict:
    """Selected views as per-view lists (sizes may differ: the photo can be the native-resolution crop).
    `registration` is registration.json or its records."""
    excluded = {s.strip() for s in cfg.exclude.split(",") if s.strip()}
    records = registration if isinstance(registration, list) else json.loads(registration.read_text())
    records = [r for r in records if r.get("selected") and f"{r['tag']}_s{r['seed']}" not in excluded]
    if records[0]["tag"] != "photo":
        raise ValueError("registration.json must list the photo first")
    prior = None
    if world is not None and (cfg.face_weight < 1.0 or cfg.face_depth_weight > 0 or cfg.face_mesh_init):
        from .priors import MeshPrior
        prior = MeshPrior(world, device, core_radius_m=cfg.face_core_m)
    images, alphas, Ks, w2cs, weights, names, color_maps = [], [], [], [], [], [], []
    face_depths, face_masks, kinds, shares = [], [], [], []
    for r in records:
        if r["tag"] == "photo" and hr_photo is not None and hr_photo.exists():
            rgba = np.asarray(Image.open(hr_photo).convert("RGBA"), np.float32) / 255.0
            rgb, alpha = rgba[..., :3], rgba[..., 3]
            canvas_h, canvas_w = np.asarray(Image.open(r["path"])).shape[:2]
            k = K.copy()
            k[0] *= rgb.shape[1] / canvas_w
            k[1] *= rgb.shape[0] / canvas_h
        else:
            img, matte = Image.open(r["path"]).convert("RGB"), Image.open(r["mask"]).convert("L")
            if matte.size != img.size:          # restored frames are larger than the canvas their matte has
                matte = matte.resize(img.size, Image.Resampling.BILINEAR)
            rgb = np.asarray(img, np.float32) / 255.0
            alpha = np.asarray(matte, np.float32) / 255.0
            if "K" in r:                        # the record's own intrinsics, for its image size
                k = np.asarray(r["K"], np.float64).copy()
            else:                               # K is for the canvas (principal point at its centre)
                k = K.copy()
                k[0] *= rgb.shape[1] / (2 * K[0, 2])
                k[1] *= rgb.shape[0] / (2 * K[1, 2])
        images.append(torch.from_numpy(rgb).to(device))
        alphas.append(torch.from_numpy(alpha)[..., None].to(device))
        Ks.append(k)
        w2cs.append(np.asarray(r["w2c"], np.float64))
        kind = "photo" if r["tag"] == "photo" else r.get("kind", "generated")
        kinds.append(kind)
        # a record may carry its own loss weight and sampling share, so one clip (the head close-ups)
        # can count for more than another (the body orbit) without changing the others
        weights.append(float(r.get("weight", {"photo": cfg.photo_weight, "generated": cfg.view_weight,
                                              "restored": cfg.restored_weight, "video": cfg.video_weight}[kind])))
        shares.append(float(r.get("share", 1.0)))
        names.append(f"{r['tag']}_s{r['seed']}")
        cmap = fdepth = fmask = None
        az = abs((r.get("azimuth_deg", 180.0) + 180.0) % 360.0 - 180.0)
        if prior is not None and az <= 110.0:
            face = prior.face_targets(torch.from_numpy(w2cs[-1]).float().to(device),
                                      torch.from_numpy(k).float().to(device), rgb.shape[1], rgb.shape[0])
            if face is not None:
                fdepth, fmask, region = face
                if r["tag"] != "photo" and cfg.face_weight < 1.0:
                    # the photo owns the face: generated faces disagree with it and with each other
                    fw = cfg.face_weight if az <= 60.0 else (1.0 + cfg.face_weight) / 2.0
                    cmap = 1.0 - (1.0 - fw) * region
        color_maps.append(cmap)
        face_depths.append(fdepth)
        face_masks.append(fmask)
    return {
        "names": names,
        "images": images,                                                    # V x (H,W,3)
        "alphas": alphas,                                                    # V x (H,W,1)
        "color_maps": color_maps,                                            # V x (H,W,1) or None
        "face_depth": face_depths,                                           # V x (H,W,1) or None
        "face_mask": face_masks,                                             # V x (H,W,1) or None
        "prior": prior,
        "K": torch.from_numpy(np.stack(Ks)).float().to(device),              # V,3,3
        "w2c": torch.from_numpy(np.stack(w2cs)).float().to(device),          # V,4,4
        "weights": torch.tensor(weights, device=device),
        "kinds": kinds,                                                      # photo / generated / restored / video
        "shares": shares,                                                    # per-view sampling multiplier
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


def visual_hull(views: dict, world: dict, cfg: SplatConfig, return_grid: bool = False):
    """Surface voxels of the space carved from the mattes, coloured from the best-facing view.
    With return_grid, also the occupancy grid {"occ", "lo", "voxel"}."""
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
    if return_grid:
        grid_info = {"occ": occ > 0, "lo": torch.tensor(lo, device=device).float(), "voxel": cfg.voxel_m}
        return points, colors, grid_info
    return points, colors


class HullCarver:
    """Space carving for Gaussians. Every true surface point projects inside every view's matte, so a
    Gaussian that reaches outside the (dilated) visual hull is a streak or spike filling a gap between
    views: its centre outside -> it is killed; an axis reaching outside -> that axis is shortened."""

    def __init__(self, grid: dict, dilate_vox: int):
        occ = grid["occ"].float()[None, None]
        if dilate_vox > 0:
            occ = F.max_pool3d(occ, 2 * dilate_vox + 1, 1, dilate_vox)
        self.occ = occ[0, 0] > 0.5
        self.lo, self.voxel = grid["lo"], grid["voxel"]
        self.shape = torch.tensor(self.occ.shape, device=self.occ.device)

    def inside(self, p: torch.Tensor) -> torch.Tensor:
        idx = torch.floor((p - self.lo) / self.voxel + 0.5).long()
        ok = ((idx >= 0) & (idx < self.shape)).all(1)
        res = torch.zeros(len(p), dtype=torch.bool, device=p.device)
        i = idx[ok]
        res[ok] = self.occ[i[:, 0], i[:, 1], i[:, 2]]
        return res

    @torch.no_grad()
    def apply(self, params, shrink: float = 0.8, reach: float = 2.0) -> tuple[torch.Tensor, int]:
        """Returns (mask of Gaussians whose centre is outside, number of axes shortened)."""
        from gsplat.utils import normalized_quat_to_rotmat
        means = params["means"]
        outside = ~self.inside(means)
        R = normalized_quat_to_rotmat(F.normalize(params["quats"], dim=1))
        s = torch.exp(params["scales"])
        shortened = 0
        for k in range(3):
            ext = reach * s[:, k:k + 1] * R[:, :, k]
            out = ~(self.inside(means + ext) & self.inside(means - ext)) & ~outside
            params["scales"].data[out, k] += math.log(shrink)
            shortened += int(out.sum())
        params["opacities"].data[outside] = math.log(0.001 / 0.999)     # pruned / relocated next refine
        return outside, shortened


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
        "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), cfg.init_opacity, device=points.device))),
        "sh0": torch.nn.Parameter(sh[:, :1].contiguous()),
        "shN": torch.nn.Parameter(sh[:, 1:].contiguous()),
    })


def render(params, viewmat, K, width, height, sh_degree, absgrad=False, with_depth=False):
    """Premultiplied colour over black, alpha, info (info["depth"]: expected depth if with_depth)."""
    from gsplat import rasterization
    colors = torch.cat([params["sh0"], params["shN"]], 1)
    out, alpha, info = rasterization(
        means=params["means"], quats=params["quats"], scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]), colors=colors, viewmats=viewmat[None], Ks=K[None],
        width=width, height=height, sh_degree=sh_degree, packed=False, absgrad=absgrad,
        near_plane=0.01, far_plane=100.0, rasterize_mode="classic",
        render_mode="RGB+ED" if with_depth else "RGB")
    if with_depth:
        info["depth"] = out[0, ..., 3:4]
    return out[0, ..., :3], alpha[0], info


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


def _resize(t: torch.Tensor, h: int, w: int, mode: str = "area") -> torch.Tensor:
    """(H, W, C) image to (h, w, C)."""
    return F.interpolate(t.permute(2, 0, 1)[None], (h, w), mode=mode)[0].permute(1, 2, 0)


def _make_strategy(cfg: SplatConfig, n_views: int):
    """(strategy, state). MCMC keeps a fixed Gaussian budget and relocates dead ones (no clone/split)."""
    if cfg.strategy == "mcmc":
        from gsplat.strategy import MCMCStrategy
        strategy = MCMCStrategy(cap_max=cfg.mcmc_cap, noise_lr=5e5, refine_start_iter=cfg.refine_start,
                                refine_stop_iter=cfg.refine_stop, refine_every=cfg.refine_every,
                                min_opacity=0.005, verbose=False)
        return strategy, strategy.initialize_state()
    from gsplat.strategy import DefaultStrategy
    strategy = DefaultStrategy(prune_opa=cfg.prune_opa, grow_grad2d=cfg.grow_grad2d, grow_scale3d=0.01,
                               grow_scale2d=0.05, prune_scale3d=cfg.prune_scale3d, prune_scale2d=0.15,
                               refine_scale2d_stop_iter=4000, refine_start_iter=cfg.refine_start,
                               refine_stop_iter=cfg.refine_stop, reset_every=cfg.reset_every,
                               refine_every=cfg.refine_every, pause_refine_after_reset=n_views + cfg.refine_every,
                               absgrad=True, revised_opacity=False, verbose=False)
    return strategy, strategy.initialize_state(scene_scale=1.0)


def needle_mask(params, ratio: float, min_len_m: float) -> torch.Tensor:
    """Gaussians much longer than they are wide: the streaks that fill gaps between views."""
    s = torch.exp(params["scales"]).sort(dim=1, descending=True).values
    return (s[:, 0] > min_len_m) & (s[:, 0] > ratio * s[:, 1])


def train(views: dict, world: dict, cfg: SplatConfig, bar=None, init=None,
          extra_loss=None, preview_dir=None) -> tuple[torch.nn.ParameterDict, dict]:
    """`init` = (points, colours) replaces the visual hull as the starting point cloud; `extra_loss(params,
    step)` adds a regulariser (e.g. headpass's surface leash); views["loss_masks"] (optional, per view
    (H,W,1) or None) excludes pixels from every image loss - neither colour nor alpha is learned there."""
    from pytorch_msssim import SSIM

    from .splatkit_vendor import PerceptualLoss

    bar = bar or NullBar()
    torch.manual_seed(cfg.seed)
    device = views["w2c"].device
    n_views = len(views["names"])

    shell = interior = None
    if cfg.face_shell:
        from .priors import FaceShell
        shell = FaceShell(world, device)
    if cfg.interior_margin > 0:
        from .priors import BodyInterior
        interior = BodyInterior(world, device)
    if init is not None:
        points, colors = init
        grid = None
        log.info("Starting from %d given points", len(points))
    else:
        points, colors, grid = visual_hull(views, world, cfg, return_grid=True)
        log.info("Visual hull: %d surface points to start from", len(points))
    carver = HullCarver(grid, cfg.hull_carve_dilate) if cfg.hull_carve and grid is not None else None
    if len(points) < 1000:
        raise RuntimeError(f"visual hull has only {len(points)} points: the view mattes or cameras disagree")
    n_face = 0
    if cfg.face_mesh_init and views.get("prior") is not None:
        from .priors import drop_face_hull_points
        points, colors = drop_face_hull_points(points, colors, world)
        face_pts, face_rgb = views["prior"].face_points(views["images"][0], views["w2c"][0], views["K"][0])
        points, colors = torch.cat([points, face_pts]), torch.cat([colors, face_rgb])
        n_face = len(face_pts)
        log.info("Face: %d starting points on the body mesh's face", n_face)
    params = init_params(points, colors, cfg)
    if n_face:
        params["scales"].data[-n_face:] = math.log(0.003 * 0.7)

    lrs = {"means": cfg.lr_means, "scales": cfg.lr_scales, "quats": cfg.lr_quats,
           "opacities": cfg.lr_opacities, "sh0": cfg.lr_sh0, "shN": cfg.lr_shN}
    optimizers = {k: torch.optim.Adam([params[k]], lr=lr, eps=1e-15) for k, lr in lrs.items()}
    means_decay = (cfg.lr_means_final / cfg.lr_means) ** (1.0 / cfg.iterations)

    strategy, state = _make_strategy(cfg, n_views)
    strategy.check_sanity(params, optimizers)

    pose_delta = torch.zeros(n_views, 6, device=device, requires_grad=True)
    color_A = torch.eye(3, device=device).repeat(n_views, 1, 1).requires_grad_(True)
    color_b = torch.zeros(n_views, 3, device=device, requires_grad=True)
    extra_opt = torch.optim.Adam([{"params": [pose_delta], "lr": cfg.pose_lr},
                                  {"params": [color_A, color_b], "lr": cfg.color_lr}])
    # only generated views get camera refinement (video frames too if video_pose_refine): the photo and
    # restored frames have exact cameras
    kinds = views.get("kinds") or ["photo"] + ["generated"] * (n_views - 1)
    alpha_w = torch.tensor([cfg.restored_alpha if k == "restored" else 1.0 for k in kinds], device=device)

    ssim = SSIM(data_range=1.0, size_average=True, channel=3).to(device)
    perceptual = PerceptualLoss(VGG_WEIGHTS).to(device) if cfg.perceptual_weight > 0 else None

    # sampling: the photo gets a fixed share, restored frames (if any) a fixed share, generated views and
    # video frames the rest (split by video_share when both are present)
    n_rest = kinds.count("restored")
    n_vid = kinds.count("video")
    n_gen = n_views - 1 - n_rest - n_vid
    rest_share = cfg.restored_share if n_rest else 0.0
    remaining = 1 - cfg.photo_sample_share - rest_share
    vid_share = (remaining * cfg.video_share if n_gen else remaining) if n_vid else 0.0
    probs = torch.tensor([cfg.photo_sample_share if k == "photo" else
                          rest_share / max(n_rest, 1) if k == "restored" else
                          vid_share / n_vid if k == "video" else
                          (remaining - vid_share) / max(n_gen, 1) for k in kinds], device=device)
    shares = views.get("shares")
    if shares and any(s != 1.0 for s in shares):
        # per-view shares redistribute within each group, so the photo keeps photo_sample_share
        w = torch.tensor(shares, device=device)
        for kind in ("generated", "restored", "video"):
            sel = torch.tensor([k == kind for k in kinds], device=device)
            if sel.any() and probs[sel].sum() > 0:
                scaled = probs[sel] * w[sel]
                probs[sel] = scaled * (probs[sel].sum() / scaled.sum())
    probs = probs if n_views > 1 else torch.ones(1, device=device)
    gen = torch.Generator(device=device).manual_seed(cfg.seed)

    start = time.perf_counter()
    reported = 0
    next_preview_check = 0.0
    if preview_dir is not None:
        preview_frame = preview_dir / "training-preview.jpg"
        preview_lease = preview_dir / ".training-preview-enabled"
        try:
            preview_frame.unlink(missing_ok=True)
        except OSError:
            pass
    rate_file = preview_dir / "training-rate.csv" if preview_dir is not None else None
    previous_elapsed = 0.0
    previous_step = 0
    if rate_file:
        try:
            rate_file.write_text("", encoding="ascii")
        except OSError:
            rate_file = None
    for step in range(cfg.iterations):
        i = int(torch.multinomial(probs, 1, generator=gen))
        img, mat = views["images"][i], views["alphas"][i]
        cmap, fdepth, fmask = views["color_maps"][i], views["face_depth"][i], views["face_mask"][i]
        lmask = views["loss_masks"][i] if views.get("loss_masks") else None
        H0, W0 = img.shape[:2]
        d = 2.0 if step < cfg.half_res_until else 1.0
        ms = cfg.multiscale_video if kinds[i] == "video" else cfg.multiscale
        if d == 1.0 and ms > 0 and H0 > cfg.ms_min_height and \
                float(torch.rand(1, generator=gen, device=device)) < ms:
            # a random zoom-out: Gaussians must also look right when they cover less than a pixel
            d = math.exp(float(torch.rand(1, generator=gen, device=device)) * math.log(H0 / cfg.ms_min_height))
        h, w = round(H0 / d), round(W0 / d)
        K = views["K"][i].clone()
        K[0] *= w / W0
        K[1] *= h / H0
        if (h, w) != (H0, W0):
            img, mat = _resize(img, h, w), _resize(mat, h, w)
            cmap = _resize(cmap, h, w) if cmap is not None else None
            lmask = _resize(lmask, h, w) if lmask is not None else None
            if fdepth is not None:
                fdepth, fmask = _resize(fdepth, h, w, "nearest"), _resize(fmask, h, w, "nearest")
        use_depth = cfg.face_depth_weight > 0 and fdepth is not None

        # camera refinement only inside [pose_start, pose_stop); afterwards the corrected camera is kept fixed
        refinable = kinds[i] == "generated" or (kinds[i] == "video" and cfg.video_pose_refine)
        pose_on = refinable and cfg.pose_start <= step < cfg.pose_stop
        viewmat = apply_delta(views["w2c"][i], pose_delta[i] if pose_on else pose_delta[i].detach())
        sh_degree = min(step // cfg.sh_degree_interval, cfg.sh_degree)
        rgb, alpha, info = render(params, viewmat, K, w, h, sh_degree, absgrad=True, with_depth=use_depth)
        strategy.step_pre_backward(params, optimizers, state, step, info)

        if i != 0:
            rgb = rgb @ color_A[i].T + color_b[i] * alpha
        if preview_dir is not None and step % 10 == 0 and time.monotonic() >= next_preview_check:
            next_preview_check = time.monotonic() + 2
            try:
                if preview_lease.exists() and time.time() - preview_lease.stat().st_mtime < 15:
                    with torch.no_grad():
                        pixels = ((rgb.detach() + (1 - alpha.detach()) * 0.5).clamp(0, 1) * 255).byte().cpu().numpy()
                    frame = Image.fromarray(pixels)
                    frame.thumbnail((768, 768))
                    temporary = preview_frame.with_suffix(".tmp")
                    frame.save(temporary, format="JPEG", quality=84)
                    temporary.replace(preview_frame)
                    next_preview_check = time.monotonic() + 5
            except (OSError, RuntimeError, ValueError) as exc:
                log.debug("Training preview unavailable: %s", exc)
        bg = torch.rand(3, device=device)
        pred = rgb + (1 - alpha) * bg
        target = img * mat + (1 - mat) * bg
        if cmap is not None:
            # scale the colour gradient (not the value) where this view should have little say
            pred = pred * cmap + pred.detach() * (1 - cmap)
        if lmask is not None:
            # excluded pixels match their target exactly: no colour, SSIM or perceptual error there
            pred = pred * lmask + target * (1 - lmask)

        l1 = (pred - target).abs().mean()
        pred_n, target_n = pred.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None]
        loss = (1 - cfg.ssim_lambda) * l1 + cfg.ssim_lambda * (1 - ssim(pred_n, target_n))
        a_err = (alpha - mat).abs() if lmask is None else (alpha - mat).abs() * lmask
        loss = loss + cfg.alpha_weight * alpha_w[i] * a_err.mean()
        if perceptual is not None:
            p_pred, p_target = _perceptual_crop(pred_n, target_n, mat, cfg.perceptual_max_px, gen)
            pw = cfg.perceptual_weight
            if cfg.perceptual_ramp_to > 0:
                f = (step / cfg.iterations - cfg.perceptual_ramp_start) /                     max(cfg.perceptual_ramp_end - cfg.perceptual_ramp_start, 1e-6)
                pw = pw + (cfg.perceptual_ramp_to - pw) * min(max(f, 0.0), 1.0)
            loss = loss + pw * perceptual(p_pred, p_target)
        loss = views["weights"][i] * loss
        if pose_on:
            loss = loss + cfg.pose_reg * (pose_delta[i] ** 2).sum()
        loss = loss + cfg.color_reg * ((color_A[i] - torch.eye(3, device=device)) ** 2).sum() + \
            cfg.color_reg * (color_b[i] ** 2).sum()
        if use_depth:
            m = fmask * (alpha.detach() > 0.5)
            loss = loss + cfg.face_depth_weight * ((info["depth"] - fdepth).abs() * m).sum() / m.sum().clamp_min(1.0)
        if cfg.opacity_reg > 0:
            loss = loss + cfg.opacity_reg * torch.sigmoid(params["opacities"]).mean()
        if cfg.scale_reg > 0:
            loss = loss + cfg.scale_reg * torch.exp(params["scales"]).mean()
        if extra_loss is not None:
            loss = loss + extra_loss(params, step)
        if cfg.aniso_weight > 0:
            s = torch.exp(params["scales"]).sort(dim=1, descending=True).values
            loss = loss + cfg.aniso_weight * F.relu(s[:, 0] / s[:, 1].clamp_min(1e-9) - cfg.aniso_ratio).mean()
        loss.backward()

        # same order as gsplat's simple_trainer: optimise, then densify/prune
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        extra_opt.step()
        extra_opt.zero_grad(set_to_none=True)
        if cfg.strategy == "mcmc":
            strategy.step_post_backward(params, optimizers, state, step, info,
                                        lr=optimizers["means"].param_groups[0]["lr"])
        else:
            strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
        for group in optimizers["means"].param_groups:
            group["lr"] *= means_decay
        if carver is not None and cfg.refine_start <= step < cfg.refine_stop and step % cfg.refine_every == 0:
            carver.apply(params)
        if shell is not None:
            shell.project(params, cfg.face_shell_m)
        if interior is not None and step % cfg.refine_every == 0 and step < cfg.iterations - 1:
            with torch.no_grad():
                deep = interior.inside(params["means"], cfg.interior_margin, cfg.interior_face_margin)
                params["opacities"].data[deep] = math.log(0.001 / 0.999)   # pruned / relocated next refine

        if step % 50 == 0 or step == cfg.iterations - 1:
            bar.update(step + 1 - reported)
            reported = step + 1
            bar.set_postfix_str(f"loss {loss.item():.4f}  gaussians {len(params['means']):,}")
            elapsed = time.perf_counter() - start
            if rate_file and (elapsed - previous_elapsed >= 2 or step + 1 == cfg.iterations):
                rate = (step + 1 - previous_step) / max(1e-9, elapsed - previous_elapsed)
                try:
                    with rate_file.open("a", encoding="ascii") as stream:
                        stream.write(f"{elapsed:.3f},{step + 1},{rate:.3f}\n")
                except OSError:
                    rate_file = None
                previous_step, previous_elapsed = step + 1, elapsed
        if step % 1000 == 0 or step == cfg.iterations - 1:
            log.debug("step %5d loss %.4f l1 %.4f gaussians %d %.1fs", step, loss.item(), l1.item(),
                      len(params["means"]), time.perf_counter() - start)

    if interior is not None:
        with torch.no_grad():
            deep = interior.inside(params["means"], cfg.interior_margin, cfg.interior_face_margin)
            params["opacities"].data[deep] = math.log(0.001 / 0.999)       # clean() removes them
        log.info("Interior: %d Gaussians inside the body cleared at the end", int(deep.sum()))
    carved = None
    if carver is not None:
        outside, shortened = carver.apply(params)
        carved = {"outside": int(outside.sum()), "axes_shortened": shortened}
        log.info("Hull carving at the end: %d outside, %d axes shortened", carved["outside"], shortened)
    with torch.no_grad():
        deltas = pose_delta.detach().clone()
        deltas[0] = 0
        refined = torch.stack([apply_delta(views["w2c"][i], deltas[i]) for i in range(n_views)])
    stats = {"train_seconds": round(time.perf_counter() - start, 1), "gaussians": len(params["means"]),
             "hull_carve_final": carved,
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
    needles = 0
    if cfg.needle_ratio > 0:
        needle = needle_mask(params, cfg.needle_ratio, cfg.needle_min_m)
        needles = int((needle & keep).sum())
        keep &= ~needle
    removed = int((~keep).sum())
    for k in list(params.keys()):
        params[k] = torch.nn.Parameter(params[k][keep])
    return {"removed": removed, "needles_removed": needles, "remaining": int(keep.sum())}


@torch.no_grad()
def bake_face_into(params, world: dict, hr_photo: Path, cfg: SplatConfig) -> dict:
    """Replace the face core with the photo on the body mesh (priors.bake_face); params changed in place.
    Skipped when the mesh's eyes and nose do not land on the photo's (a turned or hidden face)."""
    from .priors import bake_face
    device = params["means"].device
    rgba = torch.from_numpy(np.asarray(Image.open(hr_photo).convert("RGBA"), np.float32) / 255).to(device)
    K = torch.as_tensor(world["K"], device=device).float().clone()
    w, h = world["size"]
    K[0] *= rgba.shape[1] / w
    K[1] *= rgba.shape[0] / h
    misfit = _face_misfit(world, rgba, K.cpu().numpy())
    if misfit is None or misfit > 0.25:
        log.warning("Face bake skipped: %s", "no face found in the photo" if misfit is None else
                    f"the body mesh's face is {misfit:.2f} eye-distances off the photo's")
        return {"skipped": True, "misfit": misfit}
    plain = {k: v.detach() for k, v in params.items()}
    baked, stats = bake_face(plain, world, rgba, torch.as_tensor(world["photo_w2c"], device=device).float(), K,
                             radius_m=cfg.face_bake_m)
    for k in list(params.keys()):
        params[k] = torch.nn.Parameter(baked[k])
    return stats | {"misfit": round(misfit, 3)}


def _face_misfit(world: dict, rgba: torch.Tensor, K: np.ndarray) -> float | None:
    """Mean distance between the mesh's eyes/nose and the detected ones, in eye distances (None: no face)."""
    from .body import L_EYE, NOSE, R_EYE
    from .cameras import project
    try:
        from .evaluate import FaceID
        det = FaceID().detect((rgba[..., :3].cpu().numpy() * 255).astype(np.uint8))
    except Exception as exc:  # noqa: BLE001 - the check is optional (models live in ComfyUI's folder)
        log.debug("face check unavailable: %s", exc)
        return 0.0
    if det is None:
        return None
    uv, _ = project(world["kp3d"][[R_EYE, L_EYE, NOSE]], world["photo_w2c"], K)   # image-left eye first
    kps = det[1][:3]
    return float(np.linalg.norm(uv - kps, axis=1).mean() / np.linalg.norm(kps[0] - kps[1]))


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

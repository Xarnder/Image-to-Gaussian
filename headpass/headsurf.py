"""H2: one continuous head surface for the depth stand-in, instead of a face plate on a skull.

Why: the product's stand-in (the pass-2 splat) has the face baked onto the SAM 3D Body mesh's face, one flat
disc per photo pixel, and the face priors pin the face there too, while the forehead, temples, skull and hair
come from the Qwen views. The two don't meet: from the side the face is a plate in front of the skull, with a
step at its edge. Wan paints exactly what the depth control shows, so the final model inherits the step as a
seam across the forehead and temples (forehead kink 26 mm on the Nurse Man's bald head).

How, following CAP4D's lesson (face, forehead and skull must be ONE surface), as a radius map r(el, az) around
the head centre in 1-degree bins:
  1. Head frame: centre = mean of the body model's nose, eyes and ears; head region = above the chin, within
     reach of the centre, and below the mouth only inside a neck column (in_head).
  2. The old stand-in's outer surface per direction (splat_radius_map), with its colour.
  3. Face core: the body mesh's face seen from the photo camera, inside the photo's jaw contour and brows
     raised to about the hairline (face_radius_map), rim eroded.
  4. Old face: directions where the old surface lies on the mesh's face, near the face and skin-coloured.
  5. fuse: everything as offsets from the body mesh's head (a template carrying face, skull and ears as one
     surface): observed offsets = face core + old surface outside the old face; the old face's directions
     filled as a membrane; the whole field smoothed once; the face core spliced in exactly with a smoothstep.
  6. The old head is replaced by opaque discs tiling the surface in its tangent plane (surface_gaussians).

Every simpler variant left an artefact Wan would paint; they are listed in
next-stage-research/docs/headpass_results.md section 2. The same surface is the head pass's leash target.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

N_AZ, N_EL = 360, 180          # 1 degree bins


def head_frame(world: dict, ref) -> dict:
    """centre, cut height (world y), chin point; from the body model's head keypoints and the photo's chin."""
    from i2g.evaluate import _head
    from .metrics import mesh_depth
    c = _head(world)
    depth, alpha = mesh_depth(world, ref)
    u, v = ref.lm[8, :2]                     # chin landmark
    d = float(np.median(depth[int(v) - 2:int(v) + 3, int(u) - 2:int(u) + 3]))
    ray = np.linalg.inv(ref.K) @ np.array([u, v, 1.0])
    chin_cam = ray / ray[2] * d
    R, t = ref.w2c[:3, :3], ref.w2c[:3, 3]
    chin = R.T @ (chin_cam - t)
    # the cut is at the chin: below it a ray from the centre crosses both the jaw and the neck, so the radius
    # is ambiguous (a cut 2 cm lower grew strips hanging under the jaw); the old neck stays as it was
    return {"center": c, "chin": chin, "cut_y": float(chin[1])}


def in_head(p: torch.Tensor, frame: dict, reach_m: float = 0.25, neck_r_m: float = 0.09, jaw_m: float = 0.08) -> torch.Tensor:
    """The head's region: above the cut, within reach of the centre, and below ear level only inside a
    vertical neck column (so shoulders and the collar are not head). The column starts 8 cm below the
    centre (about the mouth; shoulders are ~17 cm down) and is 9 cm wide: narrower clipped wide cheeks and left
    holes in the surface."""
    c = torch.as_tensor(frame["center"], device=p.device).float()
    d = p - c
    horiz = torch.sqrt(d[:, 0] ** 2 + d[:, 2] ** 2)
    return (p[:, 1] > frame["cut_y"]) & (d.norm(dim=-1) < reach_m) & ((d[:, 1] > -jaw_m) | (horiz < neck_r_m))


def _dirs(n_az=N_AZ, n_el=N_EL, device="cuda") -> torch.Tensor:
    az = (torch.arange(n_az, device=device) + 0.5) / n_az * 2 * math.pi - math.pi
    el = (torch.arange(n_el, device=device) + 0.5) / n_el * math.pi - math.pi / 2
    E, A = torch.meshgrid(el, az, indexing="ij")
    return torch.stack([torch.cos(E) * torch.sin(A), torch.sin(E), torch.cos(E) * torch.cos(A)], -1)   # el, az, 3


def _to_bins(p: torch.Tensor, c: torch.Tensor):
    d = p - c
    rho = d.norm(dim=-1)
    az = torch.atan2(d[:, 0], d[:, 2])
    el = torch.asin((d[:, 1] / rho.clamp_min(1e-9)).clamp(-1, 1))
    ia = ((az + math.pi) / (2 * math.pi) * N_AZ).long().clamp(0, N_AZ - 1)
    ie = ((el + math.pi / 2) / math.pi * N_EL).long().clamp(0, N_EL - 1)
    return ie, ia, rho


def _icosphere_dirs(n: int = 162) -> np.ndarray:
    """Roughly uniform directions (Fibonacci sphere)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = math.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.cos(phi), np.sin(th) * np.sin(phi)], -1)


def _look_at(eye, target):
    from i2g.cameras import look_at
    return look_at(eye, target)


@torch.no_grad()
def splat_radius_map(params: dict, frame: dict, size: int = 384, dist: float = 1.0, fov_m: float = 0.5,
                     min_cos: float = 0.75, with_color: bool = False):
    """The splat's head surface per direction (el, az) around the head centre. NaN = no head there.

    Each of 162 cameras around the head sees the surface facing it; a point only votes for its direction
    when that camera looks at it roughly head-on (cos > min_cos): grazing views put silhouette-edge depth
    and semi-transparent fuzz in the wrong bins. Per bin: a cos-weighted mean, then the same mean over the
    votes within 1 cm of it (rejects stray hair and floaters), then a 3x3 median over the map."""
    from .metrics import render_depth
    dev = params["means"].device
    c = torch.as_tensor(frame["center"], device=dev).float()
    # the depth proxy's opacity sharpening, so the first surface wins over semi-transparent fuzz and interior
    o = 1 - (1 - torch.sigmoid(params["opacities"])).clamp_min(1e-6) ** 8
    params = dict(params, opacities=torch.logit(o.clamp(1e-6, 1 - 1e-6)))
    f = size * dist / fov_m
    K = np.array([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1.0]])
    vs, us = torch.meshgrid(torch.arange(size, device=dev), torch.arange(size, device=dev), indexing="ij")
    all_flat, all_rho, all_w, all_rgb = [], [], [], []
    from i2g.splat import render as splat_render
    for d in _icosphere_dirs():
        eye = frame["center"] + dist * d
        w2c = _look_at(eye, frame["center"])
        depth, alpha = render_depth(params, w2c, K, size, size)
        depth, alpha = torch.as_tensor(depth, device=dev), torch.as_tensor(alpha, device=dev)
        rgb, _, _ = splat_render(params, torch.as_tensor(w2c, device=dev).float(), torch.as_tensor(K, device=dev).float(),
                                 size, size, 0)
        ok = alpha > 0.5
        z = depth[ok]
        x = (us[ok] + 0.5 - K[0, 2]) / K[0, 0] * z
        y = (vs[ok] + 0.5 - K[1, 2]) / K[1, 1] * z
        R = torch.as_tensor(w2c[:3, :3], device=dev).float()
        t = torch.as_tensor(w2c[:3, 3], device=dev).float()
        pts = (torch.stack([x, y, z], -1) - t) @ R
        cosv = F.normalize(pts - c, dim=-1) @ torch.as_tensor(d, device=dev).float()
        keep = in_head(pts, frame) & (cosv > min_cos)
        ie, ia, rho = _to_bins(pts[keep], c)
        all_flat.append(ie * N_AZ + ia)
        all_rho.append(rho)
        all_w.append(cosv[keep] ** 4)
        all_rgb.append(rgb[ok][keep])
    flat, rho, w = torch.cat(all_flat), torch.cat(all_rho), torch.cat(all_w)
    rgbs = torch.cat(all_rgb)
    n = N_EL * N_AZ

    def wmean(mask):
        sw = torch.zeros(n, device=dev).scatter_add(0, flat[mask], w[mask])
        sr = torch.zeros(n, device=dev).scatter_add(0, flat[mask], (w * rho)[mask])
        return sr / sw.clamp_min(1e-9), sw
    m1, sw = wmean(torch.ones_like(rho, dtype=torch.bool))
    m2, sw2 = wmean((rho - m1[flat]).abs() < 0.01)
    r = torch.where(sw2 > 0, m2, m1)
    r = torch.where(sw > 0, r, torch.nan).view(N_EL, N_AZ)
    # 3x3 median ignoring NaN (azimuth wraps)
    x = F.pad(r[None, None], (1, 1, 0, 0), mode="circular")
    x = F.pad(x, (0, 0, 1, 1), mode="replicate")[0, 0]
    win = torch.stack([x[i:i + N_EL, j:j + N_AZ] for i in range(3) for j in range(3)], 0)
    med = torch.nanmedian(win, dim=0).values
    r = torch.where(torch.isnan(r), r, med)
    if not with_color:
        return r
    sw = torch.zeros(n, device=dev).scatter_add(0, flat, w)
    col = torch.stack([torch.zeros(n, device=dev).scatter_add(0, flat, w * rgbs[:, c]) for c in range(3)], -1)
    col = (col / sw.clamp_min(1e-9)[:, None]).view(N_EL, N_AZ, 3)
    return r, col


@torch.no_grad()
def face_radius_map(world: dict, ref, frame: dict, brow_raise: float = 0.9, device="cuda", erode_bins: int = 2):
    """(radius map of the body mesh's face seen from the photo, core mask) on the same grid.

    The core is the photo's face: the convex hull of the jaw contour and the brows raised by
    `brow_raise` x the brow-to-nose-tip height (to take in the forehead up to about the hairline)."""
    import cv2
    from .metrics import mesh_depth
    depth, alpha = mesh_depth(world, ref)
    lm = ref.lm[:, :2].copy()
    up = brow_raise * (ref.lm[30, 1] - ref.lm[17:27, 1].mean())
    brows = lm[17:27].copy()
    brows[:, 1] -= up
    hull = cv2.convexHull(np.concatenate([lm[:17], brows]).astype(np.int32))
    m = np.zeros((ref.h, ref.w), np.uint8)
    cv2.fillConvexPoly(m, hull, 1)
    m = m.astype(bool) & (alpha > 0.99)
    v, u = np.nonzero(m)
    z = depth[v, u]
    x = (u + 0.5 - ref.K[0, 2]) / ref.K[0, 0] * z
    y = (v + 0.5 - ref.K[1, 2]) / ref.K[1, 1] * z
    cam = np.stack([x, y, z], -1)
    pts = (cam - ref.w2c[:3, 3]) @ ref.w2c[:3, :3]
    c = torch.as_tensor(frame["center"], device=device).float()
    ie, ia, rho = _to_bins(torch.as_tensor(pts, device=device).float(), c)
    flat = ie * N_AZ + ia
    rsum = torch.zeros(N_EL * N_AZ, device=device).scatter_add(0, flat, rho)
    cnt = torch.zeros(N_EL * N_AZ, device=device).scatter_add(0, flat, torch.ones_like(rho))
    core = (cnt > 0).view(N_EL, N_AZ)
    r = (rsum / cnt.clamp_min(1)).view(N_EL, N_AZ)
    # close pinholes in the core (bins no photo pixel landed in)
    core_f = F.max_pool2d(core.float()[None, None], 3, 1, 1)[0, 0]
    core_f = -F.max_pool2d(-core_f[None, None], 3, 1, 1)[0, 0] > 0.5
    r_blur = F.avg_pool2d(torch.where(core, r, 0)[None, None], 3, 1, 1)[0, 0] / \
        F.avg_pool2d(core.float()[None, None], 3, 1, 1)[0, 0].clamp_min(1e-6)
    r = torch.where(core, r, r_blur)
    # the outermost bins of the core are seen at grazing angles from the photo camera, where the mesh depth
    # per pixel is unreliable; they drew a thin line around the face. Erode them; the blend covers the gap.
    core_f = ~_dilate(~core_f, erode_bins) if erode_bins > 0 else core_f
    return torch.where(core_f, r, torch.nan), core_f


def _smooth_nan(r: torch.Tensor, sigma_bins: float) -> torch.Tensor:
    """Gaussian smoothing that ignores NaN bins (and leaves them NaN); azimuth wraps."""
    if sigma_bins <= 0:
        return r
    k = int(math.ceil(3 * sigma_bins))
    x = torch.arange(-k, k + 1, device=r.device).float()
    g = torch.exp(-x ** 2 / (2 * sigma_bins ** 2))
    g = g / g.sum()
    valid = (~torch.isnan(r)).float()
    v = torch.nan_to_num(r, nan=0.0) * valid

    def conv(t):
        t = F.pad(t[None, None], (k, k, 0, 0), mode="circular")
        t = F.conv2d(t, g.view(1, 1, 1, -1))
        t = F.pad(t, (0, 0, k, k), mode="replicate")
        return F.conv2d(t, g.view(1, 1, -1, 1))[0, 0]
    out = conv(v) / conv(valid).clamp_min(1e-6)
    return torch.where(valid > 0, out, torch.nan)


def _fill_nan(r: torch.Tensor, iters: int = 200) -> torch.Tensor:
    """Fill NaN bins by repeated neighbour averaging (azimuth wraps)."""
    r = r.clone()
    for _ in range(iters):
        nan = torch.isnan(r)
        if not nan.any():
            break
        x = torch.nan_to_num(r, nan=0.0)
        w = (~nan).float()
        xs = F.pad(x[None, None], (1, 1, 0, 0), mode="circular")
        ws = F.pad(w[None, None], (1, 1, 0, 0), mode="circular")
        xs = F.pad(xs, (0, 0, 1, 1), mode="replicate")
        ws = F.pad(ws, (0, 0, 1, 1), mode="replicate")
        s = F.avg_pool2d(xs, 3, 1)[0, 0]
        n = F.avg_pool2d(ws, 3, 1)[0, 0]
        r = torch.where(nan & (n > 0), s / n.clamp_min(1e-9), r)
    return r


def _dilate(m: torch.Tensor, k: int) -> torch.Tensor:
    x = F.pad(m.float()[None, None], (k, k, 0, 0), mode="circular")
    return F.max_pool2d(x, 2 * k + 1, 1, (k, 0))[0, 0] > 0.5


def _fill_holes(mask: torch.Tensor, max_bins: int = 1500) -> torch.Tensor:
    """Close enclosed holes (components of ~mask smaller than max_bins): a missing bin is a see-through
    crack in the surface, which the shaded depth shows as a crumpled patch."""
    import scipy.ndimage as nd
    m = mask.cpu().numpy()
    lab, n = nd.label(~m)
    if n:
        sizes = nd.sum(~m, lab, range(1, n + 1))
        small = np.isin(lab, 1 + np.nonzero(sizes < max_bins)[0])
        m = m | small
    return torch.as_tensor(m, device=mask.device)


def _distance(mask: torch.Tensor, limit: int) -> torch.Tensor:
    """Bins from `mask` (0 inside), by repeated 1-bin dilation, capped at limit + 1."""
    dist = torch.full(mask.shape, float(limit + 1), device=mask.device)
    grown = mask.clone()
    dist[grown] = 0
    for i in range(1, limit + 1):
        nxt = _dilate(grown, 1)
        dist[nxt & ~grown] = i
        grown = nxt
    return dist


@torch.no_grad()
def mesh_radius_map(world: dict, frame: dict, n: int = 1_500_000) -> torch.Tensor:
    """The body mesh's head surface (outermost per direction, head region only)."""
    from i2g.priors import sample_surface
    pts = torch.as_tensor(sample_surface(world, n, seed=5), device="cuda").float()
    pts = pts[in_head(pts, frame)]
    ie, ia, rho = _to_bins(pts, torch.as_tensor(frame["center"], device="cuda").float())
    r = torch.zeros(N_EL * N_AZ, device="cuda").scatter_reduce(0, ie * N_AZ + ia, rho, reduce="amax")
    r = r.view(N_EL, N_AZ)
    return _fill_nan(torch.where(r > 0, r, torch.nan))


def _close_el(r: torch.Tensor, k: int) -> torch.Tensor:
    """Grey-scale closing of the radius map along elevation (each meridian), window 2k+1 bins: fills grooves
    that run across the head (a dent at the top of the forehead) narrower than the window, and keeps the
    overall size. NaN bins stay NaN."""
    if k <= 0:
        return r
    nan = torch.isnan(r)
    x = _fill_nan(r)[None, None]                               # 1, 1, el, az
    x = x.permute(0, 3, 1, 2).reshape(-1, 1, r.shape[0])       # az, 1, el
    pad = F.pad(x, (k, k), mode="replicate")
    dil = F.max_pool1d(pad, 2 * k + 1, 1)
    ero = -F.max_pool1d(F.pad(-dil, (k, k), mode="replicate"), 2 * k + 1, 1)
    out = ero.reshape(r.shape[1], r.shape[0]).T
    return torch.where(nan, r, out)


def _harmonic_fill(obs: torch.Tensor, region: torch.Tensor, iters: int = 3000) -> torch.Tensor:
    """NaN bins of obs inside region solved as a membrane (Laplace) with the observed bins fixed."""
    free = torch.isnan(obs) & region
    x = _fill_nan(obs, iters=400)
    for _ in range(iters):
        p = F.pad(x[None, None], (1, 1, 0, 0), mode="circular")
        p = F.pad(p, (0, 0, 1, 1), mode="replicate")[0, 0]
        avg = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) / 4
        x = torch.where(free, avg, x)
    return x


def fuse(r_skull: torch.Tensor, r_face: torch.Tensor, core: torch.Tensor, plate: torch.Tensor, r_mesh: torch.Tensor,
         band_m: float = 0.02, face_shift_m: float = 0.0, obs_smooth_bins: float = 1.5,
         final_smooth_bins: float = 3.0, skin_w: torch.Tensor | None = None,
         skin_smooth_bins: float = 0.0, close_bins: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """(fused radius map, gap mask). The head is ONE surface, written as offsets from the body mesh's head
    (a template that carries face, forehead, skull and ears together):

      * observed offsets: the photo face in the core (~0), and the old stand-in's outer surface (skull or
        hair) everywhere except the old face plate's directions
      * the plate's directions are filled as a membrane between the face and the surrounding surface
      * the whole field is smoothed once (no borders between separately built zones survive), and the face
        core is put back exactly, faded in over band_m at its rim

    There is deliberately no hair/skin threshold: every threshold tried drew a line on the head where it
    flipped (bald temples that sit higher than the smoothed template read as "hair"). Earlier variants and
    what they left, all painted by Wan: blending raw radii (a ridge where the mesh's bigger head met the
    real skull), zone-by-zone assembly (creases at zone borders), bridging to hair with stretched discs
    (fans)."""
    skull = ~torch.isnan(r_skull)
    rho = float(torch.nanmedian(r_face))
    k = max(1, int(round(band_m / (rho * math.pi / N_EL))))
    head = _fill_holes(skull | core | plate, max_bins=300)
    # the skull with its grooves filled: the pass-2 splat's skull is concave at the top of the forehead (a
    # 2-3 cm dent on both test subjects, from the Qwen-view carving); Wan painted it as a dip in the profile
    r_sk = _close_el(_smooth_nan(r_skull, obs_smooth_bins), close_bins)
    obs = torch.where(core, r_face + face_shift_m - r_mesh,
                      torch.where(skull & ~plate, r_sk - r_mesh, torch.nan))
    off = _harmonic_fill(obs, head)
    off = _smooth_nan(torch.where(head, off, torch.nan), final_smooth_bins)
    off = _fill_nan(off)
    if skin_w is not None and skin_smooth_bins > 0:
        # v3: a bald scalp is smooth in reality; the old stand-in's bumps there were painted by Wan as ridges.
        # Smooth much more where the surface is skin-coloured, keep hair detail where it is not.
        strong = _fill_nan(_smooth_nan(torch.where(head, off, torch.nan), skin_smooth_bins))
        off = skin_w * strong + (1 - skin_w) * off
    d_core = _distance(core, k + 1)
    t = (d_core / k).clamp(0, 1)
    w_face = 1 - t * t * (3 - 2 * t)
    off_f = _fill_nan(torch.where(core, r_face + face_shift_m - r_mesh, torch.nan))
    r = r_mesh + w_face * off_f + (1 - w_face) * off
    gap = head & ~core & (plate | ~skull | (d_core <= k))
    return torch.where(head, r, torch.nan), gap


@torch.no_grad()
def surface_gaussians(r: torch.Tensor, frame: dict, colors_from: dict | None = None, supersample: int = 2):
    """Opaque flat Gaussians tiling the fused surface. Each is a disc in the surface's own tangent plane
    (from the grid's neighbours), sized to its grid cell: discs oriented along the radius instead stand
    like shingles wherever the surface slopes away from the head centre (temples, the face-to-skull
    transition), leaving gaps that the shaded depth shows as crumpled patches."""
    dev = r.device
    n_el, n_az = N_EL * supersample, N_AZ * supersample
    rr = F.interpolate(_fill_nan(r)[None, None], size=(n_el, n_az), mode="bilinear", align_corners=False)[0, 0]
    ok = F.interpolate((~torch.isnan(r)).float()[None, None], size=(n_el, n_az), mode="nearest")[0, 0] > 0.5
    c = torch.as_tensor(frame["center"], device=dev).float()
    P = c + _dirs(n_az, n_el, dev) * rr[..., None]                          # el, az, 3
    ok &= in_head(P.view(-1, 3), frame).view(ok.shape)
    Pa = F.pad(P.permute(2, 0, 1)[None], (1, 1, 0, 0), mode="circular")
    Pa = F.pad(Pa, (0, 0, 1, 1), mode="replicate")[0].permute(1, 2, 0)
    t_az = (Pa[1:-1, 2:] - Pa[1:-1, :-2]) / 2                              # one cell along azimuth
    t_el = (Pa[2:, 1:-1] - Pa[:-2, 1:-1]) / 2                              # one cell along elevation
    nrm = F.normalize(torch.cross(t_az, t_el, dim=-1), dim=-1)
    out = F.normalize(P - c, dim=-1)
    nrm = torch.where((nrm * out).sum(-1, keepdim=True) < 0, -nrm, nrm)
    xax = F.normalize(t_az, dim=-1)
    yax = F.normalize(torch.cross(nrm, xax, dim=-1), dim=-1)
    # a disc covers its own cell; across a real depth step (cheek to hair curtain) the cell is stretched
    # and the disc would become a sheet bridging the step (seen as streaky "fans"), so cap its size
    cell = rr * math.pi / n_el
    sx = torch.minimum(t_az.norm(dim=-1), 2.5 * cell)
    sy = torch.minimum((t_el * yax).sum(-1).abs().clamp_min(1e-5), 2.5 * cell)
    pts, xax, yax, nrm, sx, sy = P[ok], xax[ok], yax[ok], nrm[ok], sx[ok], sy[ok]
    n = len(pts)
    thin = torch.minimum(sx, sy) * 0.3
    scales = torch.log(torch.stack([sx * 0.75, sy * 0.75, thin], -1).clamp_min(2e-4))
    Rm = torch.stack([xax, yax, nrm], -1)
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(Rm.cpu().numpy()).as_quat()           # x y z w
    quats = torch.as_tensor(q[:, [3, 0, 1, 2]], device=dev).float()
    sh0 = torch.zeros(n, 1, 3, device=dev)
    if colors_from is not None:
        from scipy.spatial import cKDTree
        tree = cKDTree(colors_from["means"].cpu().numpy())
        _, j = tree.query(pts.cpu().numpy(), k=1)
        sh0 = colors_from["sh0"][torch.as_tensor(j, device=dev)]
    shN = torch.zeros(n, colors_from["shN"].shape[1] if colors_from is not None else 0, 3, device=dev)
    return {"means": pts, "scales": scales, "quats": quats, "opacities": torch.full((n,), 6.0, device=dev),
            "sh0": sh0, "shN": shN}


@torch.no_grad()
def replace_head(params: dict, r: torch.Tensor, frame: dict, affected: torch.Tensor, hair_m: float = 0.03,
                 hair_opacity: float = 0.5):
    """The stand-in without its head. Every Gaussian in the head region goes, except strands clearly outside
    the new surface (more than hair_m beyond it, and fairly opaque): the depth proxy sharpens opacities, so
    even faint leftovers near the head show up in the control as crumpled patches. In the face core and
    band directions nothing is kept (the old plate and its rim)."""
    c = torch.as_tensor(frame["center"], device=params["means"].device).float()
    ie, ia, rho = _to_bins(params["means"], c)
    rf = _fill_nan(r)[ie, ia]
    head = in_head(params["means"], frame)
    strand = (rho > rf + hair_m) & (torch.sigmoid(params["opacities"]) > hair_opacity) & ~affected[ie, ia]
    drop = head & ~strand
    return {k: v[~drop] for k, v in params.items()}, int(drop.sum())


def cat_params(a: dict, b: dict) -> dict:
    return {k: torch.cat([a[k], b[k]], 0) for k in a}


def _skin_like(col: torch.Tensor, core: torch.Tensor, tol: float) -> torch.Tensor:
    """Bins whose colour is close to the face core's median colour (brightness normalised, so a shadowed
    cheek is still skin while dark hair is not)."""
    lum = col.mean(-1, keepdim=True).clamp_min(1e-3)
    chroma = col / lum
    ref_c = chroma[core].median(0).values
    ref_l = lum[core].median()
    d = (chroma - ref_c).norm(dim=-1) + 0.5 * (torch.log(lum[..., 0] / ref_l)).abs()
    return d < tol * 3


def _on_mesh_face(points: torch.Tensor, world: dict, radius_m: float, tol_m: float) -> torch.Tensor:
    """Points lying on the body mesh's face (within tol_m of it, and within radius_m of the face centre): where
    the old stand-in's face was baked or pinned. Hair beside the face lies further out and is not matched
    (a plain radius test flattened the hair next to the face into a smooth fan)."""
    from scipy.spatial import cKDTree
    from i2g.priors import in_face, sample_surface
    mesh = sample_surface(world, 400_000, seed=3)
    mesh = mesh[in_face(mesh, world, front_m=0.0, radius_m=radius_m)]
    d, _ = cKDTree(mesh).query(points.cpu().numpy(), k=1)
    return torch.as_tensor(d < tol_m, device=points.device)


def build_standin(standin_ply: Path, world: dict, ref, out_ply: Path, band_m: float = 0.02,
                  face_shift_m: float = 0.0, plate_m: float = 0.12, mesh_tol_m: float = 0.012, skin_tol: float = 0.12,
                  skin_smooth_bins: float = 0.0, close_bins: int = 0,
                  save_maps: Path | None = None) -> dict:
    from i2g.evaluate import load_ply
    from i2g.splat import export_ply
    from i2g.priors import in_face
    params = load_ply(standin_ply)
    frame = head_frame(world, ref)
    # the skull: the old stand-in's front surface, except in the directions where that front surface is the
    # old face (the baked plate and whatever the face priors pinned there) - those become a gap to bridge.
    # (Removing the plate's Gaussians first, with a 3D ball, punched holes in the temples and exposed
    # interior junk as skull.)
    r_f, core = face_radius_map(world, ref, frame)
    r_s, col = splat_radius_map(params, frame, with_color=True)
    c = torch.as_tensor(frame["center"], device="cuda").float()
    surf = c + _dirs(device="cuda") * torch.nan_to_num(r_s, nan=0.0)[..., None]
    # the old face = on the body mesh's face AND within the photo's face outline (grown by 8 bins): the bake
    # only ever covered photo face pixels, and hair framing the face lies on the (wider) mesh face too
    plate = _on_mesh_face(surf.view(-1, 3), world, plate_m, mesh_tol_m).view(N_EL, N_AZ) & ~torch.isnan(r_s)
    plate &= _dilate(core, 14)
    # ... and skin-coloured: hair framing the face also lies close to the (bigger) mesh face, and treating
    # it as old face turned it into a smooth veil. Skin = within skin_tol of the face core's median colour
    # (chroma-weighted distance, so shading does not matter much).
    plate &= _skin_like(col, core, skin_tol)
    plate = _dilate(~_dilate(~_dilate(plate, 2), 2), 2)          # closed, then grown by two bins
    r_s = torch.where(plate, torch.nan, r_s)
    skin_w = None
    if skin_smooth_bins > 0:
        sk = _skin_like(col, core, skin_tol).float()
        skin_w = _fill_nan(_smooth_nan(torch.where(torch.isnan(r_s) & ~plate, torch.nan, sk), 3.0))
        skin_w = (skin_w * ~core).clamp(0, 1)
    r, band = fuse(r_s, r_f, core, plate, mesh_radius_map(world, frame), band_m=band_m, face_shift_m=face_shift_m,
                   skin_w=skin_w, skin_smooth_bins=skin_smooth_bins, close_bins=close_bins)
    # how big the step was: splat vs face along the core's rim
    rim = _dilate(core, 1) & ~core
    step = (r - _fill_nan(r_f + face_shift_m))[rim]          # the face side of the rim vs the fused surface
    body, n_drop = replace_head(params, r, frame, core | band)
    head = surface_gaussians(r, frame, colors_from=params)
    out = cat_params(body, head)
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    export_ply(out, out_ply)
    stats = {"dropped": n_drop, "head_gaussians": len(head["means"]), "band_m": band_m,
             "face_shift_m": face_shift_m,
             "rim_step_mm_median": round(float(torch.nanmedian(step)) * 1000, 1),
             "rim_step_mm_p90abs": round(float(torch.nanquantile(step.abs()[~torch.isnan(step)], 0.9)) * 1000, 1),
             "center": [round(float(x), 4) for x in frame["center"]], "cut_y": round(frame["cut_y"], 4)}
    if save_maps is not None:
        np.savez_compressed(save_maps, r=r.cpu().numpy(), r_splat=r_s.cpu().numpy(), r_face=r_f.cpu().numpy(),
                            core=core.cpu().numpy(), band=band.cpu().numpy(), center=frame["center"],
                            cut_y=frame["cut_y"], chin=frame["chin"])
    return stats

"""Geometry prior from the SAM 3D Body mesh (the MHR body fitted to the photo).

The mesh's face lines up closely with the photo's face (eyes, nose, mouth), but hair and clothes are
not in it, so only its face region is used:
  * face depth targets and masks per training view (the face Gaussians stay on a face-shaped surface
    instead of spreading along the photo's rays, the cause of double eyes at new angles)
  * a soft face mask per view (generated views get a lower colour weight there: the photo owns the face)
  * starting points on the face surface, coloured from the photo (the visual hull has no face relief)
  * FaceShell: a thin shell the face Gaussians are projected back onto every step
  * bake_face: after training, the photo unprojected onto the face (one Gaussian per photo pixel)
  * BodyInterior: clearing Gaussians inside the whole body (measured: it hurt; off by default)
The mesh is drawn with gsplat itself as ~1M tiny opaque Gaussians, so no mesh rasteriser is needed.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .body import L_EAR, L_EYE, NOSE, R_EAR, R_EYE

SH_C0 = 0.28209479177387814


def face_frame(world: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(face centre, point between the ears, unit forward direction of the head)."""
    kp = world["kp3d"]
    ear_mid = (kp[L_EAR] + kp[R_EAR]) / 2
    fwd = kp[NOSE] - ear_mid
    return (kp[NOSE] + kp[L_EYE] + kp[R_EYE]) / 3, ear_mid, fwd / np.linalg.norm(fwd)


def in_face(points: np.ndarray, world: dict, front_m: float = 0.04, radius_m: float = 0.09) -> np.ndarray:
    """Points in front of the ears and near the face centre: forehead, eyes, nose, cheeks, mouth, chin."""
    center, ear_mid, fwd = face_frame(world)
    return ((points - ear_mid) @ fwd > front_m) & (np.linalg.norm(points - center, axis=1) < radius_m)


def sample_surface(world: dict, n: int, seed: int = 0) -> np.ndarray:
    """Area-uniform random points on the mesh."""
    rng = np.random.default_rng(seed)
    tri = world["vertices"][world["faces"]]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    idx = rng.choice(len(tri), n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = tri[idx]
    return t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])


class MeshPrior:
    def __init__(self, world: dict, device, n: int = 1_000_000, splat_m: float = 0.0015,
                 core_radius_m: float = 0.09):
        pts = sample_surface(world, n)
        self.world = world
        self.face = in_face(pts, world)
        self.means = torch.from_numpy(pts).float().to(device)
        self.colors = torch.zeros(n, 3, device=device)
        self.colors[:, 0] = torch.from_numpy(self.face).float().to(device)
        # the "core": the part of the face the photo owns (eyes, brows, nose, mouth, inner cheeks)
        self.colors[:, 1] = torch.from_numpy(in_face(pts, world, radius_m=core_radius_m)).float().to(device)
        self.quats = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(n, 1)
        self.scales = torch.full((n, 3), splat_m, device=device)
        self.opacities = torch.full((n,), 0.99, device=device)

    @torch.no_grad()
    def render(self, w2c: torch.Tensor, K: torch.Tensor, width: int, height: int, core: bool = False):
        """(face coverage HxW in 0..1, mesh depth HxW, mesh alpha HxW); with core, also the core coverage."""
        from gsplat import rasterization
        out, alpha, _ = rasterization(self.means, self.quats, self.scales, self.opacities, self.colors,
                                      viewmats=w2c[None].float(), Ks=K[None].float(), width=width,
                                      height=height, render_mode="RGB+ED", packed=False,
                                      near_plane=0.01, far_plane=100.0)
        result = out[0, ..., 0], out[0, ..., 3], alpha[0, ..., 0]
        return result + (out[0, ..., 1],) if core else result

    @torch.no_grad()
    def face_targets(self, w2c, K, width: int, height: int, margin_m: float = 0.015):
        """None if the face is not visible, else (depth HxWx1, depth mask HxWx1, soft core region HxWx1)."""
        face, depth, alpha, core_cov = self.render(w2c, K, width, height, core=True)
        core = (face > 0.95) & (alpha > 0.99)
        if core.sum() < 1000:         # face barely visible (a sliver of cheek from behind): no targets
            return None
        # stay 2 px inside the face so depth edges (face/neck, face/hair) are not targets
        mask = -F.max_pool2d(-core.float()[None, None], 5, 1, 2)[0, 0]
        # soft region: the face plus a margin, feathered
        z = float(depth[core].median())
        r = max(1, round(float(K[0, 0]) * margin_m / z))
        region = F.max_pool2d((core_cov > 0.5).float()[None, None], 2 * r + 1, 1, r)
        region = F.avg_pool2d(region, 2 * r + 1, 1, r, count_include_pad=False)[0, 0]
        return depth[..., None], mask[..., None], region[..., None]

    @torch.no_grad()
    def face_points(self, photo_rgb: torch.Tensor, photo_w2c: torch.Tensor, photo_K: torch.Tensor,
                    spacing_m: float = 0.003) -> tuple[torch.Tensor, torch.Tensor]:
        """Face-surface starting points that the photo sees, coloured from it (N,3), (N,3)."""
        h, w = photo_rgb.shape[:2]
        _, depth, alpha = self.render(photo_w2c, photo_K, w, h)
        pts = self.means[torch.from_numpy(self.face).to(self.means.device)]
        area = 0.012 / spacing_m ** 2                          # ~120 cm^2 of face
        keep = torch.randperm(len(pts), device=pts.device)[:int(area)]
        pts = pts[keep]
        cam = pts @ photo_w2c[:3, :3].T + photo_w2c[:3, 3]
        u = (photo_K[0, 0] * cam[:, 0] / cam[:, 2] + photo_K[0, 2]).long().clamp(0, w - 1)
        v = (photo_K[1, 1] * cam[:, 1] / cam[:, 2] + photo_K[1, 2]).long().clamp(0, h - 1)
        visible = ((cam[:, 2] - depth[v, u]).abs() < 0.005) & (alpha[v, u] > 0.99)
        return pts[visible], photo_rgb[v[visible], u[visible]]


class FaceShell:
    """A thin shell around the mesh's face: Gaussians near the face surface are projected back to within
    `tau` of it after every step. The expected-depth loss alone lets the face grow into a thick cloud of
    layers that averages to the right depth but shatters when seen from a new angle.

    The signed distance is precomputed on a 1.5 mm grid from dense samples of the head (so a point near
    the edge of the face is measured against the true nearest surface) and only points whose nearest
    surface is face (not scalp, ears or neck) are projected."""

    def __init__(self, world: dict, device, voxel_m: float = 0.0015, band_m: float = 0.025, n: int = 600_000):
        from scipy.spatial import cKDTree
        verts, faces = world["vertices"], world["faces"]
        center, _, fwd = face_frame(world)
        head_tris = faces[(np.linalg.norm(verts[faces].mean(1) - center, axis=1) < 0.2)]
        tri = verts[head_tris]
        normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        area = 0.5 * np.linalg.norm(normal, axis=1)
        normal /= np.maximum(2 * area[:, None], 1e-12)
        rng = np.random.default_rng(1)
        idx = rng.choice(len(tri), n, p=area / area.sum())
        u, v = rng.random(n), rng.random(n)
        flip = u + v > 1
        u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
        t = tri[idx]
        pts = t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])
        nrm = normal[idx]
        is_face = in_face(pts, world)
        if (nrm[is_face] @ fwd).mean() < 0:        # make normals point out of the head
            nrm = -nrm
        lo = pts[is_face].min(0) - band_m
        hi = pts[is_face].max(0) + band_m
        axes = [np.arange(lo[i], hi[i] + voxel_m, voxel_m) for i in range(3)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        dist, nearest = cKDTree(pts).query(grid, distance_upper_bound=band_m, workers=-1)
        found = np.isfinite(dist)
        field = np.zeros((len(grid), 5), np.float32)       # sdf, normal xyz, nearest-is-face
        near = nearest[found]
        field[found, 0] = ((grid[found] - pts[near]) * nrm[near]).sum(1)
        field[found, 1:4] = nrm[near]
        field[found, 4] = is_face[near]
        shape = tuple(len(a) for a in axes)
        # grid_sample wants (N, C, D, H, W) with coordinates ordered (x -> W, y -> H, z -> D)
        vol = torch.from_numpy(field.reshape(*shape, 5)).permute(3, 2, 1, 0)[None].contiguous()
        self.field = vol.to(device)
        self.lo = torch.tensor(lo, device=device).float()
        self.hi = torch.tensor([a[-1] for a in axes], device=device).float()   # last voxel centres

    def sample(self, points: torch.Tensor):
        """(inside grid mask, sdf, normal, face flag) for points (N, 3)."""
        inside = ((points > self.lo) & (points < self.hi)).all(1)
        p = points[inside]
        coords = (2 * (p - self.lo) / (self.hi - self.lo) - 1)[None, :, None, None]   # 1,N,1,1,3
        vals = F.grid_sample(self.field, coords, align_corners=True)[0, :, :, 0, 0].T  # N,5
        return inside, vals[:, 0], F.normalize(vals[:, 1:4], dim=1), vals[:, 4]

    @torch.no_grad()
    def project(self, params, tau_m: float = 0.002, capture_m: float = 0.02) -> int:
        means = params["means"]
        inside, sdf, normal, face = self.sample(means)
        sel = (face > 0.5) & (sdf.abs() < capture_m) & (sdf.abs() > tau_m)
        excess = sdf - sdf.clamp(-tau_m, tau_m)
        idx = torch.nonzero(inside, as_tuple=True)[0][sel]
        means.data[idx] -= excess[sel, None] * normal[sel]
        return int(sel.sum())


class BodyInterior:
    """Signed distance to the whole body mesh on a coarse grid, to find Gaussians *inside* the body.

    Nothing real is inside the body, but the optimiser parks Gaussians there: while the surface in front
    is partly transparent they tint the photo's colours, and from any other angle they show through as
    dark or colourful smudges (the "ghost" face). Clothes and hair are outside the mesh, so a margin
    below the surface is safe to clear: `margin_m` for the body (the fit can be a little too fat), a
    tighter `face_margin_m` in front of the face where the mesh matches the photo closely."""

    def __init__(self, world: dict, device, voxel_m: float = 0.006, n: int = 1_500_000):
        from scipy.spatial import cKDTree
        verts, faces = world["vertices"], world["faces"]
        tri = verts[faces]
        normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        area = 0.5 * np.linalg.norm(normal, axis=1)
        normal /= np.maximum(2 * area[:, None], 1e-12)
        rng = np.random.default_rng(2)
        idx = rng.choice(len(tri), n, p=area / area.sum())
        u, v = rng.random(n), rng.random(n)
        flip = u + v > 1
        u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
        t = tri[idx]
        pts = t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])
        nrm = normal[idx]
        center = verts.mean(0)
        if ((pts - center) * nrm).sum(1).mean() < 0:     # make normals point outwards
            nrm = -nrm
        lo, hi = verts.min(0) - 0.02, verts.max(0) + 0.02
        axes = [np.arange(lo[i], hi[i] + voxel_m, voxel_m) for i in range(3)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        # every point inside the body is within ~15 cm of its surface; farther ones are outside
        dist, nearest = cKDTree(pts).query(grid, distance_upper_bound=0.15, workers=-1)
        found = np.isfinite(dist)
        sdf = np.full(len(grid), 0.15, np.float32)
        face = np.zeros(len(grid), np.float32)
        near = nearest[found]
        sdf[found] = ((grid[found] - pts[near]) * nrm[near]).sum(1)
        face[found] = in_face(pts[near], world, front_m=0.03, radius_m=0.10)
        shape = tuple(len(a) for a in axes)
        vol = np.stack([sdf, face], -1).reshape(*shape, 2)
        self.field = torch.from_numpy(vol).permute(3, 2, 1, 0)[None].contiguous().to(device)
        self.lo = torch.tensor(lo, device=device).float()
        self.hi = torch.tensor([a[-1] for a in axes], device=device).float()

    def inside(self, points: torch.Tensor, margin_m: float = 0.03, face_margin_m: float = 0.005) -> torch.Tensor:
        """Mask of points deeper than the margin below the body surface."""
        within = ((points > self.lo) & (points < self.hi)).all(1)
        res = torch.zeros(len(points), dtype=torch.bool, device=points.device)
        p = points[within]
        coords = (2 * (p - self.lo) / (self.hi - self.lo) - 1)[None, :, None, None]
        vals = F.grid_sample(self.field, coords, align_corners=True)[0, :, :, 0, 0].T
        sdf, face = vals[:, 0], vals[:, 1]
        res[within] = sdf < torch.where(face > 0.5, -face_margin_m, -margin_m)
        return res


def _rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """(N, 3, 3) rotation matrices to (N, 4) wxyz quaternions."""
    w = torch.sqrt(torch.clamp(1 + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2], min=1e-8)) / 2
    x = torch.sqrt(torch.clamp(1 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2], min=0)) / 2
    y = torch.sqrt(torch.clamp(1 - R[:, 0, 0] + R[:, 1, 1] - R[:, 2, 2], min=0)) / 2
    z = torch.sqrt(torch.clamp(1 - R[:, 0, 0] - R[:, 1, 1] + R[:, 2, 2], min=0)) / 2
    x = torch.copysign(x, R[:, 2, 1] - R[:, 1, 2])
    y = torch.copysign(y, R[:, 0, 2] - R[:, 2, 0])
    z = torch.copysign(z, R[:, 1, 0] - R[:, 0, 1])
    return F.normalize(torch.stack([w, x, y, z], 1), dim=1)


def _photo_hair_mask(photo_rgba: torch.Tensor, u: torch.Tensor, v: torch.Tensor, dark: float = 0.55) -> torch.Tensor:
    """Hair over the face in the photo, as an (H, W) bool mask. `u, v` are the pixels of the face patch.
    Dark regions connected to the patch's outside are hair; dark regions enclosed by skin (eyes, brows,
    nostrils, lips) are features and stay."""
    import cv2
    h, w = photo_rgba.shape[:2]
    zone = np.zeros((h, w), np.uint8)
    zone[v.cpu().numpy(), u.cpu().numpy()] = 1
    zone = cv2.morphologyEx(zone, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    rgb = photo_rgba[..., :3].cpu().numpy()
    lum = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    inner = lum[zone > 0]
    skin = float(np.median(inner[inner > np.median(inner)]))          # the brighter half is skin
    ring = cv2.dilate(zone, np.ones((31, 31), np.uint8))
    darkm = ((lum < dark * skin) & (ring > 0)).astype(np.uint8)
    n, labels = cv2.connectedComponents(darkm, connectivity=8)
    outside = (ring > 0) & (zone == 0)
    hair_labels = np.unique(labels[outside & (darkm > 0)])
    mask = np.isin(labels, hair_labels[hair_labels > 0])
    mask = cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    return torch.from_numpy(mask).to(photo_rgba.device)


@torch.no_grad()
def bake_face(params: dict, world: dict, photo_rgba: torch.Tensor, photo_w2c: torch.Tensor, photo_K: torch.Tensor,
              keep_outside_m: float = 0.008, radius_m: float = 0.085, max_tilt_cos: float = 0.2,
              feather_m: float = 0.008, width: float = 0.55, thickness: float = 0.2,
              remove_under: bool = True) -> tuple[dict, dict]:
    """Replace the splat's face with the photo itself, unprojected onto the body mesh's face: one flat
    Gaussian per photo pixel, at the mesh's depth for that pixel, as wide as the pixel. From the photo's
    camera this reproduces the photo; from anywhere else it is one coherent textured surface, which a
    trained face (fitted to disagreeing generated faces) is not.

    Hair falling over the edge of the face stays 3D (the splat's own hair); surfaces steeper than
    `max_tilt_cos` to the camera are left out (their texture would smear). The splat's own Gaussians over
    the patch, from 2 cm under the skin to `keep_outside_m` above it, are removed. Over the last
    `feather_m` of the patch the baked layer fades out and the splat's own Gaussians stay underneath, so
    the edge blends instead of showing a seam. Returns (params, stats)."""
    from scipy.spatial import cKDTree
    device = params["means"].device
    h, w = photo_rgba.shape[:2]
    center, _, _ = face_frame(world)
    prior = MeshPrior(world, device, core_radius_m=radius_m)
    _, depth, mesh_alpha, core = prior.render(photo_w2c, photo_K, w, h, core=True)

    # pixel -> point on the mesh (gsplat puts pixel centres at +0.5)
    vs, us = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    x = (us + 0.5 - photo_K[0, 2]) / photo_K[0, 0]
    y = (vs + 0.5 - photo_K[1, 2]) / photo_K[1, 1]
    cam_pts = torch.stack([x * depth, y * depth, depth], -1)                       # H, W, 3 (camera frame)
    # normals from the point map (camera frame); they face the camera, so z < 0
    dx = F.pad(cam_pts[:, 2:] - cam_pts[:, :-2], (0, 0, 1, 1))
    dy = F.pad(cam_pts[2:] - cam_pts[:-2], (0, 0, 0, 0, 1, 1))
    nrm = F.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    nrm = torch.where(nrm[..., 2:3] > 0, -nrm, nrm)
    view = F.normalize(-cam_pts, dim=-1)
    tilt = (nrm * view).sum(-1)                                                    # cos(normal, view ray)

    ok = (core > 0.95) & (mesh_alpha > 0.99) & (photo_rgba[..., 3] > 0.5) & (tilt > max_tilt_cos)
    # hair over the face's border stays 3D; the inner face (eyes, brows, nose, mouth) never counts as hair
    inner_prior = MeshPrior(world, device, core_radius_m=0.06)
    _, _, _, inner = inner_prior.render(photo_w2c, photo_K, w, h, core=True)
    vi, ui = torch.nonzero(ok, as_tuple=True)
    hair = _photo_hair_mask(photo_rgba, ui, vi)
    ok &= ~(hair & (inner < 0.5))
    vi, ui = torch.nonzero(ok, as_tuple=True)
    # distance to the patch edge (the radius limit, hair holes, steep parts) in pixels -> fade weight
    import cv2
    edge_px = cv2.distanceTransform(ok.cpu().numpy().astype(np.uint8), cv2.DIST_L2, 5)
    feather_px = max(1.0, float(photo_K[0, 0]) * feather_m / float(depth[vi, ui].median()))
    fade = torch.from_numpy(np.clip(edge_px / feather_px, 0.0, 1.0)).to(device)[vi, ui]

    R, t = photo_w2c[:3, :3], photo_w2c[:3, 3]
    pts = (cam_pts[vi, ui] - t) @ R                                                # camera -> world
    n_world = nrm[vi, ui] @ R
    rgb = photo_rgba[vi, ui, :3]
    footprint = depth[vi, ui] / photo_K[0, 0] / tilt[vi, ui].clamp_min(max_tilt_cos)   # metres per pixel

    # remove the splat's own Gaussians over the baked patch
    tree = cKDTree(pts.cpu().numpy())
    means = params["means"]
    near = torch.nonzero(((means - torch.from_numpy(center).float().to(device)).norm(dim=1) < 0.16),
                         as_tuple=True)[0]
    dist, nearest = tree.query(means[near].cpu().numpy(), distance_upper_bound=0.025)
    found = torch.from_numpy(np.isfinite(dist)).to(device)
    near, nearest = near[found], torch.from_numpy(nearest[found.cpu().numpy()]).to(device)
    offset = means[near] - pts[nearest]
    along = (offset * n_world[nearest]).sum(1)
    sideways = (offset - along[:, None] * n_world[nearest]).norm(dim=1)
    hit = (along > -0.02) & (along < keep_outside_m) & (sideways < 0.0015) & (fade[nearest] > 0.7)
    drop = torch.zeros(len(means), dtype=torch.bool, device=device)
    if remove_under:
        # off by default when the trained face is trustworthy (head close-up frames): keeping it means
        # the gaps between foreshortened baked discs show real face instead of a hole
        drop[near[hit]] = True
    keep = ~drop

    # flat Gaussians in the skin, about a pixel wide
    helper = torch.where((n_world[:, 1:2].abs() < 0.9), torch.tensor([0.0, 1.0, 0.0], device=device),
                         torch.tensor([1.0, 0.0, 0.0], device=device))
    t1 = F.normalize(torch.cross(helper, n_world, dim=1), dim=1)
    t2 = torch.cross(n_world, t1, dim=1)
    quats = _rotmat_to_quat(torch.stack([t1, t2, n_world], 2))
    # wider and less flat discs survive being seen from the side: a disc of thickness 0.2 seen edge-on
    # projects to almost nothing, and the gaps between the discs are what reads as speckle off-angle
    sigma = width * footprint
    scales = torch.log(torch.stack([sigma, sigma, sigma * thickness], 1))
    k = params["shN"].shape[1]
    baked = {"means": pts, "quats": quats, "scales": scales,
             "opacities": torch.logit(0.99 * (0.05 + 0.95 * fade)),
             "sh0": ((rgb - 0.5) / SH_C0)[:, None], "shN": torch.zeros(len(pts), k, 3, device=device)}
    out = {key: torch.cat([params[key][keep], baked[key]]) for key in baked}
    return out, {"baked": len(pts), "removed": int(drop.sum())}


@torch.no_grad()
def recolour_face_from_photo(params: dict, world: dict, photo_rgba: torch.Tensor, photo_w2c: torch.Tensor,
                             photo_K: torch.Tensor, radius_m: float = 0.10, strength: float = 0.85,
                             tau_m: float = 0.008, feather_m: float = 0.02) -> tuple[dict, dict]:
    """Give the face the photo's colours without giving it new geometry: every face Gaussian the photo
    can see is tinted towards the photo pixel it lands on. Nothing is added or removed, so there is
    nothing to foreshorten into gaps, and the detail travels with the surface: a cheek lit by the photo
    keeps the photo's texture when seen from the side.

    This replaces `bake_face` for splats whose face was trained on head close-ups. The bake lays one
    flat disc per photo pixel over the face, which reproduces the photo exactly head-on but breaks into
    speckle at an angle, because the discs foreshorten into gaps and it deletes the face underneath.

    Gaussians the photo cannot see (behind the nose, under the chin) keep their trained colour, and the
    blend fades out over `feather_m` at the edge of the patch, so there is no seam."""
    from .splat import render
    device = params["means"].device
    h, w = photo_rgba.shape[:2]
    center, _, _ = face_frame(world)
    centre_t = torch.as_tensor(center, device=device).float()
    means = params["means"]
    dist = (means - centre_t).norm(dim=1)
    near = torch.nonzero(dist < radius_m, as_tuple=True)[0]
    if len(near) == 0:
        return params, {"recoloured": 0}

    sh_degree = int(round(math.sqrt(params["shN"].shape[1] + 1))) - 1
    _, alpha, info = render(params, photo_w2c, photo_K, w, h, sh_degree, with_depth=True)
    depth, alpha = info["depth"][..., 0], alpha[..., 0]

    cam = means[near] @ photo_w2c[:3, :3].T + photo_w2c[:3, 3]
    u = photo_K[0, 0] * cam[:, 0] / cam[:, 2] + photo_K[0, 2]
    v = photo_K[1, 1] * cam[:, 1] / cam[:, 2] + photo_K[1, 2]
    ui, vi = u.round().long().clamp(0, w - 1), v.round().long().clamp(0, h - 1)
    visible = ((cam[:, 2] > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
               & ((cam[:, 2] - depth[vi, ui]).abs() < tau_m) & (alpha[vi, ui] > 0.5)
               & (photo_rgba[vi, ui, 3] > 0.5))
    fade = ((radius_m - dist[near]) / feather_m).clamp(0.0, 1.0)
    weight = (strength * fade * visible.float())[:, None]
    rgb = photo_rgba[vi, ui, :3]
    sh0 = params["sh0"].clone()
    target = (rgb - 0.5) / SH_C0
    sh0[near, 0] = (1 - weight) * sh0[near, 0] + weight * target
    out = dict(params)
    out["sh0"] = sh0
    return out, {"recoloured": int((weight[:, 0] > 0.01).sum()), "considered": len(near)}


@torch.no_grad()
def face_push(world: dict, records: list[dict], device, max_push_m: float = 0.05, step_m: float = 0.0015,
              inside_share: float = 0.98) -> float:
    """How far (metres) the body mesh's face must move back, along the photo camera's rays, to fit
    inside the silhouettes of the generated views.

    SAM 3D Body reads the head's depth from one frontal photo; on the test image it placed the face
    ~2.5 cm in front of where the side views put it, so a face pinned to the mesh floated in front of
    the hair like a mask. The side views' mattes carve a tight visual hull around the head (no matte
    dilation, 2 mm voxels, one view may disagree); the answer is the smallest push that puts
    `inside_share` of the mesh's face inside it. Moving along camera rays leaves the photo view unchanged."""
    from PIL import Image

    from .cameras import camera_center
    center, _, _ = face_frame(world)
    K = torch.as_tensor(world["K"], device=device).float()
    axes = [torch.arange(center[i] - 0.16, center[i] + 0.16, 0.002, device=device) for i in range(3)]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3).float()
    votes = torch.zeros(len(grid), device=device)
    seen = torch.zeros(len(grid), device=device)
    for r in records:
        if r["tag"] == "photo" or not r.get("selected"):
            continue
        matte = torch.from_numpy(np.asarray(Image.open(r["mask"]).convert("L")) > 127).to(device)
        w2c = torch.as_tensor(np.asarray(r["w2c"]), device=device).float()
        cam = grid @ w2c[:3, :3].T + w2c[:3, 3]
        u = (K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2]).round().long()
        v = (K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]).round().long()
        h, w = matte.shape
        ok = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (cam[:, 2] > 0)
        seen += ok.float()
        hit = torch.zeros_like(ok)
        hit[ok] = matte[v[ok], u[ok]]
        votes += hit.float()
    inside = (votes >= seen - 1) & (seen >= 2)
    shape = torch.tensor([len(a) for a in axes], device=device)
    occ = inside.reshape(*shape.tolist())
    lo = torch.stack([a[0] for a in axes])

    verts = world["vertices"][in_face(world["vertices"], world)]
    c = camera_center(world["photo_w2c"])
    ray = (verts - c) / np.linalg.norm(verts - c, axis=1, keepdims=True)
    for push in np.arange(0.0, max_push_m + 1e-9, step_m):
        p = torch.as_tensor(verts + ray * push, device=device).float()
        idx = torch.floor((p - lo) / 0.002 + 0.5).long()
        ok = ((idx >= 0) & (idx < shape)).all(1)
        share = torch.zeros(len(p), dtype=torch.bool, device=device)
        share[ok] = occ[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
        if float(share.float().mean()) >= inside_share:
            return float(push)
    return float(max_push_m)


def push_head(world: dict, push_m: float) -> dict:
    """A copy of `world` whose head (mesh and head keypoints) is moved `push_m` back along the photo
    camera's rays: fully within 14 cm of the face centre, fading out by 20 cm (the neck)."""
    from .cameras import camera_center
    if push_m <= 0:
        return world
    center, _, _ = face_frame(world)
    c = camera_center(world["photo_w2c"])

    def moved(p: np.ndarray) -> np.ndarray:
        weight = np.clip((0.20 - np.linalg.norm(p - center, axis=1)) / 0.06, 0.0, 1.0)
        ray = (p - c) / np.linalg.norm(p - c, axis=1, keepdims=True)
        return p + (push_m * weight)[:, None] * ray

    out = dict(world)
    out["vertices"] = moved(world["vertices"])
    out["kp3d"] = moved(world["kp3d"])
    return out


def align_face(world: dict, records: list[dict], device) -> tuple[dict, float]:
    """(world with the head moved to fit the generated views, the push in metres)."""
    push = face_push(world, records, device)
    return push_head(world, push), push


def drop_face_hull_points(points: torch.Tensor, colors: torch.Tensor, world: dict):
    """Remove visual-hull points in front of the face: the hull has no face relief, and points there take
    the photo's face colours at the wrong depth."""
    keep = ~torch.from_numpy(in_face(points.cpu().numpy(), world, front_m=0.03, radius_m=0.11)).to(points.device)
    return points[keep], colors[keep]

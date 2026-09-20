"""Camera rigs for the control video. Frame 0 is always the photo's camera; the camera circles the
vertical axis through the person (x = z = 0), keeping the photo's distance, height and pitch.
Orbit angle 90 is the person's left (+X), as in i2g.cameras. OpenCV world-to-camera matrices.

Rigs:
  360, 2x180, 4x90  the plain body orbit, in one clip or in chained arcs
  helix             the same orbit, rising to `elev` degrees over the first `elev_ramp` degrees, so the
                    top of the head and the shoulders are seen from above
  face              the same camera positions, but zoomed in on the head (a longer lens, with the
                    principal point tracking the head). Frame 0 stays an exact crop of the photo, and
                    the head gets many more pixels than it does in a full-body frame.
"""
from __future__ import annotations

import math

import numpy as np

from i2g.cameras import azimuth_elevation, project

# arcs as (start, end) orbit angles; arc k > 0 starts where arc k - 1 ended (its first frame repeats
# the previous arc's last one, which is also its start image)
ARCS = {
    "360": [(0.0, 360.0)],
    "2x180": [(0.0, 180.0), (180.0, 360.0)],
    "4x90": [(0.0, 90.0), (90.0, 180.0), (180.0, 270.0), (270.0, 360.0)],
}
RIGS = {
    "360": {"arcs": "360"},
    "2x180": {"arcs": "2x180"},
    "4x90": {"arcs": "4x90"},
    "helix": {"arcs": "360", "elev": 30.0, "elev_ramp": 90.0},
    "face": {"arcs": "360", "zoom": 3.0, "track": "head"},
    # the face seen from above and below: a single ring at the photo's height leaves the face with no
    # close-up coverage from other elevations (identity 0.63 at eye level, 0.30 at +30 degrees). These
    # pivot about the head, so it stays the same distance away, and still start at the photo's camera.
    "face_up": {"arcs": "360", "zoom": 3.0, "track": "head", "elev": 30.0, "elev_ramp": 90.0, "pivot": "head"},
    "face_down": {"arcs": "360", "zoom": 3.0, "track": "head", "elev": -20.0, "elev_ramp": 90.0, "pivot": "head"},
    # steeper still. The photo already looks UP at the head by ~19 degrees, so a rig's elevation reaches
    # (elev - 19) degrees relative to the face: face_up lands at only ~+11, which is why the face grid
    # was still weak at +30. These two reach ~+36 and ~+76 (looking down on the crown).
    "face_high": {"arcs": "360", "zoom": 3.0, "track": "head", "elev": 55.0, "elev_ramp": 90.0, "pivot": "head"},
    "face_top": {"arcs": "360", "zoom": 3.0, "track": "head", "elev": 95.0, "elev_ramp": 90.0, "pivot": "head"},
}
LAYOUTS = ARCS      # backwards-compatible name


def rig_tag(rig: str) -> str:
    """Short prefix for a rig's view names, unique among the rigs above: 360 -> a (what the first runs
    used), face -> fa, face_up -> fu, face_down -> fd, helix -> he. Clips trained together must differ."""
    if rig == "360":
        return "a"
    parts = rig.split("_")
    return "".join(p[0] for p in parts) if len(parts) > 1 else rig[:2]


def video_K(world: dict, width: int, height: int) -> np.ndarray:
    """The photo's vertical field of view at the video size: square pixels, centred principal point."""
    f = float(world["K"][1, 1]) * height / world["size"][1]
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])


def rot_y(deg: float) -> np.ndarray:
    a = math.radians(deg)
    m = np.eye(4)
    m[0, 0], m[0, 2], m[2, 0], m[2, 2] = math.cos(a), math.sin(a), -math.sin(a), math.cos(a)
    return m


def rot_x_about(point: np.ndarray, deg: float) -> np.ndarray:
    """Rotation about the horizontal (world X) axis through `point`, signed so that a positive angle
    *raises* the camera (it ends up looking down at the subject). The world turns the opposite way to
    the camera, hence the negated angle."""
    a = math.radians(-deg)
    m = np.eye(4)
    m[1, 1], m[1, 2], m[2, 1], m[2, 2] = math.cos(a), -math.sin(a), math.sin(a), math.cos(a)
    t = np.eye(4)
    t[:3, 3] = point
    inv = np.eye(4)
    inv[:3, 3] = -point
    return t @ m @ inv


def orbit_w2c(world: dict, angle_deg: float, elev_deg: float = 0.0, pivot: np.ndarray | None = None) -> np.ndarray:
    """The photo's camera moved `angle_deg` around the person and `elev_deg` above its own height.
    The rise pivots about `pivot` (the body centre by default; the head for the face close-up rigs, so
    the camera keeps its distance to the face)."""
    point = world["target"] if pivot is None else pivot
    move = rot_y(angle_deg) @ rot_x_about(point, elev_deg)
    return world["photo_w2c"] @ np.linalg.inv(move)


def head_point(world: dict) -> np.ndarray:
    from i2g.evaluate import _head
    return _head(world)


def _framed_K(world: dict, base_K: np.ndarray, w2c: np.ndarray, zoom: float, track: str | None,
              width: int, height: int) -> np.ndarray:
    """A longer lens, with the principal point moved so `track` stays in the middle of the frame."""
    K = base_K.copy()
    K[0, 0] *= zoom
    K[1, 1] *= zoom
    if track == "head":
        uv, _ = project(head_point(world)[None], w2c, K)
        K[0, 2] += width / 2.0 - uv[0, 0]
        K[1, 2] += height / 2.0 - uv[0, 1]
    return K


def layout_cameras(world: dict, rig: str, frames_per_arc: int, width: int, height: int) -> list[dict]:
    """One entry per generated frame: arc, index in the arc, orbit angle, w2c, K, and the i2g azimuth,
    elevation and radius (the same angles registration.json uses)."""
    spec = RIGS[rig]
    base_K = video_K(world, width, height)
    zoom, track = spec.get("zoom", 1.0), spec.get("track")
    elev, ramp = spec.get("elev", 0.0), spec.get("elev_ramp", 90.0)
    pivot = head_point(world) if spec.get("pivot") == "head" else None
    cams = []
    for arc, (start, end) in enumerate(ARCS[spec["arcs"]]):
        for index, angle in enumerate(np.linspace(start, end, frames_per_arc)):
            phi = elev * min(1.0, float(angle) / ramp) if elev else 0.0
            w2c = orbit_w2c(world, float(angle), phi, pivot)
            K = _framed_K(world, base_K, w2c, zoom, track, width, height)
            az, el, radius = azimuth_elevation(w2c, world["target"])
            cams.append({"arc": arc, "index": index, "orbit_deg": float(angle), "w2c": w2c, "K": K,
                         "azimuth_deg": az, "elevation_deg": el, "radius_m": radius,
                         "rig_elev_deg": phi, "zoom": zoom})
    return cams


def is_duplicate(cam: dict) -> bool:
    """Frames whose camera another view already has: the orbit start (the photo itself), the first
    frame of every later arc (the previous arc's last frame) and the full-circle end (orbit 360)."""
    return cam["index"] == 0 or math.isclose(cam["orbit_deg"] % 360.0, 0.0, abs_tol=1e-6)

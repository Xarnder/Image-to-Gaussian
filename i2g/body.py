"""S1: fit SAM 3D Body (through ComfyUI) and turn it into the body scaffold.

The scaffold fixes the world frame (Y up, person facing +Z, feet at y=0, metres) and the photo
camera. Every generated view is later registered against it (register.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .cameras import intrinsics_from_focal, project, render_silhouette
from .comfy_client import ComfyClient

SAM3D_CHECKPOINT = "sam_3d_body_dinov3_bf16.safetensors"

# MHR70 keypoint indices (see ComfyUI comfy_extras/sam3d_body/export/glb_shared.py)
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW = 5, 6, 7, 8
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 9, 10, 11, 12, 13, 14
R_WRIST, L_WRIST, NECK = 41, 62, 69
BODY_KP = [NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW,
           L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE, R_WRIST, L_WRIST, NECK]
SKELETON = [(NECK, R_SHOULDER), (NECK, L_SHOULDER), (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
            (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST), (NECK, R_HIP), (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
            (NECK, L_HIP), (L_HIP, L_KNEE), (L_KNEE, L_ANKLE), (NECK, NOSE), (NOSE, R_EYE),
            (R_EYE, R_EAR), (NOSE, L_EYE), (L_EYE, L_EAR)]
# warm = person's right, cool = person's left (BGR for OpenCV drawing)
EDGE_COLORS = [(0, 0, 255), (255, 255, 0), (0, 85, 255), (0, 170, 255), (255, 170, 0), (255, 85, 0),
               (0, 255, 180), (0, 255, 0), (85, 255, 0), (255, 0, 0), (255, 0, 85), (255, 0, 170),
               (150, 150, 150), (170, 0, 255), (255, 0, 50), (170, 0, 255), (255, 0, 50)]


def run_sam3d(client: ComfyClient, image_path: Path, prefix: str, vfov_deg: float, dest: Path) -> Path:
    """Run SAM 3D Body on one image in ComfyUI and download the exported .npz to `dest`."""
    name = client.upload_image(image_path)
    workflow = {
        "1": {"class_type": "LoadImage", "inputs": {"image": name}},
        "2": {"class_type": "SAM3DBody_Loader", "inputs": {"model_file": SAM3D_CHECKPOINT}},
        "3": {"class_type": "SAM3DBody_Predict", "inputs": {"sam3d_body_model": ["2", 0], "image": ["1", 0],
                                                            "run_hand_refinement": True, "fov": vfov_deg,
                                                            "batch_size": 8}},
        "4": {"class_type": "I2G_SaveMHRPose", "inputs": {"mhr_pose_data": ["3", 0], "filename_prefix": prefix}},
    }
    entry = client.run(workflow, timeout=900)
    npz = [f for f in client.output_files(entry) if f.filename.endswith(".npz")]
    if not npz:
        raise RuntimeError(f"SAM 3D Body produced no output for {image_path.name}")
    return client.download(npz[0], dest)


def load_person(npz_path: Path, person: int = 0) -> dict:
    """Vertices / keypoints in the image's camera frame (OpenCV axes, metres) plus focal and faces."""
    data = np.load(npz_path)
    pre = f"frame000.person{person}."
    missing = [k for k in ("pred_vertices", "pred_cam_t", "focal_length", "pred_keypoints_3d")
               if pre + k not in data.files]
    if missing:
        raise KeyError(f"{npz_path.name} lacks {missing}; has {sorted(data.files)[:60]}")
    cam_t = data[pre + "pred_cam_t"].reshape(3).astype(np.float64)
    out = {
        "vertices": data[pre + "pred_vertices"].astype(np.float64) + cam_t,
        "kp3d": data[pre + "pred_keypoints_3d"].reshape(-1, 3).astype(np.float64) + cam_t,
        "focal": float(np.asarray(data[pre + "focal_length"]).reshape(-1)[0]),
        "faces": data["faces"].astype(np.int64),
        "image_size": tuple(int(v) for v in data["image_size"]),
    }
    if pre + "pred_keypoints_2d" in data.files:
        out["kp2d"] = data[pre + "pred_keypoints_2d"].reshape(-1, 2).astype(np.float64)
    return out


def build_world(person: dict) -> dict:
    """World frame from the photo fit: Y up (ankles -> shoulders), Z toward the camera, origin on the floor."""
    kp = person["kp3d"]
    up = kp[[L_SHOULDER, R_SHOULDER]].mean(0) - kp[[L_ANKLE, R_ANKLE]].mean(0)
    up /= np.linalg.norm(up)
    left = (kp[L_HIP] - kp[R_HIP]) + (kp[L_SHOULDER] - kp[R_SHOULDER])
    facing = np.cross(left, up)
    z = facing - facing.dot(up) * up
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    M = np.column_stack([x, up, z])            # camera = M @ world + origin
    verts = person["vertices"]
    p0 = verts.mean(0)
    origin = p0 + (float((verts @ up).min()) - p0.dot(up)) * up

    w2c = np.eye(4)
    w2c[:3, :3] = M
    w2c[:3, 3] = origin
    to_world = lambda p: (p - origin) @ M  # noqa: E731
    verts_w, kp_w = to_world(verts), to_world(kp)
    target = np.array([0.0, (verts_w[:, 1].max() + verts_w[:, 1].min()) / 2, 0.0])
    h, w = person["image_size"]
    return {
        "vertices": verts_w, "faces": person["faces"], "kp3d": kp_w,
        "photo_w2c": w2c, "K": intrinsics_from_focal(person["focal"], w, h), "size": (w, h),
        "target": target, "height_m": float(verts_w[:, 1].max()),
    }


def draw_skeleton(img: np.ndarray, uv: np.ndarray, thickness: int = 3) -> np.ndarray:
    out = img.copy()
    for (a, b), color in zip(SKELETON, EDGE_COLORS):
        pa, pb = tuple(int(v) for v in uv[a]), tuple(int(v) for v in uv[b])
        cv2.line(out, pa, pb, color, thickness, cv2.LINE_AA)
    for i in BODY_KP:
        cv2.circle(out, tuple(int(v) for v in uv[i]), thickness + 1, (255, 255, 255), -1, cv2.LINE_AA)
    return out


def overlay(canvas_bgr: np.ndarray, world: dict, w2c: np.ndarray) -> np.ndarray:
    w, h = world["size"]
    sil = render_silhouette(world["vertices"], world["faces"], w2c, world["K"], w, h)
    tinted = canvas_bgr.copy()
    tinted[sil > 0] = (0.55 * tinted[sil > 0] + 0.45 * np.array([255, 160, 0])).astype(np.uint8)
    contours, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(tinted, contours, -1, (255, 200, 0), 2, cv2.LINE_AA)
    uv, _ = project(world["kp3d"], w2c, world["K"])
    return draw_skeleton(tinted, uv)


def save_world(world: dict, out_dir: Path) -> None:
    np.savez_compressed(out_dir / "body_world.npz", vertices=world["vertices"], faces=world["faces"],
                        kp3d=world["kp3d"], photo_w2c=world["photo_w2c"], K=world["K"],
                        size=np.array(world["size"]), target=world["target"])


def load_world(out_dir: Path) -> dict:
    d = np.load(out_dir / "body_world.npz")
    return {"vertices": d["vertices"], "faces": d["faces"], "kp3d": d["kp3d"], "photo_w2c": d["photo_w2c"],
            "K": d["K"], "size": tuple(int(v) for v in d["size"]), "target": d["target"],
            "height_m": float(d["vertices"][:, 1].max())}


def silhouette_stats(world: dict, w2c: np.ndarray, alpha: np.ndarray) -> dict:
    w, h = world["size"]
    sil = render_silhouette(world["vertices"], world["faces"], w2c, world["K"], w, h) > 0
    matte = alpha > 0.5
    inter = float((sil & matte).sum())
    return {"coverage": inter / max(float(sil.sum()), 1.0),
            "iou": inter / max(float((sil | matte).sum()), 1.0)}


def summarize(world: dict) -> dict:
    from .cameras import azimuth_elevation
    az, el, r = azimuth_elevation(world["photo_w2c"], world["target"])
    return {"height_m": round(world["height_m"], 3), "photo_azimuth_deg": round(az, 2),
            "photo_elevation_deg": round(el, 2), "camera_distance_m": round(r, 3),
            "focal_px": round(float(world["K"][0, 0]), 1), "vertices": int(len(world["vertices"])),
            "faces": int(len(world["faces"]))}


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

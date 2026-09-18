"""Image-to-Gaussian v2 helper nodes for ComfyUI.

I2G_SaveMHRPose writes SAM 3D Body's MHR_POSE_DATA (per-frame, per-person vertices, keypoints,
camera, plus mesh faces) to an .npz in ComfyUI's output folder, so the v2 pipeline running outside
ComfyUI can build the body scaffold from it. Installed by scripts/v2_link_nodes.ps1 (a junction).
"""
import json
import os

import numpy as np
import torch

import folder_paths


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return (value.float() if value.is_floating_point() else value).numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (bool, int, float, np.number)):
        return np.asarray(value)
    if isinstance(value, (list, tuple)) and value:
        try:
            arr = np.asarray([_to_numpy(v) for v in value])
        except (TypeError, ValueError):
            return None
        return arr if arr.dtype != object else None
    return None


def _flatten(name, value, arrays, skipped):
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(f"{name}.{key}" if name else str(key), item, arrays, skipped)
        return
    arr = _to_numpy(value)
    if arr is None:
        skipped.append(name)
    else:
        arrays[name] = arr


class I2G_SaveMHRPose:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mhr_pose_data": ("MHR_POSE_DATA",),
            "filename_prefix": ("STRING", {"default": "i2g/body"}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("npz_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "image-to-gaussian"

    def save(self, mhr_pose_data, filename_prefix):
        arrays, skipped = {}, []
        frames = mhr_pose_data.get("frames", [])
        for f, people in enumerate(frames):
            for p, person in enumerate(people):
                _flatten(f"frame{f:03d}.person{p}", person, arrays, skipped)
        for key, value in mhr_pose_data.items():
            if key != "frames":
                _flatten(key, value, arrays, skipped)

        out_dir = folder_paths.get_output_directory()
        full_dir, filename, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, out_dir)
        base = f"{filename}_{counter:05}_"
        np.savez_compressed(os.path.join(full_dir, base + ".npz"), **arrays)
        meta = {
            "image_size": list(mhr_pose_data.get("image_size", [])),
            "people_per_frame": [len(people) for people in frames],
            "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()},
            "skipped": skipped,
        }
        with open(os.path.join(full_dir, base + ".json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=1)

        files = [{"filename": base + ext, "subfolder": subfolder, "type": "output"} for ext in (".npz", ".json")]
        return {"ui": {"i2g_files": files}, "result": (os.path.join(full_dir, base + ".npz"),)}


NODE_CLASS_MAPPINGS = {"I2G_SaveMHRPose": I2G_SaveMHRPose}
NODE_DISPLAY_NAME_MAPPINGS = {"I2G_SaveMHRPose": "Save SAM3D Body Pose Data (.npz) [i2g]"}

"""Image-to-Gaussian v2 helper nodes for ComfyUI.

I2G_SaveMHRPose writes SAM 3D Body's MHR_POSE_DATA (per-frame, per-person vertices, keypoints,
camera, plus mesh faces) to an .npz in ComfyUI's output folder, so the v2 pipeline running outside
ComfyUI can build the body scaffold from it. Installed by scripts/v2_link_nodes.ps1 (a junction).

The research code (pacgs: proxy-controlled video) also uses I2G_LoadImages / I2G_SaveImages (image
folders by absolute path) and I2G_Wan22FunControlToVideo (Wan 2.2 Fun-Control with a start image).
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


_IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp")


def _image_files(path):
    if os.path.isdir(path):
        return [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.lower().endswith(_IMAGE_EXT)]
    return [path]


class I2G_LoadImages:
    """One image file, or every image in a folder (sorted by name), as an IMAGE batch. Used by the
    research code (pacgs) to hand ComfyUI control videos and start images by absolute path."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"path": ("STRING", {"default": ""})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load"
    CATEGORY = "image-to-gaussian"

    @classmethod
    def IS_CHANGED(cls, path):
        # re-run when any file changes, even if the path string stays the same
        return ";".join(f"{f}:{os.path.getmtime(f)}:{os.path.getsize(f)}" for f in _image_files(path))

    def load(self, path):
        from PIL import Image
        files = _image_files(path)
        if not files:
            raise ValueError(f"no images in {path}")
        frames = [np.asarray(Image.open(f).convert("RGB"), np.float32) / 255.0 for f in files]
        return (torch.from_numpy(np.stack(frames)),)


class I2G_SaveImages:
    """Write an IMAGE batch as lossless PNGs <prefix>0000.png ... into an absolute directory."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "directory": ("STRING", {"default": ""}),
                             "prefix": ("STRING", {"default": "frame_"})}}

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "image-to-gaussian"

    def save(self, images, directory, prefix):
        from PIL import Image
        os.makedirs(directory, exist_ok=True)
        frames = (images.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(os.path.join(directory, f"{prefix}{i:04d}.png"))
        return {"ui": {"text": [f"saved {len(frames)} frames to {directory}"]}}


class I2G_Wan22FunControlToVideo:
    """ComfyUI's Wan22FunControlToVideo with start_image exposed. The core node's execute() already
    handles it (the Fun-Control patch embedding has start-image channels), but its schema lists only
    ref_image and control_video. start_image pins the first frame; ref_image adds reference latents."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"positive": ("CONDITIONING",), "negative": ("CONDITIONING",), "vae": ("VAE",),
                             "width": ("INT", {"default": 480, "min": 16, "max": 8192, "step": 16}),
                             "height": ("INT", {"default": 832, "min": 16, "max": 8192, "step": 16}),
                             "length": ("INT", {"default": 81, "min": 1, "max": 8192, "step": 4}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 64})},
                "optional": {"start_image": ("IMAGE",), "ref_image": ("IMAGE",), "control_video": ("IMAGE",)}}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"
    CATEGORY = "image-to-gaussian"

    def encode(self, positive, negative, vae, width, height, length, batch_size, start_image=None,
               ref_image=None, control_video=None):
        from comfy_extras.nodes_wan import Wan22FunControlToVideo
        out = Wan22FunControlToVideo.execute(positive, negative, vae, width, height, length, batch_size,
                                             ref_image=ref_image, start_image=start_image,
                                             control_video=control_video)
        return out.args


NODE_CLASS_MAPPINGS = {"I2G_SaveMHRPose": I2G_SaveMHRPose, "I2G_LoadImages": I2G_LoadImages,
                       "I2G_SaveImages": I2G_SaveImages, "I2G_Wan22FunControlToVideo": I2G_Wan22FunControlToVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"I2G_SaveMHRPose": "Save SAM3D Body Pose Data (.npz) [i2g]",
                              "I2G_LoadImages": "Load Images from Path [i2g]",
                              "I2G_SaveImages": "Save Images to Path [i2g]",
                              "I2G_Wan22FunControlToVideo": "Wan22 Fun Control To Video + start image [i2g]"}

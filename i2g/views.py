"""S2: anchor views from Qwen-Image-Edit-2511 + fal Multiple-Angles LoRA (+ Lightning 4-step) in ComfyUI.

Sampler settings follow fal's reference workflow (comfyui-workflow-multiple-angles.json): 4 steps,
cfg 1, euler/simple, AuraFlow shift 3.1, CFGNorm 1, reference latents "index_timestep_zero".
Views are requested by the LoRA's camera tokens; register.py measures where each one actually is.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .comfy_client import ComfyClient
from .log import NullBar, get_logger

log = get_logger("views")

QWEN_UNET = "qwen_image_edit_2511_fp8mixed.safetensors"
QWEN_CLIP = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
QWEN_VAE = "qwen_image_vae.safetensors"
ANGLES_LORA = "qwen-image-edit-2511-multiple-angles-lora.safetensors"
LIGHTNING_LORA = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"


# The LoRA's camera tokens as angles in our world frame (az 90 = the person's left, +X). Measured on the
# test image: "right side view" registers at az ~75-90, i.e. the LoRA orbits the camera to the viewer's right.
AZIMUTH_DEG = {"front view": 0, "front-right quarter view": 45, "right side view": 90,
               "back-right quarter view": 135, "back view": 180, "back-left quarter view": 225,
               "left side view": 270, "front-left quarter view": 315}
ELEVATION_DEG = {"low-angle shot": -30, "eye-level shot": 0, "elevated shot": 30, "high-angle shot": 60}


def expected_angles(prompt: str) -> tuple[float, float]:
    """(azimuth, elevation) in degrees that a `<sks> ...` prompt asks for, relative to eye level."""
    az = next(v for k, v in sorted(AZIMUTH_DEG.items(), key=lambda kv: -len(kv[0])) if k in prompt)
    el = next(v for k, v in ELEVATION_DEG.items() if k in prompt)
    return float(az), float(el)


@dataclass(frozen=True)
class ViewRequest:
    tag: str
    azimuth: str      # LoRA azimuth token
    elevation: str    # LoRA elevation token
    distance: str = "medium shot"

    @property
    def prompt(self) -> str:
        return f"<sks> {self.azimuth} {self.elevation} {self.distance}"


# farthest-first: back, then sides, then diagonals, then the elevated ring
DEFAULT_VIEWS = [
    ViewRequest("b", "back view", "eye-level shot"),
    ViewRequest("r", "right side view", "eye-level shot"),
    ViewRequest("l", "left side view", "eye-level shot"),
    ViewRequest("br", "back-right quarter view", "eye-level shot"),
    ViewRequest("bl", "back-left quarter view", "eye-level shot"),
    ViewRequest("fr", "front-right quarter view", "eye-level shot"),
    ViewRequest("fl", "front-left quarter view", "eye-level shot"),
    ViewRequest("f_hi", "front view", "elevated shot"),
    ViewRequest("b_hi", "back view", "elevated shot"),
    ViewRequest("r_hi", "right side view", "elevated shot"),
    ViewRequest("l_hi", "left side view", "elevated shot"),
]

VIEW_PRESETS = {
    # 7 eye-level angles: fastest usable coverage
    "draft": DEFAULT_VIEWS[:7],
    # + elevated ring (top of head, shoulders)
    "standard": DEFAULT_VIEWS,
    # + low angles (under the chin and raised arms)
    "full": DEFAULT_VIEWS + [ViewRequest("f_lo", "front view", "low-angle shot"),
                             ViewRequest("b_lo", "back view", "low-angle shot"),
                             ViewRequest("r_lo", "right side view", "low-angle shot"),
                             ViewRequest("l_lo", "left side view", "low-angle shot")],
}


def seed_values(count: int) -> list[int]:
    """101, 202, 303, ... so reruns with more seeds reuse the images already generated."""
    return [101 * (i + 1) for i in range(count)]


def planned_pairs(preset: str, seeds: int) -> list[tuple[str, int]]:
    return [(v.tag, s) for v in VIEW_PRESETS[preset] for s in seed_values(seeds)]


def qwen_angle_workflow(image_name: str, prompt: str, seed: int, width: int, height: int, prefix: str,
                        angles_strength: float = 0.9, steps: int = 4) -> dict:
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": QWEN_UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": QWEN_CLIP, "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": QWEN_VAE}},
        "4": {"class_type": "LoraLoaderModelOnly",
              "inputs": {"model": ["1", 0], "lora_name": LIGHTNING_LORA, "strength_model": 1.0}},
        "5": {"class_type": "LoraLoaderModelOnly",
              "inputs": {"model": ["4", 0], "lora_name": ANGLES_LORA, "strength_model": angles_strength}},
        "6": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["5", 0], "shift": 3.1}},
        "7": {"class_type": "CFGNorm", "inputs": {"model": ["6", 0], "strength": 1.0}},
        "8": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "9": {"class_type": "TextEncodeQwenImageEditPlus",
              "inputs": {"clip": ["2", 0], "prompt": prompt, "vae": ["3", 0], "image1": ["8", 0]}},
        "10": {"class_type": "TextEncodeQwenImageEditPlus",
               "inputs": {"clip": ["2", 0], "prompt": "", "vae": ["3", 0], "image1": ["8", 0]}},
        "11": {"class_type": "FluxKontextMultiReferenceLatentMethod",
               "inputs": {"conditioning": ["9", 0], "reference_latents_method": "index_timestep_zero"}},
        "12": {"class_type": "FluxKontextMultiReferenceLatentMethod",
               "inputs": {"conditioning": ["10", 0], "reference_latents_method": "index_timestep_zero"}},
        "13": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "14": {"class_type": "KSampler",
               "inputs": {"model": ["7", 0], "seed": seed, "steps": steps, "cfg": 1.0, "sampler_name": "euler",
                          "scheduler": "simple", "positive": ["11", 0], "negative": ["12", 0],
                          "latent_image": ["13", 0], "denoise": 1.0}},
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["3", 0]}},
        "16": {"class_type": "SaveImage", "inputs": {"images": ["15", 0], "filename_prefix": prefix}},
    }


def generate_views(client: ComfyClient, canvas: Path, out_dir: Path, name: str, width: int, height: int,
                   views=DEFAULT_VIEWS, seeds=(101, 202), bar=None) -> list[dict]:
    """Generate every (view, seed) not already in views.json; resumable after an interruption."""
    bar = bar or NullBar()
    image_name = client.upload_image(canvas)
    manifest_path = out_dir / "views.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    manifest = [m for m in manifest if Path(m["path"]).exists()]  # drop entries whose image was deleted
    done = {(m["tag"], m["seed"]) for m in manifest}
    first_job = True
    for view in views:
        for seed in seeds:
            if (view.tag, seed) in done:
                bar.update(1)
                continue
            dest = out_dir / view.tag / f"s{seed}.png"
            bar.set_postfix_str(f"{view.tag} seed {seed}" +
                                (" (loading Qwen into VRAM, ~2-5 min)" if first_job else ""))
            start = time.perf_counter()
            wf = qwen_angle_workflow(image_name, view.prompt, seed, width, height,
                                     f"i2g/{name}/views/{view.tag}_s{seed}")
            log.debug("queue %s seed %d: %s", view.tag, seed, view.prompt)
            # first job loads ~28 GB of weights from the HDD, so only per-step speed is policed
            entry = client.run(wf, timeout=1800, max_step_seconds=120)
            files = client.output_files(entry)
            if not files:
                raise RuntimeError(f"Qwen produced no image for {view.tag} seed {seed}")
            client.download(files[0], dest)
            record = asdict(view) | {"prompt": view.prompt, "seed": seed, "path": str(dest),
                                     "seconds": round(time.perf_counter() - start, 1)}
            manifest.append(record)
            manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
            log.debug("%s seed %d done in %.1f s -> %s", view.tag, seed, record["seconds"], dest)
            first_job = False
            bar.update(1)
    return manifest

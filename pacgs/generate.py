"""Wan 2.2 Fun-Control (A14B) through ComfyUI's HTTP API: depth control video + start image + reference
image -> an orbit video that follows the control pixel for pixel.

Settings follow ComfyUI's own "video_wan2_2_14B_fun_control" template (two-stage high/low-noise
sampling, euler/simple, shift 8): 4 steps at CFG 1 with the lightx2v Lightning LoRAs, or 20 steps at
CFG 3.5 without them. The start image goes through I2G_Wan22FunControlToVideo (comfy_nodes/i2g_export),
which exposes the start_image input the core node computes but does not list.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from i2g.comfy_client import ComfyClient, ComfyError
from i2g.log import get_logger

log = get_logger("pacgs.generate")

HIGH = "Wan2_2-Fun-Control-A14B-HIGH_fp8_e4m3fn_scaled_KJ_fixed.safetensors"
LOW = "Wan2_2-Fun-Control-A14B-LOW_fp8_e4m3fn_scaled_KJ_fixed.safetensors"
LORA_HIGH = "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"
LORA_LOW = "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"
TEXT_ENCODER = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
VAE = "wan_2.1_vae.safetensors"

# Wan's standard negative prompt minus its "static / motionless / greyish" terms (the subject SHOULD be
# static, and the background IS grey), plus motion of the person
NEGATIVE = ("色调艳丽，过曝，细节模糊不清，字幕，风格，作品，画作，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
            "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，杂乱的背景，三条腿，"
            "背景人很多，倒着走, walking, moving arms, moving hands, turning head, changing pose, hair blowing, "
            "camera shake, zoom, blurry, deformed, cartoon, 3d render")

NODES = ["UNETLoader", "LoraLoaderModelOnly", "ModelSamplingSD3", "CLIPLoader", "CLIPTextEncode", "VAELoader",
         "KSamplerAdvanced", "VAEDecode", "I2G_LoadImages", "I2G_SaveImages", "I2G_Wan22FunControlToVideo"]


@dataclass
class GenSettings:
    prompt: str
    width: int = 480
    height: int = 832
    length: int = 81
    steps: int = 4
    lightning: bool = True
    cfg: float = 1.0
    shift: float = 8.0
    sampler: str = "euler"
    scheduler: str = "simple"
    seed: int = 1001
    use_start_image: bool = True
    use_ref_image: bool = True

    @classmethod
    def preset(cls, name: str, prompt: str, **overrides) -> "GenSettings":
        base = {"lightning": dict(steps=4, lightning=True, cfg=1.0),
                "quality": dict(steps=20, lightning=False, cfg=3.5)}[name]
        return cls(prompt=prompt, **(base | overrides))


def check_nodes(client: ComfyClient) -> None:
    info = client.object_info()
    missing = [n for n in NODES if n not in info]
    if missing:
        raise ComfyError(f"ComfyUI lacks nodes {missing}. The I2G_* nodes come from comfy_nodes/i2g_export "
                         "(junctioned into custom_nodes); restart ComfyUI after updating them.")


def build_graph(s: GenSettings, control_dir: Path, start_image: Path | None, ref_image: Path | None,
                out_dir: Path) -> dict:
    # ComfyUI resolves relative paths against its own folder: hand it absolute ones
    control_dir, out_dir = Path(control_dir).resolve(), Path(out_dir).resolve()
    start_image = Path(start_image).resolve() if start_image is not None else None
    ref_image = Path(ref_image).resolve() if ref_image is not None else None
    half = s.steps // 2
    g: dict = {
        "high": {"class_type": "UNETLoader", "inputs": {"unet_name": HIGH, "weight_dtype": "default"}},
        "low": {"class_type": "UNETLoader", "inputs": {"unet_name": LOW, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT_ENCODER, "type": "wan", "device": "default"}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": s.prompt}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": NEGATIVE}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "control": {"class_type": "I2G_LoadImages", "inputs": {"path": str(control_dir)}},
    }
    high, low = ["high", 0], ["low", 0]
    if s.lightning:
        g["lora_high"] = {"class_type": "LoraLoaderModelOnly",
                          "inputs": {"model": high, "lora_name": LORA_HIGH, "strength_model": 1.0}}
        g["lora_low"] = {"class_type": "LoraLoaderModelOnly",
                         "inputs": {"model": low, "lora_name": LORA_LOW, "strength_model": 1.0}}
        high, low = ["lora_high", 0], ["lora_low", 0]
    g["shift_high"] = {"class_type": "ModelSamplingSD3", "inputs": {"model": high, "shift": s.shift}}
    g["shift_low"] = {"class_type": "ModelSamplingSD3", "inputs": {"model": low, "shift": s.shift}}
    cond = {"positive": ["pos", 0], "negative": ["neg", 0], "vae": ["vae", 0], "width": s.width,
            "height": s.height, "length": s.length, "batch_size": 1, "control_video": ["control", 0]}
    if start_image is not None and s.use_start_image:
        g["start"] = {"class_type": "I2G_LoadImages", "inputs": {"path": str(start_image)}}
        cond["start_image"] = ["start", 0]
    if ref_image is not None and s.use_ref_image:
        g["ref"] = {"class_type": "I2G_LoadImages", "inputs": {"path": str(ref_image)}}
        cond["ref_image"] = ["ref", 0]
    g["cond"] = {"class_type": "I2G_Wan22FunControlToVideo", "inputs": cond}
    common = {"steps": s.steps, "cfg": s.cfg, "sampler_name": s.sampler, "scheduler": s.scheduler,
              "positive": ["cond", 0], "negative": ["cond", 1]}
    g["sample_high"] = {"class_type": "KSamplerAdvanced", "inputs": common | {
        "model": ["shift_high", 0], "add_noise": "enable", "noise_seed": s.seed, "latent_image": ["cond", 2],
        "start_at_step": 0, "end_at_step": half, "return_with_leftover_noise": "enable"}}
    g["sample_low"] = {"class_type": "KSamplerAdvanced", "inputs": common | {
        "model": ["shift_low", 0], "add_noise": "disable", "noise_seed": 0, "latent_image": ["sample_high", 0],
        "start_at_step": half, "end_at_step": 10000, "return_with_leftover_noise": "disable"}}
    g["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sample_low", 0], "vae": ["vae", 0]}}
    g["save"] = {"class_type": "I2G_SaveImages", "inputs": {"images": ["decode", 0], "directory": str(out_dir),
                                                             "prefix": "frame_"}}
    return g


def generate_arcs(client: ComfyClient, s: GenSettings, control_dirs: list[Path], first_start: Path,
                  ref_image: Path, out_root: Path, max_step_seconds: float = 400.0) -> dict:
    """Generate every arc in order; arc k > 0 starts from arc k - 1's last generated frame."""
    check_nodes(client)
    out_root = Path(out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    provenance = {"settings": asdict(s), "models": {"high": HIGH, "low": LOW, "text_encoder": TEXT_ENCODER,
                                                     "vae": VAE, "loras": [LORA_HIGH, LORA_LOW] if s.lightning else []},
                  "negative": NEGATIVE, "arcs": []}
    start = first_start
    for k, control in enumerate(control_dirs):
        out_dir = out_root / f"arc{k}"
        graph = build_graph(s, control, start, ref_image, out_dir)
        (out_root / f"graph_arc{k}.json").write_text(json.dumps(graph, indent=1, ensure_ascii=False), encoding="utf-8")
        t0 = time.perf_counter()
        log.info("Arc %d: %s -> %s (%d steps%s)", k, control.name, out_dir, s.steps, ", Lightning" if s.lightning else "")
        client.run(graph, timeout=7200, max_step_seconds=max_step_seconds)
        frames = sorted(out_dir.glob("frame_*.png"))
        if len(frames) != s.length:
            raise ComfyError(f"arc {k}: expected {s.length} frames in {out_dir}, found {len(frames)}")
        seconds = round(time.perf_counter() - t0, 1)
        log.info("Arc %d done in %.1f min", k, seconds / 60)
        provenance["arcs"].append({"arc": k, "control": str(control), "start_image": str(start),
                                   "seconds": seconds, "frames": len(frames)})
        (out_root / "provenance.json").write_text(json.dumps(provenance, indent=1, ensure_ascii=False),
                                                  encoding="utf-8")
        start = frames[-1]
    return provenance

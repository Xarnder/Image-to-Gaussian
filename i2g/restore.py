"""Render -> restore -> retrain: extra training views made from the splat itself.

The first splat is rendered on a dense orbit (exact cameras, so no registration error). SeedVR2, a
one-step video restoration model, cleans and sharpens the orbit as one video, so neighbouring frames
stay consistent with each other. The restored frames join the training set at a low weight: they fill
the angles the generated views missed and pull the splat toward sharper, cleaner surfaces.
SeedVR2 runs in ComfyUI (numz/ComfyUI-SeedVR2_VideoUpscaler); frames are loaded with VideoHelperSuite.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .cameras import azimuth_elevation, orbit_camera
from .comfy_client import ComfyClient
from .log import NullBar, get_logger

log = get_logger("restore")

SEEDVR2_DIT = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
SEEDVR2_VAE = "ema_vae_fp16.safetensors"
# (elevation above the person's centre, frames around): eye level densest, then above and below
ORBIT_RINGS = ((5.0, 48), (28.0, 24), (-15.0, 24))


def orbit_cameras(world: dict, rings=ORBIT_RINGS) -> list[dict]:
    photo_az, _, radius = azimuth_elevation(world["photo_w2c"], world["target"])
    cams = []
    for el, n in rings:
        for k in range(n):
            az = (photo_az + 360.0 * k / n) % 360.0
            cams.append({"tag": f"orbit_e{el:+03.0f}_a{round(az):03d}", "azimuth_deg": az, "elevation_deg": el,
                         "radius_m": radius, "w2c": orbit_camera(world["target"], radius, az, el)})
    return cams


def render_orbit(params: dict, world: dict, cams: list[dict], out_dir: Path) -> None:
    """frames/<ring>/NNN.png (over grey) and alphas/<tag>.png, one folder per ring so each is one video."""
    from .evaluate import render
    K, (W, H) = world["K"], world["size"]
    for i, cam in enumerate(cams):
        ring = cam["tag"].split("_a")[0]
        rgb, alpha = render(params, cam["w2c"], K, W, H)
        (out_dir / "frames" / ring).mkdir(parents=True, exist_ok=True)
        (out_dir / "alphas").mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(out_dir / "frames" / ring / f"{i:03d}.png")
        Image.fromarray((alpha * 255).round().astype(np.uint8)).save(out_dir / "alphas" / f"{cam['tag']}.png")


def seedvr2_workflow(directory: Path, prefix: str, resolution: int, batch_size: int = 13, overlap: int = 3,
                     dit: str = SEEDVR2_DIT, vae: str = SEEDVR2_VAE, seed: int = 42, device: str = "cuda:0",
                     color_correction: str = "lab") -> dict:
    return {
        "1": {"class_type": "VHS_LoadImagesPath",
              "inputs": {"directory": str(directory), "image_load_cap": 0, "skip_first_images": 0,
                         "select_every_nth": 1}},
        "2": {"class_type": "SeedVR2LoadDiTModel",
              "inputs": {"model": dit, "device": device, "blocks_to_swap": 0, "swap_io_components": False,
                         "offload_device": "cpu", "cache_model": False, "attention_mode": "sdpa"}},
        "3": {"class_type": "SeedVR2LoadVAEModel",
              # tiled: untiled 21-frame batches at 1248x1872 filled the 24 GB and spilled into system RAM
              "inputs": {"model": vae, "device": device, "encode_tiled": True, "encode_tile_size": 1024,
                         "encode_tile_overlap": 128, "decode_tiled": True, "decode_tile_size": 1024,
                         "decode_tile_overlap": 128, "tile_debug": "false", "offload_device": "cpu",
                         "cache_model": False}},
        "4": {"class_type": "SeedVR2VideoUpscaler",
              "inputs": {"image": ["1", 0], "dit": ["2", 0], "vae": ["3", 0], "seed": seed,
                         "resolution": resolution, "max_resolution": 0, "batch_size": batch_size,
                         "uniform_batch_size": True, "temporal_overlap": overlap, "prepend_frames": 0,
                         "color_correction": color_correction, "input_noise_scale": 0.0,
                         "latent_noise_scale": 0.0, "offload_device": "cpu", "enable_debug": False}},
        "5": {"class_type": "SaveImage", "inputs": {"images": ["4", 0], "filename_prefix": prefix}},
    }


def restore_orbit(client: ComfyClient, out_dir: Path, cams: list[dict], name: str, scale: float = 1.5,
                  dit: str = SEEDVR2_DIT, bar=None) -> list[dict]:
    """Restore every ring as one video; returns registration-style records for the restored frames."""
    bar = bar or NullBar()
    client.free()          # SeedVR2 loads its own models: make room first
    width, height = Image.open(next((out_dir / "frames").rglob("*.png"))).size
    rings = sorted({c["tag"].split("_a")[0] for c in cams})
    restored_dir = out_dir / "restored"
    records = []
    for ring in rings:
        ring_cams = [c for c in cams if c["tag"].startswith(ring + "_a")]
        dest = [restored_dir / f"{c['tag']}.png" for c in ring_cams]
        if not all(d.exists() for d in dest):
            start = time.perf_counter()
            # one ring = one video; resolution is SeedVR2's short (width) edge
            wf = seedvr2_workflow(out_dir / "frames" / ring, f"i2g/{name}/restore/{ring}",
                                  resolution=int(round(width * scale / 2)) * 2, dit=dit)
            entry = client.run(wf, timeout=3600)      # SeedVR2 reports no sampler steps to police
            files = client.output_files(entry)
            if len(files) != len(ring_cams):
                raise RuntimeError(f"SeedVR2 returned {len(files)} frames for {len(ring_cams)} in ring {ring}")
            for f, d in zip(files, dest):         # kept at SeedVR2's size; the trainer scales K to match
                client.download(f, d)
            log.info("Restored ring %s (%d frames) in %.0f s", ring, len(ring_cams), time.perf_counter() - start)
        for c, d in zip(ring_cams, dest):
            records.append({"tag": c["tag"], "seed": 0, "kind": "restored", "path": str(d),
                            "mask": str(out_dir / "alphas" / f"{c['tag']}.png"),
                            "w2c": np.asarray(c["w2c"]).tolist(), "azimuth_deg": c["azimuth_deg"],
                            "elevation_deg": c["elevation_deg"], "radius_m": c["radius_m"],
                            "accepted": True, "selected": True})
        bar.update(1)
    (out_dir / "restored_views.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    return records

"""M0 smoke tests for the v2 environment (plan.md §6).

1. CUDA sees the RTX 4090.
2. gsplat rasterizes a random splat forward and backward on CUDA.
3. The driver's CUDA sysmem fallback is off for this process: allocating past free VRAM must fail
   with an out-of-memory error instead of silently spilling into system RAM.
4. ComfyUI answers on its HTTP API, runs a tiny LoadImage -> ImageScale -> SaveImage workflow, and
   exposes the nodes later milestones need.

Writes logs/v2_env_check.json and exits non-zero if a required check fails.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import vram
from .comfy_client import ComfyClient
from .paths import COMFY_URL, LOGS_DIR

# Node classes each milestone relies on; missing ones are reported, only M1 ones fail the check.
REQUIRED_NODES = {
    "M1 body scaffold": ["LoadImage", "SaveImage", "SAM3DBody_Loader", "SAM3DBody_Predict",
                         "SAM3DBody_Render", "I2G_SaveMHRPose"],
    "M2 anchor views": ["UNETLoader", "CLIPLoader", "VAELoader", "LoraLoaderModelOnly",
                        "TextEncodeQwenImageEditPlus", "KSampler", "VAEEncode", "VAEDecode",
                        "ModelSamplingAuraFlow"],
    "M4 orbit": ["UnetLoaderGGUF", "WanFirstLastFrameToVideo", "LoadDA3Model", "DA3Inference"],
    "M3 viewing": ["File3DToSplat", "RenderSplat", "CreateCameraInfo"],
}


def check_cuda(report: dict) -> bool:
    ok = torch.cuda.is_available()
    info = {"torch": torch.__version__, "cuda": torch.version.cuda, "available": ok}
    if ok:
        props = torch.cuda.get_device_properties(0)
        free, total = vram.free_and_total_gib()
        info.update(device=props.name, capability=f"{props.major}.{props.minor}",
                    total_gib=round(total, 2), free_gib=round(free, 2))
    report["cuda"] = info
    print(f"[cuda] {info}")
    return ok and "4090" in info.get("device", "")


def check_gsplat(report: dict) -> bool:
    import gsplat
    from gsplat import rasterization

    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(0)
    n = 200_000
    means = ((torch.rand(n, 3, device=dev, generator=gen) - 0.5) * 2).requires_grad_(True)
    quats = torch.nn.functional.normalize(torch.randn(n, 4, device=dev, generator=gen), dim=-1)
    scales = torch.full((n, 3), 0.01, device=dev)
    opacities = torch.full((n,), 0.6, device=dev)
    colors = torch.rand(n, 3, device=dev, generator=gen).requires_grad_(True)
    viewmat = torch.eye(4, device=dev)
    viewmat[2, 3] = 4.0  # world origin 4 m in front of the camera
    K = torch.tensor([[1100.0, 0.0, 512.0], [0.0, 1100.0, 768.0], [0.0, 0.0, 1.0]], device=dev)

    with vram.stage("gsplat_smoke", report):
        start = time.perf_counter()
        for _ in range(5):
            render, alpha, _meta = rasterization(means, quats, scales, opacities, colors,
                                                 viewmat[None], K[None], width=1024, height=1536)
            render.mean().backward()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - start) / 5 * 1000

    ok = bool(torch.isfinite(render).all() and alpha.max() > 0.5
              and means.grad is not None and torch.isfinite(means.grad).all() and means.grad.abs().sum() > 0)
    report["gsplat"] = {"version": gsplat.__version__, "ok": ok, "ms_per_fwd_bwd_1024x1536_200k": round(ms, 1),
                        "alpha_max": round(float(alpha.detach().max()), 3)}
    print(f"[gsplat] {report['gsplat']}")
    return ok


def check_sysmem_fallback(report: dict) -> bool:
    torch.cuda.empty_cache()
    free, _total = torch.cuda.mem_get_info()
    target = free + int(1.5 * vram.GIB)  # try to go 1.5 GiB past what is physically free
    chunk = 512 * 1024 ** 2
    blocks, allocated, hit_oom = [], 0, False
    try:
        while allocated < target:
            # zeros (not empty) so every block is written and must be resident somewhere
            blocks.append(torch.zeros(chunk, dtype=torch.uint8, device="cuda"))
            allocated += chunk
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        hit_oom = True
    finally:
        blocks.clear()
        torch.cuda.empty_cache()
    result = {"free_gib_at_start": round(free / vram.GIB, 2), "allocated_gib": round(allocated / vram.GIB, 2),
              "hit_oom": hit_oom, "fallback_disabled": hit_oom, "python": sys.executable}
    report["sysmem_fallback"] = result
    if hit_oom:
        print(f"[sysmem] OK: allocation stopped at {result['allocated_gib']} GiB with an OOM (fallback disabled)")
    else:
        print(f"[sysmem] WARNING: allocated {result['allocated_gib']} GiB > {result['free_gib_at_start']} GiB free; "
              "the driver spilled into system RAM. Set 'CUDA - Sysmem Fallback Policy' to "
              "'Prefer No Sysmem Fallback' under NVIDIA Control Panel > Manage 3D settings > Global Settings.")
    return hit_oom


def check_comfy(report: dict, url: str) -> bool:
    client = ComfyClient(url)
    if not client.is_up():
        report["comfy"] = {"up": False, "url": url}
        print(f"[comfy] not reachable at {url} (start it with scripts\\v2_start_comfy.ps1)")
        return False

    stats = client.system_stats()
    devices = stats.get("devices", [])
    info = {"up": True, "url": url, "comfyui_version": stats.get("system", {}).get("comfyui_version"),
            "vram_free_gib": round(devices[0]["vram_free"] / vram.GIB, 2) if devices else None}

    available = set(client.object_info().keys())
    info["missing_nodes"] = {group: [n for n in nodes if n not in available]
                             for group, nodes in REQUIRED_NODES.items()}

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "i2g_smoke_in.png"
        grad = np.linspace(0, 255, 320, dtype=np.uint8)
        Image.fromarray(np.stack([np.tile(grad, (480, 1))] * 3, axis=-1)).save(src)
        name = client.upload_image(src)
        workflow = {
            "1": {"class_type": "LoadImage", "inputs": {"image": name}},
            "2": {"class_type": "ImageScale", "inputs": {"image": ["1", 0], "upscale_method": "bilinear",
                                                         "width": 160, "height": 240, "crop": "disabled"}},
            "3": {"class_type": "SaveImage", "inputs": {"images": ["2", 0], "filename_prefix": "i2g/smoke"}},
        }
        start = time.perf_counter()
        entry = client.run(workflow, timeout=180, max_step_seconds=60)
        files = client.output_files(entry)
        out = client.download(files[0], Path(tmp) / "out.png")
        with Image.open(out) as img:
            size = img.size
    info["round_trip_seconds"] = round(time.perf_counter() - start, 2)
    info["round_trip_ok"] = size == (160, 240)
    client.free()

    report["comfy"] = info
    print(f"[comfy] {info}")
    return info["round_trip_ok"] and not info["missing_nodes"]["M1 body scaffold"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--comfy-url", default=COMFY_URL)
    parser.add_argument("--skip-fallback-test", action="store_true")
    args = parser.parse_args()

    report: dict = {}
    results = {"cuda": check_cuda(report)}
    if results["cuda"]:
        results["gsplat"] = check_gsplat(report)
        if not args.skip_fallback_test:
            results["sysmem_fallback"] = check_sysmem_fallback(report)
    results["comfy"] = check_comfy(report, args.comfy_url)
    report["results"] = results

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out = LOGS_DIR / "v2_env_check.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n{json.dumps(results)}\nreport: {out}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

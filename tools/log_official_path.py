#!/usr/bin/env python3
"""Log the official 4DAnyone 4090 low-memory selection before inference.py runs.

Mirrors inference.py: allocator config and auto→SDPA happen before model imports.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, "/opt/4danyone")

from fdanyone.attention import validate_attention_backend
from fdanyone.device import configure_inference_cuda_allocator, has_low_memory_gpu


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention-backend", default="auto")
    parser.add_argument("--enable-turbo", default="True")
    parser.add_argument("--gpu-ids", default="0")
    args = parser.parse_args()

    gpu_ids = [int(part) for part in args.gpu_ids.split(",") if part.strip()]
    validate_attention_backend(args.attention_backend)
    low_memory = has_low_memory_gpu(gpu_ids)
    configure_inference_cuda_allocator(use_expandable_segments=low_memory)
    resolved = args.attention_backend
    if resolved == "auto" and low_memory:
        resolved = "sdpa"

    print(f"low_memory={low_memory}")
    print(f"attention_backend_requested={args.attention_backend}")
    print(f"attention_backend_resolved={resolved}")
    print(f"enable_turbo={args.enable_turbo}")
    print(f"cuda_alloc_conf={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
    try:
        import subprocess

        rev = subprocess.check_output(
            ["git", "-C", "/opt/4danyone", "rev-parse", "HEAD"],
            text=True,
            timeout=5,
        ).strip()
        print(f"fdanyone_commit={rev}")
    except Exception as exc:
        print(f"fdanyone_commit_error={exc}")

    import torch

    print(f"torch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        return 1
    print(f"gpu_name={torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"vram_gib={props.total_memory / (1024**3):.2f}")
    print(f"cuda_capability={props.major}.{props.minor}")
    if resolved != "sdpa" and low_memory:
        print("ERROR: low-memory GPU did not resolve to sdpa", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

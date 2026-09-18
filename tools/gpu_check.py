#!/usr/bin/env python3
"""Fail the container if PyTorch cannot see the RTX 4090."""
from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
    except Exception as exc:
        print(f"ERROR: could not import torch: {exc}", file=sys.stderr)
        return 1

    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("ERROR: CUDA is unavailable. Refusing to continue on CPU.", file=sys.stderr)
        return 1

    name = torch.cuda.get_device_name(0)
    print(name)
    props = torch.cuda.get_device_properties(0)
    print(f"vram_gib={props.total_memory / (1024**3):.2f}")
    print(f"cuda_capability={props.major}.{props.minor}")

    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print(f"matmul_ok={bool(torch.isfinite(y).all().item())}")

    if "4090" not in name:
        print(f"ERROR: expected NVIDIA GeForce RTX 4090, got {name}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

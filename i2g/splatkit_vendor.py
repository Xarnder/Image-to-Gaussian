"""Code vendored from ComfyUI-SplatKit (MIT, https://github.com/mickmumpitz/ComfyUI-SplatKit),
core/splatting/training/{perceptual.py, ply.py}. perceptual.py is itself ported from 4DAnyone's
fdanyone/nerfstudio/perceptual.py (Apache-2.0). Kept close to upstream; trimmed to what v2 uses.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# --- VGG-19 perceptual objective (4DAnyone recipe: L1 pixel term + five L1 feature terms) -------

VGG19_CONV_CHANNELS = ((3, 64), (64, 64), (64, 128), (128, 128), (128, 256), (256, 256),
                       (256, 256), (256, 256), (256, 512), (512, 512), (512, 512),
                       (512, 512), (512, 512), (512, 512))
VGG19_FEATURE_LAYERS = (2, 4, 6, 10, 14)
VGG19_POOL_AFTER = frozenset((2, 4, 8, 12))
FEATURE_DIVISORS = (2.6, 4.8, 3.7, 5.6, 0.15)


class _Features(nn.Module):
    def __init__(self, weights_path: str | Path) -> None:
        super().__init__()
        from safetensors.torch import load_file
        state = load_file(str(weights_path), device="cpu")
        for index, (cin, cout) in enumerate(VGG19_CONV_CHANNELS, start=1):
            w, b = state[f"conv{index}.weight"], state[f"conv{index}.bias"]
            if tuple(w.shape) != (cout, cin, 3, 3) or tuple(b.shape) != (cout,):
                raise ValueError(f"VGG-19 weights at {weights_path} are malformed at conv{index}")
            self.register_buffer(f"conv{index}_weight", w.float().contiguous(), persistent=False)
            self.register_buffer(f"conv{index}_bias", b.float().contiguous(), persistent=False)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features, value = [], image.float().contiguous()
        for index in range(1, len(VGG19_CONV_CHANNELS) + 1):
            value = F.relu(F.conv2d(value, getattr(self, f"conv{index}_weight"),
                                    getattr(self, f"conv{index}_bias"), padding=1))
            if index in VGG19_FEATURE_LAYERS:
                features.append(value)
            if index in VGG19_POOL_AFTER:
                value = F.avg_pool2d(value, kernel_size=2, stride=2)
        return tuple(features)


class PerceptualLoss(nn.Module):
    def __init__(self, weights_path: str | Path) -> None:
        super().__init__()
        self.features = _Features(weights_path)
        self.register_buffer("imagenet_mean", torch.tensor((123.6800, 116.7790, 103.9390)).reshape(1, 3, 1, 1),
                             persistent=False)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Both [B, 3, H, W] in 0..1."""
        pred = prediction.float() * 255.0 - self.imagenet_mean
        tgt = target.float() * 255.0 - self.imagenet_mean
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                tgt_features = self.features(tgt)
            pred_features = self.features(pred)
        terms = [torch.mean(torch.abs(tgt - pred))]
        terms += [torch.mean(torch.abs(t.float() - p.float())) / d
                  for t, p, d in zip(tgt_features, pred_features, FEATURE_DIVISORS)]
        return torch.stack(terms).sum() / 255.0


# --- dependency-free binary PLY writer -------------------------------------------------------------

def write_ply(path, arrays: dict[str, np.ndarray], comments: tuple[str, ...] = ()) -> None:
    """Write one vertex element. `arrays` maps property name to a [N] or [N,k] column."""
    cols, names = [], []
    n = None
    for name, value in arrays.items():
        value = np.asarray(value)
        value = value[:, None] if value.ndim == 1 else value
        n = value.shape[0] if n is None else n
        if value.shape[0] != n:
            raise ValueError(f"property {name} has {value.shape[0]} rows, expected {n}")
        for k in range(value.shape[1]):
            cols.append(value[:, k])
            names.append(name if value.shape[1] == 1 else f"{name}_{k}")
    dtype = np.dtype([(nm, c.dtype.newbyteorder("<").str) for nm, c in zip(names, cols)])
    out = np.empty(n, dtype=dtype)
    for nm, c in zip(names, cols):
        out[nm] = c
    ply_of = {("f", 4): "float", ("f", 8): "double", ("u", 1): "uchar",
              ("i", 4): "int", ("u", 4): "uint", ("i", 2): "short", ("u", 2): "ushort"}
    header = ["ply", "format binary_little_endian 1.0"]
    header += [f"comment {c}" for c in comments]
    header.append(f"element vertex {n}")
    header += [f"property {ply_of[(dtype[nm].kind, dtype[nm].itemsize)]} {nm}" for nm in names]
    header.append("end_header")
    with open(path, "wb") as fh:
        fh.write(("\n".join(header) + "\n").encode())
        fh.write(out.tobytes())

"""BiRefNet foreground matting (weights in models/birefnet, loaded with trust_remote_code)."""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F

from .paths import MODELS_DIR, REPO_ROOT

os.environ.setdefault("HF_HOME", str(REPO_ROOT / "cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Matter:
    def __init__(self, model_dir=MODELS_DIR / "birefnet", device: str = "cuda"):
        from transformers import AutoModelForImageSegmentation

        self.device = device
        self.model = AutoModelForImageSegmentation.from_pretrained(str(model_dir), trust_remote_code=True)
        self.model.to(device).eval().half()

    @torch.inference_mode()
    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        """HxWx3 uint8 -> HxW float32 alpha in [0, 1]."""
        h, w = rgb.shape[:2]
        x = torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 255.0
        x = F.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False, antialias=True)
        x = ((x - _MEAN) / _STD).to(self.device).half()
        pred = self.model(x)[-1].sigmoid().float()
        alpha = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)
        return alpha[0, 0].clamp(0, 1).cpu().numpy()

    def close(self) -> None:
        del self.model
        torch.cuda.empty_cache()

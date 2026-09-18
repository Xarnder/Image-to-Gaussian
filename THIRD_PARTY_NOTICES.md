# Third-party notices

## ComfyUI-SplatKit (MIT)

`i2g/splatkit_vendor.py` contains code adapted from ComfyUI-SplatKit
(https://github.com/mickmumpitz/ComfyUI-SplatKit), files `core/splatting/training/perceptual.py`
and `core/splatting/training/ply.py`. The pipeline also installs SplatKit's prebuilt Windows gsplat
wheel (release `gsplat-1.4.0-pt28-cu128`), and parts of the trainer's design follow SplatKit's trainer.

```
MIT License

Copyright (c) 2026 mickmumpitz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 4DAnyone (Apache-2.0)

The VGG-19 perceptual loss in `i2g/splatkit_vendor.py` was ported by SplatKit from 4DAnyone's
`fdanyone/nerfstudio/perceptual.py` (AntResearch, Apache-2.0,
https://huggingface.co/AntResearch/4DAnyone). The VGG-19 weights the pipeline downloads come from
the same model repository.

## Libraries and models used at runtime (not included in this repository)

| Component | Source | Licence |
|---|---|---|
| gsplat | https://github.com/nerfstudio-project/gsplat | Apache-2.0 |
| PyTorch | https://pytorch.org | BSD-3-Clause |
| BiRefNet | https://huggingface.co/ZhengPeng7/BiRefNet | MIT |
| SAM 3D Body | https://github.com/facebookresearch/sam-3d-body (ComfyUI repackage: Comfy-Org/sam-3d-body) | SAM License |
| Qwen-Image-Edit-2511 | https://github.com/QwenLM/Qwen-Image (ComfyUI repackage: Comfy-Org) | Apache-2.0 |
| Multiple-Angles LoRA | https://huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA | Apache-2.0 |
| Lightning LoRA | https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning | see model card |
| ComfyUI | https://github.com/comfyanonymous/ComfyUI | GPL-3.0 (used as a separate application) |

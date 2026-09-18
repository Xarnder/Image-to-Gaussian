# Image-to-Gaussian

**One front-facing, full-body photo in → one static 3D Gaussian splat (`.ply`) out, on a single
consumer GPU.**

The pipeline cuts the person out and fits a 3D body to them. It then asks an image-edit model to draw
them from other angles, **measures** where the camera was for each drawn image, and trains a 3D
Gaussian splat on all of it. It runs natively on Windows with an RTX 4090 (24 GB) and 32 GB of RAM.

```
photo ─► cut-out (BiRefNet) ─► 3D body (SAM 3D Body) ─► new views (Qwen-Image-Edit-2511 + angles LoRA)
      ─► measure every view's camera (keypoint PnP) ─► train the splat (gsplat) ─► character.ply
```

New to the project? [pipeline_breakdown.md](pipeline_breakdown.md) explains every stage, tool and
library for beginners, and how to rebuild it from scratch.

---

## Requirements

| | |
|---|---|
| OS | Windows 10/11 (scripts are Windows-first) |
| GPU | NVIDIA with **24 GB VRAM** (tested: RTX 4090) + a recent driver |
| RAM | 32 GB |
| Disk | ~45 GB free: ~33 GB models, ~5 GB Python environment, room for outputs. An SSD makes model loading much faster. |
| Software | [Git](https://git-scm.com/), [uv](https://docs.astral.sh/uv/) (`winget install --id=astral-sh.uv -e`), [ComfyUI portable](https://github.com/comfyanonymous/ComfyUI/releases) **v0.35 or newer** (it has the SAM 3D Body and Qwen-Image-Edit nodes built in) |

---

## Installation (one time)

**1. Clone the repository**
```bash
git clone https://github.com/YOUR-USERNAME/Image-to-Gaussian.git
```
```bash
cd Image-to-Gaussian
```

**2. Tell it where ComfyUI is.** Copy the example config, then edit `comfy_root` in
`config.local.json` to your portable ComfyUI folder (the one that contains `python_embeded` and
`ComfyUI`).
```bash
copy config.example.json config.local.json
```

**3. Create the Python environment** (Python 3.11 + PyTorch 2.8 CUDA 12.8 + gsplat; about 5 min,
downloads ~3.5 GB):
```bash
powershell -ExecutionPolicy Bypass -File scripts\v2_setup_env.ps1
```

<details>
<summary>Manual alternative (if you prefer your own tools)</summary>

Needs Python **3.11** exactly, because the gsplat Windows wheel is built for it.

```bash
uv venv .venv-i2g --python 3.11
```
```bash
.\.venv-i2g\Scripts\activate
```
```bash
uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
```
```bash
uv pip install --no-deps https://github.com/mickmumpitz/ComfyUI-SplatKit/releases/download/gsplat-1.4.0-pt28-cu128/gsplat-1.4.0+pt28cu128-cp311-cp311-win_amd64.whl
```
```bash
copy .venv-i2g\Scripts\python.exe .venv-i2g\Scripts\python3.exe
```
</details>

**4. Download the models** (~33 GB; files you already have are skipped):
```bash
.\.venv-i2g\Scripts\activate
```
```bash
python3 scripts\download_models.py
```

| Model | Goes to | Size |
|---|---|---|
| BiRefNet (cut-out) | `models\birefnet\` | 0.4 GB |
| VGG-19 (perceptual loss) | `models\perceptual\` | 0.1 GB |
| Qwen-Image-Edit-2511 fp8mixed | `ComfyUI\models\diffusion_models\` | 20.5 GB |
| Qwen 2.5-VL 7B text encoder (fp8) | `ComfyUI\models\text_encoders\` | 9.4 GB |
| Qwen-Image VAE | `ComfyUI\models\vae\` | 0.25 GB |
| Multiple-Angles LoRA | `ComfyUI\models\loras\` | 0.3 GB |
| Lightning 4-step LoRA | `ComfyUI\models\loras\` | 0.85 GB |
| SAM 3D Body (dinov3, bf16) | `ComfyUI\models\detection\` | 2.8 GB |

**5. Check everything:**
```bash
python3 image_to_splat.py --check
```
It starts ComfyUI if needed (the first start takes 3–8 minutes), installs the pipeline's small
ComfyUI helper node automatically, and verifies the GPU, nodes and models. Anything wrong is printed
with a `FIX:` line.

**Strongly recommended:** NVIDIA Control Panel → *Manage 3D settings* → *Global Settings* →
**CUDA – Sysmem Fallback Policy: Prefer No Sysmem Fallback** → **Apply**. Without it, a VRAM
overflow silently spills into system RAM and runs about 20× slower instead of failing.

---

## Usage

### Easiest: double-click
Double-click **`run_image_to_splat.bat`**, drag your photo into the window, press Enter. Or drag
the photo **onto** the `.bat` file.

### Command line (VS Code terminal)
```bash
.\.venv-i2g\Scripts\activate
```
```bash
python3 image_to_splat.py --input input\your_photo.png --output_dir output\my_splat
```

Always activate the environment first. On Windows, `python3` outside it is usually just a Microsoft
Store placeholder.

More examples:
```bash
python3 image_to_splat.py -i input\your_photo.png --preset draft
```
```bash
python3 image_to_splat.py -i input\your_photo.png --from_stage train --set iterations=20000
```
```bash
python3 image_to_splat.py --help
```

| Option | Meaning |
|---|---|
| `-i`, `--input` | the photo: one person, front-facing, full body, ideally 1500+ px tall |
| `-o`, `--output_dir` | where the result goes; created if missing (default `output\<name>\v2\final`) |
| `-w`, `--working_dir` | intermediate files; created if missing (default `output\<name>\v2`) |
| `--preset draft\|standard\|high` | about 20 / 35 / 50 min; more angles, seeds and training in the higher presets |
| `--views`, `--seeds`, `--iterations`, `--fov`, `--no_hr_photo` | fine-tune a preset |
| `--set KEY=VALUE` | change any training setting (`SplatConfig` in `i2g/splat.py`); repeatable |
| `--from_stage`, `--only_stage`, `--force` | redo part of a run; old results are archived, never deleted |
| `--keep_comfy`, `--stop_comfy`, `--sage` | leave ComfyUI running for the next run / stop it / start it with SageAttention |
| `--comfy_root`, `--comfy_url` | override `config.local.json` |
| `--check`, `--debug` | test the setup / print detailed messages |

### What you get
In the output folder:
- **`character.ply`**: the splat. Drag it onto **https://superspl.at/editor** to view it. It is
  Y-up, in metres, feet at y = 0; if a viewer shows it upside down, rotate it 180°.
- `turntable.mp4`, `preview_8_angles.png`: quick looks.
- `contact_sheet.png`: the generated views and their measured cameras. Look at this first when
  something seems off.
- `photo_vs_render.png`, `report.json`, `run_summary.json`, plus `history\` (earlier results).

### Iterating on quality
Rerunning a command **only redoes the stages whose inputs or settings changed**. Changing
`--set ...` retrains in minutes without regenerating views. An interrupted run continues where it
stopped. See [quality_plan.md](quality_plan.md) for the improvement roadmap.

---

## Timing and memory (RTX 4090, models on an HDD)

| Stage | Draft | Standard | Peak VRAM |
|---|---|---|---|
| ComfyUI start (if not running) | 3–8 min | 3–8 min | – |
| Cut-out + 3D body | 2 min | 2 min | ~5 GB |
| Generate views | 10 min (7) | 14 min (22) | ~22 GB (ComfyUI offloads) |
| Measure cameras | 1 min | 3 min | ~5 GB |
| Train the splat | 4 min | 14 min | 2–3.5 GB |

---

## Troubleshooting

- **Every failure prints `FAILED:` + `FIX:`** and the path of a detailed log in
  `<working_dir>\logs\`. Rerun with `--debug` to see everything on screen.
- **"ComfyUI exited during startup" / port in use:** another ComfyUI is already on port 8188. Close
  it, or point `--comfy_url` at it.
- **Everything is extremely slow:** VRAM is spilling into system RAM. Close other GPU apps and set
  the Sysmem Fallback Policy (above). The tool aborts any sampling step that takes over 2 minutes.
- **SageAttention errors in ComfyUI's log (`tcc: undefined symbol ...`):** usually a machine-wide
  `CUDA_PATH`/`CUDA_HOME` pointing at a conda env, or ComfyUI's `python_embeded\Include` headers
  from an older Python. The pipeline starts ComfyUI with Triton's bundled CUDA to avoid the first;
  SageAttention stays off unless you pass `--sage`.
- **`python3` opens the Microsoft Store:** activate the environment first
  (`.\.venv-i2g\Scripts\activate`), or just use the `.bat`.

---

## Project layout

```
run_image_to_splat.bat     double-click launcher
image_to_splat.py          the command-line tool
config.example.json        copy to config.local.json (your ComfyUI location)
requirements.txt           Python packages (installed by scripts\v2_setup_env.ps1)
i2g\                       the pipeline package (stages, camera maths, training)
comfy_nodes\i2g_export\    small ComfyUI node, auto-linked into ComfyUI\custom_nodes
scripts\                   setup, model download, ComfyUI start/stop, status helpers
input\                     put your photos here (not published)
output\  logs\  models\    created locally, not published
docs\                      history of the abandoned 4DAnyone approach
```

Documents: [pipeline_breakdown.md](pipeline_breakdown.md) (how it works) ·
[plan.md](plan.md) (design and status) · [quality_plan.md](quality_plan.md) (what to improve next).

---

## Limitations

- **Static only.** The person is frozen in the photo's pose; there's no rig and no animation.
- **One person, front-facing, full body.** Loose clothing and complex hair are harder.
- **The back and sides are AI-generated guesses.** They are plausible, not ground truth.
- **Known artifacts:** streaks behind the legs at exact side views, and a softer face when you look
  from well above or below the photo's angle. These are being worked on in
  [quality_plan.md](quality_plan.md).
- **Windows + 24 GB NVIDIA GPU** is the tested setup.

**Please only process photos of people who have agreed to it.**

---

## History

The first attempt used 4DAnyone, a 4D multi-view video model, in Docker. It could not run on a
24 GB GPU that is also driving a display. That approach is kept for reference in
[docs/LEGACY_4DANYONE.md](docs/LEGACY_4DANYONE.md) and [docs/STATUS.md](docs/STATUS.md). The 4DAnyone
paper (arXiv 2608.20335) inspired parts of this pipeline.

---

## Credits and licences

This repository contains code only. **No model weights are included.** Each model keeps its own
licence; check them before any commercial use.

| Component | Licence |
|---|---|
| Qwen-Image-Edit-2511, Qwen-Image text encoder/VAE (Alibaba Qwen, ComfyUI repackage) | Apache-2.0 |
| Multiple-Angles LoRA (fal) | Apache-2.0 |
| Lightning LoRA (lightx2v) | see its model card |
| SAM 3D Body (Meta) | SAM License |
| BiRefNet | MIT |
| VGG-19 perceptual weights (via 4DAnyone) | Apache-2.0 |
| gsplat (dependency) | Apache-2.0 |
| ComfyUI (used as a separate application) | GPL-3.0 |
| Vendored code from ComfyUI-SplatKit | MIT: see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) |

This repository's own licence: *choose one and add a `LICENSE` file before publishing.*

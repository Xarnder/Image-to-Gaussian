# Pipeline breakdown: how Image-to-Gaussian works

A guide for an engineer who is new to this project, or to 3D reconstruction. It explains what the
pipeline does, why it is built this way, which tools and libraries it uses, and how you would rebuild
it from scratch.

> **In one sentence:** we take **one photo** of a person, use AI models to **imagine what they look
> like from other angles**, **measure where the camera was** for each imagined image, and then
> **train a 3D Gaussian splat** that matches all of those images.

Related documents:
- [plan.md](plan.md): project plan, status, decisions
- [quality_plan.md](quality_plan.md): ranked list of quality improvements
- [docs/STATUS.md](docs/STATUS.md): the abandoned first approach (4DAnyone) and why it failed

---

## Contents

1. [What goes in, what comes out](#1-what-goes-in-what-comes-out)
2. [Concepts you need first](#2-concepts-you-need-first)
3. [The big picture](#3-the-big-picture)
4. [Tech stack: tools, libraries, models](#4-tech-stack-tools-libraries-models)
5. [Stage by stage](#5-stage-by-stage)
6. [The orchestration layer (the easy-run tool)](#6-the-orchestration-layer-the-easy-run-tool)
7. [Conventions you must get right](#7-conventions-you-must-get-right)
8. [Files and data formats](#8-files-and-data-formats)
9. [Rebuilding from scratch, step by step](#9-rebuilding-from-scratch-step-by-step)
10. [Lessons learned the hard way](#10-lessons-learned-the-hard-way)
11. [Performance on the reference PC](#11-performance-on-the-reference-pc)
12. [Glossary](#12-glossary)

---

## 1. What goes in, what comes out

| | |
|---|---|
| **Input** | One photo (png/jpg/webp) of **one person**, **front-facing**, **full body** head to feet |
| **Output** | `character.ply`: a static 3D Gaussian splat, viewable from any angle (e.g. drag it onto https://superspl.at/editor) |
| **Also produced** | `turntable.mp4` (360° spin), `preview_8_angles.png`, `contact_sheet.png` (the generated views), `photo_vs_render.png`, `report.json` |
| **Not produced** | animation, a rig, a mesh. The person is frozen in the pose from the photo |
| **Runtime** | ~20 min (draft) to ~50 min (high quality) on an RTX 4090 |

---

## 2. Concepts you need first

If these are new to you, read this section slowly. Everything later builds on it.

### 2.1 What is a Gaussian splat?

A **3D Gaussian splat** (3DGS) represents a scene as hundreds of thousands of small, fuzzy, coloured
3D blobs ("Gaussians"). Each one has:

| Property | Meaning | Stored as |
|---|---|---|
| position | where it is in 3D (x, y, z) | 3 floats |
| scale | its size along its 3 axes | 3 floats (log) |
| rotation | its orientation | quaternion (4 floats) |
| opacity | how see-through it is | 1 float (logit) |
| colour | its colour, optionally changing with viewing direction | spherical-harmonics coefficients |

To draw an image, a **rasterizer** projects every blob onto the screen and blends them front to back.
This is fast (real time) and **differentiable**: we can compute how every blob parameter should
change to make the rendered image closer to a target image. **Training** a splat means repeating
"render → compare with the photo → nudge parameters" thousands of times.

**The catch:** training needs **many images of the subject from known camera positions**. We have
**one** photo. That's the whole problem this pipeline solves.

### 2.2 Cameras: intrinsics and extrinsics

A camera is described by two things:

- **Intrinsics (K)**, the lens: focal length in pixels (`fx`, `fy`) and image centre (`cx`, `cy`).
  A 3D point (X, Y, Z) in camera space lands on pixel `u = fx·X/Z + cx`, `v = fy·Y/Z + cy`.
- **Extrinsics (w2c)**, where the camera is and where it looks: a 4×4 matrix that converts world
  coordinates into camera coordinates (rotation + translation).

"**Registering**" an image means finding its extrinsics.

### 2.3 Diffusion image editing, LoRAs and "novel views"

**Qwen-Image-Edit** is a large AI model that edits an image according to a text instruction. A
**LoRA** is a small add-on file that teaches a big model a new skill without retraining it. We use a
LoRA trained to **re-draw the same subject from a different camera angle** ("novel view synthesis"),
controlled by special prompt words like `<sks> back view eye-level shot medium shot`.

**Important:** these generated views are *imagined*. They are plausible but not perfectly consistent
with each other, and the camera angle is approximate. The pipeline is designed around that fact.

### 2.4 ComfyUI

**ComfyUI** is a popular app for running AI image and video models as node graphs ("workflows"). It
also has an **HTTP API**, so our Python code can send it a workflow as JSON, wait for it and download
the result. ComfyUI is good at fitting big models into a 24 GB GPU (offloading, fp8 weights), so we
let it run the heavy models and do everything else in our own Python.

### 2.5 A 3D body model (SAM 3D Body / MHR)

**SAM 3D Body** (Meta) looks at a photo of a person and outputs a 3D human mesh in the same pose
(18,439 vertices), plus **70 keypoints** (nose, eyes, shoulders, elbows, wrists, hips, knees, ankles,
fingers, ...) in 3D and in 2D. The underlying body model is called **MHR** (Momentum Human Rig). We
use it as a **scaffold**: a 3D skeleton that tells us where the person is, how tall they are, and
which way they face.

### 2.6 PnP (Perspective-n-Point)

If you know some **3D points** (the scaffold's keypoints) and where they appear in an **image** (2D
keypoints), PnP finds the **camera pose** that makes them line up. This is how we measure each
generated image's camera.

### 2.7 Visual hull

If you cut the person out (a **matte**) in several images with known cameras, a 3D point can only be
part of the person if it projects **inside the matte in every image**. Keeping only those points
("space carving") gives a rough 3D shape, the **visual hull**. We start the splat from it.

---

## 3. The big picture

```
 photo.png
    │
    ▼  S1 INGEST ─────────── BiRefNet (our Python)
 cut-out person on a grey 832×1248 "canvas" + full-res crop
    │
    ▼  S2 BODY ───────────── SAM 3D Body (inside ComfyUI)
 3D body scaffold → world frame (metres, Y up) + the photo's camera
    │
    ▼  S3 VIEWS ──────────── Qwen-Image-Edit-2511 + Multiple-Angles LoRA (inside ComfyUI)
 7–15 angles × 1–3 seeds of imagined views (back, sides, quarters, elevated...)
    │
    ▼  S4 REGISTER ───────── BiRefNet + SAM 3D Body + PnP (ComfyUI + our Python)
 the real camera of every generated view, quality checks, best views selected, contact sheet
    │
    ▼  S5 TRAIN ──────────── gsplat (our Python, CUDA)
 visual hull → 15k training steps → cleanup → character.ply + turntable.mp4
```

**Who runs what:**

| Process | Python | Runs |
|---|---|---|
| **ComfyUI** (portable; its folder is set in `config.local.json`) | its own embedded Python 3.13 + torch 2.13 | SAM 3D Body, Qwen-Image-Edit (the big models) |
| **Our pipeline** (this repository) | `.venv-i2g`: Python 3.11 + torch 2.8 | everything else: matting, geometry, registration, training, export |

They talk over HTTP on `127.0.0.1:8188`. **Only one of them uses the GPU at a time.** Before our
own GPU work, we tell ComfyUI to unload its models (`POST /free`).

**Why this design?**

- **One photo isn't enough to train a splat**, so we generate more views (S3).
- **Generated views have unreliable camera angles**, so we *measure* them against a 3D body instead
  of trusting the prompt (S2 + S4). That is the key trick.
- **Generated views disagree slightly with each other**, so training weights the real photo highest
  and gives every generated view its own small colour and camera corrections (S5).
- **The big models don't fit a 24 GB GPU at full precision**, so they run in ComfyUI with fp8
  weights and automatic offloading; the rest runs in a lean environment.

---

## 4. Tech stack: tools, libraries, models

### 4.1 Hardware and OS (reference machine)

| Part | Reference | Minimum that should work |
|---|---|---|
| GPU | RTX 4090, 24 GB VRAM | 24 GB for Qwen at fp8 (smaller cards need GGUF-quantised Qwen) |
| RAM | 32 GB | 32 GB (ComfyUI pins ~12 GB for offloading) |
| OS | Windows 11, NVIDIA driver 596 | Windows 10/11 or Linux (scripts are Windows-first) |
| Disk | models on an HDD | an SSD makes model loading ~5× faster |

### 4.2 Tools

| Tool | Used for |
|---|---|
| **ComfyUI portable** ≥ 0.35 | runs SAM 3D Body and Qwen-Image-Edit through its HTTP API. Its built-in `SAM3DBody_*` and Qwen nodes are what we call. |
| **uv** | creates the Python 3.11 environment and installs packages fast (`scripts\v2_setup_env.ps1`) |
| **PowerShell 5.1 / cmd** | setup and helper scripts, the double-click `.bat` |
| **VS Code** | editing; `.vscode\launch.json` has run/debug entries |
| **SuperSplat** (browser) | viewing and cleaning the final `.ply` |

### 4.3 Python libraries in `.venv-i2g` (Python 3.11)

Pinned in [requirements.txt](requirements.txt).

| Library | Version | Why we need it |
|---|---|---|
| **torch** / torchvision | 2.8.0 + CUDA 12.8 | tensors on the GPU, autograd, the whole training loop |
| **gsplat** | 1.4.0 | the Gaussian-splat rasterizer (CUDA kernels) and the densification strategy. Windows wheel built by ComfyUI-SplatKit, SHA-256 checked. |
| jaxtyping, rich, ninja | pinned | gsplat's runtime dependencies |
| **numpy** | 1.26.4 | array maths everywhere |
| **opencv-python** | 4.11 | image resize/draw, `solvePnPRefineLM` (camera refinement), Rodrigues rotations |
| **pillow** | 12.x | reading and writing images |
| **transformers**, timm, einops, kornia | 4.57 / 1.0 / ... | loading and running **BiRefNet** (its code ships with the weights) |
| **pytorch-msssim** | 1.0.0 | SSIM loss during training |
| **safetensors** | 0.8 | loading model weights (VGG-19) |
| **imageio** + imageio-ffmpeg | 2.37 / 0.6 | writing `turntable.mp4` |
| **websocket-client** | 1.9 | watching ComfyUI's live progress feed (the "stuck step" watchdog) |
| **tqdm** | 4.70 | progress bars |
| scipy, trimesh, onnxruntime-gpu | installed | **not imported by the current code**; reserved for quality-plan items (mesh utilities, ViTPose keypoints, ArcFace identity checks) |

The only things we wrote ourselves are geometry, registration, training logic and orchestration.
Vendored code: the VGG-19 perceptual loss and the PLY writer come from ComfyUI-SplatKit (MIT),
credited in [i2g/splatkit_vendor.py](i2g/splatkit_vendor.py).

### 4.4 AI models

| Model | File | Lives in | Role | Size | Licence (check before commercial use) |
|---|---|---|---|---|---|
| **BiRefNet** | `model.safetensors` + code | `models\birefnet\` | cuts the person out of any image (matting) | 0.4 GB | MIT |
| **SAM 3D Body** (dinov3) | `sam_3d_body_dinov3_bf16.safetensors` | ComfyUI `models\detection\` | 3D body mesh + 70 keypoints from one image | 2.6 GB | SAM License |
| **Qwen-Image-Edit-2511** (fp8mixed) | `qwen_image_edit_2511_fp8mixed.safetensors` | ComfyUI `models\diffusion_models\` | the image-edit model that draws new views | 20.5 GB | Apache-2.0 |
| Qwen 2.5-VL 7B text encoder | `qwen_2.5_vl_7b_fp8_scaled.safetensors` | ComfyUI `models\text_encoders\` (or `clip\`) | reads the prompt + input image for Qwen | 8.7 GB | Apache-2.0 |
| Qwen-Image VAE | `qwen_image_vae.safetensors` | ComfyUI `models\vae\` | converts between pixels and Qwen's latent space | 0.24 GB | Apache-2.0 |
| **Multiple-Angles LoRA** (fal) | `qwen-image-edit-2511-multiple-angles-lora.safetensors` | ComfyUI `models\loras\` | teaches Qwen 96 camera poses (8 azimuths × 4 elevations × 3 distances) | 0.27 GB | Apache-2.0 |
| **Lightning LoRA** (lightx2v) | `Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors` | ComfyUI `models\loras\` | lets Qwen finish in 4 steps instead of ~30 | 0.8 GB | see model card |
| **VGG-19** | `imagenet-vgg-verydeep-19-conv.safetensors` | `models\perceptual\` | "perceptual" loss: compares images by features, not just pixels | 0.08 GB | from the 4DAnyone repo |

Where to download: Hugging Face `Comfy-Org/Qwen-Image-Edit_ComfyUI`, `Comfy-Org/Qwen-Image_ComfyUI`,
`fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA`, `lightx2v/Qwen-Image-Edit-2511-Lightning`,
`ZhengPeng7/BiRefNet`, `AntResearch/4DAnyone` (`perceptual/`). SAM 3D Body's original weights are
gated at `facebook/sam-3d-body-dinov3`; ComfyUI uses a repackaged bf16 file.

---

## 5. Stage by stage

Each stage reads the previous stage's files and writes its own folder, so any stage can be rerun
alone. File paths below are relative to the run's **working folder** (default
`output\<photo name>\v2\`).

### Stage 0: Preflight and ComfyUI

**Code:** [image_to_splat.py](image_to_splat.py) `preflight()`,
[i2g/comfy_process.py](i2g/comfy_process.py), [i2g/comfy_client.py](i2g/comfy_client.py)

Before any real work, the tool checks:
- the photo exists and opens
- PyTorch sees the GPU, and how much VRAM is free
- gsplat imports
- the local models exist
- the ComfyUI folder exists

Then:

1. **Install our helper node.** `comfy_nodes\i2g_export` is junctioned (a folder link) into
   `ComfyUI\custom_nodes`. It adds one node, `I2G_SaveMHRPose`, that saves SAM 3D Body's result as an
   `.npz` file so our Python can read it.
2. **Start ComfyUI** if it isn't running: hidden, on `127.0.0.1:8188`, with `--reserve-vram 1.5`.
   This takes 3–8 min the first time. `CUDA_PATH`/`CUDA_HOME` are set for that process only; see
   §10.
3. **Ask ComfyUI what it has** (`GET /object_info`): every node we need, and every model file in
   ComfyUI's own file lists.

**How the ComfyUI API works:**

| Call | Purpose |
|---|---|
| `POST /upload/image` | put an input image into ComfyUI's `input\` folder |
| `POST /prompt` with `{"prompt": <workflow>, "client_id": ...}` | queue a workflow; returns a `prompt_id` |
| `GET /history/<prompt_id>` | poll until `status.completed`; the entry lists the output files |
| `GET /view?filename=&subfolder=&type=output` | download a result file |
| `POST /free` | unload models from VRAM (we call this before our own GPU stages) |
| `ws://.../ws?clientId=...` | live progress events; our **watchdog** aborts a job whose sampling steps become minutes apart (a sign VRAM spilled into system RAM) |

An **API-format workflow** is a JSON dict of `node_id → {"class_type": ..., "inputs": {...}}`.
A link to another node's output is written `["<node_id>", <output_index>]`. Tip: build a graph in
ComfyUI's UI, then use *Workflow → Export (API)* to see the JSON.

### Stage 1: Ingest (cut-out and canvas)

**Code:** [i2g/ingest.py](i2g/ingest.py), [i2g/matting.py](i2g/matting.py)
**Output:** `00_ingest\canvas.png`, `alpha.png`, `source_hr_rgba.png`, `ingest.json`

1. **BiRefNet** predicts an alpha matte: 1 = person, 0 = background. The input is resized to
   1024×1024 and normalised with ImageNet mean/std, and the model runs in fp16.
2. Find the person's bounding box.
3. Crop a **2:3 portrait box** around them so the person fills **86%** of the height. Where the box
   leaves the photo, it's padded.
4. Resize that crop to the **canvas: 832 × 1248 px** and paint the background **neutral grey (0.5)**.
   - 832×1248 is one of Qwen's native ~1-megapixel sizes, so every generated view has the same size
     and lens.
   - A plain grey background stops the image model inventing scenery, and makes generated views easy
     to cut out.
5. Keep the same crop at **full resolution** (`source_hr_rgba.png`, e.g. 1866×2799), with its matte in
   the alpha channel. Training uses it so the front of the splat is as sharp as the photo.

### Stage 2: Body scaffold

**Code:** [i2g/body.py](i2g/body.py), [comfy_nodes/i2g_export/__init__.py](comfy_nodes/i2g_export/__init__.py)
**Output:** `01_body\sam3d_photo.npz`, `body_world.npz`, `overlay_photo.png`, `body_summary.json`

1. Send the canvas to ComfyUI: `LoadImage → SAM3DBody_Loader → SAM3DBody_Predict (fov 40°) →
   I2G_SaveMHRPose`, then download the `.npz`.
2. `load_person()` reads the mesh vertices, the 70 keypoints (3D and 2D), the camera translation
   `pred_cam_t` and the focal length. **Vertices + `pred_cam_t` = the body in the camera's
   coordinates.**
3. `build_world()` defines a **world frame** from the body:
   - **up (Y)** = from the midpoint of the ankles to the midpoint of the shoulders
   - **facing (Z)** = `cross(left, up)`, where *left* comes from the left-minus-right hips and
     shoulders. The person faces **+Z**, toward the photo camera.
   - **X** = `cross(Y, Z)` = the person's left
   - **origin** on the floor under the body centre, **units in metres**

   The photo camera's extrinsics fall out of this for free.
4. **Check:** `overlay_photo.png` draws the mesh and skeleton on the photo. `coverage` is the share of
   the body silhouette inside the matte; our test image scores 98%. If the overlay is off, everything
   downstream will be off too.

Typical result for the test image: height 1.72 m, camera 2.77 m away, 5° below the body centre.

### Stage 3: Generate views

**Code:** [i2g/views.py](i2g/views.py)
**Output:** `02_views\<tag>\s<seed>.png`, `02_views\views.json`

For every requested angle and seed, one ComfyUI workflow runs. These are the exact settings from
fal's reference workflow:

```
UNETLoader(qwen_image_edit_2511_fp8mixed) → LoRA(Lightning, 1.0) → LoRA(Multiple-Angles, 0.9)
  → ModelSamplingAuraFlow(shift 3.1) → CFGNorm(1.0)
CLIPLoader(qwen_2.5_vl_7b_fp8_scaled, type qwen_image), VAELoader(qwen_image_vae)
TextEncodeQwenImageEditPlus(prompt, image1 = canvas)   (+ an empty one for the negative)
  → FluxKontextMultiReferenceLatentMethod("index_timestep_zero")
EmptySD3LatentImage(832×1248) → KSampler(4 steps, cfg 1.0, euler, simple) → VAEDecode → SaveImage
```

- **Prompt format:** `<sks> {azimuth} {elevation} {distance}`, e.g.
  `<sks> back-left quarter view elevated shot medium shot`.
  - Azimuth tokens: `front view`, `front-right quarter view`, `right side view`,
    `back-right quarter view`, `back view`, `back-left quarter view`, `left side view`,
    `front-left quarter view`.
  - Elevation tokens: `low-angle shot` (−30°), `eye-level shot`, `elevated shot` (+30°),
    `high-angle shot` (+60°).
- **Presets** (`VIEW_PRESETS`):
  - `draft`: 7 eye-level angles
  - `standard`: + 4 elevated
  - `full`: + 4 low-angle
- **Seeds:** 101, 202, 303, ... Each seed gives a different candidate, so rerunning with more seeds
  reuses the old images.
- **Timing:** the first image loads about 28 GB of weights (2–5 min). After that, each **new
  prompt** costs about 57 s, because the text encoder and the 19.6 GB image model can't both sit in
  24 GB and swap in and out of VRAM. An **extra seed** of the same prompt costs about 6 s.
- `views.json` records every generated image, so an interrupted run resumes where it stopped.

### Stage 4: Register (measure every view's camera)

**Code:** [i2g/register.py](i2g/register.py), [i2g/cameras.py](i2g/cameras.py)
**Output:** `03_qa\masks\`, `03_qa\sam3d\` (cached body fits), `registration.json`, `contact_sheet.png`

This is the most important stage. For each generated image:

1. **Matte** it with BiRefNet.
2. **Fit SAM 3D Body to it** in ComfyUI. This gives the imagined person's keypoints in *that image's*
   camera frame (3D) and on the image (2D). Results are cached, so reruns are fast.
3. **Try several camera hypotheses**, because the pose is nearly symmetric:
   - labelling: {as given, **left/right swapped**}, since a back view with left/right swapped looks
     like a front view
   - starting guess: {**3D alignment**, its **180° mirror**, the **requested angle**}
     - The 3D alignment uses the Umeyama method: rotate, scale and shift the scaffold's keypoints
       onto the view's 3D keypoints.

   Each of the 6 combinations is refined by **PnP** (`cv2.solvePnPRefineLM`), so the scaffold's 3D
   keypoints project onto the view's 2D keypoints.
4. **Score** each one: `reprojection error (% of image height) + 0.005 × degrees away from the
   requested angle`. The small pull toward the requested angle only breaks near-ties (front vs back);
   clear 2D evidence still wins. This matters because Qwen sometimes draws the other side than we
   asked for.
5. **Quality gates:**

   | Check | Limit | Catches |
   |---|---|---|
   | pose drift (3D keypoint residual after alignment) | ≤ 10 cm | arms moved, different pose |
   | reprojection error | ≤ 2.5% of the image height | anatomy that fits no camera |
   | coverage (scaffold silhouette inside the matte) | ≥ 0.80 | cropped or shrunken body |

6. **Select:** the photo first, then accepted views from best score down, **skipping any view
   within 20° of one already kept**. Near-duplicates with different details would blur the splat.
7. **Contact sheet:** every view with its measured angle, the projected skeleton, and its metrics.
   Green = selected, amber = good but redundant, red = rejected. **Look at this image after every
   run.**

### Stage 5: Train the splat

**Code:** [i2g/splat.py](i2g/splat.py), [i2g/splatkit_vendor.py](i2g/splatkit_vendor.py), [i2g/pipeline.py](i2g/pipeline.py) `run_train()`
**Output:** `final\character.ply`, `turntable.mp4`, `preview_8_angles.png`, `photo_vs_render.png`, `report.json`

**1. Load the selected views.** Each one has its image, matte, intrinsics K, measured camera and a
weight: photo 1.0, generated 0.6. The photo uses the **full-resolution** crop, with K scaled to match.

**2. Initialise from the visual hull.**
- Build a 1.2 cm voxel grid around the scaffold, with margin.
- Keep a voxel if it falls inside the (slightly dilated) matte of every view that sees it, allowing
  one view to disagree.
- The **surface voxels** (~17–18k) become the first Gaussians, coloured from the view that faces them
  best.

**3. The training loop** (15,000 steps by default). Each step:
1. Pick a view: the photo 25% of the time, the others share the rest.
2. Render it with **gsplat** (`rasterization`) at half resolution for the first 2,000 steps, then full.
3. Paint a **random background colour** behind both the render and the target. This forces the matte
   edges to be learnt properly.
4. For generated views, apply the view's own **3×3 colour correction** (the photo's is fixed). Those
   views have slightly different lighting and white balance.
5. After step 1,000, apply the view's small **camera correction** (a rotation + translation "delta";
   the photo's is fixed).
6. Compute the loss:
   `0.8·L1 + 0.2·(1−SSIM) + 0.5·|alpha − matte| + 0.2·VGG-perceptual`, times the view's weight, plus
   small penalties that keep the corrections near zero. On large images the perceptual term uses a
   random 1024² crop inside the person.
7. Backpropagate, take an **Adam** step per parameter group (the position learning rate decays
   exponentially), then run **gsplat's `DefaultStrategy`**:
   - **densification** (clone or split Gaussians whose screen-space gradient is high), every 100
     steps from step 500 to 12,000
   - **pruning** (opacity < 0.05, too large)
   - an opacity **reset** every 3,000 steps

**4. Colour model.** Spherical harmonics at **degree 1**: 4 coefficients per colour channel. Higher
degrees overfit the gaps between our sparse views; see §10.

**5. Clean up.** Delete near-transparent Gaussians, and "floaters" that fall outside the matte in more
than 40% of the views that see them.

**6. Export** `character.ply` in the standard INRIA 3DGS layout (§7.4).

**7. Render and measure.** Render the turntable (72 frames at 5° elevation), the 8-angle preview, and
masked PSNR against every training view (photo view ~42–44 dB).

**Main knobs** (`SplatConfig`, changeable with `--set key=value`):

| Setting | Default | Effect |
|---|---|---|
| `iterations` | 15000 | longer = sharper, slower |
| `sh_degree` | 1 | view-dependent colour detail |
| `grow_grad2d` | 0.0002 | lower = more Gaussians (≈400k at this value) |
| `view_weight` / `photo_weight` | 0.6 / 1.0 | trust in generated views vs the photo |
| `perceptual_weight` | 0.2 | sharper texture vs pixel accuracy |
| `alpha_weight` | 0.5 | how strongly silhouettes are enforced |
| `pose_reg`, `pose_lr` | 10, 1e-4 | camera correction; currently too stiff (see quality_plan item 3) |

---

## 6. The orchestration layer (the easy-run tool)

| File | Role |
|---|---|
| [run_image_to_splat.bat](run_image_to_splat.bat) | double-click / drag-and-drop launcher. Runs setup the first time, calls `python3` inside `.venv-i2g`, keeps the window open at the end. |
| [image_to_splat.py](image_to_splat.py) | the command-line tool: arguments, presets, preflight, starting and stopping ComfyUI, progress bars, error messages with fixes, final summary. If you start it with any other Python, it re-launches itself inside `.venv-i2g`. |
| [i2g/pipeline.py](i2g/pipeline.py) | the five stages as functions (`run_ingest` ... `run_train`), folder layout (`RunPaths`), options (`RunOptions`), and **resume logic** |
| [i2g/log.py](i2g/log.py) / [i2g/errors.py](i2g/errors.py) | console + file logging that doesn't break progress bars / `PipelineError(message, hint)` |

**Resume logic ("stamps").** When a stage finishes, it writes `<stage folder>\stage.json` containing:
- a fingerprint of what it was built from: the photo's SHA-256, and the stamps of the stages it
  depends on
- the settings it used

On the next run, a stage is **skipped** if its stamp still matches. Change a training setting and
only training reruns. Use a different photo in the same folder, and the old run is moved to
`_archive\`. **Nothing is ever deleted.** Previous final outputs move to `final\history\<time>\`.

Dependencies: `ingest → body`, `ingest → views`, `body + views → register`, `register → train`.

Typical commands: see the top of `image_to_splat.py`, or run `python3 image_to_splat.py --help`.

---

## 7. Conventions you must get right

Most bugs in 3D code are convention mix-ups. These are the ones this project uses.

### 7.1 Camera axes: OpenCV

Camera space: **x → right, y → down, z → forward** (into the scene). Pixel `(u, v)` has its origin at
the **top-left**:

```
u = fx · X / Z + cx          v = fy · Y / Z + cy          (cx, cy = image centre)
```

SAM 3D Body uses the same convention: principal point at the image centre, vertices + `pred_cam_t`
in camera space.

### 7.2 World: Y up, metres, person faces +Z

- Feet at y = 0; x = the person's **left**.
- Cameras are stored as **world-to-camera 4×4 matrices** (`w2c`): `p_cam = R·p_world + t`.
- The camera centre is `−Rᵀt`.

### 7.3 Angles

- **Azimuth** is measured around the person from the front: az 0 = the photo camera, **az 90 =
  camera on the person's left**, 180 = behind. Elevation is positive when the camera is above the
  body centre.
- **The Multiple-Angles LoRA's "right side view"** turns the camera **to the viewer's right**, so it
  shows the person's *left* side (az ≈ 90). We measured this on real outputs.

### 7.4 PLY layout (standard INRIA 3DGS)

One vertex per Gaussian, all `float32`:

```
x y z  nx ny nz  f_dc_0..2  f_rest_0..44  opacity  scale_0..2  rot_0..3
```

- `f_dc` = spherical harmonics degree 0 (colour = `0.5 + 0.2820948·f_dc`)
- `f_rest` = higher SH bands in **channel-major** order (all red coefficients, then green, then
  blue). We train degree 1 and pad with zeros to degree 3 (45 values) so every viewer accepts it.
- `opacity` is stored **before** the sigmoid (logit); `scale` is stored as **log**; `rot` is a
  quaternion **w, x, y, z**.

### 7.5 Intrinsics

The canvas lens is fixed at a **40° vertical field of view**: `fy = (1248/2) / tan(20°) ≈ 1714 px`.
All generated views share it. The full-resolution photo uses the same K scaled by its size ratio.

---

## 8. Files and data formats

### 8.1 Repository

```
D:\Image-to-Gaussian\
├─ run_image_to_splat.bat        double-click launcher
├─ image_to_splat.py             the command-line tool
├─ plan.md, quality_plan.md, pipeline_breakdown.md
├─ requirements.txt           Python packages for .venv-i2g
├─ i2g\                          the pipeline package
│  ├─ paths.py                   default locations (override with env vars I2G_COMFY_ROOT / I2G_COMFY_URL)
│  ├─ comfy_client.py            ComfyUI HTTP/websocket client + watchdog
│  ├─ comfy_process.py           start / check / stop ComfyUI, install the helper node
│  ├─ matting.py                 BiRefNet wrapper
│  ├─ ingest.py                  Stage 1
│  ├─ body.py                    Stage 2 (+ MHR70 keypoint indices, overlays)
│  ├─ views.py                   Stage 3 (Qwen workflow, prompt tokens, presets)
│  ├─ register.py                Stage 4
│  ├─ cameras.py                 camera maths: projection, look-at, Umeyama, PnP, silhouettes
│  ├─ splat.py                   Stage 5 (training, cleaning, export, rendering)
│  ├─ splatkit_vendor.py         VGG-19 perceptual loss + PLY writer (MIT, from SplatKit)
│  ├─ pipeline.py                stages + resume stamps
│  ├─ vram.py, log.py, errors.py, env_check.py, run.py
├─ comfy_nodes\i2g_export\       our ComfyUI node (junctioned into ComfyUI\custom_nodes)
├─ scripts\                      v2_setup_env / v2_start_comfy / v2_stop_comfy / v2_check / v2_status ... (.ps1)
├─ models\birefnet\, models\perceptual\
├─ .venv-i2g\  .tools\           Python 3.11 environment, uv, standalone CPython, gsplat wheel
└─ input\  output\  logs\
```

### 8.2 One run's working folder

```
output\<name>\v2\
├─ 00_ingest\   canvas.png  alpha.png  source_hr_rgba.png  ingest.json  stage.json
├─ 01_body\     sam3d_photo.npz  body_world.npz  overlay_photo.png  body_summary.json  stage.json
├─ 02_views\    <tag>\s<seed>.png ...  views.json  stage.json
├─ 03_qa\       masks\  sam3d\  sam3d_cache.json  registration.json  contact_sheet.png  stage.json
├─ 06_train\    stage.json
├─ logs\        run_<time>.log (full DEBUG log)  comfyui_<time>.log
├─ _archive\    anything replaced (never deleted)
└─ final\       character.ply  turntable.mp4  preview_8_angles.png  contact_sheet.png
                photo_vs_render.png  report.json  run_summary.json  history\<time>\
```

### 8.3 Key JSON records

- **`views.json`**: one entry per generated image:
  `{tag, azimuth, elevation, distance, prompt, seed, path, seconds}`
- **`registration.json`**: one entry per image, the photo first. Key fields:
  - `w2c` (4×4 camera), `azimuth_deg`, `elevation_deg`, `radius_m`
  - `drift_m`, `reproj_rmse_px`, `coverage`, `score`
  - `lr_swapped`, `init` (which hypothesis won)
  - `accepted`, `reasons`, `selected`
- **`report.json`**: training config, per-view PSNR, Gaussian count, camera corrections, timing and
  peak VRAM.

---

## 9. Rebuilding from scratch, step by step

Build and **check** each step before moving on. Every step has a visible result.

1. **GPU environment.**
   - Install the NVIDIA driver.
   - Install **ComfyUI portable** (≥ 0.35) and start it once.
   - In the Control Panel, set *CUDA – Sysmem Fallback Policy* → **Prefer No Sysmem Fallback**, so a
     VRAM overflow fails fast instead of crawling (§10).
   - ✅ ComfyUI opens at `http://127.0.0.1:8188`.
2. **Models.** Download the files in §4.4 into the listed folders.
   - ✅ In the ComfyUI UI, the loaders list each file.
3. **Python environment.** Create a Python **3.11** venv. Install torch 2.8.0 + CUDA 12.8 and
   `requirements.txt`, then the gsplat 1.4.0 wheel. `scripts\v2_setup_env.ps1` does all of it
   with uv.
   - ✅ `python -c "import torch, gsplat; print(torch.cuda.is_available())"` prints `True`.
4. **ComfyUI client.** Write `upload_image`, `queue`, `wait`, `download` and `free`. Test with a
   `LoadImage → ImageScale → SaveImage` workflow.
   - ✅ The resized image comes back (`i2g/env_check.py` does this).
5. **Helper node.** Write `I2G_SaveMHRPose`, which flattens SAM 3D Body's dict into an `.npz`, and
   junction it into `custom_nodes`.
   - ✅ `/object_info/I2G_SaveMHRPose` exists after a ComfyUI restart.
6. **Stage 1.** Load BiRefNet and build the canvas.
   - ✅ `canvas.png` shows a clean cut-out on grey, with no clipped hair or feet.
7. **Stage 2.** Run SAM 3D Body, build the world frame, draw the overlay.
   - ✅ The mesh sits on the person in `overlay_photo.png`, with coverage > 0.9 and a plausible height.
8. **Stage 3.** Send one Qwen workflow for `back view`.
   - ✅ A believable back view in the same pose.
   - Then generate all angles and look at them.
9. **Stage 4.** Implement projection, Umeyama, PnP refinement, then the multi-hypothesis scoring,
   the gates and the contact sheet.
   - ✅ Back views register near 180°, sides near 90°/270°, and the skeleton overlays line up.
   - Test a *front-looking* back view on purpose; it must not register at 0°.
10. **Stage 5.**
    - Get gsplat rendering random Gaussians first. ✅ An image appears, and gradients flow.
    - Then add the visual hull, the loop and the losses. ✅ Loss falls; `photo_vs_render.png` matches.
    - Then densification, cleanup and export. ✅ `character.ply` opens in SuperSplat.
11. **Orchestration.** Stamps, CLI, `.bat`.
    - ✅ Rerunning skips everything; changing `--set iterations=...` reruns only training.

**Reading order for the code:** `paths.py` → `cameras.py` → `comfy_client.py` → `ingest.py` →
`body.py` → `views.py` → `register.py` → `splat.py` → `pipeline.py` → `image_to_splat.py`.

---

## 10. Lessons learned the hard way

| Problem | What happened | What we do now |
|---|---|---|
| **4DAnyone didn't fit** | The first approach generated **121 frames per camera angle** (a video model) for a *still* photo. That peaked at ~22 GiB on a Linux GPU with no display attached. On this Windows PC it crashed under Docker/WSL2 (`device not ready`), and on native Windows it ran **20× slower** (554 s per step). | Generate **one image per angle**. Keep every stage well under the VRAM limit. |
| **Silent VRAM spill** | By default Windows lets CUDA overflow into system RAM. Nothing crashes; everything crawls. | Set the NVIDIA *Sysmem Fallback Policy* to *Prefer No Sysmem Fallback*. The watchdog aborts steps over 2 min. `/free` before our own GPU work. |
| **SageAttention broken** | ComfyUI's embedded Python 3.13 had **Python 3.12 header files**. Also, the machine-wide `CUDA_PATH`/`CUDA_HOME` pointed at a **conda env with Python 3.8 headers**. Triton's compiler used the wrong headers and failed on every call. | Headers replaced with 3.13. ComfyUI is launched with `CUDA_PATH`/`CUDA_HOME` pointing at Triton's bundled CUDA. |
| **Front/back confusion** | With hands on the head, a back view with left/right swapped looks like a front view in 2D. SAM 3D Body sometimes reads backs as fronts. | Multi-hypothesis registration + a weak prior toward the requested angle (§5 Stage 4). |
| **The LoRA isn't obedient** | "Right side view" means camera to the *viewer's* right. "Elevated shot" barely elevates front/back views. Sometimes the other side is drawn. | **Measure** every camera; never trust the prompt. |
| **Densification too timid** | gsplat's usual threshold for this mode (0.0008) gave only 37k Gaussians for a metre-scale person. | 0.0002 → ~400k. |
| **Colour streaks** | Spherical harmonics degree 3 overfit view-dependent colour between sparse views. | Degree 1. |
| **More Gaussians ≠ better** | 400k looked sharper, but novel angles (e.g. the face slightly above the photo) got worse. | Regularisation, not raw capacity (quality_plan items 1, 2, 6). |
| **Windows quirks** | `python3` is a Microsoft Store placeholder. `cmd /c` strips quotes. PowerShell 5.1 has no `&&`. The HDD makes 20 GB model loads take minutes. | `python3.exe` lives inside the venv; the `.bat` handles quoting and drag-and-drop; the tool re-launches itself into the venv. |

---

## 11. Performance on the reference PC

RTX 4090 24 GB, 32 GB RAM, models on an HDD.

| Step | Draft preset | Standard preset | Peak VRAM |
|---|---|---|---|
| ComfyUI start (if not running) | 3–8 min | 3–8 min | ~1 GB |
| Stage 1: cut-out | 1 min | 1 min | ~3 GB |
| Stage 2: body | 1 min | 1 min | ~5 GB |
| Stage 3: views | 10 min (7 images) | 14 min (22 images) | ~22 GB (ComfyUI offloads) |
| Stage 4: register | 1 min | 3 min | ~5 GB |
| Stage 5: train | 4 min (6k steps) | 14 min (15k steps) | 2–3.5 GB |
| **Total** | **~18 min** | **~35 min** | |

Results on the test image:
- **Draft:** 148k Gaussians, 35 MB PLY, photo-view PSNR 43.8 dB.
- **Standard (run 2):** 407k Gaussians, 96 MB.

---

## 12. Glossary

| Term | Meaning |
|---|---|
| **3DGS / splat** | 3D Gaussian Splatting: a scene made of many small 3D Gaussian blobs |
| **alpha / matte** | per-pixel "how much is this the person" (1 = person, 0 = background) |
| **canvas** | our fixed 832×1248 grey-background image of the person |
| **densification** | adding Gaussians where the image is poorly explained (clone or split) |
| **extrinsics / w2c** | camera position and orientation (world → camera) |
| **fp8 / GGUF** | compressed model-weight formats that make big models fit in VRAM |
| **intrinsics / K** | camera lens parameters (focal length, image centre) |
| **LoRA** | small add-on weights that teach a model a new skill |
| **MHR / MHR70** | Meta's body model / its 70-keypoint layout |
| **novel view** | an image of the subject from an angle nobody photographed |
| **PnP** | solving camera pose from 3D↔2D point matches |
| **PSNR** | pixel accuracy in dB (higher is better; ~40+ is near-identical) |
| **registration** | measuring an image's camera pose |
| **scaffold** | our 3D body from SAM 3D Body, used as the geometric reference |
| **SH (spherical harmonics)** | how a Gaussian's colour can change with viewing direction |
| **SSIM / perceptual loss** | image-similarity measures based on structure / on neural-network features |
| **stamp** | `stage.json`: records what a stage was built from, for resume |
| **visual hull** | 3D shape carved from silhouettes seen by several cameras |
| **VRAM spill / sysmem fallback** | GPU memory silently overflowing into system RAM (very slow) |

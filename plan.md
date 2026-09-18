# Image-to-Gaussian v2: plan

**Goal:** one front-facing, full-body photo → one **static** 3D Gaussian splat of that person
(`output/<name>/final/character.ply`). Not 4D, not animation.

**Status (2026-09-18):** **M0–M3 done. First end-to-end run on the test image produced a complete
360° `character.ply`.** v2 replaces the blocked 4DAnyone path documented in
[docs/STATUS.md](docs/STATUS.md).

| Milestone | Result on `test-assets/test-image.png` |
|---|---|
| M0 environment | `.venv-i2g`: torch 2.8.0+cu128, gsplat 1.4.0 (145 ms fwd+bwd, 1024×1536, 200k Gaussians). ComfyUI API round trip 1.9 s; all M1–M4 nodes present. |
| M1 body scaffold | SAM 3D Body fit: 98% of the body silhouette inside the photo matte, height 1.72 m, camera 2.77 m away |
| M2 anchor views | 22 Qwen views (11 angles × 2 seeds), about 14 min. All 22 registered; 10 selected + the photo; drift 24–43 mm |
| M3 first splat | run 1: 37k Gaussians, 8.7 MB. run 2: 407k Gaussians, 96 MB, photo trained at 1866×2799, 14 min, 3.4 GiB peak VRAM |

- **Outputs:** `output/test-image/v2/final/` (latest) and `output/test-image/v2/runs/run{1,2}/`, each with
  `character.ply`, `turntable.mp4`, `preview_8_angles.png` and `report.json`.
- **Quality:** the front is near-exact at the photo's camera (41–43 dB masked PSNR); the back is
  plausible and consistent.
- **Known artifacts:**
  - streaks behind the calves at the pure side views (no training view between 74° and 125°)
  - the face degrades about 10° off the photo's elevation (the generated front-ish views disagree on
    facial features)

**Learned on this machine:**

- **SageAttention was broken; it's fixed (2026-09-18).** There were two causes.
  1. `python_embeded\Include` and `libs` held Python 3.12.7 headers and libs (from an old Sage
     install) under a Python 3.13.9 runtime. They're replaced with 3.13; the old ones are kept as
     `*_3.12.7_backup`.
  2. Machine-wide `CUDA_PATH` and `CUDA_HOME` point at `miniconda3\envs\nerfstudio`, whose `include`
     has Python 3.8 headers, so Triton compiled against them.

  `run_nvidia_gpu.bat` (backup: `.bak-20260918`) and `v2_start_comfy.ps1` now point both variables
  at Triton's bundled CUDA, for the ComfyUI process only. Sage now matches SDPA (cosine 0.9993) and
  attention is 3.1× faster. The v2 pipeline keeps it opt-in (`-UseSageAttention`) until tested on Qwen.
- **The sysmem fallback is still enabled** for both interpreters (D2 is not in effect yet).
- **Qwen timing:** about 57 s per new prompt, because the text encoder and the 19.6 GB image model
  swap through VRAM; about 6 s per extra seed.
- **The LoRA's "right" = camera to the viewer's right**, i.e. the person's left, az +90.
  "Elevated shot" is reliable for side views, weak for front/back.
- **Qwen sometimes renders the opposite side** (e.g. `fl s101`). Registration measures it anyway.
- **Hands-on-head is nearly bilaterally symmetric**, so the body fitter confuses front and back.
  `register.py` handles this with multi-hypothesis PnP plus a weak prior toward the requested angle.
- **Densification:** gsplat's `grow_grad2d` 0.0008 with absgrad under-densifies a metric-scale
  person (37k Gaussians); 0.0002 gives 400k.

**How it works:** [pipeline_breakdown.md](pipeline_breakdown.md) is a beginner's guide covering every
stage, tool and library, plus how to rebuild the project from scratch.

**Next:** the ranked quality roadmap is in [quality_plan.md](quality_plan.md). Sprint 1 covers items
0–4: the measurement harness, anti-streak, photo-owned face, real camera refinement, and better
coverage.

**Commands:**

- **Easy run:** double-click `run_image_to_splat.bat`, or drag a photo onto it, or run
  `python3 image_to_splat.py --input <photo> [--output_dir <dir>] [--preset draft|standard|high]`
  (`--help` lists every option). The tool:
  - starts and stops ComfyUI itself
  - reruns only the stages whose inputs or settings changed
  - archives old results instead of deleting them
- `scripts\v2_status.ps1`: progress and a stuck check
- `scripts\v2_check.ps1`: environment smoke tests
- Lower-level: `scripts\v2_start_comfy.ps1` / `scripts\v2_run.ps1`

---

## Can this work on my PC?

**Yes.** Honest odds:

| Outcome | Confidence |
|---|---|
| A complete 360° static splat of the person comes out of this pipeline on this PC: front faithful to the photo, sides and back plausible | **~90%** |
| It looks *good* from every angle (sharp, believable back, no big smears) after the tuning in M3–M5 | **~70%** |
| The hardware turns out to be the blocker | **< 5%** |

**What the remaining risk actually is.** None of it is hardware:

- Generated views can disagree on small details (back of the hair, jeans pockets), which blurs the
  splat.
- A full-body shot gives the face only about 100 px.
- The test pose, with hands behind the head, hides the armpits and the back of the head.

These are quality problems that get measured (hold-out LPIPS, §7) and worked down.

**Is the system too weak? No.**

- The RTX 4090's **24 GB VRAM** covers every stage; the largest peaks at about 20–21 GB with fp8
  weights.
- The real constraints cost *time*, not feasibility:
  - the 4090 also drives the 3440×1440 screen, so about 1 GB is never free and the GPU isn't
    exclusive
  - **32 GB system RAM** (a different memory from VRAM) is tight when ComfyUI offloads 20B models
  - the models sit on an HDD, so a cold load takes about 2 minutes

**Why it didn't work before.** 4DAnyone is a 4D (video) model:

- For a still photo it still generates **121 frames per camera angle**, which is 120 duplicates of
  the same picture. Most of its memory went into a time dimension a static splat doesn't need.
- Its official "≤24 GB" mode peaks at about **22.2 GiB on a Linux GPU that isn't driving a
  display**. This 4090 has about 23 GB left after Windows and the monitor, so there was no margin.
- **Docker/WSL2:** the allocation failed with `device not ready`.
- **SplatKit on native Windows:** the driver spilled VRAM into system RAM, so it ran at 554 s per
  step, about 20× slower. It never finished.

v2 generates one image per camera angle; the optional orbit stage's video frames are themselves
camera angles, so nothing is wasted on time.

---

## 0. Start here

### 0.1 The method on one screen

4DAnyone spends its memory on **time**: 121 frames × N views through one video DiT. That's why it
needs about 22 GiB on an exclusive Linux GPU, and why it failed on your PC. It crashed with `device not ready` in
Docker/WSL2 and crawled at 554 s/it in SplatKit on native Windows. **A static person has no time
axis**, so v2 spends the whole budget on *views*:

1. **Body scaffold.** SAM 3D Body (already in your ComfyUI) fits a 3D body to the photo. That fixes
   a world frame in metres and an exact camera rig around the person. This is 4DAnyone's "accuracy
   over density" skeleton idea, made explicit.
2. **Anchor views.** Qwen-Image-Edit-2511 + the Multiple-Angles LoRA (already downloaded) renders the
   person from 11–15 known angles. The farthest angle comes first (back, then sides, then diagonals,
   then elevated), and each new view can see the photo plus its nearest accepted neighbours. This is
   4DAnyone's *Reference Context Packing*, done with an image-edit model.
3. **Register, then gate.** Each generated image is re-fitted with SAM 3D Body and snapped to the
   scaffold (3D alignment, then 2D keypoint PnP). We use the camera the image was *actually*
   rendered from, not the one we asked for. Automatic checks reject pose, identity and silhouette
   drift. **You approve a contact sheet before anything else runs.**
4. **Dense orbit (optional).** Wan 2.2 first→last-frame clips between accepted anchors fill the
   gaps. For a static subject the video's time axis *is* the camera axis, and the anchors close the
   loop.
5. **Reconstruct.** A native-Windows gsplat trainer (forked from SplatKit's MIT trainer) fits the
   splat from known cameras, then exports `character.ply`. It adds a matte loss, camera refinement,
   per-view colour correction and per-view confidence weights, and weights the photo highest.

Every stage is budgeted at **≤ ~20 GB VRAM**, runs **natively on Windows (no Docker/WSL)**, and caches
its outputs so a run can resume.

### 0.2 Your questions, answered

**Use ComfyUI? Yes, as the generation engine only.** Your portable ComfyUI (v0.35.2, torch 2.13 +
CUDA 13.0, SageAttention) already runs 14–22B models natively on this PC. It also ships native nodes
for what v2 needs:

- SAM 3D Body: predict, and re-render from any 6-DOF camera
- Depth Anything 3: multi-view camera poses
- Qwen-Image-Edit, FLUX.2 klein and Wan 2.2

v2's Python scripts drive ComfyUI headlessly over its HTTP API on `127.0.0.1:8188` (`/upload/image`,
`/prompt`, `/history`, `/view`, `/free`), and the workflow JSONs are versioned in this repo. Matting,
registration, QA, frameset writing and splat training are plain Python in this repo's own venv. That
way ComfyUI updates can't break training, and the trainer never shares the GPU with a loaded
diffusion model.

**ComfyUI MCP? Not needed.** The HTTP API covers everything an MCP would, and the pipeline has to run
from one PowerShell command without me in the loop. An MCP you already have is harmless for
interactive debugging, but nothing here depends on it.

**FLUX.2 klein 9B or Qwen-Image-Edit-2511?** For this job, the deciding property is **camera
control**, not looks.

- Qwen-Image-Edit-2511 has the fal Multiple-Angles LoRA, already in your `loras/`. It was trained on
  3,000+ Gaussian-splat renders with 96 discrete poses (8 azimuths × 4 elevations × 3 distances).
- Nothing in your library does that for klein. `Flux-2-Multi-Angles-LoRA-v2` targets FLUX.2-dev
  (32B), which is too big for 24 GB VRAM + 32 GB RAM.

So **Qwen generates the angle-controlled anchors**, and **klein 9B is an optional low-denoise
detail/identity pass and a bake-off contender** (§7). If klein wins on measured angle accuracy,
identity and hold-out consistency, it becomes the default. Registration measures every image's
real camera, so the pipeline doesn't care which model made it.

**MiniMax H3 or Wan 2.2?** H3 (Aug 2026) is the stronger general video model. It has the best
reference system (up to 9 images) and a first/last-frame mode. For this pipeline, though:

1. Its license **excludes local use in the US, EU, UK and South Korea**. It also forbids using
   outputs "to improve any other artificial intelligence model".
2. Local weights are 768p.
3. It's heavy for 32 GB RAM: int8 DiT 19.5 GB + Qwen3-VL-32B nvfp4 text encoder 14.6 GB.
4. It has no orbit or camera LoRAs.

Wan 2.2 has orbit LoRAs and first/last-frame. It also has Fun-Control, which can follow the
scaffold's exact camera path. You already have its Lightning LoRAs, and it's Apache-2.0. **Wan 2.2 is
the default for the optional orbit stage. H3 is a contender only if its license covers you (D5).**

**Why SplatKit stalled (your photo).** The log shows `RCP 1-to-4: 1/4 [09:14 … 554.20 s/it]` while
running 4DAnyone at 121 frames × 16 views. That working set is bigger than what a 24 GB card
driving your desktop has free. On native Windows the CUDA *sysmem fallback* silently pages VRAM into
system RAM, so the job doesn't crash; it runs about 20× slower. The published 4090 figure is
~27 s per generated view (see docs/STATUS.md). The root cause is the same as the Docker
`device not ready` crash; only the symptom differs. v2 avoids it by budget (≤ ~20 GB per stage).
It also turns fallback off (D2), so an overflow fails in seconds instead of hours.

### 0.3 Decisions I need from you

| # | Decision | My recommendation | Why |
|---|---|---|---|
| D1 | Use ComfyUI with fp8/int8/GGUF weights for generation. This reverses the "no quantization / offload / ComfyUI" row in docs/STATUS.md. | **✅ Approved 2026-09-18** | 14–20B models don't fit 24 GB at bf16; your ComfyUI already runs them. |
| D2 | NVIDIA Control Panel → Manage 3D settings → **Global Settings** → **CUDA – Sysmem Fallback Policy: Prefer No Sysmem Fallback**. Global rather than per-program: each venv runs its own `python.exe`. It only affects CUDA apps, not games. | **✅ Set by you 2026-09-18.** M0 verifies it with an over-allocation test. | An overflow becomes an immediate out-of-memory error instead of a 20× crawl. |
| D3 | Free **system RAM** (32 GB, separate from the 4090's 24 GB VRAM) during runs. `.wslconfig` lets WSL/Docker take up to `memory=48GB` of system RAM. | Quit Docker Desktop while v2 runs (v2 doesn't use it). Optionally lower the cap to 12GB. | RAM is the tightest resource when ComfyUI offloads model weights. |
| D4 | Plug the monitor into the motherboard's DP/HDMI (the Ryzen iGPU), so the 4090 does compute only. | Later, and only if we're VRAM-bound | Frees about 1 GB VRAM and moves desktop compositing off the 4090. Needs the iGPU enabled in BIOS plus the AMD driver; it currently shows as "Microsoft Basic Display Adapter – Error". |
| D5 | May MiniMax H3 be used? | Only if its territory clause permits you | License excludes the US/EU/UK/South Korea, and outputs can't be used to improve other AI models. |
| D6 | Personal or commercial use? | **✅ Personal use (2026-09-18)** | Non-commercial licenses (FLUX.2 klein 9B, InsightFace, DA3 Large/Giant) are fine. |
| D7 | Run `git init` on this repo (it isn't one yet). | **Yes** | Tracks workflow JSONs, configs and experiment reports. |

### 0.4 Models: what you already have, and what to download

**Already on disk.** Paths are under
`D:\New_ComfyUI\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\models\` unless
noted.

| Role | Files | License |
|---|---|---|
| Body scaffold | `detection/sam_3d_body_dinov3_bf16.safetensors` | SAM License |
| Angle views | `diffusion_models/qwen_image_edit_2511_bf16.safetensors`, `loras/qwen-image-edit-2511-multiple-angles-lora.safetensors`, `loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors`, `clip/qwen_2.5_vl_7b_fp8_scaled.safetensors`, `vae/qwen_image_vae.safetensors` | Apache-2.0 |
| Detail pass / contender | `diffusion_models/flux-2-klein-9b-fp8.safetensors`, `text_encoders/qwen_3_8b_fp8mixed.safetensors`, `vae/flux2-vae.safetensors`, `loras/Flux2-Klein-9B-consistency-V2.safetensors` | FLUX non-commercial (9B) |
| Orbit (optional) | `diffusion_models/Wan2.2-I2V-A14B-{High,Low}Noise-Q8_0.gguf`, `loras/wan2.2_i2v_lightx2v_4steps_lora_v1_{high,low}_noise.safetensors`, `diffusion_models/Wan2_2-Fun-Control-A14B-{HIGH,LOW}_fp8_e4m3fn_scaled_KJ_fixed.safetensors`, `text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors`, `vae/wan_2.1_vae.safetensors` | Apache-2.0 |
| Orbit contender (D5) | `diffusion_models/minimax_h3_{fl2va,ref2va}_pruned_int8_convrot.safetensors`, `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`, `vae/minimax_h3_video_vae_fp16.safetensors` | MiniMax H3 Community |
| 2D keypoints | `onnx/vitpose-l-wholebody.onnx` | Apache-2.0 |
| Identity QA | `insightface/models/{antelopev2,buffalo_l}` | non-commercial |
| Matting | `splatkit/birefnet/` (also `D:\Image-to-Gaussian\models\birefnet\`) | MIT |
| Perceptual loss | `D:\Image-to-Gaussian\models\perceptual\imagenet-vgg-verydeep-19-conv.safetensors` | as upstream |

**To download** (none of these needs a license click; none is needed before M2):

| Priority | What | From | Size | Put in | Why |
|---|---|---|---|---|---|
| Recommended for M2 | `qwen_image_edit_2511_fp8mixed.safetensors` | Hugging Face `Comfy-Org/Qwen-Image-Edit_ComfyUI` → `split_files/diffusion_models/` | 20.5 GB | `models/diffusion_models/` | Your copy is the 40.9 GB bf16. That's bigger than your RAM, so every load pages from the D: HDD. |
| Optional for M4 | `wan22_i2v_14b_orbit_shot_lora` | `ostris/wan22_i2v_14b_orbit_shot_lora` | small | `models/loras/` | Single-pass 360° orbit option (trigger `orbit 360`, Apache-2.0) |
| Optional for M4 | 360Rotation-Redmond high + low noise | `artificialguybr/360Rotation-Redmond-WAN2-I2V-14B` | small | `models/loras/` | Turntable alternative (CC0) |
| Optional for M4 | Depth Anything 3 Base | `depth-anything/DA3-BASE` | ~0.5 GB | via ComfyUI's native DA3 loader | Cross-checks orbit-frame poses (Apache-2.0) |

v2 does **not** need SMPL-X, GVHMR or the 4DAnyone weights.

---

## 1. Why the old path failed, and the rules that follow

- 4DAnyone generates N views × 121 frames with a 5B Wan2.2 DiT. Its own ≤24 GiB setting peaks at
  about 22.18 GiB *allocated* on exclusive Linux.
- On this PC the 4090 also drives a 3440×1440 WDDM desktop, using about 0.7 GB at idle, and other
  apps share it.
- **Docker Desktop / WSL2:** near the ceiling, runs died with `CUDA driver error: device not ready`
  (docs/STATUS.md). NVIDIA documents that WSL2 limits pinned system memory, which offload-heavy
  workloads depend on.
- **SplatKit on native Windows:** the same workload page-faulted into system RAM through the
  driver's sysmem fallback. It ran at 554 s/it and never finished.

**Rules for v2**

1. Budget every stage at ≤ ~20 GB peak. No stage may "just fit".
2. Generation runs where memory management is proven on this machine: native-Windows ComfyUI with
   fp8/int8/GGUF weights and ComfyUI's own partial offloading.
3. Fail fast: sysmem fallback off (D2), plus a step-time watchdog (§5).
4. One GPU user at a time. Call ComfyUI's `/free` before any other GPU stage.
5. No Docker, no WSL and no COLMAP in the main path.

## 2. This PC (verified 2026-09-18)

| Item | Value |
|---|---|
| GPU | RTX 4090 24 GB, WDDM, **drives the 3440×1440 display**; ~0.7 GB used at idle; driver 596.21 (CUDA 13.2) |
| CPU / board | Ryzen 9 7900X (has an iGPU) / Gigabyte B650 UD AX-Y1 (rear DP 1.4 + HDMI 2.1 for the iGPU) |
| RAM | 32 GB DDR5-4800 (2×16); 55 GB pagefile on C: |
| Disks | C: WD_BLACK SN770 2 TB NVMe (~579 GB free). D: Seagate 8 TB SMR HDD (~2.36 TB free) |
| Toolchains | CUDA 11.8 / 12.1 / 12.8; VS Build Tools 2022 + 2026; host Python 3.14 |
| ComfyUI | Portable v0.35.2 at `D:\New_ComfyUI\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable`: Python 3.13.9, torch 2.13.0+cu130, SageAttention + Triton, launched with `--use-sage-attention --fast` |
| Splat backend | SplatKit venv: Python 3.11, torch 2.8.0+cu128, gsplat 1.4.0 Windows wheel; its CUDA smoke test passed 2026-09-18 |
| WSL | `Ubuntu` + `docker-desktop` distros; `.wslconfig` `memory=48GB` |

Models sit on the D: HDD (~150–200 MB/s), so a 20 GB checkpoint takes about 2 minutes to load cold.
That's acceptable. If it becomes the bottleneck, an `extra_model_paths.yaml` entry can point ComfyUI
at a model cache on C: (your call).

## 3. What we borrow from 4DAnyone, and what's new

| 4DAnyone idea (paper §3) | v2's static, single-image version |
|---|---|
| 3D-aware, depth-buffered skeleton conditioning; "accuracy over density" | An explicit 3D body scaffold from SAM 3D Body. Per-camera renders (occlusion-aware 3D OpenPose, normals, depth, silhouette) serve as optional control, and the scaffold is the **registration target** for every generated view. |
| Reference Context Packing: a fixed-size reference context, farthest-point view order | Farthest-first anchor order (back → sides → diagonals → elevated). Each view is conditioned on the photo plus ≤ 2 accepted nearest anchors, using multi-image input in Qwen/klein. |
| Target Context Routing: views denoised jointly in groups | No time axis, so a single first→last-frame clip jointly denoises 33–81 views in one pass. An experimental "turnaround sheet" mode denoises 4 views in one image canvas. |
| Prescribed camera rig, no COLMAP | Prescribed rig, then **measured** per-image cameras (3D alignment + keypoint PnP) and in-training camera refinement. Still no COLMAP. |
| FreeTimeGS initialised by space carving from masks | Visual hull from the registered mattes, plus samples on the scaffold surface |
| Flow loss + LPIPS/VGG perceptual objective | L1 + SSIM + VGG-19 perceptual (SplatKit defaults) + matte loss + per-view colour correction |
| Quality gate before dense generation | The same human gate, now backed by automatic per-view QA metrics |
| Known failure modes: loose garments, HMR pose errors | The scaffold overlay is checked at M1. QA flags silhouette overflow, which gets more anchors and lower orbit weight. |

**New in v2**

- **Body-anchored registration.** Any generator can feed the reconstruction, including Qwen, klein,
  H3, Wan and even cloud tools like Krea. Its camera control never has to be trusted.
- **Automatic per-view QA with regenerate-on-fail** before the human gate.
- **The photo is the only ground truth.** It gets the highest loss weight, a frozen camera and frozen
  colour.

## 4. Pipeline

```
input/<name>.png
  │
  S0 ingest ────── BiRefNet matte → canvas 1024×1536 on neutral grey (+ full-res photo crop)      [v2 venv]
  │
  S1 scaffold ──── SAM 3D Body → MHR mesh + 70 keypoints + focal → world frame + camera rig        [ComfyUI]
  │                → per-camera renders: 3D-OpenPose / normals / depth / silhouette                  [v2 venv]
  │
  S2 anchors ───── Qwen-Image-Edit-2511 + Multiple-Angles LoRA (+ Lightning), farthest-first,      [ComfyUI]
  │                k candidates per camera; optional FLUX.2 klein 9B detail pass
  │
  S3 register+QA ─ SAM 3D Body per candidate → 3D alignment → keypoint PnP → quality gates         [ComfyUI + venv]
  │                → contact_sheet.png ◀── HUMAN GATE
  │
  S4 orbit (opt.) ─ Wan 2.2 first→last-frame between anchors (or Fun-Control on scaffold renders)   [ComfyUI]
  │                → per-frame registration + smooth orbit fit
  │
  S5 reconstruct ─ frameset (transforms.json + masks + visual hull) → gsplat trainer               [splat venv]
  │
  S6 export ───── final/character.ply + turntable.mp4 + report.md
```

### S0: Ingest (v2 venv, ~1 min, ~3 GB)

- BiRefNet matte (weights already local). Crop to a tight box with an 8% margin, then composite on
  neutral grey (0.5) to make `canvas.png` at **1024×1536** (2:3, subject about 85% of the height).
  Also keep a full-resolution crop and alpha of the photo so it can be trained at native resolution.
- A plain grey background stops edit models inventing scenery and makes every generated view easy
  to matte.
- **Output:** `00_ingest/{canvas.png, alpha.png, source_hr.png, source_hr_alpha.png, ingest.json}`
- **Done when** the matte is clean at the hair edges and nothing is clipped.

### S1: Body scaffold (ComfyUI native SAM 3D Body, ~1 min, ~5 GB)

- Optionally estimate the photo's field of view with MoGe (a native ComfyUI node) and feed it to
  `SAM3DBody_Predict`'s `fov` input.
- A small custom node, `comfy_nodes/i2g_export` (junctioned into ComfyUI's `custom_nodes`), writes
  the `MHR_POSE_DATA` to `body.npz`: vertices, faces, 70 3D/2D keypoints, focal length and `cam_t`.
- **World frame:** origin between the feet on the floor, +Y up (feet → neck, orthogonalised), +Z
  toward the photo camera, units in metres.
- **Camera rig** (`rig.json`): shared intrinsics (the canvas field of view), looking at the body
  centre, radius equal to the photo camera's distance.
  - Ring A: the photo's elevation, azimuth every 45° (7 new cameras + the photo at 0°)
  - Ring B: +30°, azimuth 0/90/180/270 (top of head, shoulders, hair)
  - Ring C (optional): −20°, azimuth 0/180 (under chin and arms; your test pose needs it)
  - Face ring (S6): close-ups at −45/0/+45°
- **Renders per camera:** occlusion-aware 3D OpenPose, normals, depth, silhouette. These come from
  our renderer (SplatKit's MIT pure-torch/Triton rasterizer, or trimesh). They're cross-checked once
  against ComfyUI's `SAM3DBody_Render`, which accepts the same camera through `camera_info`.
- **Done when** `overlay_input.png` shows the mesh on the photo with keypoint reprojection error
  under ~1% of image height, and the hands-on-head pose is correct.

### S2: Anchor views (ComfyUI, ~20–40 s per candidate with Lightning, ≤ ~21 GB via ComfyUI offload)

- **Default generator:** Qwen-Image-Edit-2511 (fp8mixed) + Multiple-Angles LoRA (strength 0.8–1.0)
  + Lightning 4-step. The bake-off compares it against 20+ steps without Lightning.
- **Prompt:** `<sks> {azimuth} {elevation} {distance}` plus a short fixed description, for example
  "same person, same pose, arms raised with hands on head, plain grey studio background, full body
  head to toe".
- **Order**, farthest-first as in RCP: 180° → 90°/270° → 45°/135°/225°/315° → ring B → ring C.
- **Reference modes** (bake-off):
  - `single`: photo only, which is how the LoRA was trained
  - `nearest2`: photo + ≤ 2 accepted nearest anchors, through Qwen's multi-image input
  - `sheet` (experimental): a 4-panel turnaround in one canvas, denoised jointly like a 4DAnyone
    view group
- **k = 3** candidates per camera (different seeds); S3 picks one.
- **Optional klein 9B detail pass:** [photo, candidate] as references at low denoise (≈0.2–0.35).
  It must not move geometry, and S3 re-checks that.
- **Output:** `02_views/<cam>/cand_<k>.png` + `gen.json` (model, LoRAs, prompt, seed, steps)

### S3: Register, QA, human gate (~2–5 min, ≤ ~6 GB)

For each candidate:

1. Matte it with BiRefNet.
2. Run SAM 3D Body on the candidate. A similarity alignment of its 3D keypoints to the scaffold gives
   a coarse camera, which settles front/back/left/right without ambiguity, plus a pose-drift
   residual (PA-MPJPE).
3. Detect 2D keypoints with ViTPose-L wholebody. Refine with PnP (shared intrinsics, RANSAC +
   Levenberg–Marquardt), seeded from step 2. That gives the final camera and a reprojection RMSE.
4. Apply the gates below. The thresholds are starting values, tuned on the test image in M2.

| Gate | Starting threshold | Catches |
|---|---|---|
| Pose drift (PA-MPJPE; wrists and elbows checked explicitly) | ≤ ~70 mm | arms dropped, pose changed |
| Keypoint reprojection RMSE | ≤ 1.5% of image height | anatomy that fits no single camera |
| Azimuth error vs the requested angle | ≤ 25° | mirrored or mislabelled views (the solved camera is still used) |
| Scaffold silhouette covered by the matte | ≥ 0.9 | cropped or shrunken body |
| Identity (ArcFace cosine vs the photo, when the face is visible) | ≥ ~0.4 | a different person |
| Sharpness (variance of Laplacian inside the matte) | ≥ 0.6 × the photo | mushy views |

5. Keep the best passing candidate per camera. Regenerate failures up to 3 rounds, then report
   them.

- **Output:** `03_qa/qa_report.json`, `accepted/`, `masks/`, and `contact_sheet.png`. The sheet is
  ordered by solved azimuth, with the scaffold skeleton overlaid and metrics under each tile.
- **HUMAN GATE:** you approve the sheet before S4/S5 run.

### S4: Dense orbit, optional (ComfyUI, ~2–5 min per clip at 480×832 with Lightning, ≤ ~20 GB)

- **Default:** Wan 2.2 I2V A14B first→last-frame between consecutive accepted ring-A anchors
  (0→45→…→315→0), 8 clips of about 33 frames. The endpoints are anchors, so the orbit closes by
  construction.
- **Bake-off alternatives:**
  - Wan 2.2 Fun-Control driven by S1's 3D-OpenPose renders along the exact rig path (cameras known
    by construction)
  - Wan 2.2 + an orbit LoRA, one 81-frame 360° pass
  - MiniMax H3 first/last-frame, if D5 allows
- **Registration:** S3's keypoint solve per frame, then a smooth orbit fit (constant radius and
  elevation, monotonic azimuth spline). Depth Anything 3 multi-view poses serve as an optional
  cross-check.
- Keep every 2nd–3rd frame, with loss weight 0.2–0.4.

### S5: Reconstruct (splat venv, ~15–30 min, 4–10 GB)

- **Frameset:** nerfstudio-style `transforms.json` with **per-frame intrinsics** (photo at native
  resolution, anchors at 1024×1536, orbit frames at theirs), plus `masks/`, `sparse_pcd.ply` and
  `weights.json`. This is a superset of SplatKit's format.
- **Initialisation:** a visual hull carved from the registered mattes (4DAnyone's FreeTimeGS init),
  plus samples on the scaffold surface, coloured from the photo and anchors.
- **Trainer:** a fork of SplatKit's `core/splatting/training` (MIT, with attribution) on gsplat 1.4.0.
  It keeps the Splatfacto-parity defaults: L1 + 0.2 SSIM + 0.4 VGG-19, random background, absgrad
  densification, SH degree 3 and matte-based floater cleaning. v2 adds:
  - a matte loss on rendered alpha
  - per-view camera refinement (SE(3) deltas; photo frozen; starts after warm-up)
  - a per-view colour affine to absorb lighting drift between generated views (photo frozen)
  - per-view weights: photo 1.0, anchors 0.4–0.8 by QA score, orbit frames 0.2–0.4
  - a hold-out mode that holds out 2 anchors, for the consistency metric in §7
- **Budget:** ≤ 500k Gaussians, 30k iterations.

### S6: Export and report

- `final/character.ply`: a standard 3DGS PLY at SH degree 3, Y-up, in metres, feet at y=0, facing +Z.
- `final/turntable.mp4`, plus `final/report.md` containing:
  - per-view QA results
  - hold-out PSNR/SSIM/LPIPS and the photo-view PSNR
  - Gaussian count and file size
  - timings and peak VRAM per stage
- View the PLY in SuperSplat (browser, local file) or with ComfyUI's native splat Load/Render nodes.
- **Optional polish:** face close-ups (the angles LoRA's `close-up` distance), registered on face
  keypoints and trained with extra weight on the head.

## 5. Guardrails for this PC

- **VRAM:** our code logs torch peak memory per stage to `run_report.json` and aborts above 20 GiB.
  Our launcher starts ComfyUI with `--reserve-vram 1.5` to leave desktop headroom. Your `.bat`
  files are never touched.
- **Step-time watchdog:** abort a ComfyUI job if its first sampler step takes more than about 5× its
  baseline, since that's the sysmem-spill signature.
- **One GPU user at a time:** `POST /free` before any GPU work in S0, S3 or S5. The trainer never runs
  while ComfyUI holds models.
- **RAM:** Docker Desktop off during runs (D3). Close GPU-hungry apps: ChatGPT desktop, Insta360
  Studio, browsers with hardware acceleration, Meta Quest Link.
- **SageAttention:** your launcher enables it globally. If Qwen outputs come out black or noisy, run
  that workflow with attention patched back to SDPA (KJNodes). This gets verified in M2.
- **Resume:** each stage writes `done.json` with input hashes. Reruns skip finished stages unless
  `-Force` is passed.
- I won't change Windows, driver, BIOS or WSL settings. Those are your decisions in §0.3.

## 6. Milestones

| Milestone | Scope | Done when |
|---|---|---|
| **M0 Scaffolding** | Create `.venv-i2g` with Python 3.11, torch 2.8.0+cu128, and SplatKit's SHA-256-verified gsplat 1.4.0 wheel, plus onnxruntime-gpu, opencv, trimesh and scipy. Write `i2g/comfy_client.py`, `scripts/v2_start_comfy.ps1` and the custom node junction. | A smoke workflow round-trips through the API, and gsplat rasterizes a test splat on CUDA. |
| **M1 Body scaffold** | S0 + S1 on `test-assets/test-image.png` | `overlay_input.png` and the rig renders look right; reprojection error < 1% of height. |
| **M2 Anchors + gate** | S2 + S3 plus the contact sheet | ≥ 11 accepted views covering 360° plus the elevated ring, **and you approve the sheet**. |
| **M3 First splat** | S5 + S6 from the photo + anchors | `character.ply` opens in SuperSplat; photo-view PSNR ≥ 30 dB; hold-out LPIPS reported; no large floaters; the back is plausible. |
| **M4 Dense orbit** | S4, retrain | Hold-out metrics beat M3; otherwise S4 stays off. |
| **M5 Bake-off + polish** | §7, face close-ups | A winner is chosen per stage, with numbers. |
| **M6 One command** | `scripts/run_v2.ps1 -InputImage input\x.png -Stage all` | A new image goes to a PLY unattended except for the gate. README, AGENTS and STATUS move to v2. |

## 7. Bake-off: choose models by measurement

Every candidate gets the same input, rig, seeds and trainer settings.

- **Anchors:**
  - Qwen-2511 + angles LoRA, Lightning 4-step vs 20-step, in each reference mode
    (`single` / `nearest2` / `sheet`)
  - FLUX.2 klein 9B with multi-reference and the angle given in the prompt (+ consistency LoRA)
  - Qwen followed by the klein detail pass
- **Orbit:**
  - Wan 2.2 first→last-frame
  - Wan 2.2 Fun-Control on scaffold renders
  - Wan 2.2 + orbit LoRA
  - MiniMax H3 first/last-frame (D5)
- **Metrics per candidate:**
  - median azimuth error
  - pose-drift PA-MPJPE
  - ArcFace identity
  - QA pass rate
  - **hold-out PSNR/SSIM/LPIPS** (4DAnyone's "generated-view consistency" protocol)
  - wall time and peak VRAM
- **Rule:** hold-out LPIPS decides first, then identity, then time.

## 8. Repo layout (new files only)

```
plan.md                      this file
i2g/                         pipeline package
  ingest.py body.py views.py register.py orbit.py frameset.py report.py
  comfy_client.py cameras.py vram.py
recon/                       trainer: fork of SplatKit core/splatting/training (MIT, attributed)
comfy_workflows/             API-format JSONs: sam3d_body, qwen_angles, klein_refine, wan_flf2v, wan_fun_control, h3_fl2v
comfy_nodes/i2g_export/      custom node: SAM 3D Body pose data → .npz
scripts/run_v2.ps1           -Stage ingest|body|views|qa|orbit|train|all  [-Force]
output/<name>/v2/            00_ingest 01_body 02_views 03_qa 04_orbit 05_frameset 06_train final
```

The legacy Docker/4DAnyone files stay untouched until v2 works. After that, the README marks them
as legacy.

## 9. Risks

| Risk | Signal | Mitigation |
|---|---|---|
| Views disagree on details (back of hair, jeans pockets), giving a blurry splat | High hold-out LPIPS | Farthest-first references, k candidates, per-view colour affine, confidence weights, dense orbit |
| The angle LoRA renders a different angle than requested | Azimuth error | Harmless for reconstruction because the solved camera is used; gross mirroring is rejected |
| The generator changes the pose (arms drop) | PA-MPJPE, wrist/elbow check | Reject and regenerate, strengthen the pose wording, or condition on scaffold renders |
| The arms-up pose hides armpits, back of head and fingers | Holes in the splat | Ring B and C views plus the visual-hull init; hidden fingers are accepted |
| Face is soft in full-body views (face ≈ 100 px) | Face-crop LPIPS | S6 close-ups |
| RAM runs out while ComfyUI offloads | Paging, slow loads | Docker off, fp8mixed Qwen instead of bf16, one model family loaded at a time |
| SageAttention artefacts with Qwen | Black or noisy images | SDPA fallback per workflow |
| Loose clothing or long hair far from the body | Silhouette QA | More anchors, lower orbit weight (4DAnyone has the same failure mode) |
| gsplat on Windows | Import or rasterize fails | SplatKit's wheel already passed a CUDA smoke test here; fall back to SplatKit's own venv |
| Licenses | — | D5 / D6 |

## 10. Fallbacks, and what's deliberately not planned

- **LHM++** (Apache-2.0): turns 1–16 pose-free images into a 3DGS PLY in about 2 s using about 8 GB.
  It would be a fast baseline or geometry prior, but its dependencies are Linux-first (pytorch3d,
  diff-gaussian-rasterization, spconv), so it would need WSL Ubuntu. Only used if S5 needs a
  stronger prior.
- **4DAnyone at 41 frames:** the paper trains 6-view groups on 41-frame clips, and a static subject
  doesn't need 121. About 3× fewer video tokens might bring it under the limit. Not planned; at most
  a later 30-minute experiment, and only if v2's consistency plateaus.
- **Cloud generators** (Krea, FLUX.2 [pro] and similar): images dropped into `02_views/manual/` get
  the same registration and QA. Krea 2 is text-to-image, so it doesn't help locally with
  image-conditioned novel views.
- **Excluded:**
  - SAM 3D Objects: needs 32 GB VRAM
  - FLUX.2-dev 32B: weights + text encoder exceed 24 GB VRAM + 32 GB RAM
  - HumanOrbit / Pippo-style human-orbit models: no public weights
  - dual-boot Linux: unnecessary at ≤ 20 GB per stage

## 11. Sources

- 4DAnyone paper: [papers/4dAnyonePaper.pdf](papers/4dAnyonePaper.pdf) (arXiv 2608.20335). §3.2 skeleton conditioning, §3.3 RCP/TCR, Supp. E inference, Supp. C Table 4 frame counts
- fal Multiple-Angles LoRA: https://huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA
- Qwen-Image-Edit ComfyUI builds: https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI
- Qwen-Image releases: https://github.com/QwenLM/Qwen-Image
- FLUX.2 [klein]: https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence
- Flux-2 Multi-Angles LoRA (FLUX.2-dev): https://huggingface.co/lovis93/Flux-2-Multi-Angles-LoRA-v2
- Wan 2.2 first/last-frame in ComfyUI: https://blog.comfy.org/p/wan22-flf2v-comfyui-native-support
- Wan 2.2 orbit LoRA: https://huggingface.co/ostris/wan22_i2v_14b_orbit_shot_lora
- Wan 2.2 360 rotation LoRA: https://huggingface.co/artificialguybr/360Rotation-Redmond-WAN2-I2V-14B
- MiniMax H3 license (territories, outputs clause): https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/LICENSE
- MiniMax H3 overview: https://www.runpod.io/blog/minimax-h3-the-open-weight-omni-modal-video-model-and-what-it-takes-to-run-it
- ComfyUI-OrbitSheets (H3 turnaround sheets): https://github.com/lumosai8/ComfyUI-OrbitSheets
- HumanOrbit (Wan I2V + LoRA for 360° human orbits): https://arxiv.org/html/2602.24148v1
- SAM 3D Body: https://github.com/facebookresearch/sam-3d-body
- MHR: https://github.com/facebookresearch/MHR
- Sapiens2: https://github.com/facebookresearch/sapiens2
- Depth Anything 3: https://github.com/ByteDance-Seed/Depth-Anything-3
- LHM++: https://github.com/aigc3d/LHM-plusplus
- SAM 3D Objects setup (32 GB VRAM): https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md
- gsplat on Windows: https://github.com/nerfstudio-project/gsplat/blob/main/docs/INSTALL_WIN.md
- CUDA on WSL limitations: https://docs.nvidia.com/cuda/wsl-user-guide/index.html
- B650 UD AX specs: https://www.gigabyte.com/Motherboard/B650-UD-AX-rev-10/sp
- ComfyUI-SplatKit: https://github.com/mickmumpitz/ComfyUI-SplatKit

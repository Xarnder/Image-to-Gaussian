# PAC-GS: implementation plan (sprint 1)

> **Update, 2026-09-19 evening.** The anchors are parked. The proxy-controlled video part was built
> first (`pacgs/`, run `python -m pacgs --help`) and tested. Results:
> [pacgs_results_sprint1.md](pacgs_results_sprint1.md).
> - The generated orbit is almost as self-consistent as a rendered splat (30.4 dB on held-out frames).
> - Camera refinement hurts (the cameras are exact).
> - It fixes v2's off-angle face.
> - Its gap is elevated views.
>
> The simplifications vs this plan:
> - gsplat renders the controls (no Blender needed).
> - A helper node exposes Fun-Control's `start_image`.
> - A frozen snapshot of the v2 inputs lives in each run folder.

**Status:** plan for review, 2026-09-19.
**Research plan:** [proxy_anchor_calibrated_gaussian_splat_research_plan.md](proxy_anchor_calibrated_gaussian_splat_research_plan.md) (v0.1)
**Scope:**
- Experiments 0, 1 and 4 from the research plan.
- A cheap weak-control anchor probe (a slice of Exp 3 and Exp 8).

This covers the research plan's milestones M0–M3 (§19) and its first sprint (§20). The order is changed so the riskiest assumption is tested first and cheaply.

**How to read this:**
- §0: the assessment, and what changed from the research plan.
- §1: the sprint's questions and the decision point after it.
- §2–§5: what already exists, conventions, code layout, data formats.
- §6: the phases, each with a "done when" gate.
- §7: how to run it on this PC.
- §10: the decisions I need from you before coding starts.

---

## 0. Assessment

### 0.1 Verdict

| Claim | Confidence | Why |
|---|---|---|
| Exp 0 passes: anchor detection + PnP is sub-degree on Blender renders | 97% | Textbook PnP. The prototype numbers are in Appendix C. |
| A proxy-controlled 360° sequence with anchors can be generated on this PC (Wan 2.2 Fun-Control, 480×832) | 80% | The models, nodes and VRAM budget are all here. Unknowns: the two 14B experts swapping through 32 GB RAM, and back-view quality. |
| Anchors are found well enough for PnP in ≥ 90% of generated frames (dense control) | 70% | They're in the control video, and they're large and simple. |
| Anchor PnP finds real camera drift under dense control, beyond the ~0.35° / 1.7 cm noise floor | **15%** | See 0.2: the generator draws the anchors where the control puts them. |
| Anchor-corrected cameras beat the Blender cameras for the splat under dense control (H4) | **10–15%** | Follows from the row above. |
| Anchors stay rigid across a 360° weak-control (prompt / I2V) orbit | 20–25% (arcs ≤ 90°: ~45%) | Your Wan_3 orbit (see 0.2) |
| A splat from the calibrated video + photo looks better overall than the current v2 splat | 45–55% | For: exact cameras, full 360° coverage, no angular gaps. Against: 480p softness, identity drift on the back. |
| … and it has fewer streaks, floaters and ghosting than v2 | ~70% | Dense coverage with exact cameras removes their causes. |
| A publishable (workshop-level) result if run cleanly | ~40% | The novel part is anchors as an *instrument* for measuring camera adherence. Proxy-controlled human multiview has prior art: 4DAnyone, MagicMan, SMPL-guided animation (Champ), HumanOrbit. |

### 0.2 The main finding: under dense control, anchors confirm the Blender camera instead of measuring the generator

- **Fun-Control is pixel-aligned by construction.** ComfyUI's `Wan22FunControlToVideo` VAE-encodes the control video into the first 16 concat channels and the start image into the next 16. These are concatenated channel-wise with the noisy latent. Fun-Control models are trained with controls extracted *from* the target videos, so output and control share the same pixel grid. The model has no free "camera".
- **So anchors in the control are placed by the control.** Under dense control, the control video is the only way to make anchors persist. The generator then draws them where the control put them, and PnP returns the Blender camera within detection noise, **whatever the model did with the person**.
- **What anchors do and don't measure here.** In this regime they measure "did the model follow the control". That's useful QA, but silhouette IoU against the proxy gives the same signal without anchors. What they can't see is the error that matters here: **texture and features sliding or morphing inside a correctly framed frame**. Examples: identity drift, hair and clothing changes, a face drawn a few degrees off on the head.
- **Where anchors do carry information: weak control.** That means prompt/I2V orbits, first/last-frame, camera-parameter models and MiniMax H3. There, the model places the anchors from its own notion of the camera. That's also where they're least likely to stay rigid.
- **Wan_3 already tested this.** It's a prompt-driven orbit (no proxy) of a studio covered in fiducials, and SfM still split it into front/back camera islands. If those markers had stayed rigid, COLMAP/Postshot would have locked onto them.
- **Two effects could still make anchors useful under dense control.** Exp 1 is built to show them if they exist:
  - a **temporal lag** between output and control (the Wan VAE packs 4 frames per latent)
  - **individual frames** where the model breaks from the control

### 0.3 What turned out better than the research plan assumed

- **Anchors fit without shrinking the subject.** I simulated the real orbit with your SAM 3D Body mesh:
  - Frame 0 is the photo camera (elevation −5.4°, vertical FOV 40°), at 480×832, with the subject at 85% of the frame height.
  - A floor ring of 24 Ø10 cm spheres at r = 0.45 m (centres alternating 5 cm / 30 cm high) keeps **17–19 anchors clear of the subject in every one of 36 views**.
  - Expected PnP precision is **0.35° / 1.7 cm at 2 px detection noise** (0.18° / 0.8 cm at 1 px).
  - The current keypoint cameras are 5–12 cm off, so anchors are 3–7× more precise *if* they share the subject's camera.
  - Wider framing gains only ~0.05°, so there's no reason to make the subject smaller. Full tables are in Appendix C.
- **Anchor removal isn't needed with the v2 trainer.** At every step, `i2g/splat.py` replaces everything outside the subject matte with a random background, and its alpha loss drives the splat to zero there. So:
  - Anchors that don't overlap the subject never reach the loss.
  - We subtract the anchor footprints (known from the camera) from each matte.
  - A per-pixel ignore mask covers the rare overlap.
  - No inpainting, so Exp 3's inpainting sub-study goes away.
- **The proxy is free.**
  - P2 (rough body) is the SAM 3D Body mesh already in `01_body/body_world.npz`, in metric world coordinates. P0 is its 70 keypoints.
  - **P4 (new)** is the current v2 splat. Its depth carries the hair and clothing volume the body mesh lacks.
- **Frame 0 can be exact.** The orbit starts at the photo camera. So the video's start image is the photo, reframed by a pure scale and crop, over a Blender render of the floor and anchors. This is exact because both use the same known camera.
- **Everything runs locally.**
  - Blender 4.5.1 LTS and 5.2.1 LTS are installed.
  - ComfyUI core has `Wan22FunControlToVideo` (control video + start image + reference image), `WanAnimateToVideo`, `WanFirstLastFrameToVideo` and `WanCameraImageToVideo`.
  - On disk: Wan 2.2 Fun-Control A14B fp8, Wan 2.2 Animate 14B, Wan 2.2 I2V Q8 + Lightning, and MiniMax H3 FL2V/Ref2V.
  - Your own FLF and Wan Animate workflows are there too.

### 0.4 What changes (research plan → this plan)

| Research plan | This plan | Why |
|---|---|---|
| Exp 4: Blender cameras vs anchor cameras | 4 arms (A Blender, B anchor, C Blender + subject refinement, D anchor + subject refinement; §6 Phase 5) | Arm C separates "the camera is wrong" from "the subject slid". If A ≈ B but C > A, the error is in the content. |
| Subject-consistency check (loosely specified) | Explicit metrics: neighbour warping residual through proxy depth and known cameras, ArcFace along the orbit, silhouette IoU against the proxy, leave-frames-out splat PSNR | The dominant error is expected to be content, so measure it directly. |
| Anchor removal by inpainting | Subtract footprints from the mattes, plus an ignore mask | The trainer already ignores everything outside the matte. |
| Splatfacto first | The v2 gsplat trainer (`i2g/splat.py`). A nerfstudio `transforms.json` is exported only for portability. | It runs natively here, matches Splatfacto, and already has the matte loss, pose refinement, per-view weights, face priors and an eval harness. |
| Proxy authored by hand in Blender first | SAM 3D Body mesh (P2) generated automatically, and the v2 splat (P4) as the richer proxy. You can still edit the generated `.blend`. | The mesh already exists, is metric, and is aligned to the photo camera. |
| "Asymmetric side towers" anchor rig | Floor ring close to the feet | With a low camera, anything above ~0.2 m on the camera side covers the subject at some azimuth. The simulator picks the layout. |
| 121-frame control orbits | 81-frame arcs (4 × 90° = 320 unique frames) | 81 frames is Wan 2.2's training length. |
| Exp 3 anchor design study early | Only if Exp 1 finds drift or the probe shows anchors survive | Don't build a design study for an instrument that may read zero. |
| (not in the plan) | Weak-control probe: anchors composited into the start image of a Wan 2.2 FLF/I2V orbit | Tests anchors where they actually carry information. |
| Exp 9 intrinsic refinement | Deferred | Dense control can't zoom; it's pixel-aligned by construction. The probe checks zoom under weak control. |

---

## 1. Sprint goal and decision point

The sprint answers three questions using one subject (`test-image`) and one seed per condition. Seed variance comes in sprint 2.

- **Q1 (Exp 1):** Under dense proxy control, is there measurable camera drift?
  - Measured as anchor PnP vs the Blender camera, against the solver's own noise floor.
  - Also: temporal lag, frames that break from the control, silhouette IoU.
- **Q2 (Exp 4):** Does a splat trained on the calibrated sequence plus the photo beat v2? And which camera source wins: A, B, C or D?
- **Q3 (probe):** Under weak control, do anchors stay rigid across an orbit?
  - Measured by the single-camera reprojection RMSE, colour/identity swaps and loop closure.

**Pre-registered predictions.** These are written down now so the goalposts can't move later.

| | Prediction |
|---|---|
| Q1 | Median drift ≤ 0.3° and ≤ 1.5 cm (the noise level). ≥ 90% of frames have ≥ 8 anchors. Lag < 0.5 frame. |
| Q2 | A ≈ B (held-out PSNR within 0.2 dB). C beats A by 0–0.5 dB. Fewer needles and floaters than v2. The back is softer than v2's Qwen back. |
| Q3 | Rigidity breaks past ~90° of orbit (RMSE > 5 px in ≥ 30% of those frames). Anchors that first come into view after frame 0 change colour. |

**Decision after Exp 4** (this sets sprint 2):

| Outcome | Meaning | Sprint 2 |
|---|---|---|
| Q1 finds drift **and** B > A | Anchors correct real drift | The research plan as written: anchor design study (Exp 3), temporal BA, triage (Exp 5), more backends |
| Q1 ≈ 0 and C ≈ A | The cameras are right; the remaining error is content | Content consistency: proxy levels (Exp 2, incl. P4), arcs (Exp 6), view count (Exp 7), video upscaling, triage by warp residual. Anchors stay as QA. |
| Q1 ≈ 0 and C > A | The subject is drawn "as if" from a slightly different camera, which anchors can't see | Subject-based camera refinement is the tool. Study it against arcs and proxies. |
| Q3: anchors survive on short arcs | Anchors work for weak-control backends | The "anchors as an instrument" paper angle: benchmark camera adherence of camera-controlled video models (H3, Wan camera, LTX camera LoRA) |
| Q2: PAC-GS splat < v2 | The video route loses on this PC (e.g. on resolution) | Before dropping it: 720p, SeedVR2 video upscale, or a hybrid (v2 Qwen views + calibrated video frames) |

---

## 2. What we build on

| Asset | Where | Used for |
|---|---|---|
| World frame, photo camera, body mesh + 70 keypoints | `output/test-image/v2/01_body/body_world.npz` (`i2g.body.load_world`) | Scene origin and scale, orbit start, P0/P2 proxy |
| Photo canvas, matte, HR photo | `output/test-image/v2/00_ingest/` (`canvas.png`, `alpha.png`, `source_hr_rgba.png`) | Start image, reference image, training photo |
| v2 splat | best of `experiments/E5–E7` or `final/character.ply` | P4 proxy; baseline |
| Camera maths | `i2g/cameras.py` (`project`, `refine_pnp`, `render_silhouette`, `azimuth_elevation`) | Projection, PnP refinement |
| Trainer | `i2g/splat.py` (`load_views`, `train`, `clean`, `export_ply`) | Splats (3 small changes, §4.3) |
| Eval harness | `i2g/evaluate.py` (`load_ply`, `render`, `FaceID`, `_psnr`, `_ssim`, `_VGG`), `i2g/reregister.py` (`fit_camera`) | Metrics, turntables, hold-out camera fitting |
| ComfyUI client | `i2g/comfy_client.py` (step-time watchdog), `scripts/v2_start_comfy.ps1`, `scripts/v2_stop_comfy.ps1` | Generation |
| Matting | `i2g/matting.py` (BiRefNet) | Subject mattes of generated frames |
| Blender | `C:\Program Files\Blender Foundation\Blender 4.5\blender.exe` (4.5.1 LTS) | Building the scene; control and anchor passes |
| Video models | ComfyUI `models/diffusion_models`: `Wan2_2-Fun-Control-A14B-{HIGH,LOW}_fp8_e4m3fn_scaled_KJ_fixed`, `Wan2.2-I2V-A14B-{High,Low}Noise-Q8_0.gguf` (+ `wan2.2_i2v_lightx2v_4steps_lora_v1_*`), `Wan2_2-Animate-14B_fp8_*`, `minimax_h3_*` | Generators |
| Your workflows | ComfyUI `user/default/workflows/`: "Xander Wan 2.2 Video Start and End Frame Genration", "Wan Animate", "Xander's Wan Animate 2 August 2026 V32" | Known-good sampler settings to copy |

---

## 3. Conventions (single source of truth)

- **World:** the i2g world. Y is up, units are metres, the person faces +Z, and the origin is on the floor between the feet (`body.build_world`).
- **Cameras:** an OpenCV world-to-camera 4×4 plus K for every frame. In sprint 1, fx = fy and the principal point is at the image centre. All JSON uses this.
- **Orbit:**
  - Frame 0 *is* the photo camera: `w2c(θ) = photo_w2c · R_y(θ)⁻¹`.
  - This pivots about the vertical axis through the subject (x = z = 0), keeping the photo's distance, height and pitch.
  - Azimuth increases toward the person's left (+X), as in `i2g.cameras`. Elevation rings later add a rotation about the target.
- **Video intrinsics:**
  - The canvas's vertical FOV (40.0° for `test-image`), square pixels, principal point centred.
  - `fy = K_canvas[1,1] · H_video / H_canvas`.
  - At 480×832 the subject is 85% of the frame height and never touches the border. This was checked at all 36 azimuths.
- **Blender mapping.** This conversion lives in `blender_build.py` and nowhere else.
  - World: `p_bl = A · p_i2g` with `A = [[1,0,0],[0,0,−1],[0,1,0]]`. i2g's Y-up becomes Blender's Z-up, and the person faces Blender's −Y (its "front" view).
  - Camera: `c2w_bl = A₄ · c2w_cv · diag(1, −1, −1, 1)`.
  - Lens: `sensor_fit = 'HORIZONTAL'`, `sensor_width = 36`, `lens = fx · 36 / W`, `shift_x = shift_y = 0`.
  - Render: exactly W×H at 100%, pixel aspect 1.
- **Pixel centres:** all our code uses OpenCV's integer-centre convention. The round-trip test measures any half-pixel offset against Blender and gsplat, and it's fixed in one place.
- **Resolution:**
  - Generation sizes are multiples of 16.
  - Control renders are made at exactly the generation size, because `Wan22FunControlToVideo` centre-crop-resizes anything else. That would silently break pixel alignment.

**Convention tests** (`python -m unittest discover -s pacgs/tests`). They must pass before any generation:

1. **Axis/cube scene:** centroids of small spheres in Blender's ID pass vs `i2g.cameras.project`: < 0.3 px.
2. **Depth pass** vs analytic floor-plane depth: < 1 mm.
3. **Orbit frame 0:** reprojects the SAM 3D Body keypoints onto the photo's 2D keypoints (scaled): < 0.5 px.
4. **Export round trip:** `transforms.json` (OpenGL) back to OpenCV gives the identity to 1e-9.

---

## 4. Code layout

### 4.1 Package

```
pacgs/                       research code at the repo root, next to i2g/ (imports it)
  cli.py                     python -m pacgs <scene|simulate|render|exp0|generate|calibrate|qa|export|train|eval|report>
  config.py                  dataclass configs loaded from YAML (PyYAML is already in .venv-i2g); config hash
  manifest.py                run manifest: input hashes, tool/model versions, seeds, git commit + dirty flag
  scene/
    proxy.py                 P0 skeleton, P1 capsules (later), P2 SAM 3D Body mesh -> OBJ, P4 v2-splat depth via gsplat
    rig.py                   orbit / arcs / rings from the photo camera -> cameras.json
    anchors.py               anchor families (sphere ring v1; later variants) -> anchors.json
    simulate.py              per-frame usable anchors + expected PnP covariance (Appendix C, productised)
    blender.py               launcher: blender -b --factory-startup -P <script> -- <args>; version check
    blender_build.py         [runs inside Blender] scene.json -> scene.blend (proxy, floor, pegs, anchors, cameras)
    blender_render.py        [runs inside Blender] passes per frame; re-exports cameras + anchor centres from the .blend
    passes.py                pass readers; depth -> control encoding; z-composite (P4 depth + floor/anchors)
    compose.py               frame-0 start image: photo reframed to K_video over the floor/anchor render
  generate/
    base.py                  Generator protocol: (controls, start_image, ref_image, prompt, seed, size, length) -> frames + provenance
    comfy_graphs.py          API-format graph builders, checked against /object_info at startup
    wan_fun_control.py       dense control: Wan22FunControlToVideo, A14B high -> low two-stage sampling
    wan_flf.py               weak-control probe: WanFirstLastFrameToVideo (start = end = frame-0 composite)
    wan_animate.py           sprint 2: WanAnimateToVideo (pose + mask + background video)
    frames.py                PNG sequences, arc stitching, frame indexing
  calib/
    detect.py                anchor detection: colour + roundness in a window around the predicted position; sub-pixel centre
    assign.py                identities: Hungarian on predicted positions (prior mode); colour-consistent RANSAC (no-prior mode)
    pnp.py                   SQPnP + RANSAC -> LM refinement; 6x6 covariance; residuals
    trajectory.py            sprint 2: temporal smoothing / small BA over the fixed anchors
  qa/
    consistency.py           silhouette IoU vs proxy; neighbour warp residual; ArcFace along the orbit; sharpness
    triage.py                accept / camera-correct / reject (recorded in sprint 1, applied in Exp 5)
    report.py                contact sheets, plots, summary.md
  export/
    dataset.py               registration-style records for i2g.splat (+ K, kind="video", ignore masks)
    nerfstudio.py            transforms.json (OpenGL c2w, per-frame fl/cx/cy/w/h, mask_path)
  experiments/
    exp0_solver.py  exp1_adherence.py  exp4_cameras.py  probe_weak.py  eval_lfo.py
  tests/                     stdlib unittest: conventions, projection, PnP, detector on synthetic images

next-stage-research/
  configs/                   scene_v1.yaml  anchors_ring_v1.yaml  gen_funcontrol_480.yaml
                             exp0.yaml  exp1.yaml  exp4.yaml  probe_weak.yaml
  docs/                      research plan, this plan, results_sprint1.md

output/pacgs/<subject>/<run_id>/     (gitignored)
  manifest.json  scene/  controls/  gen/  calib/  qa/  datasets/  splats/  eval/  report/
```

### 4.2 Why a separate package

The product pipeline (`i2g/`, `image_to_splat.py`, `run_image_to_splat.bat`) stays untouched apart from the trainer changes in 4.3. The research code can churn freely without breaking the product.

### 4.3 Trainer changes (`i2g/splat.py`, backward compatible)

1. **Per-record intrinsics.** `load_views` uses `r["K"]` (scaled to the image) when present. Otherwise nothing changes.
2. **`kind: "video"`.**
   - New `SplatConfig` fields: `video_weight` (default 0.6), `video_share` (share of non-photo steps, default 0.75) and `video_pose_refine` (default off).
   - Pose deltas, alpha weight and sampling switch on `kind`, the same way `restored` does now.
3. **Ignore mask** (optional `r["ignore"]` path).
   - Inside it, the target becomes `pred.detach()` and the matte becomes `alpha.detach()`.
   - So L1, SSIM, VGG and the alpha loss all get zero gradient there.
   - This is the same trick the face weighting uses.
4. **Regression check.** With none of the new fields present, a 300-step seeded run must reproduce the old loss curve exactly.

---

## 5. Data contracts

These are adapted from research plan §13. Everything is JSON, OpenCV convention, metres.

**`scene/cameras.json`**, written by `rig.py` and re-exported from the `.blend` by `blender_render.py`:
```json
{"convention": "opencv_w2c", "units": "m", "world": "i2g: Y up, +Z front, origin on the floor between the feet",
 "width": 480, "height": 832,
 "frames": [{"frame": 0, "arc": 0, "az_deg": 0.0, "el_deg": -5.43,
             "K": [[1143.0, 0, 240.0], [0, 1143.0, 416.0], [0, 0, 1]],
             "w2c": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}]}
```

**`scene/anchors.json`:**
```json
{"rig": "ring_v1", "floor_albedo": 0.5,
 "anchors": [{"id": "A00", "family": "sphere", "radius_m": 0.05, "hue": "red", "rgb": [0.85, 0.08, 0.08],
              "world_xyz": [0.0, 0.05, 0.45], "peg": false}]}
```

**`calib/frames.json`**, one record per generated frame after calibration and QA:
```json
{"frame": 42, "arc": 0, "path": "gen/arc0/0042.png", "matte": "qa/mattes/0042.png", "ignore": "qa/ignore/0042.png",
 "K": [[1143.0, 0, 240.0], [0, 1143.0, 416.0], [0, 0, 1]],
 "w2c_blender": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
 "w2c_pnp": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
 "anchors": [{"id": "A07", "uv": [311.2, 771.8], "pred_uv": [310.9, 772.4], "conf": 0.93, "inlier": true}],
 "pnp": {"n": 17, "inliers": 17, "rmse_px": 1.4, "d_rot_deg": 0.21, "d_centre_cm": 0.9,
         "sigma_rot_deg": 0.30, "sigma_centre_cm": 1.5, "mahalanobis": 3.1},
 "qa": {"iou_proxy": 0.91, "warp_res_2": 0.031, "warp_res_8": 0.058, "arcface": 0.52, "sharpness_rel": 0.70},
 "triage": "camera_good", "reasons": []}
```

**`manifest.json`** records:
- SHA-256 of the source photo, `body_world.npz`, the proxy and every config
- Blender version, ComfyUI version, model file names and sizes
- seeds, prompts, the full ComfyUI graph JSON per arc
- the trainer config, and the repo commit + dirty flag
- start/end time and peak VRAM for each stage

---

## 6. Phases

The GPU column matters: the no-overlap rule is in §7. Coding times are rough.

### Phase 0: scaffolding (CPU, ~2–3 h)

**Tasks:**
- The `pacgs/` skeleton, CLI, YAML configs, manifest and unittest runner.
- The trainer changes in §4.3, with their regression check. The check is a 300-step run of ~1 min on the GPU; it's small enough to share with the trainings running now.

**Done when:**
- `python -m pacgs --help` lists the commands.
- The trainer regression check passes.
- A run folder gets a manifest.

### Phase 1: scene, rig, simulator, Blender passes (CPU, ~4–6 h)

**Tasks:**

1. **`rig.py`:** the orbit from the photo camera, 4 arcs × 90° × 81 frames = 320 unique frames. A 2 × 180° variant is a config switch.
2. **`anchors.py`, ring v1:**
   - 24 spheres, Ø10 cm, on a ring of r = 0.45 m.
   - Centres alternate at 0.05 m (resting on the floor) and 0.30 m (on Ø1 cm grey pegs).
   - 6 saturated hues in a non-periodic order: no two neighbours share a hue, and every (left, self, right) hue triple is unique.
   - Matte materials. The floor is a matte grey plane matched to the canvas grey (0.5).
3. **`simulate.py`**, on the real path:
   - usable anchors per frame
   - expected rotation and centre σ at 1, 2 and 3 px
   - a subject-in-frame check
   - **Gate:** ≥ 10 usable anchors in every frame, and ≤ 0.5° / 2 cm at σ = 2 px. The prototype gives 17–19 anchors and 0.35° / 1.7 cm.
4. **`proxy.py`:**
   - P2: an OBJ from `body_world.npz`.
   - P0: a keypoint JSON.
   - P4: depth renders of the v2 splat via gsplat (alpha threshold 0.5, 3×3 median filter).
5. **`blender_build.py` + `blender_render.py`** (Blender 4.5.1 LTS). Data passes use Cycles on the CPU: 1 sample, pixel filter 0.01 px (no antialiasing), float EXR. The passes are:
   - `depth`: view-space Z, from Camera Data → emission
   - `normal`: camera space
   - `id`: a flat emission value per object (subject, floor, peg, anchor k)
   - `pretty`: a lit RGB render (floor + anchors + clay proxy), Cycles at 64 spp
   - `anchors_only`: `pretty` with the subject hidden, used for frame-0 compositing
   - A shadow-catcher pass of the proxy, for a soft contact shadow

   Also:
   - Cameras and anchor centres are re-exported **from the `.blend`**, not from our own JSON, so the round-trip test compares Blender's truth against ours.
   - EXRs are converted to `.npz` inside Blender (it bundles numpy), so the venv needs no OpenEXR.
6. **`passes.py`:** depth → Fun-Control control frames (encoding in A.4). A variant, `depth_source: da2`, instead runs Depth Anything V2 (comfyui_controlnet_aux) on the clay pass. The smoke test picks one.
7. **`compose.py`:** the frame-0 start image.
   - The photo canvas is reframed to K_video: same camera and FOV, so it's a pure scale plus centre crop.
   - It's alpha-composited over `anchors_only`, with the contact shadow.
   - It also writes frame 0's subject matte and anchor footprint masks.
8. Convention tests 1–4 (§3).

**Outputs:**
- `scene/scene.blend`, `scene/cameras.json`, `scene/anchors.json`, `scene/simulate.json`
- `controls/{depth,normal,id,pretty}/NNNN.*`, `controls/start_image.png`
- `controls/contact_sheet.png`: every 10th control frame with the projected anchor IDs overlaid

**Done when:**
- all convention tests pass
- the simulator gate passes
- **you've looked at the contact sheet and opened the `.blend`**

This is the research plan's M0 exit: "one command generates proxy frames + exact camera/anchor JSON".

### Phase 2: Experiment 0, solver sanity (CPU, ~3–4 h code, ~15 min run)

**Tasks:**

1. **`detect.py`** on the Blender `pretty` renders (algorithm in A.1). It returns sub-pixel centres and a confidence per anchor.
2. **`pnp.py`:**
   - `cv2.solvePnPRansac` with `SOLVEPNP_SQPNP` and a 4 px threshold, then `solvePnPRefineLM` on the inliers.
   - Covariance: `s²(JᵀJ)⁻¹`, with s estimated from the residuals.
   - Returns w2c, RMSE, inliers, the Δ to the prior, σ_rot and σ_centre.
3. **`exp0_solver.py`,** 20 trials per cell:
   - **(a) Analytic projections + noise.** Gaussian noise σ ∈ {0, 0.5, 1, 2, 4} px × visible count ∈ {4, 6, 8, 12, 16} × outlier fraction ∈ {0, 10, 25%} (wrong identities or displaced points).
   - **(b) Real detections** on the Blender renders.
   - **(c) Perturbed-camera renders.** Render from cameras moved by known amounts (±2°, ±5 cm), while the prior stays the intended camera. The solver must recover the *perturbed* camera, which proves it measures the actual camera.
   - **(d) A 3% zoom/crop** of the image. This shows up as a coherent residual pattern plus a pose bias. It documents why intrinsics stay fixed.
4. **Plots:** rotation and centre error vs noise and anchor count, failure rate, and a histogram of detection error.

**Done when** (the research plan's gate, made concrete):
- median rotation error < 0.2° and centre error < 1 cm at σ = 1 px with ≥ 8 anchors
- median detection error on Blender renders < 0.5 px
- perturbed cameras recovered to < 0.1° / 0.5 cm
- no failures with ≥ 8 inliers and ≤ 25% outliers

**Expected:** a pass. A failure here means a conventions bug, which tests 1–4 should already have caught.

### Phase 3: generator adapter + smoke test (GPU session 1: ~3–4 h code, ~1 h GPU)

**Tasks:**

1. **`comfy_graphs.py`.**
   - At startup, it calls `ComfyClient.object_info()` and asserts that every node and input it uses exists:
     - `Wan22FunControlToVideo` with `ref_image`, `start_image` and `control_video`
     - `UNETLoader`, `CLIPLoader`, `VAELoader`, `ModelSamplingSD3`, `KSamplerAdvanced`, `LoraLoaderModelOnly`
     - `VHS_LoadImagesPath`, `SaveImage`
   - The graph is in Appendix B.
   - Sampler, scheduler, shift, CFG and the negative prompt are copied from your FLF workflow, which is known to work on this PC.
   - Output is lossless PNG. MP4 compression would hurt anchor detection and the splat.
2. **`wan_fun_control.py`.**
   - Arcs are chained: arc k's `start_image` is arc k−1's last generated frame. `ref_image` is always the photo.
   - Provenance per arc: model files and sizes, graph JSON, seed, prompt and timings.
   - The step-time watchdog (`max_step_seconds`) catches VRAM spilling into system RAM.
3. **Smoke matrix.** Each run is one arc (0→90°, 81 frames). Stop at the first good one.

| Run | Control | Sampling |
|---|---|---|
| S1 | ground-truth depth (P2 + floor + anchors) | 20 steps, no LoRA |
| S1b | the same, **without anchors** (control and start image) | 20 steps. Answers "do the anchors change the person?" |
| S2 | Depth Anything V2 on the clay render | 20 steps |
| S3 | the better of S1/S2 | lightx2v 4-step LoRAs, 4–6 steps, if they work with Fun-Control |

4. **For each run:** time per step, peak VRAM (ComfyUI `/system_stats`), quick detection + PnP on every 10th frame, and a contact sheet.

**Done when:** one configuration gives 81 frames with an identifiable, static subject and visible anchors, and its time per arc is known.

**Stop rule:** a step taking more than 5× the first step's time means VRAM is spilling to system RAM. Stop ComfyUI and report.

### Phase 4: Experiment 1, adherence over the full orbit (GPU session 2: ~1.5 h; then CPU)

**Tasks:**

1. **Generate** 4 × 90° arcs (320 frames) with the smoke-test winner, proxy P2, seeds 1001–1004.
2. **If the session has time,** repeat with **proxy P4**: v2 splat depth, z-composited with the floor/anchor depth. Same cameras and seeds.
3. **Stop ComfyUI**, then run on the CPU (plus a little GPU for BiRefNet):
   - mattes, with the anchor footprints subtracted
   - detection and PnP per frame
   - QA metrics:
     - silhouette IoU vs the proxy, at the intended camera and at the recovered camera
     - warp residual to the neighbours at ±2 and ±8 frames (A.3)
     - ArcFace vs the photo where the face is visible
     - sharpness
4. **`exp1_adherence.py` report:**
   - Δyaw, Δpitch, Δroll and Δcentre vs frame, with the solver's ±2σ band
   - anchor count, RMSE and inlier ratio vs frame
   - **lag estimate** (A.5)
   - behaviour at the arc seams
   - IoU, warp-residual and ArcFace curves
   - a contact sheet of every 8th frame: detected anchors (green), predicted anchors (red), and the proxy contour at the recovered camera (cyan)
   - `summary.md`: the Q1 verdict against the pre-registered prediction

**Done when:** `report/exp1/summary.md` answers Q1 with numbers for P2, and for P4 if it was run.

### Phase 5: Experiment 4, which cameras make the better splat (GPU session 3, ~2 h of training)

**1. `export/dataset.py`:** identical frames and mattes for every arm.
- **Training:** every 2nd frame (160 frames) plus the HR photo at its exact camera.
- **Held out** (never trained on):
  - *LFO-dense:* every 4th frame, offset 1 (80 frames, ~1° from a training frame).
  - *LFO-gap:* 4 blocks of 9 frames centred on 45°, 135°, 225° and 315° (10° gaps, away from the arc seams).

**2. Arms.** All use the same seed, the best v2 settings (the E5/E6/E7 winner's config) and 15k iterations. Three train in parallel (~4.5 GB each, within the no-overlap rule), then the fourth.

| Arm | Cameras | Trainer pose refinement |
|---|---|---|
| A | Blender (intended) | off |
| B | Anchor PnP | off |
| C | Blender | on (`pose_lr 5e-4`, `pose_reg 0.1`, as in v2) |
| D | Anchor PnP | on |
| E (optional) | Blender, trained on the S1b-style **no-anchor** generation | off |

Arm E needs one more generation session. It answers "do the anchors cost splat quality?"

**3. `eval_lfo.py`,** for each arm:
- **Masked PSNR / SSIM / VGG** on LFO-dense and LFO-gap, computed two ways:
  - (i) with the arm's own camera source
  - (ii) with each held-out camera fitted to the splat (`i2g.reregister.fit_camera`)
  - The gap (ii) − (i) *is* the remaining camera error.
- **The standard harness on the same PLY:** photo dB, face id/min, needles, floaters, sharpness.
- **Visuals:** a turntable and `compare.png`, next to the best v2 experiment.
- **C/D only:** the trainer's per-frame pose deltas plotted against the anchor − Blender deltas. Do the subject and the anchors agree on the camera?

**4. `summary.md`:** the Q2 verdict and the matching row of the decision table in §1.

**Done when:** the four arms and v2 are in one table and one compare sheet, and you've looked at the turntables. This is the research plan's M3 "first decisive test".

### Phase 6: weak-control anchor probe (GPU session 4: ~1 h; code ~2 h)

**Setup:**
- Wan 2.2 I2V A14B Q8 + lightx2v, using your FLF workflow's settings, 81 frames.
- `start_image = end_image` = the frame-0 composite (photo + anchor floor).
- A prompt-driven orbit ("the camera orbits 360° around her…"), as in Wan_3.
- A second variant with 90° of orbit ("the camera moves a quarter turn to her left").

**No-prior calibration:**
- Frame 0 has a known camera.
- Each later frame starts from the previous frame's pose as the prior (tracking). If tracking is lost, it falls back to colour-consistent RANSAC over identity hypotheses.

**Metrics:**
- single-camera reprojection RMSE: the **rigidity** test
- whether each anchor keeps its identity and colour
- how smooth the recovered path is
- loop closure: frame 80 vs frame 0
- whether the anchor camera agrees with the subject (P2 silhouette IoU at the anchor camera)

**Done when:** Q3 is answered, with a contact sheet showing where rigidity breaks.

### Phase 7: conditional follow-ups (sprint 2, chosen by the decision table)

- **Research plan experiments:**
  - Exp 2 proxy ablation: P0 pose-only, P2, P4; P1 capsules only if P0 and P2 differ.
  - Exp 6 arcs: 1 × 360° vs 2 × 180° vs 4 × 90°.
  - Exp 7 view count: 40, 80, 160, 320.
  - Exp 5 triage policies.
- **Generation quality and backends:**
  - 720×1280 generation.
  - SeedVR2 video upscale of the generated arcs. It's temporally consistent; reuse `i2g/restore.py`.
  - Wan-Animate replacement mode as the second dense backend: pose + mask + background video. The background keeps the anchors pixel-exact.
- **Scaling up the evidence:**
  - Exp 10 ground-truth benchmark. It needs a textured 3D human asset, which is a licence decision.
  - More subjects and seeds.

---

## 7. Running it on this PC

**GPU rule** (from project memory): ComfyUI stages and training never overlap. Overlap causes 5–20× slowdowns.
- The other session's queue is still running: E5/E6 are training now, then SeedVR2 restores E6's orbit, then E7 trains.
- PAC-GS GPU sessions start only after `logs/E7.out` shows "E7 done", or when you say so.
- Phases 0–2 are CPU-only (Blender runs Cycles on the CPU) and can start now.

**Session plan.** ComfyUI takes ~7 min to cold-start, so keep it up within a session and stop it at the end.

| Session | What | Estimate |
|---|---|---|
| 1 | Smoke S1, S1b, S2, S3 | ~1–1.5 h |
| 2 | Exp 1: 4 arcs P2 (+ 4 arcs P4) | 1–2 h |
| — (CPU) | Matting, calibration, QA, report | ~30 min |
| 3 | Exp 4: 4 trainings + evaluation | ~2 h |
| 4 | Weak-control probe (+ arm E generation if wanted) | ~1–2 h |

**Commands** (once implemented; run from the repo root with `.venv-i2g` activated):

| Step | Command |
|---|---|
| Tests | `python -m unittest discover -s pacgs/tests` |
| Build scene | `python -m pacgs scene --config next-stage-research/configs/scene_v1.yaml` |
| Simulate rig | `python -m pacgs simulate --run output/pacgs/test-image/r001` |
| Render passes | `python -m pacgs render --run output/pacgs/test-image/r001` |
| Exp 0 | `python -m pacgs exp0 --run output/pacgs/test-image/r001` |
| Smoke (session 1) | `python -m pacgs generate --run ... --config next-stage-research/configs/gen_funcontrol_480.yaml --arcs 0 --variant S1` |
| Exp 1 generate (session 2) | `python -m pacgs generate --run ... --arcs all` |
| Calibrate + QA | `python -m pacgs calibrate --run ...` then `python -m pacgs qa --run ...` |
| Exp 4 (session 3) | `python -m pacgs export --run ...` then `python -m pacgs train --run ... --arms A,B,C` then `--arms D` |
| Evaluate + report | `python -m pacgs eval --run ...` then `python -m pacgs report --run ...` |

---

## 8. Outputs and headline metrics

| Experiment | Key files | Headline numbers |
|---|---|---|
| Phase 1 | `scene/scene.blend`, `controls/contact_sheet.png`, `scene/simulate.json` | Usable anchors per frame, expected σ, convention test residuals |
| Exp 0 | `report/exp0/{plots,summary.md}` | Rotation/centre error vs noise, count and outliers; detection error; perturbed-camera recovery |
| Smoke | `gen/smoke/<variant>/`, `report/smoke.md` | s/step, peak VRAM, anchors found, S1 vs S1b ArcFace/sharpness |
| Exp 1 | `calib/frames.json`, `report/exp1/{drift.png,anchors.png,qa.png,contact_sheet.png,summary.md}` | Median and p90 drift vs σ, frames with ≥ 8 anchors, lag, IoU, warp residual, ArcFace |
| Exp 4 | `splats/{A,B,C,D}/character.ply`, `eval/{A..D}/eval.json`, `report/exp4/{compare.png,turntables,summary.md}` | LFO-dense/gap PSNR/SSIM/VGG (own vs fitted cameras), photo dB, face id, needles, floaters |
| Probe | `report/probe/{rigidity.png,contact_sheet.png,summary.md}` | Rigidity RMSE vs orbit angle, identity persistence, loop closure |

---

## 9. Implementation risks

| Risk | Signal | Mitigation |
|---|---|---|
| Blender↔OpenCV convention bug | Round-trip test > 0.3 px | Fix before any generation. The conversion lives in one function. |
| Swapping the two 14B experts thrashes 32 GB of RAM | Minutes between the high and low stages, heavy disk activity | One arc per graph, pagefile on the NVMe, lower resolution; a Fun-Control GGUF if one is needed (none on disk) |
| The Lightning LoRAs don't suit Fun-Control | Smeared or frozen output in S3 | Stay at 20 steps (~+10 min per arc) |
| Depth from the hairless body mesh squashes the hair | Hair clipped to the skull | P4 proxy (v2 splat depth) |
| Anchors distract the generator or change the lighting | Blobs on the subject, colour casts, S1 worse than S1b | Smaller spheres, fewer hues, prompt wording. Arm E measures the cost. |
| Drift at arc seams (chained start images) | ArcFace or warp-residual jumps at multiples of 90° | 2 × 180° variant; `ref_image` is always the photo |
| Anchors first seen after frame 0 change colour | Colour mismatch despite good position agreement | Identity comes from the position prior (dense control); colour is only a soft cue |
| BiRefNet mattes include the coloured spheres | Matte blobs at anchor positions | Subtract the projected footprints, dilated by 4 px |
| Frames too soft at 480p | Low sharpness; soft back | 720×1280 run or a SeedVR2 video pass (sprint 2) |
| GPU contention with the E5–E7 queue | Slow steps, watchdog trips | Wait for E7; watchdog on |

---

## 10. Decisions for you before we start

| # | Decision | My default |
|---|---|---|
| P1 | Accept the reordered sprint: answer Q1–Q3 first. The anchor design study, temporal BA, intrinsic refinement and inpainting come only if the data calls for them. | Yes |
| P2 | Pilot proxy: P2 (SAM 3D Body mesh) first, then P4 (v2 splat) in the same GPU session | Yes |
| P3 | Backends: Fun-Control (dense) first, Wan 2.2 FLF/I2V for the weak-control probe, Wan-Animate in sprint 2 | Yes |
| P4 | MiniMax H3 as a weak-control backend. Its licence excludes the US/EU/UK/South Korea (plan.md D5 is still open). | No, unless you confirm your location is covered |
| P5 | Blender version | 4.5.1 LTS: the long-tested bpy API. 5.2.1 LTS would also work. |
| P6 | Locations: code in `pacgs/` at the repo root, configs and docs in `next-stage-research/`, outputs in `output/pacgs/` (gitignored) | Yes |
| P7 | Add `matplotlib` to `.venv-i2g` for report plots (PyPI, ~9 MB). Tests use stdlib `unittest`, not pytest. | Yes |
| P8 | Pilot resolution 480×832; 720×1280 only after Exp 4 | Yes |
| P9 | GPU sessions wait for the E5–E7 queue to finish | Yes |

---

## Appendix A: algorithms

**A.1 Anchor detection (prior mode).**
1. **Predict** each anchor's centre and image radius from the prior camera (`r_px = R·f/z`).
   - Skip anchors predicted outside the frame or overlapping the dilated proxy silhouette.
2. **Search a window** of radius `3·r_px + 20` px around the prediction:
   - Convert to Lab.
   - Compute the distance to the anchor's reference colour. That colour is **measured on generated frame 0**, not taken from Blender, because the model shifts colours.
   - Threshold the distance map with Otsu.
   - Split the result into connected components.
3. **Score** each component: circularity × area ratio (vs `π r_px²`) × a Gaussian in its distance to the prediction. Keep the best if it scores ≥ 0.5.
4. **Centre:** the component centroid, refined by an ellipse fit to its contour. Correct the sphere's perspective offset (closed form; < 0.2 px at `r_px ≤ 25`).
5. **Confidence** = the score.

In no-prior mode, used by the probe, all blobs of each hue go into RANSAC with sampled identity hypotheses (4 anchors per hue gives ≤ 256 combinations per 4-point sample).

**A.2 PnP and covariance.**
1. SQPnP inside RANSAC (4 px threshold, 0.999 confidence), then `refine_pnp` (LM) on the inliers.
2. Covariance: `Σ = s² (JᵀJ)⁻¹`, where `s² = RSS / (2n − 6)`. The Jacobian is taken with respect to (rotation vector, camera centre).
3. **Drift is significant** when either holds:
   - the Mahalanobis distance of Δ = (anchor − Blender) exceeds χ²₆(0.999) = 22.5, or
   - |Δrot| > 3σ for ≥ 5 consecutive frames.

**A.3 Warp residual (frame i → j).**
1. Take the proxy depth `D_i` at camera i.
2. For each pixel p in (proxy mask ∩ matte ∖ ignore):
   - back-project `X = D_i(p) · K⁻¹ p` into the world
   - project it into camera j: `p' = π_j(X)`
3. The pixel is valid if both hold:
   - `|z_j(X) − D_j(p')| < 2 cm` (it isn't occluded in frame j)
   - `p'` lies inside frame j's matte
4. The residual is the median, over valid pixels, of `‖Lab_i(p) − Lab_j(p')‖` (bilinear sampling).
5. It's reported for j = i ± 2 and i ± 8, and normalised by the same statistic on Blender's `pretty` pass. That pass is perfectly consistent, so it gives the floor.
6. With P2 the proxy lacks hair, so read the residual *relative to the sequence median*. With P4 it is closer to absolute.

**A.4 Depth control encoding.**
1. View-space depth `d` → inverse depth `1/d`.
2. Pick **one** near/far pair for the whole sequence: the 1st and 99th percentiles over subject + floor within 3 m.
3. Normalise: `v = clip((1/d − 1/d_far) / (1/d_near − 1/d_far), 0, 1)`, so near is white. Pixels with no surface hit are 0.
4. Store as 8-bit grey copied to RGB.
5. Optionally blur with σ = 1 px, to look like estimator depth.

Per-frame normalisation is avoided because it makes the depth scale flicker over the video.

**A.5 Temporal lag.**
- Regress `Δyaw_t = β · ω_t + c`, where `ω_t` is the commanded yaw rate in degrees per frame.
- β is the lag in frames; positive means the output lags the control.
- Report β with a bootstrap confidence interval, per arc.

---

## Appendix B: Fun-Control graph

To be verified against `/object_info` at startup. Settings not listed here come from your FLF workflow.

```
UNETLoader  Wan2_2-Fun-Control-A14B-HIGH_fp8_e4m3fn_scaled_KJ_fixed  -> ModelSamplingSD3(shift) -> model_high
UNETLoader  Wan2_2-Fun-Control-A14B-LOW_fp8_e4m3fn_scaled_KJ_fixed   -> ModelSamplingSD3(shift) -> model_low
CLIPLoader  umt5_xxl_fp8_e4m3fn_scaled (type wan) -> CLIPTextEncode(prompt), CLIPTextEncode(negative)
VAELoader   wan_2.1_vae
VHS_LoadImagesPath  controls/depth/arcK/        -> control_video   (exactly 480x832, 81 frames)
LoadImage           start_arcK.png              -> start_image     (arc 0: frame-0 composite; later: previous arc's last frame)
LoadImage           photo canvas                -> ref_image       (identity reference; same for every arc)
Wan22FunControlToVideo(width 480, height 832, length 81) -> positive, negative, latent
KSamplerAdvanced(model_high, steps N, start 0,   end N/2, add_noise on,  return_with_leftover_noise on)
KSamplerAdvanced(model_low,  steps N, start N/2, end N,   add_noise off)
VAEDecode -> SaveImage(pacgs/<run>/arcK/)
```

**Draft prompt:** "Photorealistic studio video. A young woman stands completely still with both hands resting on top of her head, wearing a white tank top, dark skinny jeans and strappy heels. The camera slowly orbits around her at a constant distance. She stands on a plain grey studio floor inside a ring of small coloured balls. Soft even lighting, sharp focus."

**Negative prompt:** your workflow's negative, plus "camera shake, zoom, walking, turning around, moving arms".

---

## Appendix C: prototype numbers

These come from the scratch scripts written while reviewing the plan. They'll become `pacgs/scene/simulate.py`.

**Real orbit.** Frame 0 is the photo camera (azimuth 359°, elevation −5.4°, 2.77 m from the target, vertical FOV 40°). Frames are 480×832, and the subject is 85% of the frame height with no border contact at 36 azimuths. Anchors are Ø10 cm spheres. "Usable" means fully in frame and ≥ 8 px from the subject's silhouette. Precision is the first-order `σ²(JᵀJ)⁻¹`, given as the median over frames.

| Rig | Usable anchors (min / median) | σ = 1 px (rot / centre) | σ = 2 px | σ = 3 px |
|---|---|---|---|---|
| **Floor ring r 0.45 m, 24 spheres, centres at 0.05/0.30 m** | **17 / 19** | 0.18° / 0.8 cm | **0.35° / 1.7 cm** | 0.53° / 2.5 cm |
| Floor ring r 0.60 m, 24 spheres, 0.05/0.30 m | 11 / 12 | 0.15° / 0.7 cm | 0.31° / 1.5 cm | 0.46° / 2.2 cm |
| Floor ring r 0.75 m, 24 spheres, 0.05/0.30 m | 6 / 7 | 0.22° / 1.3 cm | 0.44° / 2.6 cm | 0.66° / 3.8 cm |
| r 0.60 m + top ring r 0.9 m × 8 at 2.15 m | 11 / 14 | 0.13° / 0.7 cm | 0.25° / 1.3 cm | 0.38° / 2.0 cm |
| r 0.60 m, centres at 0.05/0.20/0.45 m | 10 / 12 | 0.15° / 0.7 cm | 0.30° / 1.4 cm | 0.46° / 2.2 cm |

**Framing sweep** (idealised look-at orbit, elevation 10°, 2.77 m, best rig per row, σ = 1 px). Making the subject smaller buys very little precision:

| Vertical FOV | Subject / frame height | Best rig | Rotation / centre σ |
|---|---|---|---|
| 40° | 0.85 | floor r 0.6 m × 24 | 0.20° / 1.1 cm |
| 46° | 0.73 | floor r 0.6 m × 24 + top r 1.3 m × 8 | 0.14° / 0.8 cm |
| 52° | 0.64 | floor r 0.9 m × 24 + top r 0.9 m × 8 | 0.13° / 0.7 cm |
| 58° | 0.56 | floor r 0.9 m × 24 + top r 0.9 m × 8 | 0.13° / 0.7 cm |

For scale: the current v2 keypoint cameras are 5–12 cm off, and render-and-compare fitting (`i2g/reregister.py`) moves them 1.5–15 cm (quality_plan.md).

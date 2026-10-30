# Quality plan: a better splat on this PC

**Scope:** make `character.ply` look better from every angle, running entirely on this PC
(RTX 4090 24 GB, 32 GB RAM, ComfyUI + the v2 pipeline).

**Baseline:** runs 1 and 2 on `test-assets/test-image.png` (2026-09-18). See [plan.md](plan.md).

**How to read this:** items are ranked from **most confident** to **least confident**. Each one names
the artifact it targets, what to change, what it costs, whether it fits the PC, and how we'll know it
worked. The confidence figure is my estimate that the item **visibly improves the result without
making anything else worse**.

---

## Results of the 2026-09-19 quality sprint (measured)

Every change below was trained on the same test image and views, then scored by the harness
(`i2g/evaluate.py`, run through `i2g/experiment.py`):

- **photo dB / HR dB:** masked PSNR against the photo, at canvas size (a typical viewer size) and at
  the photo's native resolution.
- **face id:** mean ArcFace similarity (InsightFace buffalo_l) between the photo's face and the splat's
  face in 15 close-ups, from 30° left to 30° right and up to 30° above the photo's view.
  **face min** is the worst of those 15.
- **hold:** 5 generated views never used in training (back, right, back-left, front-right,
  left-elevated). Each one's camera is first fitted to the splat, because their own keypoint cameras
  are 5–12 cm off.
- **needles:** Gaussians over 8× longer than wide and over 3 cm long.

<!-- RESULTS_TABLE -->
| Run | What changed | photo dB | HR dB | face id | face min | hold dB | hold SSIM | hold VGG ↓ | hold IoU | needles |
|---|---|---|---|---|---|---|---|---|---|---|
| E0_run2 | baseline: run 2 (SH 1, HR photo, 11 views) | 30.19 | 41.45 | 0.257 | -0.023 | 20.06 | 0.801 | 0.055 | 0.936 | 548 |
| E1_sh0_ms | + flat colour (SH 0) + multiscale | 37.25 | 35.67 | 0.254 | -0.029 | 20.25 | 0.804 | 0.056 | 0.936 | 804 |
| E2_face | + face prior: depth, mesh start, photo-only face colour 0.25 | 37.33 | 35.61 | 0.222 | 0.000 | 20.35 | 0.804 | 0.057 | 0.935 | 686 |
| E3_shell | E2 + face shell | 37.22 | 35.59 | 0.242 | -0.004 | 20.24 | 0.803 | 0.057 | 0.935 | 634 |
| E3_aniso | E2 + anti-needle | 37.29 | 35.44 | 0.255 | 0.045 | 20.25 | 0.803 | 0.057 | 0.935 | 0 |
| E3_pose | E2 + camera refinement | 37.08 | 35.80 | 0.245 | -0.015 | 20.45 | 0.811 | 0.056 | 0.937 | 724 |
| E4a_cand | shell + anti-needle + cameras, face colour 0 | 36.91 | 35.45 | 0.340 | 0.031 | 20.14 | 0.803 | 0.058 | 0.936 | 0 |
| E4b_cand_fit15 | E4a on fitted cameras, 15 views (incl. 90° side) | 36.76 | 36.32 | 0.259 | 0.000 | 20.50 | 0.809 | 0.056 | 0.943 | 0 |
| E4c_interior | E4a + interior clearing (rejected) | 36.70 | 35.47 | 0.290 | 0.000 | 19.37 | 0.781 | 0.065 | 0.913 | 0 |
| E6_final_fit15 | final recipe: E4b + inner-face core + face bake (v1) | 35.41 | 35.19 | 0.526 | 0.326 | 20.73 | 0.816 | 0.054 | 0.945 | 0 |
| E6_pixbake | final recipe with the per-pixel face bake | 35.59 | 35.60 | 0.575 | 0.364 | 20.68 | 0.816 | 0.054 | 0.945 | 0 |
| E7_restore | E6 + 96 SeedVR2-restored orbit frames | 35.49 | 36.37 | 0.621 | 0.427 | 20.23 | 0.811 | 0.056 | 0.943 | 0 |
| FINAL_pipeline | **the pipeline itself, standard preset, end to end, face aligned** (hold-out columns not comparable: its second pass trains on 2 of the 5 hold-out views) | 35.68 | 35.58 | 0.583 | 0.344 | 23.69 | 0.856 | 0.042 | 0.950 | 0 |

### What each finding means

| Finding | Evidence | Now |
|---|---|---|
| The "HR photo" of run 2 made the splat alias at normal viewing sizes (darker, blotchy face) | photo dB 30.2 at canvas size vs 41.5 at native size | `multiscale=0.5`: random zoomed-out steps. Canvas size is now 37 dB, with no aliasing |
| Teal/green streaks between the legs came from view-dependent colour (spherical harmonics) extrapolating to angles no view covers | rendering run 1 without its SH band 1 removes them | `sh_degree=0` |
| Calf spikes are needle Gaussians hidden in the 74°→125° and 234°→284° gaps. They sit inside the visual hull, so hull carving does nothing | 54% of run 2's Gaussians over 6:1, 1% over 1000:1. Hull carving removed 2 | `aniso_weight=0.1` (+ `needle_ratio=10` cleanup): 0 needles left |
| Keypoint cameras are loose: render-and-compare fitting moves them 1.5–15 cm and raises silhouette agreement from ~0.90 to ~0.95 | `i2g/reregister.py` on 38 views | in-training refinement (`pose_lr=5e-4, pose_reg=0.1`) plus the new second pass (views re-selected 12° apart) |
| Camera words in the multi-angle LoRA name the *target* view, and the model gets closer to it from a nearby view | "right side view" from the photo lands at 74–77°. From the 45° view it lands at **91°** | "full" preset adds chained views (r2, l2, br2, bl2). "standard" adds r2, l2 |
| Pure profile views confuse the SAM 3D keypoints (the far limbs are hidden) | r2 at 90°: keypoint drift 11–13 cm, rejected | rescued by camera fitting (IoU 0.64 → 0.86) |
| The broken face off-angle is a thick cloud plus junk *inside* the head showing through a semi-transparent face | depth prior: 36 mm → 0.7 mm error, still ghosted. Removing the face layer reveals random colours inside the head | face shell + face bake. Interior clearing was tested: body-wide it removed real surface (hold-out −0.8 dB), so it is off |
| Face bake: the photo unprojected onto the SAM 3D face, one flat Gaussian per photo pixel | face id 0.34 → 0.53, worst face 0.03 → 0.33 (first version, random 0.5 mm samples). The per-pixel version also keeps the front view sharp | `face_bake` |
| The body mesh's head sat **~2.7 cm too far forward** relative to where the generated side views put it. Pinned to the mesh, the face floated in front of the hair like a mask. (Spotted by eye on the first "after" renders.) | Every prior-free splat put the face 31–35 mm behind the mesh face. A tight visual hull around the head needs a 24–27 mm push before 98% of the mesh face fits the side views' silhouettes | `face_align`: before any face prior, the head is moved back along the photo camera's rays by the measured amount (the photo view stays pixel-identical). Check, with each view's camera fitted per model: head pixels rendered outside the side views' silhouettes fell to 3.4% (was 4.4% with the face on the raw mesh, 4.6% for run 2) |
| **...but with the orbit pass, the push is wrong.** The orbit pass films the second-pass splat as a depth video and repaints it, so the face lands wherever that splat put it; with `face_align` every orbit result had its face 1.9–5.0 cm too far back (the push on ten runs), and side profiles showed it sunk behind the raised arm (spotted by eye on GUI runs) | test image, same world frame for every model: nose tip 9.7 cm with the push, 12.0 cm for the research model that never pushed, 12.5 cm on the body model | `face_align` is off whenever the orbit pass runs (`make_splat_config`); `--no_orbit` keeps it |
| SeedVR2 cannot share the GPU with training on this PC (VRAM spills into system RAM, everything slows 5–20×) | observed twice | pipeline runs ComfyUI work and training strictly one after another |

---

## Summary

| # | Change | Main target | Confidence | Dev time | Extra runtime per image |
|---|---|---|---|---|---|
| 0 | Measurement harness (fixed renders + metrics) | makes every other step checkable | 95% | 1–2 h | +2 min |
| 1 | Anti-streak: anisotropy loss + needle cleanup + scale cap | side-view streaks behind the legs | 85% | 30–45 min | none |
| 2 | Photo owns what it sees (face and front weighting) | blotchy or doubled face off-angle | 75% | 1–2 h | +5% train |
| 3 | Turn camera refinement on properly (photometric alignment) | general blur and ghosting | 70% | 1 h | +2 min |
| 4 | Better coverage: more candidates, low/high angles, gap-filling selection, drop the worst views | gaps at 74°→125° and 234°→284°, top of head, armpits | 70% | 1–2 h | +20–30 min, one-off per image |
| 5 | Render → repair → retrain loop (Difix3D+ or FLUX.2 klein) with known cameras | everything between the views: streaks, face, hair | 65% | 3–4 h | +20–25 min |
| 6 | MCMC densification with a Gaussian budget + opacity/scale regularisation | floaters, run 2's over-fitting | 65% | 1–2 h | ≈ same |
| 7 | Higher-quality Qwen views (no Lightning, 1.5 MP, reference to the nearest accepted view) | soft side and back detail, cross-view consistency | 55–60% | 1–2 h | +15–30 min |
| 8 | Geometry priors: Sapiens2 normals + SAM 3D Body mesh depth | flat face, lumpy surfaces, floaters | 50% | 3–4 h | +5 min |
| 9 | Wan 2.2 first→last-frame in-between views | remaining angular gaps | 45% | 3–4 h | +30–40 min |
| 10 | Head and hair close-up views | hair and back-of-head detail | 45% | 1 h | +10 min |
| 11 | Colour normalisation pre-pass | slight colour and exposure drift between views | 40% | 1 h | +1 min |
| 12 | Generator bake-off (FLUX.2 klein 9B, Wan Fun-Control, MiniMax H3) | overall consistency | 35% | 1 day | varies |
| 13 | 4DAnyone at 41 frames (native, via SplatKit's backend) | reconstruction-grade human views | 25–30% | 0.5–1 day | +30 min |
| 14 | Train our own multi-view LoRA on 3D human renders | camera accuracy + consistency | 25% | days | none after training |

**Suggested sprints:**
- **Sprint 1:** items 0–4. These are high-confidence and fix all three visible problems.
- **Sprint 2:** items 5–7.
- **Sprint 3:** items 8–11.
- **Research bets:** items 12–14.

**Status after the 2026-09-19 sprint** (numbers in "Results" above):

| # | Item | Status |
|---|---|---|
| 0 | Harness | **done**: `i2g/evaluate.py` + `i2g/experiment.py` |
| 1 | Anti-streak | **done**: anisotropy loss + needle cleanup, 0 needles left. Hull carving was tried and did nothing (the spikes sit inside the hull) |
| 2 | Photo owns the face | **done**, and went further: face prior, face shell, inner-face ownership, and the **per-pixel face bake**. This is the biggest visible win |
| 3 | Camera refinement | **done**: in training (`pose_lr 5e-4`, `pose_reg 0.1`), plus render-and-compare fitting in the second pass |
| 4 | Coverage | **done**: chained gap views (true sides at ~90°/260°, back diagonals), low angles in "full", 12° spacing after fitting |
| 5 | Render → repair → retrain | **done with SeedVR2** (`--restore`, high preset). Best face and HR front, slightly lower hold-out agreement |
| 6 | MCMC | not evaluated (implemented as `strategy=mcmc`, smoke-tested only) |
| 8 | Geometry priors | body-mesh face depth and shell are in. Body-wide interior clearing hurt and is off |
| 7, 9–14 | | not started |

---

## Baseline: what's wrong now (evidence)

| Artifact | Where | Evidence | Likely cause |
|---|---|---|---|
| Dark and teal streaks behind the calves | pure side views (90°, 270°) | turntable frames in both runs | angular gaps (no view between 74° and 125°, or 234° and 284°), plus long needle Gaussians that satisfy the sparse views |
| Blotchy or doubled face | front seen about 10° above the photo | `runs/run1_vs_run2.png`; worse in run 2 | generated front-ish views (45°, 327°, elevated front) disagree on facial features; only the photo is "true" |
| White smear in the back of the hair | back view | turntable | back, back-quarter and elevated-back views disagree on hair |
| General softness in generated regions | sides and back | visual | 1 MP Lightning 4-step views; views mis-registered by up to 1–2% of image height |
| Some views conflict with the rest | `bl` 24.2 dB, `br` 26.1, `f_hi` 28.0, `r_hi` 27.8 masked PSNR | `report.json` | view inconsistency and/or residual misregistration |
| Camera refinement effectively off | max 0.37° / 0.2 cm correction | `report.json` `pose_refinement` | `pose_reg=10` is too strong for `pose_lr=1e-4` |
| More Gaussians didn't mean better | run 2 (407k) sharper texture but a worse off-angle face than run 1 (37k) | turntables | un-regularised capacity fits the conflicts |

---

## Tier 1: high confidence (do first)

### 0. Measurement harness (95%)

**Why:** without numbers, run 1 vs run 2 was a judgement call. Every later change needs a pass or
fail.

**What:**
- **`i2g/evaluate.py` + `--stage eval`**, which writes `06_train/eval.json` and `compare.png`.
- **Fixed render set:** 24 cameras (8 azimuths × elevations {−10°, +10°, +30°}) plus a 4×4 face grid
  (±20° azimuth × {0, +10, +20}° elevation). The images are identical across runs, so any two runs can
  be compared side by side.
- **Hold-out consistency:** train with 2 accepted-but-unselected views held out (e.g. back 185° and
  right 77°), then report LPIPS/PSNR on them. This is 4DAnyone's "generated-view consistency"
  protocol.
- **Face identity:** ArcFace cosine (InsightFace `antelopev2` is already on disk) between the photo's
  face and the rendered face at each face-grid camera.
- **Streak count:** Gaussians with longest/second axis > 8 and longest axis > 4 cm.
- **Floaters:** rendered alpha outside the dilated visual hull, as a % of pixels, over the 24 renders.

**Success:** `compare.png` and `eval.json` are produced for both existing runs. Every metric
reproduces within ±2% on a rerun with the same seed.

**Fits the PC:** trivially (renders only).

### 1. Anti-streak regularisation (85%)

**Why:** the side streaks are long thin Gaussians that line up with a training ray. They look right
in the training views and wrong between them.

**What (`i2g/splat.py`):**
- **Anisotropy loss (PhysGaussian style):**
  `L_aniso = mean(relu(max_scale / min_scale − r)), r = 6, weight 0.1`, on from step 500.
- **Scale cap:** `prune_scale3d` 0.1 → 0.04 m, so nothing wider than 4 cm survives densification.
- **Post-train needle cleanup** (ported from SplatKit's `clean.py`, MIT): drop Gaussians with
  longest/second axis > 8 and longest > 3 cm. Keep the existing matte-support cleanup, with support
  0.6 → 0.7.

**Success:** the streak count falls by at least 90%. Side-view renders show no dark or teal stripes.
Hold-out LPIPS does not get worse.

**Risk:** thin hair strands and bracelet edges soften slightly. Mitigate by exempting Gaussians inside
the head/hand region, or by using r = 8.

**Fits the PC:** no change (training peak ~3.5 GiB).

### 2. The photo owns what it sees (75%)

**Why:** the photo is the only real observation. Wherever it sees the surface (face, chest, front of
the jeans), generated views should not be allowed to overwrite it with their own guesses. That
conflict is the doubled or blotchy face.

**What:**
- **Version A** (30 min, do first): per-pixel loss weights for generated views.
  1. Project the scaffold's face keypoints with each view's registered camera.
  2. Inside that head disc, for views within 90° of the photo's azimuth, the weight drops to 0.1.
  3. Apply the weights to L1, SSIM and the matte loss. For the perceptual loss, replace masked pixels
     in the target with the prediction.
- **Version B** (1–2 h): a general visibility test.
  1. Every 2k steps, render depth for the photo and for each generated view.
  2. For each generated-view pixel, back-project its surface point.
  3. If the photo sees that point (depth test within 2 cm), the pixel's weight becomes 0.2.
  4. This covers the whole front, not only the face.
- **Keep:** training the photo at native 1866×2799.

**Success:** ArcFace similarity on the face grid rises noticeably, especially at +10°/+20°. No
doubled eyebrows or mouth in `compare.png`.

**Risk:** the front geometry is then constrained by one view. If the face looks flat off-angle, that's
item 5's job (repaired off-angle face views) and item 8's (face mesh prior).

**Fits the PC:** Version B adds 2 depth renders every 2k steps: negligible.

### 3. Real camera refinement (70%)

**Why:** registration leaves 12–23 px keypoint RMSE (1–2% of the image height). Some of that is pose
differences, but some is camera error. The trainer barely corrected anything (≤ 0.37°) because the
refinement is over-regularised. Misaligned views average into blur and double edges.

**What:**
- **Staged schedule:**
  - steps 0–2k: geometry and colour only
  - steps 2k–8k: camera deltas on, `pose_lr` 5e-4, `pose_reg` 0.1
  - after 8k: camera deltas frozen, detail refinement continues
- **Log** each view's correction in `report.json`, and flag any view that moves more than 5° or
  10 cm. That's a registration failure to look at, not something to hide.
- **Optional pre-pass:** align each generated view's camera to a quick 3k-step splat of the other
  views before the main run (leave-one-out alignment).

**Success:** masked PSNR on the conflicting views (`bl`, `br`, `f_hi`, `r_hi`) rises by at least
1.5 dB. Hold-out LPIPS improves. Edges look sharper in `compare.png`.

**Risk:** cameras can "explain away" real inconsistencies. The 5°/10 cm flag and the hold-out views
catch that.

**Fits the PC:** negligible.

### 4. Better coverage and view selection (70%)

**Why:** the side gaps (74°→125°, 234°→284°) cause streaks. The top of the head, the armpits and the
underside of the raised arms are barely seen. The four weakest views drag down everything near them.

**What (`i2g/views.py`, `i2g/register.py`):**
- **Generate more:** 8 azimuths × {eye-level, elevated (30°), **low-angle (−30°)**, **high-angle (60°)**
  for front/back/sides}, with 3 seeds each.
- **Cost:** about 57 s per new prompt plus about 6 s per extra seed ≈ 25–30 min per image, once.
  Registration is cached.
- **Gap-filling selection:** replace "best-first with 20° spacing" by greedy farthest-point coverage on
  the view sphere. Each candidate is scored by registration quality **and** by its consistency with a
  quick draft splat: render its camera and measure LPIPS against the candidate. Keep 14–18 views.
- **Automatic regeneration:** any selected view whose draft-splat PSNR is more than 4 dB below the
  median is regenerated with a new seed, or dropped.
- **Target:** no eye-level azimuth gap over 35°, elevation coverage from −30° to +60°.

**Success:** side-view streaks gone (with item 1). The hold-out views and the face grid improve. The
top of the head is no longer blobby.

**Risk:** more views bring more disagreement. The consistency score is what stops that.

**Fits the PC:** ComfyUI's Qwen peaks around 21 GB, same as now.

---

## Tier 2: medium confidence (bigger potential payoff)

### 5. Render → repair → retrain loop (65%)

**Why:** this is the most promising way past "sparse, slightly inconsistent views".
1. Render the current splat from **known** cameras between and around the views: 32 cameras, 16
   azimuths × elevations {0°, +20°}, plus the face grid.
2. Let a diffusion model **repair** each render: remove streaks and blur, keep geometry.
3. Train on the repaired images at low weight, using their exact cameras (no registration needed).

Consistency comes from the 3D render, and the diffusion model only fixes artifacts. This is the idea
behind NVIDIA's Difix3D+ (CVPR 2025) and GaussianObject.

**Repair model options (both run locally):**
- **Difix3D+** (`nvidia/difix_ref`, single-step, built on SD-Turbo, NVIDIA non-commercial licence; fine
  for personal use).
  - Trained specifically to remove 3DGS/NeRF render artifacts.
  - The `difix_ref` variant takes a **reference image**; we pass the photo or the nearest anchor view.
  - Fast: about 0.1–0.5 s per image. Small: well under 8 GB.
  - Needs `diffusers` in `.venv-i2g`, plus the weights (one Hugging Face download).
- **FLUX.2 klein 9B** img2img through ComfyUI (already installed): denoise 0.25–0.4, photo as a
  reference image, prompt "same woman, sharp photo, remove artifacts". Slower, but stronger on faces
  (your `bfs_head` LoRA could help keep the face on-identity).

**Loop:** base train (15k) → render 32 views → repair → add at weight 0.3 → fine-tune 5k steps.
Repeat 2 rounds. Repaired views never override the photo; item 2's weighting applies to them too.

**Success:** the face grid's ArcFace score and turntable quality both improve. Hold-out LPIPS drops by
≥ 10%.

**Risk:**
- Too little denoise fixes nothing; too much invents inconsistent detail. Tune on the harness.
- Difix was trained on scenes, not people, so it may be weaker on faces. That's why FLUX.2 klein is
  the backup.

**Fits the PC:** yes. Difix needs < 8 GB; klein 9B fp8 in ComfyUI about 17 GB. Run it with no other
GPU user, as now.

### 6. MCMC densification with a budget (65%)

**Why:** run 1 (37k) vs run 2 (407k) showed that capacity without regularisation fits the conflicts.
gsplat's `MCMCStrategy` ("3DGS as MCMC") keeps a fixed Gaussian budget and relocates dead Gaussians
instead of cloning. It usually gives fewer floaters and better quality per Gaussian.

**What:**
- `MCMCStrategy(cap_max=300k, refine_start_iter=500, refine_stop_iter=12k, refine_every=100,
  min_opacity=0.005)`
- `opacity_reg 0.01`, `scale_reg 0.01`, initial opacity 0.5
- sweep `cap_max` ∈ {150k, 300k, 600k} on the harness

**Success:** floater % and hold-out LPIPS both beat the best DefaultStrategy run. The PLY is smaller
than run 2's.

**Fits the PC:** same or less VRAM than run 2.

### 7. Higher-quality Qwen views (55–60%)

**Why:** side and back detail can never beat the generated views. The current views are 1 MP, 4-step
Lightning.

**What (bake-off on 4 angles first, using the harness):**
- **(a)** No Lightning: 25–30 steps, true CFG 2.5–4 with an empty negative. About 30–40 s per image
  plus the model swap.
- **(b)** 1024×1536 output (1.57 MP) instead of 832×1248.
- **(c)** RCP-style multi-reference: image1 = photo, image2 = nearest accepted view.
- **(d)** Prompt suffix: "same person, same outfit, same pose, photorealistic, sharp focus, plain grey
  background".

**Success:** the per-view sharpness ratio rises and hold-out consistency doesn't drop.

**Risk:** the angles LoRA was trained with one input image and a certain resolution, so (b) and (c)
might weaken angle control. Registration will show it (prior deviation).

**Fits the PC:** same ComfyUI footprint. Slower per image.

### 8. Geometry priors: normals and body mesh (50%)

**Why:** the face and body surfaces are constrained mainly by colour from a few views. A normal or
depth prior gives smoother, more plausible surfaces off-angle and fewer floaters.

**What:**
- **Sapiens2** normal estimation (1B, Apr 2026, torch-only) on the photo and every selected view.
  Render normals from the Gaussians (from rendered depth) and apply an L1 normal loss inside the
  matte, weight 0.05.
- **SAM 3D Body** mesh, already fitted: a depth-order constraint that no visible skin surface lies
  more than 2 cm *inside* the body mesh on the arms, neck and face. The face region of the mesh gives
  the face real depth.

**Success:** fewer floaters; the face profile looks correct at ±45° in `compare.png`.

**Risk:** clothes and hair deviate from the body mesh, so apply the mesh prior only on skin
(body-part segmentation from Sapiens2).

**Fits the PC:** Sapiens2-1B needs about 4 GB of VRAM, run once per view.

### 9. Wan 2.2 in-between views (45%)

**Why:** a video model's frames are consistent with each other, so first→last-frame clips between
adjacent accepted views fill the angular gaps smoothly (plan.md M4).

**What:**
- For each adjacent pair of eye-level views, generate a 33-frame first→last-frame clip at 480×832
  with the Lightning 4-step LoRAs (already installed).
- Prompt: "camera slowly orbits the standing woman, she stays perfectly still".
- Register every 4th frame (SAM 3D Body + our solver). Initialise cameras by interpolating between
  the two end cameras.
- Train at weight 0.2.

**Success:** coverage gaps close with no new pose ghosting (hold-out stable or better).

**Risk:**
- The clip morphs between two anchors that disagree.
- The pose wobbles (arms) mid-clip; the drift check rejects those frames.
- It's the slowest option: about 3–5 min per clip, 8 clips.

**Fits the PC:** Wan 2.2 Q8 GGUF + Lightning at 480p fits (~20 GB with ComfyUI offload).

### 10. Head and hair close-ups (45%)

**What:**
- Qwen with the `close-up` distance token at the back, both backs-quarter, both sides and elevated
  back.
- Register them on the head keypoints and train at weight 0.5, with a head-crop perceptual loss.
- The front face stays photo-owned (item 2).

**Success:** the hair smear goes away; the back of the head gets real strand detail.

**Fits the PC:** same as item 4.

### 11. Colour normalisation (40%)

**What:** before training, match each generated view's per-channel mean and std, inside
skin / tank-top / jeans segments (Sapiens2 body parts or simple k-means on the matte), to the photo.
The trainer's per-view colour correction then only handles what's left.

**Success:** smaller learned colour corrections; slightly cleaner seams between front and back.
Small gain, cheap.

---

## Tier 3: research bets (lower confidence, possibly large payoff)

### 12. Generator bake-off (35%)

- **FLUX.2 klein 9B** multi-reference + `Flux2-Klein-9B-consistency-V2` (your favourite image model).
  Angles come from the prompt; registration measures them anyway.
- **Wan 2.2 Fun-Control** driven by occlusion-aware 3D skeleton renders of the SAM 3D Body scaffold
  along an exact orbit. Cameras are known by construction; this is 4DAnyone's skeleton-conditioning
  idea done with off-the-shelf models.
- **MiniMax H3** reference-to-video with the accepted views as references, *if its licence covers your
  location*.
- Decide with the harness: hold-out LPIPS, then ArcFace, then time.

### 13. 4DAnyone at 41 frames (25–30%)

The 4DAnyone paper trains 6-view groups on 41-frame clips, and a still photo doesn't need 121 frames.
About 3× fewer video tokens could bring SplatKit's native-Windows 4DAnyone under the VRAM ceiling that
made it crawl at 554 s/step.

- **What it needs:** a frame-count patch in SplatKit's vendored generator.
- **Stop rule:** if the first denoising step takes more than 60 s, abandon it.
- **If it works:** human-specific, reconstruction-grade views, the best consistency of anything here.

### 14. Train our own multi-view LoRA (25%)

HumanOrbit (Feb 2026) trained a rank-32 Wan I2V LoRA on 3,000 orbit videos rendered from 500 human
scans, on one A100.

- **On the 4090:** a Wan 2.2 LoRA at 480p with block swap, via AI Toolkit / musubi-tuner.
- **Or:** a Qwen-Edit LoRA on our own camera convention.
- **Needs:** a scan dataset (THuman2.1, CustomHumans: licence agreements) and days of preparation
  and training.
- **Payoff:** exact camera control and higher consistency for every future image.

---

## Already ruled out

- **More Gaussians alone:** run 2 showed sharper texture but worse novel views.
- **SH degree 3:** overfits view-dependent colour between sparse views. We use degree 1, padded to 3
  in the PLY for viewer compatibility.
- **Upscaling generated views independently** (SeedVR2 etc.): each view invents its own micro-detail,
  and the splat averages it into mush. Upscale only through item 5, where the render supplies
  consistency.
- **SageAttention:** speed only (3.1× faster attention), no quality change. Fine to enable once
  checked on Qwen.

---

## Definition of "better"

A change is kept only if the harness (item 0) shows all three of these:

1. **Hold-out LPIPS** does not get worse, and at least one of face ArcFace, streak count or floater %
   improves.
2. **`compare.png`** is at least as good side by side, reviewed at 0°, 90°, 180° and the face grid.
3. **Runtime** stays within this PC's budget: every stage ≤ ~20 GB VRAM, and a full image run
   ≤ ~90 min.

Sprint 1 target, versus run 1:
- side streaks gone
- face ArcFace at +10°/+20° ≥ the photo-view value − 0.1
- hold-out LPIPS −15%

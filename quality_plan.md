# Quality plan: a better splat on this PC

**Scope:** make `character.ply` look better from every angle, running entirely on this PC
(RTX 4090 24 GB, 32 GB RAM, ComfyUI + the v2 pipeline).

**Baseline:** runs 1 and 2 on `test-assets/test-image.png` (2026-09-18). See [plan.md](plan.md).

**How to read this:** items are ranked from **most confident** to **least confident**. Each one names
the artifact it targets, what to change, what it costs, whether it fits the PC, and how we'll know it
worked. The confidence figure is my estimate that the item **visibly improves the result without
making anything else worse**.

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

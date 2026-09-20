# PAC-GS sprint 1: proxy-controlled orbit video → splat (results)

**Date:** 2026-09-19. **Subject:** `test-image` (the same photo as every v2 experiment).
**Scope:** the proxy-controlled video part of the research plan only. The Super Anchors are parked, as
agreed. Design and background: [pacgs_implementation_plan.md](pacgs_implementation_plan.md).
**Code:** `pacgs/` (run `python -m pacgs --help`). **Run folder:** `output/pacgs/test-image/r1/` (gitignored).

## What was built

1. **Orbit cameras from the photo's camera.**
   - Frame 0 *is* the photo's camera (SAM 3D Body fit). The orbit circles the vertical axis through the
     person at the photo's distance, height and pitch.
   - Check: the photo's matte vs the proxy's silhouette at frame 0 gives IoU 0.99.
2. **Depth control video from a 3D proxy,** rendered with the trainer's own gsplat renderer, so control
   and training cameras share one projection.
   - **splat** proxy: the best v2 splat (E6_pixbake), which has hair and clothing volume.
     - Depth is rendered with sharpened opacities and median-filtered.
     - Plain expected depth put dents in the semi-transparent hair.
   - **mesh** proxy: the SAM 3D Body mesh (no hair). Rendered, but not generated from yet.
   - Encoding: inverse depth, near = white, one normalisation for the whole orbit.
3. **Wan 2.2 Fun-Control A14B** (your KJ fp8 weights) through ComfyUI's API.
   - A new helper node `I2G_Wan22FunControlToVideo` exposes the `start_image` input. The core node
     computes it but doesn't list it. Result: the video starts on the real photo, with the photo also
     as the reference image.
   - Settings are from ComfyUI's official Fun-Control template.
   - Output: 81 frames at 480×832 for one 360° orbit.
4. **Training set.** BiRefNet mattes of every frame, each frame's exact orbit camera and its own K, plus
   the native-resolution photo at its exact camera. The v2 trainer is used with a new frame kind,
   `video`, and settings = the best v2 config (E6_final_fit15).
5. **Scores.**
   - Leave-frames-out (LFO): 10 of 79 frames held out, at their orbit cameras and with cameras refitted.
   - The v2 harness: photo fidelity, face identity on a 15-camera face grid, needles, floaters,
     sharpness.
   - A turntable.
6. **Frozen snapshot of the v2 inputs** in `r1/subject/`, so later v2 runs in `output/test-image/v2`
   can't shift the cameras under this run.

## Results

| Splat | Held-out frames (exact / refitted cameras) | Photo dB | Photo HR dB | Face id (mean / worst) | Floaters % | Sharpness |
|---|---|---|---|---|---|---|
| v2 E6_pixbake | — (its Qwen hold-outs: 20.7 dB) | 35.59 | 35.60 | 0.575 / 0.364 | 0.031 | 14.2 |
| v2 E7_restore | — (its Qwen hold-outs: 20.2 dB) | 35.49 | 36.37 | 0.621 / 0.427 | 0.026 | 12.9 |
| **PAC-GS g1_A**: Lightning 4-step, exact cameras | **30.4** / 27.7 | **36.24** | 34.69 | 0.600 / 0.299 | **0.014** | 13.5 |
| g1_C: same video, trainer camera refinement on | 23.6 / 25.2 | 35.66 | 35.62 | 0.521 / 0.246 | 0.109 | 10.3 |
| g4_A: 20 steps at CFG 3.5, exact cameras | 25.3 / 24.4 | 36.07 | 34.86 | 0.585 / 0.300 | 0.018 | 13.0 |

The v2 hold-outs and the PAC-GS hold-outs use different protocols, so compare each column's trend, not
the two numbers directly.

**Visual comparison:** `r1/report/compare.png`.
- **Face 30° to the side.** v2 (E6, E7) shows the known patchy, doubled skin and chest smears. **PAC-GS
  g1_A has a clean, coherent face.**
- **Face from 15° above.** v2's hairline smear is gone.
- **Back and sides.** Clean for all.
- **Top 40°.** g1_A shows light streaks on the arms and crown, because nothing in the orbit looks from
  above. This is its one clear weakness.

Turntable: `r1/eval/g1_splat360_l4_A/turntable.mp4`. Splat: `r1/splats/g1_splat360_l4_A/character.ply`.

## Round 2: more camera paths (2026-09-20)

Two more clips, same method, different camera paths:

- **face**: the same orbit zoomed in on the head, the principal point tracking it, so the head fills the
  frame instead of being 40 px wide. Frame 0 is an exact crop of the native-resolution photo.
- **helix**: the same orbit rising to +30° over the first quarter turn, so the top of the head and the
  shoulders are seen.

| Splat | Clips | Held-out | Photo dB | Photo HR dB | Face id (mean / worst) | Floaters % | Sharpness | Gaussians |
|---|---|---|---|---|---|---|---|---|
| v2 E7_restore | – | – | 35.49 | 36.37 | 0.621 / 0.427 | 0.026 | 12.9 | 240k |
| v2 FINAL_pipeline (face-aligned) | – | – | 35.68 | 35.58 | 0.583 / 0.344 | 0.145 | 14.3 | 262k |
| g1_A | body | 30.4 | 36.24 | 34.69 | 0.600 / 0.299 | 0.014 | 13.5 | 269k |
| p2_body_helix | body + rising | 28.2 | 36.19 | 35.04 | 0.609 / 0.291 | 0.015 | 12.7 | 269k |
| p3_body_face | body + close-up | 25.4 | 35.55 | 35.06 | 0.629 / 0.402 | 0.017 | 13.6 | 410k |
| **p1_all_fw03** | **all three** | 25.0 | 36.08 | **35.25** | **0.642 / 0.408** | 0.016 | 12.9 | 358k |

Held-out scores are only comparable within one dataset: adding close-up and elevated frames makes the
held-out set harder, which is why the number falls as quality rises. The face grid and the visual
comparison are the meaningful columns here.

**What the clips are worth:**

- **The head close-up is the big one.** Worst-angle face identity goes from 0.30 to 0.40, and the
  average to 0.64, the best of any run in the project (v2's best was 0.62, with a SeedVR2 pass).
  Visually: real skin texture and hair strands at 30° off-centre, where v2 is blotchy and the
  body-only splat is waxy. The clip itself scores 0.61 for face identity against the photo, versus
  0.26 for the full-body clip, and follows its control best (IoU 0.98).
- **The rising clip** cleans up the view from above (fewer streaks on the arms and crown) and lifts the
  native-resolution photo score. On its own it does nothing for the face.
- **Together** they also close part of the native-resolution gap to v2 (34.69 → 35.25 dB).
- With the close-ups present, the face weighting is relaxed (`face_weight` 0 → 0.3), so the generated
  faces contribute instead of the photo owning the face core alone.

**This is now the product default.** `run_image_to_splat.bat` runs the orbit pass on all three paths in
the standard and high presets (`i2g/orbit_stage.py`, stage `orbit`, `--no_orbit` to skip).

## Round 3: the off-angle face (2026-09-20)

The head close-ups exposed a problem the earlier pipeline had been hiding: the face looked right only
from the photo's viewpoint and was speckled everywhere else. The cause was the baked photo layer, plus
the mesh-face priors and uniform view sampling that came with it. Full write-up, with the measurement
method, the mechanism, the ablations and a negative result: **[pacgs_face_fix.md](pacgs_face_fix.md)**.

Headline, measured over ten cameras around the head against the close-up frames themselves:

| | Agreement | Detail | Speckle |
|---|---|---|---|
| Before | 25.4 dB | 0.71× | 1.09 |
| After | 31.8 dB | 0.68× | 0.85 |

## Findings

1. **Proxy-controlled video is almost as self-consistent as a rendered splat.**
   - Held-out generated frames reach 30.4 dB at their exact cameras.
   - For scale: a *synthetic* orbit (the v2 splat's own renders, perfectly consistent) gave 29.7 dB in
     the dry run.
   - The frames follow the control closely (matte-vs-control IoU 0.95), and the orbit closes on itself
     (frame 80 vs frame 0: 33.7 dB).
2. **The cameras are exact; don't refine them.**
   - Refitting held-out cameras to the splat *lowers* the score (30.4 → 27.7).
   - Letting the trainer refine video-frame cameras (median move of only 0.24° / 0.6 cm) costs 6.8 dB,
     multiplies floaters by 8 and softens everything.
   - This is research-plan Q1/H4 answered for dense control: the generator follows the control
     pixel-for-pixel, so there is no camera drift to correct (see the plan's §0.2).
3. **Lightning 4-step beats 20-step here.** It's 4× faster, and the 20-step frames are less consistent
   with each other (25.3 vs 30.4 dB) and no sharper (Laplacian 2.20 vs 2.43).
4. **The weak spots are coverage and resolution, not consistency.**
   - The top 40° view is streaky: there are no elevated frames.
   - The native-resolution photo score is 0.9 dB below v2's: the 480×832 frames are soft.

## Timing on this PC (RTX 4090, models on the HDD)

| Step | Time |
|---|---|
| Control rendering (81 frames) | ~2 s |
| Generation, Lightning, one 81-frame orbit | 8 min, of which ~5 min is loading the two 14B experts |
| Generation, 20 steps | 35 min (contended with another GPU job; expect ~15 min alone) |
| Matting + QA (79 frames) | ~2 min |
| Training, 15k iterations | ~45 min with three trainings sharing the GPU |

## Not done yet

The 2×180° arc layout and the mesh-proxy ablation. The first attempt at the 2×180° variant died with a
CUDA fault after another session's training attached to the same ComfyUI; the mesh variant never started.

## Recommended next steps (in order)

1. **Elevation coverage.**
   - A second clip on a helix: start at the photo camera, rise to about +30° over the first quarter
     turn, then orbit.
   - Train on both clips. This targets the top-view streaks.
2. **720×1280 generation** (about 2.5× the generation time), or a SeedVR2 video pass on the 480p frames,
   for sharper sides and back and a better native-resolution photo score.
3. **Second round.** Use g1_A as the proxy and regenerate. Its geometry now comes from a consistent
   video, not from Qwen views.
4. **Research ablations:** mesh proxy vs splat proxy, 2×180° vs 360°, and seeds. These can run
   unattended once the GPU is free.

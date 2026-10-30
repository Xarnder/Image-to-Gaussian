# CAP4D: how it builds a head, and how we'd build a head pass from it

**Status:** 2026-09-25. A study of `existing-research-projects/cap4d` (CVPR 2025, Taubner et al.), plus a
proposed plan for our pipeline. **Implemented and measured the same day: see [headpass_results.md](headpass_results.md).**
Short version: the coherent stand-in (H2, without FLAME) is the fix; the separate head pass (H3/H4) is a small extra
on top of it and does nothing on its own; the leash hurts.
**Problem it addresses:** the face on our full-body splat behaves like a mask. In side views it sits
too far forward or too far back, and it doesn't blend into the forehead.
**Companion documents:** [pacgs_face_fix.md](pacgs_face_fix.md) (the face work so far, including the
head push), [pacgs_quality_next.md](pacgs_quality_next.md), [../../pipeline_breakdown.md](../../pipeline_breakdown.md).

---

## 1. TL;DR

- **CAP4D's head doesn't look like a mask because the face is never a separate object.** A dense face
  tracker fits one head mesh (FLAME: face, forehead, skull, ears and neck) to the photo. Every image the
  generator produces is conditioned on that mesh, rendered from that image's own camera. Every Gaussian
  of the final avatar is attached to a triangle of that same mesh. So the face, forehead and scalp can't
  come apart. They're one surface, and a model built specifically for faces measured its depth.
- **Our face depth comes from a body model and a heuristic.** SAM 3D Body's MHR mesh has a coarse
  face. `face_align` pushed it 1.9–5.0 cm along the photo's rays, and the bake/shell priors pinned face
  Gaussians to it. The first-pass splat carrying that face then becomes the depth stand-in for every
  Wan clip, and Wan paints the face exactly where the stand-in puts it. Any error in the face's depth
  therefore reaches every clip and gets trained in. Wan is doing its job; the geometry it's given is wrong.
- **We don't need CAP4D's diffusion model to get CAP4D's benefit.** What fixes the mask is its
  *geometry*: a FLAME head fitted to the photo, a leash that keeps Gaussians near that surface, and
  control renders made from it. All of that fits into our Wan/PAC-GS pipeline. Its image generator
  (MMDM) is an optional extra, with real costs (§7).
- **Your two-pass idea is right, with one change:** the head pass must be anchored to a fitted head
  mesh. Otherwise it inherits the same wrong face depth from the control video. The recommended plan:
  1. Fit FLAME to the photo and place it in our world frame. From then on it decides where the face is.
  2. Correct the depth stand-in's face with FLAME before generating the head clips.
  3. Run a separate **head pass**: a splat of head and hair only, Gaussians leashed to FLAME, trained on
     the head clips plus the native-resolution photo crop.
  4. Train the full-body pass as now.
  5. **Merge at the neck**: a cut plane plus a blend band on the neck, then a short fine-tune of only
     the Gaussians in the band. No seam goes around the face.

---

## 2. What CAP4D is

CAP4D turns one or more photos of a person into an **animatable** Gaussian head avatar. It has four
stages:

```
 reference photo(s)
    │
    ▼  1. TRACK ────────── Pixel3DMM (UV + normal prediction) → FLAME fit; RVM matte; L2CS gaze
 FLAME shape / expression / pose + a camera, per reference image          scripts/track_video_pixel3dmm.sh
    │                                                                       scripts/pixel3dmm/convert_to_flowface.py
    ▼  2. GENERATE ─────── MMDM: a multi-view latent diffusion model conditioned on FLAME renders
 840 head images at 512², each with an exactly known FLAME pose, camera and expression
                                                                            cap4d/inference/generate_images.py
    ▼  3. FIT ──────────── GaussianAvatars: Gaussians bound to FLAME triangles + a deformation U-Net
 an animatable 3D Gaussian head                                             gaussianavatars/train.py
    │
    ▼  4. ANIMATE ──────── drive FLAME with new expressions/poses, re-pose the bound Gaussians
 renders + an exported animated .ply                                        gaussianavatars/animate.py
```

The default quality setting runs 840 generated images and 100k training steps, which the README says
takes "hours". Low quality is 210 images and 25k steps.

### 2.1 Tracking: a head mesh fitted to the photo

`track_video_pixel3dmm.sh` runs **Pixel3DMM**, a network that predicts a per-pixel UV coordinate map and
a normal map of the face, then fits **FLAME 2023** to those predictions (`include_neck=False`,
`use_flame2023=True`). `convert_to_flowface.py` refits FLAME with 150 shape and 65 expression
parameters into CAP4D's format (`fit.npz`: `shape`, `expr`, `rot`, `tra`, `eye_rot`, `neck_rot`, plus
`fx, fy, cx, cy, extr`). It also:

- mattes the person with **RobustVideoMatting** (`bg/`), since the generator expects a white background
- estimates gaze with **L2CS** and turns it into eyeball rotation (clamped to 25°)

FLAME is a metric head model with about 5k vertices covering the face, skull, ears and neck. CAP4D adds
two things to it (`cap4d/flame/mouth.py`): a hemisphere for the mouth interior, and an optional
separately regressed lower jaw.

**This is the step we lack.** Our only 3D model of the head is the head of a full-body model.

### 2.2 MMDM: the morphable multi-view diffusion model

`cap4d/mmdm/`, `configs/mmdm/cap4d_mmdm_final.yaml`.

- **Base:** Stable Diffusion 2.1 (`v2-1_512-ema-pruned.ckpt`), 512×512, 64×64 latents. **No text.**
  Cross-attention is switched off (`context=None`).
- **Multi-view:** it denoises 8 frames together. At the lower-resolution levels of the UNet
  (`channel_mult ≥ 2`), self-attention runs over the tokens of all 8 frames at once (`mode="3d"` in
  `net/attention.py`). That's how the views agree with each other.
- **Reference frames are clean inputs, not targets.** In a batch of 8, R frames (up to 4) are real
  photos. Their clean latents replace the noisy ones, and the loss ignores them (`mmdm_unet.py`: `x =
  z_input * ref_mask + x * ref_mask_inv`). The rest are generated.
- **Conditioning** (`conditioning/cap4dcond.py`, 50 channels per latent pixel, added after the first
  UNet block through a zero-initialised linear layer):

  | Channels | What | Why |
  |---|---|---|
  | 42 | sin/cos positional encoding of the **canonical FLAME vertex position** rasterised at each pixel (`mesh2img.PropRenderer`, head vertices + mouth only) | tells the model *which point of the head* each pixel shows: dense correspondence, not only depth |
  | 3 | the expression's vertex offset from neutral, rasterised | expression control |
  | 3 | per-pixel **ray direction**, rotated into the first reference camera's frame | the camera, relative to the reference |
  | 1 | reference mask | real photo or to-be-generated |
  | 1 | crop mask | where the crop ran off the original image |

  Compare with our Wan control, which is a depth video. CAP4D's control is a **correspondence** map: it
  tells the model "this pixel is the tip of the nose" as well as how far away it is.
- **Schedule:** SNR shift for multi-image generation, with zero terminal SNR (`mmdm/utils.py`).
- **Training data:** NeRSemble, Ava-256, VFHQ and MEAD. These are real multi-view or video face
  captures, all FLAME-tracked with FlowFace. The model has seen real faces from calibrated rigs, which
  is why its side views are geometrically faithful.

### 2.3 Which views it generates

`cap4d/inference/data/generation_data.py`, `configs/generation/*.yaml`:

- Cameras **pivot about the head**, not the body. Yaw and pitch are sampled uniformly inside an ellipse
  of **±55° yaw, ±20° pitch** around the direction the face points (`pivot_camera_intrinsic`,
  `compute_yaw_pitch_to_face_direction`). It never generates the back of the head.
- Each frame is a **square crop around the projected FLAME head vertices + 20 % margin**, resampled to
  512 (`get_bbox_from_verts`, `CROP_MARGIN = 0.2`). The head always fills the frame.
- Expressions and eye rotations are drawn from a bank of 840 real ones (`gen_data.npz`) so that the
  avatar learns to animate. `expr_factor` scales them; 0 gives FLAME's neutral face. That's not the
  photo's own expression (see §7.2).
- Every generated image's FLAME parameters and camera are saved beside it (`generated_images/flame/*.npz`).
  **The camera of every generated image is known exactly, by construction.** This is the same
  principle as our PAC-GS orbit pass.

### 2.4 Stochastic I/O conditioning: 840 consistent images from an 8-frame model

`cap4d/mmdm/sampler.py`. The model sees only 8 frames at a time, but it has to produce 840 images that
all agree. At **every denoising step**:

1. All 840 noisy latents are shuffled into random groups of 7 (V − R).
2. Each group is paired with a random subset of the references.
3. One noise prediction per group (with classifier-free guidance, scale 2.0).
4. All 840 latents step forward together.

Because the grouping changes every step, each image is denoised alongside a different set of views at
different steps. Consistency spreads across the whole set without any image being generated
"after" another. It's an elegant fix for the problem we hit in §6.5.1 of pacgs_face_fix.md, where each
clip invents the back independently.

### 2.5 The Gaussian head: GaussianAvatars with a leash

`gaussianavatars/scene/cap4d_gaussian_model.py`, `gaussian_model.py`, `train.py`, `configs/avatar/*.yaml`.

- **Every Gaussian is bound to a triangle** (`binding`). Its position, scale and rotation are stored in
  that triangle's local frame (`compute_face_xyz_transformed`: `R_tri · xyz · scale_tri + centre_tri`).
  When the mesh moves, the Gaussians move with it.
- **The mesh is remeshed in UV space** at 128×128 (`load_uv`, `uv_remesh_flame_vertices`). Initial
  Gaussians are spread by triangle area (100k by default, at least 2 per triangle), so density is even
  over the whole head.
- **The leash:** `losses['xyz'] = relu(|local offset| − 2) · 1e-3` and a similar threshold on scale. A
  Gaussian can drift up to about two triangle-sizes from its surface freely and is penalised beyond
  that. It is **soft**. Hair can grow off the scalp, but the face can't walk off the skull. Our
  `FaceShell` was the hard version (projected back to within 2 mm every step), and we measured that it
  hurt.
- **Deformation U-Net** in UV space. It predicts per-vertex offsets from the expression offsets, only
  on `deformable_verts.txt` and with a Laplacian smoothness loss. It's for animation, so a static
  splat doesn't need it.
- **Per-frame neck rotation offset** (`neck_rot_offset`, L2-regularised). It absorbs small head-pose
  errors in the generated images.
- **Loss:** L1 + D-SSIM, with **LPIPS ramped from 0 to 0.9 of the loss between steps 10k and 60k**,
  while L1/SSIM are scaled down by `(1 − λ_lpips)`. By the end, training is mostly perceptual. That's a
  much stronger perceptual weight than ours (0.2), and it's aimed at the softness we've measured
  ("sharp ≈ 0.67" in pacgs_face_fix.md §8).
- White background, and no person matte in the loss. Only the crop mask is used.

### 2.6 What CAP4D is *not* good at (for us)

| Limitation | Consequence for us |
|---|---|
| Views only within ±55° yaw, ±20° pitch of the face | No back of the head or crown. Our Wan head rigs already cover those. |
| FLAME has no hair, and the avatar has no torso | Hair is invented around the mesh. Shoulders appear in the crops but aren't modelled. |
| 512² SD 2.1 images | Softer than a native-res photo crop. Good geometry, moderate texture. |
| "Hours" of generation; README asks for > 64 GB RAM | Our PC has 32 GB. Medium/low presets would be needed. |
| Linux-first: bash scripts, pytorch3d, Pixel3DMM install is "prone to errors" | Real porting work on Windows (§8). |
| **Licences**: CAP4D is CC BY-NC 4.0; the GaussianAvatars code carries a Toyota non-commercial notice; FLAME is non-commercial research | Fine for research. Check before any commercial use. |
| MMDM weights are not in the clone yet (`data/weights/mmdm/` has only `config_dump.yaml`); FLAME needs an account | Both need downloading before anything runs. |

---

## 3. Why our face behaves like a mask

The chain that decides where our face ends up, from [pacgs_face_fix.md](pacgs_face_fix.md) and the code:

```
SAM 3D Body (MHR) mesh ── a body model; its face is coarse
   │  face_align: head pushed back 1.9–5.0 cm along the photo's rays to fit the Qwen silhouettes
   ▼
pass-1 training ── face_depth_weight / FaceShell pin face Gaussians to the MHR face;
   │               face_bake pastes one disc per photo pixel onto the MHR face surface
   ▼
character.ply (pass 1) ── used as the depth stand-in for every Wan clip
   │
   ▼
Wan Fun-Control paints the face wherever the stand-in's depth says it is (pixel-for-pixel)
   │
   ▼
orbit-pass splat ── trained on those clips; inherits the stand-in's face depth
```

Three things make it read as a mask:

1. **The face's depth was never measured by anything that models faces.** MHR is a body model, and the
   push is a silhouette heuristic. Measured on the test image, the nose tip sits at 12.5 cm in front of
   the body's centre line on the body model and at 9.7 cm in the product's final splat (§6.6). That 2.8 cm
   is the size of the "mask set back" defect the user reported.
2. **The face layer and the rest of the head come from different geometry.** The baked layer follows
   the MHR face, while forehead, hair and scalp follow the trained splat (§4.3). Where those two
   surfaces meet, at the brow and the temples, there's a step or a gap. That's where it fails to blend.
3. **The error gets locked in.** Once the stand-in has the face at the wrong depth, every head clip
   agrees on that wrong depth. The orbit pass then fits it confidently: self-consistent, and wrong.

`face_align` is already off when the orbit pass runs, which helped (§6.6). But the pass-1 stand-in is
still built with the face priors and the bake against the MHR face, so the depth control for the face
still comes from a body model.

CAP4D avoids all three: (1) Pixel3DMM + FLAME measures the face, (2) face, forehead and scalp are one
mesh, and (3) the generator is conditioned on that mesh, so it's correct from the start.

---

## 4. Options

| | What | Fixes the mask? | Cost | Verdict |
|---|---|---|---|---|
| **A. CAP4D as a black box** | Run Pixel3DMM + MMDM + GaussianAvatars on a head crop, export a static head PLY, transform it into our world, paste it onto the body | Yes at the front, but no back of head and no real hair volume. The paste becomes the hard part. | Large: full Linux-first stack on Windows, hours per run, NC licence | Useful as a **reference result**, not as the product path |
| **B. Take CAP4D's geometry, keep our generator** | FLAME fitted to the photo decides where the face is. It corrects the Wan stand-in and anchors a separate head pass. Merge at the neck. | Yes: this addresses the root cause in §3 | Medium: FLAME fitting + alignment + a head trainer + a merge step. All native Windows, no MMDM. | **Recommended** |
| **C. B + MMDM frames** | B, plus MMDM-generated head images (their cameras are exact in our world once FLAME is there) as extra head-pass training frames | Yes, with extra front-hemisphere evidence | B + the MMDM install and hours of generation | **Experiment after B works** |
| D. Single pass with FLAME-bound head Gaussians | No separate head pass: bind the head region's Gaussians to FLAME inside the full-body training | Yes | Medium | Simpler seam-free variant of B; worth an ablation arm (§6.6) |

**Recommendation: B, then C as an experiment, with D as the comparison arm.** Your two-pass instinct is
right for two reasons. The head gets its own resolution, Gaussian budget and loss schedule. And a neck
seam is far easier to hide than a hairline-and-temple seam. But the two passes only help once
the head has correct geometry. A separate head pass trained on today's clips would reproduce today's
mask.

---

## 5. The recommended pipeline

```
 photo.png
   │
   ├─ S1–S6 as now (ingest, SAM 3D Body, Qwen views, register, train, refine) ── body world frame
   │
   ├─ H1 FIT HEAD ────── Pixel3DMM (or fallback) on the native-res head crop → FLAME
   │                     re-posed into our world with our photo camera (PnP)         NEW i2g/head.py
   │
   ├─ H2 FIX STAND-IN ── stand-in = pass-2 splat, face region re-shaped onto FLAME
   │                     (hair and bangs kept from the splat)                          NEW
   │
   ├─ S7 ORBIT ───────── Wan clips as now; the head rigs' depth now comes from the fixed stand-in
   │
   ├─ H3 HEAD PASS ───── head-only splat: FLAME-leashed skin Gaussians + free hair Gaussians,
   │                     trained on head clips + native photo crop, head-region masks,
   │                     LPIPS ramp                                                     NEW
   │
   ├─ BODY PASS ──────── the existing orbit-pass training (all clips), unchanged
   │
   └─ H4 MERGE ───────── cut at the neck, dithered blend band, colour match,
                         short band-only fine-tune → character.ply                      NEW
```

### H0. First, make the defect a number (half a day)

Before changing anything, add two measurements to `pacgs facecheck` / `i2g/evaluate.py` so that "looks
like a mask" becomes something we can score:

- **Face-depth residual vs FLAME** (once H1 exists): median and 90th-percentile signed distance from
  the splat's rendered face surface to the fitted FLAME face, in cm. Positive means the face is in front
  of FLAME, negative behind. This measures "too far forward/back" directly.
- **Brow step**: render a pure side view (±90°) and a ±45° view. Take the expected-depth profile along
  the sagittal line from the nose bridge up to the hairline and report the largest jump between
  neighbouring samples (mm), relative to the same profile on FLAME. A mask edge shows up as a spike.

Existing numbers to keep watching: nose-tip depth (§6.6), facecheck psnr/speckle/arcface per elevation,
photo dB, floaters.

### H1. Fit FLAME to the photo and put it in our world (2–4 days)

1. **Crop** the head from the native-resolution photo (`00_ingest/source_hr_rgba.png`) around the MHR
   head keypoints, with a margin, to a square of roughly 512–1024 px.
2. **Fit FLAME** with Pixel3DMM in single-image mode. That's what CAP4D was validated with, and the CAP4D
   repo has the conversion code (`scripts/pixel3dmm/convert_to_flowface.py`). Fallbacks if the Windows
   install fights back (§8): SMIRK, EMOCA/DECA or MICA, all of which regress FLAME from one image.
   Keep the photo's own expression. We want the face as it is in the photo, not neutral.
3. **Re-pose into our camera.** The tracker's camera isn't ours. Project FLAME's vertices with the
   tracker's camera to get 2D positions in the crop, map them back to full-photo pixels, then solve
   **PnP with our `K`** (FLAME vertices as 3D, those pixels as 2D). That gives the FLAME pose in our
   photo camera, and `inv(photo_w2c)` gives it in the world. It's the same trick S4 already uses.
   FLAME is metric, so the scale is right, and PnP only has to find depth and rotation.
4. **Sanity-check against MHR**: FLAME's eye/ear/nose landmarks vs MHR `kp3d`. Expect agreement to
   about 1–2 cm. Where they disagree about the face, **FLAME wins**. Log the difference, because it's
   the size of the mask error we've been living with.
5. **Save** `head_flame.npz` (vertices in world, faces, UVs, landmark indices, the fit) next to
   `body_world.npz`.

Where it goes: a new `i2g/head.py`, and a `head` stage after `body` in `i2g/pipeline.py`. It only needs
the ingest output, so it can run alongside the Qwen views on the CPU/GPU schedule without touching
ComfyUI.

**Acceptance:** FLAME projected with our photo camera overlays the photo's face within about 2 px at
the eyes and mouth corners. Save an overlay image like `body/overlay.png`.

### H2. Put the face where FLAME says, *before* Wan sees it (2–3 days)

The Wan head clips follow the stand-in's depth, so the stand-in's face has to be right. Lesson from
§6.5 of the face-fix doc: **don't replace the stand-in with a bare mesh**, because Wan then removes the
hair. Reshape only the face:

1. Build the stand-in from the pass-2 splat **without the bake** (`character_unbaked.ply` already
   exists for this).
2. Label the photo with a **face-parsing model** (skin / brows / eyes / nose / lips vs hair / hat /
   background) and lift the labels onto FLAME. That gives "skin the photo sees", "hair over the
   forehead (bangs)" and "ears".
3. For stand-in Gaussians near FLAME's skin region: move each one along the FLAME normal onto the FLAME
   surface, keeping its small offset. Delete face Gaussians more than ~1.5 cm off the surface on the
   inside (the old mask's backing layer). Leave hair Gaussians alone, including bangs, which should stay
   *in front of* FLAME. Feather the correction to zero across the hairline and temples, so hair volume
   is untouched.
4. Recompute the control depth for the head rigs (`i2g/orbit_stage.py: _controls`). Nothing else
   changes: same rigs, same Wan settings, same start frames.

Cheaper first test, with no Gaussian edits: composite the **depth images** instead. In the face-skin
region use FLAME's depth, elsewhere use the splat's, and feather across the hairline. That's
`pacgs/proxy.py`-level code. If the clips it produces fix the side-view profile, the Gaussian-level
version may not be needed.

**Acceptance:** the `restyled` check still passes (IoU vs subject ≥ 0.9). The head clips' side frames
show the nose and brow at FLAME's depth. The face-depth residual (H0) on a splat retrained from these
clips alone is below 5 mm.

**H2 alone may fix most of the reported defect.** It's the smallest change that addresses the root cause,
so it's the first thing to build and measure, before the separate head pass.

### H3. The head pass (4–6 days)

A splat of **the head and hair only**, trained in world coordinates:

- **Representation, taken from GaussianAvatars:**
  - *Skin Gaussians* bound to FLAME triangles (UV remesh at 128², area-proportional init as in
    `load_uv`), with the **soft leash** (`relu(|offset| − τ)`, τ ≈ 2 triangle-scales, λ ≈ 1e-3) and a
    scale threshold. Static: no deformation U-Net, no expression changes.
  - *Hair Gaussians*, unbound: initialised from the head clips' visual hull minus FLAME's interior,
    with a density floor on the scalp. Ordinary densification.
  - The leash is the key difference from the old `FaceShell`. It keeps the face on the skull without
    flattening it onto a 2 mm shell, which we measured as harmful.
- **Training data:** every head rig's frames (`face`, `face_up`, `face_down`, `face_high`, `face_top`),
  plus the **native-resolution photo crop** of the head at high weight. Optionally the body orbit
  frames at low weight, masked to the head (they give the head's silhouette against the body).
- **Loss masks:** each frame's loss covers only the **head region**: FLAME dilated by ~2 cm ∪ the
  hair matte, above the neck cut plane (H4). Torso pixels in the head clips are ignored, which avoids
  the invented-back problem of §6.5.1 in this pass.
- **Loss schedule, from CAP4D:** L1 + SSIM, with LPIPS ramping to a large share of the loss in the
  second half of training (start at 0.5 max and ablate up to CAP4D's 0.9). Our face is measurably soft,
  and this is the most direct lever.
- **Budget:** its own Gaussian count and iterations (start at 20k). Head frames no longer compete with
  the body clips for steps.

Where it goes: `i2g/head_splat.py` (a trainer variant: it reuses `i2g/splat.py`'s render, losses,
strategy and export, adding the binding and the leash), and a `head` stage after `orbit`.

**Acceptance:** facecheck at all five elevations ≥ the current shipped model (`f11_facedome40`), speckle
≤ 0.92. Face-depth residual < 5 mm. Brow step within 2 mm of FLAME's own profile.

### H4. Merge at the neck, not around the face (3–4 days)

```
        head pass owns ──►   ░░░░  (face, skull, hair above the band)
                            ░░░░░░
  h1 (just under the jaw) ─ ▒▒▒▒▒▒ ─┐
                            ▒▒▒▒▒▒  │ blend band on the neck, ~3–5 cm
  h0 (base of FLAME neck) ─ ▒▒▒▒▒▒ ─┘
        body pass owns ──►  ▓▓▓▓▓▓▓▓  (shoulders, torso, everything below)
```

1. **The neck axis and cut:** from FLAME's neck joint up to its head joint. `h` is height along that
   axis. The band runs from `h0` (FLAME's neck ring, above the collar) to `h1` (below the jaw line).
   Skin there is smooth and evenly lit, with no features to misalign, so it's the easiest place on a
   person to hide a seam.
2. **Ownership:**
   - Above `h1`, the head splat owns everything inside the head region (FLAME dilated ∪ head-pass hair
     hull). Body Gaussians there are deleted.
   - Below `h0`, the body splat owns everything, and head Gaussians there are deleted.
   - **Long hair crossing the band:** hair Gaussians are owned by whichever pass has them in its hair
     hull. The head pass keeps its hair down to the shoulders, and body Gaussians inside that hull are
     removed. Hair behind the shoulders stays with the body pass.
3. **Blend by dithering, not by fading opacity.** In the band, keep each head Gaussian with probability
   `w = smoothstep(h0, h1, h)` and each body Gaussian with probability `1 − w`. Multiplying opacities
   instead (two layers at 50 %) gives 75 % coverage, which makes a translucent ring. Random thinning
   gives a true cross-dissolve, because both sets are dense.
4. **Colour match:** render the band from ~16 cameras with each set alone and solve a per-channel
   affine for the head set, weighted to the band. The head clips and body clips were generated
   separately and can differ slightly in exposure.
5. **Seam fine-tune:** 2–3k steps on body-orbit + head-clip frames, **with only the band's Gaussians
   trainable** and everything else frozen. The optimiser closes whatever the dither left.
6. Export `character.ply` as usual. Keep `character_body.ply` and `character_head.ply` beside it for
   debugging.

Where it goes: `i2g/merge.py`, and a `merge` stage that writes the deliverable.

**Acceptance:** no visible ring in the turntable at eye level and at ±30°. A new *seam metric*: in
renders at 12 azimuths, the high-pass energy along the band vs the same band on the body-only splat
should not rise more than 10 %. Photo dB and floaters no worse than the body-only splat.

### H5 (experiment). MMDM frames as extra head evidence

Once FLAME sits in our world frame, every MMDM-generated image has an **exact camera in our world**,
because MMDM is conditioned on FLAME rendered from that camera. So MMDM frames can join the head pass
directly, like the Wan frames.

- What they'd add: dense, face-specialised, mutually consistent views across the ±55° yaw / ±20° pitch
  front hemisphere, where the mask artefact lives. They come from a model trained on real calibrated
  multi-view faces, and at 512² with the head filling the frame (more face pixels than our `face` rig).
- What it takes: see §7.
- How to judge it: an ablation arm with and without MMDM frames, scored by H0 + facecheck. Also use the
  MMDM frames as a **held-out reference**: a model trained without them should still agree with them.
  That gives an independent check that doesn't use our own generator's output as ground truth, which
  §8 of the face-fix doc lists as a limitation.

### H6 (ablation arm). One pass, FLAME-bound head

Option D: bind the head-region Gaussians to FLAME with the same leash, inside the ordinary full-body
orbit-pass training. There's no seam at all, but the head can't have its own budget, resolution or
loss schedule. Worth one arm, because if it scores the same as H3 + H4 it's much simpler to maintain.

---

## 6. Order of work and what each step should prove

| Step | Builds | Proves | Rough effort |
|---|---|---|---|
| 1 | H0 metrics | "mask" is a number (depth residual, brow step) | ½ day |
| 2 | H1 FLAME fit + world alignment | how far MHR's face is from a face model's (expect cm) | 2–4 days |
| 3 | H2 depth-composite stand-in (cheap version) | whether correcting the control alone fixes the profile | 1–2 days |
| 4 | H3 head pass | head quality with its own budget + leash + LPIPS ramp | 4–6 days |
| 5 | H4 merge | the neck seam can't be seen | 3–4 days |
| 6 | H6 one-pass arm | whether the two-pass complexity pays for itself | 1–2 days |
| 7 | H5 MMDM frames | whether a face-specialised generator adds front-hemisphere quality | 3–5 days (mostly install) |

Stop after step 3 and look. If correcting the stand-in fixes the side profile, the head pass is still
worth doing for sharpness and budget, but the urgency drops.

All measured on `test-image` first, then at least two more subjects before it becomes a product
default. §6.6 of the face-fix doc explains why the product has to be run end to end, not only the
research harness.

---

## 7. Using MMDM itself: what it would take

### 7.1 Install on this PC

- **pytorch3d** is used for mesh rasterisation in `mesh2img.py`. On Windows it needs a source build
  against our CUDA/torch, or we replace `PropRenderer` with gsplat (we already render meshes as
  Gaussians in `i2g/priors.py: MeshPrior`) or nvdiffrast. The replacement is about 100 lines.
- **Weights:** `scripts/download_mmdm_weights.sh` for MMDM, and the FLAME 2023 pkl after registering
  at flame.is.tue.mpg.de (then `scripts/fixes/fix_flame_pickle.py`).
- **Pixel3DMM:** installed by `scripts/install_pixel3dmm.sh`, which is bash, and whose README warns about
  version conflicts. Run it in its own venv. Don't mix it with `.venv-i2g` or ComfyUI's Python.
- A separate `.venv-cap4d` (Python 3.10, as upstream). It talks to our pipeline through files only
  (`fit.npz`, image folders), the way ComfyUI does.

### 7.2 Changes to CAP4D for a static splat

- **Keep the photo's expression:** in `GenerationDataset.init_flame_params`, use the reference's
  `expr` and `eye_rot` for every sample instead of the expression bank. `expr_factor: 0` would give
  FLAME's neutral face, not the photo's.
- **Pivot and range:** keep the head pivot. Yaw ±55 / pitch ±20 is the front hemisphere, and the Wan
  head rigs cover the rest.
- **Export cameras in our world:** apply the H1 transform to each `generated_images/flame/*.npz` camera
  and write `records.json` in the format `i2g/splat.py` loads (`kind: "video"`-style exact cameras,
  per-record intrinsics from the crop).
- **Background:** MMDM paints on white. Matte its frames with BiRefNet like any other frame.

### 7.3 Runtime and memory (estimates, not measured)

The sampler does one UNet pass per group of 7 generated + 1 reference frames, with CFG doubling the
batch. With 420 images at 50 steps (the medium preset), that's 60 groups × 50 steps = 3,000 passes of a
batch-16 SD 2.1 UNet at 64² latents. On a 4090 that's likely **30–60 minutes**. Low quality (210
images) is about half. The ">64 GB RAM" in the README looks tied to one model copy per GPU plus cached
conditioning, and with one GPU, 32 GB should work, but that needs checking. As always: **never run it
while ComfyUI or a training run holds the GPU.**

---

## 8. Things to borrow even if we never run CAP4D

These are cheap and independent of FLAME:

1. **LPIPS ramp** (§2.5): perceptual loss growing to dominate late in training. Directly targets our
   measured softness (sharp 0.67). One config field and a schedule in `i2g/splat.py`.
2. **Soft leash instead of hard shells:** `relu(distance − τ)` penalties rather than projection. Any
   prior we add in future should be soft by default. §3 of the face-fix doc shows what hard ones did.
3. **Head-pivoted cameras with the head filling the frame:** our face rigs already pivot and track.
   CAP4D's square crop with 20 % margin puts more pixels on the face than a 3× zoom into a 480×832
   frame. Worth one run: a face rig at 832×832 square, framed by the head bounding box.
4. **Stochastic I/O as a cross-clip idea:** our clips each invent the back of the head independently
   (§6.5.1). The CAP4D principle is to regroup at every step so no view is generated in isolation.
   Wan can't do this within a clip, but a cheap version would be feeding the body orbit's 180° frame to
   the head rigs as a second reference image, so all clips share one invented back.
5. **A correspondence control:** CAP4D conditions on *which point of the head* each pixel is, not only
   its depth. We can't retrain Wan, but Fun-Control variants that accept extra control channels, or a
   depth + normal pair, move in that direction. A research note, not a plan.

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| Pixel3DMM fit is off for some subjects (hair over the face, strong expressions, glasses) | Overlay check in H1 and fall back to MHR's face with a warning. Try SMIRK/EMOCA as second opinions. |
| FLAME and MHR disagree about head *rotation*, so the neck twists at the seam | Fit the band fine-tune with a small rotation freedom for the head set. Measure the angle between FLAME's and MHR's neck axes and warn above ~10°. |
| Bangs and hair over the face get pushed onto the skin in H2 | Face parsing decides what's skin. Only skin-labelled regions are reshaped. |
| Head clips and body clips differ in colour | Colour match in H4 step 4, plus the band fine-tune. |
| Hats, hoods and big hair break "head = FLAME + hull" | The hair hull is data-driven. For hats, treat them as hair: unbound Gaussians. |
| Licence (NC) for FLAME/CAP4D/GaussianAvatars code | Research use now. For commercial use, reimplement the binding and leash (simple, described above) and license FLAME, or swap in an open head model. |
| Extra passes lengthen runtime | The head pass is small (head frames only). The merge is minutes. MMDM is optional. |

---

## 10. Where CAP4D's ideas live, for reference

| Idea | CAP4D file |
|---|---|
| End-to-end script | `scripts/generate_avatar.sh` |
| FLAME fit from Pixel3DMM, matting, gaze | `scripts/track_video_pixel3dmm.sh`, `scripts/pixel3dmm/convert_to_flowface.py` |
| FLAME + mouth + lower jaw | `cap4d/flame/flame.py`, `cap4d/flame/mouth.py` |
| Head-pivoted camera sampling | `cap4d/inference/data/generation_data.py`, `cap4d/datasets/utils.py: pivot_camera_intrinsic` |
| Head crop from FLAME vertices | `cap4d/datasets/utils.py: get_bbox_from_verts` (`CROP_MARGIN = 0.2`) |
| Conditioning maps (canonical position, expression, rays) | `cap4d/mmdm/conditioning/cap4dcond.py`, `mesh2img.py` |
| Multi-view UNet (3D attention, reference latents) | `cap4d/mmdm/net/mmdm_unet.py`, `net/attention.py` |
| Stochastic I/O sampler | `cap4d/mmdm/sampler.py` |
| Gaussians bound to FLAME, UV remesh, leash, neck offsets | `gaussianavatars/scene/cap4d_gaussian_model.py`, `gaussian_model.py` |
| Losses and LPIPS ramp | `gaussianavatars/train.py`, `configs/avatar/default.yaml` |

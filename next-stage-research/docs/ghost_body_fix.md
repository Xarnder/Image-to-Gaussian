# Ghost second body in side views: diagnosis and fix (2026-09-28)

**Symptom** (V15 runs: `V15-Different-Poses/White Monochrome Studio Portrait`, `Minimalist Studio Fashion
Portrait`, `V15-Combined-Head/19B0E488-…`): the front is fine, but from 45-135 degrees the torso and legs
show a second, offset copy of the body, torn "sheets" of cloth hanging off the hips, and a grey
back-of-top smeared across the back.

## Where it comes from

The ghost is already in the first splat (`06_train`), survives the second pass (`07_refine`), and the
orbit pass then bakes it in: Wan repaints the depth video of the pass-2 splat faithfully (control IoU
0.95), so the retrained splat keeps it. Nothing downstream of registration can remove it.

**Root cause: no side views.** In all five V15 subjects with this look, every one of the 8-12 side
candidates (`r`, `l`, `r2`, `l2`, `r_hi`, `l_hi`) was rejected at registration (`03_qa/contact_sheet.png`,
red row). Older subjects (F-14, Nurse Man, ...) keep 6-8 of 8. With no view between 50 and 130 degrees
on either side, the only depth constraints are the 45/135 degree diagonals, whose silhouettes bound the
body by slabs at 45 degrees. Their intersection is a **diamond** about twice as deep as the body.
Top-down slices of the splat against the SAM 3D Body mesh show exactly that (orange = mesh, white =
Gaussians):

- torso slices at 0.45-0.75 of the height are diamonds with tips 15-20 cm in front of and behind the
  body; each diagonal view paints its colours on the diamond face it sees;
- from the side, the front and back tips of the diamond read as a second body.

**Why the side views were rejected.** Registration fits each view's camera by 2D keypoints against the
body scaffold (SAM 3D Body fitted to the front photo). The scaffold's depth is SAM 3D Body's guess
from a single front photo. From the side, the 2D keypoints measure precisely that depth, and for slim or
curvy subjects the scaffold's side-on posture (head forward, hips back) differs from Qwen's side views.
Reprojection comes out at 31-42 px (limit 31 px) and coverage at 0.68-0.79 (limit 0.80), so every side
view is rejected. The second pass cannot rescue them: it refits cameras against the first splat, which
is the diamond, so side views score IoU 0.52-0.65 (limit 0.85).

**Second bug: back views registered as front views** (19B0, walking pose). The keypoints read the back
view as a mirrored front view (az 358 instead of 180). The second pass then fitted `b_hi_s101` at
az 3 and *selected* it: a back-of-body image trained as a front view (grey back texture on the front
half, ghost limbs).

## Fix (`i2g/register.py`, `i2g/reregister.py`)

1. **Silhouette rescue** (`register.rescue_by_silhouette`, run at the end of `register_all`). A view
   rejected only for reprojection or coverage gets a second fit: the scaffold mesh is rendered as opaque
   Gaussians (`mesh_gaussians`), and the camera is fitted to the view's matte by render and compare
   (`reregister.fit_camera`, silhouette only), from both the keypoint and the requested camera. It is
   kept only if the keypoint camera and the silhouette camera both land within 25 degrees of the
   requested angle and the scaffold silhouette's coverage is at least 0.72. The second pass then fits
   it to the (now sane) first splat, as it always has for good subjects.
2. **Front/back flip guard** (`register.flipped`). A camera is flipped when it lands on the other
   front/back half from the requested angle, with both more than about 17 degrees off the side.
   - Registration prefers non-flipped hypotheses and rejects a view if all of them are flipped.
   - The second pass never takes a flipped fit.
   - Left/right mirrors (asked for front-left, got front-right) are allowed, as before.
3. **Core views are required** (`pipeline.run_register`, `register.core_gaps`). Before training, each core
   angle must have an accepted view within 25 degrees (below 40 degrees elevation):
   - the angles are the right and left sides, the back and the four diagonals (`CORE_VIEWS`);
   - an angle without one gets `--seeds` new Qwen candidates for that tag (the next seeds in the
     101, 202, ... sequence), which are registered, and this repeats for up to `--view_retries` rounds
     (default 5);
   - if an angle is still missing after that, the run stops (`PipelineError`; a batch marks the photo
     failed and goes on), unless `--no-require_core_views` is given;
   - the extra seeds are picked up again on reruns (`views.extra_pairs`).
4. **Back-left views**: SAM 3D Body reads "back-left quarter" views 10-12 cm off the scaffold's pose
   (back-right views: ~5 cm), so 8 of 12 V15 subjects lost them to "pose drift". Yet their silhouette
   fits reach coverage 0.79-0.94 and IoU 0.58-0.69, the same as the accepted back-right views (0.90-0.95,
   0.65-0.79). The rescue therefore also takes pose-drift rejections up to 15 cm, with a stricter
   coverage of at least 0.80.

Offline re-registration with the cached SAM 3D fits (no ComfyUI):

| subject | sides kept before | after | notes |
|---|---|---|---|
| White Monochrome | 0 / 12 | 12 / 12 | |
| Minimalist Studio | 0 / 12 | 10 / 12 | `r2` (both seeds) is really a left view; still rejected |
| 19B0 (walking) | 0 / 12 | 12 / 12 | `b_s202` now at az 177 (was 359); flipped `b_hi` no longer used |

Across 12 older V15 and F-14 subjects (offline, before item 4), the sides are recovered in all of them
(8 of 8 side candidates accepted in 11; 5 of 8 in Argyle). Remaining gaps: `bl` in 8 subjects, which is
what item 4 addresses; `br` in one (7E29, the same drift pattern); `fl` in Argyle.

## Results

- **White, first splat (`output/ghostfix/.../06_train`), trained on the rescued side views:** the
  diamond is gone. Torso slices at 0.45-0.75 of the height now follow the body mesh's outline (the
  splat is even slightly shallower: Qwen's side views are slimmer than SAM 3D Body's guess).
- **End-to-end, through the GUI (`output/V16-Fix-Side-Issue`, 2026-09-28):**
  - The ghost body is gone on all three subjects.
  - Minimalist then showed a new defect: a detached face plate several centimetres in front of the head
    in every view past ~40 degrees (part 2).

# Part 2: the face plate that came back with the side views (2026-10-01)

**Symptom** (V16 Minimalist): the photo's face floats about a head's depth in front of the real head
from 45 degrees on, like a mask held out in front. The contact sheet's skeleton heads also sat too low
and too far forward on the side views.

**Where it comes from** (head close-ups of every stage; `scratchpad/headstages.py`):
- In `06_train` and `07_refine`, the side views build the real head, profile and all. A second, broken
  face sits about 7 cm in front of it, exactly where the scaffold's nose is.
- The orbit pass's coherent stand-in splices the photo's face at the scaffold's face depth onto a skull
  taken from the pass-2 splat. The face is a separate plate in front, and Wan paints it.

**Root cause: the scaffold's head is too far forward.**
- SAM 3D Body reads the head's depth from one frontal photo. For some subjects it guesses the head poked
  forward and tipped down.
- Every face prior (`face_mesh_init`, face depth and shell, bake) and the coherent stand-in put the
  photo's face on the scaffold's face.
- Before the side-view fix nothing contradicted that: every side view was rejected, and the diagonals
  only bound a diamond. With the side views back, the real head forms where they see it, and the
  photo's face is left in front of it.
- So the side views are right, and making their acceptance stricter would only bring the ghost back.

**Measurement.** The nose, eye midpoint and ear midpoint are triangulated from all accepted views' SAM
3D Body keypoints (pass-2 cameras) and compared with the scaffold:

| subject | head forward of the views | defect |
|---|---|---|
| test image | 0.2-0.9 cm | none |
| Nurse Man | 0.2-0.4 cm | none |
| Lawyer, 0BDC, IMG_1267, Violinist | 0.7-2.8 cm | none visible |
| Dark-Jeans | 1.2-2.8 cm | none visible |
| **Minimalist** | **4.4-5.4 cm** | **face plate** |
| **White** | **5.6-6.3 cm** | head too long front to back; the hair happened to close the gap |

Every subject also shows the views' head 1-4 cm *higher* than the scaffold, the good ones included.
That is a systematic offset between the generated views and the photo (or SAM 3D Body's keypoints on
them), not something to correct: moving the head up would break the photo view. It is the "skeleton
head too low" seen on contact sheets.

**Why the old `face_align` was no answer.** It measured the same thing from the side views'
*silhouettes*, which hair and raised arms confuse. It pushed the test image 2.7 cm when the true
offset is under 1 cm, so it was switched off for the orbit pass (`pacgs_face_fix.md` §6.6).

## Fix (`priors.keypoint_head_push`, `pipeline._aligned_world`)

1. Each head keypoint slides along its own **photo ray**, so the photo view does not change.
2. The depth is chosen to minimise the **horizontal** reprojection error in every accepted view more
   than 30 degrees off the front or back. Vertical error is left out because of the systematic height
   offset: the photo camera sits at chest height, so its rays climb toward the head, and including the
   vertical error dragged the test image to 2.2 cm.
3. The median over nose, eyes and ears is the measurement. The first 1 cm is treated as measurement
   bias (good subjects measure 0.2-1.0 cm), and the rest is applied with `priors.push_head` (full
   within 14 cm of the face, fading out down the neck).
4. Every stage uses the moved head:
   - train: register-stage cameras;
   - refine: re-measured with its fitted cameras;
   - orbit (stand-in, face priors, head pass): the refine stage's fitted cameras.
5. `face_align` is on again with the orbit pass, with `face_align_method = "keypoints"`
   (`"silhouette"` is the old `face_push`). A push of 3 cm or more is logged as a warning.

Pushes it gives (applied, second-pass cameras):

| subject | push |
|---|---|
| Nurse Man, Fashion Grandmother | 0 |
| test image | 0 |
| Lawyer | 0.4 cm |
| Ballet Man | 0.6 cm |
| 0BDC | 0.8 cm |
| Dark-Jeans | 1.0 cm |
| IMG_1267 | 1.4 cm |
| Violinist | 1.6 cm |
| Minimalist | 4.6 cm |
| White | 6.4 cm |

## Results

**Minimalist rerun** (`output/V17-Head-Fix`): from the train stage, on V16's views and cameras,
100 min.
- **Pushes applied:** 5.6 cm in train; 4.6 cm in refine and orbit, re-measured with fitted cameras.
- **`06_train`:** the detached face is gone. The photo's face sits on the head, and the front view is a
  clean face (V16's was a smeared plate). A thin double edge remains at the front of the profile, about
  the 1 cm margin.
- **Coherent stand-in:** one continuous head with the face on the skull (V16: a face plate held in front
  of the hair).
- **Final:** clean profiles at 45, 70, 90 and 270 degrees with no plate
  (`output/V17-Head-Fix/minimalist_before_after.jpg`).
- **Still open:** skin-coloured smears on the black sleeves and the upper back. They were in V16 too and
  are unrelated to the head.

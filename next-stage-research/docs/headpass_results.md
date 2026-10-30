# Head pass: experiments and results

**Status:** 2026-09-25, complete for two subjects. Plan: [cap4d_head_pass.md](cap4d_head_pass.md). Code: `headpass/`
(`python -m headpass ...`). Runs: `output/headpass/<subject>/`.
**Subjects:** `Nurse Man` (bald, the clearest mask artefact) and `Violinist Woman` (long wavy hair framing
the face), from the product batch `output/F-14-Fixed-Face/` (standard preset, seven orbit rigs, 40k steps).

## 0. What differs from the plan

- **No FLAME.** Downloading FLAME needs an account at flame.is.tue.mpg.de, which an agent may not create.
  The plan's H1 ("fit FLAME to the photo") was replaced by what the repo already has: the body mesh's *face*
  (SAM 3D Body / MHR, seen from the photo camera) for the face, the multi-view splat for the skull and hair,
  and InsightFace's `1k3d68` 3D-landmark network (already in ComfyUI's models) as an independent face-shape
  reference for measurement. If FLAME becomes available (the user registers and drops
  `flame2023_no_jaw.pkl` in), it slots in as the face-core source in `headpass/headsurf.py: face_radius_map`.

## 1. Diagnosis (H0)

Head close-ups of four product results (`headpass/views.py: head_sheet`) and the depth that Wan was given
(`depth_sheet`, shaded) show the mechanism directly:

- The orbit pass's **stand-in** (the pass-2 splat, baked) has its face as a separate plate: the photo baked
  onto the body mesh's face, plus the face priors pinning the face to that mesh. From the side the plate
  stands in front of (Nurse Man) or inside the hair around (Violinist) the skull, with a visible edge.
- Wan paints the depth it is given, so the final model inherits the plate's edge as a seam across the
  forehead and temples.
- The face's depth pattern of every product splat equals the body mesh's to within 2 mm (`sides_mm` below),
  because the priors pin it there.

Metrics (`headpass/metrics.py`):

| Metric | Meaning |
|---|---|
| `forehead_jump_mm` | largest change in depth *slope* walking up the forehead in the photo view (3 columns). A smooth head: ~1.5-2 mm (body mesh). A plate edge: centimetres. A real hairline or hat brim also scores, so compare arms of one subject only. |
| `sides_mm` | nose-tip-to-face-side depth vs the 1k3d68 estimate. Not usable as an absolute: my sampling point (12 % in from the jaw contour) sits where the cheek curves fast, so every head reads about -40 mm. Useful only as "identical to the mesh" evidence. |
| `hidden_layer` | alpha 1.5-5 cm behind the face. **Not a useful metric**: finals are solid there (0.98-1.0) and stand-ins partly hollow; it does not separate mask from no mask. Kept in code, ignore it. |

| Subject | body mesh | pass-2 stand-in (baked) | product final |
|---|---|---|---|
| Nurse Man (bald) | 1.9 | 6.3 | **25.8** |
| Violinist Woman | 1.5 | 17.1 | 3.7 |
| Lawyer Man | 1.6 | 14.4 | 2.6 |
| Sikh Chef Man (turban) | 1.6 | 3.0 | 13.2 (turban brim) |

(forehead kink, mm, photo view.) The Nurse Man's final has a 26 mm kink on a bald forehead: the seam.

## 2. The coherent stand-in (H2)

`headpass/headsurf.py: build_standin`. The head of the stand-in is replaced by one surface, held as a radius map
r(elevation, azimuth) around the head centre (1-degree bins), and written as **offsets from the body mesh's
head** - a template that already carries face, forehead, skull and ears as one surface:

1. **Old surface**: the old stand-in's outer surface per direction, from depth renders at 162 cameras around
   the head, opacities sharpened like the depth proxy's, each point voting only for the direction it is seen
   head-on from (cos > 0.75); trimmed cos-weighted mean per bin, 3x3 median; plus its colour.
2. **Face core**: the body mesh's face seen from the photo camera, inside the photo's jaw contour and brows
   raised to about the hairline (1k3d68 landmarks); the rim (2 bins, grazing to the photo) eroded.
3. **Old face**: directions where the old surface lies on the mesh's face (within 1.2 cm, within 12 cm of the
   face centre), near the face outline and skin-coloured. These give no observation (behind them is junk).
4. **Fuse**: observed offsets = face core (~0) + old surface elsewhere; the old face's directions solved as a
   membrane; the whole field smoothed once (3 bins); the face core spliced back exactly with a smoothstep over
   2 cm.
5. The old head is replaced by opaque flat Gaussians tiling the surface, each in the local tangent plane and
   capped at 2.5 cells (so a real depth step stays a step).

Stand-ins: `output/headpass/<subject>/standins/coherent2.ply` (and the shaded depth, `coherent2_depth.jpg`).
`coherent.ply` is an earlier version (v1, radius blending with a skin/hair split) used for the `coh` set.

### Dead ends on the way (each seen in the shaded depth, i.e. what Wan is given)

| Attempt | What went wrong | Fix |
|---|---|---|
| outermost radius per bin | noisy skull (floaters, fuzz win) | cos-filtered trimmed mean + median |
| head region = everything above the chin - 2 cm | a tent down to the shoulders | head/neck capsule (`in_head`) |
| Laplace membrane on raw radii across the band | crease at the band's outer edge (matches position, not slope) | smoothstep; later offsets + one smoothing |
| skull measured with the plate still in | the plate's edge survived just outside the band | remove the plate's directions, bridge them |
| plate removed with a 3D ball first | holes in the temples exposed interior junk as "skull" | classify the plate by direction on the full surface |
| discs oriented along the radius | "crumpled" patches where the surface slopes (shingles with gaps) | discs in the local tangent plane |
| discs sized to their cell with no cap | across real depth steps (cheek to hair) they stretch into streaky "fans" | cap at 2.5 cells |
| neck column 8.5 cm from 4 cm below the centre | wide cheeks clipped: holes | 9 cm from 8 cm below |
| cut 2 cm below the chin | strips hanging under the jaw (a ray crosses jaw and neck) | cut at the chin, old neck stays |
| plate = any surface within 8.5-10 cm of the face centre | hair framing the face flattened | plate must lie on the mesh's face |
| face continued at constant radius | the rising forehead read as hair: a ledge on top of the head | continue along the mesh's head |
| blending radii between the mesh-shaped face and the real skull | a ridge on the scalp (the mesh's head is 2-3 cm bigger than the Nurse Man's skull). **Wan painted it: see the `coh` set** | blend offsets from the mesh |
| hair/skin thresholds (several) | a line wherever the threshold flips (bald temples that sit higher than the smoothed template read as "hair") | no threshold: one offset field |
| 1.2 cm band | the face-edge ridge came back | 2 cm |

**Known limitation of v2**: on the Violinist the hair lying against the right cheek comes out as a smooth
veil rather than curls (the colour test for "old face" did not remove it; cause not isolated). The face and
forehead are continuous on both subjects.

**Lesson worth keeping**: Wan follows the control's geometry faithfully, including defects. The v1 stand-in's
scalp ridge (a 1-2 cm bump) appears as a real ridge on the bald head in every frame of the regenerated head
clip, while the old stand-in's smaller wrinkles were smoothed over. Judge a stand-in by its shaded depth
(`headpass/views.py: depth_sheet`) before spending generation time on it.

## 3. Experiments

All arms use the product's own settings for the orbit pass (`i2g.orbit_stage.splat_config` with the run's
`RunOptions`: seven rigs, face clips boosted 2x, 40k steps, face priors off, same Wan seed 1001 and prompt),
so an arm differs from the product only where it says so. One training at a time on the 4090.

| Arm | What changes | Where |
|---|---|---|
| **B0** | the product result (`F-14-Fixed-Face`) | `output/F-14-Fixed-Face/<subject>/v2/final` |
| **E1** | clips regenerated from the coherent stand-in (v2); product training | `arms/e1_coh2` |
| **E2** | E1's body + a separate head pass (head clips + photo, own 30k steps, leash, perceptual ramp to 0.6), merged at the neck | `arms/e2_*` |
| **E3** | head pass on the product's own clips, merged onto B0 (isolates the head pass from the stand-in) | `arms/e3_*` |

Scores (`headpass/evaluate.py`): `kink` = max forehead kink over views at 0, +-30, +-45 deg (landmarks found on
each render); identity = ArcFace to the photo over 21 head cameras (az 0, +-20, +-40, +-60; el -15, 0, +20);
`sharp` = mean |Laplacian| on the head over the same cameras; `photo dB` = head region vs the native photo.

### 3.1 Nurse Man

| Arm | kink mm | id mean | id worst | id side 60 | sharp | photo dB |
|---|---|---|---|---|---|---|
| B0 product | 11.0 | 0.375 | 0.232 | 0.353 | 3.18 | 32.22 |
| **E1 coherent stand-in** | **4.4** | 0.367 | **0.271** | 0.345 | 3.26 | 31.83 |

Per-angle kink: B0 0.9 / -30: **11.0** / +30: 1.8 / -45: **8.3** / +45: 1.6 (the seam is on one side, as seen);
E1 3.6 / 4.4 / 1.3 / 3.2 / 1.0.

Visually (`output/headpass/nurse/report/`): E1 removes the mask edge across the forehead and temple at -45 deg;
the face flows into the temples at 60-90 deg. Identity and detail are unchanged within noise. E1's scalp shows
a slight bump at the upper back (the stand-in's residual ridge, painted by Wan) - v3 of the stand-in targets it.

Clip QA (median over each clip; v2 clips vs the product's): silhouette IoU equal (0.96-0.98); face identity in
the head clips 0.476 vs 0.492 on average (better from steep above, slightly lower at eye level).

**Head pass (E2), first version: worse.** Coloured floaters grew above the head: the loss mask ignored all
background beyond 12 px of the head, so nothing told Gaussians there to be transparent. Fixed (the mask now
excludes only body pixels). With the fix (E2b, leash 10): no floaters, but the face is blotchier than E1 (dark
patches at the forehead and eye, mottled cheek at -45 deg). Being isolated: leash off (E2c), perceptual ramp off
(E2d).

### 3.2 The dip in the forehead's side profile (raised by the user mid-run)

The user pointed at the side profile: the forehead rises from the brow, then dips before the crown. Found and
measured:

- **Where it comes from: the pass-2 splat's skull**, not the blend. Along the head's midline the old stand-in's
  radius rises to 94.7 mm at 70 deg elevation, collapses to 71.8 mm at 78 deg and climbs again (Nurse Man);
  Violinist 93 -> 84 -> 88 -> 102 mm around 66 deg. The unbaked pass-2 splat has the same dent, so it is not the
  bake: it is the Qwen-view reconstruction carving a groove across the top of the forehead. Every Wan clip made
  from that stand-in paints the groove, so the product has it too.
- **The body mesh is no alternative there**: its crown is 3.6 cm too tall for the Nurse Man against the photo's
  own head-top silhouette (rows 84 vs 112 at 1.3 mm/px); the splat is within 1.3 cm.
- **Metric** (`headpass/evaluate.py: profile_dent`): the midline profile from 40 to 88 deg elevation (upper
  forehead to crown) as a 2D curve; its depth below its own convex hull, median over +-10 meridians around the
  face. (From 10 deg it picks up the brow ridge, a natural ~6 mm concavity that the body mesh has too.)
- **Fix (stand-in v4)**: a grey-scale closing of the skull's radius map along each meridian (window 21 bins,
  ~3 cm), which fills grooves across the head and keeps its size, ears and hair volume; plus v3's extra smoothing
  on skin-coloured scalp.

| forehead-to-crown dip, mm | old stand-in | v2 | **v4** | B0 product | E1 (v2 clips) | E2c (v2, head pass) |
|---|---|---|---|---|---|---|
| Nurse Man | 11.1 | 6.6 | **1.6** | 5.5 | 2.2 | 1.8 |
| Violinist | 21.3 | 9.5 | **2.7** | 5.7 | **12.9** | - |

The Violinist's E1 is *worse* than the product here: the v2 stand-in only halved the groove and Wan painted the
rest, sharper. v4 clips and arms are the next runs.

### 3.3 Stand-in v4 and the head pass: full results

Arms trained on clips regenerated from stand-in v4 (`coh4`): **E1 v4** (product training) and **E2c v4** (E1 v4 +
head pass without leash, perceptual ramp 0.6, merged at the neck). Also listed: the v2-based arms and the two
head-pass ablations on v2 (E2b with the leash, E2d without the perceptual ramp).

**Nurse Man** (`output/headpass/nurse/report/scores_all.json`)

| Arm | kink mm | profile dip mm | id mean | id worst | id side 60 | sharp | photo dB |
|---|---|---|---|---|---|---|---|
| B0 product | 11.0 | 5.5 | 0.375 | 0.232 | 0.353 | 3.18 | 32.22 |
| E1 v2 | 4.4 | 2.2 | 0.367 | 0.271 | 0.345 | 3.26 | 31.83 |
| E2b v2 (leash 10) | 5.5 | 3.0 | 0.366 | 0.238 | 0.337 | 3.19 | 30.00 |
| E2c v2 | 3.8 | 1.8 | 0.403 | 0.275 | 0.366 | 3.34 | 31.69 |
| E2d v2 (no ramp) | 3.5 | 2.2 | 0.390 | 0.284 | 0.371 | 3.13 | 31.73 |
| E1 v4 | 4.7 | 3.6 | 0.394 | 0.291 | 0.376 | 3.28 | 31.69 |
| **E2c v4** | 4.7 | 2.8 | **0.406** | 0.240 | **0.377** | **3.37** | 31.31 |

**Violinist Woman** (`output/headpass/violinist/report/scores_all.json`)

| Arm | kink mm | profile dip mm | id mean | id worst | id side 60 | sharp | photo dB |
|---|---|---|---|---|---|---|---|
| B0 product | 3.8 | 5.7 | 0.327 | 0.120 | 0.207 | 2.45 | **34.19** |
| E1 v2 | 3.0 | 12.9 | 0.347 | 0.139 | 0.243 | 2.60 | 30.83 |
| E2c v2 | 4.8 | 4.7 | 0.372 | 0.137 | 0.298 | 2.70 | 29.33 |
| E1 v4 | 3.6 | 8.3 | 0.356 | 0.162 | 0.268 | 2.59 | 30.98 |
| **E2c v4** | 4.3 | **4.3** | 0.366 | **0.167** | 0.260 | 2.70 | 29.38 |

**What the true side profiles show** (`report/side_right.jpg`, `side_left.jpg`, `final_compare.jpg`; 80-120 deg is
where the product falls apart):

- Nurse Man, product: an angular corner at the top-front of the skull, the forehead seam, a face that reads
  as pasted on at 105-120 deg, dark smudges and streaks on the scalp on the left side. **v4 arms: one round
  skull, forehead, temples and scalp continuous to 120 deg on both sides.** Visually the clearest win of the
  project on this subject; the profile-dip metric does not show it (v2 and v4 differ by ~1 mm, within its
  noise) - trust the sheets here.
- Violinist, product: the face sinks into the hair from 80 deg on (the "mask too far back" failure); at -105
  only a sliver of profile. **v4 arms: the face stays attached and the profile reads at +-90 and at 105.**
  Identity from 60 deg rises 0.207 -> 0.26-0.30.
- **Cost on the Violinist**: a whitish smear on her right brow/temple, stronger than the product's faint one,
  and photo fidelity -3 to -5 dB. The clips there are clean (checked frame by frame), so it forms in training;
  the region coincides with the smoothed "veil" in the v2/v4 stand-in where hair lies against her right temple.
  This is the main open problem for long hair (section 4).

**The head pass**: without the leash it is a consistent but modest gain - identity mean +0.01-0.04, sharpness
+0.1, cleaner skin than E1 on the Violinist - at -0.3 to -1.6 dB photo fidelity. With the leash it is worse
(blotches). The perceptual ramp trades a little worst-angle identity for sharpness and mean identity.

### 3.4 Ablation: the head pass without the stand-in fix (E3)

Head pass (no leash, ramp 0.6) trained on the **product's own clips**, merged onto the product result at the neck.

| Subject | Arm | kink | dip | id mean | id worst | id side 60 | sharp | photo dB |
|---|---|---|---|---|---|---|---|---|
| Nurse Man | B0 | 11.0 | 5.5 | 0.375 | 0.232 | 0.353 | 3.18 | 32.22 |
| Nurse Man | E3 | 9.3 | 3.5 | 0.388 | **0.107** | 0.312 | 3.29 | 32.03 |
| Violinist | B0 | 3.8 | 5.7 | 0.327 | 0.120 | 0.207 | 2.45 | 34.19 |
| Violinist | E3 | 4.8 | 4.3 | 0.360 | **0.056** | 0.269 | 2.58 | 31.34 |

**A separate head pass on its own does not fix the mask**: the seam stays (Nurse Man kink 9.3 mm) because the
clips it trains on were painted over the plate, and worst-angle identity halves. This confirms the plan's
warning (section 4 of cap4d_head_pass.md): the geometry the generator is given decides where the face ends
up; a separate pass only helps once that geometry is right.

## 4. Conclusions and recommendation

1. **Root cause** (confirmed on both subjects): the orbit pass's depth stand-in. Its face is a separate plate
   (photo baked onto the body model's face + face priors), and its skull (Qwen-view reconstruction) has a 2-3 cm
   groove at the top of the forehead. Wan paints both into every clip.
2. **The fix that matters: the coherent stand-in (v4)** - one head surface, written as offsets from the body
   model's head, face from the photo view, skull from the splat with grooves filled. With product training it
   removes the seam (Nurse Man kink 11.0 -> 4.7 mm), keeps the head coherent to 120 deg in true side views on
   both subjects, keeps the Violinist's face out of her hair (identity at 60 deg 0.207 -> 0.268), and costs
   0.5 dB (Nurse Man) to 3 dB (Violinist) of front-on photo fidelity.
3. **The head pass** (CAP4D-style separate head, merged at the neck) is a modest extra on top of v4: +0.01
   identity, +0.1 sharpness, no visible neck seam. Keep it optional. **Never with the leash.**
4. **Open**: the Violinist's whitish right-brow smear (photo -3 to -5 dB). Suspects: the smoothed "veil" in the
   stand-in where her hair lies against the temple, and the native-res photo view (25 % of steps) disagreeing
   with the clips there. First tests: keep the old hair surface where it lies against the temple (no veil), and
   lower the photo view's weight near the hairline.
5. **Product integration: done (2026-09-25).** `i2g/orbit_stage.py` builds the v4 stand-in from the second pass's
   splat (`_coherent_standin`) and renders every depth video from it; after the orbit training it runs the head
   pass (`_head_pass`: `headpass.headtrain.train_head_core`, no leash, ramp 0.6, 30k steps) and merges it onto the
   deliverable at the neck (`headpass.merge.merge_params`), then refreshes the turntable, previews and
   photo_vs_render. Both are on by default whenever the orbit pass runs, so `run_image_to_splat.bat`,
   `run_batch_to_splat.bat` and the GUI all use it; `--no-coherent_head` / `--no-head_pass` turn them off.
   The body-only and head-only splats are kept in `08_orbit/` (`character_body.ply`, `character_head.ply`), and
   `report.json` gains a `head_pass` block.

   **End-to-end check** (`image_to_splat.py --from_stage orbit` on a copy of the Nurse Man run, `output/headtest/`):
   exit 0, orbit stage 65.7 min (Wan 23, training 24, head pass 15). Scores: kink 6.2 mm (old product 11.0),
   dip 3.5 (5.5), identity 0.393 / worst 0.262 / side 0.382 (0.375 / 0.232 / 0.353), sharpness 3.39 (3.18), photo
   31.31 dB (32.22) - the research arm E2c v4 within run-to-run noise. Figure: `output/headtest/new_vs_old.jpg`.

## 5. Reproduction

```
python -m headpass setup    --src "output/F-14-Fixed-Face/Nurse Man/v2" --exp output/headpass/nurse
python -m headpass standin  --exp output/headpass/nurse --kind coherent4 --skin_smooth 16 --close 10
python -m headpass generate --exp output/headpass/nurse --set coh4 --standin coherent4     # ComfyUI running
python -m headpass prepare  --exp output/headpass/nurse --set coh4
python -m headpass train    --exp output/headpass/nurse --set coh4 --arm e1_coh4
python -m headpass head     --exp output/headpass/nurse --set coh4 --arm e2c_head_coh4 --leash 0 --standin coherent4
python -m headpass merge    --exp output/headpass/nurse --body output/headpass/nurse/arms/e1_coh4/character.ply                             --head output/headpass/nurse/arms/e2c_head_coh4/head.ply --arm e2c_coh4_merged --standin coherent4
python -m headpass score    --exp output/headpass/nurse --arms "B0=<product ply>,E1=...,E2c=..." --out <json>
```

Figures: `output/headpass/<subject>/report/` (`final_side_left/right.jpg`, `final_above.jpg`, `final_front.jpg`)
and the stand-in depth sheets in `output/headpass/<subject>/standins/`. Results page: `output/headpass/page/`.


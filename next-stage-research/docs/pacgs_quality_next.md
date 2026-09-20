# Next stage: quality plan (orbit-video pipeline)

**Updated:** 2026-09-20, after the face fix shipped. This is an implementation plan: each experiment
names the files to change, the commands to run, what it costs and what would count as success.

**Reading order:** [pacgs_results_sprint1.md](pacgs_results_sprint1.md) (the method),
[pacgs_face_fix.md](pacgs_face_fix.md) (the face diagnosis and fix),
[pacgs_explained.md](../../pacgs_explained.md) (plain English).

---

## 1. Where we are

`run_image_to_splat.bat` (standard preset) runs: the v2 stages, then the orbit pass, which renders the
splat as a depth video along **seven** camera paths (body orbit; head close-ups level, above, below,
steeply above and over the crown; a rising body orbit), has Wan 2.2 Fun-Control repaint each as the
real person, and retrains on those frames with their exact cameras for 40k steps. About 3 h in total.

Shipped model `f11_facedome40` (round 5), measured at the five head clips' own cameras and by the v2
harness:

| | eye level | +11° | +34° | +73° (crown) | −37° | all | Photo | Floaters |
|---|---|---|---|---|---|---|---|---|
| round 4 (`f7_faceball30`) | 28.87 | 28.32 | 20.62 | 17.41 | 28.40 | 24.73 | 36.97 | 0.020% |
| **round 5 (`f11_facedome40`)** | 28.04 | 26.94 | **28.09** | **26.50** | 27.30 | **27.38** | 36.73 | **0.016%** |

(psnr in dB against each clip's own frames; speckle 1.005 → 0.924, arcface 0.433 → 0.457.)

### The measured weak points now

1. **Everything behind her is invented, and each clip invents it differently.** The vest's back is wide
   flaps in one clip, a notched racerback in another, and in the +34° clip at seed 1001 a flesh-coloured
   knot with a fabric bow, which trained straight into the model. See **Q19**, the first thing to fix.

2. **Detail is 0.6× the frames.** The reconstruction is softer than the video it trains on, so the
   pipeline, not the generator, limits the face (**Q18**, **Q14**, **Q4**).

3. **Identity is below the pasted-layer version** (face grid 0.51 versus 0.64). The layer had to go —
   it was the speckle — but the real photo detail it carried has not been replaced by anything (**Q17**).

4. **Budget dilution.** Seven clips share one training budget; the three elevations that five clips had
   already learned each gave up about 1 dB. 40k steps recovered most of it (**Q7**).

Beyond the face: the body is still built from 480×832 frames, and everything so far is one subject.

---

## 2. How every experiment below is judged

- **`pacgs facecheck`** — renders from the close-up clips' own cameras and compares with the generated
  frames: agreement (psnr), detail (`sharp`, →1.0), **speckle** (fine error with no counterpart in the
  target, →0), identity. Multi-angle by construction.
- **`pacgs eval`** — the v2 harness: photo fidelity at viewer and native size, face grid over 15
  cameras, floaters, needles, sharpness, plus held-out generated frames.
- **The visual strip** from facecheck, which is what caught the speckle in the first place.

**Do not rank face changes by identity alone.** ArcFace preferred the speckled pasted layer (0.64) over
the clean face (0.51); optimising it is how the artefact shipped in the first place.

---

## 3. Experiments, ready to run

### Q16. Face close-ups from above and below — **DONE, shipped**

Two clips added (`face_up`, `face_down`), trained as `f7_faceball30` and now the product default.
Identity from 30° above went 0.30 → 0.37 on the face grid and 0.31 → 0.55 measured at the clip's own
cameras; speckle from above fell from 1.21 to 0.92. Details in
[pacgs_face_fix.md](pacgs_face_fix.md) §6.4.

**Q16b (the steeper rings) — DONE too.** `face_up` plateaus at only ~+11° relative to the head, because
the photo already looks up at it by 19°, so two steeper rings were added: `face_high` (+55 rig = +35 at
the head) and `face_top` (+95 = +73, over the crown). Trained as `f10_facedome2`: +34° went 20.6 → 28.3
dB and the crown 17.4 → 25.8 dB, at a cost of 1–2 dB on the three angles that were already good
(training-budget dilution — see **Q7**). Both rings are now in the product default. §6.5 also records
the round's main lesson: the control proxy must be the splat, never the bare body mesh.

<details><summary>Original plan (kept for reference)</summary>

### Q16. Face close-ups from above and below  ★ do first

**Hypothesis.** The face collapses above eye level because no close-up frame ever looks down at it. Two
more zoomed head clips, one rising and one dipping, should lift face-grid identity at +30° from 0.30
towards the 0.6 it reaches at eye level.

**Design.** Reuse the head close-up rig, but let the camera change elevation as it orbits, pivoting
about the head rather than the body centre so the head stays at a constant distance:
- `face_up`: rises to +30° over the first quarter turn, then holds.
- `face_down`: dips to −20° the same way.

Frame 0 is still at the photo's camera, so the start image stays an exact crop of the
native-resolution photo, which is what makes these clips start from real detail.

**Implement.**
1. `pacgs/orbit.py`: give `RIGS` entries an optional `pivot` (`"target"` or `"head"`), and pass it
   through `layout_cameras` → `orbit_w2c` → `rot_x_about`. Add:
   ```python
   "face_up":   {"arcs": "360", "zoom": 3.0, "track": "head", "elev":  30.0, "elev_ramp": 90.0, "pivot": "head"},
   "face_down": {"arcs": "360", "zoom": 3.0, "track": "head", "elev": -20.0, "elev_ramp": 90.0, "pivot": "head"},
   ```
2. **Fix the clip tag derivation first.** `pacgs/__main__.py` and `i2g/orbit_stage.py` both derive a
   view-name prefix as `rig[:2]`, which gives `fa` for `face`, `face_up` and `face_down` alike, and
   merging would raise on the name collision. Derive a short unique tag per rig instead (initials of
   the parts, e.g. `face_up` → `fu`), in one helper both call.
3. `pacgs/facecheck.py`: accept several clips (`--variant face,face_up,face_down`) and report per clip
   as well as overall, so the elevated angles are scored against their own frames.

**Run.**
```
python -m pacgs setup     --run output/pacgs/test-image/r1 --proxies splat --rigs face_up,face_down
python -m pacgs generate  --run ... --variant u1_faceup_l4  --rig face_up   --preset lightning --seed 1001
python -m pacgs generate  --run ... --variant d1_facedown_l4 --rig face_down --preset lightning --seed 1001
python -m pacgs prepare   --run ... --variant u1_faceup_l4  --tag fu
python -m pacgs prepare   --run ... --variant d1_facedown_l4 --tag fd
python -m pacgs train     --run ... --name f6_faceball \
    --variant g1_splat360_l4,f1_face_l4,u1_faceup_l4,d1_facedown_l4,h1_helix_l4 \
    --boost f1_face_l4=2 --boost u1_faceup_l4=2 --boost d1_facedown_l4=2 \
    --set face_weight=1.0 --set face_bake=false --set face_shell=false --set face_depth_weight=0.0 \
    --set multiscale_video=0.1 --set iterations=20000
python -m pacgs facecheck --run ... --splats f5_msvideo,f6_faceball --variant f1_face_l4,u1_faceup_l4,d1_facedown_l4
```

**Cost.** ~10 min generation (two clips), ~4 min matting, ~60 min training (395 frames).

**Success.** Face-grid identity at +30° elevation ≥ 0.45 (from 0.30), no rise in speckle, and the
eye-level numbers hold. **Risk:** five clips is ~400 frames and ~3 GB of images on the GPU; if VRAM is
tight, drop the body orbit's share rather than a clip, since the close-ups now carry the detail.

</details>

### Q17. Add the photo's own head crop as a training view  ★ do next, cheap

**Hypothesis.** Identity fell from 0.64 to 0.51 when the pasted photo layer went, because nothing
replaced the real photo detail. The photo is currently trained as one full-body view, where the face is
a small part of the loss. Adding the *same photo*, cropped to the head with the matching lens, injects
real face pixels at native resolution with an exact camera — no generation, no pasted geometry.

**Implement.** `pacgs/dataset.py`: `photo_face_record(subject, world, size, zoom)` — take the face
rig's frame-0 intrinsics, crop the native-resolution photo and its matte with `proxy.reframe_photo`,
write both, and emit a record with `kind: "video"` (exact camera, no pose refinement), `weight: 1.0`
and `share: 2.0`. Add it to the merged set in `pacgs train --photo_face` and in `i2g/orbit_stage.py`.

**Cost.** Minutes of coding, no generation; one training (can share the GPU with Q16's).
**Success.** Face-grid identity ≥ 0.58 without speckle rising above 0.85. **Risk:** the photo's face is
one viewpoint; too much weight could flatten the 3D face again, which the speckle and multi-angle
detail numbers will show.

### Q18. Why is detail stuck at 0.67×?  ★ diagnostic, cheap

**Hypothesis.** Either the trainer cannot resolve what the frames contain (optimisation or Gaussian
budget), or the frames disagree with each other at fine scale (data). The answer decides whether to
spend the next GPU hours on more pixels (Q4, Q14) or on training changes (Q7).

**Implement.** Nothing new. Train on the head close-up clip alone (79 frames + photo), which removes
all competition for steps, and facecheck it against those frames:
```
python -m pacgs train --run ... --variant f1_face_l4 --name x_faceonly --set iterations=20000 \
    --set face_weight=1.0 --set face_bake=false --set face_shell=false --set face_depth_weight=0.0 \
    --set multiscale_video=0.1
python -m pacgs facecheck --run ... --splats f5_msvideo,x_faceonly
```
**Cost.** ~30 min. **Read it as:** detail ≥ 0.85 means the data is fine and the full run is
step-starved (→ Q7, more boost); detail still ≈ 0.7 means the frames disagree at fine scale (→ Q4/Q14
for more pixels, or Q8 for denser frames).

### Q14. Close-up passes over the body

**Hypothesis.** What the head close-up did for the face, a 2× pass will do for jeans seams, the
tank-top ribbing, shoe straps and fingers.

**Design.** Zoomed orbits tracking: torso and arms (2×), hips and thighs (2×), calves and shoes (2.5×),
hands (3×; they are on her head, so prominent). Same machinery as Q16, tracking different keypoints.

**Implement.** `pacgs/orbit.py`: let `track` name any body keypoint (`i2g.body` indices) or a named
point, not just `"head"`; add the four rigs. Everything downstream already handles per-frame
intrinsics.

**Cost.** ~5 min generation per pass, ~20 min for four, plus one training (~60–70 min with ~700 frames).
**Success.** Detail (`sharp`) against each clip's own frames ≥ 0.6 in those regions, no speckle rise,
and visibly sharper clothing in the turntable. **Risk:** a tight crop loses context and the model may
drift between angles; keep zoom ≤ 2.5× for the body and watch each clip's matte-vs-control IoU.

### Q4. Generate at 720×1280

**Hypothesis.** The frames themselves limit the sides and back.
**Cost.** ~2.5× generation (~12 min per clip), more VRAM, slower training.
**Risk.** Wan 14B at 720×1280×81 is near this card's ceiling; try 656×1152 first, and expect ComfyUI to
offload (slow) if it does not fit. **Success:** sharper sides and back in the turntable, native-size
photo fidelity up, no loss of frame agreement.
**Order:** after Q18 says whether more pixels are the bottleneck.

### Q5. Second round with the new splat as the stand-in

The stand-in is still the old Qwen-view splat. Regenerate all clips from `f5_msvideo` (or the Q16
winner) and retrain: a better instruction sheet, and a head shape in the face clips' control that
already matches the reconstruction.
**Cost.** ~25 min generation + one training. **Success:** frame agreement and detail both up. Run once.

### Q13. A second and third subject  ★ before more tuning

Everything is tuned on one photo, and the full `.bat` path has only just been wired. Run it end to end
on a long-or-curly-haired subject and a loose-clothing subject.
**Cost.** ~2 h 15 each, unattended. **Success:** the face and body numbers land in the same range; any
stage that breaks tells us more than another tuning round on the same subject.

### Q7. Retune the training budget for the new data  ★ partly done

`grow_grad2d`, the Gaussian budget, MCMC versus clone/split, opacity and scale regularisation, and the
boost factor (3 → 5) were all tuned for 15 disagreeing views. There are now 240–700 consistent frames.
**Cost.** three or four trainings, which can share the GPU. **Success:** detail up without speckle up.

**Done so far: the step count.** Seven clips (630 frames) at 30k steps lost 1–2 dB at the elevations
that five clips had already learned. 40k recovers most of it — eye level 27.42 → 28.04 dB, all-angle
27.00 → 27.38, arcface 0.427 → 0.457, floaters 0.017 → 0.016% — for seven more minutes, and is now the
preset. **Still untested:** the per-clip boost. All five head clips are boosted 2× equally, so the
eye-level ring, the one a viewer looks at first, gets a fifth of the close-up budget. Worth one run at
`face=3` with the rest at 2.

### Q19. One agreed back  ★ new, from the round-5 review

**Problem.** The photo shows only the front, so everything behind her is invented — and each clip
invents it *differently*. At orbit 180° the eye-level head clip gives the vest wide flaps at the
shoulders, the +11° clip a notched racerback, and the +34° clip a flesh-coloured knot with a fabric bow
at the neckline. The control depth there is smooth, so none of this is geometry: it is the video model
filling in what it cannot see. The trainer then averages the disagreement, and a hallucination that is
consistent within its own clip (the bow) trains in as a lump.

**Measured (2026-09-20 night).** Fix A was implemented and run; the reseed was run as a control.

| Model | back at 180 deg | face (own clips, mean psnr) | sharp | photo | floaters |
|---|---|---|---|---|---|
| `f11_facedome40` (+34 clip with the bow) | smeared blob under the hair | 27.37 | 0.603 | 36.73 | 0.016% |
| **`f12_dome40_hi2`** (same, clip reseeded) | **clean** | 27.13 | 0.600 | 36.56 | 0.016% |
| `f13_dome40_crop` (Fix A, head clips cropped to 1.8 head radii) | **worse**: dark streaks and a tan patch | 25.86 | **0.643** | 36.61 | 0.022% |

**The reseed is the fix; the crop is a negative result as implemented.** Cropping removed the only
close-up supervision of the shoulders and upper back, and the body orbit's 480x832 frames were not
enough to hold that region on their own, so the smear got worse rather than better. Two things the crop
*did* do: sharpness rose from 0.603 to 0.643 (cropped views skip the multiscale zoom-outs and carry more
head pixels per step) and leave-frames-out rose to 25.0 dB. Worth revisiting as a **softer crop** - 2.5
to 3 head radii, keeping the shoulders - or as a per-pixel loss weight that fades out below the neck
rather than a hard cut. The code stays: `pacgs prepare --crop`, off by default.

**Fix A, cheap and principled: crop the head clips to the head.** Their job is the head; the torso
pixels they carry are both redundant (the body orbit covers the torso, consistently, and the photo
covers the front) and actively harmful when they disagree. The rigs already keep the head at the frame
centre, so the crop is a centred box of k x the head radius, projected: half-width = f * k * r / z, with
r the largest distance from the head centre to a facial keypoint (0.097 m here) and k ≈ 1.8, which
gives 434 px of an 832 px frame. Per-record `K` already exists, so a crop is just a shifted principal
point and a smaller image; the trainer reads each view's size from its file. Crop after generation, in
`pacgs.dataset.prepare(..., crop=k)`, so no regeneration is needed. Do **not** implement it by zeroing
the matte: the visual hull is carved from the mattes, and a zeroed torso would carve the torso away.
**Cost:** one prepare (2 min) + one training. **Success:** the back artefacts disappear from renders at
+34° and the face metrics do not move.

**Fix B, if A is not enough: make one clip authoritative for the back.** Generate the body orbit first
and pass its back frame as the `ref_image` for the close-up clips instead of the photo, so they start
from an agreed back rather than inventing their own. **Cost:** one generation per variant.

**Also worth doing either way:** a reseed is a valid escape hatch for a single bad clip (the bow was
seed 1001; seed 1002 is being tested), and `iou_splat` already catches the gross case — a clip that
re-styles the subject's *silhouette* is now dropped automatically (§6.5). A bow that lies inside the
silhouette is invisible to that check; the only current detector is looking at the contact sheet.

### Q20. Detect a hallucination that stays inside the silhouette  ★ new

`iou_splat` catches a clip that re-styles the *outline* (the mesh-proxy crown clip, 0.818). It cannot
see a flesh-coloured knot painted on a white vest: that clip scored 0.977. The only detector today is a
human looking at the contact sheet, and one such clip trains straight into the model.

**Proposed metric.** Render the splat proxy in colour at each frame's camera, blur both, and take the
largest block-mean colour difference over a grid of, say, 32x32 blocks inside the matte. The proxy is
blurry and its back is itself invented, so the *absolute* value means little — but a bow appearing in
one clip and not in the proxy is a large, spatially localised, temporally persistent difference, which
is a different signature from the diffuse difference of a sharper repaint.

**Validate before trusting:** the two +34 clips are a labelled pair (seed 1001 has the bow, seed 1002
does not), as are the two crown clips. A useful detector separates them; if it does not, do not ship it.
**Then use it for:** flagging the clip in the report, and — if it is reliable — regenerating that one
clip with the next seed automatically, which costs about 7 minutes.

### Backlog (unchanged, lower priority)

- **Q8** denser frames (121 per clip) or a second seed with agreement-based triage.
- **Q9** colour-match frames to the photo before training.
- **Q10** hair-edge mattes: check BiRefNet is not eating strands.
- **Q11** SeedVR2 video upscale as a cheaper alternative to Q4.
- **Q12** a synthetic ground-truth subject: the only way to test whether the invented back is *right*
  rather than merely consistent.
- **Q6** a low-angle body pass (under chin, under arms) — partly covered by Q16's `face_down`.

---

## 4. Suggested order

| Session | What | GPU |
|---|---|---|
| 1 | **Q16** (face up/down clips) + **Q17** (photo head crop), trained together and separately so each is attributable | ~1 h 20 |
| 2 | **Q18** diagnostic, in parallel with session 1's training if VRAM allows | ~30 min |
| 3 | **Q13**: one fresh subject end to end | ~2 h 15 |
| 4 | **Q14** body close-ups, or **Q4** 720p, whichever Q18 points at | ~1 h 30 |
| 5 | **Q5** second round, then **Q7** tuning | ~2 h |

Q16 and Q17 are the two that address what is visibly wrong now. Q13 matters before more tuning: a
second subject is worth more than another decimal place on this one.

---

## 5. Done, for the record

| | Outcome |
|---|---|
| Metrics upgrade | `pacgs facecheck`: multi-angle, artefact-sensitive. Shipped. |
| Q1 photo recolour instead of a baked layer | **Negative.** Worse on every measure; alignment too loose. Kept off by default. |
| Q2/Q3 face finishing and per-clip weighting | **Shipped** as per-view `weight`/`share` plus `--boost`; the close-ups get 3×. |
| Face priors and the baked layer | **Removed** when close-ups are present; the speckle fix. |
| Per-kind anti-aliasing | **Shipped** (`multiscale_video`), resolving the sharpness/aliasing trade. |

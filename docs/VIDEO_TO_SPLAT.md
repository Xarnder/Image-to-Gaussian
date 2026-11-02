# Approved orbit video to Gaussian splat

## Front image to generated 360 video

The GUI now defaults to **Front image → 360 video → Gaussian splat**. Choose one
front image and an experiment name. **Review candidates** generates five videos by
default (configurable from 1 to 20 per batch), then pauses. Play the candidate
videos, check blinking/eye motion and the complete orbit, and choose **Approve this
video & reconstruct**. **Generate more candidates** appends fresh seeds and retains
earlier options. Reconstruction settings remain editable before approval.

Alternatively choose **Generate one and continue automatically**. This generates
one new video and sends it directly to frame extraction, camera reconstruction and
Gaussian training. It records `automatic-first` selection, rather than claiming a
person reviewed the video. Camera quality gates still apply; automatic selection
does not promise that the generated orbit is geometrically consistent.

Generation uses the versioned `workflows/minimax_360.json`, copied from
`Xander 360 View Minimax H3 Workflow.json`. It preserves the exact eye-lock prompt,
MiniMax H3 model, `minimax_h3_flf2v_lora_v1.safetensors` at strength 1, sampler,
28 steps, resolution, five-second duration calculation, and identical first/last
frame wiring. Only input image, random seed and output filename are substituted.
Disabled/bypassed nodes are respected; note text is not executable configuration.
The Qwen3-VL file in this workflow is MiniMax's text encoder, not Qwen Image Edit.
There is no image-edit, Wan, SAM body, or synthesized still-view stage on this route.

ComfyUI starts automatically if necessary. Generation is serial, shares the GPU
lock with training, and unloads models before reconstruction. Stop retains completed
videos and cancels only this generation's queued/running prompt. Refreshing the
browser restores the current batch. Saved candidates can also be restored after
restarting the GUI; an interrupted external ComfyUI job must finish or be stopped
before launching another batch. Reset clears the GUI session, not saved files.

Candidates and their exact API graphs, seeds, source/workflow hashes and selection
record are in `output/<experiment>/<image>/orbit_candidates/<fingerprint>/`.
Selected-video reconstruction is in `output/<experiment>/<candidate>/video/final/`.
Changing the source image or bundled workflow creates a separate candidate set.
The existing **360 video** import route remains available.

Close-up portraits sometimes fail full-360 camera registration even if the video
looks plausible. Full coverage is still the default; the partial recipe is an
explicit option and produces only the recovered front/side arc.

### Portrait generation-to-splat validation — 2026-10-31

Launched the attached `Test-Close_Head-Shot.png` through the real browser GUI in
automatic-first mode. MiniMax generated `orbit_a716fb19f568.mp4`, seed
`97082330915038`: 124 frames at 24 fps, 1248×1248 pixels. The bundled workflow is
byte-identical to the supplied file (SHA256
`6c30fc5c1d27de38785848107df0148f2ed9c3ddb62a0c2640805279bcf10ead`).
All 62 sampled frames registered: 29,561 sparse points, 0.6893 px mean
reprojection error, 359.9577° orbit span and 7.1263° largest gap. No partial
coverage fallback was used.

The standard 15,000-step hull run completed, including evaluation/previews, in
1,015.2 seconds with 4.45 GiB peak allocated VRAM. Held-out foreground PSNR was
29.1063 dB. The final PLY contains 769,912 Gaussians and is 190,939,787 bytes.
`tools/verify_video_export.py` reloaded it, confirmed finite parameters and exact
agreement with the normalized checkpoint, then rendered eight observed views.
The H.264 preview contains 144 frames. Front, profiles and back are coherent;
fine hair is softer, and the uniform turntable reveals small artifacts at the
cropped shoulder edges, which are weakly constrained by this head-shot video.
Scores measure fit to generated frames, not unseen real-world anatomy.

Artifacts: `output/portrait-generated-360/orbit_a716fb19f568/video/final/`, including
`character.ply`, `turntable.mp4`, `verified_comparison.jpg`, `preview_8_angles.jpg`,
`report.json` and `verification.json`. Source video, API graph and selection record
are under the portrait's `orbit_candidates/994465d18734/` directory.

23 regression tests pass. A separate, explicitly labeled browser fixture verified
five playable candidate cards, selecting the third with edited training settings,
appending five more without losing earlier options, and reset behavior. Those
fixture videos reuse an existing clip; the real GPU end-to-end test generated one
new portrait video. The real MP4 was also played in the browser, and live stage
transitions were observed through completion. The explicit `automatic-first`
approval flag was additionally checked against the cached prepared video.

The `video_to_splat.py` reconstruction entry point remains video-only. It does not run Qwen, Wan, MiniMax,
SAM 3D Body, image editing, view generation, or the older image pipeline.
Its source is an already generated MP4, selected by manual review or the GUI's explicit automatic-first policy.

For the complete experiment and explanation, see
[MiniMax orbit video to Gaussian splat](D:/Image-to-Gaussian/next-stage-research/docs/minimax_orbit_video_to_gaussian_results.md).

## UI and Windows launchers

Open `run_splat_gui.bat` and choose **360 video** under **Reconstruction method**.
Browse for an MP4, play the source preview, tick the review approval checkbox, enter
an experiment name, then click **Start**. The default is the successful recipe:
alternate frames, maximum height 1280 px, and 15,000 training iterations. Results
appear under `output/<experiment>/<video-name>/video/final/`. The UI shows frame
preparation, camera reconstruction, training progress, comparisons, and the finished
turntable. **Open working folder** leads to the run directory; the PLY is in `final/`.

Choose **Image edit** for the existing photo or folder pipeline and its presets.
The image-only settings are hidden for video runs. A new video selection clears the
approval checkbox. Stop now interrupts a video run; preparation and calibration can
be reused, but interrupted Gaussian training starts again.

Both `run_image_to_splat.bat` and `run_batch_to_splat.bat` offer a method choice when
double-clicked. Dropping an MP4 onto either selects the video method and prompts for
approval. The batch launcher retains folder processing for images; its video method
accepts a single MP4. Existing image CLI flags still pass through unchanged.

```powershell
run_image_to_splat.bat --method video --input "D:\videos\orbit.mp4" --experiment orbit-test --approved-video
```

The launcher also accepts `--iterations`, `--stride`, and `--max-height` for videos.
Selecting video does not start ComfyUI or generate new views. An existing GUI process
must be closed and relaunched after installing these code changes.

```powershell
.venv-i2g/Scripts/python.exe -m pip install pycolmap==4.2.1
.venv-i2g/Scripts/python.exe video_to_splat.py `
  --input input/MiniMax_H3_360_Orbit_00005_.mp4 `
  --output output/minimax-orbit --approved-video
```

Only pass `--approved-video` after a person has reviewed the video. Approval is
stored with the source SHA256, so a different video cannot inherit it. The supplied
MiniMax clip was explicitly approved by the user. Video generation and its review UI
are outside this entry point.

Stages:

1. Decode every second frame, omitting the closing endpoint, and resize to at most
   1280 pixels high. BiRefNet supplies foreground alpha without changing RGB pixels.
2. Estimate cameras using foreground-only SIFT matches and COLMAP bundle adjustment.
   A shared pinhole camera avoids fitting lens distortion to AI-generated warping.
   Refuse to train if fewer than 90% of the sampled views form one reconstruction.
3. Initialize a visual hull using the recovered cameras and mattes. Train the existing
   native Windows gsplat backend, with no mesh-face priors or photo bake. Every tenth
   video view is withheld from Gaussian optimization for an interpolation check.
4. Export `final/character.ply`, a rendered `turntable.mp4`, source/render comparison,
   checkpoint, and measured `report.json`.

Use `--stage prepare`, `--stage calibrate`, or `--stage train` to stop at a boundary.
Preparation and calibration are cached. A new source, stride, or resolution requires
a new output directory. Repeating training writes the same final paths; use a new
output directory to retain distinct experiments. `--iterations` defaults to 15000.

The calibration is estimated, not the known camera path of the previous PAC-GS
controlled-video experiments. A plausible-looking AI orbit can still deform. Frame
registration and rendered comparisons must be checked, rather than assuming uniform
angular motion from timestamps. Monocular reconstruction has no measured absolute
scale; this entry point normalizes the sparse foreground height to a nominal 1.7 m.

The video contains no elevated views. Top/bottom surfaces therefore remain weakly
constrained. Held-out scores measure agreement with this generated video, not fidelity
to a real person. Held-out images participate in camera calibration, but not splat
colour or silhouette training.

Relevant earlier findings: `plan.md`, `quality_plan.md`, PAC-GS sprint results,
`next-stage-research/docs/pacgs_face_fix.md`, and `ghost_body_fix.md`. Existing image
pipeline files and results remain separate.

## First completed run — 2026-10-30

Input: `input/MiniMax_H3_360_Orbit_00005_.mp4` (124 frames, 1088×1920, 24 fps).
All 62 sampled frames registered in one model: 7,324 sparse points, 0.667 px mean
reprojection error, 357.0° camera coverage. Training used 56 frames at 725×1280,
with six held out, and finished 15,000 iterations in 13.1 minutes.

`output/minimax-orbit/final/character.ply`: 216,418 Gaussians, 53,673,275 bytes.
The held-out foreground PSNR averaged 30.54 dB (range 29.28–31.66). Re-imported
PLY parameters are finite and agree with the training checkpoint after quaternion
normalization (largest difference 1.2e-7). The re-imported model was rendered and
inspected at eight views, including head close-ups. It retains a coherent front,
profile and back; fine hair and skin texture are softer than the source frames.

Additional first-run artifacts: `face_comparison.jpg`, `preview_8_angles.jpg`, and
`verification.json`. These were produced during final inspection. `comparison.jpg`
shows video on the left and splat on the right in each pair. The H.264 turntable has
144 frames and covers a complete orbit.

## Full UI repeat — 2026-10-30

A fresh run was launched through the browser with the same approved MP4 and all
15,000 iterations. Output: `output/minimax-ui-validation/MiniMax_H3_360_Orbit_00005_/video/final/`.
All 62 decoded frames and all 62 masks were byte-identical to the first run.
Calibration again registered 62/62 frames (7,306 points, 0.66735 px reprojection
error, 357.02° coverage). Training configuration and six held-out views matched.

Held-out PSNR was **30.5722 dB**, compared with **30.5415 dB** originally (+0.0307 dB).
The exported PLY contains **216,510 Gaussians** and is 53,696,091 bytes. Every
parameter is finite; re-imported position, scale, normalized rotation, opacity,
and DC colour match the checkpoint exactly. The preview contains 144 frames.
Eight paired full-body and head views were visually inspected and show equivalent
quality. Small differences are expected because camera reconstruction is rerun.

The browser was observed receiving live preparation, camera, iteration, and export
stage transitions, then showing all four stages Done and all bars at 100%.
A buffering issue found in the preliminary test was fixed in the worker output
reader. Six integration tests pass, including progress arriving before the child
exits, export percentages, approval enforcement, and preserved image routing.
Additional calibration/export phase events were added during the full run and
their UI consumption is covered by the stage test; that already-running worker
used the earlier stage-boundary export events.

`baseline_vs_ui.jpg`, `baseline_vs_ui_heads.jpg`, and `verification.json` retain
the quality evidence. `tools/verify_video_repeat.py` reproduces the PLY checks and
paired renders. Dashboard screenshots are in `output/minimax-ui-validation/`.


## Close-up videos with a missing back — 2026-10-31

Choose **Camera coverage → Front and sides** in the GUI (restart the app after
updating). This explicitly accepts a partial reconstruction. It uses only cameras
that COLMAP recovered, skips unregistered frames, and does not invent rear poses.
The preview sweeps the recovered arc and returns instead of passing through the
unobserved back. The exported PLY is a partial bust, not a complete head.

The equivalent CLI option is `--coverage partial`, supported by both
`video_to_splat.py` and `run_to_splat.py`. Full coverage remains the default and
retains its 90% registration / 45-degree maximum-gap checks. Partial mode requires
at least 12 recovered cameras, 1,000 sparse points and a 60-degree observed arc.
Cached failed calibration attempts can be reused; the requested coverage policy
is rechecked even when a successful calibration file exists. Input hashes and
approval checks are unchanged.

Partial mode defaults to sparse-point initialization; full mode retains visual-hull
initialization. For experiments, `video_to_splat.py --initialization hull|sparse`
overrides that choice. Sparse initialization uses COLMAP geometry and colours;
all training and silhouette losses still use only the registered frames. Camera
calibration (and sparse initialization) sees held-out images, so held-out scores
measure splat fitting/generalization, not an independent camera-calibration test.

### Experiment: MiniMax_H3_360_Orbit_00019_.mp4

The supplied close-up registered 46/62 frames with 13,872 sparse points, 0.683 px
reprojection error and a 138.77-degree rear gap. The usable arc is 221.23 degrees.
Two 2,500-step experiments used the same recovered cameras and held-out frames:

| Initialization | Held-out foreground PSNR | Exported Gaussians |
|---|---:|---:|
| Visual hull | 26.61 dB | 379,191 |
| Recovered sparse points | 31.21 dB | 300,784 |

The sparse result has substantially fewer cheek/hair artifacts in the profile
comparisons. Results are preserved in `output/closeup-partial-hull-2500/` and
`output/closeup-partial-sparse-2500/`. The longer selected run is stored separately
in `output/closeup-partial-final/`; its `final/report.json` records final metrics.
Python failures now write their tracebacks to `run.log` as well as the console.

The selected 7,500-step sparse run completed in 471.5 seconds (including evaluation
and previews), using 3.99 GiB peak allocated VRAM. Held-out foreground PSNR reached
**32.12 dB**. `output/closeup-partial-final/final/character.ply` contains **574,044
Gaussians**. Reloading the PLY produced finite parameters matching the checkpoint
exactly; eight front/side source-versus-PLY views were rendered and inspected in
`verified_comparison.jpg`. The partial sweep preview contains 144 frames. Fine hair
edges remain imperfect, and the unregistered rear sector is not a verified surface.
All 11 partial-coverage and existing video integration regression tests pass.


## Launcher and GUI recipes

Both `run_image_to_splat.bat` and `run_batch_to_splat.bat` now offer a video recipe
menu when launched interactively (including dropping an MP4 on the launcher):

1. Full 360: full coverage, silhouette hull, 15,000 steps.
2. Close-up / skip missing back: partial coverage, recovered sparse points, 7,500 steps.
3. Partial silhouette hull: partial coverage, hull initialization, 7,500 steps.

The GUI exposes the same three recipes plus individual controls for camera
coverage, starting geometry (automatic, recovered points, silhouette hull),
training steps, frame stride and maximum height. Selecting a recipe fills its
coverage, geometry and step defaults; explicit field edits take precedence.
No option invents cameras for missing views.

For scripted batch launches:

```powershell
run_image_to_splat.bat --method video --video-preset closeup --input "D:\clips\head.mp4" --experiment head-test --approved-video
```

Optional overrides: `--coverage full|partial`, `--initialization auto|sparse|hull`,
`--iterations 2500`, `--stride 2`, `--max-height 1280`. Fully specified commands
run without the interactive recipe prompt. Existing image routing is unchanged.
Restart an already-running GUI server after updating the Python files.


GUI verification: the close-up preset was selected in the browser, its 7,500-step
default verified, and a 2,500-step override launched via Start. Using the existing
verified frames/calibration, all four GUI stages completed and the UI showed 100%
and Finished. The output report confirms partial coverage, sparse initialization
and 2,500 steps; held-out PSNR is 31.25 dB. Artifacts are under
`output/closeup-gui-options-test/MiniMax_H3_360_Orbit_00019_/video/final/`.
Switching all three recipes was checked in the browser. Both real BAT launchers
expose the flags through their video help; all 13 routing/coverage/menu tests pass.

### Reset for another video/project
Click Reset for new project, then slide fully right and release. Cancel or an incomplete slide preserves the form. Reset clears video, project, approval, preview, console and progress while retaining recipe settings and all saved files. Active runs cannot be reset. Browser checks and all 16 regression tests passed.
### Custom MiniMax H3 prompt

Enable **Custom MiniMax H3 prompt** to edit the exact prompt loaded from the bundled working workflow. **Restore working prompt** resets the editor. With the toggle off, the unchanged workflow prompt is used regardless of editor contents. Edits apply to every candidate and every image in an automated folder batch. The effective prompt is saved in generation metadata and each submitted API graph; prompt changes use a separate candidate directory. Model, LoRA and sampling settings remain unchanged.

### Early candidate approval

The candidate count is a maximum. While a review batch is running, approve any completed video with **Approve this video, cancel rest & reconstruct**. The app interrupts its current ComfyUI job, skips the remaining candidates, waits for that job to stop, releases generation resources, and starts reconstruction from the selected video. Completed candidates remain saved. Unrelated ComfyUI jobs are not cancelled.

### Automated folder batches

The GUI now defaults to **Automatic — aim for 360°, skip unusable views**. It first applies the full-orbit checks, then falls back to the partial checks if coverage is incomplete. Partial fallback still requires at least 12 aligned cameras, 1,000 sparse points and a 60° observed arc. Fallback uses sparse initialization, limits the preview to the observed arc, and records the reason in calibration/report JSON and an incomplete-result notice in the GUI (including per-image batch results). It does not recover from unrelated decoding, GPU, or training failures. Explicit CLI `--coverage full` remains strict; `--coverage auto` enables fallback.

Under Front image → 360 video → Gaussian splat, choose **Folder of front-facing images — automated batch**. This selects automatic-first mode. Review mode is also accepted if the candidate count is exactly one; the single candidate is then accepted automatically for every image. Multi-candidate manual review is rejected for folder batches.

PNG, JPG, JPEG, WebP, BMP and TIFF images run in the chosen order. Drag rows or use Alt+Up/Alt+Down; A–Z resets the list. Include subfolders is optional. Order is saved and restored. Each image completes video generation, frame extraction, camera recovery, training and export before the next starts. Outputs are saved separately as `output/<experiment>/<image-name>/video/final/<image-name>-and-<experiment>-character.ply`; ambiguous image names receive a short hash suffix. The GUI shows each image's status and a Show PLY in Explorer button for completed results. `generated_batch.json` in the experiment folder records the source-to-output mapping and errors.

Stop after this one finishes the current image, including reconstruction. Stop now cancels the active stage and prevents subsequent images from starting. Failed images are recorded and the remaining images continue. Start again to retry failures or continue after stopping; finished splats are skipped by default. Rebuild finished ones preserves previous work under a video_previous_<timestamp> folder. The saved batch and order are restored after restarting the GUI, but execution resumes only when Start is clicked.

### Final filenames and shared GUI features

All video final deliverables use the original image name and experiment, matching the image-edit pipeline: `<image>-and-<experiment>-character.ply`, `-turntable.mp4`, `-comparison.jpg`, `-splats.pt`, and `-report.json`. Source provenance is recorded in `output_naming.json`. Duplicate image stems receive unique suffixes. Preview and Explorer links accept both new and legacy names. `tools/name_video_outputs.py --apply` repairs legacy video filenames only when the original image is known from generation records, and refuses destination collisions.

MiniMax folder mode shares image ordering, saved order, subfolder discovery, duplicate-name handling, skip-finished/rebuild, stop-now/stop-after-current, per-image status, continue-on-failure, reset, previews, console controls, and output reveal behavior. Image-edit-specific stages, presets and flags remain confined to that pipeline; MiniMax uses its own prompt, video selection and reconstruction controls.

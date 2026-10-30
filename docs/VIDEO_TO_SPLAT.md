# Approved orbit video to Gaussian splat

This is a separate video-only entry point. It does not run Qwen, Wan, MiniMax,
SAM 3D Body, image editing, view generation, or the older image pipeline.
The source is an already generated, manually approved MP4.

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

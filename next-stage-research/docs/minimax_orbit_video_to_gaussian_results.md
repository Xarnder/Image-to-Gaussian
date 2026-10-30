# MiniMax orbit video to Gaussian splat

**Date:** 30 October 2026  
**Status:** completed reconstruction and visual inspection on one manually approved video  
**Entry point:** [video_to_splat.py](D:/Image-to-Gaussian/video_to_splat.py)

A pre-generated MiniMax 360° orbit video produced a coherent static Gaussian splat using a much simpler reconstruction path than the previous image pipeline. We extracted frames, isolated the person, recovered cameras from matching foreground details, and trained the existing gsplat backend. No image-edit model, generated anchor images, body scaffold, depth-control video, or face bake was needed.

The user judged this result substantially better than previous project experiments. The measurements support a successful reconstruction of this particular clip: all 62 sampled frames registered together, camera reprojection error averaged 0.667 pixels, and six frames excluded from splat training averaged 30.54 dB foreground PSNR. This is a strong first result, rather than a controlled ranking across all earlier subjects and datasets.

## The result

| Measurement | Result |
| --- | --- |
| Supplied video | `MiniMax_H3_360_Orbit_00005_.mp4` |
| Source dimensions and rate | 1088 × 1920, 24 fps |
| Source length | 124 frames, approximately 5.17 seconds |
| Sampled images | 62, at 725 × 1280 |
| Registered cameras | 62 of 62 in one reconstruction |
| Reconstructed sparse points | 7,324 |
| Mean feature reprojection error | 0.667 pixels at the processed resolution |
| Measured camera sweep | Approximately 357° across sampled frames |
| Images used for splat training | 56 |
| Images withheld from splat training | 6 |
| Training iterations | 15,000 |
| Training loop time | 786.9 seconds, or 13.1 minutes |
| Training stage including evaluation and initial export | 796.1 seconds, or 13.3 minutes |
| Gaussians before final cleanup | 241,095 |
| Gaussians removed during cleanup | 24,677 |
| Final Gaussians | 216,418 |
| PLY size | 53,673,275 bytes, or 53.7 MB |
| Held-out foreground PSNR | Mean 30.54 dB; range 29.28–31.66 dB |

The timing excludes video generation, manual selection, environment setup, frame preparation, and camera reconstruction. It is not an image-to-finished-model runtime.

![Rendered model at eight angles](D:/Image-to-Gaussian/output/minimax-orbit/final/preview_8_angles.jpg)

The output retains a continuous body and head around the inspected orbit. Fine hair and skin texture are somewhat softer than the video. There is no visible detached face plate in these inspected views.

## The complete process

```text
Generate candidate orbit videos outside this pipeline
    ↓
Person reviews the videos and selects a successful 360° orbit
    ↓
Approved MP4 and approval recorded against its SHA256
    ↓
OpenCV extracts alternate frames and resizes them
    ↓
BiRefNet estimates a foreground alpha mask for each image
    ↓
COLMAP matches foreground features and recovers cameras
    ↓
Camera coverage is checked and the scene is normalized
    ↓
Silhouettes define a visual hull for Gaussian initialization
    ↓
gsplat optimizes one static model from the calibrated frames
    ↓
Cleanup, standard Gaussian PLY export, held-out evaluation
    ↓
Reload the PLY and inspect renders and a complete turntable
```

### Generate and approve the video

The upstream generator was identified by the user as MiniMax-H3 with Pablo Dawson’s 360 Orbit LoRA. The model card describes using the same image as the first and last keyframe, aiming for a frozen subject and a camera that completes an orbit. Its training examples were rendered from static human Gaussian splats, which makes the intended motion particularly relevant to reconstruction. These are the author’s descriptions of the LoRA, not a guarantee that every generated clip is geometrically consistent. [LoRA model card](https://huggingface.co/pablodawson/MiniMax-H3-360-Orbit-LoRA)

The user reported that roughly one in four attempts produced a sufficiently good orbit and explicitly supplied this successful MP4. That estimate is an operational observation, not a measured acceptance rate from this experiment. Manual selection remains part of the method.

Review should look for a complete orbit, a stable pose, consistent limbs and clothing, continuous camera movement, and a plausible return to the front. A clip can look attractive while changing the person’s shape; such changes make static reconstruction harder.

The reconstruction command uses `--approved-video` to record approval in `approval.json`, tied to the file’s SHA256. A different file cannot inherit that approval. The script records a human decision; it does not judge the video itself. The supplied clip’s generation prompt, seed, sampling steps, and original workflow were not captured in this run, so the reproducible procedure begins with the MP4.

### Extract the frames

OpenCV decoded the video. We kept frames 0, 2, 4, and so on through 122, omitting the final endpoint from consideration. This yielded 62 observations distributed around the orbit. The images were resized to a maximum height of 1280 pixels, giving 725 × 1280 PNGs.

This provided close neighboring views without the cost of training on all 124 frames. Angular spacing was recovered later from the images; it was not assumed to be a fixed number of degrees per frame.

### Isolate the person

The existing BiRefNet model produced a soft foreground alpha mask for every frame. The RGB images were retained as decoded and resized; segmentation did not repaint them.

Two masks served different purposes. The soft alpha supplied the silhouette and image-compositing target for training. For camera estimation, alpha was thresholded above 0.8 and eroded with a 5 × 5 kernel, keeping feature detection away from uncertain edges.

This distinction mattered because the source video contains conspicuous background markers. We did **not** treat those markers as surveyed calibration targets or rely on them for camera recovery. Foreground masking excluded the background from SIFT feature detection, so the recovered motion was supported by details on the person.

### Recover the actual cameras

COLMAP, accessed through `pycolmap`, performed structure from motion: finding recurring image details and jointly estimating their 3D positions and the camera poses that explain them. Bundle adjustment refines those estimates to reduce image reprojection error. [COLMAP tutorial](https://colmap.github.io/tutorial.html)

The implemented settings were:

| Setting | Value |
| --- | --- |
| Camera model | One shared `SIMPLE_PINHOLE` camera |
| Intrinsics | Shared focal length, centered principal point, no distortion parameters |
| Features | SIFT, up to 12,000 features per image |
| SIFT peak threshold | 0.003 |
| Matching | Exhaustive matching between image pairs |
| Execution | CPU, 12 threads |
| Initial pair minimum inliers | 60 |
| Initial minimum triangulation angle | 8° |
| Camera registration minimum inliers | 20 |
| Reconstruction selection | Connected model with the most registered images |

All 62 images registered in a single model, with 7,324 sparse 3D points. The fitted focal length was approximately 1511.1 pixels at 725 × 1280. Mean reprojection error was 0.667 pixels. Inspection of camera positions showed a continuous sweep of approximately 357°, with nearly constant distance from the foreground center.

That error measures how well matched features agree with the fitted cameras and points. It is not a direct measurement of anatomical accuracy. Together with full orbit coverage and the later renders, it provides evidence that this clip is sufficiently consistent for reconstruction.

The script requires at least 90% of sampled frames in the selected model. The current code also rejects newly computed calibrations spanning less than 300° or having an angular gap larger than 45°. For this first run, the continuous orbit was inspected explicitly; the angular checks were added afterward. Existing cached calibrations are reused.

### Normalize and initialize the scene

Camera up vectors establish a Y-up orientation, and the first view determines the front. Monocular reconstruction has arbitrary scale: the code normalizes the robust sparse foreground height to a nominal 1.7 metres. This is a convenient coordinate scale, not a measurement of the person’s actual height.

The sparse points provide robust scene bounds using the 1st and 99th percentiles. This prevents a few incorrect triangulations from creating an unnecessarily large volume. The sparse cloud is **not** directly used as the complete initial Gaussian cloud.

Instead, the trainer builds a visual hull. Imagine projecting each silhouette into 3D as a viewing cone: the space supported by those silhouettes approximates the person’s volume. The code samples its surface at a 0.012 m voxel spacing, allowing some silhouette disagreement. This produced 18,078 initial surface points. Colors were sampled from suitable views.

No SAM 3D Body mesh was fitted. Consequently, a body model’s guessed head position or profile could not constrain this initialization incorrectly.

### Train one static Gaussian splat

The new entry point reused the project’s established `i2g.splat` trainer. The renderer is gsplat, a CUDA library that renders Gaussians and supplies gradients for optimization. Training adjusts each Gaussian’s location, shape, rotation, opacity, and color so rendered views match the supplied images. [gsplat documentation](https://docs.gsplat.studio/main/)

The run used these settings:

| Component | Setting and purpose |
| --- | --- |
| Training budget | 15,000 iterations |
| Supervision | 56 images; six excluded from Gaussian optimization |
| Color model | Spherical harmonics degree 0, giving each Gaussian a view-independent color |
| Image losses | L1 pixel error plus SSIM, with SSIM weight 0.2 |
| Perceptual loss | Existing VGG-19 loss, weight 0.1 |
| Silhouette loss | Rendered alpha versus BiRefNet alpha, weight 0.5 |
| Background | Random training background, composited consistently into prediction and target |
| Initial resolution | Half resolution for the first 1,000 steps |
| Multiscale sampling | Existing trainer settings, including 0.1 for video views |
| Densification | Clone and split strategy, starting at step 500 and ending at 12,000 |
| Densification threshold | Absolute image-gradient threshold 0.0002 |
| Shape regularization | Anti-needle penalty, weight 0.1, axis-ratio threshold 6 |
| Camera adjustment during splat training | Disabled |
| Per-view color correction | Disabled |
| Face bake, shell, depth prior and mesh initialization | Disabled |

The first frame is internally tagged `photo` to satisfy the existing trainer’s input convention. It is still a video frame; no separate source photograph was supplied. Its sampling share was set to 1/62, so it did not receive the old pipeline’s strong front-photo preference. Video views had loss weight 1.0.

Fixing cameras here means keeping the **recovered** cameras after COLMAP has optimized them. It does not mean assuming that the LoRA supplied exact camera poses. This choice worked in this run; there was no separate ablation proving that allowing further refinement would be worse on this clip.

### Clean and export

Training ended with 241,095 Gaussians. Cleanup removed low-opacity Gaussians and points insufficiently supported by the foreground masks, with an additional elongated-Gaussian filter. The final model contains 216,418 Gaussians; the cleanup report records zero removals by the final needle filter.

The output is a standard binary Gaussian PLY containing position, opacity, scales, rotations, and color coefficients. The exporter normalizes quaternion rotations and pads unused spherical-harmonic bands with zeros for viewer compatibility. It is a static Gaussian representation, not an animated model or polygon mesh.

The PLY was reloaded and checked for finite values. Its parameters matched the training checkpoint after quaternion normalization, with a maximum quaternion discrepancy of approximately 1.2 × 10⁻⁷. Renders of the reloaded PLY were inspected, including head close-ups. FFmpeg encoded a 144-frame H.264 turntable.

## Why this worked so well

**The strongest explanation is better input consistency combined with dense coverage and successful camera recovery.** The existing trainer could fit a coherent set of observations without compensating for the disagreements that complicated earlier pipelines. The individual contributions below are supported by the run and previous diagnoses, but were not isolated in new ablation experiments.

### One accepted video supplied closely related views

Independent image edits can each produce a plausible person while disagreeing about the face, clothing, pose, or silhouette. This accepted clip supplied a continuous sequence whose foreground features could be tracked through the full turn. All 62 camera registrations and the held-out renders are evidence that it worked as a multiview dataset.

This does not imply that any video is more consistent than edited images. Selection of a successful clip was essential. The reported failed attempts remain part of the upstream cost.

### The side views actually constrained the body

The earlier ghost-body investigation traced major defects to missing side views. With only diagonal observations, the reconstruction could develop an exaggerated, diamond-shaped cross-section. Here the dense orbit included both sides, the back, and intermediate angles, all within one reconstruction. Those observations constrained the body’s depth directly. [Earlier diagnosis](D:/Image-to-Gaussian/next-stage-research/docs/ghost_body_fix.md)

### Camera estimation no longer depended on a guessed body mesh

The older registration stage fitted human keypoints against a scaffold inferred from one photograph. Errors in that scaffold’s profile could cause useful side views to be rejected or fitted poorly. This run estimated camera poses from repeated foreground details instead. A coherent clip gave conventional structure from motion the observations it needed.

Earlier research warned that SfM can fracture on generated videos. That warning still applies to inconsistent inputs. This result shows that an accepted MiniMax clip can meet the method’s requirements; it does not establish that camera recovery will work on every candidate.

### The face was learned with the rest of the head

Previous experiments diagnosed a detached face layer caused by baking the photo or applying face constraints at an incorrect scaffold depth. This run had no baked layer and no mesh-based face constraint. The face, hair, and skull appearance were optimized together from the orbit views. The inspected profiles remained continuous. [Face findings](D:/Image-to-Gaussian/next-stage-research/docs/pacgs_face_fix.md)

![Source and reconstructed head comparisons](D:/Image-to-Gaussian/output/minimax-orbit/final/face_comparison.jpg)

Each pair shows the source video on the left and the reloaded Gaussian PLY on the right.

### Earlier trainer improvements still contributed

The simpler orchestration retained work already validated in this project: foreground alpha supervision, visual-hull initialization, densification, multiscale training, flat color, anti-needle regularization, and cleanup. The improvement was not achieved by discarding all the research. We reused the reconstruction machinery while giving it better observations and avoiding stages that could introduce new disagreement. [Quality experiments](D:/Image-to-Gaussian/quality_plan.md)

The previous PAC-GS video experiments had already demonstrated strong results from consistent video frames. They generated those frames from a controlled proxy. This experiment supplied a successful external video and recovered its cameras afterward, removing the need to build and repaint a proxy first. [PAC-GS results](D:/Image-to-Gaussian/next-stage-research/docs/pacgs_results_sprint1.md)

## What the evaluation establishes

The six withheld video frames were 10, 30, 50, 70, 90, and 110. They participated in camera calibration, but their color and silhouette targets were excluded from Gaussian training and visual-hull initialization. The 30.54 dB score measures foreground agreement at those intervening views. It is an interpolation test of reconstruction quality, not a test against a real 3D scan.

The older experiments used different subjects, resolutions, hold-out sets, and sometimes camera-refitting protocols. Their PSNR values should not be subtracted from this result to claim a numerical improvement. The user’s preference and the visible coherence are meaningful; a controlled quantitative comparison would require matched inputs and evaluation.

The single orbit provides little evidence for surfaces viewed from above or below. Fine texture is softer than the source frames, and any stable anatomical invention in the video can be reproduced by the splat. Repeat this workflow on additional approved clips before treating the present success as a general success rate.

## Tools and environment

The reconstruction ran natively on the project’s Windows machine and RTX 4090 using the existing `.venv-i2g` environment. It did not use Docker or WSL. ComfyUI was not required for reconstruction; the launcher only checks an existing server for conflicting work and can ask it to unload models.

| Tool | Installed version or location | Role |
| --- | --- | --- |
| Python | 3.11.16 | Pipeline orchestration |
| OpenCV | 4.11.0.86 | Video decoding, resizing, mask erosion, preview frames |
| BiRefNet | Existing `models/birefnet` weights | Foreground alpha estimation |
| Transformers | 4.57.6 | Loads the local BiRefNet model |
| pycolmap | 4.2.1 | SIFT, matching, camera reconstruction and bundle adjustment |
| NumPy | 1.26.4 | Camera and scene coordinate calculations |
| PyTorch | 2.8.0+cu128 | GPU tensor operations and optimization |
| gsplat | 1.4.0 | Gaussian rasterization and densification |
| pytorch-msssim | 1.0.0 | SSIM image loss |
| VGG-19 | Existing local perceptual weights | Perceptual training loss |
| Pillow | 12.3.0 | Images and comparison sheets |
| imageio-ffmpeg | 0.6.0, bundled FFmpeg 7.1 | H.264 preview encoding |

The trainer and PLY writer are the project’s existing code, including SplatKit-derived components credited in [THIRD_PARTY_NOTICES.md](D:/Image-to-Gaussian/THIRD_PARTY_NOTICES.md). MiniMax and the LoRA belong to upstream video generation; their weights were not loaded during this reconstruction.

## Reproduce the reconstruction

From `D:\Image-to-Gaussian`, with the existing project environment and local models installed:

```powershell
.venv-i2g/Scripts/python.exe -m pip install pycolmap==4.2.1

.venv-i2g/Scripts/python.exe video_to_splat.py `
  --input "D:\Image-to-Gaussian\input\MiniMax_H3_360_Orbit_00005_.mp4" `
  --output "D:\Image-to-Gaussian\output\minimax-orbit-reproduction" `
  --approved-video --stride 2 --max-height 1280 --iterations 15000
```

Use a new output directory to preserve the original result. Preparation and calibration are cached. `--stage prepare`, `--stage calibrate`, and `--stage train` allow stopping at those boundaries. Repeating training in the same directory replaces the final outputs. The script also uses the project GPU lock to prevent overlapping pipeline runs.

For another video, replace the input and output paths and review that clip before setting `--approved-video`. Preserve the approval step even if generation later becomes automated.

## Windows launcher and UI integration

`run_splat_gui.bat` now opens a UI with an **Image edit / 360 video** method choice.
Video mode accepts a local, pre-generated MP4 and requires an explicit review
checkbox before starting. It previews that source clip, then shows extracted frames,
source/render comparisons, and the rendered turntable as they become available.
The old photo and folder routes retain their existing presets and settings.

Both command-line `.bat` launchers use `run_to_splat.py` to offer the same method
choice. An MP4 selects the video route automatically; explicit `--method image`
or `--method video` is also supported. `i2g/video_options.py` keeps the GUI and
terminal video defaults identical: stride 2, maximum height 1280, 15,000 steps.
Outputs use `output/<experiment>/<video-name>/video/final/`.

The video worker emits flushed structured progress events for preparation,
calibration, training, and export. The GUI consumes those events immediately and
streams state to the browser. A test exposed a pipe read that could buffer short
updates until the worker exited; using `read1` with incremental UTF-8 decoding
fixed it. The regression test holds a child process open and confirms its progress
arrives before it exits. Stage tests cover export percentages and completion, while
routing tests verify strict approval and preservation of the image routes.

Camera recovery reports feature extraction, matching, reconstruction, and coverage
validation phases. Training reports actual iteration counts, loss, and Gaussian
counts. Export reports view evaluation and turntable rendering counts. The overall
bar is weighted by stage; it is an estimate of total work, not a quality score.

## Full UI reproduction result

The same approved MP4 was run through the browser for all 15,000 iterations in a
fresh output directory, without reusing extracted frames or camera calibration.
Every extracted frame and mask was byte-identical to the original. All 62 views
registered again, with 0.66735 px reprojection error and 357.02° coverage.

The UI run achieved **30.5722 dB** held-out foreground PSNR versus **30.5415 dB**
for the original, a +0.0307 dB difference. Both runs used the same training config
and held-out image names. The new PLY contains **216,510 Gaussians**; re-imported
parameters are finite and match the saved checkpoint. Eight paired body and head
renders were inspected, with equivalent visual quality. This supports repeatability
for this clip; it does not establish a success rate across different videos.

Live stage changes and iteration counts were observed in the browser, including
the switch to export and final 100% completion. Preparation took 16 seconds,
camera recovery 21 seconds, training stage 13 minutes 28 seconds, and export
6 seconds. The new result is at
[UI repeat PLY](D:/Image-to-Gaussian/output/minimax-ui-validation/MiniMax_H3_360_Orbit_00005_/video/final/character.ply).
The original remains intact. See the
[operating guide's validation record](D:/Image-to-Gaussian/docs/VIDEO_TO_SPLAT.md)
for test scope and the added phase-event details.

![Original on the left and UI repeat on the right in each pair](D:/Image-to-Gaussian/output/minimax-ui-validation/MiniMax_H3_360_Orbit_00005_/video/final/baseline_vs_ui.jpg)

## Files to retain

| Artifact | Location |
| --- | --- |
| New reconstruction entry point | [video_to_splat.py](D:/Image-to-Gaussian/video_to_splat.py) |
| Short operating guide | [VIDEO_TO_SPLAT.md](D:/Image-to-Gaussian/docs/VIDEO_TO_SPLAT.md) |
| Original approved MP4 | [Source video](D:/Image-to-Gaussian/input/MiniMax_H3_360_Orbit_00005_.mp4) |
| Approval and source identity | [approval.json](D:/Image-to-Gaussian/output/minimax-orbit/approval.json), [input.json](D:/Image-to-Gaussian/output/minimax-orbit/input.json) |
| Calibration summary | [calibration.json](D:/Image-to-Gaussian/output/minimax-orbit/calibration.json) |
| Training image and camera records | [records.json](D:/Image-to-Gaussian/output/minimax-orbit/records.json) |
| Final Gaussian model | [character.ply](D:/Image-to-Gaussian/output/minimax-orbit/final/character.ply) |
| Full configuration and measured results | [report.json](D:/Image-to-Gaussian/output/minimax-orbit/final/report.json) |
| Export verification | [verification.json](D:/Image-to-Gaussian/output/minimax-orbit/final/verification.json) |
| Source and render comparisons | [comparison.jpg](D:/Image-to-Gaussian/output/minimax-orbit/final/comparison.jpg) |
| Rendered orbit | [turntable.mp4](D:/Image-to-Gaussian/output/minimax-orbit/final/turntable.mp4) |

Keep the decoded frames, masks, feature database, sparse reconstruction, and checkpoint alongside these files when preserving the full experiment. They allow calibration and evaluation to be inspected without regenerating the video.

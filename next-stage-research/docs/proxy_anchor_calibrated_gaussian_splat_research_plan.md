**PAC-GS**

Proxy-Guided, Anchor-Calibrated
Generative Multiview Reconstruction

A research plan for static human Gaussian splatting from a single image

**Research project plan - v0.1**

Idea stage: method design, implementation plan and experimental programme

19 September 2026

*Status: hypothesis-driven proposal - no performance claims are made until experiments are run.*

# Executive summary

This project investigates whether a high-quality static 3D Gaussian splat of a human character can be reconstructed from a single reference image by replacing post-hoc camera estimation with a deliberately calibrated generative capture process. The central idea is to build a rough, stationary human proxy in Blender, prescribe the camera trajectory and camera intrinsics exactly, render a dense control sequence, and use a video-to-video generative model to transform that proxy into the target character while preserving the prescribed viewpoint progression. The generated frames are then treated as synthetic multiview observations for Gaussian-splat training.

The critical technical distinction is that the generated character does not need to retain the proxy's exact body geometry. Wider shoulders, a deeper face, different torso thickness, more hair volume, or different garment volume are acceptable if the generator settles on one coherent 3D subject and renders that same subject consistently from every view. What breaks reconstruction is view-dependent morphing or an effective camera path that no longer matches the Blender camera labels.

To make camera fidelity measurable rather than assumed, the project introduces persistent 'Super Anchors': rigid, uniquely identifiable calibration objects with known 3D world coordinates that are included in the control render and required to persist in the AI-generated sequence. Their detected 2D locations provide 2D-3D correspondences. With known or approximately known camera intrinsics, per-frame camera extrinsics can be re-estimated using robust Perspective-n-Point (PnP), refined by nonlinear reprojection minimisation and optionally temporally regularised bundle adjustment. Frames can therefore be classified as camera-correct, camera-correctable, or geometrically inconsistent. Correctable frames use refined cameras; bad frames can be rejected or regenerated. The anchors are subsequently masked or inpainted before Gaussian-splat training.

The research programme is deliberately staged. It first validates camera recovery on ordinary Blender renders where ground truth is exact. It then measures how well candidate video models preserve proxy motion, framing and anchors. It compares skeleton-only control with progressively richer proxy geometry, long 360-degree sequences with shorter overlapping arcs, intended Blender cameras with anchor-corrected cameras, and different numbers and arrangements of views. Finally, it evaluates downstream Gaussian splat quality on controlled synthetic subjects, real multiview human datasets where licensing permits, and Liforma-style AI-generated characters.

|  |
| --- |
| **Core research thesis**  A static Gaussian splat does not require the final character to match the Blender proxy. It requires calibrated cameras and a set of images that are explainable by one static 3D subject. Blender supplies the intended calibration; Super Anchors tell us whether the generative renderer obeyed it and, when camera drift is modest, allow us to estimate the camera that the generated frame actually corresponds to. |

# 1. Research objective and working paper framing

## 1.1 Primary objective

Develop and experimentally validate a reproducible pipeline that converts one full-body character image into a high-quality static Gaussian splat by generating reconstruction-grade multiview imagery under a prescribed Blender camera rig, while detecting and correcting generative camera drift with persistent 3D calibration anchors.

## 1.2 Working paper title

Proxy-Guided, Anchor-Calibrated Generative Multiview Reconstruction for Single-Image Human Gaussian Splatting

## 1.3 Working project name

PAC-GS (Proxy- and Anchor-Calibrated Gaussian Splatting). This is an internal working name, not a final publication title.

## 1.4 Research question

|  |
| --- |
| **Primary question**  Can a generative video model be turned into a calibrated synthetic multiview renderer by conditioning it on a known Blender proxy sequence, and can persistent 3D anchors recover enough of any residual camera drift to materially improve static Gaussian-splat reconstruction from a single character image? |

## 1.5 Intended contribution

* A camera-first alternative to generating an orbit video and recovering cameras afterward.
* A practical separation of camera error from generative shape/appearance error.
* A persistent calibration-anchor design that survives generation and enables per-frame pose validation/correction.
* A frame-quality gate that can correct, reject or selectively regenerate synthetic views before 3D optimisation.
* An experimental study of which control representation - skeleton, volumetric proxy, silhouette, depth, normals, or combinations - produces the most reconstruction-consistent human views.
* A reproducible evaluation protocol linking camera fidelity metrics to downstream Gaussian-splat novel-view quality.

# 2. What a static Gaussian splat actually needs

At the conceptual level, the reconstruction problem is simpler than the upstream generation problem. A static Gaussian-splat optimiser is trying to find a set of 3D Gaussians whose renders reproduce the training images from their calibrated cameras. The essential information is therefore a set of mutually consistent images of one static subject and the calibrated camera associated with every image.

| **Input** | **Fundamental?** | **Role** |
| --- | --- | --- |
| RGB images | Yes | Photometric observations the splat must reproduce. |
| Camera extrinsics | Yes | Per-view position and orientation; defines where every camera ray originates and points. |
| Camera intrinsics | Yes | Focal length/FOV, principal point, image dimensions and any relevant distortion model. |
| Static cross-view geometry | Yes | All views must be explainable by one underlying 3D subject. |
| Foreground mask | No, but valuable | Prevents background from dominating optimisation and simplifies subject-only reconstruction. |
| Depth / normals | No | Useful priors or diagnostics; not a fundamental requirement. |
| Sparse point cloud / mesh | No | Useful initialisation; the geometry can be optimised without matching a pre-existing mesh. |
| Skeleton / body proxy | No for GS | Used upstream to constrain view synthesis, not required by the splat itself. |

A crucial consequence follows: the generated person may differ from the rough Blender proxy. If the proxy has 45 cm shoulders but the generative model consistently renders a 50 cm-shouldered person from every view, Gaussian splatting can simply reconstruct the latter. The proxy is a conditioning scaffold, not a geometric ground truth that the final splat must copy.

|  |
| --- |
| **Acceptable vs unacceptable variation**  Acceptable: the generator chooses a different but fixed head shape, face depth, shoulder width, arm spacing, torso thickness, hair mass or clothing volume. Unacceptable: those properties change as a function of viewpoint, or the generated image behaves as if it came from a different camera than the one assigned to it. |

# 3. Motivation: why the earlier generative-orbit approach is fragile

The earlier approach generated an orbit video and then attempted to recover camera poses with Postshot/COLMAP-style structure-from-motion. Dense adjacent frames helped local matching, but the underlying assumption of a rigid scene was violated by the video generator. Face structure, silhouette, hair, clothing, limbs and background details could drift. As a result, local sections of the orbit could track while the global reconstruction fractured, for example into inconsistent front/back camera islands.

The problem is architectural: a generative orbit is not a real camera recording a static object. If we ask an SfM system to infer camera motion after the fact, it must explain generative non-rigidity as some combination of camera movement and geometry. That is fundamentally ambiguous.

|  |
| --- |
| OLD ORDER  single image -> generative orbit -> infer camera path -> train 3D  PROPOSED ORDER  single image -> define camera path + proxy first -> generate views to those controls  -> verify/correct cameras with anchors -> train 3D |

4DAnyone is an important conceptual reference because it also shifts the problem toward prescribed viewpoints and human-structure conditioning rather than relying on background feature tracking to recover camera poses. Our proposed study is narrower and more controllable: static subjects, a user-authored Blender proxy and camera rig, and explicit calibration markers that let us measure whether a generic video model actually followed the prescribed viewpoint.

# 4. Proposed method

## 4.1 Input and static-subject assumption

The primary input is one full-body reference image of the target character. The subject is assumed to remain in a fixed pose throughout the synthetic capture. The camera moves; the person should not. Static subjects are deliberately chosen because any genuine body motion introduces a second reconstruction problem that is unnecessary for the Liforma use case.

## 4.2 Rough Blender proxy

Create a proxy that approximates pose and coarse occupied volume. It does not need to be a production character model. In early experiments it can be authored manually in Blender so the research tests the generative/reconstruction hypothesis rather than automatic fitting quality. Later work can automate the proxy from monocular pose/SMPL-X-style estimates.

| **Proxy level** | **Contents** | **Purpose** |
| --- | --- | --- |
| P0 - skeleton | Joints/bones only | Tests the minimum pose cue. |
| P1 - skeleton + primitives | Torso, head, pelvis and limb volumes | Adds silhouette, occlusion and depth cues cheaply. |
| P2 - rough body | Coarse body mesh plus head/hair/garment volumes | Strong geometric guidance without requiring final anatomy. |
| P3 - multi-pass proxy | P2 plus depth, normals, silhouette and/or segmentation passes | Tests whether explicit geometric channels improve adherence. |

## 4.3 Camera rig designed first

The camera trajectory is authored in Blender before any AI generation. For every control frame we export the full calibrated camera model. At minimum this includes the camera-to-world transform (or equivalent R/t extrinsics), focal length or horizontal/vertical field of view, sensor/image dimensions, principal point convention, and any lens shift. Lens distortion should initially be zero to keep the synthetic camera model exact.

A dense control sequence is desirable even if the final Gaussian training set uses fewer frames. Small temporal steps give the video model an easier motion-following problem. After generation and quality control, frames can be selected at the density that best balances coverage and redundancy.

## 4.4 Candidate camera layouts

* Single horizontal ring for the first proof of concept.
* Two or three pitch rings for crown, shoulders, feet and under-chin coverage.
* Dense 121-frame or similar control orbit per ring, with 24/48/96+ selected reconstruction views after validation.
* Fixed focal length and fixed subject framing in the first experiments; dynamic lenses are explicitly out of scope until camera recovery is proven.
* Optional segmented arcs with overlap (for example four 90-degree clips) to reduce long-sequence identity and geometry drift.

## 4.5 Control rendering

Render a control video from the proxy. The first experiment should avoid aesthetic complexity and maximise geometric legibility. Candidate passes include a flat-shaded body proxy, skeleton overlay, silhouette, depth, normals and segmentation. Different video models accept different conditioning formats, so the software should keep the Blender render products model-agnostic and let each backend adapter build the required input.

## 4.6 Generative video stage

The target video model receives the character reference image and the rendered control sequence. Candidate backends include current commercial/API systems and open-source video-to-video systems such as Wan- and MiniMax-class models; Seedance/Seedream-class systems can be evaluated where the exact product supports the required image/video conditioning. The research method must not depend on a single vendor.

The generator is instructed to preserve the target identity and costume, keep the subject motionless, copy the viewpoint progression of the control video, avoid independent camera shake/pan/zoom/crop, and preserve the calibration anchors exactly enough for reliable detection. The goal is not to reproduce the proxy's precise anatomy; it is to produce one consistent realistic subject under the intended viewing sequence.

# 5. Super Anchors: turning camera fidelity into a measurable quantity

## 5.1 Core idea

Place a rigid set of visually distinctive calibration objects in the Blender scene with precisely known 3D world coordinates. These markers are rendered into the proxy/control video and are required to remain visible in the final generated video. Because their 3D positions are known and their 2D image coordinates can be detected, they form the correspondences needed to estimate the camera that the AI frame actually represents.

|  |
| --- |
| **Why this matters**  Without anchors, a generated frame that looks plausible at 45 degrees may actually behave like 48 degrees with a slight dolly, crop or focal-length change. The Blender camera label would then be wrong. With rigid world anchors, that discrepancy becomes measurable and often correctable. |

## 5.2 Two anchor classes

| **Anchor class** | **Placement** | **Use** |
| --- | --- | --- |
| World / rig anchors | Rigid calibration cage, floor ring, side towers or markers around the subject | Primary camera recovery. Independent of body-shape changes. |
| Body diagnostic anchors | Optional persistent markers attached to coarse body locations | Detect body-specific drift or non-rigid behaviour after camera correction; not trusted as the primary camera reference. |

World anchors are the primary design. Body anchors are secondary diagnostics because the generated body may legitimately differ from the proxy and may also deform. Camera estimation should depend on rigid scene geometry whenever possible.

## 5.3 Anchor layout requirements

* Non-coplanar 3D distribution: markers at multiple heights and depths, not a flat poster wall.
* Unique correspondence: each marker should be distinguishable by shape, colour pattern, position or a combination. Literal tiny QR/ArUco text may not survive diffusion reliably, so several designs should be tested.
* Redundancy: target at least 8-12 reliably visible correspondences in most frames, even though PnP can mathematically operate with fewer.
* Coverage across 360 degrees: place enough markers around the subject that occlusion of one side does not collapse the solver.
* Separation from the subject silhouette where practical so marker removal/inpainting is easy.
* Known dimensions and centre points so detection noise can be translated into reprojection error.

## 5.4 Camera recovery

For frame f with known 3D anchor positions Xi, detected pixels xi and intrinsics K, estimate the frame's camera pose by minimising robust reprojection error:

|  |
| --- |
| R\_f, t\_f = argmin Σ\_i ρ( || π(K, R\_f, t\_f, X\_i) - x\_i ||² )  where π is the perspective projection function and ρ is a robust loss. |

The first implementation should use OpenCV solvePnPRansac followed by iterative refinement. If the video model introduces small systematic zoom/crop changes, a later joint optimisation can allow a tightly regularised focal-length/principal-point correction. Intrinsic optimisation should be a separate experiment because unconstrained focal length can absorb errors that should instead cause a frame to be rejected.

## 5.5 Temporal refinement

Per-frame PnP is intentionally local and interpretable. A second-stage sequence optimisation can refine the complete path by jointly minimising anchor reprojection error while penalising implausible high-frequency pose jumps. This is effectively a small calibrated bundle-adjustment problem over known fixed 3D markers, not scene reconstruction.

## 5.6 Anchor removal

The calibration rig is a measurement instrument, not part of the desired final asset. After camera estimation, generate anchor masks from detections/projections and remove the anchors by masking or inpainting. The research must measure whether inpainting introduces view-dependent artefacts. A fallback is to exclude anchor-covered pixels from the Gaussian photometric loss if the trainer supports masks, avoiding hallucinated inpainted content altogether.

# 6. Error model: what anchors can and cannot fix

| **Failure type** | **Example** | **Anchor response** | **Downstream action** |
| --- | --- | --- | --- |
| Camera drift | AI renders frame as if camera yaw/position differs from Blender | World anchors expose reprojection mismatch and allow pose re-estimation | Use corrected camera if residual becomes low. |
| Crop / zoom drift | Generator reframes subject or changes effective focal length | Anchor pattern shows coherent scale/offset error | Refine small intrinsic delta or reject frame. |
| Consistent new body shape | Final subject has wider shoulders/deeper face than proxy, consistently | Not an error if all views agree | Allow GS to reconstruct generated geometry. |
| View-dependent morphing | Shoulders/face/clothes change with viewpoint | Camera can fit anchors but subject observations remain inconsistent | Reject/regenerate; anchors do not repair anatomy. |
| Anchor corruption | Markers disappear, merge or change identity | PnP confidence drops / correspondences fail | Reject frame or use alternate anchor design. |
| Independent subject motion | Head/arms/hair move despite static proxy | World camera can still be recovered, but scene is non-static | Reject/regenerate or mask moving region; do not treat as camera error. |

This decomposition is a central scientific benefit of the method. It prevents us from incorrectly blaming all Gaussian-splat artefacts on camera pose or, conversely, assuming that corrected cameras guarantee a consistent 3D subject.

# 7. End-to-end pipeline

|  |
| --- |
| 1. Reference character image  v 2. Fit/author static proxy in Blender  v 3. Build calibrated camera path + rigid Super Anchor rig  v 4. Render dense control sequence + masks/depth/normals as configured  v 5. Generate realistic video using reference identity + control sequence  v 6. Detect Super Anchors in every generated frame  v 7. Solve/refine actual camera pose (and optionally tiny intrinsic correction)  v 8. Frame QA: accept / camera-correct / reject / regenerate  v 9. Remove or loss-mask calibration rig  v 10. Export RGB + masks + corrected camera transforms  v 11. Train static Gaussian splat (no COLMAP required)  v 12. Evaluate held-out views, geometry consistency and artefacts |

The key architectural property is that camera estimation is no longer inferred from the person's generated texture/features. Camera calibration comes from known world geometry. The person is free to be visually complex; the rig remains the measurement reference.

# 8. Research questions and hypotheses

| **ID** | **Research question / hypothesis** |
| --- | --- |
| RQ1 | How accurately do current video-to-video models follow a prescribed proxy camera trajectory when the subject is static? |
| H1 | Proxy-guided generation will exhibit lower camera-path error than unconstrained text/image-prompted orbit generation. |
| RQ2 | Does a rough volumetric human proxy provide materially better multiview consistency than a skeleton-only control sequence? |
| H2 | Skeleton + coarse body volume will reduce silhouette/occlusion drift and improve downstream novel-view quality relative to skeleton-only control. |
| RQ3 | Can persistent world anchors survive generative transformation with sufficient fidelity for robust PnP? |
| H3 | A redundant non-coplanar anchor rig with simple distinctive geometry will recover useful camera poses in a high fraction of frames. |
| RQ4 | Does anchor-corrected camera calibration improve Gaussian-splat quality over using the intended Blender cameras unchanged? |
| H4 | When generative camera drift exists, corrected cameras will reduce reprojection residuals and improve held-out view metrics/artifact rates. |
| RQ5 | When should a frame be corrected versus rejected/regenerated? |
| H5 | A threshold on anchor count/reprojection residual plus subject-consistency checks will outperform using every generated frame. |
| RQ6 | Does sequence segmentation reduce long-range identity/geometry drift? |
| H6 | Short overlapping arcs will preserve identity and static geometry better than one long 360-degree clip for at least some generators. |
| RQ7 | How many views and elevation rings are useful once camera accuracy is controlled? |
| H7 | Consistency will matter more than raw view count; more inconsistent frames can make reconstruction worse. |
| RQ8 | How model-dependent is the method? |
| H8 | The approach will transfer across multiple video generators, but anchor survival and geometric consistency will vary strongly by model and conditioning mode. |

# 9. Experimental programme

Experiments are ordered so that each stage isolates one uncertainty. We should not begin expensive Gaussian training until the preceding measurement layer is trustworthy.

## Experiment 0 - calibration solver sanity check (no AI)

Purpose: prove the Super Anchor detector/PnP pipeline independently of generative models. Render the Blender anchor rig from known cameras, add controlled image noise, occlusions, marker detection error and synthetic camera perturbations, then recover the camera.

* Vary visible anchors: 4, 6, 8, 12, 16.
* Vary 2D localisation noise: 0, 0.5, 1, 2, 4 pixels.
* Vary occlusion and outlier fraction.
* Measure rotation error, translation error, reprojection RMSE and failure rate.
* Provisional gate: solver should be comfortably sub-degree on clean Blender observations and robust to realistic localisation noise before AI is introduced.

## Experiment 1 - does the generator follow the camera control?

Generate one 360-degree static-character sequence with the intended Blender cameras and anchors. Do not train a splat yet. Compare intended versus recovered camera trajectories.

* Plot yaw/pitch/roll error over time.
* Plot camera-centre displacement relative to orbit radius.
* Plot anchor reprojection RMSE and visible-anchor count.
* Flag systematic zoom/crop drift separately from extrinsic drift.
* Repeat across seeds to estimate variance.

## Experiment 2 - proxy representation ablation

Compare P0 skeleton, P1 skeleton + primitive volumes, P2 rough body, and P3 multi-pass control while holding camera path, source image, seed policy and generator settings as constant as the backend allows.

Primary outcomes: camera adherence, subject silhouette stability, identity consistency, anchor survival, frame acceptance rate and downstream splat quality.

## Experiment 3 - Super Anchor design study

Compare marker rigs rather than committing to the first visual design. Candidate variants include coloured spheres/cubes, asymmetric shape clusters, striped plates, semantic props, side towers and a floor/overhead cage. Tiny text codes are a useful control but may be destroyed by diffusion.

* Measure marker detection/ID accuracy by yaw and elevation.
* Measure PnP condition number / pose error versus layout.
* Measure how much the rig changes the generated character or steals visual attention.
* Measure inpainting/removal quality after calibration.

## Experiment 4 - intended cameras vs anchor-corrected cameras

This is the central downstream test. Train otherwise identical static Gaussian splats from the same generated frames using (A) the original Blender cameras and (B) the anchor-estimated/refined cameras. If correction is meaningful, B should improve held-out rendering and reduce floaters/thick edges/duplicated structures.

## Experiment 5 - correct vs reject vs regenerate

Build a frame triage policy from anchor residuals and subject-consistency diagnostics. Compare four datasets: all frames, corrected-only, corrected + bad-frame rejection, and corrected + selective regeneration. The experiment determines whether removing a small number of inconsistent views is more valuable than maximising view count.

## Experiment 6 - long orbit vs overlapping arcs

Compare one continuous 360-degree sequence with 2x180, 4x90 and possibly 8x45 degree overlapping clips. Each arc begins from an anchor/reference view and uses the same underlying Blender camera path. Evaluate identity drift, body-shape drift, seam consistency at overlap regions and final splat quality.

## Experiment 7 - view count and elevation

From a dense validated sequence, train with increasing selected-view counts such as 24, 48, 96 and 192 views. Separately compare one ring versus multiple pitch rings. Because the dense generator sequence is fixed, this isolates reconstruction density rather than generation quality.

## Experiment 8 - generator/backend comparison

Run the same calibrated Blender package through multiple model backends. The comparison is not primarily about photorealism; it is about reconstruction suitability: identity preservation, static geometry consistency, camera adherence, anchor survival and effective cost/latency. Where a backend has Turbo/Base or quality modes, compare them as a separate factor.

## Experiment 9 - intrinsic drift and focal refinement

Test whether current generators introduce a coherent effective zoom/crop even when the control camera is fixed. Compare fixed known intrinsics, per-frame weakly regularised focal correction, and sequence-global intrinsic refinement. Reject any approach that reduces anchor residual by absorbing genuine non-rigid errors into implausible focal changes.

## Experiment 10 - ground-truth reconstruction benchmark

To publish credible quantitative results, the method needs test subjects for which unseen true views exist. Use controlled synthetic Blender humans first: render a complete ground-truth multiview set from a real 3D character, reveal only one front image to the generative pipeline, then compare the resulting splat against held-out real renders. Where licensing permits, add a real multiview human dataset and treat one camera as the single source image.

This experiment separates 'looks good on an AI character' from measurable reconstruction accuracy. It is essential if the eventual paper makes quality claims.

## Experiment 11 - Liforma-style target characters

Evaluate the intended product domain: AI-created, front-facing full-body characters with detailed faces, hair and clothing. Ground-truth back views are not available, so use multiview consistency metrics, reconstruction residuals, controlled novel-view inspection and blinded expert preference. These results demonstrate applicability but should not replace ground-truth benchmarks.

# 10. Evaluation metrics

## 10.1 Camera fidelity

| **Metric** | **Definition / use** |
| --- | --- |
| Anchor reprojection RMSE | Pixel error after projecting known 3D anchors with estimated camera. |
| Rotation error | Degrees between estimated and intended/ground-truth camera orientation where ground truth exists. |
| Translation error | Distance between camera centres, normalised by orbit radius or scene scale. |
| Path smoothness | Frame-to-frame pose acceleration/jitter after accounting for intended motion. |
| Visible anchor count | Confidence/support for each pose estimate. |
| PnP inlier ratio | Fraction of detected correspondences accepted by robust pose solver. |
| Intrinsic delta | Recovered focal/principal-point deviation when intrinsic refinement is enabled. |

## 10.2 Generative multiview consistency

* Identity similarity across yaw/elevation using a face/identity embedding where appropriate.
* Silhouette stability after correcting for camera pose, focusing on unexplained temporal/viewpoint morphing rather than legitimate perspective change.
* Body keypoint stability and static-pose deviation.
* Hair/clothing topology events: appearance/disappearance of major structures.
* Frame-to-frame optical/feature consistency conditioned on the known camera motion.
* Percentage of frames accepted, corrected, rejected and regenerated by the QA gate.

## 10.3 Gaussian-splat quality

* Held-out-view PSNR, SSIM and LPIPS for ground-truth synthetic/real multiview benchmarks.
* Subject-mask IoU on held-out views.
* Depth/geometry error where ground-truth depth or mesh is available.
* Artifact counts/ratings: floaters, duplicate limbs/edges, thick silhouettes, transparent haze, face blur, back-of-head failure.
* Blinded human preference for Liforma-style characters where objective ground truth is absent.
* Training stability and convergence speed as secondary metrics.

## 10.4 Efficiency

* Generation time and compute cost per accepted view.
* Regeneration rate.
* Camera-solver runtime per frame.
* Anchor-removal/inpainting cost.
* Total end-to-end time to a usable splat.

# 11. Dataset and test-asset strategy

| **Tier** | **Purpose** | **Assets** |
| --- | --- | --- |
| A - synthetic ground truth | Exact quantitative calibration/reconstruction evaluation | Blender human models with known geometry, materials, lights and cameras. Only one source view exposed to the method; all other views held out. |
| B - real multiview benchmark | Check transfer to real human imagery | Public/licensable multiview human datasets; one camera chosen as source, remaining cameras reserved for evaluation. |
| C - product-domain characters | Test intended Liforma use case | AI-generated full-body characters with varied sex, skin tone, hair, clothing, silhouette complexity and accessories. |
| D - stress cases | Expose failure boundaries | Long hair, loose dresses, translucent fabric, shiny materials, hands away from body, crossed limbs, hats, asymmetrical accessories. |

The synthetic tier should be built first because it provides exact camera and geometry ground truth. The real multiview tier should be selected only after a licence and data-use review. The product-domain tier can begin immediately but should be treated as qualitative/consistency evidence rather than objective 3D ground truth.

# 12. Implementation architecture

The codebase should separate geometry/control generation, generative rendering, calibration, dataset assembly and Gaussian training. This prevents the project from becoming one model-specific notebook and makes ablations reproducible.

| **Module** | **Responsibilities** |
| --- | --- |
| blender\_scene/ | Load or author proxy; camera rigs; anchor rigs; deterministic render settings; export RGB/control passes, camera JSON and anchor JSON. |
| generators/ | Backend adapters for local/API video models; reference-image conditioning; control-video packaging; seeds/settings; response provenance. |
| anchors/ | Marker detection/ID, confidence scores, occlusion handling, 2D centres/corners and masks. |
| calibration/ | PnP/RANSAC, pose refinement, optional intrinsic refinement, temporal optimisation, diagnostics. |
| qa/ | Frame scoring, thresholds, accept/correct/reject/regenerate decisions; subject consistency diagnostics. |
| cleanup/ | Anchor masks, removal/inpainting, foreground masks, crop bookkeeping. |
| export/ | Nerfstudio transforms.json and/or standard calibrated dataset format; consistent coordinate-system conversion. |
| splat/ | Training launcher for Splatfacto/gsplat; fixed configs; export PLY; checkpoints. |
| eval/ | Camera metrics, novel-view metrics, geometry metrics, contact sheets, plots and experiment summaries. |
| experiments/ | Versioned YAML configs, run manifests, hashes, seeds and batch execution. |

# 13. Suggested data contracts

## 13.1 Camera record

|  |
| --- |
| {  "frame": 42,  "width": 1280, "height": 720,  "fx": ..., "fy": ..., "cx": ..., "cy": ...,  "camera\_to\_world": [[...],[...],[...],[...]],  "blender\_camera\_to\_world": [[...]],  "recovered\_camera\_to\_world": [[...]],  "pose\_source": "blender|pnp|bundle\_adjusted",  "reprojection\_rmse\_px": ...,  "pnp\_inliers": ...,  "accepted": true } |

## 13.2 Anchor record

|  |
| --- |
| {  "anchor\_id": "A07",  "world\_xyz": [x, y, z],  "design": "blue\_double\_sphere",  "frame": 42,  "detected\_uv": [u, v],  "confidence": 0.98,  "inlier": true } |

## 13.3 Run manifest

Every experiment should record source-image hash, Blender scene version/hash, proxy level, camera-rig version, anchor-rig version, generator/model/version, seed, prompt, video settings, calibration settings, frame-selection thresholds, splat trainer commit/config and evaluation code commit. This is necessary for eventual paper reproducibility and for debugging model-provider drift.

# 14. Blender control-scene specification (initial)

* Stationary subject at world origin with a known scale convention (for example metres).
* Camera looks at a stable target near sternum/head centre; no auto-framing after export.
* Lens, sensor fit and resolution fixed for the whole clip in initial studies.
* Simple matte proxy materials and controlled background to reduce irrelevant appearance cues.
* Rigid calibration rig parented to world, never to camera or character.
* Anchor positions exported directly from Blender in world coordinates.
* Render a machine-readable per-frame manifest alongside images/video.
* Coordinate conversion to the training framework unit-tested with a known cube/axis scene before human experiments.

## 14.1 Initial anchor-rig prototype

Start with approximately 16-24 world markers arranged as asymmetric side towers plus rear/floor markers at multiple heights and depths. The design should guarantee 8-12 visible markers through most of the 360-degree orbit without blocking the subject. Use visually simple marker identities such as distinct colour/shape combinations rather than relying solely on small printed codes.

The anchor rig itself is an experimental variable. If current video models distort symbolic markers, larger semantic 3D forms may survive better. If they alter colour but retain geometry, shape/relative arrangement can carry identity. The detection system should therefore support multiple marker families.

# 15. Gaussian-splat training protocol

Begin with Nerfstudio Splatfacto because the current workflow already supports calibrated camera transforms and static Gaussian training. Avoid COLMAP for the proposed method: camera calibration is an input, not something to recover from the generated person. Once the method works, reproduce key results with a second implementation such as gsplat/vanilla 3DGS to demonstrate that improvements are not trainer-specific.

* Use the same image resolution, foreground-mask policy and training budget across camera-calibration ablations.
* Use a common initialisation strategy for fair comparisons; proxy/visual-hull initialisation can be a later ablation rather than a hidden advantage.
* Keep background handling explicit. For subject-only splats, random/neutral training backgrounds plus masks may reduce overfitting to the calibration scene.
* Reserve held-out validated views that are never used for training so novel-view quality can be measured.
* Record per-image residuals. High-residual frames after camera correction are valuable evidence of remaining geometric inconsistency.

# 16. Experimental methodology and statistical discipline

* Use multiple characters, not a single showcase subject. Initial pilot: 3 characters; main study: target 10-20+ across varied appearance complexity if compute allows.
* Use multiple seeds for stochastic generators. At least 3 seeds per key condition in the pilot; increase for final results if variance is high.
* Predefine primary metrics for each experiment so we do not select whichever metric happens to improve.
* Report median and distribution, not only best-case examples.
* Separate development subjects from final benchmark subjects where practical.
* Do not tune QA thresholds on the same held-out views used to report final reconstruction metrics.
* Preserve failed generations and calibration failures as part of the result distribution.
* Use paired comparisons whenever the same source/generated sequence can be reconstructed with intended versus corrected cameras.

# 17. Provisional acceptance gates

These are engineering/research gates, not claims. They can be adjusted after the first calibration pilot if they are unrealistic, but changes should be documented before comparing downstream methods.

| **Gate** | **Provisional criterion** |
| --- | --- |
| Anchor solver on Blender renders | Near-zero bias; sub-degree median rotation error under realistic detection noise. |
| Anchor survival | Enough correctly identified markers for robust PnP in >=90% of frames on at least one tested rig/model combination. |
| Camera correction usefulness | Corrected cameras materially lower held-out reconstruction error than intended cameras on sequences with measurable drift. |
| Frame QA usefulness | Reject/regenerate policy improves reconstruction quality without discarding so many views that coverage collapses. |
| Proxy-control value | At least one richer proxy representation improves consistency over skeleton-only across multiple subjects/seeds. |
| Product viability | A representative AI character yields a clean 360-degree splat with acceptable face/hair/clothing quality and no major duplicate geometry. |

# 18. Key risks and mitigation

| **Risk** | **Why it matters** | **Mitigation / experiment** |
| --- | --- | --- |
| Generator ignores anchor rig | No reliable 2D-3D correspondences | Test multiple marker geometries/scales; increase redundancy; compare model backends. |
| Generator follows anchors but morphs person | Camera is correct but 3D subject is inconsistent | Subject-consistency gate; short arcs; rejection/regeneration; richer proxy. |
| Dynamic crop/zoom | Extrinsics alone cannot explain projection | Bookkeep deterministic resize/crop; optionally weak intrinsic refinement; reject large deviations. |
| Anchor removal damages frames | Inpainting creates new multiview inconsistency | Mask anchor pixels from loss, or compare inpaint vs loss-mask strategies. |
| Calibration rig changes generative output | Markers distract model or alter lighting/composition | Place rig away from silhouette; simplify scene; quantify with no-rig control. |
| Back view hallucination changes identity/clothes | Single image underdetermines unseen appearance | Treat plausible but consistent hallucination as acceptable for product; ground-truth benchmark measures factual reconstruction separately. |
| Long clips drift | Identity/shape degrades over time | Overlapping short arcs, reference resets, seed/model ablations. |
| Overfitting to one trainer/model | Method appears to work only in one stack | Use backend adapters and reproduce core result with second GS trainer/model. |
| No objective GT on product characters | Qualitative success can hide geometry errors | Synthetic and real multiview benchmark tiers are mandatory for paper claims. |

# 19. Milestones and implementation sequence

| **Milestone** | **Deliverable** | **Exit condition** |
| --- | --- | --- |
| M0 - research harness | Repository structure, config system, run manifests, deterministic Blender export | One command generates proxy frames + exact camera/anchor JSON. |
| M1 - calibration proof | Anchor detector + PnP + synthetic perturbation tests | Camera solver passes Experiment 0 gate. |
| M2 - first AI orbit | One model, one character, one proxy, one anchor rig | Full generated sequence with per-frame recovered camera diagnostics. |
| M3 - first splat A/B | Blender-camera vs corrected-camera splats | Measured downstream comparison completed. |
| M4 - proxy/anchor ablations | Experiments 2-3 | Best control/anchor design identified on development set. |
| M5 - robust sequence pipeline | Arc segmentation + QA/reject/regenerate + cleanup | Repeatable end-to-end runs across several characters. |
| M6 - quantitative benchmark | Synthetic + real multiview where possible | Held-out metrics with multiple subjects/seeds. |
| M7 - paper-ready study | Full ablations, figures, failure cases, reproducible configs | Evidence supports or falsifies core claims; manuscript can be written honestly. |

# 20. Immediate coding plan: first research sprint

The first implementation sprint should deliberately avoid integrating every possible model. We want to know whether the measurement concept works before building a general product pipeline.

**1.** Create a research repository/folder with versioned experiment configs and a results directory; pin Blender/Python/package versions.

**2.** Write a Blender Python script that creates/loads a static proxy, generates a camera orbit, instantiates the first asymmetric 3D anchor rig, renders control frames, and exports per-frame intrinsics/extrinsics plus anchor world coordinates.

**3.** Write a synthetic anchor detector for the first marker family and validate it on non-AI Blender renders.

**4.** Implement OpenCV solvePnPRansac + iterative pose refinement; produce plots of intended vs recovered camera and reprojection error.

**5.** Add a perturbation test harness that injects known camera and detection errors and confirms recovery accuracy.

**6.** Choose one video backend that can consume the reference image plus proxy/control video. Generate a single dense 360 sequence while asking it to keep anchors and subject static.

**7.** Run anchor detection and camera recovery on the generated sequence. Produce a per-frame QA report/contact sheet before any Gaussian training.

**8.** Export two calibrated datasets from exactly the same generated RGB frames: one with Blender cameras, one with recovered cameras.

**9.** Train two otherwise identical Splatfacto runs and compare held-out views/artifacts. This is the first decisive experiment.

**10.** Only if the result is promising, add proxy levels, anchor variants, shorter arcs, multiple models, automatic regeneration and full benchmark automation.

|  |
| --- |
| **First decisive test**  Generate one controlled sequence, recover its cameras from Super Anchors, then train two splats from the same pixels: intended Blender cameras vs recovered cameras. If recovered cameras do not improve a sequence that clearly exhibits measured camera drift, we need to understand why before scaling the project. |

# 21. Minimal software deliverables for Experiment 0-4

|  |
| --- |
| research/pacgs/  blender/  build\_scene.py  camera\_rig.py  anchor\_rig.py  render\_controls.py  generators/  base.py  <backend>.py  anchors/  detect.py  identities.py  calibration/  pnp.py  refine.py  trajectory.py  qa/  score\_frames.py  report.py  export/  nerfstudio.py  splat/  train.py  eval/  camera\_metrics.py  reconstruction\_metrics.py  make\_report.py  configs/  experiments/  tests/ |

The exact repository location can be chosen during implementation. The important point is the modular boundary: camera/anchor calibration must be testable with no generative model, and Gaussian training must accept a generic calibrated dataset rather than know how the images were generated.

# 22. Paper structure if the research succeeds

**1.** Introduction - single-image human 3D is underdetermined; generative models can hallucinate unseen views but are not calibrated renderers.

**2.** Related work - Gaussian splatting, single-image human reconstruction, generative novel-view synthesis, camera-conditioned diffusion/video, human multiview generation, calibration/PnP.

**3.** Problem formulation - static subject, one reference image, prescribed control cameras, generated observations with possible camera and geometry drift.

**4.** Method - Blender proxy guidance, Super Anchor rig, pose recovery/refinement, frame QA, calibrated dataset export and static GS training.

**5.** Experimental setup - datasets, generator backends, proxy/anchor designs, metrics and training protocol.

**6.** Results - camera fidelity, calibration correction, reconstruction metrics, ablations and cost.

**7.** Failure analysis - view-dependent anatomy changes, anchor destruction, intrinsic drift, hair/clothing/topology failures.

**8.** Discussion - what constitutes a valid synthetic multiview observation; limits of single-image unseen-side hallucination; product vs factual reconstruction.

**9.** Conclusion - evidence for or against treating guided video generation as a measurable calibrated synthetic capture process.

# 23. What would count as a publishable result?

A publication does not require every generative model to work. A strong result could be positive or partially negative, provided the experiments isolate the cause. Examples of potentially publishable findings include:

* Persistent world anchors reveal that plausible-looking video-to-video orbit sequences exhibit measurable, systematic camera drift that materially harms 3DGS, and PnP correction recovers a significant portion of the lost quality.
* A coarse volumetric proxy is sufficient to turn a generic video model into a usable calibrated multiview generator for static humans, while skeleton-only conditioning is not.
* Camera correction alone reaches a ceiling because the dominant failure is view-dependent shape morphing; the study quantifies that boundary and identifies control representations that reduce it.
* Short overlapping generative arcs plus anchor calibration outperform long 360 clips, showing a practical way to trade temporal continuity for multiview rigidity.
* A simple frame-rejection/regeneration policy based on calibration residuals provides a large reconstruction-quality gain at modest extra generation cost.

The eventual manuscript should report whichever conclusion the data supports. The research plan is designed so that even a failure of the full pipeline yields useful measurements about where current generative video ceases to behave like a calibrated renderer.

# 24. Summary of the idea developed today

The discussion converged on a camera-first view of the problem. A static Gaussian splat fundamentally needs calibrated cameras and consistent observations of one static subject. The exact Blender proxy geometry is not sacred: the generator may invent a different but fixed body, face, hair and clothing volume. The fatal errors are effective camera drift and view-dependent geometry changes.

The proposed solution is therefore to define the camera trajectory in Blender, render a skeleton or preferably a rough volumetric human proxy under those exact cameras, use the first/reference character image to transform that control sequence into a realistic video, and preserve a rigid set of Super Anchors in the generated output. The known 3D anchor coordinates and detected 2D positions make camera drift observable. Frames whose drift is coherent can have their cameras re-estimated; frames whose subject geometry is inconsistent can be rejected or regenerated. Once the calibration rig is removed or masked, the final calibrated images can train a static Gaussian splat directly without COLMAP.

The project now needs code and evidence. The immediate priority is not to add more conceptual complexity but to build the smallest rigorous harness that can answer the first two questions: (1) does a current video model preserve enough of the proxy camera movement and anchors to make calibration recoverable, and (2) does using the recovered cameras actually improve the resulting Gaussian splat?

# Appendix A - Initial experiment matrix

| **Factor** | **Pilot levels** | **Later levels** |
| --- | --- | --- |
| Proxy | Skeleton; primitive body | Rough mesh; depth/normals/multi-pass |
| Sequence | 1x360 | 2x180; 4x90; 8x45 with overlap |
| Views used for GS | 48 | 24; 96; 192 |
| Elevation | 1 ring | 2-3 rings |
| Camera labels | Blender; PnP-corrected | Temporal BA; weak intrinsic refinement |
| Anchors | One 16-24 marker rig | Shape/colour/layout variants; body diagnostics |
| Frame policy | All; correct-only | Reject; selective regenerate |
| Generator | One chosen backend | Wan / MiniMax / commercial/API alternatives |
| Subjects | 1-3 pilot | 10-20+ main study, plus ground-truth benchmarks |
| GS trainer | Splatfacto | Second gsplat/vanilla implementation for replication |

# Appendix B - Decision logic for each generated frame

|  |
| --- |
| detect world anchors  v < minimum correspondences? ── yes ──> REJECT / REGENERATE  │ no  v robust PnP + refinement  v reprojection residual too high? ── yes ──> REJECT / REGENERATE  │ no  v pose differs from Blender beyond small threshold?  ├─ yes -> CAMERA-CORRECTABLE: use recovered pose  └─ no -> CAMERA-GOOD: keep Blender or recovered pose consistently  v subject consistency check  v non-rigid / identity failure? ── yes ──> REJECT / REGENERATE  │ no  v ACCEPT -> remove/mask anchors -> export calibrated training view |

# Appendix C - Open questions to resolve empirically

* What anchor geometry survives each generator best?
* Should the calibration rig be behind the person, around the person, or split into foreground/background structures?
* Can we preserve markers strongly enough without causing the generator to simplify the human subject?
* Does a rough body proxy materially improve view-dependent geometry consistency over a skeleton?
* How much camera drift is typical by yaw angle and sequence duration?
* Can intrinsics be treated as fixed, or do some models impose systematic reframing that requires correction?
* Does masking anchors from the GS loss outperform generative inpainting for reconstruction consistency?
* How aggressively should high-residual frames be rejected?
* Does restarting the reference every 90 degrees create identity seams, and can overlap resolve them?
* At what point does adding more views stop helping because the generator's multiview inconsistency dominates?
* Which failure mode dominates high-quality splat reconstruction: camera drift, body-shape morphing, texture/identity drift, or unseen-side hallucination?

**End of v0.1 research plan**

Next action: implement Experiment 0-4 before expanding the scope.

### Document screenshots

![img-001](proxy_anchor_calibrated_gaussian_splat_research_plan_media/img-001.png)

## Image Index

Use the image `ID` with the matching `File` when linking nearby text to a figure.

| ID | File | Origin |
|----|------|--------|
| `img-001` | `proxy_anchor_calibrated_gaussian_splat_research_plan_media/img-001.png` | screenshot |

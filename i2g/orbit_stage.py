"""Stage "orbit": the calibrated turntable-video pass (the research plan's PAC-GS, see pacgs_explained.md).

The splat trained from the Qwen views is good enough as a *stand-in*: its shape carries hair, clothes
and shoes. So we render it as a depth video along camera paths we choose ourselves, and let Wan 2.2
Fun-Control repaint that depth video into the real person, starting from the photo itself. The frames
follow the depth pixel for pixel, so every frame's camera is known exactly instead of being estimated
from keypoints, and a video model keeps one identity across the whole turn instead of a dozen
separately drawn views disagreeing. The splat is then retrained on those frames plus the photo.

Camera paths (pacgs.orbit.RIGS, `--orbit_rigs`):
  360        one full circle at the photo's height: the coverage the Qwen views used to provide
  face       the same circle zoomed in on the head, so the face is not 40 px wide
  face_up    the head close-up rising to 30 degrees: the face seen from above
  face_down  the head close-up dipping to 20 degrees below: the face seen from underneath
  face_high  steeper still (55 degrees), where the shoulders and the arms start to hide the face
  face_top   over the crown (95 degrees), looking down on the top of the head
  helix      the body orbit rising to 30 degrees, so the top of the head and shoulders are seen

A single head close-up ring leaves the face with no coverage from other elevations: identity measured
0.63 at eye level and 0.30 from 30 degrees above, until face_up and face_down were added. The photo
already looks up at the head by about 19 degrees, so a rig reaches (elev - 19) degrees relative to the
face: face_up lands at +11, face_high at +35, face_top at +73.

Every rig's control comes from the splat, never from the body mesh: a stand-in without hair or clothes
tells the video model to remove them (it drew a bun and a strapless top over the crown), and one such
clip drags every other view with it. `pacgs.dataset.prepare` flags that case as `restyled` and this
stage drops the clip.

The heavy lifting lives in the `pacgs` package (the research code for this method); this module is the
product pipeline's wrapper around it.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from PIL import Image

from .errors import PipelineError
from .log import NullBar, get_logger

log = get_logger("orbit")


def _controls(paths, world: dict, proxy_ply: Path, rigs: list[str], width: int, height: int, frames: int,
              subject: Path, bar) -> dict[str, list[dict]]:
    """Render the depth control video and the start image for every rig; returns the cameras per rig."""
    from pacgs.orbit import layout_cameras
    from pacgs.proxy import SplatProxy, encode_depth, inverse_depth_range, photo_source, reframe_photo
    stage = paths.stage_dir("orbit")
    proxy = SplatProxy(proxy_ply)
    cams_by_rig = {}
    try:
        for rig in rigs:
            cams = layout_cameras(world, rig, frames, width, height)
            cams_by_rig[rig] = cams
            rng = inverse_depth_range(proxy, cams[::5], width, height)
            for cam in cams:
                out = stage / "controls" / rig / f"arc{cam['arc']}"
                out.mkdir(parents=True, exist_ok=True)
                depth, alpha = proxy.depth(cam["w2c"], cam["K"], width, height)
                Image.fromarray(encode_depth(depth, alpha, rng)).save(out / f"frame_{cam['index']:04d}.png")
            source, source_K = photo_source(subject, world, cams[0]["zoom"])
            (stage / "start").mkdir(parents=True, exist_ok=True)
            reframe_photo(source, source_K, cams[0]["K"], width, height).save(stage / "start" / f"{rig}.png")
            log.debug("Orbit controls for rig %s: %d frames", rig, len(cams))
            bar.update(1)
    finally:
        del proxy
        torch.cuda.empty_cache()
    return cams_by_rig


def splat_config(opts, rigs: list[str]):
    """The training settings for the orbit pass: the run's settings, plus what the head close-ups change.

    With close-ups in the training set the face priors are not merely unnecessary, they are what made
    the face look bad away from the front. Measured over ten cameras around the head against the
    close-up frames themselves (`pacgs facecheck`, see next-stage-research/docs/pacgs_face_fix.md):

      baked photo layer, priors on   agreement 25.4 dB, speckle 1.09  ("detail" that is mostly wrong)
      layer off, priors off          agreement 30.7 dB, speckle 0.88, but soft (detail 0.60x)
      + close-ups at 3x the steps    agreement 31.7 dB, speckle 0.85
      + no zoom-out blur on video    agreement 32.2 dB, speckle 0.83, detail 0.67x   <- shipped

    The baked layer is one flat disc per photo pixel, pinned to the body mesh's face: it reproduces the
    photo head-on and breaks into speckle at an angle. The zoom-out blur (`multiscale`) is anti-aliasing
    for the native-resolution photo; applied to 480p video frames it only blurs them, so it is now per
    frame kind. Anything the operator set explicitly (--set) wins over all of this."""
    from .pipeline import make_splat_config
    cfg = make_splat_config(opts)
    if "face" in rigs:
        for key, value in (("face_weight", 1.0), ("face_bake", False), ("face_shell", False),
                           ("face_depth_weight", 0.0), ("multiscale_video", 0.1)):
            if key not in opts.splat:
                setattr(cfg, key, value)
    if opts.orbit_iterations:            # 0 when the operator asked for a specific iteration count
        cfg.iterations = opts.orbit_iterations
    return cfg


def run_orbit(paths, opts, client, make_bar=None) -> dict:
    from pacgs.dataset import merge, prepare
    from pacgs.generate import GenSettings, generate_arcs
    from pacgs.orbit import ARCS, RIGS, rig_tag
    from pacgs.proxy import make_proxy

    from .pipeline import (_aligned_world, _archive_outputs, _free_comfy_for_training, _train_and_export,
                           make_splat_config, write_stamp)

    if not opts.orbit:
        log.info("Orbit video pass is off (the standard and high presets and --orbit turn it on)")
        write_stamp(paths, opts, "orbit", {"skipped": True})
        return {"skipped": True}

    proxy_ply = paths.output / "character.ply"
    if not proxy_ply.exists():
        raise PipelineError("the orbit pass needs the splat from the earlier stages",
                            hint="Run without --only_stage orbit, or rerun with --from_stage train.")
    rigs = [r.strip() for r in opts.orbit_rigs.split(",") if r.strip()]
    unknown = [r for r in rigs if r not in RIGS]
    if unknown:
        raise PipelineError(f"unknown orbit camera rig(s): {unknown}",
                            hint=f"Known rigs: {', '.join(RIGS)}")
    width, height = opts.orbit_size
    cfg = splat_config(opts, rigs)
    world = _aligned_world(paths, cfg)
    stage = paths.stage_dir("orbit")
    t0 = time.perf_counter()

    bar = make_bar(total=len(rigs), desc="Rendering control videos", unit="rig") if make_bar else NullBar()
    try:
        cams_by_rig = _controls(paths, world, proxy_ply, rigs, width, height, opts.orbit_frames,
                                paths.work, bar)
    finally:
        bar.close()

    settings = GenSettings.preset("lightning", opts.orbit_prompt or _describe(paths), width=width, height=height,
                                  length=opts.orbit_frames, seed=opts.orbit_seed)
    clips = sum(len(ARCS[RIGS[r]["arcs"]]) for r in rigs)
    bar = make_bar(total=clips, desc="Generating orbit video (Wan)", unit="clip") if make_bar else NullBar()
    try:
        for rig in rigs:
            controls = [stage / "controls" / rig / f"arc{k}" for k in range(len(ARCS[RIGS[rig]["arcs"]]))]
            start = stage / "start" / f"{rig}.png"
            generate_arcs(client, settings, controls, start, start, stage / "gen" / rig)
            bar.update(len(controls))
    finally:
        bar.close()
    _free_comfy_for_training(client)

    splat_proxy = make_proxy("splat", world, proxy_ply)
    datasets = []
    bar = make_bar(total=sum(len(c) for c in cams_by_rig.values()), desc="Cutting out the person",
                   unit="frame") if make_bar else NullBar()
    try:
        for rig in rigs:
            data = stage / "data" / rig
            qa = prepare(stage / "gen" / rig, cams_by_rig[rig], data, paths.work, world,
                         splat_proxy, splat_proxy, holdout_every=opts.orbit_holdout_every, tag=rig_tag(rig))
            datasets.append((rig, data, qa))
            bar.update(len(cams_by_rig[rig]))
    finally:
        bar.close()
        del splat_proxy
        torch.cuda.empty_cache()
    # A clip whose matte disagrees with the splat's own silhouette has re-styled the person (different
    # hair, different clothes). One such clip drags every other view towards it, so drop it - unless they
    # all failed, which means something more basic is wrong and the numbers should be seen in the report.
    kept = [(r, d) for r, d, qa in datasets if not qa["restyled"]]
    for rig, _, qa in datasets:
        if qa["restyled"]:
            log.warning("Dropping the %s clip: matte-vs-subject IoU %.3f, so the video re-styled the "
                        "person (hair or clothing)", rig, qa["iou_splat_median"])
    if not kept:
        log.warning("Every clip disagrees with the subject's silhouette; training on them anyway")
        kept = [(r, d) for r, d, _ in datasets]
    rigs = [r for r, _ in kept]
    datasets = [d for _, d in kept]
    # the close-ups are where the face detail is; without this each would be one clip among many
    boost = {r: opts.orbit_face_boost for r in rigs if r.startswith("face")}
    summary = merge(datasets, stage / "data" / "all", boost) if len(datasets) > 1 else None
    data_dir = stage / "data" / ("all" if summary else rigs[0])
    records = json.loads((data_dir / "records.json").read_text())
    holdout = json.loads((data_dir / "qa.json").read_text())["summary"]["holdout"]
    cfg.exclude = ",".join(s for s in (cfg.exclude, ",".join(holdout)) if s)
    log.info("Orbit video: %d frames from %s, %d held out (%.1f min so far)", len(records) - 1,
             " + ".join(rigs), len(holdout), (time.perf_counter() - t0) / 60)

    _archive_outputs(paths, copy=True)
    result = _train_and_export(paths, opts, cfg, world, records, "orbit", make_bar)
    write_stamp(paths, opts, "orbit", {"gaussians": result["gaussians"], "psnr": result.pop("psnr"),
                                       "rigs": rigs, "frames": len(records) - 1, "holdout": holdout})
    return result


def _describe(paths) -> str:
    """The prompt for the video model. A plain description works: the depth video carries the shape and
    the photo carries the identity, so the words only have to set the scene."""
    return ("A person standing completely still in a photo studio, seen in a slow orbit around them. "
            "The camera moves at a constant distance and height while the person stays perfectly still. "
            "Plain light grey studio background, soft even lighting, photorealistic, sharp focus, high detail.")

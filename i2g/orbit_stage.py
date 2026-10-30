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

The head fix (next-stage-research/docs/headpass_results.md, on by default):
  coherent_head  the depth videos are rendered from a stand-in whose head is ONE continuous surface
                 (headpass.headsurf, v4): the second pass's splat carries the face as a plate baked onto the
                 body model's face, glued onto a skull with a groove across the top of the forehead, and Wan
                 painted both - a mask edge around the face and a dip in the side profile. Measured on two
                 subjects: forehead kink 11.0 -> 4.7 mm, the head holds together to 120 deg in true side views,
                 the face no longer sinks into long hair.
  head_pass      then a separate head splat, trained on the head close-ups and the photo only (own 30k steps,
                 a perceptual weight that ramps up late), merged onto the body at the neck with a dithered
                 5 cm band. A small extra gain on top of coherent_head (identity +0.01-0.04, a little sharper);
                 ~20 min.

The heavy lifting lives in the `pacgs` and `headpass` packages (the research code for these methods); this
module is the product pipeline's wrapper around them.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
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


def _coherent_standin(paths, world: dict, proxy_ply: Path):
    """(stand-in PLY, head maps) with one continuous head; (proxy_ply, None) if no face is found in the photo."""
    from headpass.headsurf import build_standin
    from headpass.metrics import Reference
    stage = paths.stage_dir("orbit")
    out = stage / "standin_coherent.ply"
    try:
        ref = Reference(paths.work, world)
    except RuntimeError as exc:
        log.warning("Head fix skipped (%s): the depth video uses the second pass's splat as it is", exc)
        return proxy_ply, None
    stats = build_standin(proxy_ply, world, ref, out, skin_smooth_bins=16.0, close_bins=10,
                          save_maps=stage / "standin_coherent_maps.npz")
    (stage / "standin_coherent.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    log.info("Head fix: the depth videos use one continuous head (face, forehead and skull as one surface)")
    return out, dict(np.load(stage / "standin_coherent_maps.npz"))


def _head_pass(paths, opts, world: dict, rigs: list[str], standin_ply: Path, maps: dict, make_bar=None) -> dict:
    """Train a head splat on the head close-ups + photo, merge it onto the deliverable at the neck, and refresh
    the previews. The body-only and head-only splats are kept in the stage folder."""
    import shutil

    from headpass.headtrain import head_records, train_head_core
    from headpass.merge import merge_params

    from .evaluate import load_ply
    from .pipeline import export_previews
    from .splat import export_ply
    stage = paths.stage_dir("orbit")
    head_rigs = [r for r in rigs if r.startswith("face")]
    records = head_records(stage / "data", head_rigs)
    if len(records) < 20:
        log.warning("Head pass skipped: only %d head close-up frames", len(records) - 1)
        return {"skipped": True}
    cfg = splat_config(opts, rigs)
    cfg.iterations = opts.head_iterations
    cfg.perceptual_ramp_to = 0.6
    cfg.exclude = ""
    hr = paths.stage_dir("ingest") / "source_hr_rgba.png" if opts.photo_hr else None
    bar = make_bar(total=cfg.iterations, desc="Training the head", unit="it") if make_bar else NullBar()
    try:
        head, report = train_head_core(records, world, cfg, maps, standin_ply, hr, 0.0, stage / "head", bar)
    finally:
        bar.close()
    body_ply = paths.deliverable("character.ply")
    shutil.copy2(body_ply, stage / "character_body.ply")
    export_ply(head, stage / "character_head.ply")
    merged, stats = merge_params(load_ply(body_ply), head, maps)
    export_ply(merged, body_ply)
    export_previews(paths, merged, world)
    _photo_vs_render(paths, world, merged)
    report["merge"] = stats
    rj = paths.deliverable("report.json")
    if rj.exists():
        full = json.loads(rj.read_text())
        full["head_pass"] = report
        rj.write_text(json.dumps(full, indent=1, default=str), encoding="utf-8")
    log.info("Head pass merged at the neck: %d head + %d body Gaussians", stats["head_kept"], stats["body_kept"])
    return {"head_gaussians": stats["head_kept"], "head_minutes": report["train_minutes"]}


def _photo_vs_render(paths, world: dict, params) -> None:
    """photo_vs_render.png for the merged splat (the photo and the render side by side at the photo's camera)."""
    import cv2

    from pacgs.proxy import photo_source

    from .splat import render_rgb
    src, K = photo_source(paths.work, world, 2.0)
    rgba = np.asarray(Image.open(src).convert("RGBA")).astype(np.float32)
    a = rgba[..., 3:] / 255.0
    gt = (rgba[..., :3] * a + 127.5 * (1 - a)).round().astype(np.uint8)
    h, w = gt.shape[:2]
    with torch.no_grad():
        pred = render_rgb(params, torch.as_tensor(world["photo_w2c"], device="cuda").float(),
                          torch.as_tensor(K, device="cuda").float(), w, h)
    pair = np.concatenate([gt, pred], 1)
    pair = cv2.resize(pair, (pair.shape[1] * 1248 // h, 1248), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(paths.deliverable("photo_vs_render.png")), cv2.cvtColor(pair, cv2.COLOR_RGB2BGR))


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

    proxy_ply = paths.find_deliverable("character.ply")
    if proxy_ply is None:
        raise PipelineError("the orbit pass needs the splat from the earlier stages",
                            hint="Run without --only_stage orbit, or rerun with --from_stage train.")
    rigs = [r.strip() for r in opts.orbit_rigs.split(",") if r.strip()]
    unknown = [r for r in rigs if r not in RIGS]
    if unknown:
        raise PipelineError(f"unknown orbit camera rig(s): {unknown}",
                            hint=f"Known rigs: {', '.join(RIGS)}")
    width, height = opts.orbit_size
    cfg = splat_config(opts, rigs)
    fitted = paths.stage_dir("refine") / "registration_fitted.json"
    world = _aligned_world(paths, cfg, json.loads(fitted.read_text()) if opts.refine and fitted.exists() else None)
    stage = paths.stage_dir("orbit")
    t0 = time.perf_counter()

    control_ply, head_maps = proxy_ply, None
    if opts.coherent_head:
        control_ply, head_maps = _coherent_standin(paths, world, proxy_ply)
    bar = make_bar(total=len(rigs), desc="Rendering control videos", unit="rig") if make_bar else NullBar()
    try:
        cams_by_rig = _controls(paths, world, control_ply, rigs, width, height, opts.orbit_frames,
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

    # QA: frames against the control they followed, and against the subject's own silhouette (the second pass)
    control_proxy = make_proxy("splat", world, control_ply)
    splat_proxy = make_proxy("splat", world, proxy_ply) if control_ply != proxy_ply else control_proxy
    datasets = []
    bar = make_bar(total=sum(len(c) for c in cams_by_rig.values()), desc="Cutting out the person",
                   unit="frame") if make_bar else NullBar()
    try:
        for rig in rigs:
            data = stage / "data" / rig
            qa = prepare(stage / "gen" / rig, cams_by_rig[rig], data, paths.work, world,
                         control_proxy, splat_proxy, holdout_every=opts.orbit_holdout_every, tag=rig_tag(rig))
            datasets.append((rig, data, qa))
            bar.update(len(cams_by_rig[rig]))
    finally:
        bar.close()
        del splat_proxy, control_proxy
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
    if opts.head_pass and head_maps is not None and any(r.startswith("face") for r in rigs):
        result |= _head_pass(paths, opts, world, rigs, control_ply, head_maps, make_bar)
    write_stamp(paths, opts, "orbit", {"gaussians": result["gaussians"], "psnr": result.pop("psnr"),
                                       "rigs": rigs, "frames": len(records) - 1, "holdout": holdout,
                                       "coherent_head": head_maps is not None,
                                       "head_pass": bool(result.get("head_gaussians"))})
    return result


def _describe(paths) -> str:
    """The prompt for the video model. A plain description works: the depth video carries the shape and
    the photo carries the identity, so the words only have to set the scene."""
    return ("A person standing completely still in a photo studio, seen in a slow orbit around them. "
            "The camera moves at a constant distance and height while the person stays perfectly still. "
            "Plain light grey studio background, soft even lighting, photorealistic, sharp focus, high detail.")

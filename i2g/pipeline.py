"""The v2 pipeline as resumable stages: photo -> cut-out -> 3D body -> views -> cameras -> splat.

Each finished stage writes `<stage dir>/stage.json` recording what it was built from (the photo's
hash, the upstream stages' stamps) and the settings it used. A rerun skips a stage whose stamp still
matches, so iterating on training settings only retrains. Stale data is moved to `_archive/`,
never deleted.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .comfy_client import ComfyClient, WatchdogTripped
from .errors import PipelineError
from .log import NullBar, get_logger
from .paths import (MODELS_DIR, OUTPUT_KINDS, REPO_ROOT, deliverable_path, deliverable_stem,
                    find_deliverable, safe_run_name)

log = get_logger("pipeline")

STAGES = ["ingest", "body", "views", "register", "train", "refine", "orbit"]
STAGE_TITLES = {"ingest": "Cut out the person", "body": "Fit a 3D body", "views": "Generate views (Qwen)",
                "register": "Measure view cameras (new candidates for any missing side)",
                "train": "Train the Gaussian splat",
                "refine": "Second pass: refit cameras, retrain",
                "orbit": "Orbit video pass: turntable video of a rebuilt head (Wan), retrain, head pass"}
STAGE_DIRS = {"ingest": "00_ingest", "body": "01_body", "views": "02_views", "register": "03_qa",
              "train": "06_train", "refine": "07_refine", "orbit": "08_orbit"}
DEPENDS = {"ingest": [], "body": ["ingest"], "views": ["ingest"], "register": ["body", "views"],
           "train": ["register"], "refine": ["train"], "orbit": ["refine"]}


@dataclass
class RunPaths:
    work: Path      # intermediate files for every stage
    output: Path    # the deliverables (<photo>-and-<experiment>-character.ply, ...)
    name: str       # photo stem (or --name)
    experiment: str = ""  # --experiment; also in the folder path and file names

    def stage_dir(self, stage: str) -> Path:
        return self.work / STAGE_DIRS[stage]

    @property
    def logs(self) -> Path:
        return self.work / "logs"

    def deliverable(self, kind: str) -> Path:
        """Where this run writes a final file: alice-and-exp3-character.ply, ..."""
        return deliverable_path(self.output, self.name, kind, self.experiment)

    def find_deliverable(self, kind: str) -> Path | None:
        """Existing final file: the current name, or older names from earlier runs."""
        return find_deliverable(self.output, self.name, kind, self.experiment)

    def deliverables_on_disk(self, *, include_contact_sheet: bool = True) -> list[Path]:
        found, seen = [], set()
        photo = safe_run_name(self.name)
        stem = deliverable_stem(self.name, self.experiment)
        for kind in OUTPUT_KINDS:
            if not include_contact_sheet and kind == "contact_sheet.png":
                continue
            for candidate in (self.output / f"{stem}-{kind}", self.output / f"{photo}-{kind}",
                              self.output / kind):
                if candidate.exists():
                    key = candidate.resolve()
                    if key not in seen:
                        found.append(candidate)
                        seen.add(key)
        return found

    def clear_legacy(self, kind: str) -> None:
        """Remove older unprefixed / photo-only files once the current name has been written."""
        dest = self.deliverable(kind)
        leftovers = [self.output / kind, self.output / f"{safe_run_name(self.name)}-{kind}"]
        for legacy in leftovers:
            if dest.exists() and legacy.exists() and dest.resolve() != legacy.resolve():
                legacy.unlink()

    def create(self) -> None:
        for d in [self.work, self.output, self.logs] + [self.stage_dir(s) for s in STAGES]:
            d.mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> dict[str, Path]:
        """The folder map the stage modules (register.py) expect."""
        return {"root": self.work, "ingest": self.stage_dir("ingest"), "body": self.stage_dir("body"),
                "views": self.stage_dir("views"), "qa": self.stage_dir("register"),
                "train": self.stage_dir("train"), "final": self.output}


@dataclass
class RunOptions:
    name: str
    input_image: Path
    vfov_deg: float = 40.0
    views: str = "standard"          # i2g.views.VIEW_PRESETS
    seeds: int = 2
    photo_hr: bool = True            # train the photo view at its native resolution
    splat: dict = field(default_factory=dict)   # SplatConfig overrides, e.g. {"iterations": 20000}
    refine: bool = False             # second pass: fit every view's camera to the first splat, retrain
    restore: bool = False            # ... plus SeedVR2-restored orbit renders of the first splat as views
    orbit: bool = False              # orbit video pass: a calibrated turntable video (Wan), then retrain
    # camera paths for that video (i2g/orbit_stage.py, pacgs.orbit.RIGS)
    orbit_rigs: str = "360,face,face_up,face_down,face_high,face_top,helix"
    orbit_frames: int = 81                      # frames per clip (Wan 2.2 is trained on 81)
    orbit_size: tuple[int, int] = (480, 832)    # width, height of the generated frames
    orbit_seed: int = 1001
    orbit_prompt: str = ""                      # empty: a generic studio-orbit description
    orbit_holdout_every: int = 8                # 1 in N frames kept out of training, to score the result
    orbit_face_boost: float = 2.0               # training steps per head close-up frame, relative to a body frame
                                                # (2.0 with three face clips: measured as f6/f7; 3.0 was for one)
    orbit_iterations: int = 40000               # the orbit pass trains longer than the earlier passes (0: as preset)
    # The head fix (next-stage-research/docs/headpass_results.md), used when the orbit pass runs:
    coherent_head: bool = True                  # Wan's depth video shows one continuous head (no face plate, no
                                                # forehead groove) instead of the second pass's splat as it is
    head_pass: bool = True                      # then a separate head splat, trained on the head close-ups and
                                                # merged onto the body at the neck
    head_iterations: int = 30000
    # every core angle (both sides, the back, the four diagonals) must have a usable view before training;
    # an angle without one gets new candidates, up to this many rounds (ghost_body_fix.md)
    require_core_views: bool = True
    view_retries: int = 5


# --- stamps ---------------------------------------------------------------------------------------

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _stamp_path(paths: RunPaths, stage: str) -> Path:
    return paths.stage_dir(stage) / "stage.json"


def _settings(opts: RunOptions, stage: str) -> dict:
    if stage in ("train", "refine", "orbit"):
        # the full effective config, so a change of the defaults in splat.py also retrains
        from .splat import config_dict
        s = {"photo_hr": opts.photo_hr, "splat": config_dict(make_splat_config(opts))}
        if stage == "refine":
            s |= {"refine": opts.refine, "restore": opts.restore}
        if stage == "orbit":
            s |= {"orbit": opts.orbit, "rigs": opts.orbit_rigs, "frames": opts.orbit_frames,
                  "size": list(opts.orbit_size), "seed": opts.orbit_seed, "prompt": opts.orbit_prompt,
                  "coherent_head": opts.coherent_head, "head_pass": opts.head_pass,
                  "head_iterations": opts.head_iterations}
    else:
        s = {"ingest": {}, "body": {"vfov_deg": opts.vfov_deg}, "views": {"views": opts.views, "seeds": opts.seeds},
             "register": {"views": opts.views, "seeds": opts.seeds, "vfov_deg": opts.vfov_deg,
                          # v2: silhouette rescue of side views, front/back flip guard, core-view retries
                          "method": 2, "require_core_views": opts.require_core_views}}[stage]
    return json.loads(json.dumps(s, sort_keys=True))


def _upstream(paths: RunPaths, opts: RunOptions, stage: str) -> str:
    h = hashlib.sha256()
    if stage == "ingest":
        h.update(file_sha256(opts.input_image).encode())
    for dep in DEPENDS[stage]:
        p = _stamp_path(paths, dep)
        h.update(p.read_bytes() if p.exists() else b"missing")
    return h.hexdigest()


def write_stamp(paths: RunPaths, opts: RunOptions, stage: str, info: dict) -> None:
    data = {"stage": stage, "upstream": _upstream(paths, opts, stage), "settings": _settings(opts, stage),
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "info": info}
    _stamp_path(paths, stage).write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")


def stage_status(paths: RunPaths, opts: RunOptions, stage: str) -> tuple[bool, str]:
    """(up to date?, why)."""
    stamp_file = _stamp_path(paths, stage)
    if not stamp_file.exists():
        if _adopt_legacy(paths, opts, stage):
            return True, "reusing results from an earlier run"
        return False, "not run yet"
    stamp = json.loads(stamp_file.read_text())
    if stamp.get("upstream") != _upstream(paths, opts, stage):
        return False, "an earlier stage changed"
    if stamp.get("settings") != _settings(opts, stage):
        return False, "settings changed"
    if stage in ("train", "refine") and paths.find_deliverable("character.ply") is None:
        return False, "output missing"
    return True, "up to date"


def _adopt_legacy(paths: RunPaths, opts: RunOptions, stage: str) -> bool:
    """Accept results written before stamps existed (the first v2 runs), then stamp them."""
    d = paths.stage_dir(stage)
    ok = False
    if stage == "ingest" and (d / "ingest.json").exists():
        source = Path(json.loads((d / "ingest.json").read_text())["source"])
        source = source if source.is_absolute() else REPO_ROOT / source
        ok = source.exists() and file_sha256(source) == file_sha256(opts.input_image)
    elif stage == "body":
        ok = (d / "body_world.npz").exists() and _stamp_path(paths, "ingest").exists()
    elif stage == "views" and (d / "views.json").exists():
        from .views import planned_pairs
        have = {(m["tag"], m["seed"]) for m in json.loads((d / "views.json").read_text()) if Path(m["path"]).exists()}
        ok = set(planned_pairs(opts.views, opts.seeds)) <= have and _stamp_path(paths, "ingest").exists()
    elif stage == "register" and (d / "registration.json").exists():
        from .views import planned_pairs
        have = {(r["tag"], r["seed"]) for r in json.loads((d / "registration.json").read_text())}
        ok = set(planned_pairs(opts.views, opts.seeds)) <= have and all(
            _stamp_path(paths, s).exists() for s in DEPENDS["register"])
    if ok:
        write_stamp(paths, opts, stage, {"adopted": True})
    return ok


def archive_dir(paths: RunPaths, directory: Path, label: str) -> None:
    if not directory.exists() or not any(directory.iterdir()):
        return
    dest = paths.work / "_archive" / f"{label}_{time.strftime('%Y%m%d-%H%M%S')}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(directory), str(dest))
    directory.mkdir(parents=True, exist_ok=True)
    log.info("Moved old %s results to %s", label, dest)


def check_same_photo(paths: RunPaths, opts: RunOptions) -> None:
    """If the working folder holds a different photo's run, archive it rather than mixing results."""
    ingest_json = paths.stage_dir("ingest") / "ingest.json"
    if not ingest_json.exists():
        return
    meta = json.loads(ingest_json.read_text())
    old = meta.get("source_sha256")
    if old is None:
        source = Path(meta["source"])
        source = source if source.is_absolute() else REPO_ROOT / source
        old = file_sha256(source) if source.exists() else None
    if old is not None and old != file_sha256(opts.input_image):
        log.warning("The working folder holds a run for a different photo; archiving it and starting fresh")
        for stage in STAGES:
            archive_dir(paths, paths.stage_dir(stage), f"previous_photo_{STAGE_DIRS[stage]}")


# --- stages ---------------------------------------------------------------------------------------

def run_ingest(paths: RunPaths, opts: RunOptions, client: ComfyClient, make_bar=None) -> dict:
    from .ingest import ingest
    from .matting import Matter

    bar = make_bar(total=2, desc="Cut-out", unit="step") if make_bar else NullBar()
    client.free()
    birefnet = MODELS_DIR / "birefnet"
    try:
        matter = Matter(birefnet)
    except Exception as exc:
        raise PipelineError(f"could not load the BiRefNet matting model from {birefnet}: {exc}",
                            hint="That folder needs model.safetensors, config.json, birefnet.py and "
                                 "BiRefNet_config.py (Hugging Face: ZhengPeng7/BiRefNet).") from exc
    bar.update(1)
    try:
        meta = ingest(opts.input_image, paths.stage_dir("ingest"), matter)
    except RuntimeError as exc:
        if "no person" in str(exc):
            raise PipelineError("no person was found in the photo",
                                hint="Use a front-facing, full-body photo of one person, clearly separated "
                                     "from the background.") from exc
        raise
    finally:
        matter.close()
    bar.update(1)
    bar.close()

    x0, y0, x1, y1 = meta["person_bbox_source"]
    w, h = meta["source_size"]
    if y1 - y0 < 600:
        log.warning("The person is only %d px tall in the photo; the splat will be soft. A larger photo helps.",
                    y1 - y0)
    if y0 <= 2 or y1 >= h - 2:
        log.warning("The person touches the top or bottom edge of the photo (cropped head or feet?). "
                    "A full-body photo with some margin works best.")
    meta["source_sha256"] = file_sha256(opts.input_image)
    (paths.stage_dir("ingest") / "ingest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("Cut-out: person %d x %d px in a %d x %d photo", x1 - x0, y1 - y0, w, h)
    write_stamp(paths, opts, "ingest", {"person_px": [x1 - x0, y1 - y0]})
    return meta


def run_body(paths: RunPaths, opts: RunOptions, client: ComfyClient, make_bar=None) -> dict:
    from . import body

    bar = make_bar(total=3, desc="3D body fit", unit="step") if make_bar else NullBar()
    d = paths.stage_dir("body")
    npz = body.run_sam3d(client, paths.stage_dir("ingest") / "canvas.png", f"i2g/{opts.name}/body_photo",
                         opts.vfov_deg, d / "sam3d_photo.npz")
    client.free()
    bar.update(1)
    try:
        person = body.load_person(npz)
    except KeyError as exc:
        raise PipelineError("SAM 3D Body did not return a body for the photo",
                            hint="Check 01_body and 00_ingest/canvas.png: the person should be clearly visible "
                                 "and upright.") from exc
    world = body.build_world(person)
    body.save_world(world, d)
    bar.update(1)

    canvas = cv2.imread(str(paths.stage_dir("ingest") / "canvas.png"))
    cv2.imwrite(str(d / "overlay_photo.png"), body.overlay(canvas, world, world["photo_w2c"]))
    alpha = cv2.imread(str(paths.stage_dir("ingest") / "alpha.png"), cv2.IMREAD_GRAYSCALE) / 255.0
    summary = body.summarize(world) | body.silhouette_stats(world, world["photo_w2c"], alpha)
    body.write_json(d / "body_summary.json", summary)
    bar.update(1)
    bar.close()

    log.info("3D body: %.2f m tall, camera %.2f m away, %.0f%% of the body inside the cut-out",
             summary["height_m"], summary["camera_distance_m"], summary["coverage"] * 100)
    if summary["coverage"] < 0.85:
        log.warning("The body fit overlaps the cut-out poorly (%.0f%%). Check 01_body/overlay_photo.png; "
                    "unusual poses or loose clothing can confuse the fit.", summary["coverage"] * 100)
    if not 1.3 < summary["height_m"] < 2.2:
        log.warning("Estimated height %.2f m looks unusual; check 01_body/overlay_photo.png", summary["height_m"])
    write_stamp(paths, opts, "body", summary)
    return summary


SLOW_GPU_HINT = ("A sampling step took minutes: VRAM spilled into system RAM. Close other GPU apps (browsers, games, "
                 "other AI tools), set NVIDIA Control Panel > CUDA - Sysmem Fallback Policy > Prefer No Sysmem "
                 "Fallback, and run again.")


def _slow_comfy_error(exc: Exception, client: ComfyClient) -> PipelineError:
    """A ComfyUI job that crawled or timed out. The usual cause is a second GPU job (another run, a test,
    a ComfyUI tab) paging model weights out of VRAM and RAM, so name it when there is one."""
    from .gpu_lock import describe, other_gpu_jobs
    others = other_gpu_jobs()
    if others:
        return PipelineError(f"{exc}. Another Image-to-Gaussian job was using the GPU at the same time: "
                             f"{describe(others)}",
                             hint="Only one job can use the GPU at a time. Wait for that one to finish (or stop "
                                  "it), then Start again without Rebuild; finished stages are kept.")
    try:
        foreign = client.busy()
    except Exception:  # noqa: BLE001
        foreign = False
    if foreign:
        return PipelineError(f"{exc}. ComfyUI was also running other jobs (another app or a ComfyUI tab)",
                             hint="Let ComfyUI finish or cancel its other jobs, then Start again without Rebuild.")
    return PipelineError(str(exc), hint=SLOW_GPU_HINT)


def run_views(paths: RunPaths, opts: RunOptions, client: ComfyClient, make_bar=None) -> dict:
    from .ingest import CANVAS_H, CANVAS_W
    from .views import VIEW_PRESETS, generate_views, seed_values

    views, seeds = VIEW_PRESETS[opts.views], seed_values(opts.seeds)
    bar = make_bar(total=len(views) * len(seeds), desc="Generating views", unit="img") if make_bar else NullBar()
    start = time.perf_counter()
    try:
        manifest = generate_views(client, paths.stage_dir("ingest") / "canvas.png", paths.stage_dir("views"),
                                  opts.name, CANVAS_W, CANVAS_H, views, seeds, bar)
    except (WatchdogTripped, TimeoutError) as exc:
        raise _slow_comfy_error(exc, client) from exc
    finally:
        bar.close()
        client.free()
    log.info("Views: %d images ready (%.1f min)", len(manifest), (time.perf_counter() - start) / 60)
    write_stamp(paths, opts, "views", {"images": len(manifest)})
    return {"images": len(manifest)}


def run_register(paths: RunPaths, opts: RunOptions, client: ComfyClient, make_bar=None) -> dict:
    """Register every view, then make sure each core angle (both sides, the back, the four diagonals) has a
    usable one: for an angle that has none, generate new candidates (new seeds) and register again, up to
    opts.view_retries rounds. Without side views the splat grows a ghost second body (ghost_body_fix.md),
    so by default the run stops here rather than train on an incomplete set."""
    from . import body
    from .ingest import CANVAS_H, CANVAS_W
    from .matting import Matter
    from .register import CORE_VIEWS, core_gaps, register_all
    from .views import VIEW_PRESETS, extra_pairs, generate_views, next_seeds, planned_pairs

    views_dir = paths.stage_dir("views")
    world = body.load_world(paths.stage_dir("body"))
    requests = {v.tag: v for v in VIEW_PRESETS[opts.views] if v.source is None}
    pairs = set(planned_pairs(opts.views, opts.seeds)) | extra_pairs(views_dir, opts.views, opts.seeds)

    def measure(desc: str) -> list[dict]:
        bar = make_bar(total=len(pairs), desc=desc, unit="view") if make_bar else NullBar()
        try:
            return register_all(client, matter, world, paths.as_dict(), opts.name, opts.vfov_deg, pairs, bar)
        finally:
            bar.close()

    client.free()
    matter = Matter(MODELS_DIR / "birefnet")
    try:
        records = measure("Measuring cameras")
        for attempt in range(1, opts.view_retries + 1):
            gaps = [tag for tag in core_gaps(records) if tag in requests]
            if not gaps:
                break
            log.warning("No usable view yet for %s: generating %d new candidates for each (retry %d of %d)",
                        ", ".join(f"{t} ({CORE_VIEWS[t]:.0f} deg)" for t in gaps), opts.seeds, attempt,
                        opts.view_retries)
            new = {tag: next_seeds(views_dir, tag, opts.seeds) for tag in gaps}
            bar = make_bar(total=sum(map(len, new.values())), desc=f"Regenerating views (retry {attempt})",
                           unit="img") if make_bar else NullBar()
            try:
                for tag, seeds in new.items():
                    generate_views(client, paths.stage_dir("ingest") / "canvas.png", views_dir, opts.name,
                                   CANVAS_W, CANVAS_H, [requests[tag]], seeds, bar)
            except (WatchdogTripped, TimeoutError) as exc:
                raise _slow_comfy_error(exc, client) from exc
            finally:
                bar.close()
                client.free()
            pairs |= {(tag, s) for tag, seeds in new.items() for s in seeds}
            records = measure(f"Measuring cameras (retry {attempt})")
    finally:
        matter.close()
        client.free()

    selected = [r for r in records[1:] if r.get("selected")]
    accepted = [r for r in records[1:] if r.get("accepted")]
    sheet = paths.stage_dir("register") / "contact_sheet.png"
    if sheet.exists():
        shutil.copy2(sheet, paths.deliverable("contact_sheet.png"))
        paths.clear_legacy("contact_sheet.png")
    log.info("Cameras: %d of %d views usable, %d selected for training (plus the photo). Contact sheet: %s",
             len(accepted), len(records) - 1, len(selected), paths.deliverable("contact_sheet.png"))
    gaps = core_gaps(records)
    if gaps:
        missing = ", ".join(f"{t} ({CORE_VIEWS[t]:.0f} deg)" for t in gaps)
        if opts.require_core_views:
            raise PipelineError(f"no usable view for {missing} after {opts.view_retries} rounds of new candidates; "
                                "training without it gives a ghost second body or a smeared side",
                                hint="Open contact_sheet.png to see why they were rejected. Rerun with "
                                     "--view_retries 10 for more attempts, or --no-require_core_views to "
                                     "train anyway.")
        log.warning("No usable view for %s; training anyway (--no-require_core_views). Expect a ghost body or "
                    "smears on that side.", missing)
    if len(selected) < 3:
        raise PipelineError(f"only {len(selected)} generated views are usable; that is too few for a 360° splat",
                            hint="Open contact_sheet.png to see why views were rejected, then rerun with "
                                 "--seeds 3 (more candidates per angle) or --force --from_stage views.")
    write_stamp(paths, opts, "register", {"selected": [f"{r['tag']}_s{r['seed']}" for r in selected],
                                          "accepted": len(accepted), "extra_views": sorted(
                                              f"{t}_s{s}" for t, s in extra_pairs(views_dir, opts.views, opts.seeds))})
    return {"selected": len(selected), "accepted": len(accepted)}


def make_splat_config(opts: RunOptions):
    from .splat import SplatConfig
    cfg = SplatConfig()
    valid = {f.name: f for f in dataclasses.fields(SplatConfig)}
    for key, value in opts.splat.items():
        if key not in valid:
            raise PipelineError(f"unknown training setting '{key}'",
                                hint="Valid --set keys: " + ", ".join(sorted(valid)))
        setattr(cfg, key, value)
    # face_align used to be off with the orbit pass: its silhouette measurement pushed the test image's
    # head 2.7 cm back when the views agreed with the body mesh within 1 cm, and the orbit video repainted
    # the face sunk behind the raised arm. The keypoint measurement (face_align_method="keypoints") does
    # not make that mistake (test image 0.2-0.9 cm, so no push), and the orbit pass needs it: with the
    # side views in training, a head the body mesh put 4-6 cm too far forward left the photo's face as a
    # detached plate in front of the real head (next-stage-research/docs/ghost_body_fix.md).
    return cfg


def parse_setting(text: str):
    """'key=value' -> (key, typed value) using SplatConfig's field types."""
    from .splat import SplatConfig
    if "=" not in text:
        raise PipelineError(f"--set expects key=value, got '{text}'", hint="Example: --set iterations=20000")
    key, raw = (s.strip() for s in text.split("=", 1))
    fields = {f.name: f for f in dataclasses.fields(SplatConfig)}
    if key not in fields:
        raise PipelineError(f"unknown training setting '{key}'", hint="Valid --set keys: " + ", ".join(sorted(fields)))
    kind = type(getattr(SplatConfig(), key))
    try:
        value = (raw.lower() in ("1", "true", "yes", "on")) if kind is bool else kind(raw)
    except ValueError as exc:
        raise PipelineError(f"--set {key}: '{raw}' is not a valid {kind.__name__}") from exc
    return key, value


def _train_and_export(paths: RunPaths, opts: RunOptions, cfg, world: dict, registration, stage: str,
                      make_bar=None) -> dict:
    """Train on `registration` (registration.json or records), clean, export the deliverables."""
    import imageio.v2 as imageio
    import torch

    from . import vram
    from .cameras import azimuth_elevation, orbit_camera
    from .splat import clean, config_dict, export_ply, load_views, psnr, render_rgb, train

    views = load_views(registration, world["K"], cfg, "cuda",
                       hr_photo=paths.stage_dir("ingest") / "source_hr_rgba.png" if opts.photo_hr else None,
                       world=world)
    log.info("Training on %d views (%s), %d iterations", len(views["names"]),
             ", ".join(n for n in views["names"] if not n.startswith("orbit_")) +
             (f" + {sum(k == 'restored' for k in views['kinds'])} restored orbit frames"
              if "restored" in views["kinds"] else ""), cfg.iterations)

    report: dict = {"stage": stage, "config": config_dict(cfg), "views": views["names"]}
    bar = make_bar(total=cfg.iterations, desc="Training splat", unit="it") if make_bar else NullBar()
    try:
        with vram.stage(stage, report):
            params, stats = train(views, world, cfg, bar, preview_dir=paths.work)
    except torch.cuda.OutOfMemoryError as exc:
        raise PipelineError("the GPU ran out of memory while training",
                            hint=f"Close other GPU apps and ComfyUI tabs, then rerun with --from_stage {stage}. "
                                 "If it persists: --no_hr_photo, or --set grow_grad2d=0.0004 (fewer Gaussians).") \
            from exc
    finally:
        bar.close()
    refined = torch.tensor(stats.pop("refined_w2c"), device="cuda")
    report |= stats

    bar = make_bar(total=4, desc="Exporting", unit="step") if make_bar else NullBar()
    report["clean"] = clean(params, views, refined, cfg)
    if cfg.face_bake:
        from .splat import bake_face_into
        export_ply(params, paths.stage_dir(stage) / "character_unbaked.ply")   # the bake can be redone from this
        report["face_bake"] = bake_face_into(params, world, paths.stage_dir("ingest") / "source_hr_rgba.png", cfg)
    bar.update(1)
    export_ply(params, paths.deliverable("character.ply"))
    paths.clear_legacy("character.ply")
    bar.update(1)

    per_view = {}
    for i, view_name in enumerate(views["names"]):
        if views["kinds"][i] == "restored":
            continue
        img = (views["images"][i].cpu().numpy() * 255).round().astype(np.uint8)
        mat = views["alphas"][i][..., 0].cpu().numpy()
        vh, vw = img.shape[:2]
        pred = render_rgb(params, refined[i], views["K"][i], vw, vh)
        gt = (img * mat[..., None] + 127.5 * (1 - mat[..., None])).round().astype(np.uint8)
        per_view[view_name] = round(psnr(pred, gt, mat > 0.5), 2)
        if i == 0:
            pair = np.concatenate([gt, pred], 1)
            pair = cv2.resize(pair, (pair.shape[1] * 1248 // vh, 1248), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(paths.deliverable("photo_vs_render.png")), cv2.cvtColor(pair, cv2.COLOR_RGB2BGR))
            paths.clear_legacy("photo_vs_render.png")
    report["psnr_masked"] = per_view
    bar.update(1)

    export_previews(paths, params, world)
    paths.deliverable("report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    paths.clear_legacy("report.json")
    bar.update(1)
    bar.close()

    ply_mb = paths.deliverable("character.ply").stat().st_size / 1024 ** 2
    log.info("Splat: %s Gaussians (%.0f MB), photo-view PSNR %.1f dB, %.1f min training, peak VRAM %.1f GiB",
             f"{report['clean']['remaining']:,}", ply_mb, per_view[views["names"][0]], report["train_seconds"] / 60,
             report["stages"][0]["peak_allocated_gib"])
    del views, params
    torch.cuda.empty_cache()
    return {"gaussians": report["clean"]["remaining"], "ply_mb": round(ply_mb, 1),
            "train_minutes": round(report["train_seconds"] / 60, 1), "psnr": per_view}


def export_previews(paths: RunPaths, params, world: dict) -> None:
    """turntable.mp4 and preview_8_angles.png of a splat, as the deliverables."""
    import imageio.v2 as imageio
    import torch

    from .cameras import azimuth_elevation, orbit_camera
    from .splat import render_rgb
    K = torch.tensor(world["K"], device="cuda").float()
    w, h = world["size"]
    _, _, radius = azimuth_elevation(world["photo_w2c"], world["target"])
    frames = [render_rgb(params, torch.tensor(orbit_camera(world["target"], radius, az, 5.0), device="cuda").float(),
                         K, w, h) for az in np.linspace(0, 360, 73)[:-1]]
    imageio.mimsave(paths.deliverable("turntable.mp4"), frames, fps=24, quality=8, macro_block_size=8)
    paths.clear_legacy("turntable.mp4")
    tiles = [frames[j] for j in range(0, 72, 9)]
    grid = np.concatenate([np.concatenate(tiles[:4], 1), np.concatenate(tiles[4:], 1)], 0)
    cv2.imwrite(str(paths.deliverable("preview_8_angles.png")),
                cv2.cvtColor(cv2.resize(grid, (grid.shape[1] // 2, grid.shape[0] // 2), interpolation=cv2.INTER_AREA),
                             cv2.COLOR_RGB2BGR))
    paths.clear_legacy("preview_8_angles.png")


def _free_comfy_for_training(client: ComfyClient) -> None:
    """Unload ComfyUI's models before training, unless it is busy with other jobs (a ComfyUI shared with
    other work): unloading would disrupt them, and training beside them is slow, so say so."""
    if client.busy():
        log.warning("ComfyUI is busy with other jobs: leaving its models loaded. Training shares the GPU with "
                    "them and can be much slower (VRAM spills into system RAM) until they finish.")
        return
    client.free()


def _view_keypoints(paths: RunPaths, records: list[dict]) -> dict:
    """SAM 3D Body's 2D keypoints of every registered view (from the register stage's cache)."""
    from . import body
    cache_path = paths.stage_dir("register") / "sam3d_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    out = {}
    for r in records[1:]:
        name = f"{r['tag']}_s{r['seed']}"
        hits = [v for k, v in cache.items() if k == name or k.startswith(name + "@")]
        if hits and Path(hits[0]).exists():
            person = body.load_person(Path(hits[0]))
            if "kp2d" in person:
                out[name] = person["kp2d"]
    return out


def _aligned_world(paths: RunPaths, cfg, records: list[dict] | None = None) -> dict:
    """The body scaffold with its head moved along the photo's rays to where the generated views put it
    (priors.keypoint_head_push). Everything that places the photo's face in 3D uses this head: the face
    priors in training, and the orbit pass's head stand-in. `records`: the views with their cameras
    (default: the register stage's; the second pass and the orbit pass give their fitted, tighter ones)."""
    from . import body
    from .priors import face_push, keypoint_head_push, push_head
    world = body.load_world(paths.stage_dir("body"))
    if not cfg.face_align:
        return world
    if records is None:
        records = json.loads((paths.stage_dir("register") / "registration.json").read_text())
    if cfg.face_align_method == "silhouette":
        push = face_push(world, records, "cuda")
        log.info("Face: the body mesh's head moved %.1f cm back to fit the generated views' silhouettes", push * 100)
    else:
        push, report = keypoint_head_push(world, records, _view_keypoints(paths, records))
        log.info("Face: the generated views put the head %.1f cm %s than the body mesh (from %d side and "
                 "diagonal views; nose/eyes/ears %s cm); moved it %.1f cm along the photo's rays",
                 abs(report.get("measured_cm", 0.0)), "further back" if report.get("measured_cm", 0) >= 0
                 else "further forward", report.get("views", 0), report.get("per_point_cm", {}), push * 100)
        if abs(push) >= 0.03:
            log.warning("Face: SAM 3D Body put this head %.0f cm off where the generated views see it; the head "
                        "was moved to match (without this the photo's face sits in front of the head as a "
                        "mask).", abs(report["measured_cm"]))
    return push_head(world, push)


def run_train(paths: RunPaths, opts: RunOptions, client: ComfyClient, make_bar=None) -> dict:
    cfg = make_splat_config(opts)
    _archive_outputs(paths)
    _free_comfy_for_training(client)
    world = _aligned_world(paths, cfg)
    result = _train_and_export(paths, opts, cfg, world, paths.stage_dir("register") / "registration.json",
                               "train", make_bar)
    write_stamp(paths, opts, "train", {"gaussians": result["gaussians"], "psnr": result.pop("psnr")})
    return result


def run_refine(paths: RunPaths, opts: RunOptions, client: ComfyClient, make_bar=None) -> dict:
    """Second pass. The first splat is a consensus of all views, so each view's camera can now be fitted
    to it by render-and-compare (much tighter than keypoints, and it rescues side views the body fitter
    misread). The views are re-selected with those cameras and the splat is retrained from scratch;
    with --restore, SeedVR2-cleaned orbit renders of the first splat join as extra low-weight views."""
    import torch

    from .evaluate import load_ply
    from .register import select_views
    from .reregister import refit_records

    if not opts.refine:
        log.info("Second pass is off (the standard/high presets and --refine turn it on)")
        write_stamp(paths, opts, "refine", {"skipped": True})
        return {"skipped": True}
    cfg = make_splat_config(opts)
    _free_comfy_for_training(client)
    world = _aligned_world(paths, cfg)
    refine_dir = paths.stage_dir("refine")
    records = json.loads((paths.stage_dir("register") / "registration.json").read_text())
    ply = paths.find_deliverable("character.ply")
    if ply is None:
        raise PipelineError("the second pass needs the splat from the train stage",
                            hint="Rerun with --from_stage train.")
    params = load_ply(ply)
    bar = make_bar(total=len(records) - 1, desc="Fitting cameras to the splat", unit="view") if make_bar else NullBar()
    try:
        records = refit_records(params, world, records, bar=bar)
    finally:
        bar.close()
    select_views(records, min_separation_deg=12.0)     # fitted cameras are reliable enough to use more views
    (refine_dir / "registration_fitted.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    kept = [f"{r['tag']}_s{r['seed']}" for r in records[1:] if r.get("selected")]
    log.info("Cameras fitted: %d of %d views usable, %d selected", sum(bool(r.get("accepted")) for r in records[1:]),
             len(records) - 1, len(kept))
    world = _aligned_world(paths, cfg, records)        # re-measure the head with the fitted cameras

    if opts.restore:
        from .restore import orbit_cameras, render_orbit, restore_orbit
        cams = orbit_cameras(world)
        render_orbit(params, world, cams, refine_dir / "orbit")
        del params
        torch.cuda.empty_cache()
        bar = make_bar(total=3, desc="Restoring orbit (SeedVR2)", unit="ring") if make_bar else NullBar()
        try:
            records = records + restore_orbit(client, refine_dir / "orbit", cams, opts.name, bar=bar)
        finally:
            bar.close()
            client.free()
    else:
        del params
        torch.cuda.empty_cache()

    _archive_outputs(paths, copy=True)
    result = _train_and_export(paths, opts, cfg, world, records, "refine", make_bar)
    write_stamp(paths, opts, "refine", {"gaussians": result["gaussians"], "psnr": result.pop("psnr"),
                                        "selected": kept})
    return result


def _archive_outputs(paths: RunPaths, copy: bool = False) -> None:
    """Keep every previous result in output/history/<time>/ before writing new files. With copy, the
    current files also stay in place until overwritten (the second pass: if it is interrupted, the
    first pass's splat is still the result and the train stage stays done)."""
    existing = paths.deliverables_on_disk(include_contact_sheet=False)
    if not existing:
        return
    dest = paths.output / "history" / time.strftime("%Y%m%d-%H%M%S")
    dest.mkdir(parents=True, exist_ok=True)
    for src in existing:
        (shutil.copy2 if copy else shutil.move)(str(src), str(dest / src.name))
    log.info("Previous result %s %s", "copied to" if copy else "moved to", dest)


def run_orbit(paths: RunPaths, opts: RunOptions, client: ComfyClient, make_bar=None) -> dict:
    from .orbit_stage import run_orbit as _run
    return _run(paths, opts, client, make_bar)


RUNNERS = {"ingest": run_ingest, "body": run_body, "views": run_views, "register": run_register,
           "train": run_train, "refine": run_refine, "orbit": run_orbit}

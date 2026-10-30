"""Experiment runner for the head-pass research: the product's orbit pass, split into steps so arms can share
generated clips.

    exp/                       one subject (made by `setup` from a finished product run)
      00_ingest/, 01_body/     copied from the product run (the world frame is identical, so results compare)
      config.json              source run, subject, the product's RunOptions
      standins/<name>.ply      depth stand-ins: old (the product's) and coherent (H2)
      gen/<set>/controls, start, frames/<rig>, data/<rig>, data/all
      arms/<arm>/character.ply, report.json, sheets

Settings for training come from the product (i2g.orbit_stage.splat_config with the run's own options), so
an arm differs from the product only where the arm says so.

    python -m headpass setup    --src "output/F-14-Fixed-Face/Nurse Man/v2" --exp output/headpass/nurse
    python -m headpass standin  --exp ... --kind coherent
    python -m headpass generate --exp ... --set coh --standin coherent
    python -m headpass prepare  --exp ... --set coh
    python -m headpass train    --exp ... --set coh --arm e1_coherent
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from i2g.log import get_logger

log = get_logger("headpass")


# --- setup ---------------------------------------------------------------------------------------------

def setup(src: Path, exp: Path) -> dict:
    exp.mkdir(parents=True, exist_ok=True)
    for stage in ("00_ingest", "01_body"):
        shutil.copytree(src / stage, exp / stage, dirs_exist_ok=True)
    (exp / "03_qa").mkdir(exist_ok=True)              # the photo's record (pacgs.dataset.photo_record)
    shutil.copy2(src / "03_qa" / "registration.json", exp / "03_qa" / "registration.json")
    summary = next((src / "final").glob("*-run_summary.json"))
    opts = json.loads(summary.read_text())["options"]
    hist = sorted((src / "final" / "history").iterdir())
    cfg = {"src": str(src), "name": opts["name"], "options": opts,
           "product_final": str(next((src / "final").glob("*-character.ply"))),
           "product_standin": str(next(hist[-1].glob("*character.ply"))),
           "product_pass1": str(next(hist[0].glob("*character.ply"))),
           "product_orbit_data": str(src / "08_orbit" / "data" / "all")}
    (exp / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    (exp / "standins").mkdir(exist_ok=True)
    shutil.copy2(cfg["product_standin"], exp / "standins" / "old.ply")
    log.info("Set up %s from %s", exp, src)
    return cfg


def load(exp: Path):
    from i2g.body import load_world
    cfg = json.loads((exp / "config.json").read_text())
    return cfg, load_world(exp / "01_body")


def run_options(cfg: dict):
    from i2g.pipeline import RunOptions
    o = dict(cfg["options"])
    o["input_image"] = Path(o["input_image"])
    o["orbit_size"] = tuple(o["orbit_size"])
    return RunOptions(**o)


# --- stand-ins -----------------------------------------------------------------------------------------

def standin(exp: Path, kind: str, **kw) -> Path:
    from .headsurf import build_standin
    from .metrics import Reference
    cfg, world = load(exp)
    out = exp / "standins" / f"{kind}.ply"
    if kind == "old":
        shutil.copy2(cfg["product_standin"], out)
        return out
    ref = Reference(exp, world)
    stats = build_standin(exp / "standins" / "old.ply", world, ref, out, save_maps=exp / "standins" / f"{kind}_maps.npz",
                          **kw)
    (exp / "standins" / f"{kind}.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    log.info("Stand-in %s: %s", kind, stats)
    return out


# --- generation (ComfyUI must be running) --------------------------------------------------------------

def generate(exp: Path, gset: str, standin_name: str, rigs: list[str] | None = None, seed: int | None = None,
             url: str | None = None) -> dict:
    from i2g.comfy_client import ComfyClient
    from i2g.orbit_stage import _describe
    from i2g.paths import COMFY_URL
    from pacgs.generate import GenSettings, generate_arcs
    from pacgs.orbit import ARCS, RIGS, layout_cameras
    from pacgs.proxy import SplatProxy, encode_depth, inverse_depth_range, photo_source, reframe_photo

    cfg, world = load(exp)
    opts = run_options(cfg)
    rigs = rigs or [r.strip() for r in opts.orbit_rigs.split(",") if r.strip()]
    width, height = opts.orbit_size
    root = exp / "gen" / gset
    root.mkdir(parents=True, exist_ok=True)
    meta = {"standin": standin_name, "rigs": rigs, "seed": seed or opts.orbit_seed, "size": [width, height],
            "frames": opts.orbit_frames}
    (root / "gen.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    proxy = SplatProxy(exp / "standins" / f"{standin_name}.ply")
    try:
        for rig in rigs:
            cams = layout_cameras(world, rig, opts.orbit_frames, width, height)
            rng = inverse_depth_range(proxy, cams[::5], width, height)
            for cam in cams:
                out = root / "controls" / rig / f"arc{cam['arc']}"
                out.mkdir(parents=True, exist_ok=True)
                depth, alpha = proxy.depth(cam["w2c"], cam["K"], width, height)
                Image.fromarray(encode_depth(depth, alpha, rng)).save(out / f"frame_{cam['index']:04d}.png")
            source, source_K = photo_source(exp, world, cams[0]["zoom"])
            (root / "start").mkdir(exist_ok=True)
            reframe_photo(source, source_K, cams[0]["K"], width, height).save(root / "start" / f"{rig}.png")
    finally:
        del proxy
        torch.cuda.empty_cache()
    client = ComfyClient(url or COMFY_URL)
    settings = GenSettings.preset("lightning", opts.orbit_prompt or _describe(None), width=width, height=height,
                                  length=opts.orbit_frames, seed=meta["seed"])
    t0 = time.perf_counter()
    for rig in rigs:
        if (root / "frames" / rig / "arc0" / f"frame_{opts.orbit_frames - 1:04d}.png").exists():
            log.info("Clip %s already generated", rig)
            continue
        controls = [root / "controls" / rig / f"arc{k}" for k in range(len(ARCS[RIGS[rig]["arcs"]]))]
        start = root / "start" / f"{rig}.png"
        generate_arcs(client, settings, controls, start, start, root / "frames" / rig)
        log.info("Clip %s generated (%.1f min so far)", rig, (time.perf_counter() - t0) / 60)
    client.free()
    return meta


def prepare_set(exp: Path, gset: str, boost: float | None = None) -> dict:
    from pacgs.dataset import merge, prepare
    from pacgs.orbit import layout_cameras, rig_tag
    from pacgs.proxy import SplatProxy
    cfg, world = load(exp)
    opts = run_options(cfg)
    root = exp / "gen" / gset
    meta = json.loads((root / "gen.json").read_text())
    width, height = meta["size"]
    proxy = SplatProxy(exp / "standins" / f"{meta['standin']}.ply")
    # QA against the subject's own silhouette uses the product's stand-in, as the product does
    subject = SplatProxy(exp / "standins" / "old.ply")
    out = {}
    try:
        for rig in meta["rigs"]:
            cams = layout_cameras(world, rig, meta["frames"], width, height)
            out[rig] = prepare(root / "frames" / rig, cams, root / "data" / rig, exp, world, proxy, subject,
                               holdout_every=opts.orbit_holdout_every, tag=rig_tag(rig))
    finally:
        del proxy, subject
        torch.cuda.empty_cache()
    kept = [r for r in meta["rigs"] if not out[r]["restyled"]] or meta["rigs"]
    for r in meta["rigs"]:
        if r not in kept:
            log.warning("Dropping %s: re-styled (IoU vs subject %.3f)", r, out[r]["iou_splat_median"])
    b = opts.orbit_face_boost if boost is None else boost
    merge([root / "data" / r for r in kept], root / "data" / "all", {r: b for r in kept if r.startswith("face")})
    (root / "data" / "qa_summary.json").write_text(json.dumps({"kept": kept, "per_rig": out}, indent=1),
                                                   encoding="utf-8")
    return out


# --- training ------------------------------------------------------------------------------------------

def product_config(cfg: dict, rigs: list[str], overrides: dict | None = None):
    from i2g.orbit_stage import splat_config
    opts = run_options(cfg)
    c = splat_config(opts, rigs)
    for k, v in (overrides or {}).items():
        setattr(c, k, v)
    return c


def train_arm(exp: Path, arm: str, gset: str | None, overrides: dict | None = None, data_dir: Path | None = None,
              records_filter=None) -> dict:
    """Train one arm on a prepared set with the product's orbit-pass settings (+ overrides)."""
    from i2g.splat import clean, config_dict, export_ply, load_views, psnr, render_rgb, train
    cfg, world = load(exp)
    data = data_dir or (exp / "gen" / gset / "data" / "all")
    records = json.loads((data / "records.json").read_text())
    if records_filter is not None:
        records = records_filter(records)
    holdout = json.loads((data / "qa.json").read_text())["summary"]["holdout"]
    rigs = json.loads((exp / "gen" / gset / "gen.json").read_text())["rigs"] if gset else \
        [r.strip() for r in cfg["options"]["orbit_rigs"].split(",")]
    c = product_config(cfg, rigs, overrides)
    c.exclude = ",".join(s for s in (c.exclude, ",".join(holdout)) if s)
    out = exp / "arms" / arm
    out.mkdir(parents=True, exist_ok=True)
    views = load_views(records, world["K"], c, "cuda", hr_photo=exp / "00_ingest" / "source_hr_rgba.png"
                       if cfg["options"]["photo_hr"] else None, world=world)
    log.info("Arm %s: %d views, %d iterations", arm, len(views["names"]), c.iterations)
    t0 = time.perf_counter()
    params, stats = train(views, world, c, None)
    refined = torch.tensor(stats.pop("refined_w2c"), device="cuda")
    report = {"arm": arm, "gset": gset, "config": config_dict(c), "train_minutes": round((time.perf_counter() - t0) / 60, 1)}
    report |= {k: v for k, v in stats.items() if isinstance(v, (int, float, str, list, dict))}
    report["clean"] = clean(params, views, refined, c)
    export_ply(params, out / "character.ply")
    i = 0
    img = (views["images"][i].cpu().numpy() * 255).round().astype(np.uint8)
    mat = views["alphas"][i][..., 0].cpu().numpy()
    pred = render_rgb(params, refined[i], views["K"][i], img.shape[1], img.shape[0])
    gt = (img * mat[..., None] + 127.5 * (1 - mat[..., None])).round().astype(np.uint8)
    report["photo_psnr"] = round(psnr(pred, gt, mat > 0.5), 2)
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    del views, params
    torch.cuda.empty_cache()
    log.info("Arm %s done: photo %.2f dB, %.1f min", arm, report["photo_psnr"], report["train_minutes"])
    return report

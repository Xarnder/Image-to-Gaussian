"""Train a splat on a prepared video dataset with the v2 trainer (i2g.splat).

The base settings are the best v2 experiment's (its report.json config), so a comparison with v2
changes only the training views. Held-out frames go into cfg.exclude.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from i2g.log import get_logger

log = get_logger("pacgs.train")


def base_config(v2_report: Path):
    from i2g.splat import SplatConfig
    cfg = SplatConfig()
    if v2_report.exists():
        known = set(SplatConfig.__dataclass_fields__)
        for key, value in json.loads(v2_report.read_text())["config"].items():
            if key in known and key != "exclude":
                setattr(cfg, key, value)
    cfg.exclude = ""
    return cfg


def train_splat(data_dir: Path, out_dir: Path, subject_run: Path, v2_report: Path,
                settings: list[tuple[str, object]]) -> dict:
    from i2g import body
    from i2g.splat import bake_face_into, clean, config_dict, export_ply, load_views, train

    out_dir.mkdir(parents=True, exist_ok=True)
    records = json.loads((data_dir / "records.json").read_text())
    qa = json.loads((data_dir / "qa.json").read_text())["summary"]
    cfg = base_config(v2_report)
    for key, value in settings:
        setattr(cfg, key, value)
    cfg.exclude = ",".join(s for s in (cfg.exclude, ",".join(qa["holdout"])) if s)

    world = body.load_world(subject_run / "01_body")
    hr = subject_run / "00_ingest" / "source_hr_rgba.png"
    views = load_views(records, world["K"], cfg, "cuda", hr_photo=hr, world=world)
    log.info("Training %s on %d views (photo + %d video frames), %d held out; %s", out_dir.name,
             len(views["names"]), len(views["names"]) - 1, len(qa["holdout"]),
             ", ".join(f"{k}={v}" for k, v in settings) or "base settings")
    t0 = time.perf_counter()
    params, stats = train(views, world, cfg)
    refined = torch.tensor(stats.pop("refined_w2c"), device="cuda")
    report = {"config": config_dict(cfg), "settings": {k: v for k, v in settings}, "views": views["names"],
              "holdout": qa["holdout"], **stats}
    report["clean"] = clean(params, views, refined, cfg)
    if cfg.face_bake:
        export_ply(params, out_dir / "character_unbaked.ply")
        report["face_bake"] = bake_face_into(params, world, hr, cfg)
    export_ply(params, out_dir / "character.ply")
    report["total_seconds"] = round(time.perf_counter() - t0, 1)
    (out_dir / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    log.info("Trained %s in %.1f min: %d Gaussians", out_dir.name, report["total_seconds"] / 60,
             report["clean"]["remaining"])
    del views, params
    torch.cuda.empty_cache()
    return report

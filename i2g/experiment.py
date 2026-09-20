"""Train splat variants on an existing run's registered views and score them (developer tool).

Each experiment trains with the given settings, writes experiments/<name>/character.ply and
report.json, then runs i2g.evaluate with a fixed set of hold-out views (kept out of training).

    python3 -m i2g.experiment --run output/test-image/v2 --name E1 --set sh_degree=0 --set multiscale=0.5
    python3 -m i2g.experiment --run output/test-image/v2 --name E0_run2 --eval_ply output/test-image/v2/runs/run2/character.ply
    python3 -m i2g.experiment --run output/test-image/v2 --compare E0_run2 E1 E2
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .log import get_logger, setup_logging

log = get_logger("experiment")
HOLDOUT = "b_s202,r_s101,bl_s202,fr_s101,l_hi_s101"
TABLE = [("photo_psnr", "photo dB"), ("photo_psnr_hr", "HR dB"), ("face_mean", "face id"), ("face_min", "face min"), ("face_depth_mm", "face mm"),
         ("holdout_psnr", "hold dB"), ("holdout_ssim", "hold ssim"), ("holdout_vgg", "hold vgg"),
         ("holdout_iou", "hold iou"), ("needles_8x_3cm", "needles"), ("floater_pct", "floater%"),
         ("sharpness", "sharp"), ("gaussians", "gaussians")]


def restored_records(run_dir: Path, source: str, dit: str | None = None) -> list[dict]:
    """Render experiment `source`'s splat on the orbit and restore it with SeedVR2 (starts ComfyUI if needed)."""
    from . import body
    from .comfy_client import ComfyClient
    from .comfy_process import start_comfy, stop_comfy
    from .evaluate import load_ply
    from .paths import COMFY_ROOT, COMFY_URL, LOGS_DIR
    from .restore import SEEDVR2_DIT, orbit_cameras, render_orbit, restore_orbit

    src = run_dir / "experiments" / source
    orbit_dir = src / ("orbit" if not dit else f"orbit_{Path(dit).stem}")
    done = orbit_dir / "restored_views.json"
    if done.exists() and all(Path(r["path"]).exists() for r in json.loads(done.read_text())):
        return json.loads(done.read_text())
    world = body.load_world(run_dir / "01_body")
    cams = orbit_cameras(world)
    if not (orbit_dir / "frames").exists():
        params = load_ply(src / "character.ply")
        render_orbit(params, world, cams, orbit_dir)
        del params
        torch.cuda.empty_cache()
    client = ComfyClient(COMFY_URL)
    proc = None if client.is_up() else start_comfy(COMFY_ROOT, COMFY_URL, LOGS_DIR, sage=False, timeout_s=900)
    try:
        return restore_orbit(client, orbit_dir, cams, name=f"exp_{source}", dit=dit or SEEDVR2_DIT)
    finally:
        client.free()
        if proc is not None:
            stop_comfy(proc, client)


def train_one(run_dir: Path, name: str, settings: list[tuple[str, object]], registration: Path | list | None,
              hr_photo: bool, holdout: str) -> dict:
    from . import body
    from .evaluate import evaluate
    from .splat import SplatConfig, clean, config_dict, export_ply, load_views, train

    out = run_dir / "experiments" / name
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out / "train.log")
    cfg = SplatConfig()
    for key, value in settings:
        setattr(cfg, key, value)
    cfg.exclude = ",".join(s for s in (cfg.exclude, holdout) if s)
    log.info("Experiment %s: %s", name, ", ".join(f"{k}={v}" for k, v in settings) or "defaults")

    world = body.load_world(run_dir / "01_body")
    if cfg.face_align:
        from .priors import align_face
        world, push = align_face(world, json.loads((run_dir / "03_qa" / "registration.json").read_text()), "cuda")
        log.info("Face: head moved %.1f cm back to fit the views", push * 100)
    views = load_views(registration or run_dir / "03_qa" / "registration.json", world["K"], cfg, "cuda",
                       hr_photo=run_dir / "00_ingest" / "source_hr_rgba.png" if hr_photo else None, world=world)
    log.info("Training on %d views: %s", len(views["names"]), ", ".join(views["names"]))
    t0 = time.perf_counter()
    params, stats = train(views, world, cfg)
    refined = torch.tensor(stats.pop("refined_w2c"), device="cuda")
    report = {"config": config_dict(cfg), "views": views["names"], **stats}
    report["clean"] = clean(params, views, refined, cfg)
    if cfg.face_bake:
        from .splat import bake_face_into
        export_ply(params, out / "character_unbaked.ply")        # so the bake can be redone without training
        report["face_bake"] = bake_face_into(params, world, run_dir / "00_ingest" / "source_hr_rgba.png", cfg)
    export_ply(params, out / "character.ply")
    report["total_seconds"] = round(time.perf_counter() - t0, 1)
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    log.info("Trained %s in %.1f min: %d Gaussians", name, report["total_seconds"] / 60, report["clean"]["remaining"])
    del views, params
    torch.cuda.empty_cache()
    return evaluate(run_dir, out / "character.ply", out / "eval", holdout=holdout.split(","))


def compare(run_dir: Path, names: list[str], sheet: Path | None) -> None:
    from .evaluate import compare_sheet
    rows, sheets = [], []
    for name in names:
        ev = run_dir / "experiments" / name / "eval" / "eval.json"
        if not ev.exists():
            print(f"{name}: no eval.json yet")
            continue
        e = json.loads(ev.read_text())
        rows.append([name] + [e.get(k) for k, _ in TABLE])
        sheets.append((name, ev.parent / "sheet_row.png"))
    head = ["experiment"] + [h for _, h in TABLE]
    widths = [max(len(str(x)) for x in col) for col in zip(head, *rows)]
    for r in [head] + rows:
        print("  ".join(str(x).rjust(w) for x, w in zip(r, widths)))
    if sheet and sheets:
        compare_sheet(sheets, sheet)
        print("sheet:", sheet)


def main() -> None:
    from .pipeline import parse_setting
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--name")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--registration", type=Path, help="alternative registration.json (view-set experiments)")
    ap.add_argument("--restore_from", metavar="NAME", help="add SeedVR2-restored orbit frames of experiment NAME")
    ap.add_argument("--seedvr2_dit", help="SeedVR2 DiT file (default: the 3B fp8 model)")
    ap.add_argument("--no_hr_photo", action="store_true")
    ap.add_argument("--holdout", default=HOLDOUT)
    ap.add_argument("--eval_ply", type=Path, help="only evaluate this PLY as experiment --name")
    ap.add_argument("--compare", nargs="+", metavar="NAME")
    ap.add_argument("--sheet", type=Path)
    args = ap.parse_args()

    if args.compare:
        compare(args.run, args.compare, args.sheet or args.run / "experiments" / "compare.png")
        return
    if args.eval_ply:
        from .evaluate import evaluate
        out = args.run / "experiments" / args.name
        setup_logging(out / "eval.log")
        res = evaluate(args.run, args.eval_ply, out / "eval", holdout=args.holdout.split(","))
    else:
        registration = args.registration
        setup_logging(args.run / "experiments" / args.name / "train.log")
        if args.restore_from:
            base = json.loads((registration or args.run / "03_qa" / "registration.json").read_text())
            registration = base + restored_records(args.run, args.restore_from, args.seedvr2_dit)
        res = train_one(args.run, args.name, [parse_setting(s) for s in args.set], registration,
                        not args.no_hr_photo, args.holdout)
    print(json.dumps({k: v for k, v in res.items() if k not in ("holdout", "face")}, indent=1))


if __name__ == "__main__":
    main()

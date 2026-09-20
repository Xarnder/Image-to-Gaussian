"""python -m pacgs <command>: proxy-controlled orbit video -> splat, one stage per command.

  facecheck the face from every angle: renders vs the head close-up frames (psnr, sharpness, speckle)
  recolour  tint the face Gaussians the photo can see towards the photo (instead of a baked layer)
  setup     orbit cameras, depth control videos (mesh and/or splat proxy), the start image
  generate  Wan 2.2 Fun-Control orbit video in ComfyUI (ComfyUI must be running; GPU alone)
  sheet     quick look at a generation: frames above their control frames (CPU only)
  prepare   matte the frames, score them, write the training records + contact sheet (--crop trains a
            head clip on the head only; --out prepares the same clip twice under different names)
  train     train a splat on a prepared variant (v2 trainer, best v2 settings + --set overrides)
  eval      leave-frames-out scores, the v2 harness, turntable
  compare   table + compare sheet of several splats (and a v2 experiment)

Typical order: setup -> (start ComfyUI) generate ... -> (stop ComfyUI) prepare -> train -> eval -> compare.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from i2g.log import get_logger, setup_logging

log = get_logger("pacgs")

DEFAULT_SUBJECT = Path("output/test-image/v2")
DEFAULT_BASELINES = "E6_pixbake,E7_restore"      # the first one is also the splat proxy
DEFAULT_CONFIG_EXP = "E6_final_fit15"            # training settings (E6_pixbake has no report.json)
DEFAULT_PROMPT = ("A young woman with shoulder-length wavy dark brown hair stands completely still with both hands "
                  "resting on top of her head. She wears a fitted white tank top, dark grey skinny jeans and strappy "
                  "tan heeled sandals. The camera slowly orbits around her at a constant distance and height while she "
                  "stays perfectly still. Plain light grey studio background, soft even lighting, photorealistic, "
                  "sharp focus, high detail.")


def _config(run: Path) -> dict:
    return json.loads((run / "config.json").read_text())


def _world(cfg: dict) -> dict:
    from i2g import body
    return body.load_world(Path(cfg["subject"]) / "01_body")


def _cams(cfg: dict, world: dict, rig: str) -> list[dict]:
    from .orbit import layout_cameras
    return layout_cameras(world, rig, cfg["frames"], cfg["width"], cfg["height"])


def cmd_setup(args) -> None:
    from PIL import Image

    from .proxy import encode_depth, inverse_depth_range, make_proxy, photo_source, reframe_photo
    from .subject import snapshot
    run = args.run
    run.mkdir(parents=True, exist_ok=True)
    width, height = (int(v) for v in args.size.lower().split("x"))
    baselines = args.baselines.split(",")
    subject = snapshot(args.subject, run / "subject", baselines, args.config_exp)
    old = _config(run) if (run / "config.json").exists() else {}
    cfg = {"source": str(args.subject), "subject": str(subject),
           "splat": str(args.splat or subject / "experiments" / baselines[0] / "character.ply"),
           "width": width, "height": height, "frames": args.frames, "prompt": args.prompt,
           "baselines": baselines, "config_experiment": args.config_exp,
           "ranges": {} if args.fresh_ranges else old.get("ranges", {})}
    world = _world(cfg)
    (run / "start").mkdir(exist_ok=True)
    rigs = args.rigs.split(",")
    for rig in rigs:                                  # the start image: exactly what this rig's frame 0 sees
        cam0 = _cams(cfg, world, rig)[0]
        source, source_K = photo_source(Path(cfg["subject"]), world, cam0["zoom"])
        start = reframe_photo(source, source_K, cam0["K"], width, height)
        start.save(run / "start" / f"{rig}.png")
        if rig == "360":
            start.save(run / "start" / "photo.png")   # the name the first runs used
    for kind in args.proxies.split(","):
        proxy = make_proxy(kind, world, Path(cfg["splat"]))
        for rig in rigs:
            cams = _cams(cfg, world, rig)
            key = f"{kind}:{rig}"                     # one depth normalisation per proxy and rig
            rng = cfg["ranges"].get(key) or list(inverse_depth_range(proxy, cams[::5], width, height))
            cfg["ranges"][key] = rng
            t0 = time.perf_counter()
            for cam in cams:
                d = run / "controls" / kind / rig / f"arc{cam['arc']}"
                d.mkdir(parents=True, exist_ok=True)
                depth, alpha = proxy.depth(cam["w2c"], cam["K"], width, height)
                Image.fromarray(encode_depth(depth, alpha, rng)).save(d / f"frame_{cam['index']:04d}.png")
            log.info("Control %s/%s: %d frames in %.1f s", kind, rig, len(cams), time.perf_counter() - t0)
            _control_sheet(run, kind, rig, cfg, world)
        del proxy
    (run / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    log.info("Setup done: %s", run)


def _control_sheet(run: Path, kind: str, rig: str, cfg: dict, world: dict) -> None:
    """The start image next to control frames every 45 degrees of the rig's first arc."""
    import cv2
    ctrl = run / "controls" / kind / rig / "arc0"
    if not ctrl.exists():
        return
    tiles = [cv2.imread(str(run / "start" / f"{rig}.png"))]
    for j in range(0, cfg["frames"] - 1, max(1, (cfg["frames"] - 1) // 8)):
        img = cv2.imread(str(ctrl / f"frame_{j:04d}.png"))
        cv2.putText(img, f"{360.0 * j / (cfg['frames'] - 1):.0f} deg", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 200, 255), 2, cv2.LINE_AA)
        tiles.append(img)
    sheet = np.concatenate(tiles, 1)
    cv2.imwrite(str(run / f"controls_{kind}_{rig}.jpg"),
                cv2.resize(sheet, (sheet.shape[1] // 2, sheet.shape[0] // 2), interpolation=cv2.INTER_AREA))


def cmd_generate(args) -> None:
    from i2g.comfy_client import ComfyClient

    from .generate import GenSettings, generate_arcs
    from .orbit import ARCS, RIGS
    run = args.run
    cfg = _config(run)
    s = GenSettings.preset(args.preset, args.prompt or cfg["prompt"], width=cfg["width"], height=cfg["height"],
                           length=cfg["frames"], seed=args.seed, use_start_image=not args.no_start,
                           use_ref_image=not args.no_ref)
    n_arcs = len(ARCS[RIGS[args.rig]["arcs"]])
    controls = [run / "controls" / args.proxy / args.rig / f"arc{k}" for k in range(n_arcs)]
    missing = [str(c) for c in controls if not c.exists()]
    if missing:
        raise SystemExit(f"control frames missing (run setup with --proxies {args.proxy} --rigs {args.rig}): "
                         f"{missing}")
    start = run / "start" / f"{args.rig}.png"
    if not start.exists():
        start = run / "start" / "photo.png"
    client = ComfyClient()
    if not client.is_up():
        raise SystemExit("ComfyUI is not running: scripts\\v2_start_comfy.ps1 (and keep the GPU free of training)")
    out = run / "gen" / args.variant
    variant = {"proxy": args.proxy, "rig": args.rig, "preset": args.preset, "seed": args.seed,
               "start_image": not args.no_start, "ref_image": not args.no_ref}
    out.mkdir(parents=True, exist_ok=True)
    (out / "variant.json").write_text(json.dumps(variant, indent=1), encoding="utf-8")
    generate_arcs(client, s, controls, start, start, out)


def cmd_sheet(args) -> None:
    """CPU-only look at a generation: every n-th frame above its control frame (no matting needed)."""
    import cv2
    run = args.run
    variant = json.loads((run / "gen" / args.variant / "variant.json").read_text())
    rig = variant.get("rig") or variant["layout"]
    frames = sorted((run / "gen" / args.variant).glob("arc*/frame_*.png"))
    picks = frames[:: max(1, len(frames) // args.count)][: args.count]
    cols = []
    for f in picks:
        gen = cv2.imread(str(f))
        ctrl = cv2.imread(str(run / "controls" / variant["proxy"] / rig / f.parent.name / f.name))
        cv2.putText(gen, f"{f.parent.name}/{f.stem[-4:]}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
        cols.append(np.concatenate([gen, ctrl], 0))
    sheet = np.concatenate(cols, 1)
    out = run / "report" / f"{args.variant}_frames.jpg"
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), cv2.resize(sheet, (sheet.shape[1] // 2, sheet.shape[0] // 2), interpolation=cv2.INTER_AREA))
    print(out)


def cmd_prepare(args) -> None:
    from .dataset import prepare
    from .proxy import make_proxy
    run = args.run
    cfg = _config(run)
    world = _world(cfg)
    variant = json.loads((run / "gen" / args.variant / "variant.json").read_text())
    rig = variant.get("rig") or variant["layout"]
    control = make_proxy(variant["proxy"], world, Path(cfg["splat"]))
    splat = control if variant["proxy"] == "splat" else make_proxy("splat", world, Path(cfg["splat"]))
    from .orbit import rig_tag
    prepare(run / "gen" / args.variant, _cams(cfg, world, rig), run / "data" / (args.out or args.variant),
            Path(cfg["subject"]), world, control, splat, holdout_every=args.holdout_every,
            tag=args.tag or rig_tag(rig), crop=args.crop)


def cmd_train(args) -> None:
    from i2g.pipeline import parse_setting

    from .dataset import merge
    from .train import train_splat
    run = args.run
    cfg = _config(run)
    subject = Path(cfg["subject"])
    settings = [parse_setting(s) for s in args.set]
    variants = args.variant.split(",")
    boost = {}
    for item in args.boost or []:
        clip, _, factor = item.partition("=")
        boost[clip] = float(factor)
    data = run / "data" / variants[0]
    if len(variants) > 1:                        # several clips (body orbit + head close-up + ...) in one splat
        data = run / "data" / f"_{args.name}"
        merge([run / "data" / v for v in variants], data, boost)
    train_splat(data, run / "splats" / args.name, subject,
                subject / "experiments" / cfg["config_experiment"] / "report.json", settings)


def cmd_eval(args) -> None:
    from .evaluate import evaluate_splat
    run = args.run
    cfg = _config(run)
    report = json.loads((run / "splats" / args.name / "report.json").read_text())
    variant = args.variant or _variant_of(run, args.name, report)
    evaluate_splat(run / "splats" / args.name / "character.ply", run / "data" / variant, Path(cfg["subject"]),
                   run / "eval" / args.name)


def _variant_of(run: Path, name: str, report: dict) -> str:
    for d in (run / "data").iterdir():
        if (d / "qa.json").exists() and json.loads((d / "qa.json").read_text())["summary"]["holdout"] == report["holdout"]:
            return d.name
    raise SystemExit(f"cannot tell which dataset {name} was trained on; pass --variant")


def cmd_facecheck(args) -> None:
    from .facecheck import check
    run = args.run
    res = check(run, args.splats.split(","), args.variant.split(","), run / "report" / "facecheck")
    keys = ("psnr", "sharp", "speckle", "arcface", "arcface_worst")
    print("| splat | clip | " + " | ".join(keys) + " |")
    print("|---" * (len(keys) + 2) + "|")
    for name, r in res["splats"].items():
        for v, c in r["clips"].items():
            print(f"| {name} | {c['rig']} | " + " | ".join(str(c["mean"].get(k)) for k in keys) + " |")
        if len(r["clips"]) > 1:
            print(f"| **{name}** | **all** | " + " | ".join(f"**{r['mean'].get(k)}**" for k in keys) + " |")
    print("sheet:", run / "report" / "facecheck" / "facecheck.jpg")


def cmd_recolour(args) -> None:
    """Tint the face Gaussians the photo can see towards the photo, in place of a baked-on layer."""
    import torch
    from PIL import Image

    from i2g.body import load_world
    from i2g.evaluate import load_ply
    from i2g.priors import recolour_face_from_photo
    from i2g.splat import export_ply
    run = args.run
    cfg = _config(run)
    subject = Path(cfg["subject"])
    world = load_world(subject / "01_body")
    params = load_ply(run / "splats" / args.splat / "character.ply")
    rgba = torch.from_numpy(np.asarray(Image.open(subject / "00_ingest" / "source_hr_rgba.png").convert("RGBA"),
                                       np.float32) / 255).cuda()
    K = np.asarray(world["K"], np.float64).copy()
    K[0] *= rgba.shape[1] / world["size"][0]
    K[1] *= rgba.shape[0] / world["size"][1]
    out, stats = recolour_face_from_photo(params, world, rgba,
                                          torch.as_tensor(world["photo_w2c"], device="cuda").float(),
                                          torch.as_tensor(K, device="cuda").float(),
                                          radius_m=args.radius, strength=args.strength)
    dest = run / "splats" / args.name
    dest.mkdir(parents=True, exist_ok=True)
    export_ply(out, dest / "character.ply")
    (dest / "report.json").write_text(json.dumps({"recoloured_from": args.splat, "radius_m": args.radius,
                                                  "strength": args.strength, **stats}, indent=1), encoding="utf-8")
    log.info("Recoloured %d of %d face Gaussians from the photo -> %s", stats["recoloured"], stats["considered"],
             dest / "character.ply")


def cmd_compare(args) -> None:
    from i2g.evaluate import compare_sheet
    run = args.run
    cfg = _config(run)
    keys = [("lfo_orbit_psnr", "LFO dB"), ("lfo_fitted_psnr", "LFO fit dB"), ("lfo_orbit_vgg", "LFO vgg"),
            ("lfo_fit_move_cm_median", "fit cm"), ("photo_psnr", "photo dB"), ("photo_psnr_hr", "HR dB"),
            ("face_mean", "face id"), ("face_min", "face min"), ("needles_8x_3cm", "needles"),
            ("floater_pct", "floaters %"), ("sharpness", "sharp"), ("gaussians", "gaussians")]
    rows, sheets = [], []
    for base in (args.v2.split(",") if args.v2 else cfg["baselines"]):
        v2 = Path(cfg["subject"]) / "experiments" / base / "eval"
        if (v2 / "eval.json").exists():
            rows.append((f"v2 {base}", json.loads((v2 / "eval.json").read_text())))
            sheets.append((f"v2\n{base}", v2 / "sheet_row.png"))
    for name in args.splats.split(","):
        ev = run / "eval" / name / "eval.json"
        if ev.exists():
            rows.append((name, json.loads(ev.read_text())))
            sheets.append((name.replace("_", "\n", 1), ev.parent / "sheet_row.png"))
    lines = ["| splat | " + " | ".join(h for _, h in keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for name, e in rows:
        lines.append(f"| {name} | " + " | ".join("" if e.get(k) is None else str(e.get(k)) for k, _ in keys) + " |")
    table = "\n".join(lines)
    (run / "report").mkdir(exist_ok=True)
    (run / "report" / "compare.md").write_text(table + "\n", encoding="utf-8")
    compare_sheet(sheets, run / "report" / "compare.png")
    print(table)
    print("sheet:", run / "report" / "compare.png")


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m pacgs", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("setup")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--subject", type=Path, default=DEFAULT_SUBJECT, help="the v2 run folder of the photo")
    p.add_argument("--baselines", default=DEFAULT_BASELINES,
                   help="v2 experiments to compare against; the first one's splat is the splat proxy")
    p.add_argument("--config_exp", default=DEFAULT_CONFIG_EXP, help="v2 experiment whose training settings to use")
    p.add_argument("--splat", type=Path, help="splat proxy PLY (default: the first baseline's character.ply)")
    p.add_argument("--size", default="480x832")
    p.add_argument("--frames", type=int, default=81, help="frames per arc (4k+1)")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--proxies", default="splat,mesh")
    p.add_argument("--rigs", default="360,face,helix", help="camera rigs to render controls for (pacgs.orbit.RIGS)")
    p.add_argument("--fresh_ranges", action="store_true", help="recompute the depth normalisation")
    p = sub.add_parser("generate")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--proxy", default="splat", choices=["splat", "mesh"])
    p.add_argument("--rig", default="360", help="360 (body orbit), face (head close-up), helix (rising orbit), 2x180")
    p.add_argument("--preset", default="lightning", choices=["lightning", "quality"])
    p.add_argument("--seed", type=int, default=1001)
    p.add_argument("--prompt")
    p.add_argument("--no_start", action="store_true", help="no start image (reference image only)")
    p.add_argument("--no_ref", action="store_true", help="no reference image")
    p = sub.add_parser("sheet")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--count", type=int, default=12)
    p = sub.add_parser("prepare")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--holdout_every", type=int, default=8)
    p.add_argument("--tag", help="view-name prefix (default: from the rig); clips trained together need different ones")
    p.add_argument("--out", help="dataset name under data/ (default: the variant), so one clip can be "
                                 "prepared twice, e.g. cropped and uncropped")
    p.add_argument("--crop", type=float, help="train only on a box of this many head radii around the head "
                                              "(1.8 keeps the head, the hair and a little shoulder); the torso "
                                              "of a head close-up is where each clip invents its own version of "
                                              "what the photo never saw")
    p = sub.add_parser("train")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--variant", required=True, help="one prepared variant, or several comma-separated to merge")
    p.add_argument("--boost", action="append", metavar="CLIP=FACTOR",
                   help="give a clip more of the training steps, e.g. --boost f1_face_l4=3 (repeatable)")
    p.add_argument("--name", required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p = sub.add_parser("eval")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--variant")
    p = sub.add_parser("facecheck")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--splats", required=True, help="comma-separated splat names under splats/")
    p.add_argument("--variant", default="f1_face_l4", help="head close-up clip(s) to score against, comma-separated")
    p = sub.add_parser("recolour")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--splat", required=True, help="splat to recolour")
    p.add_argument("--name", required=True, help="name for the recoloured copy")
    p.add_argument("--radius", type=float, default=0.10)
    p.add_argument("--strength", type=float, default=0.85)
    p = sub.add_parser("compare")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--splats", required=True)
    p.add_argument("--v2", help="comma-separated v2 experiments (default: the run's baselines)")
    args = ap.parse_args()
    setup_logging(args.run / "logs" / f"{args.cmd}_{time.strftime('%Y%m%d-%H%M%S')}.log")
    {"setup": cmd_setup, "generate": cmd_generate, "sheet": cmd_sheet, "prepare": cmd_prepare, "train": cmd_train,
     "eval": cmd_eval, "facecheck": cmd_facecheck, "recolour": cmd_recolour,
     "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()

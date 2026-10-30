"""python -m headpass <command> ... (see headpass/experiment.py)"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from i2g.log import setup_logging


def _overrides(items: list[str]) -> dict:
    from i2g.pipeline import parse_setting
    return dict(parse_setting(s) for s in items)


def main() -> None:
    ap = argparse.ArgumentParser(prog="headpass")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup"); s.add_argument("--src", type=Path, required=True); s.add_argument("--exp", type=Path, required=True)
    s = sub.add_parser("standin"); s.add_argument("--exp", type=Path, required=True); s.add_argument("--kind", default="coherent")
    s.add_argument("--band_m", type=float, default=0.02); s.add_argument("--face_shift_m", type=float, default=0.0)
    s.add_argument("--skin_smooth", type=float, default=0.0, help="v3: extra smoothing (bins) on skin-coloured scalp")
    s.add_argument("--close", type=int, default=0, help="v4: fill grooves across the head narrower than 2k+1 bins")
    s = sub.add_parser("generate"); s.add_argument("--exp", type=Path, required=True); s.add_argument("--set", required=True)
    s.add_argument("--standin", required=True); s.add_argument("--rigs", default=""); s.add_argument("--seed", type=int)
    s = sub.add_parser("prepare"); s.add_argument("--exp", type=Path, required=True); s.add_argument("--set", required=True)
    s = sub.add_parser("train"); s.add_argument("--exp", type=Path, required=True); s.add_argument("--set", required=True)
    s.add_argument("--arm", required=True); s.add_argument("--o", action="append", default=[], help="key=value SplatConfig override")
    s = sub.add_parser("head"); s.add_argument("--exp", type=Path, required=True); s.add_argument("--set", required=True)
    s.add_argument("--arm", required=True); s.add_argument("--standin", default="coherent2")
    s.add_argument("--iterations", type=int, default=30000); s.add_argument("--leash", type=float, default=0.0)
    s.add_argument("--ramp", type=float, default=0.6); s.add_argument("--o", action="append", default=[])
    s = sub.add_parser("merge"); s.add_argument("--exp", type=Path, required=True); s.add_argument("--body", type=Path, required=True)
    s.add_argument("--head", type=Path, required=True); s.add_argument("--arm", required=True); s.add_argument("--standin", default="coherent2")
    s = sub.add_parser("score"); s.add_argument("--exp", type=Path, required=True); s.add_argument("--arms", required=True,
                                                                                            help="name=ply,name=ply")
    s.add_argument("--set", default=None); s.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    from . import experiment as ex
    setup_logging(Path("output/headpass/logs") / f"{a.cmd}.log")
    if a.cmd == "setup":
        ex.setup(a.src, a.exp)
    elif a.cmd == "standin":
        ex.standin(a.exp, a.kind, **({} if a.kind == "old" else {"band_m": a.band_m, "face_shift_m": a.face_shift_m,
                                                                 "skin_smooth_bins": a.skin_smooth, "close_bins": a.close}))
    elif a.cmd == "generate":
        ex.generate(a.exp, a.set, a.standin, [r for r in a.rigs.split(",") if r] or None, a.seed)
    elif a.cmd == "prepare":
        print(json.dumps({k: {kk: v[kk] for kk in ("iou_control_median", "iou_splat_median", "face_id_median", "restyled")}
                          for k, v in ex.prepare_set(a.exp, a.set).items()}, indent=1))
    elif a.cmd == "train":
        ex.train_arm(a.exp, a.arm, a.set, _overrides(a.o))
    elif a.cmd == "head":
        from .headtrain import train_head
        train_head(a.exp, a.arm, a.set, iterations=a.iterations, leash_weight=a.leash, perceptual_ramp_to=a.ramp,
                   overrides=_overrides(a.o), standin=a.standin)
    elif a.cmd == "merge":
        from .merge import merge
        out = a.exp / "arms" / a.arm
        st = merge(a.body, a.head, a.exp / "standins" / f"{a.standin}_maps.npz", out / "character.ply")
        (out / "merge.json").write_text(json.dumps(st, indent=1), encoding="utf-8")
    elif a.cmd == "score":
        from .evaluate import score
        arms = dict(x.split("=", 1) for x in a.arms.split(","))
        res = score(a.exp, {k: Path(v) for k, v in arms.items()}, a.set)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

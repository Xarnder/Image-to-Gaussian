"""A frozen copy of the v2 run data a PAC-GS run depends on, in the v2 folder layout (so i2g.evaluate works
on it unchanged): the body fit, the photo (canvas, matte, native resolution), the accepted registered views
(only for the visual hull of the floater metric) and the baseline v2 splat. Later v2 runs in the same
folder cannot then shift the world frame or cameras under an existing PAC-GS run."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from i2g.log import get_logger

log = get_logger("pacgs.subject")


def snapshot(v2_dir: Path, dest: Path, baselines: list[str], config_experiment: str) -> Path:
    """baselines: v2 experiments to compare against (splat + scores; the first is the splat proxy);
    config_experiment: the v2 experiment whose training settings PAC-GS splats start from."""
    if (dest / "01_body" / "body_world.npz").exists():
        return dest
    for sub, names in (("01_body", ["body_world.npz", "body_summary.json"]),
                       ("00_ingest", ["canvas.png", "alpha.png", "source_hr_rgba.png", "ingest.json"])):
        (dest / sub).mkdir(parents=True, exist_ok=True)
        for name in names:
            if (v2_dir / sub / name).exists():
                shutil.copy2(v2_dir / sub / name, dest / sub / name)
    qa = dest / "03_qa"
    (qa / "views").mkdir(parents=True, exist_ok=True)
    (qa / "masks").mkdir(exist_ok=True)
    kept = []
    for r in json.loads((v2_dir / "03_qa" / "registration.json").read_text()):
        if r["tag"] == "photo":
            r = r | {"path": str(dest / "00_ingest" / "canvas.png"), "mask": str(dest / "00_ingest" / "alpha.png")}
        elif r.get("accepted"):
            name = f"{r['tag']}_s{r['seed']}.png"
            shutil.copy2(r["path"], qa / "views" / name)
            shutil.copy2(r["mask"], qa / "masks" / name)
            r = r | {"path": str(qa / "views" / name), "mask": str(qa / "masks" / name)}
        else:
            continue
        kept.append(r)
    (qa / "registration.json").write_text(json.dumps(kept, indent=1), encoding="utf-8")
    for name in dict.fromkeys(baselines + [config_experiment]):
        src, exp = v2_dir / "experiments" / name, dest / "experiments" / name
        (exp / "eval").mkdir(parents=True, exist_ok=True)
        for rel in ("character.ply", "report.json", "eval/eval.json", "eval/sheet_row.png"):
            if (src / rel).exists() and (name in baselines or rel == "report.json"):
                shutil.copy2(src / rel, exp / rel)
    log.info("Snapshot of %s (%d registered views, baselines %s, settings from %s) in %s", v2_dir, len(kept),
             ", ".join(baselines), config_experiment, dest)
    return dest

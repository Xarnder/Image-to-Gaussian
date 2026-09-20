#!/usr/bin/env python3
r"""
Batch Image-to-Gaussian: a folder of photos -> one 3D Gaussian splat per photo, one at a time.

Each photo is a separate `image_to_splat.py` run in its own process, so one bad photo cannot take the
batch down and the GPU is fully released between characters. ComfyUI is started once and kept for the
whole batch. Photos that already have a finished `character.ply` are skipped, so stopping and starting
again continues where you left off.

EXAMPLE (VS Code terminal, in D:\Image-to-Gaussian):
    .\.venv-i2g\Scripts\activate
    python3 batch_to_splat.py --input D:\photos\characters

MORE EXAMPLES
    python3 batch_to_splat.py -i D:\photos\characters -o D:\splats     # one subfolder per photo
    python3 batch_to_splat.py -i D:\photos\front.png                   # a single photo also works
    python3 batch_to_splat.py -i D:\photos --recursive                 # include subfolders
    python3 batch_to_splat.py -i D:\photos --preset draft              # rough results, ~20 min each
    python3 batch_to_splat.py -i D:\photos --dry_run                   # list what it would do
    python3 batch_to_splat.py -i D:\photos --redo                      # rebuild finished ones too
    python3 batch_to_splat.py -i D:\photos --no_orbit                  # any image_to_splat.py option
                                                                       # is passed through to every run

Or double-click run_batch_to_splat.bat, or drag a folder (or a photo) onto it.

Stopping: Ctrl-C stops the batch and the photo it is on. To stop *between* photos instead, create a
file called `stop.txt` in the input folder; the current photo finishes and the batch ends cleanly.

Result per photo: <output_dir>\<photo name>\v2\final\character.ply (default: output\<photo name>\v2\...),
plus batch_summary.json next to them with the status and duration of every photo.
"""
from __future__ import annotations

import argparse
import json
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
from image_to_splat import IMAGE_TYPES, PRESETS  # noqa: E402  (light: no torch, no models)

SINGLE = REPO_ROOT / "image_to_splat.py"
STOP_FILE = "stop.txt"
# options that this script sets per photo, or that make no sense for a batch
RESERVED = {"-i", "--input", "-o", "--output_dir", "-w", "--working_dir", "--name", "--stop_comfy", "--check"}
# rough wall-clock per photo, for the "about N hours" estimate only (RTX 4090, models cached)
PRESET_MINUTES = {"draft": 20, "standard": 180, "high": 210}


def find_photos(where: Path, recursive: bool) -> list[Path]:
    """Every supported image under `where`, sorted the way a file manager would show them."""
    if where.is_file():
        return [where]
    files = (where.rglob("*") if recursive else where.glob("*"))
    return sorted((f for f in files if f.is_file() and f.suffix.lower() in IMAGE_TYPES),
                  key=lambda f: (str(f.parent).lower(), f.name.lower()))


def run_names(photos: list[Path]) -> dict[Path, str]:
    """A run name per photo: the file stem, or <parent>_<stem> where two photos share a stem."""
    stems: dict[str, int] = {}
    for p in photos:
        stems[p.stem.lower()] = stems.get(p.stem.lower(), 0) + 1
    names = {}
    for p in photos:
        name = p.stem if stems[p.stem.lower()] == 1 else f"{p.parent.name}_{p.stem}"
        names[p] = re.sub(r'[<>:"/\\|?*]', "_", name).strip(". ") or "photo"
    return names


def fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h} h {m:02d} min" if h else f"{m} min {s:02d} s"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Turn a folder of photos into one Gaussian splat each, one after another.",
        epilog="Any other option is passed through to image_to_splat.py, for example --no_orbit or "
               "--set iterations=50000. Run image_to_splat.py --help to see them.")
    ap.add_argument("-i", "--input", type=Path, required=True,
                    help="folder of photos, or a single photo")
    ap.add_argument("-o", "--output_dir", type=Path,
                    help=r"folder to hold one subfolder per photo (default: output\<photo name>\v2)")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="standard",
                    help="quality preset for every photo (default: standard, the best pipeline)")
    ap.add_argument("--recursive", action="store_true", help="also take photos from subfolders")
    ap.add_argument("--redo", action="store_true",
                    help="do not skip photos that already have a character.ply (each run still redoes only "
                         "the stages whose settings changed; add --force to rebuild regardless)")
    ap.add_argument("--limit", type=int, help="stop after this many photos (the rest stay for the next run)")
    ap.add_argument("--dry_run", action="store_true", help="list what would run, then exit")
    args, extra = ap.parse_known_args(argv)

    bad = sorted(RESERVED.intersection(a.split("=")[0] for a in extra))
    if bad:
        ap.error(f"{', '.join(bad)} is set per photo by this script; use --input / --output_dir instead")

    where = args.input.expanduser().resolve()
    if not where.exists():
        ap.error(f"not found: {where}")
    photos = find_photos(where, args.recursive)
    if not photos:
        ap.error(f"no photos in {where} (looked for {', '.join(sorted(IMAGE_TYPES))}"
                 f"{'' if args.recursive else '; add --recursive for subfolders'})")
    names = run_names(photos)
    base = args.output_dir.expanduser().resolve() if args.output_dir else REPO_ROOT / "output"
    # --from_stage or --force means the operator wants stages redone, so "already finished" cannot skip
    rerunning = args.redo or any(a.split("=")[0] in ("--from_stage", "--only_stage", "--force") for a in extra)
    stop_file = (where if where.is_dir() else where.parent) / STOP_FILE

    plan = []
    for photo in photos:
        work = base / names[photo] / "v2"
        done = (work / "final" / "character.ply").exists()
        plan.append({"photo": str(photo), "name": names[photo], "work": str(work),
                     "status": "done" if (done and not rerunning) else "todo"})
    todo = [p for p in plan if p["status"] == "todo"]
    if args.limit:
        for p in todo[args.limit:]:
            p["status"] = "not reached"
        todo = todo[:args.limit]

    print(f"Batch: {len(photos)} photo(s) in {where}")
    print(f"  preset {args.preset}{' ' + ' '.join(extra) if extra else ''}")
    print(f"  results in {base}\\<name>\\v2\\final")
    root = where if where.is_dir() else where.parent
    for p in plan:
        mark = {"done": "skip (already built)", "todo": "build", "not reached": "later (--limit)"}[p["status"]]
        print(f"  {mark:22s} {p['name']:28s} <- {Path(p['photo']).relative_to(root)}")
    if not todo:
        print("Nothing to do. Use --redo to rebuild the finished ones.")
        return 0
    print(f"\n{len(todo)} to build, about {fmt(len(todo) * PRESET_MINUTES.get(args.preset, 180) * 60)} in total.")
    print(f"Stop between photos by creating {stop_file}; Ctrl-C stops immediately.\n")
    if args.dry_run:
        return 0

    base.mkdir(parents=True, exist_ok=True)
    summary_path = base / "batch_summary.json"
    started = time.perf_counter()
    interrupted = False
    for index, item in enumerate(todo, 1):
        if stop_file.exists():
            print(f"[batch] {STOP_FILE} found: stopping. Delete it and run again to continue.")
            for rest in todo[index - 1:]:
                rest["status"] = "not reached"
            break
        left = f", about {fmt((len(todo) - index + 1) * ((time.perf_counter() - started) / max(index - 1, 1)))} left" \
            if index > 1 else ""
        print(f"\n{'=' * 78}\n[batch] {index}/{len(todo)}  {Path(item['photo']).name}{left}\n{'=' * 78}", flush=True)
        cmd = [sys.executable, str(SINGLE), "--input", item["photo"], "--name", item["name"],
               "--working_dir", item["work"], "--preset", args.preset, "--keep_comfy", *extra]
        t0 = time.perf_counter()
        try:
            code = subprocess.call(cmd)
        except KeyboardInterrupt:
            code, interrupted = -signal.SIGINT, True
        item["seconds"] = round(time.perf_counter() - t0, 1)
        item["exit_code"] = code
        ply = Path(item["work"]) / "final" / "character.ply"
        if ply.exists():
            item["ply"] = str(ply)
            item["ply_mb"] = round(ply.stat().st_size / 1024 ** 2, 1)
        if interrupted:
            item["status"] = "stopped"
        elif code != 0:
            item["status"] = "failed"
        else:                     # --only_stage and --from_stage can finish without producing a splat
            item["status"] = "built" if ply.exists() else "ran, no character.ply"
        summary_path.write_text(json.dumps({"input": str(where), "preset": args.preset, "extra": extra,
                                            "photos": plan}, indent=1), encoding="utf-8")
        print(f"[batch] {item['status']} in {fmt(item['seconds'])}: {Path(item['photo']).name}", flush=True)
        if interrupted:
            break

    if not any(a == "--keep_comfy" for a in extra):
        subprocess.call([sys.executable, str(SINGLE), "--stop_comfy"])

    built = [p for p in plan if p["status"] in ("built", "ran, no character.ply")]
    failed = [p for p in plan if p["status"] == "failed"]
    print(f"\n{'=' * 78}\n[batch] {len(built)} built, {len(failed)} failed, "
          f"{sum(1 for p in plan if p['status'] == 'done')} already there, "
          f"{sum(1 for p in plan if p['status'] == 'not reached')} left, in {fmt(time.perf_counter() - started)}")
    for p in built:
        size = f"{p['ply_mb']} MB" if "ply_mb" in p else "no character.ply"
        print(f"  ok      {p['name']:28s} {fmt(p['seconds']):>12s}  {size}")
    for p in failed:
        print(f"  FAILED  {p['name']:28s} exit {p['exit_code']} - see {Path(p['work']) / 'logs'}")
    print(f"[batch] summary: {summary_path}")
    if interrupted:
        return 130
    return min(len(failed), 125)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[batch] stopped by Ctrl-C. Run the same command again to continue where it left off.")
        sys.exit(130)

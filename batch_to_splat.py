#!/usr/bin/env python3
r"""
Batch Image-to-Gaussian: a folder of photos -> one 3D Gaussian splat per photo, one at a time.

Each photo is a separate `image_to_splat.py` run in its own process, so one bad photo cannot take the
batch down and the GPU is fully released between characters. ComfyUI is started once and kept for the
whole batch. Photos that already have a finished `<photo>-and-<experiment>-character.ply` are skipped, so stopping and
starting again continues where you left off.

EXAMPLE (VS Code terminal, in D:\Image-to-Gaussian):
    .\.venv-i2g\Scripts\activate
    python3 batch_to_splat.py --input D:\photos\characters --experiment 3

MORE EXAMPLES
    python3 batch_to_splat.py -i D:\photos\characters -e face-v2
    python3 batch_to_splat.py -i D:\photos\characters -e 3 -o D:\splats
    python3 batch_to_splat.py -i D:\photos\front.png -e 3
    python3 batch_to_splat.py -i D:\photos -e 3 --recursive
    python3 batch_to_splat.py -i D:\photos -e 3 --preset draft
    python3 batch_to_splat.py -i D:\photos -e 3 --dry_run
    python3 batch_to_splat.py -i D:\photos -e 3 --redo
    python3 batch_to_splat.py -i D:\photos -e 3 --no_orbit

Or double-click run_batch_to_splat.bat, or drag a folder (or a photo) onto it.

Stopping: Ctrl-C (or the GUI **Stop now**) stops the batch and the photo it is on, then frees the GPU.
The unfinished photo is kept; run the same command again to continue that character, then the rest.
To stop *after* the current photo instead, create `stop.txt` in the input folder (or the GUI
**Stop after this one**); that photo finishes, ComfyUI is stopped, and a later Start continues the rest.

Result per photo: output\<experiment>\<photo name>\v2\final\<photo name>-and-<experiment>-character.ply
(default; or under --output_dir\<experiment>\...), plus batch_summary.json in the experiment folder.
The same <photo>-and-<experiment>- prefix is used for the turntable, previews, contact sheet and reports.
"""
from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
from image_to_splat import IMAGE_TYPES, PRESETS, resolve_experiment  # noqa: E402
from i2g.paths import find_deliverable, safe_run_name  # noqa: E402

SINGLE = REPO_ROOT / "image_to_splat.py"
STOP_FILE = "stop.txt"
STOP_AFTER_EXIT = 129  # current photo finished; remaining photos wait for the next run
# Windows STATUS_CONTROL_C_EXIT when Ctrl-C / Ctrl-Break kills the process
WIN_CTRL_C_EXIT = 0xC000013A
# options that this script sets per photo, or that make no sense for a batch
RESERVED = {"-i", "--input", "-o", "--output_dir", "-w", "--working_dir", "--name",
            "-e", "--experiment", "--stop_comfy", "--check"}
# rough wall-clock per photo, for the "about N hours" estimate only (RTX 4090, models cached)
PRESET_MINUTES = {"draft": 20, "standard": 200, "high": 230}     # standard/high include the ~20 min head pass


def find_photos(where: Path, recursive: bool) -> list[Path]:
    """Every supported image under `where`, sorted the way a file manager would show them."""
    if where.is_file():
        return [where]
    files = (where.rglob("*") if recursive else where.glob("*"))
    return sorted((f for f in files if f.is_file() and f.suffix.lower() in IMAGE_TYPES),
                  key=lambda f: (str(f.parent).lower(), f.name.lower()))


def photo_rel(photo: Path, root: Path) -> str:
    """Path of `photo` relative to the input folder, for display and --order files."""
    try:
        return str(photo.resolve().relative_to(root.resolve()))
    except ValueError:
        return photo.name


def _order_key(text: str) -> str:
    return text.replace("\\", "/").strip().strip("/").casefold()


def _photo_id(photo: Path) -> str:
    try:
        return str(photo.resolve()).replace("\\", "/").casefold()
    except OSError:
        return str(photo).replace("\\", "/").casefold()


def load_order_file(path: Path) -> list[str]:
    """One photo per line; `#` comments and blank lines ignored. Accepts a UTF-8 BOM."""
    if not path.is_file():
        raise FileNotFoundError(f"order file not found: {path}")
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip().strip('"').strip("'")
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def apply_photo_order(photos: list[Path], order: list[str], root: Path) -> tuple[list[Path], list[str]]:
    """Put `photos` in the sequence listed in `order`.

    Each entry may be a path relative to `root`, an absolute path under `root`, a unique
    filename, or a unique stem. Entries outside the folder, unknown names, and ambiguous
    names are skipped (warning). Duplicates are ignored. Photos not mentioned stay at the
    end, in their original order, so a partial list cannot silently drop work.
    """
    root = root.resolve()
    photos = list(photos)
    by_rel: dict[str, Path] = {}
    by_name: dict[str, list[Path]] = {}
    by_stem: dict[str, list[Path]] = {}
    for photo in photos:
        by_rel[_order_key(photo_rel(photo, root))] = photo
        by_name.setdefault(photo.name.casefold(), []).append(photo)
        by_stem.setdefault(photo.stem.casefold(), []).append(photo)

    out: list[Path] = []
    seen: set[str] = set()
    warnings: list[str] = []

    def take(photo: Path | None, raw: str, why_missing: str) -> None:
        if photo is None:
            warnings.append(why_missing)
            return
        ident = _photo_id(photo)
        if ident in seen:
            return
        seen.add(ident)
        out.append(photo)

    for raw in order:
        raw = raw.strip().strip('"').strip("'")
        if not raw or raw.startswith("#"):
            continue
        looks_abs = Path(raw).is_absolute() or (len(raw) >= 3 and raw[1] == ":" and raw[0].isalpha())
        if looks_abs:
            try:
                resolved = Path(raw).expanduser().resolve()
                rel = resolved.relative_to(root)
            except ValueError:
                warnings.append(f"ignored (outside folder): {raw}")
                continue
            except (OSError, RuntimeError):
                warnings.append(f"ignored (bad path): {raw}")
                continue
            take(by_rel.get(_order_key(str(rel))), raw, f"not a photo in this folder: {raw}")
            continue
        key = _order_key(raw)
        photo = by_rel.get(key)
        if photo is not None:
            take(photo, raw, "")
            continue
        name = Path(raw.replace("\\", "/")).name
        named = by_name.get(name.casefold(), [])
        if len(named) == 1:
            take(named[0], raw, "")
            continue
        if len(named) > 1:
            warnings.append(f"ambiguous name (use a relative path): {raw}")
            continue
        stemmed = by_stem.get(name.casefold(), [])
        if len(stemmed) == 1:
            take(stemmed[0], raw, "")
            continue
        if len(stemmed) > 1:
            warnings.append(f"ambiguous name (use a relative path): {raw}")
            continue
        warnings.append(f"not in this folder: {raw}")

    for photo in photos:
        ident = _photo_id(photo)
        if ident not in seen:
            seen.add(ident)
            out.append(photo)
    return out, warnings


def run_names(photos: list[Path]) -> dict[Path, str]:
    """A run name per photo: the file stem, or <parent>_<stem> where two photos share a stem."""
    stems: dict[str, int] = {}
    for p in photos:
        stems[p.stem.lower()] = stems.get(p.stem.lower(), 0) + 1
    names = {}
    for p in photos:
        name = p.stem if stems[p.stem.lower()] == 1 else f"{p.parent.name}_{p.stem}"
        names[p] = safe_run_name(name)
    # Same-stem files with different extensions (or sanitized names) can still collide.
    import hashlib
    counts = {}
    for name in names.values():
        counts[name.casefold()] = counts.get(name.casefold(), 0) + 1
    for p, name in list(names.items()):
        if counts[name.casefold()] > 1:
            names[p] = name + "_" + hashlib.sha256(str(p.resolve()).encode()).hexdigest()[:10]
    return names


def fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h} h {m:02d} min" if h else f"{m} min {s:02d} s"


def is_interrupt_code(code: int) -> bool:
    """True when a child exited because of Ctrl-C / Stop now, so the batch must not start the next photo."""
    if code in (-2, -signal.SIGINT, 130):
        return True
    return (code & 0xFFFFFFFF) == WIN_CTRL_C_EXIT


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Turn a folder of photos into one Gaussian splat each, one after another.",
        epilog="Any other option is passed through to image_to_splat.py, for example --no_orbit or "
               "--set iterations=50000. Run image_to_splat.py --help to see them.")
    ap.add_argument("-i", "--input", type=Path, required=True,
                    help="folder of photos, or a single photo")
    ap.add_argument("-e", "--experiment",
                    help="name or number of this experiment; results go under output\\<experiment>\\ "
                         "(asked for if omitted)")
    ap.add_argument("-o", "--output_dir", type=Path,
                    help=r"parent folder for experiments (default: output). Each run is "
                         r"output_dir\<experiment>\<photo>\v2")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="standard",
                    help="quality preset for every photo (default: standard, the best pipeline)")
    ap.add_argument("--recursive", action="store_true", help="also take photos from subfolders")
    ap.add_argument("--separate-ply", action="store_true", help="keep final PLY files in individual project folders")
    ap.add_argument("--project-names", type=Path, help="Reviewed image path to project name mapping")
    ap.add_argument("--order", type=Path,
                    help="text file of photos in the order to build them (relative to --input or absolute, "
                         "one per line). Photos not listed stay at the end. Default is A–Z")
    ap.add_argument("--redo", action="store_true",
                    help="do not skip photos that already have a finished splat (each run still redoes only "
                         "the stages whose settings changed; add --force to rebuild regardless)")
    ap.add_argument("--limit", type=int, help="stop after this many photos (the rest stay for the next run)")
    ap.add_argument("--dry_run", action="store_true", help="list what would run, then exit")
    ap.add_argument("--stop_file", type=Path,
                    help="if this file appears during the batch, finish the current photo then stop "
                         f"(default: {STOP_FILE} in the input folder)")
    args, extra = ap.parse_known_args(argv)

    bad = sorted(RESERVED.intersection(a.split("=")[0] for a in extra))
    if bad:
        ap.error(f"{', '.join(bad)} is set per photo by this script; use --input / --output_dir / --experiment instead")

    where = args.input.expanduser().resolve()
    if not where.exists():
        ap.error(f"not found: {where}")
    photos = find_photos(where, args.recursive)
    if not photos:
        ap.error(f"no photos in {where} (looked for {', '.join(sorted(IMAGE_TYPES))}"
                 f"{'' if args.recursive else '; add --recursive for subfolders'})")
    folder = where if where.is_dir() else where.parent
    order_path = None
    order_warn: list[str] = []
    if args.order:
        order_path = args.order.expanduser().resolve()
        try:
            wanted = load_order_file(order_path)
        except OSError as exc:
            ap.error(str(exc))
        photos, order_warn = apply_photo_order(photos, wanted, folder)
    from i2g.paths import reviewed_names
    names = reviewed_names(photos, json.loads(args.project_names.read_text(encoding="utf-8"))) if args.project_names else run_names(photos)
    experiment = resolve_experiment(args.experiment, required=True, parser=ap)
    root = args.output_dir.expanduser().resolve() if args.output_dir else REPO_ROOT / "output"
    base = root / experiment
    # --from_stage or --force means the operator wants stages redone, so "already finished" cannot skip
    rerunning = args.redo or any(a.split("=")[0] in ("--from_stage", "--only_stage", "--force") for a in extra)
    stop_file = args.stop_file.expanduser().resolve() if args.stop_file else folder / STOP_FILE

    plan = []
    for photo in photos:
        work = base / names[photo] / "v2"
        from i2g.batch_outputs import organize_ply
        existing = find_deliverable(work / "final", names[photo], "character.ply", experiment=experiment)
        done = existing is not None
        if done and not rerunning and not args.dry_run:
            organize_ply(existing, work / 'final', base / 'final_ply', not args.separate_ply)
        plan.append({"photo": str(photo), "name": names[photo], "work": str(work),
                     "status": "done" if (done and not rerunning) else "todo"})
    todo = [p for p in plan if p["status"] == "todo"]
    if args.limit:
        for p in todo[args.limit:]:
            p["status"] = "not reached"
        todo = todo[:args.limit]

    print(f"Batch: {len(photos)} photo(s) in {where}")
    print(f"  experiment {experiment}")
    print(f"  preset {args.preset}{' ' + ' '.join(extra) if extra else ''}")
    if order_path is not None:
        print(f"  order from {order_path}")
        for msg in order_warn:
            print(f"  [order] {msg}")
    print(f"  results in {base}\\<name>\\v2\\final\\<name>-and-{experiment}-character.ply")
    root = where if where.is_dir() else where.parent
    for p in plan:
        mark = {"done": "skip (already built)", "todo": "build", "not reached": "later (--limit)"}[p["status"]]
        print(f"  {mark:22s} {p['name']:28s} <- {Path(p['photo']).relative_to(root)}")
    if not todo:
        print("Nothing to do. Use --redo to rebuild the finished ones.")
        return 0
    print(f"\n{len(todo)} to build, about {fmt(len(todo) * PRESET_MINUTES.get(args.preset, 180) * 60)} in total.")
    print(f"Stop after the current photo: create {stop_file}")
    print("Stop immediately (and free the GPU): Ctrl-C. Start again without --redo to continue.\n")
    if args.dry_run:
        return 0

    base.mkdir(parents=True, exist_ok=True)
    summary_path = base / "batch_summary.json"
    if stop_file.exists():
        stop_file.unlink()
        print(f"[batch] removed leftover {stop_file} from a previous stop")
    started = time.perf_counter()
    interrupted = False
    stopped_after = False
    for index, item in enumerate(todo, 1):
        if stop_file.exists():
            left = todo[index - 1:]
            for rest in left:
                rest["status"] = "not reached"
            names_left = [Path(x["photo"]).name for x in left]
            shown = ", ".join(names_left[:8]) + ("…" if len(names_left) > 8 else "")
            print(f"[batch] stop after current: not starting {len(left)} remaining ({shown}). "
                  f"Start again without --redo to continue.", flush=True)
            stop_file.unlink(missing_ok=True)
            stopped_after = True
            break
        left = f", about {fmt((len(todo) - index + 1) * ((time.perf_counter() - started) / max(index - 1, 1)))} left" \
            if index > 1 else ""
        print(f"\n{'=' * 78}\n[batch] {index}/{len(todo)}  {Path(item['photo']).name}{left}\n{'=' * 78}", flush=True)
        cmd = [sys.executable, str(SINGLE), "--input", item["photo"], "--name", item["name"],
               "--experiment", experiment, "--working_dir", item["work"], "--preset", args.preset,
               "--keep_comfy", *extra]
        t0 = time.perf_counter()
        try:
            code = subprocess.call(cmd)
        except KeyboardInterrupt:
            code, interrupted = -signal.SIGINT, True
        if is_interrupt_code(code):
            interrupted = True
        item["seconds"] = round(time.perf_counter() - t0, 1)
        item["exit_code"] = code
        ply = find_deliverable(Path(item["work"]) / "final", item["name"], "character.ply",
                               experiment=experiment)
        if ply is not None:
            if code == 0:
                ply = organize_ply(ply, Path(item['work']) / 'final', base / 'final_ply', not args.separate_ply)
            item["ply"] = str(ply)
            item["ply_mb"] = round(ply.stat().st_size / 1024 ** 2, 1)
        if interrupted:
            item["status"] = "stopped"
        elif code != 0:
            item["status"] = "failed"
        else:                     # --only_stage and --from_stage can finish without producing a splat
            item["status"] = "built" if ply is not None else "ran, no splat"
        reason = "interrupt" if interrupted else None
        summary_path.write_text(json.dumps({"input": str(where), "experiment": experiment,
                                            "preset": args.preset, "extra": extra, "stop": reason,
                                            "photos": plan}, indent=1), encoding="utf-8")
        print(f"[batch] {item['status']} in {fmt(item['seconds'])}: {Path(item['photo']).name}", flush=True)
        if interrupted:
            print("[batch] stopped immediately. Finished stages of this photo are kept; "
                  "Start again without --redo to continue, then the rest of the batch.", flush=True)
            break

    stop_file.unlink(missing_ok=True)
    reason = "after_current" if stopped_after else ("interrupt" if interrupted else None)
    summary_path.write_text(json.dumps({"input": str(where), "experiment": experiment,
                                        "preset": args.preset, "extra": extra, "stop": reason,
                                        "photos": plan}, indent=1), encoding="utf-8")
    if not any(a == "--keep_comfy" for a in extra):
        subprocess.call([sys.executable, str(SINGLE), "--stop_comfy"])

    built = [p for p in plan if p["status"] in ("built", "ran, no splat")]
    failed = [p for p in plan if p["status"] == "failed"]
    print(f"\n{'=' * 78}\n[batch] {len(built)} built, {len(failed)} failed, "
          f"{sum(1 for p in plan if p['status'] == 'done')} already there, "
          f"{sum(1 for p in plan if p['status'] == 'not reached')} left, in {fmt(time.perf_counter() - started)}")
    for p in built:
        size = f"{p['ply_mb']} MB" if "ply_mb" in p else "no splat"
        print(f"  ok      {p['name']:28s} {fmt(p['seconds']):>12s}  {size}")
    for p in failed:
        print(f"  FAILED  {p['name']:28s} exit {p['exit_code']} - see {Path(p['work']) / 'logs'}")
    print(f"[batch] summary: {summary_path}")
    if interrupted:
        return 130
    if stopped_after:
        return STOP_AFTER_EXIT
    return min(len(failed), 125)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[batch] stopped by Ctrl-C. Run the same command again to continue where it left off.")
        sys.exit(130)

#!/usr/bin/env python3
r"""
Image-to-Gaussian: one front-facing, full-body photo -> one static 3D Gaussian splat (.ply).

EXAMPLE (VS Code terminal, in D:\Image-to-Gaussian):
    .\.venv-i2g\Scripts\activate
    python3 image_to_splat.py --input input\front.png --output_dir output\my_splat

MORE EXAMPLES
    python3 image_to_splat.py -i input\front.png --preset draft          # quick rough result (~20 min)
    python3 image_to_splat.py -i input\front.png --preset high           # more views, longer training (~50 min)
    python3 image_to_splat.py -i input\front.png --from_stage train --set iterations=20000   # retrain only
    python3 image_to_splat.py -i input\front.png --working_dir D:\splat_work\me --keep_comfy
    python3 image_to_splat.py --check                                    # test the setup without a run
    python3 image_to_splat.py --stop_comfy                               # stop a ComfyUI kept by --keep_comfy
    python3 image_to_splat.py --help                                     # every option

Or double-click run_image_to_splat.bat, or drag a photo onto it.

Result: <output_dir>\character.ply. Open it by dragging it onto https://superspl.at/editor
Also written: turntable.mp4, preview_8_angles.png, contact_sheet.png, photo_vs_render.png, report.json.
Stages rerun only when their inputs or settings change, so iterating on training settings is cheap.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
VENV_DIR = REPO_ROOT / ".venv-i2g"


def _relaunch_in_venv() -> None:
    """Run with the pipeline's own Python (torch, gsplat live there), whichever python3 started us."""
    if Path(sys.prefix).resolve() == VENV_DIR.resolve():
        return
    for exe in ("python3.exe", "python.exe"):
        python = VENV_DIR / "Scripts" / exe
        if python.exists():
            sys.exit(subprocess.call([str(python), str(Path(__file__).resolve()), *sys.argv[1:]]))
    print(f"ERROR: the pipeline's Python environment {VENV_DIR} does not exist yet.\n"
          f"FIX:   powershell -ExecutionPolicy Bypass -File {REPO_ROOT / 'scripts' / 'v2_setup_env.ps1'}\n"
          f"       (one time, ~5 min, downloads ~3.5 GB), then run this command again.")
    sys.exit(2)


if __name__ == "__main__":
    _relaunch_in_venv()

import argparse  # noqa: E402  (after the venv relaunch on purpose)
import json  # noqa: E402
import time  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))
from i2g.paths import COMFY_ROOT as DEFAULT_COMFY_ROOT, COMFY_URL as DEFAULT_COMFY_URL  # noqa: E402

# (set comfy_root in config.local.json once; see config.example.json)
STAGES = ["ingest", "body", "views", "register", "train"]
PRESETS = {
    # views: which camera angles; seeds: candidates per angle; iterations: splat training steps
    "draft": {"views": "draft", "seeds": 1, "iterations": 6000, "photo_hr": False},
    "standard": {"views": "standard", "seeds": 2, "iterations": 15000, "photo_hr": True},
    "high": {"views": "full", "seeds": 3, "iterations": 25000, "photo_hr": True},
}
IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="image_to_splat.py",
        description="Turn one front-facing, full-body photo into a static 3D Gaussian splat (.ply).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="presets: draft = 7 angles x 1 seed, 6k iterations (~20 min); standard = 11 angles x 2 seeds, "
               "15k iterations (~35 min); high = 15 angles x 3 seeds, 25k iterations (~50 min). "
               "First ComfyUI start adds ~5-8 min.")
    io = p.add_argument_group("input and output")
    io.add_argument("-i", "--input", type=Path, help="front-facing full-body photo (png, jpg, webp)")
    io.add_argument("-o", "--output_dir", type=Path,
                    help="folder for the finished splat; created if missing (default: <working_dir>\\final)")
    io.add_argument("-w", "--working_dir", type=Path,
                    help="folder for intermediate files; created if missing (default: output\\<name>\\v2)")
    io.add_argument("--name", help="run name used for default folders (default: the photo's file name)")

    q = p.add_argument_group("quality")
    q.add_argument("--preset", choices=PRESETS, default="standard", help="speed/quality preset (default: standard)")
    q.add_argument("--views", choices=["draft", "standard", "full"], help="camera angles to generate (overrides preset)")
    q.add_argument("--seeds", type=int, help="candidate images per angle (overrides preset)")
    q.add_argument("--iterations", type=int, help="splat training iterations (overrides preset)")
    q.add_argument("--no_hr_photo", action="store_true",
                   help="train the photo at canvas size instead of full resolution (faster, softer front)")
    q.add_argument("--fov", type=float, default=40.0, help="assumed vertical field of view of the photo (degrees)")
    q.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a training setting from i2g/splat.py SplatConfig, e.g. --set sh_degree=0 "
                        "(repeatable)")

    r = p.add_argument_group("re-running")
    r.add_argument("--from_stage", choices=STAGES, help="redo this stage and every stage after it")
    r.add_argument("--only_stage", choices=STAGES, help="run just this one stage")
    r.add_argument("--force", action="store_true",
                   help="redo stages even when up to date; old results are archived, never deleted")

    c = p.add_argument_group("ComfyUI")
    c.add_argument("--comfy_root", type=Path, default=DEFAULT_COMFY_ROOT,
                   help=f"portable ComfyUI folder containing python_embeded and ComfyUI (default from "
                        f"config.local.json: {DEFAULT_COMFY_ROOT})")
    c.add_argument("--comfy_url", default=DEFAULT_COMFY_URL, help=f"ComfyUI address (default {DEFAULT_COMFY_URL})")
    c.add_argument("--keep_comfy", action="store_true",
                   help="leave ComfyUI running at the end if this script started it (next run starts faster)")
    c.add_argument("--no_start_comfy", action="store_true", help="fail instead of starting ComfyUI")
    c.add_argument("--sage", action="store_true", help="start ComfyUI with SageAttention (faster)")
    c.add_argument("--comfy_timeout", type=float, default=900, help="seconds to wait for ComfyUI to start")
    c.add_argument("--stop_comfy", action="store_true",
                   help="stop a ComfyUI left running by --keep_comfy, then exit (never touches one you started)")

    d = p.add_argument_group("diagnostics")
    d.add_argument("--check", action="store_true",
                   help="check Python, GPU, ComfyUI and model files, then stop")
    d.add_argument("--debug", action="store_true", help="print detailed messages (always written to the log file)")
    return p


def fmt_minutes(seconds: float) -> str:
    return f"{seconds / 60:.1f} min" if seconds >= 60 else f"{seconds:.0f} s"


def make_bar(total=None, desc="", unit="it", **kwargs):
    from tqdm import tqdm
    return tqdm(total=total, desc=f"  {desc}", unit=unit, dynamic_ncols=True, leave=True, file=sys.stdout,
                mininterval=0.5, **kwargs)


def preflight(args, input_image: Path | None, log) -> None:
    """Fail early, with a fix, on anything that would break the run later."""
    from i2g.errors import PipelineError
    from i2g.paths import MODELS_DIR

    if input_image is not None:
        if not input_image.is_file():
            raise PipelineError(f"input photo not found: {input_image}",
                                hint="Check the path. Tip: drag the photo onto run_image_to_splat.bat.")
        if input_image.suffix.lower() not in IMAGE_TYPES:
            raise PipelineError(f"{input_image.name} is not a supported image type",
                                hint="Use a .png, .jpg or .webp photo.")
        from PIL import Image, UnidentifiedImageError
        try:
            with Image.open(input_image) as img:
                w, h = img.size
        except (UnidentifiedImageError, OSError) as exc:
            raise PipelineError(f"cannot read {input_image}: {exc}", hint="Re-save the photo as PNG or JPG.") from exc
        log.info("  ok   photo %s (%d x %d)", input_image.name, w, h)
        if h < 1000:
            log.warning("       the photo is only %d px tall; 1500+ px gives a much sharper splat", h)

    import torch
    if not torch.cuda.is_available():
        raise PipelineError("PyTorch cannot see the NVIDIA GPU",
                            hint="Update the NVIDIA driver and reboot; check that nvidia-smi works.")
    free, total = torch.cuda.mem_get_info()
    log.info("  ok   GPU %s (%.1f of %.1f GiB free)", torch.cuda.get_device_name(0), free / 1024 ** 3,
             total / 1024 ** 3)
    if free / 1024 ** 3 < 6:
        log.warning("       little free VRAM; close other GPU apps (games, browsers with video, other AI tools)")
    import gsplat
    log.info("  ok   torch %s, gsplat %s", torch.__version__, gsplat.__version__)

    for rel, what in (("birefnet/model.safetensors", "BiRefNet cut-out model (Hugging Face ZhengPeng7/BiRefNet)"),
                      ("perceptual/imagenet-vgg-verydeep-19-conv.safetensors",
                       "VGG-19 perceptual-loss weights (Hugging Face AntResearch/4DAnyone, perceptual/)")):
        if not (MODELS_DIR / rel).exists():
            raise PipelineError(f"missing {MODELS_DIR / rel} ({what})",
                                hint="Run  python3 scripts\\download_models.py --local_only  then run again.")
    log.info("  ok   local models in %s", MODELS_DIR)

    if not (args.comfy_root / "python_embeded" / "python.exe").exists():
        raise PipelineError(f"no portable ComfyUI at {args.comfy_root}",
                            hint="Set comfy_root in config.local.json (copy config.example.json) to the folder that "
                                 "holds python_embeded and ComfyUI, or pass --comfy_root.")

    try:
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq Docker Desktop.exe"], capture_output=True,
                             text=True, timeout=10).stdout
        if "Docker Desktop.exe" in out:
            log.warning("       Docker Desktop is running and can hold a lot of RAM; quitting it speeds up this run")
    except (OSError, subprocess.SubprocessError):
        pass


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.stop_comfy:
        os.environ["I2G_COMFY_ROOT"] = str(args.comfy_root)
        sys.path.insert(0, str(REPO_ROOT))
        from i2g.comfy_client import ComfyClient
        from i2g.comfy_process import stop_kept_comfy
        from i2g.log import setup_logging
        setup_logging(None)
        if not stop_kept_comfy(args.comfy_root, ComfyClient(args.comfy_url)):
            print("No ComfyUI started by this tool is running (a ComfyUI you opened yourself is left alone).")
        return 0
    if args.input is None and not args.check:
        parser.error("--input is required (the front-facing photo), unless you use --check")

    input_image = args.input.expanduser().resolve() if args.input else None
    name = args.name or (input_image.stem if input_image else "setup-check")
    work = (args.working_dir or REPO_ROOT / "output" / name / "v2").expanduser().resolve()
    output = (args.output_dir or work / "final").expanduser().resolve()

    # i2g reads these at import time
    os.environ["I2G_COMFY_ROOT"] = str(args.comfy_root)
    os.environ["I2G_COMFY_URL"] = args.comfy_url
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sys.path.insert(0, str(REPO_ROOT))

    from i2g.log import get_logger, setup_logging
    from i2g.pipeline import STAGE_TITLES, RunOptions, RunPaths, archive_dir, check_same_photo, parse_setting, \
        stage_status

    paths = RunPaths(work=work, output=output)
    # only create folders once the photo is known to exist (a typo shouldn't leave empty folders behind)
    real_run = not args.check and input_image is not None and input_image.is_file()
    if real_run:
        paths.create()
    log_dir = paths.logs if real_run else REPO_ROOT / "logs"
    log_file = log_dir / f"run_{time.strftime('%Y%m%d-%H%M%S')}.log"
    setup_logging(log_file, debug=args.debug)
    log = get_logger("cli")

    preset = dict(PRESETS[args.preset])
    for key, value in (("views", args.views), ("seeds", args.seeds), ("iterations", args.iterations)):
        if value is not None:
            preset[key] = value
    if args.no_hr_photo:
        preset["photo_hr"] = False

    from i2g.comfy_client import ComfyClient
    from i2g.errors import PipelineError
    client = ComfyClient(args.comfy_url)
    comfy_proc = None
    started = time.perf_counter()
    try:
        splat_overrides = {"iterations": preset["iterations"]}
        for item in args.set:
            key, value = parse_setting(item)
            splat_overrides[key] = value
        opts = RunOptions(name=name, input_image=input_image or Path(), vfov_deg=args.fov, views=preset["views"],
                          seeds=preset["seeds"], photo_hr=preset["photo_hr"], splat=splat_overrides)

        log.info("=" * 78)
        log.info("Image-to-Gaussian   photo: %s", input_image or "(setup check)")
        if not args.check:
            log.info("  preset %s (%s views x %d seeds, %d iterations%s)", args.preset, opts.views, opts.seeds,
                     splat_overrides["iterations"], "" if opts.photo_hr else ", photo at canvas size")
            log.info("  working folder: %s", work)
            log.info("  output folder:  %s", output)
        log.info("  log file:       %s", log_file)
        log.info("=" * 78)

        log.info("Checking the setup")
        preflight(args, input_image, log)

        from i2g.comfy_process import check_comfy, ensure_node_link, start_comfy
        link_created = ensure_node_link(args.comfy_root)
        if client.is_up():
            if link_created:
                raise PipelineError("the pipeline's helper node was just installed, but ComfyUI is already running",
                                    hint="Close ComfyUI (or run scripts\\v2_stop_comfy.ps1) and run this again.")
            log.info("  ok   using the ComfyUI already running at %s", args.comfy_url)
        elif args.no_start_comfy:
            raise PipelineError(f"ComfyUI is not running at {args.comfy_url}",
                                hint="Start ComfyUI, or drop --no_start_comfy so this script starts it.")
        else:
            comfy_proc = start_comfy(args.comfy_root, args.comfy_url, log_dir, args.sage, args.comfy_timeout,
                                     make_bar)
        info = check_comfy(client)
        log.info("  ok   ComfyUI %s on %s (%.1f GiB VRAM free); all nodes and model files found",
                 info["comfyui_version"], info["device"], info["vram_free_gib"])
        if args.check:
            log.info("Setup check passed. Run with --input <photo> to make a splat.")
            return 0

        check_same_photo(paths, opts)
        if args.only_stage:
            stages = [args.only_stage]
        elif args.from_stage:
            stages = STAGES[STAGES.index(args.from_stage):]
        else:
            stages = STAGES
        requested = set(stages) if (args.only_stage or args.from_stage or args.force) else set()

        from i2g.pipeline import RUNNERS
        timings, results = {}, {}
        for n, stage in enumerate(stages, 1):
            title = f"[{n}/{len(stages)}] {STAGE_TITLES[stage]}"
            up_to_date, why = stage_status(paths, opts, stage)
            if up_to_date and stage not in requested:
                log.info("%s: skipped (%s)", title, why)
                continue
            if args.force and stage == "views":
                archive_dir(paths, paths.stage_dir("views"), "views")
                archive_dir(paths, paths.stage_dir("register"), "cameras")
            log.info("%s%s", title, "" if stage in requested else f" ({why})")
            t0 = time.perf_counter()
            results[stage] = RUNNERS[stage](paths, opts, client, make_bar)
            timings[stage] = round(time.perf_counter() - t0, 1)
            log.info("%s: done in %s", title, fmt_minutes(timings[stage]))

        total = time.perf_counter() - started
        summary = {"input": str(input_image), "preset": args.preset, "options": opts.__dict__ | {
            "input_image": str(opts.input_image)}, "stage_seconds": timings, "results": results,
            "total_seconds": round(total, 1), "log_file": str(log_file), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
        (output / "run_summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")

        ply = output / "character.ply"
        log.info("=" * 78)
        if ply.exists():
            log.info("DONE in %s  ->  %s  (%.0f MB)", fmt_minutes(total), ply, ply.stat().st_size / 1024 ** 2)
            log.info("  also: turntable.mp4, preview_8_angles.png, contact_sheet.png, photo_vs_render.png")
            log.info("  view it: open https://superspl.at/editor and drag character.ply onto the page")
            log.info("  iterate: python3 image_to_splat.py -i \"%s\" --from_stage train --set KEY=VALUE",
                     input_image)
        else:
            log.info("Finished the requested stages in %s (no character.ply yet: run the train stage)",
                     fmt_minutes(total))
        log.info("=" * 78)
        return 0

    except PipelineError as exc:
        log.error("FAILED: %s", exc)
        if exc.hint:
            log.error("FIX:    %s", exc.hint)
        log.error("Details: %s (finished stages are kept; rerunning the same command continues)", log_file)
        return 2
    except KeyboardInterrupt:
        log.warning("Stopped (Ctrl+C). Finished stages are kept; rerun the same command to continue.")
        try:
            client.interrupt()
        except Exception:  # noqa: BLE001
            pass
        return 130
    except Exception as exc:  # noqa: BLE001 - last-resort report with the full traceback in the log
        log.debug("unexpected error", exc_info=True)
        log.error("FAILED with an unexpected %s: %s", type(exc).__name__, exc)
        log.error("Details (full traceback): %s   Rerun with --debug to see it here.", log_file)
        return 1
    finally:
        if comfy_proc is not None and not args.keep_comfy:
            from i2g.comfy_process import stop_comfy
            stop_comfy(comfy_proc, client)
        elif comfy_proc is not None:
            log.info("ComfyUI left running at %s (--keep_comfy)", args.comfy_url)


if __name__ == "__main__":
    sys.exit(main())

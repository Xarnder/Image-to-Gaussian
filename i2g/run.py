"""Minimal stage runner kept for scripts/v2_run.ps1. The user-facing tool is image_to_splat.py.

    python -m i2g.run --image test-assets/test-image.png --stage ingest|body|views|register|train|all

Outputs go to output/<image name>/v2/ (final files in .../final). Stages always run when asked.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .comfy_client import ComfyClient
from .log import setup_logging
from .paths import OUTPUT_DIR
from .pipeline import RUNNERS, STAGES, RunOptions, RunPaths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--stage", default="all", choices=STAGES + ["all"])
    args = parser.parse_args()

    name = args.image.stem
    work = OUTPUT_DIR / name / "v2"
    paths = RunPaths(work=work, output=work / "final")
    paths.create()
    setup_logging(paths.logs / "i2g_run.log")
    client = ComfyClient()
    if not client.is_up():
        raise SystemExit("ComfyUI is not running; start it with scripts\\v2_start_comfy.ps1")
    opts = RunOptions(name=name, input_image=args.image.resolve())
    for stage in STAGES if args.stage == "all" else [args.stage]:
        RUNNERS[stage](paths, opts, client)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

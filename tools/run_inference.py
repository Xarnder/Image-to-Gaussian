#!/usr/bin/env python3
"""Run official 4DAnyone inference with logging, timing, and GPU memory sampling."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def run(cmd: list[str], log_file: Path | None = None) -> int:
    print("+", " ".join(cmd), flush=True)
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    handle = log_file.open("a", encoding="utf-8") if log_file else None
    try:
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if handle:
                handle.write(line)
                handle.flush()
    finally:
        if handle:
            handle.close()
    return process.wait()


def read_gpu_memory_mib() -> int | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return int(result.stdout.strip().splitlines()[0].split()[0])
    except Exception:
        return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--views-per-layer", type=int, default=6)
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--model-dir", default="/workspace/models")
    parser.add_argument("--gvhmr-root", default="/opt/4danyone/third_party/GVHMR")
    parser.add_argument("--enable-turbo", default="True")
    parser.add_argument("--attention-backend", default="auto")
    args = parser.parse_args()

    log_file = Path(args.log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not (output_dir / "cameras.json").is_file():
        print(f"ERROR: incomplete output directory exists: {output_dir}", file=sys.stderr)
        return 1

    check = run([sys.executable, "/opt/pipeline/gpu_check.py"], log_file)
    if check != 0:
        return check

    download = run(
        [
            sys.executable,
            "/opt/4danyone/scripts/download_model.py",
            "--model_dir",
            args.model_dir,
            "--gvhmr_root",
            args.gvhmr_root,
        ],
        log_file,
    )
    if download != 0:
        return download

    config = run(
        [
            sys.executable,
            "/opt/pipeline/log_official_path.py",
            "--attention-backend",
            args.attention_backend,
            "--enable-turbo",
            args.enable_turbo,
            "--gpu-ids",
            "0",
        ],
        log_file,
    )
    if config != 0:
        return config

    samples: list[int] = []
    # Do not poll nvidia-smi while the child owns the GPU. WSL2 has reported
    # "CUDA driver error: device not ready" from concurrent driver queries
    # during large conv3d. Sample once after inference instead.
    started = time.time()
    inference = run(
        [
            sys.executable,
            "/opt/4danyone/inference.py",
            "--video_path",
            args.video_path,
            "--output_dir",
            args.output_dir,
            "--views_per_layer",
            str(args.views_per_layer),
            f"--enable_turbo={args.enable_turbo}",
            "--attention_backend",
            args.attention_backend,
            "--gpu_ids=[0]",
            "--model_dir",
            args.model_dir,
            "--gvhmr_root",
            args.gvhmr_root,
        ],
        log_file,
    )
    elapsed = time.time() - started
    used = read_gpu_memory_mib()
    if used is not None:
        samples.append(used)

    metadata_path = output_dir / "metadata.json"
    metadata = None
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            print("4danyone_metadata_keys=" + ",".join(sorted(metadata.keys())), flush=True)
            for key in ("attention_backend", "enable_turbo", "peak_cuda_allocated_bytes", "timings"):
                if key in metadata:
                    print(f"metadata_{key}={metadata[key]!r}", flush=True)
        except Exception as exc:
            print(f"metadata_read_error={exc}", flush=True)

    report = {
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": elapsed,
        "exit_code": inference,
        "peak_gpu_memory_mib": max(samples) if samples else None,
        "gpu_samples": len(samples),
        "output_dir": args.output_dir,
        "views_per_layer": args.views_per_layer,
        "cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    }
    report_path = output_dir / "run_report.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return inference


if __name__ == "__main__":
    raise SystemExit(main())

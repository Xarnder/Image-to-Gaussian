#!/usr/bin/env python3
"""Local browser UI for image_to_splat.py / batch_to_splat.py.

Starts a tiny HTTP server on 127.0.0.1, opens the page, and runs the real pipeline as a
subprocess so the console in the browser matches the usual terminal.
"""
from __future__ import annotations

import json
import codecs
import os
import queue
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
from batch_to_splat import (  # noqa: E402
    PRESET_MINUTES,
    STOP_AFTER_EXIT,
    apply_photo_order,
    find_photos,
    is_interrupt_code,
    photo_rel,
    run_names,
)
from image_to_splat import IMAGE_TYPES, PRESETS  # noqa: E402
from i2g.paths import OUTPUT_DIR, existing_experiments, find_deliverable, safe_run_name  # noqa: E402
from i2g.pipeline import STAGE_TITLES, STAGES  # noqa: E402
from i2g.video_options import VIDEO_STAGES, video_command, video_output


def video_stage_plan():
    return [{"id": key, "index": i+1, "title": title, "detail": detail, "included": True,
             "estimate": seconds, "estimate_hms": fmt_estimate(seconds)}
            for i, (key, title, detail, seconds) in enumerate(VIDEO_STAGES)]

GUI_DIR = REPO_ROOT / "gui"
VENV_PY = REPO_ROOT / ".venv-i2g" / "Scripts" / "python3.exe"
SINGLE = REPO_ROOT / "image_to_splat.py"
BATCH = REPO_ROOT / "batch_to_splat.py"
HOST = "127.0.0.1"
PORTS = range(8765, 8780)
HISTORY = 4000
PREVIEW_CAP = 2000
CREATE_NEW_PROCESS_GROUP = 0x00000200
MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".mp4": "video/mp4",
}
SKIP_STATUSES = (
    "built", "done", "ran, no splat", "failed", "stopped",
    "later (--limit)", "left for next start", "not reached",
)
# Angle counts from i2g.views.VIEW_PRESETS (draft / standard / full).
VIEW_COUNTS = {"draft": 7, "standard": 13, "full": 19}
STAGE_SHORT = {
    "ingest": "Cut out the person",
    "body": "Fit a 3D body",
    "views": "Generate views",
    "register": "Measure cameras",
    "train": "Train the splat",
    "refine": "Second pass",
    "orbit": "Orbit video pass",
}

# Rough share of a standard run, used only for the ETA bar (orbit dominates). From measured stage times
# (Nurse Man, F-14 batch) plus the ~20 min head pass that now ends the orbit stage.
STAGE_WEIGHTS = (0.005, 0.01, 0.13, 0.015, 0.11, 0.11, 0.52)
STAGE_RE = re.compile(r"\[(\d+)/(\d+)\]\s+(.+)")
DONE_IN_RE = re.compile(r"done in\s+(\d+(?:\.\d+)?)\s*(min|s)\b", re.I)
BATCH_RE = re.compile(r"^\[batch\]\s+(\d+)/(\d+)\s+(.+?)(?:, about |\s*$)")
BATCH_STOP_AFTER_RE = re.compile(r"^\[batch\]\s+stop after current")
PLAN_RE = re.compile(r"^ {2}(skip \(already built\)|build|later \(--limit\))\s+(.+?)\s+<-\s+(.+)$")
TQDM_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)%")
TQDM_LEFT_RE = re.compile(r"<(\d+:\d+(?::\d+)?)")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mK]")
CTRL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def venv_python() -> str:
    for exe in (VENV_PY, VENV_PY.with_name("python.exe")):
        if exe.exists():
            return str(exe)
    return sys.executable


def fmt_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds != seconds:  # noqa: PLR0124  NaN
        return "—"
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h} h {m:02d} min"
    if m:
        return f"{m} min {s:02d} s"
    return f"{s} s"


def fmt_estimate(seconds: float | None) -> str:
    """Coarse estimate for a stage (avoid fake second-level precision)."""
    if seconds is None or seconds < 0 or seconds != seconds:  # noqa: PLR0124  NaN
        return ""
    if seconds < 45:
        return f"about {max(1, int(round(seconds)))} s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"about {max(1, int(round(minutes)))} min"
    h, m = divmod(int(round(minutes)), 60)
    if m == 0:
        return f"about {h} h"
    return f"about {h} h {m:02d} min"


def parse_done_in(text: str) -> float | None:
    """Seconds from a pipeline line like 'done in 12.3 min' or 'done in 8 s'."""
    match = DONE_IN_RE.search(text)
    if not match:
        return None
    value = float(match.group(1))
    return value * 60.0 if match.group(2).lower().startswith("min") else value


def stage_estimate_seconds(preset: str) -> dict[str, float]:
    """Share of PRESET_MINUTES across stages that this preset actually runs."""
    spec = PRESETS.get(preset, PRESETS["standard"])
    total = PRESET_MINUTES.get(preset, 180) * 60.0
    included: list[tuple[str, float]] = []
    for key, weight in zip(STAGES, STAGE_WEIGHTS, strict=True):
        if key == "refine" and not spec.get("refine"):
            continue
        if key == "orbit" and not spec.get("orbit"):
            continue
        included.append((key, weight))
    weight_sum = sum(weight for _, weight in included) or 1.0
    return {key: total * (weight / weight_sum) for key, weight in included}


def parse_clock(text: str) -> float | None:
    parts = text.split(":")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return None
    if len(nums) == 2:
        return nums[0] * 60 + nums[1]
    if len(nums) == 3:
        return nums[0] * 3600 + nums[1] * 60 + nums[2]
    return None


def strip_ansi(text: str) -> str:
    return CTRL_CHARS_RE.sub("", ANSI_RE.sub("", text)).replace("\x00", "")


def stage_plan(preset: str, no_orbit: bool = False) -> list[dict]:
    """The character pipeline for this preset: which stages run, and a short explanation of each."""
    spec = PRESETS.get(preset, PRESETS["standard"])
    view_key = str(spec.get("views") or "standard")
    n_angles = VIEW_COUNTS.get(view_key, VIEW_COUNTS["standard"])
    seeds = int(spec.get("seeds") or 1)
    tries = "1 try" if seeds == 1 else f"{seeds} tries"
    refine_on = bool(spec.get("refine"))
    restore = bool(spec.get("restore"))
    orbit_on = bool(spec.get("orbit")) and not no_orbit
    if restore:
        refine_detail = "Refit cameras, SeedVR2-cleaned frames, retrain"
    elif refine_on:
        refine_detail = "Refit cameras to the splat, then retrain"
    else:
        refine_detail = "Off in the draft preset"
    if orbit_on:
        orbit_detail = "Head fix, Wan turntable video, retrain, head pass"
    elif no_orbit:
        orbit_detail = "Off — Skip orbit video is ticked"
    else:
        orbit_detail = "Off in the draft preset"
    details = {
        "ingest": "Isolate the person on a clean canvas",
        "body": "Estimate pose and body shape",
        "views": f"{n_angles} camera angles × {tries} each",
        "register": "Fit each view to the 3D body; regenerate any missing side",
        "train": f"{int(spec.get('iterations') or 0):,} training steps",
        "refine": refine_detail,
        "orbit": orbit_detail,
    }
    included = {
        "ingest": True, "body": True, "views": True, "register": True, "train": True,
        "refine": refine_on, "orbit": orbit_on,
    }
    estimates = stage_estimate_seconds(preset if preset in PRESETS else "standard")
    items = []
    for n, key in enumerate(STAGES, 1):
        title = STAGE_SHORT.get(key) or STAGE_TITLES.get(key, key).split(":")[0]
        estimate = estimates.get(key)
        items.append({
            "id": key,
            "index": n,
            "title": title,
            "detail": details[key],
            "included": included[key],
            "estimate": estimate,
            "estimate_hms": fmt_estimate(estimate) if included[key] else "",
        })
    return items


def match_stage_key(index: int, total: int, title: str) -> str | None:
    if total == 4 and 1 <= index <= 4:
        return VIDEO_STAGES[index-1][0]
    text = (title or "").casefold()
    for key, full in STAGE_TITLES.items():
        head = full.split(":")[0].strip().casefold()
        short = STAGE_SHORT.get(key, "").casefold()
        if text == head or text.startswith(head) or (short and (text == short or text.startswith(short))):
            return key
    if 1 <= index <= len(STAGES) and total == len(STAGES):
        return STAGES[index - 1]
    return None


VRAM_HINT = (
    "Close other GPU apps (games, browsers with video, other AI tools) and extra ComfyUI tabs. "
    "Wait a few seconds if the driver reset, then Start again without Rebuild — finished stages are kept."
)
CRASH_HINT = (
    "Finished stages are kept. Start again without Rebuild to continue this character. "
    "If it keeps crashing, try the draft preset or tick Skip orbit video."
)
BUSY_HINT = (
    "Only one job can use the GPU at a time. Wait for the other one to finish (or stop it), then Start again "
    "without Rebuild; finished stages are kept. New runs now wait for each other automatically."
)
BUSY_RE = re.compile(r"Another Image-to-Gaussian job was using the GPU|ComfyUI was also running other jobs", re.I)
GPU_WAIT_RE = re.compile(r"Waiting for the GPU: another Image-to-Gaussian job is using it \((.*?)\)\. ", re.I)
FAILED_RE = re.compile(r"^FAILED(?: with an unexpected (\w+))?:?\s+(.*)$", re.I)
FIX_RE = re.compile(r"^FIX:\s+(.*)$", re.I)
VRAM_RE = re.compile(
    r"out of memory|OutOfMemoryError|CUDA out of memory|ran out of memory|"
    r"VRAM spill|VramBudgetExceeded|sysmem fallback|peaked at .+\sGiB \(budget",
    re.I)
CUDA_RE = re.compile(
    r"CUDA driver error|device not ready|cudaError|illegal memory access|GPU has fallen off",
    re.I)
COMFY_RE = re.compile(
    r"ComfyUI (reported an error|exited|did not answer|is not running|cannot see)|"
    r"execution_error|WatchdogTripped|sampler step took",
    re.I)
BATCH_RESULT_RE = re.compile(
    r"^\[batch\]\s+(built|failed|stopped|ran, no splat)\s+in\b", re.I)
NT_ACCESS = 0xC0000005
NT_HEAP = 0xC0000374
NT_ABORT = 0xC0000409
NT_STACK = 0xC00000FD


def _unsigned(code: int) -> int:
    return code & 0xFFFFFFFF


def error_from_text(text: str) -> dict | None:
    raw = (text or "").strip()
    if not raw:
        return None
    failed = FAILED_RE.match(raw)
    if failed:
        unexpected, rest = failed.group(1), (failed.group(2) or "").strip()
        kind, title = "pipeline", "The pipeline stopped with an error"
        if unexpected:
            kind, title = "crash", f"Unexpected {unexpected}"
        if BUSY_RE.search(raw):
            return {"kind": "busy", "title": "Another job was using the GPU", "message": rest or raw,
                    "hint": BUSY_HINT}
        if VRAM_RE.search(rest) or VRAM_RE.search(raw):
            kind, title = "vram", "The GPU ran out of memory"
        elif CUDA_RE.search(rest) or CUDA_RE.search(raw):
            kind, title = "cuda", "The GPU driver reported an error"
        elif COMFY_RE.search(rest) or COMFY_RE.search(raw):
            kind, title = "comfy", "ComfyUI failed"
        return {"kind": kind, "title": title, "message": rest or raw, "hint": VRAM_HINT if kind == "vram" else ""}
    fix = FIX_RE.match(raw)
    if fix:
        return {"kind": "hint", "title": "", "message": "", "hint": fix.group(1).strip()}
    if VRAM_RE.search(raw):
        return {"kind": "vram", "title": "The GPU ran out of memory", "message": raw, "hint": VRAM_HINT}
    if CUDA_RE.search(raw):
        return {
            "kind": "cuda", "title": "The GPU driver reported an error", "message": raw,
            "hint": "Wait a few seconds for the driver to recover, close other GPU apps, then Start again.",
        }
    if COMFY_RE.search(raw):
        hint = VRAM_HINT if "sampler step" in raw.casefold() or "spill" in raw.casefold() else (
            "Check that ComfyUI is installed and the models are in place, then Start again.")
        return {"kind": "comfy", "title": "ComfyUI failed", "message": raw, "hint": hint}
    return None


def error_from_exit(code: int) -> dict:
    u = _unsigned(code)
    if code == 2:
        return {"kind": "pipeline", "title": "The pipeline stopped with an error",
                "message": "A step failed and wrote FAILED / FIX lines in the console.",
                "hint": "Read the red panel and the console. Finished stages are kept; Start again without Rebuild."}
    if code == 1:
        return {"kind": "crash", "title": "The pipeline hit an unexpected error",
                "message": "Python reported an unexpected exception. The full traceback is in the run log.",
                "hint": CRASH_HINT + " Tick Debug log for more detail next time."}
    if 1 <= code <= 125:
        return {"kind": "batch", "title": "The batch finished with failures",
                "message": f"{code} character(s) failed. See FAILED lines in the console and each run’s logs folder.",
                "hint": "Start again without Rebuild to retry the failed ones (leave Rebuild unticked so finished splats stay skipped)."}
    if u == NT_ACCESS:
        return {"kind": "crash", "title": "The pipeline process crashed",
                "message": "Windows ended Python with an access violation. This often follows a GPU driver reset after it ran out of memory.",
                "hint": VRAM_HINT}
    if u in (NT_HEAP, NT_ABORT, NT_STACK):
        return {"kind": "crash", "title": "The pipeline process aborted",
                "message": f"Windows reported a fatal abort (code 0x{u:08X}).",
                "hint": VRAM_HINT}
    if code in (137, 9, -9, 139, -11):
        return {"kind": "crash", "title": "The process was killed",
                "message": f"The pipeline was terminated (exit {code}).",
                "hint": CRASH_HINT}
    return {"kind": "crash", "title": "The pipeline stopped unexpectedly",
            "message": f"Exit code {code} (0x{u:08X}).",
            "hint": CRASH_HINT}


KIND_RANK = {"hint": 0, "batch": 1, "pipeline": 1, "crash": 2, "comfy": 2, "cuda": 3, "vram": 4, "busy": 5}


class RunState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.running = False
        self.mode = ""
        self.input_path = ""
        self.experiment = ""
        self.preset = "standard"
        self.no_orbit = False
        self.stage_flags: dict[str, str] = {s: "pending" for s in STAGES}
        self.photos: list[dict] = []
        self.history: list[dict] = []
        self.subscribers: list[queue.Queue] = []
        self.started_at = 0.0
        self.image_started_at = 0.0
        self.image_index = 0
        self.image_total = 1
        self.image_name = ""
        self.stage_index = 0
        self.stage_total = 7
        self.stage_title = ""
        self.tqdm_frac = 0.0
        self.tqdm_left: float | None = None
        self.tqdm_desc = ""
        self.stage_started_at: dict[str, float] = {}
        self.stage_took: dict[str, float] = {}
        self.finished_times: list[float] = []
        self.exit_code: int | None = None
        self.status = "idle"
        self.message = "Pick a photo or a folder, then Start."
        self.stop_mode: str | None = None  # None | "after" | "now"
        self.stop_file: Path | None = None
        self.error: dict | None = None

    def snapshot(self) -> dict:
        with self.lock:
            now = time.time()
            elapsed = (now - self.started_at) if self.running and self.started_at else 0.0
            image_elapsed = (now - self.image_started_at) if self.running and self.image_started_at else 0.0
            image_frac = self._image_frac()
            batch_frac = self._batch_frac(image_frac)
            stage_frac = min(1.0, max(self.tqdm_frac, 0.0))
            if not self.running and self.status == "done":
                image_frac = batch_frac = stage_frac = 1.0
            jobs_done, jobs_total = self._job_counts()
            preset_s = 1070 if self.mode == "video" else PRESET_MINUTES.get(self.preset, 180) * 60
            if image_frac > 0.04:
                image_eta = max(0.0, image_elapsed / image_frac - image_elapsed)
            else:
                image_eta = max(0.0, preset_s - image_elapsed)
            if self.mode == "video":
                # Stage weights drive the bar, but are not measured elapsed time.
                # Estimate remaining time from the active stage and pending stages.
                image_eta = sum(
                    (row.get("eta") or 0.) if row["status"] == "running"
                    else float(row.get("estimate") or 0.) if row["status"] == "pending" else 0.
                    for row in self._stages()
                )
            remaining_after = max(jobs_total - jobs_done - (1 if self.running and self.image_index else 0), 0)
            avg = (sum(self.finished_times) / len(self.finished_times)) if self.finished_times else preset_s
            batch_eta = image_eta + remaining_after * avg
            return {
                "app": "i2g-gui",
                "running": self.running,
                "status": self.status,
                "message": self.message,
                "mode": self.mode,
                "input": self.input_path,
                "experiment": self.experiment,
                "preset": self.preset,
                "no_orbit": self.no_orbit,
                "photos": list(self.photos),
                "stages": self._safe_stages(),
                "image_index": self.image_index,
                "image_total": self.image_total,
                "image_name": self.image_name,
                "stage_index": self.stage_index,
                "stage_total": self.stage_total,
                "stage_title": self.stage_title,
                "tqdm_desc": self.tqdm_desc,
                "elapsed": elapsed,
                "elapsed_hms": fmt_duration(elapsed),
                "image_elapsed": image_elapsed,
                "image_elapsed_hms": fmt_duration(image_elapsed),
                "image_eta": image_eta,
                "image_eta_hms": fmt_duration(image_eta),
                "batch_eta": batch_eta if self.mode == "batch" else image_eta,
                "batch_eta_hms": fmt_duration(batch_eta if self.mode == "batch" else image_eta),
                "stage_eta_hms": fmt_duration(self.tqdm_left if self.running else None),
                "batch_percent": round(100.0 * batch_frac, 1),
                "image_percent": round(100.0 * image_frac, 1),
                "stage_percent": round(100.0 * stage_frac, 1),
                "percent": round(100.0 * batch_frac, 1),
                "jobs_done": jobs_done,
                "jobs_total": jobs_total,
                "exit_code": self.exit_code,
                "stop_mode": self.stop_mode,
                "error": dict(self.error) if self.error else None,
                "preview": self._safe_preview(),
                **self._work_paths(),
            }

    def _current_photo(self) -> dict | None:
        photos = self.photos
        for photo in photos:
            if photo.get("current"):
                return photo
        name = self.image_name
        if name:
            name_cf = name.casefold()
            stem_cf = Path(name).stem.casefold()
            for photo in photos:
                file_name = Path(str(photo.get("file") or "")).name
                src_name = Path(str(photo.get("path") or "")).name
                run = str(photo.get("name") or "").casefold()
                if name_cf in (file_name.casefold(), src_name.casefold()) or run in (stem_cf, name_cf):
                    return photo
        for photo in photos:
            status = str(photo.get("status") or "")
            if status.startswith("skip") or status in SKIP_STATUSES:
                continue
            return photo
        if len(photos) == 1:
            return photos[0]
        if self.input_path:
            src = Path(self.input_path)
            if src.is_file():
                return {"name": safe_run_name(src.stem), "file": src.name, "path": str(src.resolve())}
        return None

    def _work_paths(self) -> dict:
        """v2 working directory for the current character (final files live in final/)."""
        empty = {"work_dir": "", "final_dir": "", "run_name": "", "work_exists": False, "final_exists": False}
        current = self._current_photo()
        experiment = safe_run_name(self.experiment) if self.experiment else ""
        run_name = ""
        if current is not None:
            run_name = safe_run_name(str(current.get("name") or Path(str(current.get("file") or "")).stem))
        elif self.image_name:
            run_name = safe_run_name(Path(self.image_name).stem)
        if not experiment or not run_name:
            return {**empty, "run_name": run_name}
        work = OUTPUT_DIR / experiment / run_name / ("video" if self.mode == "video" else "v2")
        final = work / "final"
        return {
            "work_dir": str(work),
            "final_dir": str(final),
            "run_name": run_name,
            "work_exists": work.is_dir(),
            "final_exists": final.is_dir(),
        }

    def _viewer(self) -> dict:
        """Source photo, then contact sheet, then turntable as each file appears."""
        current = self._current_photo()
        if current is None:
            return {"name": "", "file": "", "active": None, "items": []}
        source = Path(str(current.get("path") or ""))
        if not source.is_file():
            rel = str(current.get("file") or "")
            base = Path(self.input_path) if self.input_path else None
            if base is not None:
                candidate = base / rel if base.is_dir() else base
                if candidate.is_file():
                    source = candidate
        run_name = safe_run_name(str(current.get("name") or Path(str(current.get("file") or source)).stem))
        experiment = safe_run_name(self.experiment) if self.experiment else ""
        items: list[dict] = []
        if self.mode == "video":
            work = video_output(experiment, source)
            for path, item_id, label, kind in (
                (source, "source_video", "Approved source video", "video"),
                (work/"contact_sheet.jpg", "frames", "Extracted frames", "image"),
                (work/"final"/"comparison.jpg", "comparison", "Video vs splat", "image"),
                (work/"final"/"turntable.mp4", "turntable", "Gaussian turntable", "video"),
            ):
                asset = _media_item(path if path.is_file() else None, item_id, label, kind)
                if asset:
                    items.append(asset)
            return {"name": run_name, "file": source.name, "active": items[-1]["id"] if items else None,
                    "items": items}
        photo = _media_item(source if source.is_file() else None, "photo", "Photo", "image")
        if photo:
            items.append(photo)
        if experiment and run_name:
            work = OUTPUT_DIR / experiment / run_name / "v2"
            final = work / "final"
            for path, item_id, label, kind in (
                (work / "00_ingest" / "canvas.png", "cutout", "Cut-out", "image"),
                (work / "01_body" / "overlay_photo.png", "body_fit", "Body fit", "image"),
            ):
                asset = _media_item(path if path.is_file() else None, item_id, label, kind)
                if asset:
                    items.append(asset)
            items.extend(_view_gallery_items(work / "02_views"))
            contact = find_deliverable(final, run_name, "contact_sheet.png", experiment)
            if contact is None:
                qa = work / "03_qa" / "contact_sheet.png"
                if qa.is_file():
                    contact = qa
            for asset in (
                _media_item(contact, "contact_sheet", "Contact sheet", "image"),
                _media_item(
                    find_deliverable(final, run_name, "photo_vs_render.png", experiment),
                    "photo_vs_render", "Photo vs render", "image"),
                _media_item(
                    find_deliverable(final, run_name, "preview_8_angles.png", experiment),
                    "preview_8", "8 angles", "image"),
                _media_item(
                    find_deliverable(final, run_name, "turntable.mp4", experiment),
                    "turntable", "Turntable", "video"),
            ):
                if asset:
                    items.append(asset)
        file_label = str(current.get("file") or self.image_name or source.name)
        return {
            "name": run_name,
            "file": file_label,
            "active": items[-1]["id"] if items else None,
            "items": items,
        }

    def _safe_preview(self) -> dict:
        try:
            return self._viewer()
        except Exception:
            return {"name": "", "file": "", "active": None, "items": []}

    def _safe_stages(self) -> list[dict]:
        try:
            return self._stages()
        except Exception:
            return []

    def note_error(self, incoming: dict, *, fatal: bool = False) -> None:
        if not incoming:
            return
        prev = self.error or {}
        incoming_kind = incoming.get("kind") or "pipeline"
        prev_kind = prev.get("kind") or ""
        if incoming_kind == "hint":
            kind = prev_kind or "pipeline"
            title = prev.get("title") or "The pipeline stopped with an error"
        elif KIND_RANK.get(incoming_kind, 0) >= KIND_RANK.get(prev_kind, 0):
            kind = incoming_kind
            title = incoming.get("title") or prev.get("title") or "Something went wrong"
        else:
            kind = prev_kind or incoming_kind
            title = prev.get("title") or incoming.get("title") or "Something went wrong"
        self.error = {
            "kind": kind,
            "title": title,
            "message": (incoming.get("message") or prev.get("message") or "").strip(),
            "hint": (incoming.get("hint") or prev.get("hint") or "").strip(),
            "photo": self.image_name,
            "stage": self.stage_title,
            "fatal": bool(fatal or prev.get("fatal")),
        }

    def reset_stages(self) -> None:
        self.stage_flags = {s: "pending" for s in ([v[0] for v in VIDEO_STAGES] if self.mode == "video" else STAGES)}
        self.stage_index = 0
        self.stage_title = ""
        self.tqdm_frac = 0.0
        self.tqdm_left = None
        self.tqdm_desc = ""
        self.stage_started_at = {}
        self.stage_took = {}

    def _finish_stage_clock(self, key: str, now: float | None = None) -> None:
        if key in self.stage_took:
            return
        started = self.stage_started_at.get(key)
        if started:
            self.stage_took[key] = max(0.0, (now or time.time()) - started)

    def mark_stage(self, key: str, status: str) -> None:
        keys = [v[0] for v in VIDEO_STAGES] if self.mode == "video" else STAGES
        if key not in keys:
            return
        now = time.time()
        if status == "running":
            for other, flag in list(self.stage_flags.items()):
                if other != key and flag == "running":
                    self._finish_stage_clock(other, now)
                    self.stage_flags[other] = "done"
            self.stage_started_at.setdefault(key, now)
        elif status in ("done", "failed"):
            self._finish_stage_clock(key, now)
        elif status == "skipped":
            self.stage_took.pop(key, None)
        self.stage_flags[key] = status
        if status in ("running", "done", "skipped", "failed"):
            for earlier in keys[: keys.index(key)]:
                if self.stage_flags.get(earlier) == "pending":
                    self.stage_flags[earlier] = "done"

    def _stages(self) -> list[dict]:
        plan = video_stage_plan() if self.mode == "video" else stage_plan(self.preset or "standard", self.no_orbit)
        idx = self.stage_index
        finished = (not self.running) and self.status == "done"
        stopped = (not self.running) and self.status in ("stopped", "stopped_after", "failed")
        now = time.time()
        out = []
        for item in plan:
            key = item["id"]
            flag = self.stage_flags.get(key, "pending")
            n = item["index"]
            if not item["included"]:
                status = "off"
            elif flag == "failed" or (
                self.status == "failed" and not self.running and idx > 0 and n == idx
            ):
                status = "failed"
            elif finished:
                status = "skipped" if flag == "skipped" else "done"
            elif flag in ("done", "skipped", "running"):
                status = flag
            elif idx > 0 and n < idx:
                status = "done"
            elif idx > 0 and n == idx and self.running:
                status = "running"
            elif stopped and idx > 0 and n == idx:
                status = "running"
            else:
                status = "pending"
            row = dict(item)
            row["status"] = status
            row["percent"] = round(100.0 * min(1.0, max(self.tqdm_frac, 0.0)), 1) if status == "running" else None
            if status == "running" and self.tqdm_desc:
                row["note"] = self.tqdm_desc
            estimate = item.get("estimate")
            if status == "off":
                row["estimate_hms"] = ""
            if status == "running":
                started = self.stage_started_at.get(key)
                elapsed = (now - started) if started else 0.0
                frac = min(1.0, max(self.tqdm_frac, 0.0))
                if self.tqdm_left is not None:
                    eta = max(0.0, self.tqdm_left)
                elif frac > 0.04:
                    eta = max(0.0, elapsed / frac - elapsed)
                elif isinstance(estimate, (int, float)):
                    eta = max(0.0, float(estimate) - elapsed)
                else:
                    eta = None
                row["started_at"] = started
                row["elapsed"] = elapsed
                row["elapsed_hms"] = fmt_duration(elapsed)
                row["eta"] = eta
                row["eta_hms"] = fmt_duration(eta) if eta is not None and eta >= 0.5 else ""
            elif status in ("done", "failed") and key in self.stage_took:
                took = self.stage_took[key]
                row["took"] = took
                row["took_hms"] = fmt_duration(took)
            out.append(row)
        return out

    def _image_frac(self) -> float:
        n = max(self.stage_total, 1)
        idx = min(max(self.stage_index, 0), n)
        if idx <= 0:
            return min(1.0, self.tqdm_frac)
        weights = (0.15, 0.05, 0.77, 0.03) if self.mode == "video" else STAGE_WEIGHTS
        if n != len(weights):
            return min(1.0, (idx - 1 + self.tqdm_frac) / n)
        done = sum(weights[: idx - 1])
        current = weights[idx - 1] * min(1.0, max(self.tqdm_frac, 0.0))
        return min(1.0, done + current)

    def _job_counts(self) -> tuple[int, int]:
        photos = self.photos
        if photos:
            done = 0
            for photo in photos:
                if photo.get("current"):
                    continue
                status = str(photo.get("status") or "")
                if status.startswith("skip") or status in (
                    "built", "done", "ran, no splat", "failed", "stopped",
                ):
                    done += 1
            return done, len(photos)
        done = max(self.image_index - (1 if self.running else 0), 0)
        return done, max(self.image_total, 1)

    def _batch_frac(self, image_frac: float) -> float:
        photos = self.photos
        if photos:
            acc = 0.0
            for photo in photos:
                status = str(photo.get("status") or "")
                if photo.get("current"):
                    acc += image_frac
                elif status.startswith("skip") or status in (
                    "built", "done", "ran, no splat", "failed", "stopped",
                ):
                    acc += 1.0
            return min(1.0, acc / len(photos))
        total = max(self.image_total, 1)
        done = max(self.image_index - 1, 0)
        return min(1.0, (done + image_frac) / total)

    def emit(self, event: dict) -> None:
        event = dict(event)
        event.setdefault("t", time.time())
        with self.lock:
            self.history.append(event)
            if len(self.history) > HISTORY:
                self.history = self.history[-HISTORY:]
            subs = list(self.subscribers)
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=500)
        with self.lock:
            self.subscribers.append(q)
            replay = list(self.history[-400:])
        for event in replay:
            try:
                q.put_nowait(event)
            except queue.Full:
                break
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)


STATE = RunState()


def browse(kind: str) -> str | None:
    """Native file dialog in a helper process (tkinter is unreliable from an HTTP worker thread)."""
    picker = r"""
import sys
from tkinter import Tk, filedialog
root = Tk()
root.withdraw()
try:
    root.wm_attributes("-topmost", 1)
except Exception:
    pass
kind, initial = sys.argv[1], sys.argv[2]
if kind == "folder":
    path = filedialog.askdirectory(initialdir=initial, title="Choose a folder of photos")
elif kind == "video":
    path = filedialog.askopenfilename(initialdir=initial, title="Choose a pre-generated 360 video",
                                      filetypes=[("MP4 video", "*.mp4")])
else:
    path = filedialog.askopenfilename(
        initialdir=initial, title="Choose a photo",
        filetypes=[("Photos", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"), ("All files", "*.*")])
root.destroy()
sys.stdout.buffer.write(path.encode("utf-8"))
"""
    try:
        out = subprocess.check_output(
            [sys.executable, "-c", picker, kind, str(REPO_ROOT / "input")],
            cwd=str(REPO_ROOT), timeout=600)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    text = out.decode("utf-8", "replace").strip()
    return text or None


def _under_output(path: Path) -> bool:
    try:
        path.resolve().relative_to(OUTPUT_DIR.resolve())
        return True
    except ValueError:
        return False


def reveal_under_output(path: Path) -> dict:
    """Open Explorer (or the OS file manager) at a folder under output/."""
    root = OUTPUT_DIR.resolve()
    path = path.resolve()
    if path != root and not _under_output(path):
        return {"ok": False, "error": "That folder is outside output\\."}
    shown = path
    while shown != root and not shown.exists():
        shown = shown.parent
    if not shown.exists() and root.exists():
        shown = root
    if not shown.exists():
        return {"ok": False, "error": f"That folder has not been created yet:\n{path}"}
    try:
        if sys.platform == "win32":
            os.startfile(str(shown))  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(shown)])
        else:
            subprocess.Popen(["xdg-open", str(shown)])
    except OSError as exc:
        return {"ok": False, "error": f"Could not open the folder: {exc}"}
    return {"ok": True, "path": str(shown), "requested": str(path)}


def open_character_folder(body: dict) -> dict:
    """Open this character's v2 working directory (final files live in final/)."""
    experiment = str(body.get("experiment") or "").strip()
    run_name = str(body.get("name") or "").strip()
    where = str(body.get("where") or "work").strip().lower()
    with STATE.lock:
        if not experiment:
            experiment = STATE.experiment or ""
        if not run_name:
            current = STATE._current_photo()
            if current:
                run_name = str(current.get("name") or Path(str(current.get("file") or "")).stem)
            elif STATE.image_name:
                run_name = Path(STATE.image_name).stem
    experiment = str(experiment or "").strip()
    run_name = str(run_name or "").strip()
    if not experiment or not run_name:
        return {
            "ok": False,
            "error": "No character folder yet. Set an experiment and start a run, or wait until a photo is in progress.",
        }
    experiment = safe_run_name(experiment)
    run_name = safe_run_name(run_name)
    work = OUTPUT_DIR / experiment / run_name / ("video" if body.get("mode", STATE.mode) == "video" else "v2")
    target = work / "final" if where == "final" else work
    return reveal_under_output(target)


def preview(path: str, recursive: bool, mode: str = "") -> dict:
    where = Path(path).expanduser()
    if not where.exists():
        return {"ok": False, "error": f"not found: {where}"}
    if mode == "video":
        if not where.is_file() or where.suffix.lower() != ".mp4":
            return {"ok": False, "error": "Choose a pre-generated 360° MP4 video."}
        return {"ok": True, "mode": "video", "count": 1, "path": str(where.resolve()),
                "photos": [{"file": where.name, "name": where.stem, "path": str(where.resolve())}]}
    photos = find_photos(where, recursive)
    root = where if where.is_dir() else where.parent
    items = []
    for p in photos[:PREVIEW_CAP]:
        items.append({
            "file": photo_rel(p, root),
            "name": p.stem,
            "path": str(p.resolve()),
        })
    mode = "single" if where.is_file() else "batch"
    return {"ok": True, "mode": mode, "count": len(photos), "photos": items,
            "truncated": max(0, len(photos) - PREVIEW_CAP),
            "path": str(where.resolve()), "root": str(root.resolve())}


def order_lines(raw) -> list[str]:
    if not isinstance(raw, list):
        return []
    lines: list[str] = []
    for item in raw:
        if isinstance(item, dict):
            text = str(item.get("file") or item.get("path") or item.get("name") or "").strip()
        else:
            text = str(item).strip()
        if text:
            lines.append(text)
    return lines


def ordered_photos(where: Path, recursive: bool, order: list[str] | None) -> tuple[list[Path], list[str]]:
    photos = find_photos(where, recursive)
    root = where if where.is_dir() else where.parent
    if not order:
        return photos, []
    return apply_photo_order(photos, order, root)


def write_order_file(experiment: str, lines: list[str]) -> Path:
    dest = REPO_ROOT / "output" / experiment
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "gui_order.txt"
    body = "# Image-to-Gaussian GUI processing order (first line runs first)\n"
    body += "\n".join(lines) + "\n"
    path.write_text(body, encoding="utf-8")
    return path


def strip_flag(extra: list[str], flag: str) -> list[str]:
    out: list[str] = []
    skip = False
    for arg in extra:
        if skip:
            skip = False
            continue
        if arg == flag:
            skip = True
            continue
        if arg.startswith(flag + "="):
            continue
        out.append(arg)
    return out


def _safe_image(raw: str) -> Path | None:
    text = unquote(raw or "").strip()
    if not text:
        return None
    try:
        path = Path(text).expanduser().resolve()
    except (OSError, RuntimeError):
        return None
    if not path.is_file() or path.suffix.lower() not in IMAGE_TYPES:
        return None
    return path


def _safe_media(raw: str) -> Path | None:
    text = unquote(raw or "").strip()
    if not text:
        return None
    try:
        path = Path(text).expanduser().resolve()
    except (OSError, RuntimeError):
        return None
    if not path.is_file() or path.suffix.lower() not in MEDIA_TYPES:
        return None
    return path


def _view_gallery_items(views_root: Path) -> list[dict]:
    """One gallery tile per generated camera folder, using the latest seed image."""
    items: list[dict] = []
    if not views_root.is_dir():
        return items
    try:
        folders = sorted((p for p in views_root.iterdir() if p.is_dir()), key=lambda p: p.name)
    except OSError:
        return items
    for folder in folders:
        try:
            pngs = sorted(folder.glob("s*.png")) or sorted(folder.glob("*.png"))
        except OSError:
            continue
        if not pngs:
            continue
        label = "View " + folder.name.replace("_", " ")
        asset = _media_item(pngs[-1], f"view_{folder.name}", label, "image")
        if asset:
            items.append(asset)
    return items


def _media_item(path: Path | None, item_id: str, label: str, kind: str) -> dict | None:
    if path is None:
        return None
    try:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            return None
        mtime = int(resolved.stat().st_mtime)
    except OSError:
        return None
    return {
        "id": item_id,
        "label": label,
        "kind": kind,
        "url": f"/api/media?p={quote(str(resolved), safe='')}&t={mtime}",
        "mtime": mtime,
    }


def _parse_range(header: str, size: int) -> tuple[int, int] | None:
    if not header or not header.startswith("bytes=") or size <= 0:
        return None
    spec = header[6:].strip()
    if "," in spec:
        return None
    start_s, _, end_s = spec.partition("-")
    try:
        if start_s == "":
            suffix = int(end_s)
            start = max(size - suffix, 0)
            end = size - 1
        elif end_s == "":
            start = int(start_s)
            end = size - 1
        else:
            start = int(start_s)
            end = int(end_s)
    except ValueError:
        return None
    if start < 0 or end < start or start >= size:
        return None
    return start, min(end, size - 1)


def build_command(body: dict) -> tuple[list[str], str]:
    if body.get("mode") == "video":
        return video_command(venv_python(), body), "video"
    path = Path(body.get("input") or "").expanduser()
    if not path.exists():
        raise ValueError(f"not found: {path}")
    experiment = str(body.get("experiment") or "").strip()
    if not experiment:
        raise ValueError("experiment name or number is required")
    preset = str(body.get("preset") or "standard")
    if preset not in PRESETS:
        raise ValueError(f"unknown preset: {preset}")
    extra_raw = str(body.get("extra") or "").strip()
    extra = extra_raw.split() if extra_raw else []
    extra = strip_flag(extra, "--order")
    extra = strip_flag(extra, "--stop_file")
    mode = str(body.get("mode") or ("single" if path.is_file() else "batch"))
    py = venv_python()
    if mode == "single":
        if not path.is_file():
            raise ValueError("one-photo mode needs a photo file, not a folder")
        cmd = [py, "-u", str(SINGLE), "--input", str(path), "--experiment", experiment, "--preset", preset]
    else:
        cmd = [py, "-u", str(BATCH), "--input", str(path), "--experiment", experiment, "--preset", preset]
        if body.get("recursive"):
            cmd.append("--recursive")
        if body.get("redo"):
            cmd.append("--redo")
    if body.get("no_orbit"):
        cmd.append("--no_orbit")
    if body.get("debug"):
        cmd.append("--debug")
    cmd.extend(extra)
    return cmd, mode


def ingest_line(kind: str, raw: str) -> None:
    text = strip_ansi(raw).rstrip()
    if text.startswith("[video-event] "):
        try:
            event = json.loads(text[len("[video-event] "):])
            key = event["stage"]
            index = [v[0] for v in VIDEO_STAGES].index(key)
            fraction = min(1., max(0., float(event.get("fraction", 0.))))
        except (ValueError, KeyError, TypeError):
            return
        with STATE.lock:
            STATE.stage_index, STATE.stage_total = index+1, 4
            STATE.stage_title = VIDEO_STAGES[index][1]
            STATE.mark_stage(key, "done" if fraction == 1. else "running")
            STATE.tqdm_frac = fraction
            STATE.tqdm_desc = str(event.get("detail", ""))
            STATE.message = STATE.stage_title
        STATE.emit({"type": "stage", **STATE.snapshot()})
        return
    if not text and kind != "bar":
        return
    STATE.emit({"type": kind, "text": text})
    if kind == "bar":
        pct = TQDM_PCT_RE.search(text)
        left = TQDM_LEFT_RE.search(text)
        desc = re.match(r"^\s*(.+?):\s+\d", text)
        changed = False
        with STATE.lock:
            old_pct = round(STATE.tqdm_frac * 100)
            if pct:
                STATE.tqdm_frac = min(1.0, float(pct.group(1)) / 100.0)
            if left:
                STATE.tqdm_left = parse_clock(left.group(1))
            if desc:
                STATE.tqdm_desc = desc.group(1).strip()[:80]
            changed = round(STATE.tqdm_frac * 100) != old_pct
        if changed:
            STATE.emit({"type": "status", **STATE.snapshot()})
        return

    waiting = GPU_WAIT_RE.search(text)
    if waiting or "the GPU is free now" in text:
        with STATE.lock:
            STATE.message = (f"Waiting for the GPU: another job is using it ({waiting.group(1)[:120]})"
                             if waiting else "The GPU is free; starting")
        STATE.emit({"type": "status", **STATE.snapshot()})
        return

    spotted = error_from_text(text)
    if spotted:
        with STATE.lock:
            STATE.note_error(spotted)
            key = match_stage_key(STATE.stage_index, STATE.stage_total, STATE.stage_title) if STATE.stage_index else None
            if key and spotted.get("kind") != "hint":
                STATE.mark_stage(key, "failed")
        STATE.emit({"type": "error", **STATE.snapshot()})

    plan = PLAN_RE.match(text)
    if plan:
        status, name, rel = plan.group(1), plan.group(2).strip(), plan.group(3).strip()
        rel_key = rel.replace("\\", "/").casefold()
        with STATE.lock:
            hit = None
            for photo in STATE.photos:
                file_key = str(photo.get("file") or "").replace("\\", "/").casefold()
                if file_key == rel_key or photo.get("name") == name:
                    hit = photo
                    break
            if hit is None:
                STATE.photos.append({"name": name, "file": rel, "status": status, "current": False})
            else:
                hit["status"] = status
                hit["name"] = name
                hit["file"] = rel
        return

    batch = BATCH_RE.match(text)
    if batch:
        index, total, name = int(batch.group(1)), int(batch.group(2)), batch.group(3).strip()
        with STATE.lock:
            if STATE.image_started_at and STATE.image_index:
                STATE.finished_times.append(time.time() - STATE.image_started_at)
            STATE.image_index = index
            STATE.image_total = max(total, 1)
            STATE.image_name = name
            STATE.image_started_at = time.time()
            STATE.stage_index = 0
            STATE.tqdm_frac = 0.0
            STATE.tqdm_left = None
            STATE.tqdm_desc = ""
            STATE.reset_stages()
            STATE.status = "stopping" if STATE.stop_mode else "running"
            if STATE.stop_mode == "after":
                STATE.message = f"Finishing {name}, then stopping"
            else:
                STATE.message = f"Working on {name} ({index}/{total})"
            for photo in STATE.photos:
                file_name = Path(str(photo.get("file") or "")).name
                photo["current"] = (
                    file_name.casefold() == name.casefold()
                    or photo.get("name") == Path(name).stem
                    or str(photo.get("file") or "").replace("\\", "/").casefold().endswith("/" + name.casefold())
                )
                if photo["current"] and STATE.stop_mode == "after":
                    photo["status"] = "finishing, then stop"
        STATE.emit({"type": "image", **STATE.snapshot()})
        return

    item_done = BATCH_RESULT_RE.match(text)
    if item_done:
        status = item_done.group(1)
        with STATE.lock:
            for photo in STATE.photos:
                if photo.get("current"):
                    photo["status"] = status
                    photo["current"] = status not in ("failed", "stopped")
                    break
            if status == "failed":
                STATE.note_error({
                    "kind": "batch",
                    "title": "This character failed",
                    "message": text.strip(),
                    "hint": "The batch continues with the next photo if there is one. "
                            "Start again later without Rebuild to retry this one.",
                })
                key = match_stage_key(STATE.stage_index, STATE.stage_total, STATE.stage_title) if STATE.stage_index else None
                if key:
                    STATE.mark_stage(key, "failed")
        if status == "failed":
            STATE.emit({"type": "error", **STATE.snapshot()})
        else:
            STATE.emit({"type": "status", **STATE.snapshot()})
        return

    if BATCH_STOP_AFTER_RE.match(text):
        with STATE.lock:
            STATE.stop_mode = STATE.stop_mode or "after"
            for photo in STATE.photos:
                if photo.get("current"):
                    continue
                if str(photo.get("status") or "") in ("queued", "build", "todo", ""):
                    photo["status"] = "left for next start"
                    photo["current"] = False
        STATE.emit({"type": "status", **STATE.snapshot()})
        return

    stage = STAGE_RE.search(text)
    if stage:
        index, total, title = int(stage.group(1)), int(stage.group(2)), stage.group(3).strip()
        title = title.split(":")[0].strip()
        with STATE.lock:
            STATE.stage_index = index
            STATE.stage_total = max(total, 1)
            STATE.stage_title = title
            if "skipped" in text.casefold():
                STATE.tqdm_frac = 1.0
                flag = "skipped"
            elif "done in" in text.casefold():
                STATE.tqdm_frac = 1.0
                flag = "done"
            else:
                flag = "running"
                STATE.tqdm_frac = 0.0
            key = match_stage_key(index, STATE.stage_total, title)
            if key:
                STATE.mark_stage(key, flag)
                took = parse_done_in(text)
                if flag == "done" and took is not None:
                    STATE.stage_took[key] = took
            STATE.tqdm_left = None
            STATE.tqdm_desc = ""
            if not STATE.image_name and STATE.input_path:
                STATE.image_name = Path(STATE.input_path).name
                if not STATE.image_started_at:
                    STATE.image_started_at = time.time()
                    STATE.image_index = max(STATE.image_index, 1)
            STATE.message = title
        STATE.emit({"type": "stage", **STATE.snapshot()})


def pump_output(proc: subprocess.Popen) -> None:
    assert proc.stdout is not None
    buf = ""
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    try:
        while True:
            # Deliver flushed progress immediately instead of waiting to fill a buffer.
            chunk = proc.stdout.read1(4096)
            if not chunk:
                break
            buf += decoder.decode(chunk)
            buf = buf.replace("\r\n", "\n")
            while True:
                newline, cr = buf.find("\n"), buf.find("\r")
                if newline == -1 and cr == -1:
                    break
                if cr != -1 and (newline == -1 or cr < newline):
                    piece, buf = buf[:cr], buf[cr + 1:]
                    ingest_line("bar", piece)
                else:
                    piece, buf = buf[:newline], buf[newline + 1:]
                    ingest_line("log", piece.rstrip("\r"))
        if buf.strip():
            ingest_line("log", buf)
    except Exception as exc:  # noqa: BLE001
        STATE.emit({"type": "log", "text": f"[gui] reader error: {exc}"})


def start_run(body: dict) -> dict:
    with STATE.lock:
        if STATE.running:
            return {"ok": False, "error": "a run is already in progress"}
    try:
        cmd, mode = build_command(body)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    path_obj = Path(body["input"]).expanduser().resolve()
    path = str(path_obj)
    recursive = bool(body.get("recursive"))
    order = order_lines(body.get("order"))
    photos, order_warn = ([path_obj], []) if mode == "video" else ordered_photos(path_obj, recursive, order if mode == "batch" else None)
    if mode == "batch" and not photos:
        return {"ok": False, "error": f"no photos in {path}"}
    root = path_obj if path_obj.is_dir() else path_obj.parent
    if mode == "batch" and photos:
        rels = [photo_rel(p, root) for p in photos]
        experiment = str(body.get("experiment") or "").strip()
        order_file = write_order_file(experiment, rels)
        cmd.extend(["--order", str(order_file)])
        stop_path = REPO_ROOT / "output" / experiment / "stop.txt"
        cmd.extend(["--stop_file", str(stop_path)])
    else:
        stop_path = None
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["COLUMNS"] = "100"
    kwargs: dict = {"args": cmd, "cwd": str(REPO_ROOT), "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT,
                    "env": env}
    if os.name == "nt":
        kwargs["creationflags"] = CREATE_NEW_PROCESS_GROUP
    try:
        proc = subprocess.Popen(**kwargs)
    except OSError as exc:
        return {"ok": False, "error": f"could not start the pipeline: {exc}"}

    with STATE.lock:
        STATE.proc = proc
        STATE.running = True
        STATE.mode = mode
        STATE.input_path = path
        STATE.experiment = str(body.get("experiment") or "")
        STATE.preset = str(body.get("preset") or "standard")
        extra_raw = str(body.get("extra") or "")
        STATE.no_orbit = mode != "video" and (bool(body.get("no_orbit")) or "--no_orbit" in extra_raw.split())
        STATE.reset_stages()
        named = run_names(photos) if photos else {}
        STATE.photos = [{
            "name": named.get(p, safe_run_name(p.stem)),
            "file": photo_rel(p, root),
            "path": str(p.resolve()),
            "status": "queued",
            "current": False,
        } for p in photos]
        if mode in ("single", "video") and STATE.photos:
            STATE.photos[0]["current"] = True
        STATE.started_at = time.time()
        STATE.image_started_at = time.time() if mode in ("single", "video") else 0.0
        STATE.image_index = 1 if mode in ("single", "video") else 0
        STATE.image_total = max(len(photos), 1) if mode == "batch" else 1
        STATE.image_name = Path(path).name if mode in ("single", "video") else ""
        STATE.stage_index = 0
        STATE.stage_total = 4 if mode == "video" else 7
        STATE.stage_title = ""
        STATE.tqdm_frac = 0.0
        STATE.tqdm_left = None
        STATE.tqdm_desc = ""
        STATE.finished_times = []
        STATE.exit_code = None
        STATE.status = "running"
        STATE.message = "Starting video reconstruction…" if mode == "video" else "Starting… finished photos are skipped, so a stopped batch continues."
        STATE.stop_mode = None
        STATE.error = None
        STATE.stop_file = stop_path
        if stop_path is not None:
            stop_path.unlink(missing_ok=True)
    STATE.emit({"type": "log", "text": "$ " + subprocess.list2cmdline(cmd)})
    if body.get("redo"):
        STATE.emit({"type": "log", "text": "[gui] Rebuild is on: finished splats will be built again."})
    elif mode == "batch":
        STATE.emit({"type": "log", "text":
                    "[gui] Finished splats are skipped, so a stopped batch continues where it left off."})
    for msg in order_warn:
        STATE.emit({"type": "log", "text": f"[gui] order: {msg}"})
    STATE.emit({"type": "status", **STATE.snapshot()})
    threading.Thread(target=_watch, args=(proc,), daemon=True).start()
    return {"ok": True, "pid": proc.pid, "mode": mode, "cmd": cmd}


def write_stop_record(kind: str, code: int | None) -> None:
    exp = (STATE.experiment or "").strip()
    if not exp:
        return
    dest = REPO_ROOT / "output" / exp
    try:
        dest.mkdir(parents=True, exist_ok=True)
        payload = {
            "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "stop": kind,
            "exit_code": code,
            "photo": STATE.image_name,
            "experiment": exp,
            "resume": "Start again without Rebuild finished ones. Already-built splats are skipped; "
                      "an interrupted photo continues from its last finished stage.",
        }
        (dest / "last_stop.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    except OSError:
        pass


def free_gpu() -> None:
    """Stop the ComfyUI this pipeline started so VRAM is released after an interrupt."""
    py = venv_python()
    STATE.emit({"type": "log", "text": "[gui] stopping ComfyUI to free the GPU…"})
    try:
        out = subprocess.run([py, str(SINGLE), "--stop_comfy"], cwd=str(REPO_ROOT),
                             capture_output=True, text=True, timeout=90)
    except (OSError, subprocess.TimeoutExpired) as exc:
        STATE.emit({"type": "log", "text": f"[gui] could not stop ComfyUI: {exc}"})
        return
    note = (out.stdout or out.stderr or "").strip().splitlines()
    STATE.emit({"type": "log", "text": "[gui] " + (note[-1] if note else "GPU released.")})


def _watch(proc: subprocess.Popen) -> None:
    pump_output(proc)
    code = proc.wait()
    with STATE.lock:
        stop_mode = STATE.stop_mode
        who = STATE.image_name or "the current character"
        STATE.running = False
        STATE.proc = None
        STATE.exit_code = code
        if stop_file := STATE.stop_file:
            try:
                stop_file.unlink(missing_ok=True)
            except OSError:
                pass
        if stop_mode == "after" or code == STOP_AFTER_EXIT:
            STATE.stop_mode = "after"
            STATE.status = "stopped_after"
            STATE.message = (
                f"Stopped after {who}. Remaining photos will run when you Start again "
                "(leave Rebuild finished ones unticked)."
            )
            STATE.tqdm_frac = 1.0
        elif stop_mode == "now" or is_interrupt_code(code):
            STATE.stop_mode = "now"
            STATE.status = "stopped"
            STATE.message = (
                f"Stopped immediately during {who}. GPU is being freed. "
                "Start again to continue this character, then the rest of the batch."
            )
        elif code == 0:
            STATE.status = "done"
            STATE.message = "Finished."
            STATE.tqdm_frac = 1.0
            STATE.stop_mode = None
            STATE.error = None
            if STATE.mode == "video":
                for item in STATE.photos:
                    item["status"], item["current"] = "done", False
            for key in STATE.stage_flags:
                flag = STATE.stage_flags.get(key)
                if flag == "running":
                    STATE.mark_stage(key, "done")
                elif flag == "pending":
                    STATE.stage_flags[key] = "done"
        else:
            STATE.status = "failed"
            failed_n = sum(1 for p in STATE.photos if str(p.get("status") or "") == "failed")
            if STATE.mode == "batch" and failed_n:
                STATE.note_error({
                    "kind": "batch",
                    "title": f"{failed_n} character(s) failed",
                    "message": (STATE.error or {}).get("message") or f"Batch exit {code}.",
                    "hint": "Start again without Rebuild to retry the failed ones. Finished splats stay skipped.",
                }, fatal=True)
            else:
                STATE.note_error(error_from_exit(code), fatal=True)
            err = STATE.error or {}
            STATE.message = err.get("title") or f"Failed (exit {code})."
            key = match_stage_key(STATE.stage_index, STATE.stage_total, STATE.stage_title) if STATE.stage_index else None
            if key:
                STATE.mark_stage(key, "failed")
    should_free = (
        stop_mode in ("after", "now")
        or is_interrupt_code(code)
        or code == STOP_AFTER_EXIT
        or (code not in (0, None) and not is_interrupt_code(code))
    )
    if should_free:
        free_gpu()
        if stop_mode in ("after", "now") or is_interrupt_code(code) or code == STOP_AFTER_EXIT:
            write_stop_record("after" if (stop_mode == "after" or code == STOP_AFTER_EXIT) else "now", code)
        else:
            write_stop_record("failed", code)
    STATE.emit({"type": "done", "code": code, **STATE.snapshot()})


def stop_run(how: str = "now") -> dict:
    how = "after" if how == "after" else "now"
    with STATE.lock:
        proc = STATE.proc
        if not STATE.running or proc is None:
            return {"ok": False, "error": "nothing is running"}
        if STATE.stop_mode == "now":
            return {"ok": False, "error": "already stopping"}
        if how == "after" and STATE.stop_mode == "after":
            return {"ok": True, "how": "after", "already": True}
        if how == "after" and STATE.mode != "batch":
            how = "now"
        STATE.stop_mode = how
        STATE.status = "stopping"
        who = STATE.image_name or "this character"
        if how == "after":
            STATE.message = (
                f"Stop requested: {who} will finish, then the batch ends. "
                "Start is disabled until then. Stop now still interrupts immediately."
            )
            for photo in STATE.photos:
                if photo.get("current"):
                    photo["status"] = "finishing, then stop"
                elif str(photo.get("status") or "") in ("queued", "build", "todo", ""):
                    photo["status"] = "left for next start"
            stop_path = STATE.stop_file
        else:
            STATE.message = f"Stopping now ({who}) and freeing the GPU…"
            stop_path = None

    if how == "after":
        if stop_path is None:
            with STATE.lock:
                STATE.stop_mode = None
                STATE.status = "running"
            return {"ok": False, "error": "no stop file for this run"}
        try:
            stop_path.parent.mkdir(parents=True, exist_ok=True)
            stop_path.write_text("stop after current photo\n", encoding="utf-8")
        except OSError as exc:
            with STATE.lock:
                STATE.stop_mode = None
                STATE.status = "running"
            return {"ok": False, "error": f"could not request stop: {exc}"}
        STATE.emit({"type": "log", "text":
                    f"[gui] stop after this character: {who} will finish, then the rest wait. "
                    "Start again without Rebuild to continue."})
        STATE.emit({"type": "status", **STATE.snapshot()})
        return {"ok": True, "how": "after"}

    STATE.emit({"type": "log", "text":
                "[gui] stop now: interrupting this character, ending the batch, freeing the GPU. "
                "Start again without Rebuild to continue."})
    STATE.emit({"type": "status", **STATE.snapshot()})
    try:
        if os.name == "nt":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.send_signal(signal.SIGINT)
    except OSError:
        proc.terminate()
    threading.Timer(8.0, _kill_if_needed, args=(proc,)).start()
    return {"ok": True, "how": "now"}


def _kill_if_needed(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    STATE.emit({"type": "log", "text": "[gui] still running — killing the process tree"})
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    else:
        proc.kill()
    if proc.poll() is None:
        proc.kill()
        STATE.emit({"type": "log", "text": "[gui] process killed"})


class GuiHTTPServer(ThreadingHTTPServer):
    """One listener per port. Windows SO_REUSEADDR otherwise lets several GUIs bind 8765
    at once, and Chrome then shows ERR_EMPTY_RESPONSE."""

    allow_reuse_address = False
    daemon_threads = True

    def server_bind(self) -> None:
        if sys.platform == "win32" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def _listening_pids(port: int) -> set[int]:
    try:
        raw = subprocess.check_output(
            ["netstat", "-ano", "-p", "tcp"], text=True, errors="replace", timeout=8,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return set()
    locals_ = {
        f"{HOST}:{port}",
        f"0.0.0.0:{port}",
        f"[::]:{port}",
        f"[::1]:{port}",
    }
    pids: set[int] = set()
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[-2].upper() != "LISTENING":
            continue
        if parts[1] in locals_ and parts[-1].isdigit():
            pids.add(int(parts[-1]))
    return pids


def _splat_gui_process_table() -> dict[int, tuple[int, str]]:
    """pid -> (parent_pid, command line) for Image-to-Gaussian GUI processes."""
    if sys.platform != "win32":
        return {}
    script = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.CommandLine -like '*splat_gui.py*' } | "
        "Select-Object ProcessId, ParentProcessId, CommandLine | "
        "ConvertTo-Json -Compress"
    )
    try:
        raw = subprocess.check_output(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
            text=True, errors="replace", timeout=12,
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {}
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    rows = payload if isinstance(payload, list) else [payload]
    table: dict[int, tuple[int, str]] = {}
    for row in rows:
        try:
            pid = int(row.get("ProcessId"))
            ppid = int(row.get("ParentProcessId") or 0)
        except (TypeError, ValueError):
            continue
        table[pid] = (ppid, str(row.get("CommandLine") or ""))
    return table


def _is_our_gui_cmd(cmdline: str) -> bool:
    text = cmdline.casefold().replace("/", "\\")
    if "splat_gui.py" not in text:
        return False
    repo = str(REPO_ROOT).casefold().replace("/", "\\")
    return repo in text or "splat_gui.py" in text


def _kill_pid_tree(pid: int) -> None:
    if pid <= 0:
        return
    creationflags = 0
    if sys.platform == "win32":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.run(
        ["taskkill", "/F", "/T", "/PID", str(pid)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=12,
        creationflags=creationflags,
    )


def reclaim_stale_gui(port: int) -> bool:
    """Kill hung Image-to-Gaussian GUI listeners on this port so a new one can bind."""
    listeners = _listening_pids(port)
    if not listeners:
        return False
    table = _splat_gui_process_table()
    ours = {pid for pid in listeners if _is_our_gui_cmd(table.get(pid, (0, ""))[1])}
    if not ours:
        return False
    roots: set[int] = set()
    for pid in ours:
        ppid = table.get(pid, (0, ""))[0]
        if ppid in table and _is_our_gui_cmd(table[ppid][1]):
            roots.add(ppid)
        else:
            roots.add(pid)
    print(f"[gui] port {port} is stuck on a previous GUI (pid {', '.join(map(str, sorted(ours)))}). Taking it over.", flush=True)
    for pid in roots:
        _kill_pid_tree(pid)
    deadline = time.time() + 4.0
    while time.time() < deadline:
        left = _listening_pids(port) & ours
        if not left:
            return True
        time.sleep(0.15)
    return not (_listening_pids(port) & ours)


def probe_gui(port: int) -> dict | None:
    req = urllib.request.Request(
        f"http://{HOST}:{port}/api/status",
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if data.get("app") == "i2g-gui":
        return data
    if "status" in data and "message" in data:
        return data
    return None


def bind_gui(port: int) -> GuiHTTPServer | None:
    try:
        return GuiHTTPServer((HOST, port), Handler)
    except OSError:
        return None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("[gui] " + (fmt % args) + "\n")

    def _send(self, status: int, body: bytes, content_type: str, extra: dict | None = None) -> None:
        extra = extra or {}
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if "Cache-Control" not in extra:
            self.send_header("Cache-Control", "no-store")
        for key, value in extra.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict) -> None:
        raw = json.dumps(payload, default=str).encode("utf-8")
        self._send(status, raw, "application/json; charset=utf-8")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid JSON: {exc}") from exc
        return data if isinstance(data, dict) else {}

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            page = (GUI_DIR / "index.html").read_bytes()
            self._send(200, page, "text/html; charset=utf-8")
            return
        if path == "/api/status":
            self._json(200, STATE.snapshot())
            return
        if path == "/api/experiments":
            self._json(200, {"experiments": existing_experiments()})
            return
        if path == "/api/presets":
            self._json(200, {
                "presets": list(PRESETS),
                "minutes": PRESET_MINUTES,
                "plans": {**{name: stage_plan(name, False) for name in PRESETS}, "video": video_stage_plan()},
            })
            return
        if path == "/api/events":
            self._sse()
            return
        if path == "/api/thumb":
            self._thumb()
            return
        if path == "/api/media":
            self._media()
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            body = self._read_json()
        except ValueError as exc:
            self._json(400, {"ok": False, "error": str(exc)})
            return
        if path == "/api/browse":
            picked = browse(str(body.get("kind") or "file"))
            self._json(200, {"ok": True, "path": picked, "cancelled": picked is None})
            return
        if path == "/api/preview":
            self._json(200, preview(str(body.get("input") or ""), bool(body.get("recursive")), str(body.get("mode") or "")))
            return
        if path == "/api/run":
            try:
                result = start_run(body)
            except Exception as exc:
                result = {"ok": False, "error": f"could not start: {exc}"}
            self._json(200 if result.get("ok") else 400, result)
            return
        if path == "/api/stop":
            how = str(body.get("how") or "now")
            self._json(200, stop_run(how))
            return
        if path == "/api/open_folder":
            result = open_character_folder(body)
            self._json(200 if result.get("ok") else 400, result)
            return
        self._json(404, {"error": "not found"})

    def _thumb(self) -> None:
        qs = parse_qs(urlparse(self.path).query)
        raw = (qs.get("p") or [""])[0]
        image = _safe_image(raw)
        if image is None:
            self._json(404, {"error": "image not found"})
            return
        try:
            from io import BytesIO
            from PIL import Image
            with Image.open(image) as im:
                has_alpha = im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info)
                im.thumbnail((72, 96))
                buf = BytesIO()
                if has_alpha:
                    im = im.convert("RGBA")
                    im.save(buf, "PNG")
                    ctype = "image/png"
                else:
                    im = im.convert("RGB")
                    im.save(buf, "JPEG", quality=70)
                    ctype = "image/jpeg"
                body = buf.getvalue()
        except Exception:
            self._json(415, {"error": "could not read image"})
            return
        self._send(200, body, ctype, {"Cache-Control": "private, max-age=3600"})

    def _media(self) -> None:
        qs = parse_qs(urlparse(self.path).query)
        raw = (qs.get("p") or [""])[0]
        path = _safe_media(raw)
        if path is None:
            self._json(404, {"error": "media not found"})
            return
        self._stream_file(path, MEDIA_TYPES[path.suffix.lower()])

    def _stream_file(self, path: Path, content_type: str) -> None:
        try:
            size = path.stat().st_size
            fh = path.open("rb")
        except OSError:
            self._json(404, {"error": "media not found"})
            return
        rng = _parse_range(self.headers.get("Range") or "", size)
        try:
            if rng:
                start, end = rng
                length = end - start + 1
                fh.seek(start)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                length = size
                self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "private, max-age=30")
            self.end_headers()
            remaining = length
            while remaining > 0:
                chunk = fh.read(min(256 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, ConnectionError):
            pass
        finally:
            fh.close()

    def _sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        q = STATE.subscribe()
        try:
            self.wfile.write(b"data: " + json.dumps({"type": "hello", **STATE.snapshot()}).encode("utf-8") + b"\n\n")
            self.wfile.flush()
            while True:
                try:
                    event = q.get(timeout=1.0)
                except queue.Empty:
                    tick = {"type": "tick", **STATE.snapshot()}
                    self.wfile.write(b"data: " + json.dumps(tick).encode("utf-8") + b"\n\n")
                    self.wfile.flush()
                    continue
                self.wfile.write(b"data: " + json.dumps(event, default=str).encode("utf-8") + b"\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        finally:
            STATE.unsubscribe(q)


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Browser UI for Image-to-Gaussian.")
    parser.add_argument("--port", type=int, help="port (default: first free port from 8765)")
    parser.add_argument("--no-open", action="store_true", help="do not open a browser tab")
    args = parser.parse_args(argv)
    if not (GUI_DIR / "index.html").exists():
        print(f"ERROR: missing {GUI_DIR / 'index.html'}")
        return 2
    httpd = None
    port = None
    wanted = [args.port] if args.port else list(PORTS)
    for candidate in wanted:
        existing = probe_gui(candidate)
        if existing:
            url = f"http://{HOST}:{candidate}/"
            print(f"Image-to-Gaussian GUI already running at {url}", flush=True)
            if args.no_open:
                print("This window can be closed.", flush=True)
            else:
                print("Opened a browser tab. You can close this window.", flush=True)
                webbrowser.open(url)
            return 0
        httpd = bind_gui(candidate)
        if httpd is None and reclaim_stale_gui(candidate):
            httpd = bind_gui(candidate)
        if httpd is not None:
            port = candidate
            break
        if args.port:
            print(f"ERROR: port {args.port} is already in use by another program.")
            return 1
    if httpd is None or port is None:
        print("ERROR: could not bind a local port (tried 8765–8779).")
        return 1
    url = f"http://{HOST}:{port}/"
    print(f"Image-to-Gaussian GUI  {url}", flush=True)
    print("Leave this window open. Close it (or Ctrl-C) to stop the UI.", flush=True)
    if not args.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping GUI.")
    finally:
        httpd.server_close()
        if STATE.running:
            stop_run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

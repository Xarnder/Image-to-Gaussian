"""One Image-to-Gaussian run on the GPU at a time.

Two runs at once do not fail cleanly on this PC: a 24 GB card and 32 GB of system RAM hold one Qwen /
Wan model plus one training, not two of each. Windows then pages model weights between VRAM, RAM and
the disk, and a Qwen step that takes 2 s takes 20+ minutes (2026-09-28: a stray test run training and
running Wan while a GUI batch generated views; the batch's Qwen steps took 743 s and 1652 s, and it
failed with "GPU ran out of memory").

So every run takes a machine-wide lock before it touches the GPU, and a second run waits for the first
to finish instead of colliding with it. The lock is an OS file lock: it is released when the process
exits, even if it crashes or is killed, so it can never go stale. Runs from older code (or research
scripts) do not take the lock, so the waiting run also looks for their processes by command line.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .paths import REPO_ROOT

LOCK_DIR = REPO_ROOT / "cache"
LOCK_FILE = LOCK_DIR / "gpu.lock"
INFO_FILE = LOCK_DIR / "gpu.lock.json"
# command-line fragments of processes that use the GPU heavily (the pipeline and its research scripts)
GPU_JOBS = ("image_to_splat.py", "i2g.experiment", "i2g.reregister", "-m headpass", "-m pacgs", "pacgs\\run",
            "pacgs/run")


def _lock_fd(fd: int) -> bool:
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _processes() -> list[dict]:
    """[{pid, ppid, cmd}] for every python process (Windows: via CIM; elsewhere: ps)."""
    try:
        if os.name == "nt":
            out = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 "Get-CimInstance Win32_Process -Filter \"name like 'python%'\" | "
                 "Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"],
                capture_output=True, text=True, timeout=30).stdout.strip()
            rows = json.loads(out) if out else []
            rows = rows if isinstance(rows, list) else [rows]
            return [{"pid": r["ProcessId"], "ppid": r["ParentProcessId"], "cmd": r.get("CommandLine") or ""}
                    for r in rows]
        out = subprocess.run(["ps", "-eo", "pid=,ppid=,args="], capture_output=True, text=True, timeout=30).stdout
        rows = []
        for line in out.splitlines():
            parts = line.split(None, 2)
            if len(parts) == 3 and "python" in parts[2]:
                rows.append({"pid": int(parts[0]), "ppid": int(parts[1]), "cmd": parts[2]})
        return rows
    except (OSError, ValueError, subprocess.SubprocessError):
        return []


def other_gpu_jobs() -> list[dict]:
    """Other pipeline or research processes running now (not this one, its parents or its children)."""
    procs = _processes()
    by_pid = {p["pid"]: p for p in procs}
    family, pid = set(), os.getpid()
    while pid in by_pid and pid not in family:          # this process and its ancestors (batch, GUI)
        family.add(pid)
        pid = by_pid[pid]["ppid"]
    family.add(os.getpid())
    family.add(os.getppid())
    changed = True
    while changed:                                     # ... and its descendants
        changed = False
        for p in procs:
            if p["ppid"] in family and p["pid"] not in family and p["ppid"] != p["pid"]:
                family.add(p["pid"])
                changed = True
    jobs = [p for p in procs if p["pid"] not in family and any(k in p["cmd"] for k in GPU_JOBS)]
    # a venv launcher and the interpreter it starts are one job: keep the outermost
    pids = {p["pid"] for p in jobs}
    return [p for p in jobs if p["ppid"] not in pids]


def describe(jobs: list[dict]) -> str:
    out = []
    for p in jobs:
        cmd = p["cmd"]
        cut = min((cmd.find(k) for k in GPU_JOBS if k in cmd), default=0)
        out.append(f"PID {p['pid']}: {cmd[cut:cut + 160]}")
    return "; ".join(out)


def holder() -> dict | None:
    """Who holds the lock (as it wrote itself down), or None."""
    try:
        return json.loads(INFO_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class GpuLock:
    def __init__(self) -> None:
        self._fh = None

    def try_acquire(self, label: str) -> bool:
        LOCK_DIR.mkdir(parents=True, exist_ok=True)
        fh = open(LOCK_FILE, "a+b")
        if not _lock_fd(fh.fileno()):
            fh.close()
            return False
        self._fh = fh
        INFO_FILE.write_text(json.dumps({"pid": os.getpid(), "run": label, "since": time.strftime("%H:%M"),
                                         "command": " ".join(sys.argv)[:300]}), encoding="utf-8")
        return True

    def release(self) -> None:
        if self._fh is not None:
            try:
                INFO_FILE.unlink(missing_ok=True)
            except OSError:
                pass
            self._fh.close()
            self._fh = None


def acquire(label: str, log, poll_s: float = 20.0, report_every_s: float = 300.0) -> GpuLock:
    """Block until no other run uses the GPU, then hold the lock (release() or process exit frees it)."""
    lock = GpuLock()
    waited, last_report, first = 0.0, -report_every_s, True
    while True:
        if lock.try_acquire(label):
            others = other_gpu_jobs()           # older runs and research scripts do not take the lock
            if not others:
                if waited:
                    log.info("  ok   the GPU is free now (waited %.0f min)", waited / 60)
                return lock
            lock.release()
            who = describe(others)
        else:
            h = holder() or {}
            who = f"PID {h.get('pid', '?')}: {h.get('run', 'another run')} (since {h.get('since', '?')})"
        if first or waited - last_report >= report_every_s:
            log.warning("Waiting for the GPU: another Image-to-Gaussian job is using it (%s). Two at once "
                        "make both crawl or fail, so this run starts when that one finishes%s.", who,
                        "" if first else f" (waiting {waited / 60:.0f} min)")
            last_report, first = waited, False
        time.sleep(poll_s)
        waited += poll_s

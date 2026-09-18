"""Per-stage GPU memory accounting and the VRAM budget guard (plan.md §5)."""
from __future__ import annotations

import contextlib
import time

import torch

BUDGET_GIB = 20.0
GIB = 1024 ** 3


class VramBudgetExceeded(RuntimeError):
    pass


@contextlib.contextmanager
def stage(name: str, report: dict | None = None, budget_gib: float = BUDGET_GIB):
    """Record wall time and peak allocated/reserved VRAM of a block; fail if it broke the budget."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    yield
    torch.cuda.synchronize()
    entry = {
        "stage": name,
        "seconds": round(time.perf_counter() - start, 2),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / GIB, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / GIB, 3),
    }
    if report is not None:
        report.setdefault("stages", []).append(entry)
    if entry["peak_allocated_gib"] > budget_gib:
        raise VramBudgetExceeded(f"{name} peaked at {entry['peak_allocated_gib']} GiB (budget {budget_gib} GiB)")


def free_and_total_gib() -> tuple[float, float]:
    free, total = torch.cuda.mem_get_info()
    return free / GIB, total / GIB

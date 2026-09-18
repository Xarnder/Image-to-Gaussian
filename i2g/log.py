"""Logging for the v2 pipeline.

Console lines go through tqdm.write so they never break a progress bar. Everything, including DEBUG
detail and Python warnings, also goes to a log file for diagnosing failures.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT_NAME = "i2g"


def get_logger(name: str = "") -> logging.Logger:
    return logging.getLogger(f"{ROOT_NAME}.{name}" if name else ROOT_NAME)


class _TqdmHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            from tqdm import tqdm
            tqdm.write(self.format(record), file=sys.stdout)
        except Exception:  # noqa: BLE001 - logging must never raise
            self.handleError(record)


def setup_logging(log_file: Path | None, debug: bool = False) -> None:
    root = logging.getLogger(ROOT_NAME)
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.propagate = False

    console = _TqdmHandler()
    console.setLevel(logging.DEBUG if debug else logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
        root.addHandler(file_handler)
        # library warnings (torch, gsplat, transformers) go to the file only, not the console
        logging.captureWarnings(True)
        warnings_logger = logging.getLogger("py.warnings")
        warnings_logger.handlers.clear()
        warnings_logger.addHandler(file_handler)
        warnings_logger.propagate = False


class NullBar:
    """Stand-in for a tqdm bar when no progress display is wanted."""

    def update(self, n: int = 1) -> None:
        pass

    def set_postfix_str(self, s: str = "", refresh: bool = True) -> None:
        pass

    def close(self) -> None:
        pass

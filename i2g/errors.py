"""Errors that carry a plain-language fix for the person running the tool."""
from __future__ import annotations


class PipelineError(RuntimeError):
    """A failure the user can act on: `str(err)` says what broke, `err.hint` says how to fix it."""

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.hint = hint

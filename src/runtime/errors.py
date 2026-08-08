"""Structured error types for the runtime.

Every error carries a stable ``code`` string so downstream consumers
(monitoring, eval, retry logic) can switch on codes rather than
exception type strings.
"""

from __future__ import annotations

from typing import Any


class RuntimeError(Exception):
    """Base for all runtime-raised errors.

    Not to be confused with the Python builtin ``RuntimeError`` — this is
    the runtime's own structured error base. Callers should catch the
    specific subclasses below.
    """

    def __init__(
        self,
        *,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


class RetryableError(RuntimeError):
    """A transient failure the caller may retry (timeout, Gemini 500, etc.)."""


class InvalidInputError(RuntimeError):
    """The input is fundamentally invalid (corrupt image, wrong type, etc.).

    Retrying with the same input will not help.
    """

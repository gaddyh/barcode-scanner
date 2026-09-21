"""Structured error types for the runtime.

Every error carries a stable ``code`` string so downstream consumers
(monitoring, eval, retry logic) can switch on codes rather than
exception type strings.

Error taxonomy (ported from echo-v2, merged with barcode-scanner's
structured ``code``/``message``/``details`` pattern):

- ``ExecutionError`` — base for all runtime execution failures.
- ``RetryableError`` — a transient failure that may succeed on a later
  attempt (timeout, Gemini 500, connection refused).
- ``TimeoutError`` — an operation exceeded its configured timeout.
  Retryable.
- ``PermanentError`` — a failure that should not be retried.
- ``IndeterminateError`` — an operation whose outcome is unknown (e.g.
  timed out mid-flight on an irreversible write). Not retryable, not
  permanent: the side effect may or may not have happened. For
  idempotent irreversible writes, this is persisted as an
  ``IndeterminateOutcome`` so the same key cannot re-run until
  reconciled.
- ``InvalidInputError`` — the input is fundamentally invalid. Subclass
  of ``PermanentError`` (retrying with the same input will not help).
"""

from __future__ import annotations

from typing import Any


class ExecutionError(Exception):
    """Base class for all runtime execution failures.

    Carries a stable ``code`` string and optional ``details`` dict so
    downstream consumers (monitoring, eval, retry logic) can switch on
    codes rather than exception type strings.
    """

    def __init__(
        self,
        message: str = "",
        *,
        code: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or type(self).__name__
        self.details = details or {}


class RetryableError(ExecutionError):
    """A transient failure that may succeed on a later attempt."""


class TimeoutError(RetryableError):
    """An operation exceeded its configured timeout. Retryable."""


class PermanentError(ExecutionError):
    """A failure that should not be retried."""


class IndeterminateError(ExecutionError):
    """An operation whose outcome is unknown (e.g. timed out mid-flight).

    Not retryable, not permanent: the side effect may or may not have
    happened. For idempotent irreversible writes, this is persisted as an
    ``IndeterminateOutcome`` so the same key cannot re-run until
    reconciled.
    """


class InvalidInputError(PermanentError):
    """The input is fundamentally invalid (corrupt image, wrong type, etc.).

    Subclass of ``PermanentError`` — retrying with the same input will
    not help.
    """

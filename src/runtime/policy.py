"""ExecutionPolicy — per-operation retry, timeout, and safety configuration.

Ported from echo-v2's ``ExecutionPolicy`` structure, but with
product-specific timeout values derived from barcode-scanner's own P95
measurements (NOT copied from echo-v2). A scanner regression should not
happen because echo-v2 happened to use 5 seconds.

P95 baselines (from ``tests/eval/baseline_frozen.json``):
- Scanner-only P95: ~2.3s → SCAN_COMPUTE timeout = P95 × 3 ≈ 7s
- Gemini audit P95: ~4.6s → EXTERNAL_READ timeout = 15s (headroom)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionPolicy:
    """Retry, timeout, and safety configuration for one operation type.

    Attributes:
        max_attempts: Maximum number of attempts (including the first).
            ``max_attempts=1`` means no retry.
        timeout_seconds: Per-attempt timeout in seconds. ``None`` means no
            timeout. Must be ``> 0`` if set.
        retry_delay_seconds: Delay between retry attempts in seconds.
            ``0.0`` means no delay.
        irreversible_write: If ``True``, an unexpected failure or timeout
            is treated as ``IndeterminateError`` (the side effect may have
            happened). The executor does NOT retry irreversible writes —
            safety is structural, not dependent on ``max_attempts=1``.
    """

    max_attempts: int = 1
    timeout_seconds: float | None = None
    retry_delay_seconds: float = 0.0
    irreversible_write: bool = False

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {self.max_attempts}")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError(
                f"timeout_seconds must be > 0 or None, got {self.timeout_seconds}"
            )
        if self.retry_delay_seconds < 0:
            raise ValueError(
                f"retry_delay_seconds must be >= 0, got {self.retry_delay_seconds}"
            )


# --- Product-specific policies ---------------------------------------------

#: No retry, no timeout. Generic preset for tests and trivial operations.
NO_RETRY = ExecutionPolicy(max_attempts=1)

#: Local deterministic scanner. No retry — the scanner is deterministic,
#: retrying the same image is pointless. Timeout = scanner P95 × 3 for
#: headroom on slow images, but bounded.
SCAN_COMPUTE = ExecutionPolicy(
    max_attempts=1,
    timeout_seconds=7.0,
)

#: Gemini audit (read-like, retryable). ``max_attempts=3`` with bounded
#: timeout and short backoff. 15s is a starting point derived from
#: observed Gemini P95 (~4.6s) with headroom.
EXTERNAL_READ = ExecutionPolicy(
    max_attempts=3,
    timeout_seconds=15.0,
    retry_delay_seconds=0.5,
)

#: Priority draft order (irreversible write). ``max_attempts=1`` — the
#: executor does NOT retry irreversible writes. An unexpected failure or
#: timeout is treated as ``IndeterminateError`` (the side effect may have
#: happened). Idempotency is required for safe retry.
EXTERNAL_WRITE = ExecutionPolicy(
    max_attempts=1,
    timeout_seconds=10.0,
    irreversible_write=True,
)

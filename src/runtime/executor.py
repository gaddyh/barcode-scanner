"""execute() — the minimal runtime execution boundary.

Milestone 1: timing, error normalization, sync-to-async bridging only.
Milestone 2 plugs observability (LangSmith tracing, metadata propagation)
into this function.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any, TypeVar

from src.runtime.context import RunContext
from src.runtime.errors import InvalidInputError, RetryableError

T = TypeVar("T")


async def execute(
    operation: Callable[..., T],
    input: Any,
    context: RunContext,
    *,
    timeout: float | None = None,
) -> T:
    """Execute ``operation(input, context=context)`` within the runtime.

    Sync operations are bridged to a thread via ``asyncio.to_thread``.
    Async operations are awaited directly. Exceptions are normalized into
    structured ``RetryableError`` / ``InvalidInputError`` types.

    Args:
        operation: A callable accepting ``(input, context=context)``.
            May be sync or async.
        input: The primary input passed to ``operation``.
        context: The ``RunContext`` for this execution.
        timeout: Optional timeout in seconds. If exceeded, raises
            ``RetryableError`` with code ``"timeout"``.

    Returns:
        Whatever ``operation`` returns.

    Raises:
        RetryableError: On timeout or any unexpected exception.
        InvalidInputError: On invalid input (e.g. wrong type).
    """
    start = time.monotonic()

    async def _run() -> T:
        if asyncio.iscoroutinefunction(operation):
            return await operation(input, context=context)
        return await asyncio.to_thread(operation, input, context=context)

    try:
        if timeout is not None:
            result = await asyncio.wait_for(_run(), timeout=timeout)
        else:
            result = await _run()
        return result
    except TimeoutError as exc:
        raise RetryableError(
            code="timeout",
            message=f"Operation timed out after {timeout}s",
            details={"timeout_seconds": timeout, "elapsed": time.monotonic() - start},
        ) from exc
    except (RetryableError, InvalidInputError):
        # Already structured — pass through.
        raise
    except Exception as exc:
        raise RetryableError(
            code=type(exc).__name__,
            message=str(exc),
            details={"elapsed": time.monotonic() - start},
        ) from exc

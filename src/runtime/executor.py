"""execute() — the runtime execution boundary with observability.

Wraps any operation with LangSmith tracing (when enabled), timing,
error normalization, and sync-to-async bridging. The ``RunContext.run_id``
is added to trace metadata as ``run_id`` — it is an application
correlation ID, NOT the LangSmith run ID.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any, TypeVar

from src.observability.tracing import emit_metadata, trace_operation
from src.runtime.context import RunContext
from src.runtime.errors import InvalidInputError, RetryableError

T = TypeVar("T")


async def execute(
    operation: Callable[..., T],
    input: Any,
    context: RunContext,
    *,
    name: str | None = None,
    run_type: str = "chain",
    tags: list[str] | None = None,
    timeout: float | None = None,
) -> T:
    """Execute ``operation(input, context=context)`` within the runtime.

    When ``name`` is provided, the operation is wrapped with LangSmith
    tracing. Sync operations are bridged to a thread via
    ``asyncio.to_thread``. Exceptions are normalized into structured
    ``RetryableError`` / ``InvalidInputError`` types.

    Args:
        operation: A callable accepting ``(input, context=context)``.
            May be sync or async.
        input: The primary input passed to ``operation``.
        context: The ``RunContext`` for this execution.
        name: Optional LangSmith trace name. When provided, the operation
            is wrapped with ``@traceable(name=name, ...)``.
        run_type: LangSmith run type ("chain", "tool", "retriever").
        tags: Optional LangSmith trace tags.
        timeout: Optional timeout in seconds. If exceeded, raises
            ``RetryableError`` with code ``"timeout"``.

    Returns:
        Whatever ``operation`` returns.

    Raises:
        RetryableError: On timeout or any unexpected exception.
        InvalidInputError: On invalid input (e.g. wrong type).
    """
    start = time.monotonic()

    # Wrap with LangSmith tracing when a name is provided.
    op = operation
    if name is not None:
        op = trace_operation(
            name=name,
            run_type=run_type,
            tags=tags,
            metadata={
                "run_id": context.run_id,
                "session_id": context.session_id,
                "source": context.source,
                **context.metadata,
            },
        )(operation)

    async def _run() -> T:
        if asyncio.iscoroutinefunction(op):
            return await op(input, context=context)
        return await asyncio.to_thread(op, input, context=context)

    try:
        if timeout is not None:
            result = await asyncio.wait_for(_run(), timeout=timeout)
        else:
            result = await _run()

        elapsed_ms = int((time.monotonic() - start) * 1000)
        emit_metadata(context, elapsed_ms=elapsed_ms, final_status="ok")
        return result

    except TimeoutError as exc:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        emit_metadata(
            context,
            elapsed_ms=elapsed_ms,
            final_status="timeout",
            error_code="timeout",
        )
        raise RetryableError(
            code="timeout",
            message=f"Operation timed out after {timeout}s",
            details={"timeout_seconds": timeout, "elapsed_ms": elapsed_ms},
        ) from exc
    except (RetryableError, InvalidInputError):
        # Already structured — pass through, but still emit metadata.
        elapsed_ms = int((time.monotonic() - start) * 1000)
        emit_metadata(context, elapsed_ms=elapsed_ms, final_status="error")
        raise
    except Exception as exc:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        emit_metadata(
            context,
            elapsed_ms=elapsed_ms,
            final_status="error",
            error_code=type(exc).__name__,
        )
        raise RetryableError(
            code=type(exc).__name__,
            message=str(exc),
            details={"elapsed_ms": elapsed_ms},
        ) from exc

"""execute() — the runtime execution boundary with policy, idempotency, and observability.

Wraps any operation with:
- **ExecutionPolicy** — retry, timeout, and irreversible-write safety.
- **Idempotency** — terminal-outcome caching with owner-token fencing so
  repeat calls with the same key short-circuit and concurrent duplicates wait.
- **LangSmith tracing** — when ``name`` is provided, the operation is wrapped
  with ``@traceable(name=name, ...)``.
- **Structured error normalization** — exceptions are classified into
  ``RetryableError`` / ``PermanentError`` / ``IndeterminateError`` per the
  policy.

Ported from echo-v2's ``runtime/executor.py``, adapted to barcode-scanner:
- Preserves ``name=`` as a parameter (NOT moved into ``RunContext``).
- Preserves barcode-scanner's LangSmith tracing via ``trace_operation`` +
  ``emit_metadata``.
- Uses barcode-scanner's structured ``ExecutionError`` (code/message/details).

The ``RunContext.run_id`` is added to trace metadata as ``run_id`` — it is an
application correlation ID, NOT the LangSmith run ID.
"""

from __future__ import annotations

import asyncio
import builtins
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Generic, TypeVar, cast

from src.observability.tracing import emit_metadata, trace_operation
from src.runtime.context import RunContext
from src.runtime.errors import (
    ExecutionError,
    IndeterminateError,
    InvalidInputError,
    PermanentError,
    RetryableError,
    TimeoutError,
)
from src.runtime.idempotency import (
    IdempotencyStore,
    IndeterminateOutcome,
    PermanentFailureOutcome,
    ReserveStatus,
    StoredOutcome,
    SuccessOutcome,
)
from src.runtime.policy import NO_RETRY, ExecutionPolicy

__all__ = ["ExecutionResult", "execute"]

TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")

Operation = Callable[..., TOutput | Awaitable[TOutput]]

@dataclass(frozen=True)
class ExecutionResult(Generic[TOutput]):
    """Result of :func:`execute` — carries the value and run metadata."""

    value: TOutput
    duration_ms: float
    attempts: int


async def _run_operation(
    operation: Operation[TOutput],
    input_: Any,
    context: RunContext,
    **op_kwargs: Any,
) -> TOutput:
    """Run ``operation``, bridging sync to async via ``asyncio.to_thread``."""
    if inspect.iscoroutinefunction(operation):
        return cast(
            "TOutput", await operation(input_, context=context, **op_kwargs)
        )
    result = await asyncio.to_thread(
        operation, input_, context=context, **op_kwargs
    )
    if inspect.isawaitable(result):
        return cast("TOutput", await result)
    return result


async def _run_with_retries(
    operation: Operation[TOutput],
    input_: Any,
    context: RunContext,
    policy: ExecutionPolicy,
    name: str | None,
    idempotency_key: str | None,
    **op_kwargs: Any,
) -> ExecutionResult[TOutput]:
    """Run ``operation`` with retry, timeout, and LangSmith tracing.

    Emits LangSmith metadata at each terminal state (ok, timeout, error).
    """
    start = perf_counter()

    # Wrap with LangSmith tracing when a name is provided.
    op: Operation[TOutput] = operation
    if name is not None:
        op = trace_operation(
            name=name,
            metadata={
                "run_id": context.run_id,
                "session_id": context.session_id,
                "source": context.source,
                **context.metadata,
            },
        )(operation)

    for attempt in range(1, policy.max_attempts + 1):
        try:
            value = await asyncio.wait_for(
                _run_operation(op, input_, context, **op_kwargs),
                timeout=policy.timeout_seconds,
            )

            duration_ms = (perf_counter() - start) * 1000
            emit_metadata(
                context,
                elapsed_ms=int(duration_ms),
                final_status="ok",
                attempts=attempt,
            )
            return ExecutionResult(
                value=value,
                duration_ms=duration_ms,
                attempts=attempt,
            )

        except builtins.TimeoutError as exc:
            wrapped: ExecutionError = TimeoutError(
                f"Operation timed out after {policy.timeout_seconds}s",
                code="timeout",
                details={
                    "timeout_seconds": policy.timeout_seconds,
                    "attempt": attempt,
                },
            )

            # An irreversible write that times out may have already produced
            # its side effect. Do NOT retry — propagate so execute() can store
            # INDETERMINATE. Safety is structural, not dependent on
            # max_attempts=1.
            if policy.irreversible_write:
                duration_ms = (perf_counter() - start) * 1000
                emit_metadata(
                    context,
                    elapsed_ms=int(duration_ms),
                    final_status="indeterminate",
                    error_code="timeout",
                    attempts=attempt,
                )
                raise IndeterminateError(
                    f"Operation timed out during irreversible run {context.run_id}",
                    code="timeout",
                    details={"timeout_seconds": policy.timeout_seconds},
                ) from exc

            if attempt >= policy.max_attempts:
                duration_ms = (perf_counter() - start) * 1000
                emit_metadata(
                    context,
                    elapsed_ms=int(duration_ms),
                    final_status="timeout",
                    error_code="timeout",
                    attempts=attempt,
                )
                raise wrapped from exc

            if policy.retry_delay_seconds > 0:
                await asyncio.sleep(policy.retry_delay_seconds)

        except RetryableError as exc:
            if attempt >= policy.max_attempts:
                duration_ms = (perf_counter() - start) * 1000
                emit_metadata(
                    context,
                    elapsed_ms=int(duration_ms),
                    final_status="error",
                    error_code=exc.code,
                    attempts=attempt,
                )
                raise

            if policy.retry_delay_seconds > 0:
                await asyncio.sleep(policy.retry_delay_seconds)

        except IndeterminateError as exc:
            # Explicitly raised by an integration that knows the outcome is
            # ambiguous. Not retryable. Emit indeterminate (not failed) and
            # propagate so execute() can store INDETERMINATE.
            duration_ms = (perf_counter() - start) * 1000
            emit_metadata(
                context,
                elapsed_ms=int(duration_ms),
                final_status="indeterminate",
                error_code=exc.code,
                attempts=attempt,
            )
            raise

        except (PermanentError, InvalidInputError) as exc:
            # Already structured — pass through, but still emit metadata.
            duration_ms = (perf_counter() - start) * 1000
            emit_metadata(
                context,
                elapsed_ms=int(duration_ms),
                final_status="error",
                error_code=exc.code,
                attempts=attempt,
            )
            raise

        except ExecutionError as exc:
            # Base-class fallback (should not normally be raised directly).
            duration_ms = (perf_counter() - start) * 1000
            emit_metadata(
                context,
                elapsed_ms=int(duration_ms),
                final_status="error",
                error_code=exc.code,
                attempts=attempt,
            )
            raise

        except Exception as exc:
            # For irreversible writes, an unexpected error after the request
            # was submitted (e.g. KeyError parsing an unfamiliar response
            # shape) means the side effect may have happened. Conservatively
            # treat as indeterminate rather than claiming "definitely did not
            # send." For reads/compute, wrap as PermanentError.
            if policy.irreversible_write:
                wrapped = IndeterminateError(
                    f"Unexpected failure during irreversible run {context.run_id}",
                    code=type(exc).__name__,
                )
                duration_ms = (perf_counter() - start) * 1000
                emit_metadata(
                    context,
                    elapsed_ms=int(duration_ms),
                    final_status="indeterminate",
                    error_code=type(exc).__name__,
                    attempts=attempt,
                )
            else:
                wrapped = PermanentError(
                    f"Unexpected failure during run {context.run_id}",
                    code=type(exc).__name__,
                )
                duration_ms = (perf_counter() - start) * 1000
                emit_metadata(
                    context,
                    elapsed_ms=int(duration_ms),
                    final_status="error",
                    error_code=type(exc).__name__,
                    attempts=attempt,
                )
            raise wrapped from exc

    raise RuntimeError("unreachable")  # pragma: no cover


def _replay_failure(
    outcome: PermanentFailureOutcome,
    context: RunContext,
) -> PermanentError:
    return PermanentError(
        f"Idempotent operation previously failed permanently: {outcome.error_message}",
        code="idempotent_replay",
        details={
            "original_error_type": outcome.error_type,
            "run_id": context.run_id,
        },
    )


def _replay_indeterminate(
    outcome: IndeterminateOutcome,
    context: RunContext,
) -> IndeterminateError:
    return IndeterminateError(
        f"Idempotent operation previously ended with unknown outcome: {outcome.error_message}",
        code="idempotent_replay",
        details={
            "original_error_type": outcome.error_type,
            "run_id": context.run_id,
        },
    )


async def _handle_outcome(
    outcome: StoredOutcome[TOutput],
    context: RunContext,
    idempotency_key: str,
) -> ExecutionResult[TOutput]:
    """Replay a cached terminal outcome without re-running the operation."""
    if isinstance(outcome, SuccessOutcome):
        emit_metadata(
            context,
            final_status="ok",
            idempotent="hit",
            idempotency_key=idempotency_key,
        )
        return ExecutionResult(
            value=outcome.value,
            duration_ms=outcome.duration_ms,
            attempts=outcome.attempts,
        )

    if isinstance(outcome, IndeterminateOutcome):
        emit_metadata(
            context,
            final_status="indeterminate",
            idempotent="replay",
            idempotency_key=idempotency_key,
        )
        raise _replay_indeterminate(outcome, context)

    # PermanentFailureOutcome
    emit_metadata(
        context,
        final_status="error",
        idempotent="replay",
        idempotency_key=idempotency_key,
    )
    raise _replay_failure(outcome, context)


async def execute(
    operation: Operation[TOutput],
    input: Any,
    context: RunContext,
    *,
    name: str | None = None,
    run_type: str = "chain",
    tags: list[str] | None = None,
    policy: ExecutionPolicy = NO_RETRY,
    idempotency_key: str | None = None,
    idempotency_store: IdempotencyStore[TOutput] | None = None,
    **op_kwargs: Any,
) -> TOutput:
    """Execute ``operation(input, context=context, **op_kwargs)`` within the runtime.

    When ``name`` is provided, the operation is wrapped with LangSmith
    tracing. Sync operations are bridged to a thread via
    ``asyncio.to_thread``. Exceptions are normalized into structured
    ``RetryableError`` / ``PermanentError`` / ``IndeterminateError`` types
    per the ``policy``.

    Args:
        operation: A callable accepting ``(input, context=context, **kwargs)``.
            May be sync or async.
        input: The primary input passed to ``operation``.
        context: The ``RunContext`` for this execution.
        name: Optional LangSmith trace name. When provided, the operation
            is wrapped with ``@traceable(name=name, ...)``.
        run_type: LangSmith run type ("chain", "tool", "retriever").
        tags: Optional LangSmith trace tags.
        policy: Execution policy (retry, timeout, irreversible-write safety).
            Defaults to ``NO_RETRY``.
        idempotency_key: Optional idempotency key. Must be provided together
            with ``idempotency_store``.
        idempotency_store: Optional idempotency store. Must be provided
            together with ``idempotency_key``.
        **op_kwargs: Additional keyword arguments passed to ``operation``.

    Returns:
        Whatever ``operation`` returns.

    Raises:
        RetryableError: On timeout or transient failure (if retries exhausted).
        PermanentError: On permanent failure or unexpected non-irreversible error.
        IndeterminateError: On unknown-outcome failure of an irreversible write.
        InvalidInputError: On invalid input (e.g. wrong type).
    """
    if (idempotency_key is None) != (idempotency_store is None):
        raise ValueError(
            "idempotency_key and idempotency_store must be provided together"
        )

    # No idempotency: simple retry path.
    if idempotency_key is None or idempotency_store is None:
        result = await _run_with_retries(
            operation=operation,
            input_=input,
            context=context,
            policy=policy,
            name=name,
            idempotency_key=None,
            **op_kwargs,
        )
        return result.value

    store = idempotency_store
    key = idempotency_key

    # 1. Fast path: a terminal outcome is already cached.
    cached = await store.get(key)
    if cached is not None:
        result = await _handle_outcome(cached, context, key)
        return result.value

    # 2. Try to claim the key.
    reserve_result = await store.reserve(key)

    if reserve_result.status == ReserveStatus.COMPLETED:
        outcome = await store.get(key)
        if outcome is not None:
            result = await _handle_outcome(outcome, context, key)
            return result.value
        # Rare race: outcome vanished. Fall through to re-reserve.  # pragma: no cover
        reserve_result = await store.reserve(key)  # pragma: no cover

    if reserve_result.status == ReserveStatus.IN_PROGRESS:
        emit_metadata(
            context,
            idempotent="waiting",
            idempotency_key=key,
        )
        outcome = await store.wait_for_completion(key)
        result = await _handle_outcome(outcome, context, key)
        return result.value

    # 3. ACQUIRED: this caller owns execution.
    owner_token = reserve_result.owner_token
    assert owner_token is not None  # ACQUIRED always carries a token

    try:
        result = await _run_with_retries(
            operation=operation,
            input_=input,
            context=context,
            policy=policy,
            name=name,
            idempotency_key=key,
            **op_kwargs,
        )
    except asyncio.CancelledError:
        # Cancellation tells us the caller stopped waiting, NOT that the side
        # effect didn't happen (especially under to_thread, which can't stop
        # the underlying thread). For irreversible writes, conservatively
        # treat as indeterminate.
        if policy.irreversible_write:
            await asyncio.shield(
                store.put_indeterminate(
                    key,
                    owner_token,
                    IndeterminateOutcome(
                        error_type="CancelledError",
                        error_message=f"Operation cancelled during run {context.run_id}",
                    ),
                )
            )
        else:
            await asyncio.shield(store.release(key, owner_token))
        raise
    except (PermanentError, InvalidInputError) as exc:
        await asyncio.shield(
            store.put_failure(
                key,
                owner_token,
                PermanentFailureOutcome(
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                ),
            )
        )
        raise
    except IndeterminateError as exc:
        await asyncio.shield(
            store.put_indeterminate(
                key,
                owner_token,
                IndeterminateOutcome(
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                ),
            )
        )
        raise
    except TimeoutError:
        # Reversible timeout (irreversible timeouts are converted to
        # IndeterminateError by _run_with_retries before reaching here).
        await asyncio.shield(store.release(key, owner_token))
        raise
    except RetryableError:
        # Unambiguous "did not complete" (e.g. 429, connection refused before
        # send, or LostOwnershipError from a store write). Release regardless
        # of irreversible_write — do NOT over-block idempotency keys for
        # trivially-retryable conditions.
        await asyncio.shield(store.release(key, owner_token))
        raise
    except ExecutionError:
        # Base-class fallback (should not normally be raised directly).
        await asyncio.shield(store.release(key, owner_token))
        raise
    except Exception as exc:  # pragma: no cover
        # Already wrapped by _run_with_retries as PermanentError or
        # IndeterminateError. This arm should not normally fire from the owner
        # path; keep as a defensive put_failure if it ever does.
        await asyncio.shield(
            store.put_failure(
                key,
                owner_token,
                PermanentFailureOutcome(
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                ),
            )
        )
        raise  # pragma: no cover

    await asyncio.shield(
        store.put_success(
            key,
            owner_token,
            SuccessOutcome(
                value=result.value,
                attempts=result.attempts,
                duration_ms=result.duration_ms,
            ),
        )
    )
    return result.value

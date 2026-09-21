"""Tests for the runtime executor — retry, timeout, indeterminate, and
non-idempotent paths.

Targets >95% coverage of ``src/runtime/executor.py``.
"""

from __future__ import annotations

import asyncio
import time
import uuid

import pytest

from src.runtime.context import RunContext
from src.runtime.errors import (
    ExecutionError,
    IndeterminateError,
    InvalidInputError,
    PermanentError,
    RetryableError,
    TimeoutError,
)
from src.runtime.executor import execute
from src.runtime.idempotency import (
    IdempotencyStore,
    IndeterminateOutcome,
    InMemoryIdempotencyStore,
    LostOwnershipError,
    PermanentFailureOutcome,
    ReserveStatus,
    SuccessOutcome,
)
from src.runtime.policy import (
    EXTERNAL_READ,
    EXTERNAL_WRITE,
    NO_RETRY,
    SCAN_COMPUTE,
    ExecutionPolicy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ctx() -> RunContext:
    return RunContext(
        run_id="run-test",
        session_id="sess-test",
        source="eval",
    )


# ---------------------------------------------------------------------------
# Basic execution (non-idempotent path)
# ---------------------------------------------------------------------------


async def test_execute_returns_operation_result():
    async def double(x: int, *, context, **kw):
        return x * 2

    result = await execute(double, 21, _ctx(), policy=NO_RETRY)
    assert result == 42


async def test_execute_sync_operation():
    def double(x: int, *, context, **kw):
        return x * 2

    result = await execute(double, 21, _ctx(), policy=NO_RETRY)
    assert result == 42


async def test_execute_sync_operation_returning_awaitable():
    """A sync operation that returns a coroutine is awaited."""

    async def inner():
        return 99

    def op(_x, *, context, **kw):
        return inner()

    result = await execute(op, 0, _ctx(), policy=NO_RETRY)
    assert result == 99


async def test_execute_passes_op_kwargs():
    captured: dict = {}

    async def op(x: int, *, context, extra=None, **kw):
        captured["extra"] = extra
        return x

    result = await execute(op, 5, _ctx(), policy=NO_RETRY, extra="hello")
    assert result == 5
    assert captured["extra"] == "hello"


async def test_execute_with_name_wraps_tracing():
    """When name= is provided, the operation is wrapped with LangSmith tracing."""
    captured: dict = {}

    async def op(x: int, *, context, **kw):
        captured["called"] = True
        return x

    result = await execute(op, 5, _ctx(), name="test_op", policy=NO_RETRY)
    assert result == 5
    assert captured["called"] is True


# ---------------------------------------------------------------------------
# Timeout handling
# ---------------------------------------------------------------------------


async def test_timeout_reversible_raises_timeout_error():
    """A timeout on a reversible operation raises TimeoutError (retryable)."""

    async def slow_op(_x, *, context, **kw):
        await asyncio.sleep(1.0)
        return 42

    with pytest.raises(TimeoutError):
        await execute(
            slow_op,
            0,
            _ctx(),
            policy=ExecutionPolicy(max_attempts=1, timeout_seconds=0.05),
        )


async def test_timeout_irreversible_raises_indeterminate():
    """A timeout on an irreversible write raises IndeterminateError."""

    async def slow_op(_x, *, context, **kw):
        await asyncio.sleep(1.0)
        return 42

    with pytest.raises(IndeterminateError):
        await execute(
            slow_op,
            0,
            _ctx(),
            policy=ExecutionPolicy(
                max_attempts=1, timeout_seconds=0.05, irreversible_write=True
            ),
        )


async def test_timeout_retries_then_succeeds():
    """A timeout on a retryable policy retries and may succeed."""
    attempts = 0

    async def op(_x, *, context, **kw):
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            await asyncio.sleep(1.0)
        return 42

    result = await execute(
        op,
        0,
        _ctx(),
        policy=ExecutionPolicy(
            max_attempts=3, timeout_seconds=0.05, retry_delay_seconds=0.0
        ),
    )
    assert result == 42
    assert attempts == 2


async def test_timeout_with_retry_delay():
    """Retry delay is observed between timeout retries."""
    attempts = 0
    timestamps: list[float] = []

    async def op(_x, *, context, **kw):
        nonlocal attempts
        attempts += 1
        timestamps.append(time.perf_counter())
        if attempts < 2:
            await asyncio.sleep(1.0)
        return 42

    result = await execute(
        op,
        0,
        _ctx(),
        policy=ExecutionPolicy(
            max_attempts=3, timeout_seconds=0.05, retry_delay_seconds=0.1
        ),
    )
    assert result == 42
    assert attempts == 2
    # At least 0.1s delay between attempts.
    assert timestamps[1] - timestamps[0] >= 0.05


# ---------------------------------------------------------------------------
# Retryable errors
# ---------------------------------------------------------------------------


async def test_retryable_error_retries_then_succeeds():
    attempts = 0

    async def op(_x, *, context, **kw):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RetryableError("transient")
        return 42

    result = await execute(op, 0, _ctx(), policy=EXTERNAL_READ)
    assert result == 42
    assert attempts == 3


async def test_retryable_error_exhausts_retries():
    attempts = 0

    async def op(_x, *, context, **kw):
        nonlocal attempts
        attempts += 1
        raise RetryableError("always fails")

    with pytest.raises(RetryableError):
        await execute(op, 0, _ctx(), policy=EXTERNAL_READ)
    assert attempts == 3


async def test_retryable_error_with_retry_delay():
    """Retry delay is observed between RetryableError retries."""
    attempts = 0
    timestamps: list[float] = []

    async def op(_x, *, context, **kw):
        nonlocal attempts
        attempts += 1
        timestamps.append(time.perf_counter())
        if attempts < 2:
            raise RetryableError("transient")
        return 42

    result = await execute(
        op,
        0,
        _ctx(),
        policy=ExecutionPolicy(
            max_attempts=3, retry_delay_seconds=0.1, timeout_seconds=None
        ),
    )
    assert result == 42
    assert timestamps[1] - timestamps[0] >= 0.05


# ---------------------------------------------------------------------------
# Permanent / InvalidInput errors
# ---------------------------------------------------------------------------


async def test_permanent_error_not_retried():
    attempts = 0

    async def op(_x, *, context, **kw):
        nonlocal attempts
        attempts += 1
        raise PermanentError("bad")

    with pytest.raises(PermanentError):
        await execute(op, 0, _ctx(), policy=EXTERNAL_READ)
    assert attempts == 1


async def test_invalid_input_error_not_retried():
    attempts = 0

    async def op(_x, *, context, **kw):
        nonlocal attempts
        attempts += 1
        raise InvalidInputError("bad input")

    with pytest.raises(InvalidInputError):
        await execute(op, 0, _ctx(), policy=EXTERNAL_READ)
    assert attempts == 1


async def test_indeterminate_error_not_retried():
    attempts = 0

    async def op(_x, *, context, **kw):
        nonlocal attempts
        attempts += 1
        raise IndeterminateError("unknown")

    with pytest.raises(IndeterminateError):
        await execute(op, 0, _ctx(), policy=EXTERNAL_READ)
    assert attempts == 1


async def test_execution_error_base_class_not_retried():
    """A bare ExecutionError (base class) is not retried."""
    attempts = 0

    async def op(_x, *, context, **kw):
        nonlocal attempts
        attempts += 1
        raise ExecutionError("base")

    with pytest.raises(ExecutionError):
        await execute(op, 0, _ctx(), policy=EXTERNAL_READ)
    assert attempts == 1


# ---------------------------------------------------------------------------
# Unexpected exceptions
# ---------------------------------------------------------------------------


async def test_unexpected_error_reversible_wraps_as_permanent():
    """An unexpected exception on a reversible op is wrapped as PermanentError."""

    async def op(_x, *, context, **kw):
        raise KeyError("boom")

    with pytest.raises(PermanentError):
        await execute(op, 0, _ctx(), policy=NO_RETRY)


async def test_unexpected_error_irreversible_wraps_as_permanent():
    """An unexpected exception on an irreversible write is wrapped as
    PermanentError, NOT IndeterminateError.

    Per AGENTS.md, the executor does NOT upgrade a generic unexpected
    exception to IndeterminateError on its own — only adapter-classified
    IndeterminateError and executor timeout/cancellation become indeterminate.
    The adapter owns classifying pre-submit vs post-submit at the integration
    boundary.
    """

    async def op(_x, *, context, **kw):
        raise KeyError("boom")

    with pytest.raises(PermanentError):
        await execute(op, 0, _ctx(), policy=EXTERNAL_WRITE)


# ---------------------------------------------------------------------------
# Idempotent execution — owner path
# ---------------------------------------------------------------------------


async def test_idempotent_success_stores_and_replays():
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()
    call_count = 0

    async def op(x: int, *, context, **kw):
        nonlocal call_count
        call_count += 1
        return x * 2

    r1 = await execute(
        op, 5, _ctx(), policy=NO_RETRY, idempotency_key="k", idempotency_store=store
    )
    assert r1 == 10
    assert call_count == 1

    r2 = await execute(
        op, 5, _ctx(), policy=NO_RETRY, idempotency_key="k", idempotency_store=store
    )
    assert r2 == 10
    assert call_count == 1  # replayed, not re-run


async def test_idempotent_permanent_error_stores_and_replays():
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()

    async def failing(_x, *, context, **kw):
        raise PermanentError("bad")

    with pytest.raises(PermanentError):
        await execute(
            failing,
            0,
            _ctx(),
            policy=NO_RETRY,
            idempotency_key="k",
            idempotency_store=store,
        )

    async def ok(_x, *, context, **kw):
        return 99

    with pytest.raises(PermanentError, match="bad"):
        await execute(
            ok,
            0,
            _ctx(),
            policy=NO_RETRY,
            idempotency_key="k",
            idempotency_store=store,
        )


async def test_idempotent_invalid_input_error_stores_as_failure():
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()

    async def failing(_x, *, context, **kw):
        raise InvalidInputError("bad input")

    with pytest.raises(InvalidInputError):
        await execute(
            failing,
            0,
            _ctx(),
            policy=NO_RETRY,
            idempotency_key="k",
            idempotency_store=store,
        )

    async def ok(_x, *, context, **kw):
        return 99

    with pytest.raises(PermanentError, match="bad input"):
        await execute(
            ok,
            0,
            _ctx(),
            policy=NO_RETRY,
            idempotency_key="k",
            idempotency_store=store,
        )


async def test_idempotent_indeterminate_error_stores_and_replays():
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()

    async def failing(_x, *, context, **kw):
        raise IndeterminateError("unknown")

    with pytest.raises(IndeterminateError):
        await execute(
            failing,
            0,
            _ctx(),
            policy=EXTERNAL_WRITE,
            idempotency_key="k",
            idempotency_store=store,
        )

    async def ok(_x, *, context, **kw):
        return 99

    with pytest.raises(IndeterminateError, match="unknown"):
        await execute(
            ok,
            0,
            _ctx(),
            policy=EXTERNAL_WRITE,
            idempotency_key="k",
            idempotency_store=store,
        )


async def test_idempotent_timeout_irreversible_stores_indeterminate():
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()

    async def slow(_x, *, context, **kw):
        await asyncio.sleep(1.0)
        return 42

    with pytest.raises(IndeterminateError):
        await execute(
            slow,
            0,
            _ctx(),
            policy=ExecutionPolicy(
                max_attempts=1, timeout_seconds=0.05, irreversible_write=True
            ),
            idempotency_key="k",
            idempotency_store=store,
        )

    # Re-call should replay indeterminate.
    async def ok(_x, *, context, **kw):
        return 99

    with pytest.raises(IndeterminateError):
        await execute(
            ok,
            0,
            _ctx(),
            policy=EXTERNAL_WRITE,
            idempotency_key="k",
            idempotency_store=store,
        )


async def test_idempotent_timeout_reversible_releases_claim():
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()

    async def slow(_x, *, context, **kw):
        await asyncio.sleep(1.0)
        return 42

    with pytest.raises(TimeoutError):
        await execute(
            slow,
            0,
            _ctx(),
            policy=ExecutionPolicy(max_attempts=1, timeout_seconds=0.05),
            idempotency_key="k",
            idempotency_store=store,
        )

    # Key was released — a subsequent call can re-run.
    async def ok(_x, *, context, **kw):
        return 99

    result = await execute(
        ok,
        0,
        _ctx(),
        policy=NO_RETRY,
        idempotency_key="k",
        idempotency_store=store,
    )
    assert result == 99


async def test_idempotent_retryable_error_releases_claim():
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()

    async def failing(_x, *, context, **kw):
        raise RetryableError("transient")

    with pytest.raises(RetryableError):
        await execute(
            failing,
            0,
            _ctx(),
            policy=EXTERNAL_WRITE,
            idempotency_key="k",
            idempotency_store=store,
        )

    # Key was released — a subsequent call can re-run.
    async def ok(_x, *, context, **kw):
        return 99

    result = await execute(
        ok,
        0,
        _ctx(),
        policy=EXTERNAL_WRITE,
        idempotency_key="k",
        idempotency_store=store,
    )
    assert result == 99


async def test_idempotent_execution_error_base_releases_claim():
    """A bare ExecutionError on the owner path releases the claim."""
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()

    async def failing(_x, *, context, **kw):
        raise ExecutionError("base")

    with pytest.raises(ExecutionError):
        await execute(
            failing,
            0,
            _ctx(),
            policy=EXTERNAL_WRITE,
            idempotency_key="k",
            idempotency_store=store,
        )

    # Key was released — a subsequent call can re-run.
    async def ok(_x, *, context, **kw):
        return 99

    result = await execute(
        ok,
        0,
        _ctx(),
        policy=EXTERNAL_WRITE,
        idempotency_key="k",
        idempotency_store=store,
    )
    assert result == 99


async def test_idempotent_unexpected_error_irreversible_stores_failure():
    """An unexpected exception on an irreversible write is stored as FAILURE
    (PermanentError), NOT INDETERMINATE.

    Per AGENTS.md, the executor does NOT upgrade a generic unexpected
    exception to IndeterminateError on its own — only adapter-classified
    IndeterminateError and executor timeout/cancellation become indeterminate.
    """
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()

    async def failing(_x, *, context, **kw):
        raise KeyError("boom")

    with pytest.raises(PermanentError):
        await execute(
            failing,
            0,
            _ctx(),
            policy=EXTERNAL_WRITE,
            idempotency_key="k",
            idempotency_store=store,
        )

    # Re-call should replay failure.
    async def ok(_x, *, context, **kw):
        return 99

    with pytest.raises(PermanentError):
        await execute(
            ok,
            0,
            _ctx(),
            policy=EXTERNAL_WRITE,
            idempotency_key="k",
            idempotency_store=store,
        )


async def test_idempotent_unexpected_error_reversible_stores_failure():
    """An unexpected exception on a reversible op is stored as FAILURE."""
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()

    async def failing(_x, *, context, **kw):
        raise KeyError("boom")

    with pytest.raises(PermanentError):
        await execute(
            failing,
            0,
            _ctx(),
            policy=NO_RETRY,
            idempotency_key="k",
            idempotency_store=store,
        )

    # Re-call should replay the failure.
    async def ok(_x, *, context, **kw):
        return 99

    with pytest.raises(PermanentError):
        await execute(
            ok,
            0,
            _ctx(),
            policy=NO_RETRY,
            idempotency_key="k",
            idempotency_store=store,
        )


# ---------------------------------------------------------------------------
# Idempotent execution — waiter path
# ---------------------------------------------------------------------------


async def test_idempotent_waiter_shares_success_result():
    """A waiter (IN_PROGRESS) shares the owner's success result."""
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_op(_x, *, context, **kw):
        started.set()
        await release.wait()
        return 42

    async def call_once():
        return await execute(
            slow_op,
            0,
            _ctx(),
            policy=NO_RETRY,
            idempotency_key="k",
            idempotency_store=store,
        )

    t1 = asyncio.create_task(call_once())
    await started.wait()
    t2 = asyncio.create_task(call_once())
    await asyncio.sleep(0.05)
    release.set()

    r1 = await t1
    r2 = await t2
    assert r1 == 42
    assert r2 == 42


async def test_idempotent_waiter_shares_failure_result():
    """A waiter shares the owner's permanent failure."""
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()
    started = asyncio.Event()
    release = asyncio.Event()

    async def failing(_x, *, context, **kw):
        started.set()
        await release.wait()
        raise PermanentError("bad")

    async def call_once():
        return await execute(
            failing,
            0,
            _ctx(),
            policy=NO_RETRY,
            idempotency_key="k",
            idempotency_store=store,
        )

    t1 = asyncio.create_task(call_once())
    await started.wait()
    t2 = asyncio.create_task(call_once())
    await asyncio.sleep(0.05)
    release.set()

    with pytest.raises(PermanentError):
        await t1
    with pytest.raises(PermanentError):
        await t2


async def test_idempotent_waiter_shares_indeterminate_result():
    """A waiter shares the owner's indeterminate outcome."""
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()
    started = asyncio.Event()
    release = asyncio.Event()

    async def failing(_x, *, context, **kw):
        started.set()
        await release.wait()
        raise IndeterminateError("unknown")

    async def call_once():
        return await execute(
            failing,
            0,
            _ctx(),
            policy=EXTERNAL_WRITE,
            idempotency_key="k",
            idempotency_store=store,
        )

    t1 = asyncio.create_task(call_once())
    await started.wait()
    t2 = asyncio.create_task(call_once())
    await asyncio.sleep(0.05)
    release.set()

    with pytest.raises(IndeterminateError):
        await t1
    with pytest.raises(IndeterminateError):
        await t2


# ---------------------------------------------------------------------------
# Idempotent execution — cancellation
# ---------------------------------------------------------------------------


async def test_cancellation_irreversible_stores_indeterminate():
    """Cancelling an irreversible write stores INDETERMINATE."""
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()
    started = asyncio.Event()

    async def slow_op(_x, *, context, **kw):
        started.set()
        await asyncio.sleep(10.0)
        return 42

    async def call():
        return await execute(
            slow_op,
            0,
            _ctx(),
            policy=EXTERNAL_WRITE,
            idempotency_key="k",
            idempotency_store=store,
        )

    task = asyncio.create_task(call())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The key should be INDETERMINATE — re-call replays it.
    async def ok(_x, *, context, **kw):
        return 99

    with pytest.raises(IndeterminateError):
        await execute(
            ok,
            0,
            _ctx(),
            policy=EXTERNAL_WRITE,
            idempotency_key="k",
            idempotency_store=store,
        )


async def test_cancellation_reversible_releases_claim():
    """Cancelling a reversible operation releases the claim."""
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()
    started = asyncio.Event()

    async def slow_op(_x, *, context, **kw):
        started.set()
        await asyncio.sleep(10.0)
        return 42

    async def call():
        return await execute(
            slow_op,
            0,
            _ctx(),
            policy=NO_RETRY,
            idempotency_key="k",
            idempotency_store=store,
        )

    task = asyncio.create_task(call())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The key was released — a subsequent call can re-run.
    async def ok(_x, *, context, **kw):
        return 99

    result = await execute(
        ok,
        0,
        _ctx(),
        policy=NO_RETRY,
        idempotency_key="k",
        idempotency_store=store,
    )
    assert result == 99


# ---------------------------------------------------------------------------
# Idempotent execution — validation
# ---------------------------------------------------------------------------


async def test_key_without_store_raises_value_error():
    async def op(_x, *, context, **kw):
        return 1

    with pytest.raises(ValueError, match="must be provided together"):
        await execute(op, 0, _ctx(), idempotency_key="k")


async def test_store_without_key_raises_value_error():
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()

    async def op(_x, *, context, **kw):
        return 1

    with pytest.raises(ValueError, match="must be provided together"):
        await execute(op, 0, _ctx(), idempotency_store=store)


# ---------------------------------------------------------------------------
# Idempotent execution — COMPLETED reserve path
# ---------------------------------------------------------------------------


async def test_idempotent_completed_reserve_replays_outcome():
    """When reserve returns COMPLETED, the cached outcome is replayed."""
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()
    store.seed_outcome(
        "k", SuccessOutcome(value=42, attempts=1, duration_ms=1.0)
    )

    call_count = 0

    async def op(_x, *, context, **kw):
        nonlocal call_count
        call_count += 1
        return 99

    result = await execute(
        op,
        0,
        _ctx(),
        policy=NO_RETRY,
        idempotency_key="k",
        idempotency_store=store,
    )
    assert result == 42
    assert call_count == 0


# ---------------------------------------------------------------------------
# SCAN_COMPUTE policy
# ---------------------------------------------------------------------------


async def test_scan_compute_no_retry():
    """SCAN_COMPUTE does not retry (deterministic scanner)."""
    attempts = 0

    async def op(_x, *, context, **kw):
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise RetryableError("transient")
        return 42

    with pytest.raises(RetryableError):
        await execute(op, 0, _ctx(), policy=SCAN_COMPUTE)
    assert attempts == 1


# ---------------------------------------------------------------------------
# COMPLETED reserve path (outcome not in fast-path cache but reserve says COMPLETED)
# ---------------------------------------------------------------------------


async def test_idempotent_completed_reserve_with_vanished_outcome():
    """When reserve returns COMPLETED but get() returns None (rare race),
    the code falls through to re-reserve. We simulate this with a custom store.
    """

    class FlakyStore(InMemoryIdempotencyStore[int]):
        """A store where get() returns None the first time, then the seeded outcome."""

        def __init__(self) -> None:
            super().__init__()
            self._get_calls = 0

        async def get(self, key: str):
            self._get_calls += 1
            # First call (fast path) returns None; second call (after
            # COMPLETED) returns the seeded outcome.
            if self._get_calls <= 1:
                return None
            return await super().get(key)

    store = FlakyStore()
    store.seed_outcome("k", SuccessOutcome(value=42, attempts=1, duration_ms=1.0))

    call_count = 0

    async def op(_x, *, context, **kw):
        nonlocal call_count
        call_count += 1
        return 99

    result = await execute(
        op,
        0,
        _ctx(),
        policy=NO_RETRY,
        idempotency_key="k",
        idempotency_store=store,
    )
    assert result == 42
    assert call_count == 0


# ---------------------------------------------------------------------------
# Owner path: TimeoutError on irreversible write stores indeterminate
# ---------------------------------------------------------------------------


async def test_idempotent_owner_timeout_irreversible_stores_indeterminate():
    """A TimeoutError from _run_with_retries on an irreversible write is
    stored as INDETERMINATE in the owner path."""
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()

    async def slow(_x, *, context, **kw):
        await asyncio.sleep(1.0)
        return 42

    with pytest.raises(IndeterminateError):
        await execute(
            slow,
            0,
            _ctx(),
            policy=ExecutionPolicy(
                max_attempts=1, timeout_seconds=0.05, irreversible_write=True
            ),
            idempotency_key="k",
            idempotency_store=store,
        )

    # Re-call should replay indeterminate.
    async def ok(_x, *, context, **kw):
        return 99

    with pytest.raises(IndeterminateError):
        await execute(
            ok,
            0,
            _ctx(),
            policy=EXTERNAL_WRITE,
            idempotency_key="k",
            idempotency_store=store,
        )


# ---------------------------------------------------------------------------
# Owner path: ExecutionError base class releases claim
# ---------------------------------------------------------------------------


async def test_idempotent_owner_execution_error_releases_claim():
    """A bare ExecutionError from _run_with_retries releases the claim."""
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()

    async def failing(_x, *, context, **kw):
        raise ExecutionError("base")

    with pytest.raises(ExecutionError):
        await execute(
            failing,
            0,
            _ctx(),
            policy=EXTERNAL_WRITE,
            idempotency_key="k",
            idempotency_store=store,
        )

    # Key was released — a subsequent call can re-run.
    async def ok(_x, *, context, **kw):
        return 99

    result = await execute(
        ok,
        0,
        _ctx(),
        policy=EXTERNAL_WRITE,
        idempotency_key="k",
        idempotency_store=store,
    )
    assert result == 99


# ---------------------------------------------------------------------------
# Owner path: LostOwnershipError propagates
# ---------------------------------------------------------------------------


async def test_idempotent_owner_lost_ownership_propagates():
    """If the store raises LostOwnershipError during put_*, it propagates."""

    class LosingStore(InMemoryIdempotencyStore[int]):
        """A store that loses ownership on put_success."""

        async def put_success(self, key, owner_token, outcome):
            raise LostOwnershipError("lost ownership during put_success")

    store = LosingStore()

    async def op(_x, *, context, **kw):
        return 42

    with pytest.raises(LostOwnershipError):
        await execute(
            op,
            0,
            _ctx(),
            policy=NO_RETRY,
            idempotency_key="k",
            idempotency_store=store,
        )


# ---------------------------------------------------------------------------
# Owner path: TimeoutError on reversible write releases claim
# ---------------------------------------------------------------------------


async def test_idempotent_owner_timeout_reversible_releases_claim():
    """A TimeoutError from _run_with_retries on a reversible op releases the claim."""
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()

    async def slow(_x, *, context, **kw):
        await asyncio.sleep(1.0)
        return 42

    with pytest.raises(TimeoutError):
        await execute(
            slow,
            0,
            _ctx(),
            policy=ExecutionPolicy(max_attempts=1, timeout_seconds=0.05),
            idempotency_key="k",
            idempotency_store=store,
        )

    # Key was released — a subsequent call can re-run.
    async def ok(_x, *, context, **kw):
        return 99

    result = await execute(
        ok,
        0,
        _ctx(),
        policy=NO_RETRY,
        idempotency_key="k",
        idempotency_store=store,
    )
    assert result == 99


# ---------------------------------------------------------------------------
# Hard edge cases — token fencing, concurrent waiters, store errors
# ---------------------------------------------------------------------------


async def test_idempotent_token_fencing_stale_owner_cannot_overwrite():
    """A stale owner (whose lease was reclaimed) cannot overwrite a new owner's
    outcome. The store raises LostOwnershipError on token mismatch."""
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()

    # Owner A acquires the key.
    reserve_a = await store.reserve("k")
    assert reserve_a.status == ReserveStatus.ACQUIRED
    token_a = reserve_a.owner_token

    # Simulate lease reclaim: owner B acquires the same key after A's lease
    # expired. In the in-memory store there's no lease expiry, so we manually
    # remove A's in-progress entry and let B reserve.
    store._in_progress.pop("k", None)
    reserve_b = await store.reserve("k")
    assert reserve_b.status == ReserveStatus.ACQUIRED
    token_b = reserve_b.owner_token

    # Now A tries to write with its stale token — must fail.
    with pytest.raises(LostOwnershipError):
        await store.put_success(
            "k",
            token_a,
            SuccessOutcome(value=1, attempts=1, duration_ms=1.0),
        )

    # B can still write with its valid token.
    await store.put_success(
        "k",
        token_b,
        SuccessOutcome(value=2, attempts=1, duration_ms=1.0),
    )
    assert (await store.get("k")).value == 2


async def test_idempotent_multiple_waiters_share_result():
    """Multiple concurrent waiters all receive the owner's success result."""
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()
    started = asyncio.Event()
    release = asyncio.Event()
    call_count = 0

    async def slow_op(_x, *, context, **kw):
        nonlocal call_count
        call_count += 1
        started.set()
        await release.wait()
        return 42

    async def call_once():
        return await execute(
            slow_op,
            0,
            _ctx(),
            policy=NO_RETRY,
            idempotency_key="k",
            idempotency_store=store,
        )

    t1 = asyncio.create_task(call_once())
    await started.wait()
    t2 = asyncio.create_task(call_once())
    t3 = asyncio.create_task(call_once())
    await asyncio.sleep(0.05)
    release.set()

    r1 = await t1
    r2 = await t2
    r3 = await t3
    assert r1 == r2 == r3 == 42
    assert call_count == 1


async def test_idempotent_waiter_after_owner_released_retries():
    """If the owner releases without storing an outcome, the waiter gets
    RetryableError (so it can re-attempt from the top)."""
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()

    # Owner acquires.
    reserve = await store.reserve("k")
    token = reserve.owner_token

    # Waiter starts waiting.
    async def wait_once():
        return await store.wait_for_completion("k")

    wait_task = asyncio.create_task(wait_once())
    await asyncio.sleep(0.05)

    # Owner releases without storing an outcome.
    await store.release("k", token)

    # Waiter should get RetryableError.
    with pytest.raises(RetryableError):
        await wait_task


async def test_idempotent_renew_lease_valid_token():
    """renew_lease returns True when the caller still owns the key."""
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()
    reserve = await store.reserve("k")
    token = reserve.owner_token
    assert await store.renew_lease("k", token) is True


async def test_idempotent_renew_lease_wrong_token():
    """renew_lease returns False when the caller's token doesn't match."""
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()
    reserve = await store.reserve("k")
    _token = reserve.owner_token
    wrong_token = uuid.uuid4()
    assert await store.renew_lease("k", wrong_token) is False


async def test_idempotent_renew_lease_no_claim():
    """renew_lease returns False when there's no in-progress claim."""
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()
    assert await store.renew_lease("k", uuid.uuid4()) is False


async def test_idempotent_put_failure_with_wrong_token_raises():
    """put_failure with a wrong token raises LostOwnershipError."""
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()
    reserve = await store.reserve("k")
    _token = reserve.owner_token
    wrong_token = uuid.uuid4()
    with pytest.raises(LostOwnershipError):
        await store.put_failure(
            "k",
            wrong_token,
            PermanentFailureOutcome(error_type="X", error_message="bad"),
        )


async def test_idempotent_put_indeterminate_with_wrong_token_raises():
    """put_indeterminate with a wrong token raises LostOwnershipError."""
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()
    reserve = await store.reserve("k")
    _token = reserve.owner_token
    wrong_token = uuid.uuid4()
    with pytest.raises(LostOwnershipError):
        await store.put_indeterminate(
            "k",
            wrong_token,
            IndeterminateOutcome(error_type="X", error_message="bad"),
        )


async def test_idempotent_release_with_wrong_token_raises():
    """release with a wrong token raises LostOwnershipError."""
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()
    reserve = await store.reserve("k")
    _token = reserve.owner_token
    wrong_token = uuid.uuid4()
    with pytest.raises(LostOwnershipError):
        await store.release("k", wrong_token)


async def test_idempotent_put_with_no_in_progress_raises():
    """put_success with no in-progress claim raises LostOwnershipError."""
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()
    with pytest.raises(LostOwnershipError):
        await store.put_success(
            "k",
            uuid.uuid4(),
            SuccessOutcome(value=1, attempts=1, duration_ms=1.0),
        )


async def test_idempotent_wait_for_completion_after_terminal():
    """wait_for_completion after the owner already stored a terminal outcome
    returns the cached outcome (fall back to terminal lookup)."""
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()
    store.seed_outcome("k", SuccessOutcome(value=42, attempts=1, duration_ms=1.0))
    outcome = await store.wait_for_completion("k")
    assert isinstance(outcome, SuccessOutcome)
    assert outcome.value == 42


async def test_idempotent_wait_for_completion_no_outcome_raises():
    """wait_for_completion with no in-progress claim and no terminal outcome
    raises RetryableError."""
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()
    with pytest.raises(RetryableError):
        await store.wait_for_completion("k")


async def test_idempotent_store_satisfies_protocol():
    """InMemoryIdempotencyStore satisfies the IdempotencyStore protocol."""
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()
    assert isinstance(store, IdempotencyStore)


# ---------------------------------------------------------------------------
# Hard edge cases — operation name + tracing
# ---------------------------------------------------------------------------


async def test_execute_with_name_and_idempotency():
    """Tracing and idempotency work together."""
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()
    call_count = 0

    async def op(x: int, *, context, **kw):
        nonlocal call_count
        call_count += 1
        return x * 2

    r1 = await execute(
        op, 5, _ctx(), name="test_op", policy=NO_RETRY,
        idempotency_key="k", idempotency_store=store,
    )
    r2 = await execute(
        op, 5, _ctx(), name="test_op", policy=NO_RETRY,
        idempotency_key="k", idempotency_store=store,
    )
    assert r1 == r2 == 10
    assert call_count == 1


async def test_execute_with_name_and_retry():
    """Tracing works with retry policies."""
    attempts = 0

    async def op(_x, *, context, **kw):
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise RetryableError("transient")
        return 42

    result = await execute(
        op, 0, _ctx(), name="test_op", policy=EXTERNAL_READ,
    )
    assert result == 42
    assert attempts == 2


# ---------------------------------------------------------------------------
# Hard edge cases — policy boundaries
# ---------------------------------------------------------------------------


async def test_no_retry_policy_does_not_retry():
    """NO_RETRY policy executes exactly once."""
    attempts = 0

    async def op(_x, *, context, **kw):
        nonlocal attempts
        attempts += 1
        raise RetryableError("always")

    with pytest.raises(RetryableError):
        await execute(op, 0, _ctx(), policy=NO_RETRY)
    assert attempts == 1


async def test_external_read_retries_three_times():
    """EXTERNAL_READ retries up to 3 attempts."""
    attempts = 0

    async def op(_x, *, context, **kw):
        nonlocal attempts
        attempts += 1
        raise RetryableError("always")

    with pytest.raises(RetryableError):
        await execute(op, 0, _ctx(), policy=EXTERNAL_READ)
    assert attempts == 3


async def test_external_write_no_retry():
    """EXTERNAL_WRITE does not retry (irreversible)."""
    attempts = 0

    async def op(_x, *, context, **kw):
        nonlocal attempts
        attempts += 1
        raise RetryableError("always")

    with pytest.raises(RetryableError):
        await execute(op, 0, _ctx(), policy=EXTERNAL_WRITE)
    assert attempts == 1


# ---------------------------------------------------------------------------
# Hard edge cases — None timeout
# ---------------------------------------------------------------------------


async def test_none_timeout_no_timeout():
    """A policy with timeout_seconds=None never times out."""
    async def op(_x, *, context, **kw):
        await asyncio.sleep(0.05)
        return 42

    result = await execute(
        op,
        0,
        _ctx(),
        policy=ExecutionPolicy(max_attempts=1, timeout_seconds=None),
    )
    assert result == 42


# ---------------------------------------------------------------------------
# Hard edge cases — sync operation with retry
# ---------------------------------------------------------------------------


async def test_sync_operation_with_retry():
    """A sync operation is retried via asyncio.to_thread."""
    attempts = 0

    def op(_x, *, context, **kw):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RetryableError("transient")
        return 42

    result = await execute(op, 0, _ctx(), policy=EXTERNAL_READ)
    assert result == 42
    assert attempts == 3


async def test_sync_operation_returning_awaitable_with_retry():
    """A sync operation returning an awaitable is retried."""
    attempts = 0

    async def inner():
        return 42

    def op(_x, *, context, **kw):
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise RetryableError("transient")
        return inner()

    result = await execute(op, 0, _ctx(), policy=EXTERNAL_READ)
    assert result == 42
    assert attempts == 2

"""Tests for IdempotencyStore protocol, InMemoryIdempotencyStore, and
executor idempotency integration.

Ported from echo-v2's ``tests/runtime/test_idempotency.py``, adapted to
barcode-scanner's runtime (RunContext without operation_name, structured
errors with code/message/details).
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from src.runtime.context import RunContext
from src.runtime.errors import (
    IndeterminateError,
    PermanentError,
    RetryableError,
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
from src.runtime.policy import EXTERNAL_WRITE, NO_RETRY

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
# InMemoryIdempotencyStore unit tests
# ---------------------------------------------------------------------------


async def test_reserve_acquires_then_in_progress_then_completed():
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()

    r1 = await store.reserve("k")
    assert r1.status == ReserveStatus.ACQUIRED
    assert r1.owner_token is not None

    r2 = await store.reserve("k")
    assert r2.status == ReserveStatus.IN_PROGRESS
    assert r2.owner_token is None

    await store.put_success(
        "k", r1.owner_token, SuccessOutcome(value=1, attempts=1, duration_ms=1.0)
    )
    r3 = await store.reserve("k")
    assert r3.status == ReserveStatus.COMPLETED


async def test_get_returns_none_for_unknown_key():
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()
    assert await store.get("missing") is None


async def test_put_success_resolves_waiter():
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()

    r = await store.reserve("k")
    assert r.owner_token is not None

    # Start a waiter before storing the outcome.
    waiter_task = asyncio.create_task(store.wait_for_completion("k"))
    await asyncio.sleep(0.01)  # let the waiter register

    await store.put_success(
        "k", r.owner_token, SuccessOutcome(value=42, attempts=1, duration_ms=1.0)
    )
    outcome = await waiter_task
    assert isinstance(outcome, SuccessOutcome)
    assert outcome.value == 42


async def test_put_failure_resolves_waiter():
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()

    r = await store.reserve("k")
    assert r.owner_token is not None

    waiter_task = asyncio.create_task(store.wait_for_completion("k"))
    await asyncio.sleep(0.01)

    await store.put_failure(
        "k",
        r.owner_token,
        PermanentFailureOutcome(error_type="ValueError", error_message="bad"),
    )
    outcome = await waiter_task
    assert isinstance(outcome, PermanentFailureOutcome)
    assert outcome.error_type == "ValueError"


async def test_release_wakes_waiter_with_retryable_error():
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()

    r = await store.reserve("k")
    assert r.owner_token is not None

    waiter_task = asyncio.create_task(store.wait_for_completion("k"))
    await asyncio.sleep(0.01)

    await store.release("k", r.owner_token)

    with pytest.raises(RetryableError):
        await waiter_task


async def test_release_without_waiter_is_noop():
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()
    r = await store.reserve("k")
    assert r.owner_token is not None

    await store.release("k", r.owner_token)
    assert await store.get("k") is None


async def test_wait_for_completion_after_release_raises_retryable():
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()
    r = await store.reserve("k")
    assert r.owner_token is not None
    await store.release("k", r.owner_token)

    with pytest.raises(RetryableError):
        await store.wait_for_completion("k")


async def test_wrong_owner_token_raises_lost_ownership():
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()
    r = await store.reserve("k")
    assert r.owner_token is not None

    wrong_token = uuid.uuid4()
    with pytest.raises(LostOwnershipError):
        await store.put_success(
            "k", wrong_token, SuccessOutcome(value=1, attempts=1, duration_ms=1.0)
        )


async def test_renew_lease_returns_true_for_owner_false_for_non_owner():
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()
    r = await store.reserve("k")
    assert r.owner_token is not None

    assert await store.renew_lease("k", r.owner_token) is True
    assert await store.renew_lease("k", uuid.uuid4()) is False


async def test_in_memory_store_satisfies_protocol():
    """Structural check: InMemoryIdempotencyStore satisfies IdempotencyStore."""
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()
    assert isinstance(store, IdempotencyStore)


# ---------------------------------------------------------------------------
# Executor idempotency integration tests
# ---------------------------------------------------------------------------


async def test_cached_success_returned_without_rerunning():
    """A cached success outcome is returned without re-running the operation."""
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()
    store.seed_outcome(
        "k",
        SuccessOutcome(value=42, attempts=1, duration_ms=1.0),
    )

    call_count = 0

    async def op(_input: int, *, context, **kw):
        nonlocal call_count
        call_count += 1
        return 99

    result = await execute(
        op,
        1,
        _ctx(),
        policy=NO_RETRY,
        idempotency_key="k",
        idempotency_store=store,
    )
    assert result == 42
    assert call_count == 0


async def test_cached_permanent_failure_replayed_as_permanent_error():
    """A cached permanent failure is replayed as PermanentError."""
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()
    store.seed_outcome(
        "k",
        PermanentFailureOutcome(error_type="ValueError", error_message="bad input"),
    )

    async def op(_input: int, *, context, **kw):
        return 99

    with pytest.raises(PermanentError, match="bad input"):
        await execute(
            op,
            1,
            _ctx(),
            policy=NO_RETRY,
            idempotency_key="k",
            idempotency_store=store,
        )


async def test_retryable_failure_not_cached_subsequent_call_reruns():
    """A retryable failure does not cache — the next call re-runs."""
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()
    call_count = 0

    async def failing_op(_input: int, *, context, **kw):
        nonlocal call_count
        call_count += 1
        raise RetryableError("transient")

    with pytest.raises(RetryableError):
        await execute(
            failing_op,
            1,
            _ctx(),
            policy=NO_RETRY,
            idempotency_key="k",
            idempotency_store=store,
        )

    # Second call should re-run (not cached).
    async def ok_op(_input: int, *, context, **kw):
        nonlocal call_count
        call_count += 1
        return 42

    result = await execute(
        ok_op,
        1,
        _ctx(),
        policy=NO_RETRY,
        idempotency_key="k",
        idempotency_store=store,
    )
    assert result == 42
    assert call_count == 2


async def test_concurrent_duplicates_wait_and_share_result():
    """Two concurrent calls with the same key: one runs, the other waits."""
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_op(_input: int, *, context, **kw):
        started.set()
        await release.wait()
        return 42

    async def call_once():
        return await execute(
            slow_op,
            1,
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


async def test_indeterminate_outcome_blocks_re_execution():
    """An indeterminate outcome blocks re-execution — the key stays locked."""
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()
    store.seed_outcome(
        "k",
        IndeterminateOutcome(error_type="TimeoutError", error_message="timed out"),
    )

    async def op(_input: int, *, context, **kw):
        return 99

    with pytest.raises(IndeterminateError, match="timed out"):
        await execute(
            op,
            1,
            _ctx(),
            policy=EXTERNAL_WRITE,
            idempotency_key="k",
            idempotency_store=store,
        )


async def test_key_without_store_raises_value_error():
    async def op(_input: int, *, context, **kw):
        return 1

    with pytest.raises(ValueError, match="must be provided together"):
        await execute(op, 1, _ctx(), idempotency_key="k")


async def test_store_without_key_raises_value_error():
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()

    async def op(_input: int, *, context, **kw):
        return 1

    with pytest.raises(ValueError, match="must be provided together"):
        await execute(op, 1, _ctx(), idempotency_store=store)


async def test_no_idempotency_zero_behavior_change():
    """Without idempotency, execute() behaves exactly as before."""
    call_count = 0

    async def op(x: int, *, context, **kw):
        nonlocal call_count
        call_count += 1
        return x * 2

    result = await execute(op, 5, _ctx(), policy=NO_RETRY)
    assert result == 10
    assert call_count == 1


async def test_explicit_indeterminate_error_stores_indeterminate():
    """An IndeterminateError from the operation is stored as INDETERMINATE."""
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()

    async def op(_input: int, *, context, **kw):
        raise IndeterminateError("unknown outcome")

    with pytest.raises(IndeterminateError):
        await execute(
            op,
            1,
            _ctx(),
            policy=EXTERNAL_WRITE,
            idempotency_key="k",
            idempotency_store=store,
        )

    # Re-call should replay the indeterminate outcome.
    async def ok_op(_input: int, *, context, **kw):
        return 99

    with pytest.raises(IndeterminateError, match="unknown outcome"):
        await execute(
            ok_op,
            1,
            _ctx(),
            policy=EXTERNAL_WRITE,
            idempotency_key="k",
            idempotency_store=store,
        )


async def test_permanent_error_stores_failure():
    """A PermanentError from the operation is stored as FAILURE."""
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()

    async def op(_input: int, *, context, **kw):
        raise PermanentError("bad input")

    with pytest.raises(PermanentError):
        await execute(
            op,
            1,
            _ctx(),
            policy=NO_RETRY,
            idempotency_key="k",
            idempotency_store=store,
        )

    # Re-call should replay the failure.
    async def ok_op(_input: int, *, context, **kw):
        return 99

    with pytest.raises(PermanentError, match="bad input"):
        await execute(
            ok_op,
            1,
            _ctx(),
            policy=NO_RETRY,
            idempotency_key="k",
            idempotency_store=store,
        )


async def test_success_stores_and_replays():
    """A successful operation is stored and replayed on the next call."""
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()
    call_count = 0

    async def op(x: int, *, context, **kw):
        nonlocal call_count
        call_count += 1
        return x * 2

    result1 = await execute(
        op,
        5,
        _ctx(),
        policy=NO_RETRY,
        idempotency_key="k",
        idempotency_store=store,
    )
    assert result1 == 10
    assert call_count == 1

    # Second call should replay the cached result.
    result2 = await execute(
        op,
        5,
        _ctx(),
        policy=NO_RETRY,
        idempotency_key="k",
        idempotency_store=store,
    )
    assert result2 == 10
    assert call_count == 1


async def test_retryable_failure_on_irreversible_write_releases_claim():
    """A RetryableError on an irreversible write releases the claim (not indeterminate)."""
    store: InMemoryIdempotencyStore[int] = InMemoryIdempotencyStore()

    async def op(_input: int, *, context, **kw):
        raise RetryableError("connection refused")

    with pytest.raises(RetryableError):
        await execute(
            op,
            1,
            _ctx(),
            policy=EXTERNAL_WRITE,
            idempotency_key="k",
            idempotency_store=store,
        )

    # Key should be released — a subsequent call can re-run.
    async def ok_op(_input: int, *, context, **kw):
        return 42

    result = await execute(
        ok_op,
        1,
        _ctx(),
        policy=EXTERNAL_WRITE,
        idempotency_key="k",
        idempotency_store=store,
    )
    assert result == 42

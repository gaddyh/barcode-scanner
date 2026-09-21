"""Integration tests for PostgresIdempotencyStore.

These tests require a live Postgres instance. Set DATABASE_URL env var
or use the default test connection. Tests are skipped if DATABASE_URL
is not set.

Run: pytest tests/runtime/test_postgres_idempotency.py -v
"""

from __future__ import annotations

import os
import uuid

import pytest

from src.db import create_pool, init_db
from src.runtime.idempotency import (
    IndeterminateOutcome,
    LostOwnershipError,
    PermanentFailureOutcome,
    ReserveStatus,
    SuccessOutcome,
)
from src.runtime.postgres_idempotency import PostgresIdempotencyStore

TEST_DB_URL = os.getenv("DATABASE_URL", "")

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not TEST_DB_URL,
        reason="DATABASE_URL not set — skipping live Postgres integration tests",
    ),
]


@pytest.fixture
async def pool():
    """Create a connection pool, initialize schema, and clean up test data."""
    p = await create_pool(TEST_DB_URL, min_size=1, max_size=3)
    await init_db(p)
    yield p
    # Clean up idempotency test rows.
    async with p.acquire() as conn:
        await conn.execute("DELETE FROM idempotency_operations")
    await p.close()


@pytest.fixture
def store(pool):
    return PostgresIdempotencyStore(pool, lease_seconds=2)


# ---------------------------------------------------------------------------
# reserve + reclaim
# ---------------------------------------------------------------------------


async def test_reserve_acquires_then_in_progress_then_completed(pool, store):
    r1 = await store.reserve("k1")
    assert r1.status == ReserveStatus.ACQUIRED
    assert r1.owner_token is not None

    r2 = await store.reserve("k1")
    assert r2.status == ReserveStatus.IN_PROGRESS
    assert r2.owner_token is None

    await store.put_success(
        "k1", r1.owner_token, SuccessOutcome(value=1, attempts=1, duration_ms=1.0)
    )
    r3 = await store.reserve("k1")
    assert r3.status == ReserveStatus.COMPLETED


async def test_get_returns_none_for_unknown_key(store):
    assert await store.get("missing") is None


async def test_get_returns_none_for_in_progress(store):
    r = await store.reserve("k2")
    assert r.owner_token is not None
    assert await store.get("k2") is None


# ---------------------------------------------------------------------------
# owner writes (token-guarded)
# ---------------------------------------------------------------------------


async def test_put_success_stores_and_completes(store):
    r = await store.reserve("k3")
    assert r.owner_token is not None

    await store.put_success(
        "k3", r.owner_token, SuccessOutcome(value=42, attempts=1, duration_ms=1.0)
    )

    outcome = await store.get("k3")
    assert isinstance(outcome, SuccessOutcome)
    assert outcome.value == 42
    assert outcome.attempts == 1


async def test_put_failure_stores(store):
    r = await store.reserve("k4")
    assert r.owner_token is not None

    await store.put_failure(
        "k4",
        r.owner_token,
        PermanentFailureOutcome(error_type="ValueError", error_message="bad"),
    )

    outcome = await store.get("k4")
    assert isinstance(outcome, PermanentFailureOutcome)
    assert outcome.error_type == "ValueError"
    assert outcome.error_message == "bad"


async def test_put_indeterminate_stores(store):
    r = await store.reserve("k5")
    assert r.owner_token is not None

    await store.put_indeterminate(
        "k5",
        r.owner_token,
        IndeterminateOutcome(error_type="TimeoutError", error_message="timed out"),
    )

    outcome = await store.get("k5")
    assert isinstance(outcome, IndeterminateOutcome)
    assert outcome.error_type == "TimeoutError"


async def test_wrong_owner_token_raises_lost_ownership(store):
    r = await store.reserve("k6")
    assert r.owner_token is not None

    wrong_token = uuid.uuid4()
    with pytest.raises(LostOwnershipError):
        await store.put_success(
            "k6", wrong_token, SuccessOutcome(value=1, attempts=1, duration_ms=1.0)
        )


async def test_release_clears_claim(store):
    r = await store.reserve("k7")
    assert r.owner_token is not None

    await store.release("k7", r.owner_token)
    assert await store.get("k7") is None

    # After release, the key can be re-reserved.
    r2 = await store.reserve("k7")
    assert r2.status == ReserveStatus.ACQUIRED


async def test_release_with_wrong_token_raises_lost_ownership(store):
    r = await store.reserve("k8")
    assert r.owner_token is not None

    wrong_token = uuid.uuid4()
    with pytest.raises(LostOwnershipError):
        await store.release("k8", wrong_token)


# ---------------------------------------------------------------------------
# lease expiry + reclaim
# ---------------------------------------------------------------------------


async def test_expired_lease_can_be_reclaimed(pool):
    """A lease that has expired can be reclaimed by a new reserve()."""
    store = PostgresIdempotencyStore(pool, lease_seconds=1)

    r1 = await store.reserve("k9")
    assert r1.status == ReserveStatus.ACQUIRED
    assert r1.owner_token is not None

    # Wait for the lease to expire.
    import asyncio

    await asyncio.sleep(1.5)

    # A new reserve should reclaim the expired lease.
    r2 = await store.reserve("k9")
    assert r2.status == ReserveStatus.ACQUIRED
    assert r2.owner_token is not None
    assert r2.owner_token != r1.owner_token


async def test_renew_lease_extends(store):
    r = await store.reserve("k10")
    assert r.owner_token is not None

    assert await store.renew_lease("k10", r.owner_token) is True
    assert await store.renew_lease("k10", uuid.uuid4()) is False


# ---------------------------------------------------------------------------
# wait_for_completion
# ---------------------------------------------------------------------------


async def test_wait_for_completion_returns_outcome(store):
    r = await store.reserve("k11")
    assert r.owner_token is not None

    import asyncio

    async def store_after_delay():
        await asyncio.sleep(0.1)
        await store.put_success(
            "k11",
            r.owner_token,
            SuccessOutcome(value=42, attempts=1, duration_ms=1.0),
        )

    asyncio.create_task(store_after_delay())
    outcome = await store.wait_for_completion("k11")
    assert isinstance(outcome, SuccessOutcome)
    assert outcome.value == 42

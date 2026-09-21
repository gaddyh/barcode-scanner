"""Tests for PostgresIdempotencyStore.

The first section contains live-Postgres integration tests that require a
DATABASE_URL env var and are skipped otherwise.

The second section contains mock-based unit tests that exercise every branch
of the store without a real database (fake asyncpg pool/connection objects).
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.runtime.errors import RetryableError
from src.runtime.idempotency import (
    IndeterminateOutcome,
    LostOwnershipError,
    PermanentFailureOutcome,
    ReserveResult,
    ReserveStatus,
    SuccessOutcome,
)
from src.runtime.postgres_idempotency import (
    PostgresIdempotencyStore,
    _row_to_outcome,
    _serialize_failure,
    _serialize_indeterminate,
    _serialize_success,
)

TEST_DB_URL = os.getenv("DATABASE_URL", "")

# Live integration tests are skipped unless DATABASE_URL is set.
skip_no_db = pytest.mark.skipif(
    not TEST_DB_URL,
    reason="DATABASE_URL not set — skipping live Postgres integration tests",
)

# asyncio_mode = "auto" in pyproject.toml marks async tests automatically.


# ---------------------------------------------------------------------------
# Live Postgres integration tests (skipped without DATABASE_URL)
# ---------------------------------------------------------------------------


@pytest.fixture
async def pool():
    """Create a connection pool, initialize schema, and clean up test data."""
    from src.db import create_pool, init_db

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


@skip_no_db
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


@skip_no_db
async def test_get_returns_none_for_unknown_key(store):
    assert await store.get("missing") is None


@skip_no_db
async def test_get_returns_none_for_in_progress(store):
    r = await store.reserve("k2")
    assert r.owner_token is not None
    assert await store.get("k2") is None


# ---------------------------------------------------------------------------
# owner writes (token-guarded)
# ---------------------------------------------------------------------------


@skip_no_db
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


@skip_no_db
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


@skip_no_db
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


@skip_no_db
async def test_wrong_owner_token_raises_lost_ownership(store):
    r = await store.reserve("k6")
    assert r.owner_token is not None

    wrong_token = uuid.uuid4()
    with pytest.raises(LostOwnershipError):
        await store.put_success(
            "k6", wrong_token, SuccessOutcome(value=1, attempts=1, duration_ms=1.0)
        )


@skip_no_db
async def test_release_clears_claim(store):
    r = await store.reserve("k7")
    assert r.owner_token is not None

    await store.release("k7", r.owner_token)
    assert await store.get("k7") is None

    # After release, the key can be re-reserved.
    r2 = await store.reserve("k7")
    assert r2.status == ReserveStatus.ACQUIRED


@skip_no_db
async def test_release_with_wrong_token_raises_lost_ownership(store):
    r = await store.reserve("k8")
    assert r.owner_token is not None

    wrong_token = uuid.uuid4()
    with pytest.raises(LostOwnershipError):
        await store.release("k8", wrong_token)


# ---------------------------------------------------------------------------
# lease expiry + reclaim
# ---------------------------------------------------------------------------


@skip_no_db
async def test_expired_lease_can_be_reclaimed(pool):
    """A lease that has expired can be reclaimed by a new reserve()."""
    import asyncio

    store = PostgresIdempotencyStore(pool, lease_seconds=1)

    r1 = await store.reserve("k9")
    assert r1.status == ReserveStatus.ACQUIRED
    assert r1.owner_token is not None

    # Wait for the lease to expire.
    await asyncio.sleep(1.5)

    # A new reserve should reclaim the expired lease.
    r2 = await store.reserve("k9")
    assert r2.status == ReserveStatus.ACQUIRED
    assert r2.owner_token is not None
    assert r2.owner_token != r1.owner_token


@skip_no_db
async def test_renew_lease_extends(store):
    r = await store.reserve("k10")
    assert r.owner_token is not None

    assert await store.renew_lease("k10", r.owner_token) is True
    assert await store.renew_lease("k10", uuid.uuid4()) is False


# ---------------------------------------------------------------------------
# wait_for_completion
# ---------------------------------------------------------------------------


@skip_no_db
async def test_wait_for_completion_returns_outcome(store):
    import asyncio

    r = await store.reserve("k11")
    assert r.owner_token is not None

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


# ===========================================================================
# Mock-based unit tests (no real Postgres required)
# ===========================================================================


def _make_pool_conn():
    """Build a fake asyncpg pool + connection with AsyncMock methods.

    Returns ``(pool, conn)``. The connection's ``fetchval``, ``fetchrow`` and
    ``execute`` are ``AsyncMock`` objects whose ``side_effect``/``return_value``
    can be configured per test.
    """

    conn = MagicMock()
    conn.fetchval = AsyncMock()
    conn.fetchrow = AsyncMock()
    conn.execute = AsyncMock()

    class FakeAcquire:
        async def __aenter__(self_inner):
            return conn

        async def __aexit__(self_inner, *exc):
            return False

    pool = MagicMock()
    pool.acquire = MagicMock(return_value=FakeAcquire())
    return pool, conn


def _make_store(**kwargs):
    """Build a store backed by a fake pool; returns ``(store, conn)``."""
    pool, conn = _make_pool_conn()
    store = PostgresIdempotencyStore(pool, **kwargs)
    return store, conn


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------


async def test_get_unknown_key_returns_none():
    store, conn = _make_store()
    conn.fetchrow.return_value = None
    assert await store.get("missing") is None
    conn.fetchrow.assert_awaited_once()


async def test_get_in_progress_returns_none():
    store, conn = _make_store()
    conn.fetchrow.return_value = {"state": "IN_PROGRESS", "outcome": None}
    assert await store.get("k") is None


async def test_get_success_outcome():
    store, conn = _make_store()
    payload = json.dumps(
        {"tag": "success", "value": 42, "attempts": 3, "duration_ms": 1.5}
    )
    conn.fetchrow.return_value = {"state": "SUCCESS", "outcome": payload}
    outcome = await store.get("k")
    assert isinstance(outcome, SuccessOutcome)
    assert outcome.value == 42
    assert outcome.attempts == 3
    assert outcome.duration_ms == 1.5


async def test_get_failure_outcome():
    store, conn = _make_store()
    payload = json.dumps(
        {"tag": "failure", "error_type": "ValueError", "error_message": "bad"}
    )
    conn.fetchrow.return_value = {"state": "FAILURE", "outcome": payload}
    outcome = await store.get("k")
    assert isinstance(outcome, PermanentFailureOutcome)
    assert outcome.error_type == "ValueError"
    assert outcome.error_message == "bad"


async def test_get_indeterminate_outcome():
    store, conn = _make_store()
    payload = json.dumps(
        {"tag": "indeterminate", "error_type": "TimeoutError", "error_message": "x"}
    )
    conn.fetchrow.return_value = {"state": "INDETERMINATE", "outcome": payload}
    outcome = await store.get("k")
    assert isinstance(outcome, IndeterminateOutcome)
    assert outcome.error_type == "TimeoutError"


async def test_get_outcome_as_dict_not_string():
    """asyncpg may return JSONB already parsed as a dict."""
    store, conn = _make_store()
    payload = {"tag": "success", "value": 7, "attempts": 1, "duration_ms": 0.0}
    conn.fetchrow.return_value = {"state": "SUCCESS", "outcome": payload}
    outcome = await store.get("k")
    assert isinstance(outcome, SuccessOutcome)
    assert outcome.value == 7


# ---------------------------------------------------------------------------
# reserve
# ---------------------------------------------------------------------------


async def test_reserve_acquired_on_fresh_insert():
    store, conn = _make_store()
    conn.fetchval.return_value = "new-key"
    r = await store.reserve("new-key")
    assert r.status == ReserveStatus.ACQUIRED
    assert r.owner_token is not None
    assert isinstance(r.owner_token, uuid.UUID)
    # Only the INSERT fetchval should have been called.
    assert conn.fetchval.await_count == 1


async def test_reserve_in_progress_when_lease_active():
    store, conn = _make_store()
    future = datetime(2025, 1, 1, 12, 0, 5, tzinfo=UTC)
    now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
    conn.fetchval.side_effect = [None, now]
    conn.fetchrow.return_value = {
        "state": "IN_PROGRESS",
        "lease_expires_at": future,
    }
    r = await store.reserve("k")
    assert r.status == ReserveStatus.IN_PROGRESS
    assert r.owner_token is None


async def test_reserve_completed_when_terminal_state():
    store, conn = _make_store()
    conn.fetchval.return_value = None
    conn.fetchrow.return_value = {"state": "SUCCESS", "lease_expires_at": None}
    r = await store.reserve("k")
    assert r.status == ReserveStatus.COMPLETED
    assert r.owner_token is None


async def test_reserve_completed_when_failure_state():
    store, conn = _make_store()
    conn.fetchval.return_value = None
    conn.fetchrow.return_value = {"state": "FAILURE", "lease_expires_at": None}
    r = await store.reserve("k")
    assert r.status == ReserveStatus.COMPLETED


async def test_reserve_reclaims_expired_lease():
    store, conn = _make_store()
    expired = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
    now = datetime(2025, 1, 1, 12, 0, 5, tzinfo=UTC)
    # fetchval calls: INSERT -> None, SELECT now() -> now, UPDATE reclaim -> "k"
    conn.fetchval.side_effect = [None, now, "k"]
    conn.fetchrow.return_value = {
        "state": "IN_PROGRESS",
        "lease_expires_at": expired,
    }
    r = await store.reserve("k")
    assert r.status == ReserveStatus.ACQUIRED
    assert r.owner_token is not None


async def test_reserve_reclaims_when_lease_expires_at_null():
    """A NULL lease_expires_at is treated as expired (reclaim attempted)."""
    store, conn = _make_store()
    now = datetime(2025, 1, 1, 12, 0, 5, tzinfo=UTC)
    conn.fetchval.side_effect = [None, now, "k"]
    conn.fetchrow.return_value = {"state": "IN_PROGRESS", "lease_expires_at": None}
    r = await store.reserve("k")
    assert r.status == ReserveStatus.ACQUIRED
    assert r.owner_token is not None


async def test_reserve_reclaim_race_lost_to_in_progress():
    """Reclaim UPDATE affects 0 rows; re-read shows still IN_PROGRESS."""
    store, conn = _make_store()
    expired = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
    now = datetime(2025, 1, 1, 12, 0, 5, tzinfo=UTC)
    # fetchval: INSERT -> None, now() -> now, reclaim -> None (lost race)
    conn.fetchval.side_effect = [None, now, None]
    # fetchrow: first read (IN_PROGRESS), second read (IN_PROGRESS)
    conn.fetchrow.side_effect = [
        {"state": "IN_PROGRESS", "lease_expires_at": expired},
        {"state": "IN_PROGRESS"},
    ]
    r = await store.reserve("k")
    assert r.status == ReserveStatus.IN_PROGRESS
    assert r.owner_token is None


async def test_reserve_reclaim_race_lost_to_completed():
    """Reclaim UPDATE affects 0 rows; re-read shows terminal state."""
    store, conn = _make_store()
    expired = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
    now = datetime(2025, 1, 1, 12, 0, 5, tzinfo=UTC)
    conn.fetchval.side_effect = [None, now, None]
    conn.fetchrow.side_effect = [
        {"state": "IN_PROGRESS", "lease_expires_at": expired},
        {"state": "SUCCESS"},
    ]
    r = await store.reserve("k")
    assert r.status == ReserveStatus.COMPLETED


async def test_reserve_uses_configured_lease_seconds():
    store, conn = _make_store(lease_seconds=99)
    conn.fetchval.return_value = "k"
    await store.reserve("k")
    # The 4th positional arg to the INSERT fetchval is lease_seconds.
    args = conn.fetchval.await_args.args
    assert args[4] == 99


# ---------------------------------------------------------------------------
# put_success / put_failure / put_indeterminate
# ---------------------------------------------------------------------------


async def test_put_success_stores_outcome():
    store, conn = _make_store()
    conn.execute.return_value = "UPDATE 1"
    token = uuid.uuid4()
    outcome = SuccessOutcome(value=42, attempts=2, duration_ms=3.5)
    await store.put_success("k", token, outcome)
    assert conn.execute.await_count == 1
    args = conn.execute.await_args.args
    # args: (sql, state, json_outcome, key, str(token))
    assert args[1] == "SUCCESS"
    payload = json.loads(args[2])
    assert payload == {
        "tag": "success",
        "value": 42,
        "attempts": 2,
        "duration_ms": 3.5,
    }
    assert args[3] == "k"
    assert args[4] == str(token)


async def test_put_success_lost_ownership():
    store, conn = _make_store()
    conn.execute.return_value = "UPDATE 0"
    with pytest.raises(LostOwnershipError, match="ownership lost"):
        await store.put_success(
            "k", uuid.uuid4(), SuccessOutcome(value=1, attempts=1, duration_ms=1.0)
        )


async def test_put_failure_stores_outcome():
    store, conn = _make_store()
    conn.execute.return_value = "UPDATE 1"
    token = uuid.uuid4()
    outcome = PermanentFailureOutcome(error_type="ValueError", error_message="bad")
    await store.put_failure("k", token, outcome)
    args = conn.execute.await_args.args
    assert args[1] == "FAILURE"
    payload = json.loads(args[2])
    assert payload == {
        "tag": "failure",
        "error_type": "ValueError",
        "error_message": "bad",
    }


async def test_put_failure_lost_ownership():
    store, conn = _make_store()
    conn.execute.return_value = "UPDATE 0"
    with pytest.raises(LostOwnershipError):
        await store.put_failure(
            "k", uuid.uuid4(), PermanentFailureOutcome("E", "m")
        )


async def test_put_indeterminate_stores_outcome():
    store, conn = _make_store()
    conn.execute.return_value = "UPDATE 1"
    token = uuid.uuid4()
    outcome = IndeterminateOutcome(error_type="TimeoutError", error_message="x")
    await store.put_indeterminate("k", token, outcome)
    args = conn.execute.await_args.args
    assert args[1] == "INDETERMINATE"
    payload = json.loads(args[2])
    assert payload == {
        "tag": "indeterminate",
        "error_type": "TimeoutError",
        "error_message": "x",
    }


async def test_put_indeterminate_lost_ownership():
    store, conn = _make_store()
    conn.execute.return_value = "UPDATE 0"
    with pytest.raises(LostOwnershipError):
        await store.put_indeterminate(
            "k", uuid.uuid4(), IndeterminateOutcome("E", "m")
        )


# ---------------------------------------------------------------------------
# release
# ---------------------------------------------------------------------------


async def test_release_success():
    store, conn = _make_store()
    conn.execute.return_value = "DELETE 1"
    token = uuid.uuid4()
    await store.release("k", token)
    args = conn.execute.await_args.args
    assert args[1] == "k"
    assert args[2] == str(token)


async def test_release_lost_ownership():
    store, conn = _make_store()
    conn.execute.return_value = "DELETE 0"
    with pytest.raises(LostOwnershipError, match="ownership lost"):
        await store.release("k", uuid.uuid4())


# ---------------------------------------------------------------------------
# renew_lease
# ---------------------------------------------------------------------------


async def test_renew_lease_true_when_owner_matches():
    store, conn = _make_store()
    conn.execute.return_value = "UPDATE 1"
    token = uuid.uuid4()
    assert await store.renew_lease("k", token) is True
    args = conn.execute.await_args.args
    # args: (sql, lease_seconds, key, str(token))
    assert args[2] == "k"
    assert args[3] == str(token)


async def test_renew_lease_false_when_owner_mismatch():
    store, conn = _make_store()
    conn.execute.return_value = "UPDATE 0"
    assert await store.renew_lease("k", uuid.uuid4()) is False


async def test_renew_lease_uses_configured_lease_seconds():
    store, conn = _make_store(lease_seconds=77)
    conn.execute.return_value = "UPDATE 1"
    await store.renew_lease("k", uuid.uuid4())
    args = conn.execute.await_args.args
    assert args[1] == 77


# ---------------------------------------------------------------------------
# wait_for_completion
# ---------------------------------------------------------------------------


async def test_wait_for_completion_returns_outcome_immediately():
    store, _ = _make_store()
    outcome = SuccessOutcome(value=42, attempts=1, duration_ms=1.0)
    store.get = AsyncMock(return_value=outcome)
    result = await store.wait_for_completion("k")
    assert result is outcome
    store.get.assert_awaited_once_with("k")


async def test_wait_for_completion_polls_then_returns():
    store, _ = _make_store()
    outcome = SuccessOutcome(value=1, attempts=1, duration_ms=1.0)
    store.get = AsyncMock(side_effect=[None, outcome])
    store.reserve = AsyncMock(
        return_value=ReserveResult(ReserveStatus.IN_PROGRESS)
    )
    result = await store.wait_for_completion("k")
    assert result is outcome
    # get called twice (first None, then outcome); reserve called once.
    assert store.get.await_count == 2
    assert store.reserve.await_count == 1


async def test_wait_for_completion_reclaim_raises_retryable():
    store, _ = _make_store()
    token = uuid.uuid4()
    store.get = AsyncMock(return_value=None)
    store.reserve = AsyncMock(
        return_value=ReserveResult(ReserveStatus.ACQUIRED, token)
    )
    store.release = AsyncMock()
    with pytest.raises(RetryableError, match="lease expired"):
        await store.wait_for_completion("k")
    store.release.assert_awaited_once_with("k", token)


async def test_wait_for_completion_completed_returns_outcome():
    store, _ = _make_store()
    outcome = SuccessOutcome(value=5, attempts=1, duration_ms=1.0)
    store.get = AsyncMock(side_effect=[None, outcome])
    store.reserve = AsyncMock(return_value=ReserveResult(ReserveStatus.COMPLETED))
    result = await store.wait_for_completion("k")
    assert result is outcome


async def test_wait_for_completion_completed_but_no_outcome_raises():
    store, _ = _make_store()
    store.get = AsyncMock(return_value=None)
    store.reserve = AsyncMock(return_value=ReserveResult(ReserveStatus.COMPLETED))
    with pytest.raises(RetryableError, match="did not complete"):
        await store.wait_for_completion("k")


async def test_wait_for_completion_timeout_raises_retryable(monkeypatch):
    import src.runtime.postgres_idempotency as mod

    monkeypatch.setattr(mod, "_POLL_TIMEOUT_SECONDS", 0.0)
    monkeypatch.setattr(mod, "_POLL_INTERVAL_SECONDS", 0.0)
    store, _ = _make_store()
    store.get = AsyncMock(return_value=None)
    store.reserve = AsyncMock(
        return_value=ReserveResult(ReserveStatus.IN_PROGRESS)
    )
    with pytest.raises(RetryableError, match="Timed out"):
        await store.wait_for_completion("k")


# ---------------------------------------------------------------------------
# serialization helpers
# ---------------------------------------------------------------------------


def test_serialize_success():
    payload = _serialize_success(SuccessOutcome(value=1, attempts=2, duration_ms=3.0))
    assert payload == {
        "tag": "success",
        "value": 1,
        "attempts": 2,
        "duration_ms": 3.0,
    }


def test_serialize_failure():
    payload = _serialize_failure(
        PermanentFailureOutcome(error_type="E", error_message="m")
    )
    assert payload == {"tag": "failure", "error_type": "E", "error_message": "m"}


def test_serialize_indeterminate():
    payload = _serialize_indeterminate(
        IndeterminateOutcome(error_type="E", error_message="m")
    )
    assert payload == {
        "tag": "indeterminate",
        "error_type": "E",
        "error_message": "m",
    }


def test_row_to_outcome_none():
    assert _row_to_outcome(None) is None


def test_row_to_outcome_success():
    outcome = _row_to_outcome(
        json.dumps({"tag": "success", "value": 9, "attempts": 1, "duration_ms": 0.0})
    )
    assert isinstance(outcome, SuccessOutcome)
    assert outcome.value == 9


def test_row_to_outcome_failure():
    outcome = _row_to_outcome(
        json.dumps({"tag": "failure", "error_type": "E", "error_message": "m"})
    )
    assert isinstance(outcome, PermanentFailureOutcome)


def test_row_to_outcome_indeterminate():
    outcome = _row_to_outcome(
        json.dumps({"tag": "indeterminate", "error_type": "E", "error_message": "m"})
    )
    assert isinstance(outcome, IndeterminateOutcome)


def test_row_to_outcome_dict_input():
    outcome = _row_to_outcome({"tag": "success", "value": 1, "attempts": 1})
    assert isinstance(outcome, SuccessOutcome)
    assert outcome.value == 1
    assert outcome.duration_ms == 0.0


def test_row_to_outcome_unknown_tag_returns_none():
    assert _row_to_outcome(json.dumps({"tag": "unknown"})) is None


def test_row_to_outcome_missing_fields_use_defaults():
    outcome = _row_to_outcome(json.dumps({"tag": "success", "value": 1}))
    assert isinstance(outcome, SuccessOutcome)
    assert outcome.attempts == 0
    assert outcome.duration_ms == 0.0


def test_row_to_outcome_failure_missing_fields():
    outcome = _row_to_outcome(json.dumps({"tag": "failure"}))
    assert isinstance(outcome, PermanentFailureOutcome)
    assert outcome.error_type == ""
    assert outcome.error_message == ""


# ---------------------------------------------------------------------------
# constructor
# ---------------------------------------------------------------------------


def test_store_uses_default_lease_seconds():
    pool, _ = _make_pool_conn()
    store = PostgresIdempotencyStore(pool)
    assert store._lease_seconds == 30


def test_store_custom_lease_seconds():
    pool, _ = _make_pool_conn()
    store = PostgresIdempotencyStore(pool, lease_seconds=10)
    assert store._lease_seconds == 10

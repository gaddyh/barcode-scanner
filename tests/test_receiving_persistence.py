"""Live Postgres integration tests for ReceivingSessionStore (PR A).

Requires DATABASE_URL to be set. Skipped otherwise — same pattern as
``tests/runtime/test_postgres_idempotency.py``. Run locally with:

    docker run -d --name pg-test -p 5433:5432 \\
        -e POSTGRES_USER=scanner -e POSTGRES_PASSWORD=scanner \\
        -e POSTGRES_DB=scanner postgres:16-alpine
    DATABASE_URL=postgres://scanner:scanner@localhost:5433/scanner \\
        pytest tests/test_receiving_persistence.py -v

In CI, the postgres:16-alpine service provides DATABASE_URL.
"""

from __future__ import annotations

import os

import pytest

from src.db import create_pool, init_db
from src.domain.receiving import ReceivingSessionStatus
from src.session_repository import ReceivingSessionStore

TEST_DB_URL = os.getenv("DATABASE_URL", "")

skip_no_db = pytest.mark.skipif(
    not TEST_DB_URL,
    reason="DATABASE_URL not set — skipping live Postgres integration tests",
)


@pytest.fixture
async def pool():
    """Create a connection pool, initialize schema, and clean up test data."""
    if not TEST_DB_URL:
        pytest.skip("DATABASE_URL not set")
    p = await create_pool(TEST_DB_URL, min_size=1, max_size=3)
    await init_db(p)
    yield p
    async with p.acquire() as conn:
        await conn.execute("DELETE FROM session_missing")
        await conn.execute("DELETE FROM session_items")
        await conn.execute("DELETE FROM sessions")
    await p.close()


@pytest.fixture
def store(pool):
    return ReceivingSessionStore(pool)


# ---------------------------------------------------------------------------
# create + get
# ---------------------------------------------------------------------------


@skip_no_db
async def test_create_and_get(store):
    await store.create_receiving_session(
        "sess-1",
        customer_id="cust-acme",
        branch_id="branch-acme-main",
        action="create_order",
        participant_id="participant-1",
    )
    session = await store.get_receiving_session("sess-1")
    assert session is not None
    assert session.session_id == "sess-1"
    assert session.customer_id == "cust-acme"
    assert session.branch_id == "branch-acme-main"
    assert session.action == "create_order"
    assert session.status == ReceivingSessionStatus.ACTIVE
    assert not session.frozen
    assert session.external_order_id is None
    assert session.boxes == []


@skip_no_db
async def test_get_nonexistent_returns_none(store):
    assert await store.get_receiving_session("missing") is None


# ---------------------------------------------------------------------------
# find_open_submission_by_participant
# ---------------------------------------------------------------------------


@skip_no_db
async def test_find_open_submission_by_participant(store):
    await store.create_receiving_session(
        "sess-1",
        customer_id="cust-acme",
        branch_id="branch-acme-main",
        action="create_order",
        participant_id="participant-1",
    )
    found = await store.find_open_submission_by_participant("participant-1")
    assert found is not None
    assert found.session_id == "sess-1"


@skip_no_db
async def test_find_open_submission_returns_none_for_unknown_participant(store):
    assert await store.find_open_submission_by_participant("unknown") is None


@skip_no_db
async def test_find_open_submission_excludes_submitted(store):
    await store.create_receiving_session(
        "sess-1",
        customer_id="cust-acme",
        branch_id="branch-acme-main",
        action="create_order",
        participant_id="participant-1",
    )
    # Freeze + mark submitted.
    await store.freeze_submission("sess-1", {"items": []})
    await store.mark_submitted("sess-1", 42)
    assert await store.find_open_submission_by_participant("participant-1") is None


@skip_no_db
async def test_find_open_submission_includes_submission_unknown(store):
    await store.create_receiving_session(
        "sess-1",
        customer_id="cust-acme",
        branch_id="branch-acme-main",
        action="create_order",
        participant_id="participant-1",
    )
    await store.freeze_submission("sess-1", {"items": []})
    await store.mark_submission_unknown("sess-1")
    found = await store.find_open_submission_by_participant("participant-1")
    assert found is not None
    assert found.status == ReceivingSessionStatus.SUBMISSION_UNKNOWN


# ---------------------------------------------------------------------------
# freeze_submission (CAS: ACTIVE → SUBMITTING)
# ---------------------------------------------------------------------------


@skip_no_db
async def test_freeze_persists_payload_and_hash(store, pool):
    await store.create_receiving_session(
        "sess-1",
        customer_id="cust-acme",
        branch_id="branch-acme-main",
        action="create_order",
        participant_id="participant-1",
    )
    payload = {"session_id": "sess-1", "items": [{"barcode_value": "AAA", "quantity": 2}]}
    ok = await store.freeze_submission("sess-1", payload)
    assert ok is True

    session = await store.get_receiving_session("sess-1")
    assert session is not None
    assert session.status == ReceivingSessionStatus.SUBMITTING
    assert session.frozen is True

    # Frozen payload is persisted verbatim.
    frozen = await store.get_frozen_payload("sess-1")
    assert frozen == payload

    # Hash is persisted.
    import hashlib
    import json

    expected_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT frozen_payload_hash FROM sessions WHERE id = $1",
            "sess-1",
        )
    assert row["frozen_payload_hash"] == expected_hash


@skip_no_db
async def test_freeze_cas_only_one_winner(store):
    await store.create_receiving_session(
        "sess-1",
        customer_id="cust-acme",
        branch_id="branch-acme-main",
        action="create_order",
        participant_id="participant-1",
    )
    # First freeze wins.
    ok1 = await store.freeze_submission("sess-1", {"items": []})
    ok2 = await store.freeze_submission("sess-1", {"items": ["different"]})
    assert ok1 is True
    assert ok2 is False

    # The frozen payload is from the winner.
    frozen = await store.get_frozen_payload("sess-1")
    assert frozen == {"items": []}


@skip_no_db
async def test_freeze_nonexistent_returns_false(store):
    ok = await store.freeze_submission("missing", {"items": []})
    assert ok is False


# ---------------------------------------------------------------------------
# mark_submitted (CAS: SUBMITTING → SUBMITTED)
# ---------------------------------------------------------------------------


@skip_no_db
async def test_mark_submitted(store):
    await store.create_receiving_session(
        "sess-1",
        customer_id="cust-acme",
        branch_id="branch-acme-main",
        action="create_order",
        participant_id="participant-1",
    )
    await store.freeze_submission("sess-1", {"items": []})
    ok = await store.mark_submitted("sess-1", 99)
    assert ok is True

    session = await store.get_receiving_session("sess-1")
    assert session is not None
    assert session.status == ReceivingSessionStatus.SUBMITTED
    assert session.external_order_id == 99


@skip_no_db
async def test_mark_submitted_cas_rejects_active(store):
    await store.create_receiving_session(
        "sess-1",
        customer_id="cust-acme",
        branch_id="branch-acme-main",
        action="create_order",
        participant_id="participant-1",
    )
    # Not frozen yet — should fail.
    ok = await store.mark_submitted("sess-1", 99)
    assert ok is False


# ---------------------------------------------------------------------------
# mark_submission_unknown (CAS: SUBMITTING → SUBMISSION_UNKNOWN)
# ---------------------------------------------------------------------------


@skip_no_db
async def test_mark_submission_unknown(store):
    await store.create_receiving_session(
        "sess-1",
        customer_id="cust-acme",
        branch_id="branch-acme-main",
        action="create_order",
        participant_id="participant-1",
    )
    await store.freeze_submission("sess-1", {"items": []})
    ok = await store.mark_submission_unknown("sess-1")
    assert ok is True

    session = await store.get_receiving_session("sess-1")
    assert session is not None
    assert session.status == ReceivingSessionStatus.SUBMISSION_UNKNOWN


@skip_no_db
async def test_mark_submission_unknown_idempotent(store):
    """Re-transitioning SUBMISSION_UNKNOWN → SUBMISSION_UNKNOWN succeeds."""
    await store.create_receiving_session(
        "sess-1",
        customer_id="cust-acme",
        branch_id="branch-acme-main",
        action="create_order",
        participant_id="participant-1",
    )
    await store.freeze_submission("sess-1", {"items": []})
    ok1 = await store.mark_submission_unknown("sess-1")
    ok2 = await store.mark_submission_unknown("sess-1")
    assert ok1 is True
    assert ok2 is True


# ---------------------------------------------------------------------------
# revert_to_active (CAS: SUBMITTING → ACTIVE)
# ---------------------------------------------------------------------------


@skip_no_db
async def test_revert_to_active_clears_frozen_payload(store):
    await store.create_receiving_session(
        "sess-1",
        customer_id="cust-acme",
        branch_id="branch-acme-main",
        action="create_order",
        participant_id="participant-1",
    )
    await store.freeze_submission("sess-1", {"items": [{"v": "AAA"}]})
    ok = await store.revert_to_active("sess-1")
    assert ok is True

    session = await store.get_receiving_session("sess-1")
    assert session is not None
    assert session.status == ReceivingSessionStatus.ACTIVE
    assert not session.frozen

    # Frozen payload is cleared.
    assert await store.get_frozen_payload("sess-1") is None


@skip_no_db
async def test_revert_to_active_cas_rejects_submitted(store):
    await store.create_receiving_session(
        "sess-1",
        customer_id="cust-acme",
        branch_id="branch-acme-main",
        action="create_order",
        participant_id="participant-1",
    )
    await store.freeze_submission("sess-1", {"items": []})
    await store.mark_submitted("sess-1", 42)
    ok = await store.revert_to_active("sess-1")
    assert ok is False


# ---------------------------------------------------------------------------
# Restart survival — the whole point of persistence
# ---------------------------------------------------------------------------


@skip_no_db
async def test_frozen_payload_survives_store_recreation(pool):
    """A new store instance (simulating a restart) reads the frozen payload."""
    store1 = ReceivingSessionStore(pool)
    await store1.create_receiving_session(
        "sess-1",
        customer_id="cust-acme",
        branch_id="branch-acme-main",
        action="create_order",
        participant_id="participant-1",
    )
    payload = {"session_id": "sess-1", "items": [{"barcode_value": "AAA", "quantity": 3}]}
    await store1.freeze_submission("sess-1", payload)
    await store1.mark_submission_unknown("sess-1")

    # Simulate a restart — new store instance, same pool.
    store2 = ReceivingSessionStore(pool)
    session = await store2.get_receiving_session("sess-1")
    assert session is not None
    assert session.status == ReceivingSessionStatus.SUBMISSION_UNKNOWN

    frozen = await store2.get_frozen_payload("sess-1")
    assert frozen == payload

    # Retry: the store can still mark it submitted with the same frozen payload.
    ok = await store2.mark_submitted("sess-1", 77)
    assert ok is True

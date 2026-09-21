"""Tests for the end-to-end receiving flow.

Covers:
- Domain layer: PhysicalBox, ReceivingSession, Discrepancy, aggregate
  quantities (multiset), state transitions, frozen enforcement.
- Receiving router: create session, upload image, submit, double-submit
  idempotency, SUBMISSION_UNKNOWN, pre-submit failure, invalid state.
- Hard edge cases: empty session submit rejected, duplicate barcode values
  count separately, frozen session rejects edits, indeterminate replay.

API tests require DATABASE_URL and are skipped otherwise — same pattern as
``tests/runtime/test_postgres_idempotency.py``. Run locally with:

    docker run -d --name pg-test -p 5433:5432 \\
        -e POSTGRES_USER=scanner -e POSTGRES_PASSWORD=scanner \\
        -e POSTGRES_DB=scanner postgres:16-alpine
    DATABASE_URL=postgres://scanner:scanner@localhost:5433/scanner \\
        pytest tests/test_receiving.py -v
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from src.domain.receiving import (
    Discrepancy,
    PhysicalBox,
    ReceivingSession,
    ReceivingSessionStatus,
)
from src.integrations.priority.models import (
    Branch,
    CreateDraftOrderResult,
    Customer,
)
from src.runtime.errors import IndeterminateError, PermanentError, RetryableError

TEST_DB_URL = os.getenv("DATABASE_URL", "")

skip_no_db = pytest.mark.skipif(
    not TEST_DB_URL,
    reason="DATABASE_URL not set — skipping live Postgres API tests",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client_no_raise():
    """TestClient that doesn't re-raise server exceptions (for 500 paths).

    Used by domain-level tests that don't hit the DB.
    """
    from src.main import app

    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
async def db_pool():
    """Create a Postgres pool for API tests, init schema, clean up between tests."""
    if not TEST_DB_URL:
        pytest.skip("DATABASE_URL not set")
    from src.db import create_pool, init_db

    p = await create_pool(TEST_DB_URL, min_size=1, max_size=3)
    await init_db(p)
    yield p
    async with p.acquire() as conn:
        await conn.execute("DELETE FROM session_missing")
        await conn.execute("DELETE FROM session_items")
        await conn.execute("DELETE FROM sessions")
        await conn.execute("DELETE FROM idempotency_operations")
    await p.close()


@pytest.fixture
def receiving_store(db_pool):
    """ReceivingSessionStore backed by the test Postgres pool."""
    from src.session_repository import ReceivingSessionStore

    return ReceivingSessionStore(db_pool)


@pytest.fixture
def in_memory_idempotency_store():
    """In-memory idempotency store for API tests.

    The Postgres idempotency store has its own comprehensive tests in
    ``tests/runtime/test_postgres_idempotency.py``. Here we use in-memory
    to focus on the receiving state machine.
    """
    from src.runtime.idempotency import InMemoryIdempotencyStore

    store = InMemoryIdempotencyStore[dict[str, Any]]()
    return store


@pytest.fixture
async def async_client(receiving_store, in_memory_idempotency_store):
    """httpx.AsyncClient with patched stores, using ASGI transport.

    Using AsyncClient (not TestClient) ensures the same event loop is used
    for both the test setup and the HTTP calls — asyncpg connections are
    tied to their event loop.
    """
    from src.main import app

    with patch(
        "src.api.receiving._get_receiving_store",
        return_value=receiving_store,
    ), patch(
        "src.api.receiving._get_idempotency_store",
        return_value=in_memory_idempotency_store,
    ):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            yield client


def _make_session(
    session_id: str = "sess-1",
    customer_id: str = "cust-acme",
    branch_id: str = "branch-acme-main",
    action: str = "create_order",
    boxes: list[PhysicalBox] | None = None,
    status: ReceivingSessionStatus = ReceivingSessionStatus.ACTIVE,
) -> ReceivingSession:
    """Create a ReceivingSession domain object (for unit tests)."""
    return ReceivingSession(
        session_id=session_id,
        customer_id=customer_id,
        branch_id=branch_id,
        action=action,
        boxes=list(boxes) if boxes else [],
        status=status,
    )


async def _make_session_in_db(
    store: Any,
    session_id: str = "sess-1",
    customer_id: str = "cust-acme",
    branch_id: str = "branch-acme-main",
    action: str = "create_order",
    boxes: list[PhysicalBox] | None = None,
    participant_id: str | None = None,
) -> ReceivingSession:
    """Create a receiving session in the Postgres store (for API tests).

    Boxes are added by directly inserting session_items rows, since the
    upload_image endpoint goes through analyze_image which needs real
    barcode images. For submit/inspect tests we need boxes pre-populated.
    """
    await store.create_receiving_session(
        session_id,
        customer_id=customer_id,
        branch_id=branch_id,
        action=action,
        participant_id=participant_id,
    )

    # Insert boxes directly into session_items.
    if boxes:
        async with store._pool.acquire() as conn:
            for i, box in enumerate(boxes):
                await conn.execute(
                    """INSERT INTO session_items
                           (session_id, barcode_value, barcode_format, label_index,
                            source_image)
                       VALUES ($1, $2, $3, $4, 0)""",
                    session_id,
                    box.barcode_value,
                    box.barcode_format or None,
                    box.label_index if box.label_index is not None else i,
                )

    session = await store.get_receiving_session(session_id)
    assert session is not None
    return session  # type: ignore[no-any-return]


def _mock_gateway_success(order_id: int = 42):
    """Mock PriorityGateway that succeeds.

    Returns real Customer/Branch dataclasses (not dicts) to match the
    actual LocalPriorityGateway contract.
    """
    gateway = MagicMock()
    gateway.create_draft_order = AsyncMock(
        return_value=CreateDraftOrderResult(
            order_id=order_id, session_id="sess-1", status="draft"
        )
    )
    gateway.customers = AsyncMock(
        return_value=[
            Customer(id="cust-acme", name="Acme Retail"),
            Customer(id="cust-northstar", name="Northstar Shoes"),
        ]
    )
    gateway.branches = AsyncMock(
        return_value=[
            Branch(id="branch-acme-main", name="Acme Main Store", customer_id="cust-acme"),
            Branch(id="branch-acme-outlet", name="Acme Outlet", customer_id="cust-acme"),
        ]
    )
    return gateway


def _mock_gateway_indeterminate():
    """Mock PriorityGateway that raises IndeterminateError."""
    gateway = MagicMock()
    gateway.create_draft_order = AsyncMock(
        side_effect=IndeterminateError("connection lost after submit")
    )
    gateway.customers = AsyncMock(
        return_value=[
            Customer(id="cust-acme", name="Acme Retail"),
            Customer(id="cust-northstar", name="Northstar Shoes"),
        ]
    )
    gateway.branches = AsyncMock(
        return_value=[
            Branch(id="branch-acme-main", name="Acme Main Store", customer_id="cust-acme"),
            Branch(id="branch-acme-outlet", name="Acme Outlet", customer_id="cust-acme"),
        ]
    )
    return gateway


def _mock_gateway_retryable():
    """Mock PriorityGateway that raises RetryableError (pre-submit)."""
    gateway = MagicMock()
    gateway.create_draft_order = AsyncMock(
        side_effect=RetryableError("connection refused")
    )
    gateway.customers = AsyncMock(
        return_value=[
            Customer(id="cust-acme", name="Acme Retail"),
            Customer(id="cust-northstar", name="Northstar Shoes"),
        ]
    )
    gateway.branches = AsyncMock(
        return_value=[
            Branch(id="branch-acme-main", name="Acme Main Store", customer_id="cust-acme"),
            Branch(id="branch-acme-outlet", name="Acme Outlet", customer_id="cust-acme"),
        ]
    )
    return gateway


def _mock_gateway_permanent():
    """Mock PriorityGateway that raises PermanentError."""
    gateway = MagicMock()
    gateway.create_draft_order = AsyncMock(
        side_effect=PermanentError("duplicate session_id")
    )
    gateway.customers = AsyncMock(
        return_value=[
            Customer(id="cust-acme", name="Acme Retail"),
            Customer(id="cust-northstar", name="Northstar Shoes"),
        ]
    )
    gateway.branches = AsyncMock(
        return_value=[
            Branch(id="branch-acme-main", name="Acme Main Store", customer_id="cust-acme"),
            Branch(id="branch-acme-outlet", name="Acme Outlet", customer_id="cust-acme"),
        ]
    )
    return gateway


# ===========================================================================
# Domain: PhysicalBox
# ===========================================================================


class TestPhysicalBox:
    def test_create(self):
        box = PhysicalBox(barcode_value="1234567890123", barcode_format="Code128")
        assert box.barcode_value == "1234567890123"
        assert box.barcode_format == "Code128"
        assert box.label_index is None

    def test_empty_barcode_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            PhysicalBox(barcode_value="")

    def test_frozen(self):
        box = PhysicalBox(barcode_value="x")
        with pytest.raises(AttributeError):
            box.barcode_value = "y"  # type: ignore[misc]

    def test_duplicate_values_are_separate_instances(self):
        b1 = PhysicalBox(barcode_value="AAA")
        b2 = PhysicalBox(barcode_value="AAA")
        assert b1 is not b2
        assert b1.barcode_value == b2.barcode_value


# ===========================================================================
# Domain: Discrepancy
# ===========================================================================


class TestDiscrepancy:
    def test_complete(self):
        d = Discrepancy(expected=6, found=6)
        assert d.missing == 0
        assert d.is_complete

    def test_missing(self):
        d = Discrepancy(expected=6, found=4)
        assert d.missing == 2
        assert not d.is_complete

    def test_zero_expected_not_complete(self):
        d = Discrepancy(expected=0, found=0)
        assert not d.is_complete

    def test_found_exceeds_expected(self):
        d = Discrepancy(expected=3, found=5)
        assert d.missing == 0
        assert d.is_complete


# ===========================================================================
# Domain: ReceivingSession — state transitions
# ===========================================================================


class TestReceivingSessionTransitions:
    def test_starts_active(self):
        s = ReceivingSession("s1", "c1", "b1", "create_order")
        assert s.status == ReceivingSessionStatus.ACTIVE
        assert not s.frozen

    def test_freeze_transitions_to_submitting(self):
        s = _make_session()
        s.freeze()
        assert s.status == ReceivingSessionStatus.SUBMITTING
        assert s.frozen

    def test_freeze_from_non_active_raises(self):
        s = _make_session(status=ReceivingSessionStatus.SUBMITTED)
        with pytest.raises(ValueError, match="Cannot freeze"):
            s.freeze()

    def test_mark_submitted(self):
        s = _make_session()
        s.freeze()
        s.mark_submitted(42)
        assert s.status == ReceivingSessionStatus.SUBMITTED
        assert s.external_order_id == 42

    def test_mark_submitted_from_active_raises(self):
        s = _make_session()
        with pytest.raises(ValueError, match="Cannot mark submitted"):
            s.mark_submitted(42)

    def test_mark_submitted_from_submission_unknown(self):
        s = _make_session()
        s.freeze()
        s.mark_submission_unknown()
        s.mark_submitted(42)
        assert s.status == ReceivingSessionStatus.SUBMITTED
        assert s.external_order_id == 42

    def test_mark_submission_unknown(self):
        s = _make_session()
        s.freeze()
        s.mark_submission_unknown()
        assert s.status == ReceivingSessionStatus.SUBMISSION_UNKNOWN

    def test_mark_submission_unknown_from_active_raises(self):
        s = _make_session()
        with pytest.raises(ValueError, match="Cannot mark submission_unknown"):
            s.mark_submission_unknown()

    def test_revert_to_active(self):
        s = _make_session()
        s.freeze()
        s.revert_to_active()
        assert s.status == ReceivingSessionStatus.ACTIVE
        assert not s.frozen

    def test_revert_from_submitted_raises(self):
        s = _make_session(status=ReceivingSessionStatus.SUBMITTED)
        with pytest.raises(ValueError, match="Cannot revert"):
            s.revert_to_active()


# ===========================================================================
# Domain: ReceivingSession — frozen enforcement
# ===========================================================================


class TestReceivingSessionFrozen:
    def test_add_box_when_active(self):
        s = _make_session()
        s.add_box(PhysicalBox(barcode_value="AAA"))
        assert len(s.boxes) == 1

    def test_add_boxes_when_active(self):
        s = _make_session()
        s.add_boxes([PhysicalBox(barcode_value="AAA"), PhysicalBox(barcode_value="BBB")])
        assert len(s.boxes) == 2

    def test_add_box_when_frozen_raises(self):
        s = _make_session()
        s.freeze()
        with pytest.raises(ValueError, match="frozen"):
            s.add_box(PhysicalBox(barcode_value="AAA"))

    def test_add_boxes_when_frozen_raises(self):
        s = _make_session()
        s.freeze()
        with pytest.raises(ValueError, match="frozen"):
            s.add_boxes([PhysicalBox(barcode_value="AAA")])


# ===========================================================================
# Domain: ReceivingSession — aggregate quantities (multiset)
# ===========================================================================


class TestAggregateQuantities:
    def test_single_box(self):
        s = _make_session(boxes=[PhysicalBox(barcode_value="AAA")])
        assert s.aggregate_quantities() == [("AAA", "", 1)]

    def test_duplicate_values_count_separately(self):
        s = _make_session(
            boxes=[
                PhysicalBox(barcode_value="AAA"),
                PhysicalBox(barcode_value="AAA"),
                PhysicalBox(barcode_value="AAA"),
            ]
        )
        assert s.aggregate_quantities() == [("AAA", "", 3)]

    def test_mixed_values(self):
        s = _make_session(
            boxes=[
                PhysicalBox(barcode_value="BBB", barcode_format="Code128"),
                PhysicalBox(barcode_value="AAA"),
                PhysicalBox(barcode_value="BBB", barcode_format="Code128"),
                PhysicalBox(barcode_value="AAA"),
            ]
        )
        result = s.aggregate_quantities()
        assert result == [("AAA", "", 2), ("BBB", "Code128", 2)]

    def test_empty_session(self):
        s = _make_session()
        assert s.aggregate_quantities() == []

    def test_format_from_first_occurrence(self):
        s = _make_session(
            boxes=[
                PhysicalBox(barcode_value="AAA", barcode_format="Code128"),
                PhysicalBox(barcode_value="AAA", barcode_format="EAN13"),
            ]
        )
        result = s.aggregate_quantities()
        assert result == [("AAA", "Code128", 2)]


# ===========================================================================
# Domain: ReceivingSession — idempotency key
# ===========================================================================


class TestIdempotencyKey:
    def test_key_format(self):
        s = _make_session(session_id="abc-123")
        assert s.idempotency_key == "priority:draft:abc-123"

    def test_key_is_stable(self):
        s = _make_session(session_id="xyz")
        assert s.idempotency_key == s.idempotency_key


# ===========================================================================
# Domain: ReceivingSession — discrepancy
# ===========================================================================


class TestSessionDiscrepancy:
    def test_no_expected(self):
        s = _make_session(boxes=[PhysicalBox(barcode_value="A")])
        assert s.discrepancy.expected == 0
        assert s.discrepancy.found == 1
        assert not s.discrepancy.is_complete

    def test_complete(self):
        s = _make_session(boxes=[PhysicalBox(barcode_value="A")])
        s.expected_count = 1
        assert s.discrepancy.is_complete

    def test_missing(self):
        s = _make_session(boxes=[PhysicalBox(barcode_value="A")])
        s.expected_count = 3
        assert s.discrepancy.missing == 2


# ===========================================================================
# Router: POST /receiving/sessions — create session
# ===========================================================================


@skip_no_db
class TestCreateSession:
    async def test_create_session_success(self, async_client):
        with patch(
            "src.api.receiving._get_priority_gateway",
            return_value=_mock_gateway_success(),
        ):
            resp = await async_client.post(
                "/receiving/sessions",
                data={
                    "customer_id": "cust-acme",
                    "branch_id": "branch-acme-main",
                    "action": "create_order",
                },
            )
        assert resp.status_code == 201
        body = resp.json()
        assert body["status"] == "active"
        assert body["customer_id"] == "cust-acme"
        assert body["box_count"] == 0
        assert "session_id" in body

    async def test_missing_customer_id(self, async_client):
        resp = await async_client.post(
            "/receiving/sessions",
            data={
                "customer_id": "  ",
                "branch_id": "branch-acme-main",
                "action": "create_order",
            },
        )
        assert resp.status_code == 422
        assert resp.json()["detail"]["code"] == "order_context_required"

    async def test_invalid_action(self, async_client):
        with patch(
            "src.api.receiving._get_priority_gateway",
            return_value=_mock_gateway_success(),
        ):
            resp = await async_client.post(
                "/receiving/sessions",
                data={
                    "customer_id": "cust-acme",
                    "branch_id": "branch-acme-main",
                    "action": "invalid",
                },
            )
        assert resp.status_code == 422
        assert resp.json()["detail"]["code"] == "invalid_action"

    async def test_unknown_customer_rejected(self, async_client):
        with patch(
            "src.api.receiving._get_priority_gateway",
            return_value=_mock_gateway_success(),
        ):
            resp = await async_client.post(
                "/receiving/sessions",
                data={
                    "customer_id": "cust-nonexistent",
                    "branch_id": "branch-acme-main",
                    "action": "create_order",
                },
            )
        assert resp.status_code == 422
        assert resp.json()["detail"]["code"] == "unknown_customer"

    async def test_unknown_branch_rejected(self, async_client):
        with patch(
            "src.api.receiving._get_priority_gateway",
            return_value=_mock_gateway_success(),
        ):
            resp = await async_client.post(
                "/receiving/sessions",
                data={
                    "customer_id": "cust-acme",
                    "branch_id": "branch-nonexistent",
                    "action": "create_order",
                },
            )
        assert resp.status_code == 422
        assert resp.json()["detail"]["code"] == "unknown_or_mismatched_branch"

    async def test_open_session_blocks_new_for_same_participant(
        self, async_client, receiving_store
    ):
        gateway = _mock_gateway_success()
        with patch("src.api.receiving._get_priority_gateway", return_value=gateway):
            resp1 = await async_client.post(
                "/receiving/sessions",
                data={
                    "customer_id": "cust-acme",
                    "branch_id": "branch-acme-main",
                    "action": "create_order",
                    "participant_id": "operator-1",
                },
            )
            assert resp1.status_code == 201
            # Same participant — should get 409.
            resp2 = await async_client.post(
                "/receiving/sessions",
                data={
                    "customer_id": "cust-acme",
                    "branch_id": "branch-acme-main",
                    "action": "create_order",
                    "participant_id": "operator-1",
                },
            )
        assert resp2.status_code == 409
        assert resp2.json()["detail"]["code"] == "open_session_exists"
        assert resp2.json()["detail"]["existing_session_id"] == resp1.json()["session_id"]


# ===========================================================================
# Router: POST /receiving/sessions/{id}/images — upload image
# ===========================================================================


@skip_no_db
class TestUploadImage:
    async def test_session_not_found(self, async_client):
        resp = await async_client.post(
            "/receiving/sessions/nonexistent/images",
            files={"file": ("test.jpg", b"fake", "image/jpeg")},
        )
        assert resp.status_code == 404

    async def test_upload_when_submitting_rejects(
        self, async_client, receiving_store
    ):
        # Create a session, then freeze it.
        await _make_session_in_db(receiving_store, "sess-1")
        await receiving_store.freeze_submission("sess-1", {"items": []})

        resp = await async_client.post(
            "/receiving/sessions/sess-1/images",
            files={"file": ("test.jpg", b"fake", "image/jpeg")},
        )
        assert resp.status_code == 409
        assert resp.json()["detail"]["code"] == "session_not_editable"


# ===========================================================================
# Router: POST /receiving/sessions/{id}/submit — submit + idempotency
# ===========================================================================


@skip_no_db
class TestSubmitSession:
    async def test_session_not_found(self, async_client):
        resp = await async_client.post("/receiving/sessions/nonexistent/submit")
        assert resp.status_code == 404

    async def test_submit_empty_session_rejected(
        self, async_client, receiving_store
    ):
        await _make_session_in_db(receiving_store, "sess-1")

        with patch(
            "src.api.receiving._get_priority_gateway",
            return_value=_mock_gateway_success(),
        ):
            resp = await async_client.post("/receiving/sessions/sess-1/submit")
        assert resp.status_code == 422
        assert resp.json()["detail"]["code"] == "empty_order"

    async def test_submit_with_boxes(
        self, async_client, receiving_store
    ):
        await _make_session_in_db(
            receiving_store,
            "sess-1",
            boxes=[
                PhysicalBox(barcode_value="AAA", barcode_format="Code128"),
                PhysicalBox(barcode_value="AAA", barcode_format="Code128"),
                PhysicalBox(barcode_value="BBB", barcode_format="Code128"),
            ],
        )

        with patch(
            "src.api.receiving._get_priority_gateway",
            return_value=_mock_gateway_success(order_id=99),
        ):
            resp = await async_client.post("/receiving/sessions/sess-1/submit")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "submitted"
        assert body["order_id"] == 99
        assert body["items"] == [
            {"barcode_value": "AAA", "barcode_format": "Code128", "quantity": 2},
            {"barcode_value": "BBB", "barcode_format": "Code128", "quantity": 1},
        ]

    async def test_double_submit_returns_same_order_id(
        self, async_client, receiving_store
    ):
        await _make_session_in_db(
            receiving_store,
            "sess-1",
            boxes=[PhysicalBox(barcode_value="AAA")],
        )

        gateway = _mock_gateway_success(order_id=42)
        with patch("src.api.receiving._get_priority_gateway", return_value=gateway):
            resp1 = await async_client.post("/receiving/sessions/sess-1/submit")
            resp2 = await async_client.post("/receiving/sessions/sess-1/submit")
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        assert resp1.json()["order_id"] == 42
        assert resp2.json()["order_id"] == 42
        assert resp2.json()["idempotent"] is True
        # Gateway called only once (idempotency cache hit on second submit).
        assert gateway.create_draft_order.call_count == 1

    async def test_indeterminate_transitions_to_submission_unknown(
        self, async_client, receiving_store
    ):
        await _make_session_in_db(
            receiving_store,
            "sess-1",
            boxes=[PhysicalBox(barcode_value="AAA")],
        )

        with patch(
            "src.api.receiving._get_priority_gateway",
            return_value=_mock_gateway_indeterminate(),
        ):
            resp = await async_client.post("/receiving/sessions/sess-1/submit")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "submission_unknown"
        assert body["error"]["code"] == "submission_unknown"
        assert body["retry_recommended"] is True

    async def test_retry_after_indeterminate_replays_same_outcome(
        self, async_client, receiving_store
    ):
        await _make_session_in_db(
            receiving_store,
            "sess-1",
            boxes=[PhysicalBox(barcode_value="AAA")],
        )

        gateway = _mock_gateway_indeterminate()
        with patch("src.api.receiving._get_priority_gateway", return_value=gateway):
            resp1 = await async_client.post("/receiving/sessions/sess-1/submit")
            resp2 = await async_client.post("/receiving/sessions/sess-1/submit")
        assert resp1.status_code == 200
        assert resp1.json()["status"] == "submission_unknown"
        assert resp2.status_code == 200
        assert resp2.json()["status"] == "submission_unknown"
        # Gateway called once; second call replays the cached indeterminate.
        assert gateway.create_draft_order.call_count == 1

    async def test_new_session_after_indeterminate_is_rejected(
        self, async_client, receiving_store
    ):
        """After an indeterminate outcome, creating a NEW session for the
        same participant is rejected until the indeterminate session is
        resolved. The user must retry the existing session.

        Full external-reference reconciliation is deferred to MVP — see
        AGENTS.md "Known gaps".
        """
        await _make_session_in_db(
            receiving_store,
            "sess-1",
            boxes=[PhysicalBox(barcode_value="AAA")],
            participant_id="operator-1",
        )

        gateway = _mock_gateway_indeterminate()
        with patch("src.api.receiving._get_priority_gateway", return_value=gateway):
            # First submit: indeterminate.
            resp1 = await async_client.post("/receiving/sessions/sess-1/submit")
            assert resp1.json()["status"] == "submission_unknown"

            # Try to create a new session for the same participant: rejected.
            resp2 = await async_client.post(
                "/receiving/sessions",
                data={
                    "customer_id": "cust-acme",
                    "branch_id": "branch-acme-main",
                    "action": "create_order",
                    "participant_id": "operator-1",
                },
            )
        assert resp2.status_code == 409
        assert resp2.json()["detail"]["code"] == "open_session_exists"
        assert resp2.json()["detail"]["existing_status"] == "submission_unknown"

    async def test_retryable_error_reverts_to_active(
        self, async_client, receiving_store
    ):
        await _make_session_in_db(
            receiving_store,
            "sess-1",
            boxes=[PhysicalBox(barcode_value="AAA")],
        )

        with patch(
            "src.api.receiving._get_priority_gateway",
            return_value=_mock_gateway_retryable(),
        ):
            resp = await async_client.post("/receiving/sessions/sess-1/submit")
        assert resp.status_code == 502
        assert resp.json()["detail"]["code"] == "submission_failed"

        # Session should be back to ACTIVE.
        session = await receiving_store.get_receiving_session("sess-1")
        assert session is not None
        assert session.status == ReceivingSessionStatus.ACTIVE

    async def test_permanent_error_reverts_to_active(
        self, async_client, receiving_store
    ):
        await _make_session_in_db(
            receiving_store,
            "sess-1",
            boxes=[PhysicalBox(barcode_value="AAA")],
        )

        with patch(
            "src.api.receiving._get_priority_gateway",
            return_value=_mock_gateway_permanent(),
        ):
            resp = await async_client.post("/receiving/sessions/sess-1/submit")
        assert resp.status_code == 409
        assert resp.json()["detail"]["code"] == "submission_permanent_error"

        session = await receiving_store.get_receiving_session("sess-1")
        assert session is not None
        assert session.status == ReceivingSessionStatus.ACTIVE

    async def test_submit_from_submitted_returns_cached(
        self, async_client, receiving_store
    ):
        await _make_session_in_db(
            receiving_store,
            "sess-1",
            boxes=[PhysicalBox(barcode_value="AAA")],
        )
        await receiving_store.freeze_submission(
            "sess-1", {"items": [{"barcode_value": "AAA", "quantity": 1}]}
        )
        await receiving_store.mark_submitted("sess-1", 55)

        resp = await async_client.post("/receiving/sessions/sess-1/submit")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "submitted"
        assert body["order_id"] == 55
        assert body["idempotent"] is True


# ===========================================================================
# Router: GET /receiving/sessions/{id} — inspect
# ===========================================================================


@skip_no_db
class TestGetSession:
    async def test_get_active_session(
        self, async_client, receiving_store
    ):
        await _make_session_in_db(
            receiving_store,
            "sess-1",
            boxes=[PhysicalBox(barcode_value="AAA")],
        )
        # Update expected_count.
        async with receiving_store._pool.acquire() as conn:
            await conn.execute(
                "UPDATE sessions SET expected_count = 3 WHERE id = $1",
                "sess-1",
            )

        resp = await async_client.get("/receiving/sessions/sess-1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "active"
        assert body["box_count"] == 1
        assert body["expected_count"] == 3
        assert body["items"] == [
            {"barcode_value": "AAA", "barcode_format": "", "quantity": 1}
        ]
        assert body["discrepancy"]["missing"] == 2

    async def test_get_not_found(self, async_client):
        resp = await async_client.get("/receiving/sessions/nonexistent")
        assert resp.status_code == 404

    async def test_get_submitted_session(
        self, async_client, receiving_store
    ):
        await _make_session_in_db(
            receiving_store,
            "sess-1",
            boxes=[PhysicalBox(barcode_value="AAA")],
        )
        await receiving_store.freeze_submission(
            "sess-1", {"items": [{"barcode_value": "AAA", "quantity": 1}]}
        )
        await receiving_store.mark_submitted("sess-1", 42)

        resp = await async_client.get("/receiving/sessions/sess-1")
        body = resp.json()
        assert body["status"] == "submitted"
        assert body["external_order_id"] == 42
        assert body["frozen"] is True

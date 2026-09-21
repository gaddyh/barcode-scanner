"""Tests for the end-to-end receiving flow (PR #5 vertical slice).

Covers:
- Domain layer: PhysicalBox, ReceivingSession, Discrepancy, aggregate
  quantities (multiset), state transitions, frozen enforcement.
- Receiving router: create session, upload image, submit, double-submit
  idempotency, SUBMISSION_UNKNOWN, pre-submit failure, invalid state.
- Hard edge cases: empty session submit, duplicate barcode values count
  separately, frozen session rejects edits, indeterminate replay.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.receiving import _idempotency_store, _sessions
from src.domain.receiving import (
    Discrepancy,
    PhysicalBox,
    ReceivingSession,
    ReceivingSessionStatus,
)
from src.integrations.priority.models import (
    CreateDraftOrderResult,
)
from src.runtime.errors import IndeterminateError, PermanentError, RetryableError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_sessions():
    """Clear the in-memory session store between tests."""
    _sessions.clear()
    _idempotency_store._outcomes.clear()
    _idempotency_store._in_progress.clear()
    yield
    _sessions.clear()
    _idempotency_store._outcomes.clear()
    _idempotency_store._in_progress.clear()


@pytest.fixture
def client_no_raise():
    """TestClient that doesn't re-raise server exceptions (for 500 paths)."""
    from src.main import app

    return TestClient(app, raise_server_exceptions=False)


def _make_session(
    session_id: str = "sess-1",
    customer_id: str = "cust-acme",
    branch_id: str = "branch-acme-main",
    action: str = "create_order",
    boxes: list[PhysicalBox] | None = None,
    status: ReceivingSessionStatus = ReceivingSessionStatus.ACTIVE,
) -> ReceivingSession:
    session = ReceivingSession(
        session_id=session_id,
        customer_id=customer_id,
        branch_id=branch_id,
        action=action,
        boxes=list(boxes) if boxes else [],
        status=status,
    )
    _sessions[session_id] = session
    return session


def _mock_gateway_success(order_id: int = 42):
    """Mock PriorityGateway that succeeds."""
    gateway = MagicMock()
    gateway.create_draft_order = AsyncMock(
        return_value=CreateDraftOrderResult(
            order_id=order_id, session_id="sess-1", status="draft"
        )
    )
    return gateway


def _mock_gateway_indeterminate():
    """Mock PriorityGateway that raises IndeterminateError."""
    gateway = MagicMock()
    gateway.create_draft_order = AsyncMock(
        side_effect=IndeterminateError("connection lost after submit")
    )
    return gateway


def _mock_gateway_retryable():
    """Mock PriorityGateway that raises RetryableError (pre-submit)."""
    gateway = MagicMock()
    gateway.create_draft_order = AsyncMock(
        side_effect=RetryableError("connection refused")
    )
    return gateway


def _mock_gateway_permanent():
    """Mock PriorityGateway that raises PermanentError."""
    gateway = MagicMock()
    gateway.create_draft_order = AsyncMock(
        side_effect=PermanentError("duplicate session_id")
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


class TestCreateSession:
    def test_create_session_success(self, client_no_raise):
        resp = client_no_raise.post(
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

    def test_missing_customer_id(self, client_no_raise):
        resp = client_no_raise.post(
            "/receiving/sessions",
            data={
                "customer_id": "  ",
                "branch_id": "branch-acme-main",
                "action": "create_order",
            },
        )
        assert resp.status_code == 422
        assert resp.json()["detail"]["code"] == "order_context_required"

    def test_invalid_action(self, client_no_raise):
        resp = client_no_raise.post(
            "/receiving/sessions",
            data={
                "customer_id": "cust-acme",
                "branch_id": "branch-acme-main",
                "action": "invalid",
            },
        )
        assert resp.status_code == 422
        assert resp.json()["detail"]["code"] == "invalid_action"


# ===========================================================================
# Router: POST /receiving/sessions/{id}/images — upload image
# ===========================================================================


class TestUploadImage:
    def test_session_not_found(self, client_no_raise):
        resp = client_no_raise.post(
            "/receiving/sessions/nonexistent/images",
            files={"file": ("test.jpg", b"fake", "image/jpeg")},
        )
        assert resp.status_code == 404

    def test_upload_when_submitting_rejects(self, client_no_raise):
        _make_session(status=ReceivingSessionStatus.SUBMITTING)
        resp = client_no_raise.post(
            "/receiving/sessions/sess-1/images",
            files={"file": ("test.jpg", b"fake", "image/jpeg")},
        )
        assert resp.status_code == 409
        assert resp.json()["detail"]["code"] == "session_not_editable"


# ===========================================================================
# Router: POST /receiving/sessions/{id}/submit — submit + idempotency
# ===========================================================================


class TestSubmitSession:
    def test_session_not_found(self, client_no_raise):
        resp = client_no_raise.post("/receiving/sessions/nonexistent/submit")
        assert resp.status_code == 404

    def test_submit_empty_session(self, client_no_raise):
        _make_session()
        with patch(
            "src.api.receiving._get_priority_gateway",
            return_value=_mock_gateway_success(),
        ):
            resp = client_no_raise.post("/receiving/sessions/sess-1/submit")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "submitted"
        assert body["order_id"] == 42
        assert body["items"] == []

    def test_submit_with_boxes(self, client_no_raise):
        _make_session(
            boxes=[
                PhysicalBox(barcode_value="AAA", barcode_format="Code128"),
                PhysicalBox(barcode_value="AAA", barcode_format="Code128"),
                PhysicalBox(barcode_value="BBB", barcode_format="Code128"),
            ]
        )
        with patch(
            "src.api.receiving._get_priority_gateway",
            return_value=_mock_gateway_success(order_id=99),
        ):
            resp = client_no_raise.post("/receiving/sessions/sess-1/submit")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "submitted"
        assert body["order_id"] == 99
        assert body["items"] == [
            {"barcode_value": "AAA", "barcode_format": "Code128", "quantity": 2},
            {"barcode_value": "BBB", "barcode_format": "Code128", "quantity": 1},
        ]

    def test_double_submit_returns_same_order_id(self, client_no_raise):
        _make_session(boxes=[PhysicalBox(barcode_value="AAA")])
        gateway = _mock_gateway_success(order_id=42)
        with patch("src.api.receiving._get_priority_gateway", return_value=gateway):
            resp1 = client_no_raise.post("/receiving/sessions/sess-1/submit")
            resp2 = client_no_raise.post("/receiving/sessions/sess-1/submit")
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        assert resp1.json()["order_id"] == 42
        assert resp2.json()["order_id"] == 42
        assert resp2.json()["idempotent"] is True
        # Gateway called only once (idempotency cache hit on second submit).
        assert gateway.create_draft_order.call_count == 1

    def test_indeterminate_transitions_to_submission_unknown(self, client_no_raise):
        _make_session(boxes=[PhysicalBox(barcode_value="AAA")])
        with patch(
            "src.api.receiving._get_priority_gateway",
            return_value=_mock_gateway_indeterminate(),
        ):
            resp = client_no_raise.post("/receiving/sessions/sess-1/submit")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "submission_unknown"
        assert body["error"]["code"] == "submission_unknown"
        assert body["retry_recommended"] is True

    def test_retry_after_indeterminate_replays_same_outcome(
        self, client_no_raise
    ):
        _make_session(boxes=[PhysicalBox(barcode_value="AAA")])
        gateway = _mock_gateway_indeterminate()
        with patch("src.api.receiving._get_priority_gateway", return_value=gateway):
            resp1 = client_no_raise.post("/receiving/sessions/sess-1/submit")
            resp2 = client_no_raise.post("/receiving/sessions/sess-1/submit")
        assert resp1.status_code == 200
        assert resp1.json()["status"] == "submission_unknown"
        assert resp2.status_code == 200
        assert resp2.json()["status"] == "submission_unknown"
        # Gateway called once; second call replays the cached indeterminate.
        assert gateway.create_draft_order.call_count == 1

    def test_new_session_succeeds_after_indeterminate(self, client_no_raise):
        """After an indeterminate outcome, a NEW session can succeed.

        The indeterminate session is stuck in SUBMISSION_UNKNOWN — the
        idempotency store replays the indeterminate outcome on retry.
        To create a new order, the user must create a new session (or
        reconcile the indeterminate outcome, which is out of scope for
        PR #5).
        """
        # First session: indeterminate.
        _make_session(
            session_id="sess-1", boxes=[PhysicalBox(barcode_value="AAA")]
        )
        gateway = MagicMock()
        gateway.create_draft_order = AsyncMock(
            side_effect=[
                IndeterminateError("connection lost"),
                # Second call (new session) succeeds.
                CreateDraftOrderResult(
                    order_id=77, session_id="sess-2", status="draft"
                ),
            ]
        )
        with patch("src.api.receiving._get_priority_gateway", return_value=gateway):
            resp1 = client_no_raise.post("/receiving/sessions/sess-1/submit")
            # Create a new session and submit.
            _make_session(
                session_id="sess-2", boxes=[PhysicalBox(barcode_value="BBB")]
            )
            resp2 = client_no_raise.post("/receiving/sessions/sess-2/submit")
        assert resp1.json()["status"] == "submission_unknown"
        assert resp2.json()["status"] == "submitted"
        assert resp2.json()["order_id"] == 77

    def test_retryable_error_reverts_to_active(self, client_no_raise):
        _make_session(boxes=[PhysicalBox(barcode_value="AAA")])
        with patch(
            "src.api.receiving._get_priority_gateway",
            return_value=_mock_gateway_retryable(),
        ):
            resp = client_no_raise.post("/receiving/sessions/sess-1/submit")
        assert resp.status_code == 502
        assert resp.json()["detail"]["code"] == "submission_failed"
        # Session should be back to ACTIVE.
        assert _sessions["sess-1"].status == ReceivingSessionStatus.ACTIVE

    def test_permanent_error_reverts_to_active(self, client_no_raise):
        _make_session(boxes=[PhysicalBox(barcode_value="AAA")])
        with patch(
            "src.api.receiving._get_priority_gateway",
            return_value=_mock_gateway_permanent(),
        ):
            resp = client_no_raise.post("/receiving/sessions/sess-1/submit")
        assert resp.status_code == 409
        assert resp.json()["detail"]["code"] == "submission_permanent_error"
        assert _sessions["sess-1"].status == ReceivingSessionStatus.ACTIVE

    def test_submit_from_submitted_returns_cached(self, client_no_raise):
        s = _make_session(boxes=[PhysicalBox(barcode_value="AAA")])
        s.freeze()
        s.mark_submitted(55)
        resp = client_no_raise.post("/receiving/sessions/sess-1/submit")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "submitted"
        assert body["order_id"] == 55
        assert body["idempotent"] is True


# ===========================================================================
# Router: GET /receiving/sessions/{id} — inspect
# ===========================================================================


class TestGetSession:
    def test_get_active_session(self, client_no_raise):
        s = _make_session(boxes=[PhysicalBox(barcode_value="AAA")])
        s.expected_count = 3
        resp = client_no_raise.get("/receiving/sessions/sess-1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "active"
        assert body["box_count"] == 1
        assert body["expected_count"] == 3
        assert body["items"] == [{"barcode_value": "AAA", "barcode_format": "", "quantity": 1}]
        assert body["discrepancy"]["missing"] == 2

    def test_get_not_found(self, client_no_raise):
        resp = client_no_raise.get("/receiving/sessions/nonexistent")
        assert resp.status_code == 404

    def test_get_submitted_session(self, client_no_raise):
        s = _make_session(boxes=[PhysicalBox(barcode_value="AAA")])
        s.freeze()
        s.mark_submitted(42)
        resp = client_no_raise.get("/receiving/sessions/sess-1")
        body = resp.json()
        assert body["status"] == "submitted"
        assert body["external_order_id"] == 42
        assert body["frozen"] is True

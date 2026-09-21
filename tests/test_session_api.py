"""Tests for the session and tracing API endpoints."""

from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from src.ingest.session_graph import SessionResult
from src.ingest.session_models import SessionStatus


def _png_bytes(width: int = 100, height: int = 100) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# /barcode/session — validation
# ---------------------------------------------------------------------------


def test_session_invalid_action(client: pytest.fixture) -> None:
    response = client.post(
        "/barcode/session",
        files={"file": ("img.png", _png_bytes(), "image/png")},
        data={
            "participant_id": "p1",
            "customer_id": "C1",
            "branch_id": "B1",
            "action": "invalid_action",
        },
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_action"


def test_session_missing_order_context(client: pytest.fixture) -> None:
    response = client.post(
        "/barcode/session",
        files={"file": ("img.png", _png_bytes(), "image/png")},
        data={
            "participant_id": "p1",
            "customer_id": "",
            "branch_id": "B1",
            "action": "create_order",
        },
    )
    assert response.status_code == 422


def test_session_unsupported_type(client: pytest.fixture) -> None:
    response = client.post(
        "/barcode/session",
        files={"file": ("img.gif", b"GIF89a", "image/gif")},
        data={
            "participant_id": "p1",
            "customer_id": "C1",
            "branch_id": "B1",
            "action": "create_order",
        },
    )
    assert response.status_code == 415


# ---------------------------------------------------------------------------
# /barcode/session — happy path
# ---------------------------------------------------------------------------


def _make_session_result(
    status: SessionStatus = SessionStatus.COMPLETE,
    found_count: int = 2,
) -> SessionResult:
    return SessionResult(
        session_id="sess-1",
        status=status,
        expected_count=found_count,
        found_count=found_count,
        missing_count=0,
        image_count=1,
        items=[],
        missing=[],
    )


def test_session_complete_creates_order(
    monkeypatch: pytest.MonkeyPatch, client: pytest.fixture
) -> None:
    """A complete session triggers create_order on the Priority repo."""
    fake_repo = MagicMock()
    fake_repo.create_order = AsyncMock(return_value=1)
    monkeypatch.setattr("src.api.routes._get_priority_repo", lambda: fake_repo)
    monkeypatch.setattr(
        "src.api.routes._get_session_repo",
        lambda: MagicMock(),
    )

    session_result = _make_session_result(SessionStatus.COMPLETE)
    with patch(
        "src.ingest.session_graph.run_session_graph",
        new=AsyncMock(return_value=session_result),
    ):
        response = client.post(
            "/barcode/session",
            files={"file": ("img.png", _png_bytes(), "image/png")},
            data={
                "participant_id": "p1",
                "customer_id": "C1",
                "branch_id": "B1",
                "action": "create_order",
            },
        )
    assert response.status_code == 200
    assert response.json()["status"] == "complete"
    fake_repo.create_order.assert_awaited_once()


def test_session_complete_priority_error(
    monkeypatch: pytest.MonkeyPatch, client: pytest.fixture
) -> None:
    """A PriorityError during create_order returns 502."""
    from src.integrations.priority import PriorityError

    fake_repo = MagicMock()
    fake_repo.create_order = AsyncMock(side_effect=PriorityError("boom"))
    monkeypatch.setattr("src.api.routes._get_priority_repo", lambda: fake_repo)
    monkeypatch.setattr(
        "src.api.routes._get_session_repo",
        lambda: MagicMock(),
    )

    session_result = _make_session_result(SessionStatus.COMPLETE)
    with patch(
        "src.ingest.session_graph.run_session_graph",
        new=AsyncMock(return_value=session_result),
    ):
        response = client.post(
            "/barcode/session",
            files={"file": ("img.png", _png_bytes(), "image/png")},
            data={
                "participant_id": "p1",
                "customer_id": "C1",
                "branch_id": "B1",
                "action": "create_order",
            },
        )
    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "priority_order_failed"


def test_session_active_no_order(
    monkeypatch: pytest.MonkeyPatch, client: pytest.fixture
) -> None:
    """An active session does NOT trigger create_order."""
    fake_repo = MagicMock()
    fake_repo.create_order = AsyncMock(return_value=1)
    monkeypatch.setattr("src.api.routes._get_priority_repo", lambda: fake_repo)
    monkeypatch.setattr(
        "src.api.routes._get_session_repo",
        lambda: MagicMock(),
    )

    session_result = _make_session_result(SessionStatus.ACTIVE)
    with patch(
        "src.ingest.session_graph.run_session_graph",
        new=AsyncMock(return_value=session_result),
    ):
        response = client.post(
            "/barcode/session",
            files={"file": ("img.png", _png_bytes(), "image/png")},
            data={
                "participant_id": "p1",
                "customer_id": "C1",
                "branch_id": "B1",
                "action": "create_order",
            },
        )
    assert response.status_code == 200
    assert response.json()["status"] == "active"
    fake_repo.create_order.assert_not_called()


# ---------------------------------------------------------------------------
# /barcode/session/{id} — GET
# ---------------------------------------------------------------------------


def test_get_session_not_found(
    monkeypatch: pytest.MonkeyPatch, client: pytest.fixture
) -> None:
    fake_repo = MagicMock()
    fake_repo.to_result = AsyncMock(return_value=None)
    monkeypatch.setattr("src.api.routes._get_session_repo", lambda: fake_repo)

    response = client.get("/barcode/session/sess-1")
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "session_not_found"


def test_get_session_found(
    monkeypatch: pytest.MonkeyPatch, client: pytest.fixture
) -> None:
    session_result = _make_session_result(SessionStatus.ACTIVE)
    fake_repo = MagicMock()
    fake_repo.to_result = AsyncMock(return_value=session_result)
    monkeypatch.setattr("src.api.routes._get_session_repo", lambda: fake_repo)

    response = client.get("/barcode/session/sess-1")
    assert response.status_code == 200
    assert response.json()["session_id"] == "sess-1"


# ---------------------------------------------------------------------------
# /barcode/session/{id} — DELETE
# ---------------------------------------------------------------------------


def test_close_session_not_found(
    monkeypatch: pytest.MonkeyPatch, client: pytest.fixture
) -> None:
    fake_repo = MagicMock()
    fake_repo.to_result = AsyncMock(return_value=None)
    monkeypatch.setattr("src.api.routes._get_session_repo", lambda: fake_repo)

    response = client.delete("/barcode/session/sess-1")
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "session_not_found"


def test_close_session_success(
    monkeypatch: pytest.MonkeyPatch, client: pytest.fixture
) -> None:
    """Closing a session returns the final state."""
    session_result = _make_session_result(SessionStatus.CLOSED)
    fake_repo = MagicMock()
    # First to_result (existence check) returns the session,
    # second to_result (after close) returns the closed session.
    fake_repo.to_result = AsyncMock(return_value=session_result)
    fake_repo.close_session = AsyncMock(return_value=None)
    monkeypatch.setattr("src.api.routes._get_session_repo", lambda: fake_repo)

    response = client.delete("/barcode/session/sess-1")
    assert response.status_code == 200
    assert response.json()["status"] == "closed"
    fake_repo.close_session.assert_awaited_once()


def test_close_session_returns_closed_dict(
    monkeypatch: pytest.MonkeyPatch, client: pytest.fixture
) -> None:
    """If to_result returns None after close, return a minimal closed dict."""
    fake_repo = MagicMock()
    # First call returns a session, second returns None (session gone).
    fake_repo.to_result = AsyncMock(side_effect=[
        _make_session_result(SessionStatus.ACTIVE),
        None,
    ])
    fake_repo.close_session = AsyncMock(return_value=None)
    monkeypatch.setattr("src.api.routes._get_session_repo", lambda: fake_repo)

    response = client.delete("/barcode/session/sess-1")
    assert response.status_code == 200
    assert response.json()["session_id"] == "sess-1"
    assert response.json()["status"] == "closed"


# ---------------------------------------------------------------------------
# /barcode/session/select
# ---------------------------------------------------------------------------


def test_select_no_session_pending(
    monkeypatch: pytest.MonkeyPatch, client: pytest.fixture
) -> None:
    fake_repo = MagicMock()
    fake_repo.find_active_by_participant = AsyncMock(return_value=None)
    monkeypatch.setattr("src.api.routes._get_session_repo", lambda: fake_repo)

    response = client.post(
        "/barcode/session/select",
        data={"participant_id": "p1", "barcode_value": "VAL1"},
    )
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "no_selection_pending"


def test_select_wrong_status(
    monkeypatch: pytest.MonkeyPatch, client: pytest.fixture
) -> None:
    fake_repo = MagicMock()
    fake_repo.find_active_by_participant = AsyncMock(
        return_value={"id": "sess-1", "status": "active"}
    )
    monkeypatch.setattr("src.api.routes._get_session_repo", lambda: fake_repo)

    response = client.post(
        "/barcode/session/select",
        data={"participant_id": "p1", "barcode_value": "VAL1"},
    )
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "no_selection_pending"


def test_select_success(
    monkeypatch: pytest.MonkeyPatch, client: pytest.fixture
) -> None:
    fake_repo = MagicMock()
    fake_repo.find_active_by_participant = AsyncMock(
        return_value={"id": "sess-1", "status": "needs_user_selection"}
    )
    monkeypatch.setattr("src.api.routes._get_session_repo", lambda: fake_repo)

    session_result = _make_session_result(SessionStatus.COMPLETE)
    with patch(
        "src.ingest.session_graph.select_candidate",
        new=AsyncMock(return_value=session_result),
    ):
        response = client.post(
            "/barcode/session/select",
            data={"participant_id": "p1", "barcode_value": "VAL1"},
        )
    assert response.status_code == 200
    assert response.json()["status"] == "complete"


def test_select_value_error(
    monkeypatch: pytest.MonkeyPatch, client: pytest.fixture
) -> None:
    fake_repo = MagicMock()
    fake_repo.find_active_by_participant = AsyncMock(
        return_value={"id": "sess-1", "status": "needs_user_selection"}
    )
    monkeypatch.setattr("src.api.routes._get_session_repo", lambda: fake_repo)

    with patch(
        "src.ingest.session_graph.select_candidate",
        new=AsyncMock(side_effect=ValueError("bad barcode")),
    ):
        response = client.post(
            "/barcode/session/select",
            data={"participant_id": "p1", "barcode_value": "BAD"},
        )
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "selection_error"


# ---------------------------------------------------------------------------
# /customers — error path
# ---------------------------------------------------------------------------


def test_customers_priority_error(
    monkeypatch: pytest.MonkeyPatch, client: pytest.fixture
) -> None:
    from src.integrations.priority import PriorityError

    fake_repo = MagicMock()
    fake_repo.customers = AsyncMock(side_effect=PriorityError("down"))
    monkeypatch.setattr("src.api.routes._get_priority_repo", lambda: fake_repo)

    response = client.get("/customers")
    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "priority_unavailable"


def test_branches_priority_error(
    monkeypatch: pytest.MonkeyPatch, client: pytest.fixture
) -> None:
    from src.integrations.priority import PriorityError

    fake_repo = MagicMock()
    fake_repo.branches = AsyncMock(side_effect=PriorityError("down"))
    monkeypatch.setattr("src.api.routes._get_priority_repo", lambda: fake_repo)

    response = client.get("/customers/C1/branches")
    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "priority_unavailable"


def test_branches_empty_customer_id(
    monkeypatch: pytest.MonkeyPatch, client: pytest.fixture
) -> None:
    response = client.get("/customers/%20/branches")
    assert response.status_code == 400

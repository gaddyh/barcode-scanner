"""Extended tests for src/api/routes.py to raise coverage above 95%.

Covers endpoints and code paths not already tested in test_api.py and
test_session_api.py: /feedback, /barcode/scan (error/edge paths),
/barcode/analyze (all outcomes), /barcode/session (size/action paths),
/customers (empty results), helper functions, and session candidates.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import zxingcpp
from PIL import Image, UnidentifiedImageError

from src.api.routes import _attach_image_to_run, _sanitize_scan_inputs
from src.config import get_settings
from src.config import settings as _settings
from src.ingest.scanner import BarcodeScanner
from src.ingest.session_graph import SessionResult
from src.ingest.session_models import SessionItem, SessionStatus
from src.main import app
from tests._zxing_fake import make_read_result

_VALID_UUID = "550e8400-e29b-41d4-a716-446655440000"


def _png_bytes(width: int = 100, height: int = 100) -> bytes:
    """Generate a small PNG image for upload tests."""
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


def _make_session_result(
    status: SessionStatus = SessionStatus.COMPLETE,
    found_count: int = 2,
) -> SessionResult:
    """Build a SessionResult for session tests."""
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


def _analyze_result(outcome: str = "complete") -> dict:
    """Build a mock analyze_image_async result dict."""
    base: dict = {
        "outcome": outcome,
        "ok": True,
        "found": [{"barcode_value": "123", "label_index": 0}],
        "unassigned": [],
        "missing": [],
        "summary": {
            "found_count": 1,
            "missing_count": 0,
            "unassigned_count": 0,
            "all_found": True,
            "visible_label_count": 1,
            "recovery": {
                "attempted": False,
                "labels_tried": 0,
                "barcodes_found": 0,
                "labels_resolved": 0,
            },
        },
        "image_width": 100,
        "image_height": 100,
        "audit_available": True,
    }
    if outcome == "needs_better_photo":
        base["found"] = []
        base["missing"] = [{"label_index": 0}]
        base["summary"]["found_count"] = 0
        base["summary"]["missing_count"] = 1
        base["summary"]["all_found"] = False
        base["annotated_image_b64"] = "base64data"
        base["message"] = "Please send a better photo"
    elif outcome == "retryable_error":
        base["found"] = []
        base["summary"]["found_count"] = 0
        base["summary"]["all_found"] = False
        base["audit_available"] = False
    return base


def _mock_run_repo() -> MagicMock:
    """Build a mock RunRepository with async no-op methods."""
    repo = MagicMock()
    repo.create_run = AsyncMock(return_value=None)
    repo.complete_run = AsyncMock(return_value=None)
    repo.fail_run = AsyncMock(return_value=None)
    return repo


@pytest.fixture
def small_max_upload() -> Iterator[None]:
    """Override get_settings with a small max_upload_bytes (100 bytes)."""
    small = replace(_settings, max_upload_bytes=100)
    app.dependency_overrides[get_settings] = lambda: small
    yield
    app.dependency_overrides.pop(get_settings, None)


# ---------------------------------------------------------------------------
# POST /feedback
# ---------------------------------------------------------------------------


def test_feedback_happy_path(
    client: pytest.fixture
) -> None:
    """Valid feedback returns 200 with score."""
    with patch("src.api.routes.submit_upload_feedback", return_value=1):
        response = client.post("/feedback", json={
            "trace_id": _VALID_UUID,
            "correct": True,
        })
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "recorded"
    assert body["score"] == 1
    assert body["trace_id"] == _VALID_UUID


def test_feedback_incorrect_returns_zero(
    client: pytest.fixture
) -> None:
    """Feedback marked incorrect returns score 0."""
    with patch("src.api.routes.submit_upload_feedback", return_value=0):
        response = client.post("/feedback", json={
            "trace_id": _VALID_UUID,
            "correct": False,
        })
    assert response.status_code == 200
    assert response.json()["score"] == 0


def test_feedback_with_comment(
    client: pytest.fixture
) -> None:
    """Feedback with a comment is accepted."""
    with patch("src.api.routes.submit_upload_feedback", return_value=1):
        response = client.post("/feedback", json={
            "trace_id": _VALID_UUID,
            "correct": True,
            "comment": "Looks good",
        })
    assert response.status_code == 200


def test_feedback_langsmith_error(
    client: pytest.fixture
) -> None:
    """LangSmithError returns 502."""
    from langsmith.utils import LangSmithError

    with patch(
        "src.api.routes.submit_upload_feedback",
        side_effect=LangSmithError("LangSmith down"),
    ):
        response = client.post("/feedback", json={
            "trace_id": _VALID_UUID,
            "correct": True,
        })
    assert response.status_code == 502
    assert "Could not submit feedback" in response.json()["detail"]


def test_feedback_missing_correct(client: pytest.fixture) -> None:
    """Missing 'correct' field returns 422."""
    response = client.post("/feedback", json={"trace_id": _VALID_UUID})
    assert response.status_code == 422


def test_feedback_missing_trace_id(client: pytest.fixture) -> None:
    """Missing 'trace_id' field returns 422."""
    response = client.post("/feedback", json={"correct": True})
    assert response.status_code == 422


def test_feedback_invalid_uuid(client: pytest.fixture) -> None:
    """Invalid UUID for trace_id returns 422."""
    response = client.post("/feedback", json={
        "trace_id": "not-a-uuid",
        "correct": True,
    })
    assert response.status_code == 422


def test_feedback_comment_too_long(client: pytest.fixture) -> None:
    """Comment exceeding 2000 chars returns 422."""
    response = client.post("/feedback", json={
        "trace_id": _VALID_UUID,
        "correct": True,
        "comment": "x" * 2001,
    })
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /barcode/scan — extended paths
# ---------------------------------------------------------------------------


def test_scan_too_large(
    client: pytest.fixture, small_max_upload: None
) -> None:
    """Upload exceeding max_upload_bytes returns 413."""
    response = client.post(
        "/barcode/scan",
        files={"file": ("img.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "image_too_large"


def test_scan_invalid_image_unidentified(
    client: pytest.fixture,
    override_scanner: pytest.fixture,
) -> None:
    """UnidentifiedImageError from scanner returns 422."""
    mock_scanner = MagicMock()
    mock_scanner.scan_bytes = MagicMock(
        side_effect=UnidentifiedImageError("bad image")
    )
    override_scanner["scanner"] = mock_scanner

    response = client.post(
        "/barcode/scan",
        files={"file": ("img.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_image"


def test_scan_invalid_image_oserror(
    client: pytest.fixture,
    override_scanner: pytest.fixture,
) -> None:
    """OSError from scanner returns 422."""
    mock_scanner = MagicMock()
    mock_scanner.scan_bytes = MagicMock(side_effect=OSError("disk error"))
    override_scanner["scanner"] = mock_scanner

    response = client.post(
        "/barcode/scan",
        files={"file": ("img.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 422


def test_scan_invalid_image_value_error(
    client: pytest.fixture,
    override_scanner: pytest.fixture,
) -> None:
    """ValueError from scanner returns 422."""
    mock_scanner = MagicMock()
    mock_scanner.scan_bytes = MagicMock(side_effect=ValueError("bad value"))
    override_scanner["scanner"] = mock_scanner

    response = client.post(
        "/barcode/scan",
        files={"file": ("img.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 422


def test_scan_persist_create_run_failure(
    client_no_raise: pytest.fixture,
    override_scanner: pytest.fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failure during persist_create_run propagates as 500."""
    mock_repo = MagicMock()
    mock_repo.create_run = AsyncMock(side_effect=RuntimeError("DB down"))
    monkeypatch.setattr("src.api.routes._get_run_repo", lambda: mock_repo)

    override_scanner["scanner"] = BarcodeScanner()
    monkeypatch.setattr(zxingcpp, "read_barcodes", lambda _img, **kw: [])

    response = client_no_raise.post(
        "/barcode/scan",
        files={"file": ("img.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 500


def test_scan_complete_run_failure_swallowed(
    client: pytest.fixture,
    override_scanner: pytest.fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failure during repo.complete_run is swallowed; response is 200."""
    mock_repo = _mock_run_repo()
    mock_repo.complete_run = AsyncMock(side_effect=RuntimeError("DB down"))
    monkeypatch.setattr("src.api.routes._get_run_repo", lambda: mock_repo)

    override_scanner["scanner"] = BarcodeScanner()
    monkeypatch.setattr(
        zxingcpp, "read_barcodes",
        lambda _img, **kw: [make_read_result("1234567890123")],
    )

    response = client.post(
        "/barcode/scan",
        files={"file": ("img.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "found"


def test_scan_response_fields(
    client_no_raise: pytest.fixture,
    override_scanner: pytest.fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify all ScanResponse fields are populated correctly."""
    calls = {"n": 0}

    def _first_only(_img, **kw):
        calls["n"] += 1
        return [make_read_result("1234567890123")] if calls["n"] == 1 else []

    monkeypatch.setattr(zxingcpp, "read_barcodes", _first_only)
    override_scanner["scanner"] = BarcodeScanner()

    png = _png_bytes(200, 150)
    response = client_no_raise.post(
        "/barcode/scan",
        files={"file": ("photo.png", png, "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "found"
    assert body["count"] == 1
    assert body["source"] == "web"
    assert body["image_width"] == 200
    assert body["image_height"] == 150
    assert body["filename"] == "photo.png"
    assert body["upload_bytes"] == len(png)
    assert body["elapsed_ms"] >= 0
    assert len(body["upload_id"]) > 0
    assert len(body["trace_id"]) > 0
    assert body["barcodes"][0]["value"] == "1234567890123"


# ---------------------------------------------------------------------------
# POST /barcode/analyze
# ---------------------------------------------------------------------------


def test_analyze_unsupported_type(client: pytest.fixture) -> None:
    """Unsupported content type returns 415."""
    response = client.post(
        "/barcode/analyze",
        files={"file": ("img.gif", b"GIF89a", "image/gif")},
    )
    assert response.status_code == 415


def test_analyze_too_large(
    client: pytest.fixture, small_max_upload: None
) -> None:
    """Upload exceeding max_upload_bytes returns 413."""
    response = client.post(
        "/barcode/analyze",
        files={"file": ("img.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "image_too_large"


def _patch_analyze_common(
    monkeypatch: pytest.MonkeyPatch,
    result: dict | None = None,
    side_effect: Exception | None = None,
) -> MagicMock:
    """Patch common deps for /barcode/analyze tests. Returns mock repo."""
    mock_repo = _mock_run_repo()
    monkeypatch.setattr("src.api.routes._get_run_repo", lambda: mock_repo)
    if side_effect is not None:
        monkeypatch.setattr(
            "src.api.routes.analyze_image_async",
            AsyncMock(side_effect=side_effect),
        )
    else:
        monkeypatch.setattr(
            "src.api.routes.analyze_image_async",
            AsyncMock(return_value=result),
        )
    mock_versions = MagicMock()
    mock_versions.model_dump.return_value = {"v": "1"}
    monkeypatch.setattr(
        "src.api.routes.collect_versions", lambda: mock_versions
    )
    return mock_repo


def test_analyze_complete(
    client: pytest.fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Complete outcome returns 200 with result."""
    _patch_analyze_common(monkeypatch, result=_analyze_result("complete"))

    response = client.post(
        "/barcode/analyze",
        files={"file": ("img.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["outcome"] == "complete"
    assert "upload_id" in body
    assert "trace_id" in body
    assert body["source"] == "web"
    assert body["filename"] == "img.png"


def test_analyze_needs_better_photo(
    client: pytest.fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """needs_better_photo outcome returns 200 with annotated image."""
    _patch_analyze_common(
        monkeypatch, result=_analyze_result("needs_better_photo")
    )

    response = client.post(
        "/barcode/analyze",
        files={"file": ("img.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["outcome"] == "needs_better_photo"
    assert body["annotated_image_b64"] == "base64data"
    assert body["message"] == "Please send a better photo"


def test_analyze_retryable_error(
    client: pytest.fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retryable_error outcome returns 200 with result dict."""
    _patch_analyze_common(
        monkeypatch, result=_analyze_result("retryable_error")
    )

    response = client.post(
        "/barcode/analyze",
        files={"file": ("img.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["outcome"] == "retryable_error"


def test_analyze_exception(
    client_no_raise: pytest.fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exception from analyze_image_async returns 500."""
    mock_repo = _patch_analyze_common(
        monkeypatch, side_effect=RuntimeError("gemini down")
    )

    response = client_no_raise.post(
        "/barcode/analyze",
        files={"file": ("img.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 500
    mock_repo.fail_run.assert_awaited_once()


def test_analyze_complete_run_failure_swallowed(
    client: pytest.fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failure during complete_run_from_dict is swallowed; returns 200."""
    _patch_analyze_common(monkeypatch, result=_analyze_result("complete"))

    with patch(
        "src.api.routes.complete_run_from_dict",
        new=AsyncMock(side_effect=RuntimeError("DB down")),
    ):
        response = client.post(
            "/barcode/analyze",
            files={"file": ("img.png", _png_bytes(), "image/png")},
        )
    assert response.status_code == 200
    assert response.json()["outcome"] == "complete"


def test_analyze_persist_create_run_failure(
    client_no_raise: pytest.fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failure during persist_create_run propagates as 500."""
    mock_repo = MagicMock()
    mock_repo.create_run = AsyncMock(side_effect=RuntimeError("DB down"))
    monkeypatch.setattr("src.api.routes._get_run_repo", lambda: mock_repo)
    mock_versions = MagicMock()
    mock_versions.model_dump.return_value = {"v": "1"}
    monkeypatch.setattr(
        "src.api.routes.collect_versions", lambda: mock_versions
    )

    response = client_no_raise.post(
        "/barcode/analyze",
        files={"file": ("img.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 500


# ---------------------------------------------------------------------------
# POST /barcode/session — extended paths
# ---------------------------------------------------------------------------


def test_session_too_large(
    client: pytest.fixture, small_max_upload: None
) -> None:
    """Session upload exceeding max_upload_bytes returns 413."""
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
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "image_too_large"


def test_session_verify_action_complete(
    client: pytest.fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """verify_order_before_shipment action does NOT auto-create an order.

    Order creation is now handled exclusively via
    POST /receiving/sessions/{id}/submit. The /barcode/session flow returns
    the session result; the frontend creates the order via /receiving/submit.
    """
    fake_priority = MagicMock()
    fake_priority.create_order = AsyncMock(return_value=1)
    monkeypatch.setattr(
        "src.api.routes._get_priority_repo", lambda: fake_priority
    )
    monkeypatch.setattr(
        "src.api.routes._get_session_repo", lambda: MagicMock()
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
                "action": "verify_order_before_shipment",
            },
        )
    assert response.status_code == 200
    assert response.json()["status"] == "complete"
    # Order creation is NOT called — it's handled via /receiving/submit.
    fake_priority.create_order.assert_not_awaited()


def test_session_missing_branch_id(
    client: pytest.fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whitespace-only branch_id returns 422 order_context_required."""
    response = client.post(
        "/barcode/session",
        files={"file": ("img.png", _png_bytes(), "image/png")},
        data={
            "participant_id": "p1",
            "customer_id": "C1",
            "branch_id": "   ",
            "action": "create_order",
        },
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "order_context_required"


# ---------------------------------------------------------------------------
# GET /customers — extended
# ---------------------------------------------------------------------------


def test_customers_empty_list(
    client: pytest.fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty customer list returns 200 with empty items."""
    fake_repo = MagicMock()
    fake_repo.customers = AsyncMock(return_value=[])
    monkeypatch.setattr(
        "src.api.routes._get_priority_repo", lambda: fake_repo
    )
    response = client.get("/customers")
    assert response.status_code == 200
    assert response.json() == {"items": []}


def test_branches_empty_list(
    client: pytest.fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty branch list returns 200 with empty items."""
    fake_repo = MagicMock()
    fake_repo.branches = AsyncMock(return_value=[])
    monkeypatch.setattr(
        "src.api.routes._get_priority_repo", lambda: fake_repo
    )
    response = client.get("/customers/C1/branches")
    assert response.status_code == 200
    assert response.json() == {"items": []}


def test_branches_whitespace_customer_id(
    client: pytest.fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whitespace-only customer_id returns 400."""
    response = client.get("/customers/%20%20/branches")
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# GET /barcode/session/{id} — with candidates
# ---------------------------------------------------------------------------


def test_get_session_with_candidates(
    client: pytest.fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Session in needs_user_selection includes candidates."""
    session_result = SessionResult(
        session_id="sess-1",
        status=SessionStatus.NEEDS_USER_SELECTION,
        expected_count=2,
        found_count=1,
        missing_count=1,
        candidates=[
            SessionItem(barcode_value="CAND1"),
            SessionItem(barcode_value="CAND2"),
        ],
    )
    fake_repo = MagicMock()
    fake_repo.to_result = AsyncMock(return_value=session_result)
    monkeypatch.setattr(
        "src.api.routes._get_session_repo", lambda: fake_repo
    )

    response = client.get("/barcode/session/sess-1")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "needs_user_selection"
    assert len(body["candidates"]) == 2
    assert body["candidates"][0]["barcode_value"] == "CAND1"
    assert body["candidates"][1]["barcode_value"] == "CAND2"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def test_sanitize_scan_inputs_with_file() -> None:
    """_sanitize_scan_inputs extracts filename and content_type."""
    file = MagicMock()
    file.filename = "photo.png"
    file.content_type = "image/png"
    result = _sanitize_scan_inputs({"file": file})
    assert result == {"filename": "photo.png", "content_type": "image/png"}


def test_sanitize_scan_inputs_no_file() -> None:
    """_sanitize_scan_inputs returns None when no file key."""
    result = _sanitize_scan_inputs({})
    assert result == {"filename": None, "content_type": None}


def test_sanitize_scan_inputs_none_file() -> None:
    """_sanitize_scan_inputs handles None file value."""
    result = _sanitize_scan_inputs({"file": None})
    assert result == {"filename": None, "content_type": None}


def test_attach_image_to_run_with_run() -> None:
    """_attach_image_to_run sets attachments when a run is active."""
    with patch("src.api.routes.ls.get_current_run_tree") as mock_get:
        run = MagicMock()
        mock_get.return_value = run
        _attach_image_to_run(b"img-bytes", "image/png")
        assert "uploaded_image" in run.attachments
        assert run.attachments["uploaded_image"].mime_type == "image/png"
        assert run.attachments["uploaded_image"].data == b"img-bytes"


def test_attach_image_to_run_no_run() -> None:
    """_attach_image_to_run does nothing when no run is active."""
    with patch("src.api.routes.ls.get_current_run_tree") as mock_get:
        mock_get.return_value = None
        _attach_image_to_run(b"img-bytes", "image/png")
        # Should not raise


# ---------------------------------------------------------------------------
# POST /barcode/session/select — missing fields
# ---------------------------------------------------------------------------


def test_select_missing_participant_id(client: pytest.fixture) -> None:
    """Missing participant_id returns 422."""
    response = client.post(
        "/barcode/session/select",
        data={"barcode_value": "VAL1"},
    )
    assert response.status_code == 422


def test_select_missing_barcode_value(client: pytest.fixture) -> None:
    """Missing barcode_value returns 422."""
    response = client.post(
        "/barcode/session/select",
        data={"participant_id": "p1"},
    )
    assert response.status_code == 422

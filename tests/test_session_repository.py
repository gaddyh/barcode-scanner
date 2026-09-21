"""Tests for src/session_repository.py — session, items, and missing repos."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import asyncpg
import pytest

from src.ingest.session_models import (
    MissingItem,
    SessionItem,
    SessionStatus,
)
from src.session_repository import NoOpSessionRepository, SessionRepository

# ---------------------------------------------------------------------------
# Fake asyncpg pool / connection helpers
# ---------------------------------------------------------------------------


class FakeAcquire:
    """Async context manager that yields ``conn``."""

    def __init__(self, conn: MagicMock) -> None:
        self._conn = conn

    async def __aenter__(self) -> MagicMock:
        return self._conn

    async def __aexit__(self, *exc: object) -> bool:
        return False


def _make_pool(
    *,
    execute_return: str = "INSERT 0 1",
    fetch_rows: list[dict] | None = None,
    fetchrow: dict | None = None,
) -> tuple[MagicMock, MagicMock]:
    """Build a fake asyncpg pool and its underlying connection."""
    conn = MagicMock()
    conn.execute = AsyncMock(return_value=execute_return)
    conn.fetch = AsyncMock(return_value=fetch_rows or [])
    conn.fetchrow = AsyncMock(return_value=fetchrow)

    pool = MagicMock()
    pool.acquire = MagicMock(return_value=FakeAcquire(conn))
    return pool, conn


def _make_pool_raising(exc: Exception) -> tuple[MagicMock, MagicMock]:
    """Build a fake pool whose conn methods raise ``exc``."""
    conn = MagicMock()
    conn.execute = AsyncMock(side_effect=exc)
    conn.fetch = AsyncMock(side_effect=exc)
    conn.fetchrow = AsyncMock(side_effect=exc)

    pool = MagicMock()
    pool.acquire = MagicMock(return_value=FakeAcquire(conn))
    return pool, conn


def _make_session_item(
    *,
    barcode_value: str = "BC1",
    barcode_format: str | None = "EAN_13",
    barcode_bbox: dict | None = None,
    label_bbox: dict | None = None,
    label_index: int | None = 0,
    match_basis: str | None = "containment",
    source_image: int = 0,
) -> SessionItem:
    return SessionItem(
        barcode_value=barcode_value,
        barcode_format=barcode_format,
        barcode_bbox=barcode_bbox,
        label_bbox=label_bbox,
        label_index=label_index,
        match_basis=match_basis,
        source_image=source_image,
    )


def _make_missing_item(
    *,
    label_index: int | None = 0,
    label_bbox: dict | None = None,
    barcode_bbox: dict | None = None,
    status: str = "not_visible",
    source_image: int = 0,
    resolved: bool = False,
) -> MissingItem:
    return MissingItem(
        label_index=label_index,
        label_bbox=label_bbox,
        barcode_bbox=barcode_bbox,
        status=status,
        source_image=source_image,
        resolved=resolved,
    )


def _make_session_row(
    *,
    sid: str = "sess-1",
    status: str = "active",
    expected_count: int = 10,
    found_count: int = 5,
    missing_count: int = 5,
    image_count: int = 1,
    message: str | None = None,
    customer_id: str | None = "C1",
    branch_id: str | None = "B1",
    action: str | None = "create_order",
    channel: str | None = "whatsapp",
    participant_id: str | None = "user-1",
) -> dict:
    return {
        "id": sid,
        "status": status,
        "expected_count": expected_count,
        "found_count": found_count,
        "missing_count": missing_count,
        "image_count": image_count,
        "message": message,
        "customer_id": customer_id,
        "branch_id": branch_id,
        "action": action,
        "channel": channel,
        "participant_id": participant_id,
    }


# ===========================================================================
# SessionRepository (Postgres)
# ===========================================================================


# ---------------------------------------------------------------------------
# create_session
# ---------------------------------------------------------------------------


async def test_pg_create_session():
    pool, conn = _make_pool()
    repo = SessionRepository(pool)
    await repo.create_session(
        "sess-1", source="web", channel="whatsapp",
        participant_id="user-1", customer_id="C1",
        branch_id="B1", action="create_order",
    )
    conn.execute.assert_called_once()
    args = conn.execute.call_args.args
    assert args[1] == "sess-1"
    assert args[2] == "web"
    assert args[3] == "whatsapp"
    assert args[4] == "user-1"
    assert args[5] == "C1"
    assert args[6] == "B1"
    assert args[7] == "create_order"


async def test_pg_create_session_minimal():
    pool, conn = _make_pool()
    repo = SessionRepository(pool)
    await repo.create_session("sess-1")
    conn.execute.assert_called_once()
    args = conn.execute.call_args.args
    assert args[1] == "sess-1"
    assert args[2] is None


async def test_pg_create_session_error_propagates():
    pool, _ = _make_pool_raising(asyncpg.PostgresError("boom"))
    repo = SessionRepository(pool)
    with pytest.raises(asyncpg.PostgresError, match="boom"):
        await repo.create_session("sess-1")


# ---------------------------------------------------------------------------
# get_session
# ---------------------------------------------------------------------------


async def test_pg_get_session_found():
    row = _make_session_row()
    pool, _ = _make_pool(fetchrow=row)
    repo = SessionRepository(pool)
    result = await repo.get_session("sess-1")
    assert result is not None
    assert result["id"] == "sess-1"
    assert result["status"] == "active"


async def test_pg_get_session_not_found():
    pool, _ = _make_pool(fetchrow=None)
    repo = SessionRepository(pool)
    result = await repo.get_session("missing")
    assert result is None


async def test_pg_get_session_error_propagates():
    pool, _ = _make_pool_raising(OSError("conn refused"))
    repo = SessionRepository(pool)
    with pytest.raises(OSError, match="conn refused"):
        await repo.get_session("sess-1")


# ---------------------------------------------------------------------------
# find_active_by_participant
# ---------------------------------------------------------------------------


async def test_pg_find_active_by_participant_found():
    row = _make_session_row()
    pool, _ = _make_pool(fetchrow=row)
    repo = SessionRepository(pool)
    result = await repo.find_active_by_participant("whatsapp", "user-1")
    assert result is not None
    assert result["id"] == "sess-1"


async def test_pg_find_active_by_participant_not_found():
    pool, _ = _make_pool(fetchrow=None)
    repo = SessionRepository(pool)
    result = await repo.find_active_by_participant("whatsapp", "nobody")
    assert result is None


async def test_pg_find_active_by_participant_error_propagates():
    pool, _ = _make_pool_raising(asyncpg.PostgresError("boom"))
    repo = SessionRepository(pool)
    with pytest.raises(asyncpg.PostgresError, match="boom"):
        await repo.find_active_by_participant("whatsapp", "user-1")


# ---------------------------------------------------------------------------
# update_session
# ---------------------------------------------------------------------------


async def test_pg_update_session_no_fields_returns_early():
    pool, conn = _make_pool()
    repo = SessionRepository(pool)
    await repo.update_session("sess-1")
    conn.execute.assert_not_called()


async def test_pg_update_session_status_only():
    pool, conn = _make_pool()
    repo = SessionRepository(pool)
    await repo.update_session("sess-1", status=SessionStatus.COMPLETE)
    conn.execute.assert_called_once()
    sql = conn.execute.call_args.args[0]
    assert "status" in sql
    assert "completed_at = NOW()" in sql
    assert "updated_at = NOW()" in sql
    assert "last_activity_at = NOW()" in sql


async def test_pg_update_session_status_closed():
    pool, conn = _make_pool()
    repo = SessionRepository(pool)
    await repo.update_session("sess-1", status=SessionStatus.CLOSED)
    sql = conn.execute.call_args.args[0]
    assert "closed_at = NOW()" in sql


async def test_pg_update_session_status_active_no_timestamp():
    pool, conn = _make_pool()
    repo = SessionRepository(pool)
    await repo.update_session("sess-1", status=SessionStatus.ACTIVE)
    sql = conn.execute.call_args.args[0]
    assert "completed_at" not in sql
    assert "closed_at" not in sql


async def test_pg_update_session_all_fields():
    pool, conn = _make_pool()
    repo = SessionRepository(pool)
    await repo.update_session(
        "sess-1",
        status=SessionStatus.ACTIVE,
        expected_count=10,
        found_count=5,
        missing_count=3,
        image_count=2,
        message="send another photo",
        candidates=[{"barcode": "BC1"}],
    )
    conn.execute.assert_called_once()
    sql = conn.execute.call_args.args[0]
    assert "status" in sql
    assert "expected_count" in sql
    assert "found_count" in sql
    assert "missing_count" in sql
    assert "image_count" in sql
    assert "message" in sql
    assert "candidates" in sql
    args = conn.execute.call_args.args
    assert args[1] == "sess-1"
    # candidates should be JSON-serialized
    assert json.loads(args[-1]) == [{"barcode": "BC1"}]


async def test_pg_update_session_candidates_json():
    pool, conn = _make_pool()
    repo = SessionRepository(pool)
    candidates = [{"barcode": "BC1"}, {"barcode": "BC2"}]
    await repo.update_session("sess-1", candidates=candidates)
    args = conn.execute.call_args.args
    assert json.loads(args[2]) == candidates


async def test_pg_update_session_error_propagates():
    pool, _ = _make_pool_raising(OSError("conn refused"))
    repo = SessionRepository(pool)
    with pytest.raises(OSError, match="conn refused"):
        await repo.update_session("sess-1", status=SessionStatus.ACTIVE)


# ---------------------------------------------------------------------------
# close_session
# ---------------------------------------------------------------------------


async def test_pg_close_session_success():
    pool, _ = _make_pool(execute_return="UPDATE 1")
    repo = SessionRepository(pool)
    result = await repo.close_session("sess-1")
    assert result is True


async def test_pg_close_session_already_closed():
    pool, _ = _make_pool(execute_return="UPDATE 0")
    repo = SessionRepository(pool)
    result = await repo.close_session("sess-1")
    assert result is False


async def test_pg_close_session_error_propagates():
    pool, _ = _make_pool_raising(asyncpg.PostgresError("boom"))
    repo = SessionRepository(pool)
    with pytest.raises(asyncpg.PostgresError, match="boom"):
        await repo.close_session("sess-1")


# ---------------------------------------------------------------------------
# expire_session
# ---------------------------------------------------------------------------


async def test_pg_expire_session():
    pool, conn = _make_pool()
    repo = SessionRepository(pool)
    await repo.expire_session("sess-1")
    conn.execute.assert_called_once()
    sql = conn.execute.call_args.args[0]
    assert "expired" in sql
    assert "active" in sql


async def test_pg_expire_session_error_propagates():
    pool, _ = _make_pool_raising(OSError("conn refused"))
    repo = SessionRepository(pool)
    with pytest.raises(OSError, match="conn refused"):
        await repo.expire_session("sess-1")


# ---------------------------------------------------------------------------
# add_item
# ---------------------------------------------------------------------------


async def test_pg_add_item_inserted():
    pool, _ = _make_pool(execute_return="INSERT 0 1")
    repo = SessionRepository(pool)
    item = _make_session_item()
    result = await repo.add_item("sess-1", item)
    assert result is True


async def test_pg_add_item_duplicate():
    pool, _ = _make_pool(execute_return="INSERT 0 0")
    repo = SessionRepository(pool)
    item = _make_session_item()
    result = await repo.add_item("sess-1", item)
    assert result is False


async def test_pg_add_item_with_bbox():
    pool, conn = _make_pool()
    item = _make_session_item(
        barcode_bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10},
        label_bbox={"x1": 0, "y1": 0, "x2": 20, "y2": 20},
    )
    repo = SessionRepository(pool)
    await repo.add_item("sess-1", item)
    conn.execute.assert_called_once()
    args = conn.execute.call_args.args
    # barcode_bbox and label_bbox should be JSON strings
    assert json.loads(args[4]) == {"x1": 0, "y1": 0, "x2": 10, "y2": 10}
    assert json.loads(args[5]) == {"x1": 0, "y1": 0, "x2": 20, "y2": 20}


async def test_pg_add_item_without_bbox():
    pool, conn = _make_pool()
    item = _make_session_item(barcode_bbox=None, label_bbox=None)
    repo = SessionRepository(pool)
    await repo.add_item("sess-1", item)
    args = conn.execute.call_args.args
    assert args[4] is None
    assert args[5] is None


async def test_pg_add_item_error_propagates():
    pool, _ = _make_pool_raising(asyncpg.PostgresError("boom"))
    repo = SessionRepository(pool)
    with pytest.raises(asyncpg.PostgresError, match="boom"):
        await repo.add_item("sess-1", _make_session_item())


# ---------------------------------------------------------------------------
# get_items
# ---------------------------------------------------------------------------


async def test_pg_get_items_found():
    rows = [
        {
            "barcode_value": "BC1", "barcode_format": "EAN_13",
            "barcode_bbox": json.dumps({"x1": 0, "y1": 0}),
            "label_bbox": json.dumps({"x1": 0, "y1": 0}),
            "label_index": 0, "match_basis": "containment",
            "source_image": 0,
        },
        {
            "barcode_value": "BC2", "barcode_format": None,
            "barcode_bbox": None, "label_bbox": None,
            "label_index": 1, "match_basis": None,
            "source_image": 1,
        },
    ]
    pool, _ = _make_pool(fetch_rows=rows)
    repo = SessionRepository(pool)
    result = await repo.get_items("sess-1")
    assert len(result) == 2
    assert result[0].barcode_value == "BC1"
    assert result[0].barcode_bbox == {"x1": 0, "y1": 0}
    assert result[0].label_bbox == {"x1": 0, "y1": 0}
    assert result[1].barcode_value == "BC2"
    assert result[1].barcode_bbox is None
    assert result[1].label_bbox is None


async def test_pg_get_items_empty():
    pool, _ = _make_pool(fetch_rows=[])
    repo = SessionRepository(pool)
    result = await repo.get_items("sess-1")
    assert result == []


async def test_pg_get_items_error_propagates():
    pool, _ = _make_pool_raising(OSError("conn refused"))
    repo = SessionRepository(pool)
    with pytest.raises(OSError, match="conn refused"):
        await repo.get_items("sess-1")


# ---------------------------------------------------------------------------
# add_missing
# ---------------------------------------------------------------------------


async def test_pg_add_missing():
    pool, conn = _make_pool()
    repo = SessionRepository(pool)
    item = _make_missing_item(
        label_bbox={"x1": 0, "y1": 0},
        barcode_bbox={"x1": 0, "y1": 0},
    )
    await repo.add_missing("sess-1", item)
    conn.execute.assert_called_once()
    args = conn.execute.call_args.args
    assert args[1] == "sess-1"
    assert json.loads(args[3]) == {"x1": 0, "y1": 0}
    assert json.loads(args[4]) == {"x1": 0, "y1": 0}


async def test_pg_add_missing_without_bbox():
    pool, conn = _make_pool()
    repo = SessionRepository(pool)
    item = _make_missing_item(label_bbox=None, barcode_bbox=None)
    await repo.add_missing("sess-1", item)
    args = conn.execute.call_args.args
    assert args[3] is None
    assert args[4] is None


async def test_pg_add_missing_error_propagates():
    pool, _ = _make_pool_raising(asyncpg.PostgresError("boom"))
    repo = SessionRepository(pool)
    with pytest.raises(asyncpg.PostgresError, match="boom"):
        await repo.add_missing("sess-1", _make_missing_item())


# ---------------------------------------------------------------------------
# get_missing
# ---------------------------------------------------------------------------


async def test_pg_get_missing_found():
    rows = [
        {
            "label_index": 0,
            "label_bbox": json.dumps({"x1": 0, "y1": 0}),
            "barcode_bbox": json.dumps({"x1": 0, "y1": 0}),
            "status": "not_visible", "source_image": 0,
            "resolved": False,
        },
        {
            "label_index": 1,
            "label_bbox": None, "barcode_bbox": None,
            "status": "not_visible", "source_image": 1,
            "resolved": True,
        },
    ]
    pool, _ = _make_pool(fetch_rows=rows)
    repo = SessionRepository(pool)
    result = await repo.get_missing("sess-1")
    assert len(result) == 2
    assert result[0].label_index == 0
    assert result[0].label_bbox == {"x1": 0, "y1": 0}
    assert result[0].resolved is False
    assert result[1].label_index == 1
    assert result[1].label_bbox is None
    assert result[1].resolved is True


async def test_pg_get_missing_empty():
    pool, _ = _make_pool(fetch_rows=[])
    repo = SessionRepository(pool)
    result = await repo.get_missing("sess-1")
    assert result == []


async def test_pg_get_missing_error_propagates():
    pool, _ = _make_pool_raising(OSError("conn refused"))
    repo = SessionRepository(pool)
    with pytest.raises(OSError, match="conn refused"):
        await repo.get_missing("sess-1")


# ---------------------------------------------------------------------------
# resolve_missing
# ---------------------------------------------------------------------------


async def test_pg_resolve_missing():
    pool, conn = _make_pool()
    repo = SessionRepository(pool)
    await repo.resolve_missing("sess-1", 5, 2)
    conn.execute.assert_called_once()
    args = conn.execute.call_args.args
    assert args[1] == "sess-1"
    assert args[2] == 5
    assert args[3] == 2


async def test_pg_resolve_missing_error_propagates():
    pool, _ = _make_pool_raising(asyncpg.PostgresError("boom"))
    repo = SessionRepository(pool)
    with pytest.raises(asyncpg.PostgresError, match="boom"):
        await repo.resolve_missing("sess-1", 5, 2)


# ---------------------------------------------------------------------------
# clear_missing
# ---------------------------------------------------------------------------


async def test_pg_clear_missing():
    pool, conn = _make_pool()
    repo = SessionRepository(pool)
    await repo.clear_missing("sess-1")
    conn.execute.assert_called_once()
    sql = conn.execute.call_args.args[0]
    assert "DELETE FROM session_missing" in sql


async def test_pg_clear_missing_error_propagates():
    pool, _ = _make_pool_raising(OSError("conn refused"))
    repo = SessionRepository(pool)
    with pytest.raises(OSError, match="conn refused"):
        await repo.clear_missing("sess-1")


# ---------------------------------------------------------------------------
# load_session_state
# ---------------------------------------------------------------------------


async def test_pg_load_session_state_found():
    session_row = _make_session_row()
    item_rows = [
        {
            "barcode_value": "BC1", "barcode_format": "EAN_13",
            "barcode_bbox": None, "label_bbox": None,
            "label_index": 0, "match_basis": "containment",
            "source_image": 0,
        },
    ]
    missing_rows = [
        {
            "label_index": 5, "label_bbox": None, "barcode_bbox": None,
            "status": "not_visible", "source_image": 0, "resolved": False,
        },
    ]

    conn = MagicMock()
    conn.fetchrow = AsyncMock(return_value=session_row)
    # First fetch = items, second fetch = missing
    conn.fetch = AsyncMock(side_effect=[item_rows, missing_rows])
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=FakeAcquire(conn))

    repo = SessionRepository(pool)
    result = await repo.load_session_state("sess-1")
    assert result is not None
    assert result["session"]["id"] == "sess-1"
    assert len(result["items"]) == 1
    assert result["items"][0].barcode_value == "BC1"
    assert len(result["missing"]) == 1
    assert result["missing"][0].label_index == 5


async def test_pg_load_session_state_not_found():
    pool, _ = _make_pool(fetchrow=None)
    repo = SessionRepository(pool)
    result = await repo.load_session_state("missing")
    assert result is None


async def test_pg_load_session_state_no_items_no_missing():
    session_row = _make_session_row()
    conn = MagicMock()
    conn.fetchrow = AsyncMock(return_value=session_row)
    conn.fetch = AsyncMock(side_effect=[[], []])
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=FakeAcquire(conn))

    repo = SessionRepository(pool)
    result = await repo.load_session_state("sess-1")
    assert result is not None
    assert result["items"] == []
    assert result["missing"] == []


# ---------------------------------------------------------------------------
# to_result
# ---------------------------------------------------------------------------


async def test_pg_to_result_found():
    session_row = _make_session_row(
        status="complete", expected_count=10, found_count=10,
        missing_count=0, image_count=2,
    )
    item_rows = [
        {
            "barcode_value": "BC1", "barcode_format": "EAN_13",
            "barcode_bbox": None, "label_bbox": None,
            "label_index": 0, "match_basis": "containment",
            "source_image": 0,
        },
    ]
    missing_rows: list[dict] = []

    conn = MagicMock()
    conn.fetchrow = AsyncMock(return_value=session_row)
    conn.fetch = AsyncMock(side_effect=[item_rows, missing_rows])
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=FakeAcquire(conn))

    repo = SessionRepository(pool)
    result = await repo.to_result("sess-1")
    assert result is not None
    assert result.session_id == "sess-1"
    assert result.status == SessionStatus.COMPLETE
    assert result.expected_count == 10
    assert result.found_count == 10
    assert result.missing_count == 0
    assert result.image_count == 2
    assert len(result.items) == 1
    assert len(result.missing) == 0
    assert result.customer_id == "C1"
    assert result.branch_id == "B1"
    assert result.action == "create_order"


async def test_pg_to_result_not_found():
    pool, _ = _make_pool(fetchrow=None)
    repo = SessionRepository(pool)
    result = await repo.to_result("missing")
    assert result is None


async def test_pg_to_result_filters_resolved_missing():
    session_row = _make_session_row(status="active")
    missing_rows = [
        {
            "label_index": 0, "label_bbox": None, "barcode_bbox": None,
            "status": "not_visible", "source_image": 0, "resolved": False,
        },
        {
            "label_index": 1, "label_bbox": None, "barcode_bbox": None,
            "status": "not_visible", "source_image": 0, "resolved": True,
        },
    ]
    conn = MagicMock()
    conn.fetchrow = AsyncMock(return_value=session_row)
    conn.fetch = AsyncMock(side_effect=[[], missing_rows])
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=FakeAcquire(conn))

    repo = SessionRepository(pool)
    result = await repo.to_result("sess-1")
    assert result is not None
    # Only the unresolved missing item should be in result
    assert len(result.missing) == 1
    assert result.missing[0].label_index == 0


# ===========================================================================
# NoOpSessionRepository
# ===========================================================================


# ---------------------------------------------------------------------------
# create_session
# ---------------------------------------------------------------------------


async def test_noop_create_session():
    repo = NoOpSessionRepository()
    await repo.create_session(
        "sess-1", source="web", channel="whatsapp",
        participant_id="user-1", customer_id="C1",
        branch_id="B1", action="create_order",
    )
    session = await repo.get_session("sess-1")
    assert session is not None
    assert session["id"] == "sess-1"
    assert session["status"] == "active"
    assert session["source"] == "web"
    assert session["channel"] == "whatsapp"
    assert session["participant_id"] == "user-1"
    assert session["customer_id"] == "C1"
    assert session["branch_id"] == "B1"
    assert session["action"] == "create_order"
    assert session["expected_count"] == 0
    assert session["found_count"] == 0
    assert session["missing_count"] == 0
    assert session["image_count"] == 0
    assert session["message"] is None
    assert "last_activity_at" in session


async def test_noop_create_session_already_exists():
    repo = NoOpSessionRepository()
    await repo.create_session("sess-1", source="web")
    first = await repo.get_session("sess-1")
    assert first is not None
    assert first["source"] == "web"
    # Create again with different source — should NOT overwrite
    await repo.create_session("sess-1", source="whatsapp")
    second = await repo.get_session("sess-1")
    assert second["source"] == "web"


async def test_noop_create_session_minimal():
    repo = NoOpSessionRepository()
    await repo.create_session("sess-1")
    session = await repo.get_session("sess-1")
    assert session is not None
    assert session["source"] is None
    assert session["channel"] is None


# ---------------------------------------------------------------------------
# get_session
# ---------------------------------------------------------------------------


async def test_noop_get_session_not_found():
    repo = NoOpSessionRepository()
    result = await repo.get_session("missing")
    assert result is None


async def test_noop_get_session_found():
    repo = NoOpSessionRepository()
    await repo.create_session("sess-1")
    result = await repo.get_session("sess-1")
    assert result is not None
    assert result["id"] == "sess-1"


# ---------------------------------------------------------------------------
# find_active_by_participant
# ---------------------------------------------------------------------------


async def test_noop_find_active_by_participant_found():
    repo = NoOpSessionRepository()
    await repo.create_session(
        "sess-1", channel="whatsapp", participant_id="user-1",
    )
    result = await repo.find_active_by_participant("whatsapp", "user-1")
    assert result is not None
    assert result["id"] == "sess-1"


async def test_noop_find_active_by_participant_needs_user_selection():
    repo = NoOpSessionRepository()
    await repo.create_session(
        "sess-1", channel="whatsapp", participant_id="user-1",
    )
    await repo.update_session("sess-1", status=SessionStatus.NEEDS_USER_SELECTION)
    result = await repo.find_active_by_participant("whatsapp", "user-1")
    assert result is not None
    assert result["id"] == "sess-1"


async def test_noop_find_active_by_participant_not_found():
    repo = NoOpSessionRepository()
    await repo.create_session(
        "sess-1", channel="whatsapp", participant_id="user-1",
    )
    result = await repo.find_active_by_participant("whatsapp", "nobody")
    assert result is None


async def test_noop_find_active_by_participant_wrong_channel():
    repo = NoOpSessionRepository()
    await repo.create_session(
        "sess-1", channel="whatsapp", participant_id="user-1",
    )
    result = await repo.find_active_by_participant("web", "user-1")
    assert result is None


async def test_noop_find_active_by_participant_closed_not_returned():
    repo = NoOpSessionRepository()
    await repo.create_session(
        "sess-1", channel="whatsapp", participant_id="user-1",
    )
    await repo.close_session("sess-1")
    result = await repo.find_active_by_participant("whatsapp", "user-1")
    assert result is None


# ---------------------------------------------------------------------------
# update_session
# ---------------------------------------------------------------------------


async def test_noop_update_session_existing():
    repo = NoOpSessionRepository()
    await repo.create_session("sess-1")
    await repo.update_session(
        "sess-1", status=SessionStatus.COMPLETE,
        expected_count=10, found_count=10,
    )
    session = await repo.get_session("sess-1")
    assert session["status"] == "complete"
    assert session["expected_count"] == 10
    assert session["found_count"] == 10


async def test_noop_update_session_creates_if_missing():
    repo = NoOpSessionRepository()
    await repo.update_session("sess-1", expected_count=5)
    session = await repo.get_session("sess-1")
    assert session is not None
    assert session["id"] == "sess-1"
    assert session["expected_count"] == 5


async def test_noop_update_session_skips_none_values():
    repo = NoOpSessionRepository()
    await repo.create_session("sess-1")
    await repo.update_session(
        "sess-1", expected_count=10, found_count=None,
    )
    session = await repo.get_session("sess-1")
    assert session["expected_count"] == 10
    # found_count should remain 0 (default), not None
    assert session["found_count"] == 0


async def test_noop_update_session_status_string():
    repo = NoOpSessionRepository()
    await repo.create_session("sess-1")
    await repo.update_session("sess-1", status="expired")
    session = await repo.get_session("sess-1")
    assert session["status"] == "expired"


async def test_noop_update_session_sets_last_activity():
    repo = NoOpSessionRepository()
    await repo.create_session("sess-1")
    before = (await repo.get_session("sess-1"))["last_activity_at"]
    await repo.update_session("sess-1", expected_count=5)
    after = (await repo.get_session("sess-1"))["last_activity_at"]
    assert after >= before


# ---------------------------------------------------------------------------
# close_session
# ---------------------------------------------------------------------------


async def test_noop_close_session_active():
    repo = NoOpSessionRepository()
    await repo.create_session("sess-1")
    result = await repo.close_session("sess-1")
    assert result is True
    session = await repo.get_session("sess-1")
    assert session["status"] == "closed"


async def test_noop_close_session_complete():
    repo = NoOpSessionRepository()
    await repo.create_session("sess-1")
    await repo.update_session("sess-1", status=SessionStatus.COMPLETE)
    result = await repo.close_session("sess-1")
    assert result is True


async def test_noop_close_session_already_closed():
    repo = NoOpSessionRepository()
    await repo.create_session("sess-1")
    await repo.close_session("sess-1")
    result = await repo.close_session("sess-1")
    assert result is False


async def test_noop_close_session_not_found():
    repo = NoOpSessionRepository()
    result = await repo.close_session("missing")
    assert result is False


async def test_noop_close_session_expired():
    repo = NoOpSessionRepository()
    await repo.create_session("sess-1")
    await repo.expire_session("sess-1")
    result = await repo.close_session("sess-1")
    assert result is False


# ---------------------------------------------------------------------------
# expire_session
# ---------------------------------------------------------------------------


async def test_noop_expire_session_active():
    repo = NoOpSessionRepository()
    await repo.create_session("sess-1")
    await repo.expire_session("sess-1")
    session = await repo.get_session("sess-1")
    assert session["status"] == "expired"


async def test_noop_expire_session_not_active():
    repo = NoOpSessionRepository()
    await repo.create_session("sess-1")
    await repo.close_session("sess-1")
    await repo.expire_session("sess-1")
    session = await repo.get_session("sess-1")
    assert session["status"] == "closed"


async def test_noop_expire_session_not_found():
    repo = NoOpSessionRepository()
    await repo.expire_session("missing")
    # Should not raise
    assert await repo.get_session("missing") is None


# ---------------------------------------------------------------------------
# add_item
# ---------------------------------------------------------------------------


async def test_noop_add_item_new():
    repo = NoOpSessionRepository()
    item = _make_session_item()
    result = await repo.add_item("sess-1", item)
    assert result is True
    items = await repo.get_items("sess-1")
    assert len(items) == 1
    assert items[0].barcode_value == "BC1"


async def test_noop_add_item_duplicate():
    repo = NoOpSessionRepository()
    item = _make_session_item(label_index=0, source_image=0)
    result1 = await repo.add_item("sess-1", item)
    assert result1 is True
    result2 = await repo.add_item("sess-1", item)
    assert result2 is False
    items = await repo.get_items("sess-1")
    assert len(items) == 1


async def test_noop_add_item_different_source_image():
    repo = NoOpSessionRepository()
    item1 = _make_session_item(label_index=0, source_image=0)
    item2 = _make_session_item(
        barcode_value="BC2", label_index=0, source_image=1,
    )
    await repo.add_item("sess-1", item1)
    await repo.add_item("sess-1", item2)
    items = await repo.get_items("sess-1")
    assert len(items) == 2


async def test_noop_add_item_none_label_index_no_dedup():
    repo = NoOpSessionRepository()
    item1 = _make_session_item(label_index=None, barcode_value="BC1")
    item2 = _make_session_item(label_index=None, barcode_value="BC1")
    r1 = await repo.add_item("sess-1", item1)
    r2 = await repo.add_item("sess-1", item2)
    assert r1 is True
    assert r2 is True
    items = await repo.get_items("sess-1")
    assert len(items) == 2


async def test_noop_add_item_creates_session_if_missing():
    repo = NoOpSessionRepository()
    item = _make_session_item()
    await repo.add_item("sess-1", item)
    items = await repo.get_items("sess-1")
    assert len(items) == 1


# ---------------------------------------------------------------------------
# get_items
# ---------------------------------------------------------------------------


async def test_noop_get_items_empty():
    repo = NoOpSessionRepository()
    result = await repo.get_items("sess-1")
    assert result == []


async def test_noop_get_items_returns_copy():
    repo = NoOpSessionRepository()
    item = _make_session_item()
    await repo.add_item("sess-1", item)
    items1 = await repo.get_items("sess-1")
    items1.append(_make_session_item(barcode_value="BC2"))
    items2 = await repo.get_items("sess-1")
    assert len(items2) == 1


# ---------------------------------------------------------------------------
# add_missing
# ---------------------------------------------------------------------------


async def test_noop_add_missing():
    repo = NoOpSessionRepository()
    item = _make_missing_item()
    await repo.add_missing("sess-1", item)
    missing = await repo.get_missing("sess-1")
    assert len(missing) == 1


async def test_noop_add_missing_multiple():
    repo = NoOpSessionRepository()
    await repo.add_missing("sess-1", _make_missing_item(label_index=0))
    await repo.add_missing("sess-1", _make_missing_item(label_index=1))
    missing = await repo.get_missing("sess-1")
    assert len(missing) == 2


async def test_noop_add_missing_creates_session_if_missing():
    repo = NoOpSessionRepository()
    await repo.add_missing("sess-1", _make_missing_item())
    missing = await repo.get_missing("sess-1")
    assert len(missing) == 1


# ---------------------------------------------------------------------------
# get_missing
# ---------------------------------------------------------------------------


async def test_noop_get_missing_empty():
    repo = NoOpSessionRepository()
    result = await repo.get_missing("sess-1")
    assert result == []


async def test_noop_get_missing_returns_copy():
    repo = NoOpSessionRepository()
    await repo.add_missing("sess-1", _make_missing_item())
    missing1 = await repo.get_missing("sess-1")
    missing1.append(_make_missing_item(label_index=99))
    missing2 = await repo.get_missing("sess-1")
    assert len(missing2) == 1


# ---------------------------------------------------------------------------
# resolve_missing
# ---------------------------------------------------------------------------


async def test_noop_resolve_missing():
    repo = NoOpSessionRepository()
    await repo.add_missing("sess-1", _make_missing_item(label_index=0))
    await repo.add_missing("sess-1", _make_missing_item(label_index=1))
    await repo.resolve_missing("sess-1", 0, 2)
    missing = await repo.get_missing("sess-1")
    assert missing[0].resolved is True
    assert missing[1].resolved is False


async def test_noop_resolve_missing_not_found():
    repo = NoOpSessionRepository()
    await repo.add_missing("sess-1", _make_missing_item(label_index=0))
    # Resolving a non-existent label_index should not raise
    await repo.resolve_missing("sess-1", 99, 2)
    missing = await repo.get_missing("sess-1")
    assert missing[0].resolved is False


async def test_noop_resolve_missing_empty_session():
    repo = NoOpSessionRepository()
    await repo.resolve_missing("missing", 0, 1)
    # Should not raise


# ---------------------------------------------------------------------------
# clear_missing
# ---------------------------------------------------------------------------


async def test_noop_clear_missing():
    repo = NoOpSessionRepository()
    await repo.add_missing("sess-1", _make_missing_item(label_index=0))
    await repo.add_missing("sess-1", _make_missing_item(label_index=1))
    await repo.clear_missing("sess-1")
    missing = await repo.get_missing("sess-1")
    assert missing == []


async def test_noop_clear_missing_empty_session():
    repo = NoOpSessionRepository()
    await repo.clear_missing("missing")
    # Should not raise


# ---------------------------------------------------------------------------
# load_session_state
# ---------------------------------------------------------------------------


async def test_noop_load_session_state_found():
    repo = NoOpSessionRepository()
    await repo.create_session("sess-1", source="web")
    await repo.add_item("sess-1", _make_session_item())
    await repo.add_missing("sess-1", _make_missing_item())
    state = await repo.load_session_state("sess-1")
    assert state is not None
    assert state["session"]["id"] == "sess-1"
    assert state["session"]["source"] == "web"
    assert len(state["items"]) == 1
    assert len(state["missing"]) == 1


async def test_noop_load_session_state_not_found():
    repo = NoOpSessionRepository()
    result = await repo.load_session_state("missing")
    assert result is None


async def test_noop_load_session_state_excludes_internal_keys():
    repo = NoOpSessionRepository()
    await repo.create_session("sess-1")
    await repo.add_item("sess-1", _make_session_item())
    state = await repo.load_session_state("sess-1")
    assert state is not None
    # Internal keys (starting with _) should not be in the session dict
    assert "_items" not in state["session"]
    assert "_missing" not in state["session"]


async def test_noop_load_session_state_no_items_no_missing():
    repo = NoOpSessionRepository()
    await repo.create_session("sess-1")
    state = await repo.load_session_state("sess-1")
    assert state is not None
    assert state["items"] == []
    assert state["missing"] == []


# ---------------------------------------------------------------------------
# to_result
# ---------------------------------------------------------------------------


async def test_noop_to_result_found():
    repo = NoOpSessionRepository()
    await repo.create_session("sess-1", customer_id="C1", branch_id="B1")
    await repo.update_session(
        "sess-1", status=SessionStatus.COMPLETE,
        expected_count=10, found_count=10, missing_count=0,
        image_count=2,
    )
    await repo.add_item("sess-1", _make_session_item())
    result = await repo.to_result("sess-1")
    assert result is not None
    assert result.session_id == "sess-1"
    assert result.status == SessionStatus.COMPLETE
    assert result.expected_count == 10
    assert result.found_count == 10
    assert result.missing_count == 0
    assert result.image_count == 2
    assert len(result.items) == 1
    assert len(result.missing) == 0
    assert result.customer_id == "C1"
    assert result.branch_id == "B1"


async def test_noop_to_result_not_found():
    repo = NoOpSessionRepository()
    result = await repo.to_result("missing")
    assert result is None


async def test_noop_to_result_filters_resolved_missing():
    repo = NoOpSessionRepository()
    await repo.create_session("sess-1")
    await repo.add_missing("sess-1", _make_missing_item(label_index=0))
    await repo.add_missing(
        "sess-1", _make_missing_item(label_index=1, resolved=True),
    )
    result = await repo.to_result("sess-1")
    assert result is not None
    assert len(result.missing) == 1
    assert result.missing[0].label_index == 0


async def test_noop_to_result_default_status():
    """Session created via update_session (no explicit status) defaults to active."""
    repo = NoOpSessionRepository()
    await repo.update_session("sess-1", expected_count=5)
    result = await repo.to_result("sess-1")
    assert result is not None
    assert result.status == SessionStatus.ACTIVE


async def test_noop_to_result_with_message():
    repo = NoOpSessionRepository()
    await repo.create_session("sess-1")
    await repo.update_session("sess-1", message="send another photo")
    result = await repo.to_result("sess-1")
    assert result is not None
    assert result.message == "send another photo"


# ---------------------------------------------------------------------------
# Full lifecycle integration (NoOp)
# ---------------------------------------------------------------------------


async def test_noop_full_lifecycle():
    """Test a full session lifecycle with the NoOp repository."""
    repo = NoOpSessionRepository()
    # Create
    await repo.create_session("sess-1", source="web")
    # Add items
    await repo.add_item("sess-1", _make_session_item(
        barcode_value="BC1", label_index=0, source_image=0,
    ))
    await repo.add_item("sess-1", _make_session_item(
        barcode_value="BC2", label_index=1, source_image=0,
    ))
    # Add missing
    await repo.add_missing("sess-1", _make_missing_item(label_index=2))
    # Resolve one missing
    await repo.resolve_missing("sess-1", 2, 1)
    # Update session
    await repo.update_session(
        "sess-1", status=SessionStatus.COMPLETE,
        expected_count=3, found_count=3, missing_count=0, image_count=2,
    )
    # Load state
    state = await repo.load_session_state("sess-1")
    assert state is not None
    assert len(state["items"]) == 2
    assert len(state["missing"]) == 1
    assert state["missing"][0].resolved is True
    # Convert to result
    result = await repo.to_result("sess-1")
    assert result is not None
    assert result.status == SessionStatus.COMPLETE
    assert result.found_count == 3
    # Resolved missing should be filtered from result
    assert len(result.missing) == 0
    # Close
    closed = await repo.close_session("sess-1")
    assert closed is True
    session = await repo.get_session("sess-1")
    assert session["status"] == "closed"

"""Tests for the session model and repository.

Tests use the NoOpSessionRepository (in-memory) so they run without a DB.
The PostgresSessionRepository is exercised in test_db.py against the real DB.
"""

from __future__ import annotations

import pytest

from src.ingest.session_models import (
    ImageResult,
    MissingItem,
    SessionItem,
    SessionResult,
    SessionStatus,
)
from src.session_repository import NoOpSessionRepository

# ---------------------------------------------------------------------------
# Model tests (sync — no asyncio mark)
# ---------------------------------------------------------------------------


def test_session_status_values() -> None:
    assert SessionStatus.ACTIVE == "active"
    assert SessionStatus.COMPLETE == "complete"
    assert SessionStatus.EXPIRED == "expired"
    assert SessionStatus.FAILED == "failed"


def test_session_item_dedup_key() -> None:
    """SessionItem identity is barcode_value."""
    item1 = SessionItem(barcode_value="123456789", barcode_format="Code128")
    item2 = SessionItem(barcode_value="123456789", barcode_format="EAN13")
    item3 = SessionItem(barcode_value="987654321")

    assert item1.barcode_value == item2.barcode_value  # same item
    assert item1.barcode_value != item3.barcode_value  # different item


def test_missing_item_defaults() -> None:
    m = MissingItem(label_index=7)
    assert m.resolved is False
    assert m.status == "not_visible"
    assert m.source_image == 0


def test_session_result_shape() -> None:
    r = SessionResult(
        session_id="S123",
        status=SessionStatus.ACTIVE,
        expected_count=12,
        found_count=11,
        missing_count=1,
    )
    assert r.status == SessionStatus.ACTIVE
    assert r.items == []
    assert r.missing == []
    assert r.image_count == 0
    assert r.latest_image is None


def test_image_result_shape() -> None:
    ir = ImageResult(
        image_index=0,
        status="complete",
        found=[SessionItem(barcode_value="111")],
        missing=[MissingItem(label_index=7)],
        visible_label_count=12,
        found_count=11,
        missing_count=1,
    )
    assert ir.image_index == 0
    assert len(ir.found) == 1
    assert len(ir.missing) == 1
    assert ir.audit_available is False


# ---------------------------------------------------------------------------
# NoOpSessionRepository tests (async)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_noop_create_and_get_session() -> None:
    repo = NoOpSessionRepository()
    await repo.create_session("S1", source="web")
    s = await repo.get_session("S1")
    assert s is not None
    assert s["id"] == "S1"
    assert s["status"] == "active"
    assert s["source"] == "web"


@pytest.mark.asyncio
async def test_noop_get_nonexistent_session() -> None:
    repo = NoOpSessionRepository()
    assert await repo.get_session("nope") is None


@pytest.mark.asyncio
async def test_noop_update_session() -> None:
    repo = NoOpSessionRepository()
    await repo.create_session("S1")
    await repo.update_session(
        "S1",
        status=SessionStatus.COMPLETE,
        expected_count=12,
        found_count=12,
        missing_count=0,
        image_count=1,
    )
    s = await repo.get_session("S1")
    assert s["status"] == "complete"
    assert s["expected_count"] == 12
    assert s["found_count"] == 12


@pytest.mark.asyncio
async def test_noop_add_item_dedup() -> None:
    repo = NoOpSessionRepository()
    await repo.create_session("S1")

    item1 = SessionItem(barcode_value="111", barcode_format="Code128")
    item2 = SessionItem(barcode_value="111", barcode_format="EAN13")  # same value
    item3 = SessionItem(barcode_value="222")

    assert await repo.add_item("S1", item1) is True   # newly inserted
    assert await repo.add_item("S1", item2) is False  # deduplicated
    assert await repo.add_item("S1", item3) is True   # newly inserted

    items = await repo.get_items("S1")
    assert len(items) == 2  # 111 and 222, not 3


@pytest.mark.asyncio
async def test_noop_add_and_resolve_missing() -> None:
    repo = NoOpSessionRepository()
    await repo.create_session("S1")

    m1 = MissingItem(label_index=7, status="not_visible")
    m2 = MissingItem(label_index=3, status="blurred")

    await repo.add_missing("S1", m1)
    await repo.add_missing("S1", m2)

    missing = await repo.get_missing("S1")
    assert len(missing) == 2
    assert all(not m.resolved for m in missing)

    await repo.resolve_missing("S1", 7, resolved_by_image=1)
    missing = await repo.get_missing("S1")
    resolved = [m for m in missing if m.resolved]
    unresolved = [m for m in missing if not m.resolved]
    assert len(resolved) == 1
    assert resolved[0].label_index == 7
    assert len(unresolved) == 1
    assert unresolved[0].label_index == 3


@pytest.mark.asyncio
async def test_noop_clear_missing() -> None:
    repo = NoOpSessionRepository()
    await repo.create_session("S1")
    await repo.add_missing("S1", MissingItem(label_index=1))
    await repo.add_missing("S1", MissingItem(label_index=2))

    await repo.clear_missing("S1")
    missing = await repo.get_missing("S1")
    assert len(missing) == 0


@pytest.mark.asyncio
async def test_noop_to_result() -> None:
    repo = NoOpSessionRepository()
    await repo.create_session("S1", source="web")
    await repo.add_item("S1", SessionItem(barcode_value="111"))
    await repo.add_item("S1", SessionItem(barcode_value="222"))
    await repo.add_missing("S1", MissingItem(label_index=7))
    await repo.update_session(
        "S1",
        status=SessionStatus.ACTIVE,
        expected_count=3,
        found_count=2,
        missing_count=1,
        image_count=1,
        message="Send a photo of box 7",
    )

    result = await repo.to_result("S1")
    assert result is not None
    assert result.session_id == "S1"
    assert result.status == SessionStatus.ACTIVE
    assert result.expected_count == 3
    assert result.found_count == 2
    assert result.missing_count == 1
    assert len(result.items) == 2
    assert len(result.missing) == 1
    assert result.image_count == 1
    assert result.message == "Send a photo of box 7"


@pytest.mark.asyncio
async def test_noop_to_result_nonexistent() -> None:
    repo = NoOpSessionRepository()
    assert await repo.to_result("nope") is None


@pytest.mark.asyncio
async def test_noop_load_session_state() -> None:
    repo = NoOpSessionRepository()
    await repo.create_session("S1")
    await repo.add_item("S1", SessionItem(barcode_value="111"))
    await repo.add_missing("S1", MissingItem(label_index=7))

    state = await repo.load_session_state("S1")
    assert state is not None
    assert state["session"]["id"] == "S1"
    assert len(state["items"]) == 1
    assert len(state["missing"]) == 1


@pytest.mark.asyncio
async def test_noop_load_session_state_nonexistent() -> None:
    repo = NoOpSessionRepository()
    assert await repo.load_session_state("nope") is None

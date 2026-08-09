"""Tests for the SessionGraph — multi-image ingest orchestration.

Tests mock analyze_image() so no real barcode scanning or Gemini calls happen.
Uses NoOpSessionRepository for in-memory session state.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.ingest.session_graph import run_session_graph
from src.ingest.session_models import SessionStatus
from src.session_repository import NoOpSessionRepository


@pytest.mark.asyncio
async def test_session_complete_on_first_image(tmp_path: Path) -> None:
    """All boxes found in one photo → session complete immediately."""
    repo = NoOpSessionRepository()
    img = tmp_path / "img.png"
    img.write_bytes(b"fake")

    mock_result = {
        "outcome": "complete",
        "audit_available": True,
        "found": [
            {"barcode_value": "111", "barcode_format": "Code128", "label_index": 1},
            {"barcode_value": "222", "barcode_format": "Code128", "label_index": 2},
        ],
        "missing": [],
        "unassigned": [],
        "summary": {"visible_label_count": 2, "found_count": 2, "missing_count": 0},
    }

    with patch("src.ingest.analyze.analyze_image", return_value=mock_result):
        result = await run_session_graph("S1", img, repo=repo)

    assert result.status == SessionStatus.COMPLETE
    assert result.expected_count == 2
    assert result.found_count == 2
    assert result.missing_count == 0
    assert result.image_count == 1
    assert len(result.items) == 2
    assert result.message is None


@pytest.mark.asyncio
async def test_session_needs_more_images(tmp_path: Path) -> None:
    """One box missing → session active, message tells user which box."""
    repo = NoOpSessionRepository()
    img = tmp_path / "img.png"
    img.write_bytes(b"fake")

    mock_result = {
        "outcome": "needs_better_photo",
        "audit_available": True,
        "found": [
            {"barcode_value": "111", "label_index": 1},
            {"barcode_value": "222", "label_index": 2},
        ],
        "missing": [
            {"label_index": 3, "status": "not_visible", "label_bbox": {}, "barcode_bbox": {}},
        ],
        "unassigned": [],
        "summary": {"visible_label_count": 3, "found_count": 2, "missing_count": 1},
    }

    with patch("src.ingest.analyze.analyze_image", return_value=mock_result):
        result = await run_session_graph("S1", img, repo=repo)

    assert result.status == SessionStatus.ACTIVE
    assert result.expected_count == 3
    assert result.found_count == 2
    assert result.missing_count == 1
    assert result.image_count == 1
    assert len(result.missing) == 1
    assert result.missing[0].label_index == 3
    assert "box" in (result.message or "").lower()
    assert "3" in (result.message or "")


@pytest.mark.asyncio
async def test_session_second_image_resolves_missing(tmp_path: Path) -> None:
    """Photo #2 provides the missing barcode → session complete."""
    repo = NoOpSessionRepository()
    img = tmp_path / "img.png"
    img.write_bytes(b"fake")

    # Photo #1: 3 visible, 2 found, 1 missing (label 3)
    mock1 = {
        "outcome": "needs_better_photo",
        "audit_available": True,
        "found": [
            {"barcode_value": "111", "label_index": 1},
            {"barcode_value": "222", "label_index": 2},
        ],
        "missing": [
            {"label_index": 3, "status": "not_visible", "label_bbox": {}, "barcode_bbox": {}},
        ],
        "unassigned": [],
        "summary": {"visible_label_count": 3, "found_count": 2, "missing_count": 1},
    }

    # Photo #2: close-up of box 3, barcode found
    mock2 = {
        "outcome": "complete",
        "audit_available": True,
        "found": [
            {"barcode_value": "333", "label_index": 3},
        ],
        "missing": [],
        "unassigned": [],
        "summary": {"visible_label_count": 1, "found_count": 1, "missing_count": 0},
    }

    with patch("src.ingest.analyze.analyze_image", return_value=mock1):
        result1 = await run_session_graph("S1", img, repo=repo)

    assert result1.status == SessionStatus.ACTIVE
    assert result1.found_count == 2
    assert result1.missing_count == 1

    with patch("src.ingest.analyze.analyze_image", return_value=mock2):
        result2 = await run_session_graph("S1", img, repo=repo)

    assert result2.status == SessionStatus.COMPLETE
    assert result2.found_count == 3
    assert result2.missing_count == 0
    assert result2.image_count == 2
    assert len(result2.items) == 3
    # The new item should have source_image=1 (second image, 0-indexed)
    new_item = [i for i in result2.items if i.barcode_value == "333"][0]
    assert new_item.source_image == 1


@pytest.mark.asyncio
async def test_session_dedup_across_images(tmp_path: Path) -> None:
    """Same barcode in both photos → counted once, not twice."""
    repo = NoOpSessionRepository()
    img = tmp_path / "img.png"
    img.write_bytes(b"fake")

    mock1 = {
        "outcome": "complete",
        "audit_available": True,
        "found": [
            {"barcode_value": "111", "label_index": 1},
            {"barcode_value": "222", "label_index": 2},
        ],
        "missing": [],
        "unassigned": [],
        "summary": {"visible_label_count": 2, "found_count": 2, "missing_count": 0},
    }

    mock2 = {
        "outcome": "complete",
        "audit_available": True,
        "found": [
            {"barcode_value": "111", "label_index": 1},  # same barcode
            {"barcode_value": "222", "label_index": 2},  # same barcode
        ],
        "missing": [],
        "unassigned": [],
        "summary": {"visible_label_count": 2, "found_count": 2, "missing_count": 0},
    }

    with patch("src.ingest.analyze.analyze_image", return_value=mock1):
        await run_session_graph("S1", img, repo=repo)

    with patch("src.ingest.analyze.analyze_image", return_value=mock2):
        result2 = await run_session_graph("S1", img, repo=repo)

    assert result2.found_count == 2  # not 4
    assert len(result2.items) == 2
    assert result2.image_count == 2


@pytest.mark.asyncio
async def test_session_multi_box_photo_resolves_partial(tmp_path: Path) -> None:
    """Photo #2 has 3 boxes including the missing one → merge resolves it."""
    repo = NoOpSessionRepository()
    img = tmp_path / "img.png"
    img.write_bytes(b"fake")

    # Photo #1: 4 visible, 3 found, 1 missing (label 4)
    mock1 = {
        "outcome": "needs_better_photo",
        "audit_available": True,
        "found": [
            {"barcode_value": "111", "label_index": 1},
            {"barcode_value": "222", "label_index": 2},
            {"barcode_value": "333", "label_index": 3},
        ],
        "missing": [
            {"label_index": 4, "status": "not_visible", "label_bbox": {}, "barcode_bbox": {}},
        ],
        "unassigned": [],
        "summary": {"visible_label_count": 4, "found_count": 3, "missing_count": 1},
    }

    # Photo #2: 3 boxes (2 already known + the missing one)
    mock2 = {
        "outcome": "complete",
        "audit_available": True,
        "found": [
            {"barcode_value": "222", "label_index": 2},  # already known
            {"barcode_value": "333", "label_index": 3},  # already known
            {"barcode_value": "444", "label_index": 4},  # resolves missing!
        ],
        "missing": [],
        "unassigned": [],
        "summary": {"visible_label_count": 3, "found_count": 3, "missing_count": 0},
    }

    with patch("src.ingest.analyze.analyze_image", return_value=mock1):
        result1 = await run_session_graph("S1", img, repo=repo)

    assert result1.status == SessionStatus.ACTIVE
    assert result1.found_count == 3

    with patch("src.ingest.analyze.analyze_image", return_value=mock2):
        result2 = await run_session_graph("S1", img, repo=repo)

    assert result2.status == SessionStatus.COMPLETE
    assert result2.found_count == 4  # 3 original + 1 new (deduped)
    assert result2.missing_count == 0
    assert len(result2.items) == 4


@pytest.mark.asyncio
async def test_session_audit_failure(tmp_path: Path) -> None:
    """Audit failure on first image → session failed, no items accumulated."""
    repo = NoOpSessionRepository()
    img = tmp_path / "img.png"
    img.write_bytes(b"fake")

    mock_result = {
        "outcome": "retryable_error",
        "audit_available": False,
        "found": [],
        "missing": [],
        "unassigned": [],
        "summary": {},
        "error": {"code": "audit_failed", "message": "Gemini error"},
    }

    with patch("src.ingest.analyze.analyze_image", return_value=mock_result):
        result = await run_session_graph("S1", img, repo=repo)

    assert result.status == SessionStatus.FAILED
    assert result.found_count == 0
    assert result.latest_image is not None
    assert result.latest_image.error is not None


@pytest.mark.asyncio
async def test_session_resume_after_failure(tmp_path: Path) -> None:
    """Photo #1 fails, photo #2 succeeds → session works from scratch."""
    repo = NoOpSessionRepository()
    img = tmp_path / "img.png"
    img.write_bytes(b"fake")

    mock_fail = {
        "outcome": "retryable_error",
        "audit_available": False,
        "found": [],
        "missing": [],
        "unassigned": [],
        "summary": {},
        "error": {"code": "audit_failed", "message": "Gemini error"},
    }

    mock_ok = {
        "outcome": "complete",
        "audit_available": True,
        "found": [
            {"barcode_value": "111", "label_index": 1},
            {"barcode_value": "222", "label_index": 2},
        ],
        "missing": [],
        "unassigned": [],
        "summary": {"visible_label_count": 2, "found_count": 2, "missing_count": 0},
    }

    with patch("src.ingest.analyze.analyze_image", return_value=mock_fail):
        result1 = await run_session_graph("S1", img, repo=repo)

    assert result1.status == SessionStatus.FAILED

    # Second image — session was never created (failed), so this is a new session.
    with patch("src.ingest.analyze.analyze_image", return_value=mock_ok):
        result2 = await run_session_graph("S1", img, repo=repo)

    assert result2.status == SessionStatus.COMPLETE
    assert result2.found_count == 2

"""Tests for the SessionGraph — multi-image ingest orchestration.

Tests mock analyze_image() so no real barcode scanning or Gemini calls happen.
Uses NoOpSessionRepository for in-memory session state.
"""

from __future__ import annotations

from datetime import UTC
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.ingest.session_graph import run_session_graph, select_candidate
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

    with patch("src.ingest.analyze.analyze_image_async", new=AsyncMock(return_value=mock_result)):
        result = await run_session_graph(img, repo=repo, channel="web", participant_id="test-user-1")

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

    with patch("src.ingest.analyze.analyze_image_async", new=AsyncMock(return_value=mock_result)):
        result = await run_session_graph(img, repo=repo, channel="web", participant_id="test-user-1")

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

    with patch("src.ingest.analyze.analyze_image_async", new=AsyncMock(return_value=mock1)):
        result1 = await run_session_graph(img, repo=repo, channel="web", participant_id="test-user-1")

    assert result1.status == SessionStatus.ACTIVE
    assert result1.found_count == 2
    assert result1.missing_count == 1

    with patch("src.ingest.analyze.analyze_image_async", new=AsyncMock(return_value=mock2)):
        result2 = await run_session_graph(img, repo=repo, channel="web", participant_id="test-user-1")

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

    with patch("src.ingest.analyze.analyze_image_async", new=AsyncMock(return_value=mock1)):
        result1 = await run_session_graph(img, repo=repo, channel="web", participant_id="test-user-1")

    assert result1.status == SessionStatus.COMPLETE
    session1 = result1.session_id

    # Session is complete — second photo starts a NEW session (not rejected).
    # The old session is immutable; the new photo begins fresh accumulation.
    with patch("src.ingest.analyze.analyze_image_async", new=AsyncMock(return_value=mock2)):
        result2 = await run_session_graph(img, repo=repo, channel="web", participant_id="test-user-1")

    assert result2.session_id != session1  # new session
    assert result2.status == SessionStatus.COMPLETE
    assert result2.found_count == 2  # fresh session, 2 found in this photo
    assert result2.image_count == 1  # first image of new session


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

    with patch("src.ingest.analyze.analyze_image_async", new=AsyncMock(return_value=mock1)):
        result1 = await run_session_graph(img, repo=repo, channel="web", participant_id="test-user-1")

    assert result1.status == SessionStatus.ACTIVE
    assert result1.found_count == 3

    with patch("src.ingest.analyze.analyze_image_async", new=AsyncMock(return_value=mock2)):
        result2 = await run_session_graph(img, repo=repo, channel="web", participant_id="test-user-1")

    assert result2.status == SessionStatus.COMPLETE
    assert result2.found_count == 4  # 3 original + 1 new (deduped)
    assert result2.missing_count == 0
    assert len(result2.items) == 4


@pytest.mark.asyncio
async def test_session_ambiguous_more_new_than_missing(tmp_path: Path) -> None:
    """Photo #2 has 2 new barcodes but only 1 missing → ask user to pick."""
    repo = NoOpSessionRepository()
    img = tmp_path / "img.png"
    img.write_bytes(b"fake")

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

    mock2 = {
        "outcome": "complete",
        "audit_available": True,
        "found": [
            {"barcode_value": "444", "label_index": 1},  # new
            {"barcode_value": "555", "label_index": 2},  # new — ambiguous!
        ],
        "missing": [],
        "unassigned": [],
        "summary": {"visible_label_count": 2, "found_count": 2, "missing_count": 0},
    }

    with patch("src.ingest.analyze.analyze_image_async", new=AsyncMock(return_value=mock1)):
        result1 = await run_session_graph(img, repo=repo, channel="web", participant_id="test-user-1")

    assert result1.status == SessionStatus.ACTIVE
    assert result1.found_count == 3
    assert result1.missing_count == 1

    with patch("src.ingest.analyze.analyze_image_async", new=AsyncMock(return_value=mock2)):
        result2 = await run_session_graph(img, repo=repo, channel="web", participant_id="test-user-1")

    # 2 new barcodes, 1 missing → ambiguous, ask user.
    assert result2.status == SessionStatus.NEEDS_USER_SELECTION
    assert result2.found_count == 3  # unchanged — nothing added
    assert result2.missing_count == 1  # still missing
    assert len(result2.candidates) == 2
    candidate_barcodes = {c.barcode_value for c in result2.candidates}
    assert candidate_barcodes == {"444", "555"}


@pytest.mark.asyncio
async def test_session_9_of_12_then_6_new_boxes_should_ask(tmp_path: Path) -> None:
    """Photo 1: 9/12 (3 missing). Photo 2: 6 boxes, 4 new → should ask, not complete."""
    repo = NoOpSessionRepository()
    img = tmp_path / "img.png"
    img.write_bytes(b"fake")

    mock1 = {
        "outcome": "needs_better_photo",
        "audit_available": True,
        "found": [
            {"barcode_value": "111", "label_index": 1},
            {"barcode_value": "222", "label_index": 2},
            {"barcode_value": "333", "label_index": 3},
            {"barcode_value": "444", "label_index": 4},
            {"barcode_value": "555", "label_index": 5},
            {"barcode_value": "666", "label_index": 6},
            {"barcode_value": "777", "label_index": 7},
            {"barcode_value": "888", "label_index": 8},
            {"barcode_value": "999", "label_index": 9},
        ],
        "missing": [
            {"label_index": 10, "status": "not_visible", "label_bbox": {}, "barcode_bbox": {}},
            {"label_index": 11, "status": "not_visible", "label_bbox": {}, "barcode_bbox": {}},
            {"label_index": 12, "status": "not_visible", "label_bbox": {}, "barcode_bbox": {}},
        ],
        "unassigned": [],
        "summary": {"visible_label_count": 12, "found_count": 9, "missing_count": 3},
    }

    # Photo 2: 6 boxes. 2 already known + 4 new. 4 > 3 missing → ambiguous.
    mock2 = {
        "outcome": "complete",
        "audit_available": True,
        "found": [
            {"barcode_value": "111", "label_index": 1},  # already known
            {"barcode_value": "222", "label_index": 2},  # already known
            {"barcode_value": "AAA", "label_index": 3},  # new
            {"barcode_value": "BBB", "label_index": 4},  # new
            {"barcode_value": "CCC", "label_index": 5},  # new
            {"barcode_value": "DDD", "label_index": 6},  # new
        ],
        "missing": [],
        "unassigned": [],
        "summary": {"visible_label_count": 6, "found_count": 6, "missing_count": 0},
    }

    with patch("src.ingest.analyze.analyze_image_async", new=AsyncMock(return_value=mock1)):
        result1 = await run_session_graph(img, repo=repo, channel="web", participant_id="test-user-1")
    assert result1.status == SessionStatus.ACTIVE
    assert result1.found_count == 9
    assert result1.missing_count == 3

    with patch("src.ingest.analyze.analyze_image_async", new=AsyncMock(return_value=mock2)):
        result2 = await run_session_graph(img, repo=repo, channel="web", participant_id="test-user-1")

    # 4 new > 3 missing → should ask, not complete
    assert result2.status == SessionStatus.NEEDS_USER_SELECTION
    assert result2.found_count == 9  # unchanged
    assert result2.missing_count == 3  # still missing
    assert len(result2.candidates) == 4
    candidate_barcodes = {c.barcode_value for c in result2.candidates}
    assert candidate_barcodes == {"AAA", "BBB", "CCC", "DDD"}


@pytest.mark.asyncio
async def test_session_9_of_12_then_6_boxes_3_new_3_known(tmp_path: Path) -> None:
    """Photo 1: 9/12 (3 missing). Photo 2: 6 boxes, 3 known + 3 new → accept, complete.

    3 new == 3 missing → exact match → accept all, resolve all → complete.
    This is correct: the 3 new barcodes resolve the 3 missing labels.
    """
    repo = NoOpSessionRepository()
    img = tmp_path / "img.png"
    img.write_bytes(b"fake")

    mock1 = {
        "outcome": "needs_better_photo",
        "audit_available": True,
        "found": [
            {"barcode_value": "111", "label_index": 1},
            {"barcode_value": "222", "label_index": 2},
            {"barcode_value": "333", "label_index": 3},
            {"barcode_value": "444", "label_index": 4},
            {"barcode_value": "555", "label_index": 5},
            {"barcode_value": "666", "label_index": 6},
            {"barcode_value": "777", "label_index": 7},
            {"barcode_value": "888", "label_index": 8},
            {"barcode_value": "999", "label_index": 9},
        ],
        "missing": [
            {"label_index": 10, "status": "not_visible", "label_bbox": {}, "barcode_bbox": {}},
            {"label_index": 11, "status": "not_visible", "label_bbox": {}, "barcode_bbox": {}},
            {"label_index": 12, "status": "not_visible", "label_bbox": {}, "barcode_bbox": {}},
        ],
        "unassigned": [],
        "summary": {"visible_label_count": 12, "found_count": 9, "missing_count": 3},
    }

    # Photo 2: 6 boxes. 3 already known + 3 new. 3 == 3 missing → accept.
    mock2 = {
        "outcome": "complete",
        "audit_available": True,
        "found": [
            {"barcode_value": "111", "label_index": 1},  # already known
            {"barcode_value": "222", "label_index": 2},  # already known
            {"barcode_value": "333", "label_index": 3},  # already known
            {"barcode_value": "AAA", "label_index": 4},  # new
            {"barcode_value": "BBB", "label_index": 5},  # new
            {"barcode_value": "CCC", "label_index": 6},  # new
        ],
        "missing": [],
        "unassigned": [],
        "summary": {"visible_label_count": 6, "found_count": 6, "missing_count": 0},
    }

    with patch("src.ingest.analyze.analyze_image_async", new=AsyncMock(return_value=mock1)):
        result1 = await run_session_graph(img, repo=repo, channel="web", participant_id="test-user-1")
    assert result1.status == SessionStatus.ACTIVE
    assert result1.found_count == 9

    with patch("src.ingest.analyze.analyze_image_async", new=AsyncMock(return_value=mock2)):
        result2 = await run_session_graph(img, repo=repo, channel="web", participant_id="test-user-1")

    # 3 new == 3 missing → exact match → accept all → complete
    assert result2.status == SessionStatus.COMPLETE
    assert result2.found_count == 12  # 9 + 3
    assert result2.missing_count == 0


@pytest.mark.asyncio
async def test_session_dedup_same_barcode_not_ambiguous(tmp_path: Path) -> None:
    """Missing 1, photo has 2 new but same value (same product, 2 boxes) → accept, don't ask."""
    repo = NoOpSessionRepository()
    img = tmp_path / "img.png"
    img.write_bytes(b"fake")

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

    # Photo #2: 3 found — 1 known + 2 new (same barcode value, 2 boxes of same product)
    mock2 = {
        "outcome": "complete",
        "audit_available": True,
        "found": [
            {"barcode_value": "222", "label_index": 1},  # already known (neighbor)
            {"barcode_value": "444", "label_index": 2},  # new — resolves missing
            {"barcode_value": "444", "label_index": 3},  # same value, different box
        ],
        "missing": [],
        "unassigned": [],
        "summary": {"visible_label_count": 3, "found_count": 3, "missing_count": 0},
    }

    with patch("src.ingest.analyze.analyze_image_async", new=AsyncMock(return_value=mock1)):
        result1 = await run_session_graph(img, repo=repo, channel="web", participant_id="test-user-1")
    assert result1.status == SessionStatus.ACTIVE
    assert result1.missing_count == 1

    with patch("src.ingest.analyze.analyze_image_async", new=AsyncMock(return_value=mock2)):
        result2 = await run_session_graph(img, repo=repo, channel="web", participant_id="test-user-1")

    # 2 new found but same value → 1 unique new → matches 1 missing → accept
    assert result2.status == SessionStatus.COMPLETE
    assert result2.found_count == 4  # 3 original + 1 new unique
    assert result2.missing_count == 0
    assert len(result2.candidates) == 0  # no ambiguity


@pytest.mark.asyncio
async def test_session_multi_rephoto_two_missing(tmp_path: Path) -> None:
    """2 missing → photo #2 resolves 1 → still active → photo #3 resolves the other → complete."""
    repo = NoOpSessionRepository()
    img = tmp_path / "img.png"
    img.write_bytes(b"fake")

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
            {"label_index": 5, "status": "not_visible", "label_bbox": {}, "barcode_bbox": {}},
        ],
        "unassigned": [],
        "summary": {"visible_label_count": 5, "found_count": 3, "missing_count": 2},
    }

    # Photo #2: only 1 of the 2 missing boxes (user photographed just one)
    mock2 = {
        "outcome": "complete",
        "audit_available": True,
        "found": [
            {"barcode_value": "444", "label_index": 1},  # resolves 1 missing
        ],
        "missing": [],
        "unassigned": [],
        "summary": {"visible_label_count": 1, "found_count": 1, "missing_count": 0},
    }

    # Photo #3: the other missing box
    mock3 = {
        "outcome": "complete",
        "audit_available": True,
        "found": [
            {"barcode_value": "555", "label_index": 1},  # resolves the last missing
        ],
        "missing": [],
        "unassigned": [],
        "summary": {"visible_label_count": 1, "found_count": 1, "missing_count": 0},
    }

    with patch("src.ingest.analyze.analyze_image_async", new=AsyncMock(return_value=mock1)):
        result1 = await run_session_graph(img, repo=repo, channel="web", participant_id="test-user-1")
    assert result1.status == SessionStatus.ACTIVE
    assert result1.found_count == 3
    assert result1.missing_count == 2

    with patch("src.ingest.analyze.analyze_image_async", new=AsyncMock(return_value=mock2)):
        result2 = await run_session_graph(img, repo=repo, channel="web", participant_id="test-user-1")
    assert result2.status == SessionStatus.ACTIVE  # still 1 missing
    assert result2.found_count == 4
    assert result2.missing_count == 1

    with patch("src.ingest.analyze.analyze_image_async", new=AsyncMock(return_value=mock3)):
        result3 = await run_session_graph(img, repo=repo, channel="web", participant_id="test-user-1")
    assert result3.status == SessionStatus.COMPLETE
    assert result3.found_count == 5
    assert result3.missing_count == 0
    assert result3.image_count == 3


@pytest.mark.asyncio
async def test_session_expected_count_updates_on_more_labels(tmp_path: Path) -> None:
    """Photo #2 shows more visible labels than #1 → expected_count updated."""
    repo = NoOpSessionRepository()
    img = tmp_path / "img.png"
    img.write_bytes(b"fake")

    # Photo #1: 4 visible, 3 found, 1 missing
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

    # Photo #2: 6 visible (user stepped back, saw 2 more boxes), 1 new barcode
    mock2 = {
        "outcome": "needs_better_photo",
        "audit_available": True,
        "found": [
            {"barcode_value": "444", "label_index": 1},  # resolves the 1 missing
        ],
        "missing": [
            {"label_index": 5, "status": "not_visible", "label_bbox": {}, "barcode_bbox": {}},
            {"label_index": 6, "status": "not_visible", "label_bbox": {}, "barcode_bbox": {}},
        ],
        "unassigned": [],
        "summary": {"visible_label_count": 6, "found_count": 1, "missing_count": 2},
    }

    with patch("src.ingest.analyze.analyze_image_async", new=AsyncMock(return_value=mock1)):
        result1 = await run_session_graph(img, repo=repo, channel="web", participant_id="test-user-1")
    assert result1.status == SessionStatus.ACTIVE
    assert result1.expected_count == 4
    assert result1.found_count == 3
    assert result1.missing_count == 1

    with patch("src.ingest.analyze.analyze_image_async", new=AsyncMock(return_value=mock2)):
        result2 = await run_session_graph(img, repo=repo, channel="web", participant_id="test-user-1")
    # expected_count updated from 4 → 6 (photo #2 saw more labels)
    assert result2.expected_count == 6
    # found 4 (3 + 1 new), missing 2 (the 2 new ones from the wider view)
    assert result2.found_count == 4
    assert result2.missing_count == 2
    assert result2.status == SessionStatus.ACTIVE


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

    with patch("src.ingest.analyze.analyze_image_async", new=AsyncMock(return_value=mock_result)):
        result = await run_session_graph(img, repo=repo, channel="web", participant_id="test-user-1")

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

    with patch("src.ingest.analyze.analyze_image_async", new=AsyncMock(return_value=mock_fail)):
        result1 = await run_session_graph(img, repo=repo, channel="web", participant_id="test-user-1")

    assert result1.status == SessionStatus.FAILED

    # Second image — session was never created (failed), so this is a new session.
    with patch("src.ingest.analyze.analyze_image_async", new=AsyncMock(return_value=mock_ok)):
        result2 = await run_session_graph(img, repo=repo, channel="web", participant_id="test-user-1")

    assert result2.status == SessionStatus.COMPLETE
    assert result2.found_count == 2


# ---------------------------------------------------------------------------
# Session lifecycle tests (M16C)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_reject_photo_on_complete(tmp_path: Path) -> None:
    """Complete session + new photo → rejected, not silently reopened."""
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

    with patch("src.ingest.analyze.analyze_image_async", new=AsyncMock(return_value=mock1)):
        result1 = await run_session_graph(img, repo=repo, channel="web", participant_id="test-user-1")

    assert result1.status == SessionStatus.COMPLETE
    session1 = result1.session_id

    # Session is complete — next photo starts a NEW session automatically.
    mock2 = {
        "outcome": "complete",
        "audit_available": True,
        "found": [{"barcode_value": "333", "label_index": 3}],
        "missing": [],
        "unassigned": [],
        "summary": {"visible_label_count": 1, "found_count": 1, "missing_count": 0},
    }

    with patch("src.ingest.analyze.analyze_image_async", new=AsyncMock(return_value=mock2)):
        result2 = await run_session_graph(img, repo=repo, channel="web", participant_id="test-user-1")

    assert result2.session_id != session1  # new session
    assert result2.status == SessionStatus.COMPLETE
    assert result2.found_count == 1  # fresh session
    assert result2.image_count == 1  # first image of new session


@pytest.mark.asyncio
async def test_session_close_then_new_session(tmp_path: Path) -> None:
    """Closed session → next photo starts a new session."""
    repo = NoOpSessionRepository()
    img = tmp_path / "img.png"
    img.write_bytes(b"fake")

    mock1 = {
        "outcome": "needs_better_photo",
        "audit_available": True,
        "found": [{"barcode_value": "111", "label_index": 1}],
        "missing": [
            {"label_index": 2, "status": "not_visible",
             "label_bbox": {}, "barcode_bbox": {}},
        ],
        "unassigned": [],
        "summary": {"visible_label_count": 2, "found_count": 1, "missing_count": 1},
    }

    with patch("src.ingest.analyze.analyze_image_async", new=AsyncMock(return_value=mock1)):
        result1 = await run_session_graph(img, repo=repo, channel="web", participant_id="test-user-1")

    assert result1.status == SessionStatus.ACTIVE
    session1 = result1.session_id

    # Close the session.
    closed = await repo.close_session(session1)
    assert closed is True

    # Next photo — closed session not found → new session created.
    with patch("src.ingest.analyze.analyze_image_async", new=AsyncMock(return_value=mock1)):
        result2 = await run_session_graph(img, repo=repo, channel="web", participant_id="test-user-1")

    assert result2.session_id != session1  # new session
    assert result2.status == SessionStatus.ACTIVE
    assert result2.found_count == 1  # fresh session
    assert result2.image_count == 1  # first image of new session


@pytest.mark.asyncio
async def test_session_lazy_expiry(tmp_path: Path) -> None:
    """Session that's been inactive past TTL is expired on next access."""
    from datetime import datetime, timedelta

    repo = NoOpSessionRepository()
    img = tmp_path / "img.png"
    img.write_bytes(b"fake")

    mock1 = {
        "outcome": "needs_better_photo",
        "audit_available": True,
        "found": [{"barcode_value": "111", "label_index": 1}],
        "missing": [
            {"label_index": 2, "status": "not_visible",
             "label_bbox": {}, "barcode_bbox": {}},
        ],
        "unassigned": [],
        "summary": {"visible_label_count": 2, "found_count": 1, "missing_count": 1},
    }

    with patch("src.ingest.analyze.analyze_image_async", new=AsyncMock(return_value=mock1)):
        result1 = await run_session_graph(img, repo=repo, channel="web", participant_id="test-user-1")

    assert result1.status == SessionStatus.ACTIVE
    session1 = result1.session_id

    # Simulate the session being old — manually set last_activity_at in the past.
    old_time = datetime.now(UTC) - timedelta(minutes=31)
    repo._sessions[session1]["last_activity_at"] = old_time

    # Next access — session is found (still 'active' in DB) but lazy expiry
    # check marks it expired and rejects the photo. A new session is created.
    with patch("src.ingest.analyze.analyze_image_async", new=AsyncMock(return_value=mock1)):
        result2 = await run_session_graph(img, repo=repo, channel="web", participant_id="test-user-1")

    # The old session should now be expired.
    assert repo._sessions[session1]["status"] == "expired"
    # A new session was created for this photo.
    assert result2.session_id != session1
    assert result2.status == SessionStatus.ACTIVE
    assert result2.found_count == 1  # fresh session
    assert result2.image_count == 1  # first image of new session


# ---------------------------------------------------------------------------
# WhatsApp session tests (M16C)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_whatsapp_session_resolved_by_sender(tmp_path: Path) -> None:
    """WhatsApp: session_id resolved by sender, not sent by client."""
    repo = NoOpSessionRepository()
    img = tmp_path / "img.png"
    img.write_bytes(b"fake")

    mock1 = {
        "outcome": "needs_better_photo",
        "audit_available": True,
        "found": [{"barcode_value": "111", "label_index": 1}],
        "missing": [
            {"label_index": 2, "status": "not_visible",
             "label_bbox": {}, "barcode_bbox": {}},
        ],
        "unassigned": [],
        "summary": {"visible_label_count": 2, "found_count": 1, "missing_count": 1},
    }

    mock2 = {
        "outcome": "complete",
        "audit_available": True,
        "found": [{"barcode_value": "222", "label_index": 2}],
        "missing": [],
        "unassigned": [],
        "summary": {"visible_label_count": 1, "found_count": 1, "missing_count": 0},
    }

    # First photo — no session_id, resolved by sender.
    with patch("src.ingest.analyze.analyze_image_async", new=AsyncMock(return_value=mock1)):
        result1 = await run_session_graph(
            img, repo=repo, channel="whatsapp", participant_id="+972501234567"
        )

    assert result1.status == SessionStatus.ACTIVE
    assert result1.found_count == 1
    session_id = result1.session_id
    assert session_id is not None

    # Second photo — also no session_id, should resolve to same session.
    with patch("src.ingest.analyze.analyze_image_async", new=AsyncMock(return_value=mock2)):
        result2 = await run_session_graph(
            img, repo=repo, channel="whatsapp", participant_id="+972501234567"
        )

    assert result2.session_id == session_id  # same session
    assert result2.status == SessionStatus.COMPLETE
    assert result2.found_count == 2
    assert result2.image_count == 2


@pytest.mark.asyncio
async def test_whatsapp_complete_starts_new_session(tmp_path: Path) -> None:
    """WhatsApp: when session is complete, next photo starts a new session."""
    repo = NoOpSessionRepository()
    img = tmp_path / "img.png"
    img.write_bytes(b"fake")

    mock_complete = {
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

    # First photo — completes the session.
    with patch("src.ingest.analyze.analyze_image_async", new=AsyncMock(return_value=mock_complete)):
        result1 = await run_session_graph(
            img, repo=repo, channel="whatsapp", participant_id="+972501234567"
        )

    assert result1.status == SessionStatus.COMPLETE
    session1 = result1.session_id

    # Second photo — session is complete, should start a new session.
    with patch("src.ingest.analyze.analyze_image_async", new=AsyncMock(return_value=mock_complete)):
        result2 = await run_session_graph(
            img, repo=repo, channel="whatsapp", participant_id="+972501234567"
        )

    assert result2.session_id != session1  # new session
    assert result2.status == SessionStatus.COMPLETE
    assert result2.found_count == 2
    assert result2.image_count == 1  # first image of new session


@pytest.mark.asyncio
async def test_whatsapp_different_senders_different_sessions(tmp_path: Path) -> None:
    """WhatsApp: different senders get different sessions."""
    repo = NoOpSessionRepository()
    img = tmp_path / "img.png"
    img.write_bytes(b"fake")

    mock = {
        "outcome": "needs_better_photo",
        "audit_available": True,
        "found": [{"barcode_value": "111", "label_index": 1}],
        "missing": [
            {"label_index": 2, "status": "not_visible",
             "label_bbox": {}, "barcode_bbox": {}},
        ],
        "unassigned": [],
        "summary": {"visible_label_count": 2, "found_count": 1, "missing_count": 1},
    }

    with patch("src.ingest.analyze.analyze_image_async", new=AsyncMock(return_value=mock)):
        result1 = await run_session_graph(
            img, repo=repo, channel="whatsapp", participant_id="+972111111111"
        )
        result2 = await run_session_graph(
            img, repo=repo, channel="whatsapp", participant_id="+972222222222"
        )

    assert result1.session_id != result2.session_id
    assert result1.status == SessionStatus.ACTIVE
    assert result2.status == SessionStatus.ACTIVE


# ---------------------------------------------------------------------------
# User selection tests (M16E)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_select_candidate_resolves_missing(tmp_path: Path) -> None:
    """User selects a candidate → resolves missing → complete."""
    repo = NoOpSessionRepository()
    img = tmp_path / "img.png"
    img.write_bytes(b"fake")

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

    mock2 = {
        "outcome": "complete",
        "audit_available": True,
        "found": [
            {"barcode_value": "444", "label_index": 1},
            {"barcode_value": "555", "label_index": 2},
        ],
        "missing": [],
        "unassigned": [],
        "summary": {"visible_label_count": 2, "found_count": 2, "missing_count": 0},
    }

    with patch("src.ingest.analyze.analyze_image_async", new=AsyncMock(return_value=mock1)):
        result1 = await run_session_graph(img, repo=repo, channel="web", participant_id="test-user-1")
    assert result1.status == SessionStatus.ACTIVE
    assert result1.missing_count == 1

    with patch("src.ingest.analyze.analyze_image_async", new=AsyncMock(return_value=mock2)):
        result2 = await run_session_graph(img, repo=repo, channel="web", participant_id="test-user-1")
    assert result2.status == SessionStatus.NEEDS_USER_SELECTION
    assert len(result2.candidates) == 2
    session_id = result2.session_id

    # User selects "444"
    result3 = await select_candidate(session_id, "444", repo=repo)
    assert result3.status == SessionStatus.COMPLETE
    assert result3.found_count == 4  # 3 + 1 selected
    assert result3.missing_count == 0
    assert len(result3.candidates) == 0


@pytest.mark.asyncio
async def test_select_candidate_wrong_barcode_raises(tmp_path: Path) -> None:
    """Selecting a barcode not in candidates raises ValueError."""
    repo = NoOpSessionRepository()
    img = tmp_path / "img.png"
    img.write_bytes(b"fake")

    mock1 = {
        "outcome": "needs_better_photo",
        "audit_available": True,
        "found": [{"barcode_value": "111", "label_index": 1}],
        "missing": [
            {"label_index": 2, "status": "not_visible", "label_bbox": {}, "barcode_bbox": {}},
        ],
        "unassigned": [],
        "summary": {"visible_label_count": 2, "found_count": 1, "missing_count": 1},
    }

    mock2 = {
        "outcome": "complete",
        "audit_available": True,
        "found": [
            {"barcode_value": "444", "label_index": 1},
            {"barcode_value": "555", "label_index": 2},
        ],
        "missing": [],
        "unassigned": [],
        "summary": {"visible_label_count": 2, "found_count": 2, "missing_count": 0},
    }

    with patch("src.ingest.analyze.analyze_image_async", new=AsyncMock(return_value=mock1)):
        await run_session_graph(img, repo=repo, channel="web", participant_id="test-user-1")

    with patch("src.ingest.analyze.analyze_image_async", new=AsyncMock(return_value=mock2)):
        result2 = await run_session_graph(img, repo=repo, channel="web", participant_id="test-user-1")
    session_id = result2.session_id

    # Select a barcode not in candidates
    with pytest.raises(ValueError, match="not in candidates"):
        await select_candidate(session_id, "999", repo=repo)


@pytest.mark.asyncio
async def test_select_candidate_wrong_status_raises(tmp_path: Path) -> None:
    """Selecting on a session not in needs_user_selection raises ValueError."""
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

    with patch("src.ingest.analyze.analyze_image_async", new=AsyncMock(return_value=mock1)):
        result1 = await run_session_graph(img, repo=repo, channel="web", participant_id="test-user-1")
    # Session is complete, not needs_user_selection
    with pytest.raises(ValueError, match="not awaiting selection"):
        await select_candidate(result1.session_id, "111", repo=repo)

"""Tests for the SQLite annotation store — CRUD, idempotency, and export.

No network, no Gemini. Uses a temp DB path for isolation.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from src.evals.annotation_store import (
    AnnotationCandidate,
    create_candidate,
    export_to_dataset_json,
    get_candidate,
    get_stats,
    init_db,
    list_pending,
    list_reviewed,
    submit_review,
)


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    """Isolated temp DB for each test."""
    path = str(tmp_path / "test_annotations.db")
    init_db(path)
    return path


@pytest.fixture
def dataset_path(tmp_path: Path) -> Path:
    """Minimal dataset.json for export tests."""
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps({
        "summary": {
            "description": "test",
            "image_count": 0,
            "expected_barcode_symbol_count": 0,
            "expected_decoded_count": 0,
            "expected_outcome": "complete",
            "images": [],
        },
        "images": [],
    }))
    return path


@pytest.fixture
def samples_dir(tmp_path: Path) -> Path:
    d = tmp_path / "samples"
    d.mkdir()
    return d


def _make_candidate(
    run_id: str | None = None,
    image_ref: str | None = "/tmp/fake.jpg",
    status: str = "needs_user_input",
) -> AnnotationCandidate:
    return AnnotationCandidate(
        id=str(uuid4()),
        run_id=run_id or str(uuid4()),
        session_id="test-session",
        source="cli",
        image_ref=image_ref,
        status=status,
        scanner_count=11,
        vision_count=12,
        recovery_attempted=True,
        recovery_labels_resolved=0,
        issues=[{"code": "MISSING_LABELS", "severity": "warning", "message": "1 missing"}],
    )


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


def test_create_and_get_candidate(db_path: str) -> None:
    cand = _make_candidate()
    created = create_candidate(cand, db_path=db_path)
    assert created is True

    fetched = get_candidate(cand.id, db_path=db_path)
    assert fetched is not None
    assert fetched.run_id == cand.run_id
    assert fetched.status == cand.status
    assert fetched.scanner_count == cand.scanner_count
    assert fetched.issues == cand.issues


def test_list_pending_returns_unreviewed(db_path: str) -> None:
    c1 = _make_candidate()
    c2 = _make_candidate()
    create_candidate(c1, db_path=db_path)
    create_candidate(c2, db_path=db_path)

    pending = list_pending(db_path=db_path)
    assert len(pending) == 2
    # Newest first
    assert pending[0].created_at >= pending[1].created_at


def test_list_pending_excludes_reviewed(db_path: str) -> None:
    cand = _make_candidate()
    create_candidate(cand, db_path=db_path)
    submit_review(
        cand.id, [{"value": "123", "format": "Code128"}], "complete", "tester", db_path=db_path
    )

    pending = list_pending(db_path=db_path)
    assert len(pending) == 0


def test_submit_review_sets_fields(db_path: str) -> None:
    cand = _make_candidate()
    create_candidate(cand, db_path=db_path)

    ok = submit_review(
        cand.id,
        [{"value": "7297501154117", "format": "Code128"}],
        "complete",
        "gaddy",
        db_path=db_path,
    )
    assert ok is True

    fetched = get_candidate(cand.id, db_path=db_path)
    assert fetched is not None
    assert fetched.reviewed_at is not None
    assert fetched.reviewed_by == "gaddy"
    assert fetched.expected_outcome == "complete"
    assert fetched.expected_barcodes == [{"value": "7297501154117", "format": "Code128"}]


def test_submit_review_already_reviewed_returns_false(db_path: str) -> None:
    cand = _make_candidate()
    create_candidate(cand, db_path=db_path)
    submit_review(cand.id, [{"value": "123"}], "complete", "tester", db_path=db_path)

    # Second review attempt should fail.
    ok = submit_review(cand.id, [{"value": "456"}], "complete", "other", db_path=db_path)
    assert ok is False


def test_get_candidate_not_found(db_path: str) -> None:
    assert get_candidate("nonexistent", db_path=db_path) is None


def test_get_stats(db_path: str) -> None:
    c1 = _make_candidate()
    c2 = _make_candidate()
    c3 = _make_candidate()
    create_candidate(c1, db_path=db_path)
    create_candidate(c2, db_path=db_path)
    create_candidate(c3, db_path=db_path)

    submit_review(c1.id, [{"value": "123"}], "complete", "tester", db_path=db_path)

    stats = get_stats(db_path=db_path)
    assert stats["pending"] == 2
    assert stats["reviewed"] == 1
    assert stats["exported"] == 0
    assert stats["total"] == 3


# ---------------------------------------------------------------------------
# Idempotency — duplicate run_id
# ---------------------------------------------------------------------------


def test_duplicate_run_id_is_ignored(db_path: str) -> None:
    """Emit same INGEST_COMPLETED event twice → one SQLite row."""
    run_id = str(uuid4())
    cand = _make_candidate(run_id=run_id)

    first = create_candidate(cand, db_path=db_path)
    second = create_candidate(cand, db_path=db_path)

    assert first is True
    assert second is False  # INSERT OR IGNORE

    pending = list_pending(db_path=db_path)
    assert len(pending) == 1


def test_duplicate_run_id_one_dataset_entry(
    db_path: str, dataset_path: Path, samples_dir: Path, tmp_path: Path
) -> None:
    """Full loop: duplicate event → one row → review → export → one dataset entry."""
    run_id = str(uuid4())

    # Create a fake image file.
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"fake jpeg")

    cand = _make_candidate(run_id=run_id, image_ref=str(image))
    create_candidate(cand, db_path=db_path)
    create_candidate(cand, db_path=db_path)  # duplicate

    # Review.
    submit_review(
        cand.id,
        [{"value": "7297501154117", "format": "Code128"}],
        "complete",
        "tester",
        db_path=db_path,
    )

    # Export.
    count = export_to_dataset_json(
        db_path=db_path, dataset_path=dataset_path, samples_dir=samples_dir
    )
    assert count == 1

    # Verify dataset.json has one new entry.
    with dataset_path.open() as f:
        data = json.load(f)
    assert len(data["images"]) == 1
    assert data["images"][0]["annotation_run_id"] == run_id

    # Verify the image was copied.
    expected_image = samples_dir / f"annotation_{run_id}.jpg"
    assert expected_image.exists()

    # Verify candidate is marked exported.
    fetched = get_candidate(cand.id, db_path=db_path)
    assert fetched is not None
    assert fetched.added_to_dataset is True


# ---------------------------------------------------------------------------
# Failed export — missing image_ref
# ---------------------------------------------------------------------------


def test_export_skips_missing_image_ref(
    db_path: str, dataset_path: Path, samples_dir: Path
) -> None:
    """Reviewed candidate with image_ref=None → export skips it → added_to_dataset stays false."""
    cand = _make_candidate(image_ref=None)
    create_candidate(cand, db_path=db_path)
    submit_review(cand.id, [{"value": "123"}], "complete", "tester", db_path=db_path)

    count = export_to_dataset_json(
        db_path=db_path, dataset_path=dataset_path, samples_dir=samples_dir
    )
    assert count == 0

    # Candidate should still be eligible for export.
    fetched = get_candidate(cand.id, db_path=db_path)
    assert fetched is not None
    assert fetched.added_to_dataset is False

    # Dataset unchanged.
    with dataset_path.open() as f:
        data = json.load(f)
    assert len(data["images"]) == 0


def test_export_skips_nonexistent_image_file(
    db_path: str, dataset_path: Path, samples_dir: Path
) -> None:
    """Reviewed candidate with image_ref pointing to missing file → skipped."""
    cand = _make_candidate(image_ref="/nonexistent/path.jpg")
    create_candidate(cand, db_path=db_path)
    submit_review(cand.id, [{"value": "123"}], "complete", "tester", db_path=db_path)

    count = export_to_dataset_json(
        db_path=db_path, dataset_path=dataset_path, samples_dir=samples_dir
    )
    assert count == 0

    fetched = get_candidate(cand.id, db_path=db_path)
    assert fetched is not None
    assert fetched.added_to_dataset is False


# ---------------------------------------------------------------------------
# Export — summary recalculation
# ---------------------------------------------------------------------------


def test_export_recalculates_summary(
    db_path: str, dataset_path: Path, samples_dir: Path, tmp_path: Path
) -> None:
    """Export updates the summary block with new counts."""
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"fake jpeg")

    cand = _make_candidate(image_ref=str(image))
    create_candidate(cand, db_path=db_path)
    submit_review(
        cand.id,
        [{"value": "111", "format": "Code128"}, {"value": "222", "format": "Code128"}],
        "complete",
        "tester",
        db_path=db_path,
    )

    export_to_dataset_json(
        db_path=db_path, dataset_path=dataset_path, samples_dir=samples_dir
    )

    with dataset_path.open() as f:
        data = json.load(f)
    assert data["summary"]["image_count"] == 1
    assert data["summary"]["expected_barcode_symbol_count"] == 2
    assert data["summary"]["expected_decoded_count"] == 2
    assert len(data["summary"]["images"]) == 1


def test_export_no_reviewed_candidates_returns_zero(
    db_path: str, dataset_path: Path, samples_dir: Path
) -> None:
    """No reviewed candidates → export returns 0."""
    count = export_to_dataset_json(
        db_path=db_path, dataset_path=dataset_path, samples_dir=samples_dir
    )
    assert count == 0


def test_list_reviewed_excludes_exported(
    db_path: str, dataset_path: Path, samples_dir: Path, tmp_path: Path
) -> None:
    """After export, reviewed candidates are no longer in list_reviewed."""
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"fake jpeg")

    cand = _make_candidate(image_ref=str(image))
    create_candidate(cand, db_path=db_path)
    submit_review(cand.id, [{"value": "123"}], "complete", "tester", db_path=db_path)

    assert len(list_reviewed(db_path=db_path)) == 1

    export_to_dataset_json(
        db_path=db_path, dataset_path=dataset_path, samples_dir=samples_dir
    )

    assert len(list_reviewed(db_path=db_path)) == 0

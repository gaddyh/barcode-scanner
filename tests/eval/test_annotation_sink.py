"""Tests for the AnnotationCandidateSink — interesting detection and routing.

No network, no Gemini. Uses a temp DB path via env var override.
"""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import pytest

from src.evals.annotation_sink import (
    AnnotationCandidateSink,
    _is_interesting,
    register_annotation_sink,
)
from src.evals.annotation_store import list_pending
from src.observability.event_sink import get_sinks, reset_sinks
from src.runtime.events import DomainEvent, EventType


@pytest.fixture(autouse=True)
def _isolate_sinks_and_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Reset sinks and use a temp DB for each test."""
    reset_sinks()
    db_path = str(tmp_path / "test_sink.db")
    monkeypatch.setenv("ANNOTATION_DB_PATH", db_path)
    yield
    reset_sinks()


def _make_ingest_completed_event(
    run_id: str | None = None,
    status: str = "complete",
    scanner_count: int = 6,
    vision_count: int = 6,
    recovery_attempted: bool = False,
    recovery_labels_resolved: int = 0,
    image_ref: str | None = None,
) -> DomainEvent:
    return DomainEvent(
        type=EventType.INGEST_COMPLETED,
        run_id=run_id or str(uuid4()),
        session_id="test-session",
        payload={
            "status": status,
            "source": "cli",
            "image_ref": image_ref,
            "found_count": scanner_count,
            "missing_count": 0,
            "scanner_count": scanner_count,
            "vision_count": vision_count,
            "recovery_attempted": recovery_attempted,
            "recovery_labels_resolved": recovery_labels_resolved,
            "elapsed_ms": 3000,
            "issues": [],
        },
    )


# ---------------------------------------------------------------------------
# Interesting failure rules
# ---------------------------------------------------------------------------


def test_complete_not_interesting() -> None:
    assert not _is_interesting({
        "status": "complete",
        "scanner_count": 6,
        "vision_count": 6,
        "recovery_attempted": False,
        "recovery_labels_resolved": 0,
    })


def test_needs_user_input_is_interesting() -> None:
    assert _is_interesting({
        "status": "needs_user_input",
        "scanner_count": 11,
        "vision_count": 12,
        "recovery_attempted": True,
        "recovery_labels_resolved": 0,
    })


def test_failed_is_interesting() -> None:
    assert _is_interesting({
        "status": "failed",
        "scanner_count": 0,
        "vision_count": 0,
        "recovery_attempted": False,
        "recovery_labels_resolved": 0,
    })


def test_needs_retry_is_interesting() -> None:
    assert _is_interesting({
        "status": "needs_retry",
        "scanner_count": 0,
        "vision_count": 0,
        "recovery_attempted": False,
        "recovery_labels_resolved": 0,
    })


def test_scanner_vision_mismatch_is_interesting() -> None:
    assert _is_interesting({
        "status": "complete",
        "scanner_count": 5,
        "vision_count": 6,
        "recovery_attempted": False,
        "recovery_labels_resolved": 0,
    })


def test_recovery_failed_is_interesting() -> None:
    assert _is_interesting({
        "status": "complete",
        "scanner_count": 6,
        "vision_count": 6,
        "recovery_attempted": True,
        "recovery_labels_resolved": 0,
    })


def test_recovery_succeeded_not_interesting() -> None:
    assert not _is_interesting({
        "status": "complete",
        "scanner_count": 6,
        "vision_count": 6,
        "recovery_attempted": True,
        "recovery_labels_resolved": 1,
    })


# ---------------------------------------------------------------------------
# Sink routing — only INGEST_COMPLETED
# ---------------------------------------------------------------------------


def test_sink_ignores_non_ingest_completed_events() -> None:
    sink = AnnotationCandidateSink()
    event = DomainEvent(
        type=EventType.IMAGE_RECEIVED,
        run_id=str(uuid4()),
        session_id="test",
        payload={"source": "cli"},
    )
    sink.emit(event)
    assert len(list_pending()) == 0


def test_sink_ignores_uninteresting_complete() -> None:
    sink = AnnotationCandidateSink()
    event = _make_ingest_completed_event(status="complete", scanner_count=6, vision_count=6)
    sink.emit(event)
    assert len(list_pending()) == 0


def test_sink_captures_interesting_failure() -> None:
    sink = AnnotationCandidateSink()
    event = _make_ingest_completed_event(
        status="needs_user_input",
        scanner_count=11,
        vision_count=12,
        recovery_attempted=True,
        recovery_labels_resolved=0,
        image_ref="/tmp/photo.jpg",
    )
    sink.emit(event)

    pending = list_pending()
    assert len(pending) == 1
    assert pending[0].status == "needs_user_input"
    assert pending[0].image_ref == "/tmp/photo.jpg"
    assert pending[0].scanner_count == 11
    assert pending[0].vision_count == 12


# ---------------------------------------------------------------------------
# Duplicate event — idempotency
# ---------------------------------------------------------------------------


def test_duplicate_event_creates_one_row() -> None:
    """Emit same INGEST_COMPLETED event twice → one SQLite row."""
    sink = AnnotationCandidateSink()
    run_id = str(uuid4())
    event = _make_ingest_completed_event(
        run_id=run_id,
        status="needs_user_input",
        scanner_count=11,
        vision_count=12,
    )

    sink.emit(event)
    sink.emit(event)  # duplicate

    pending = list_pending()
    assert len(pending) == 1
    assert pending[0].run_id == run_id


# ---------------------------------------------------------------------------
# Write failure — best-effort, logged, doesn't raise
# None: We can't easily force a SQLite failure without breaking the DB,
# but we can verify the sink doesn't raise on a non-existent DB path
# (init_db creates the directory, so we test with a read-only path).
# ---------------------------------------------------------------------------


def test_sink_does_not_raise_on_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sink write failure is logged but doesn't propagate."""
    # Point to a path inside a read-only directory.
    ro_dir = tmp_path / "readonly"
    ro_dir.mkdir()
    os.chmod(str(ro_dir), 0o444)
    monkeypatch.setenv("ANNOTATION_DB_PATH", str(ro_dir / "test.db"))

    sink = AnnotationCandidateSink()
    event = _make_ingest_completed_event(status="failed")

    # Should not raise.
    sink.emit(event)


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------


def test_register_annotation_sink_is_idempotent() -> None:
    sink1 = register_annotation_sink()
    sink2 = register_annotation_sink()
    assert sink1 is sink2
    # Only one annotation sink in the registry.
    annotation_sinks = [s for s in get_sinks() if isinstance(s, AnnotationCandidateSink)]
    assert len(annotation_sinks) == 1

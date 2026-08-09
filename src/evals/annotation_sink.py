"""AnnotationCandidateSink — captures interesting failures from the event stream.

Listens to ``INGEST_COMPLETED`` events. If the run matches "interesting
failure" rules (scanner≠vision, recovery failed, needs_user_input, or
error), a candidate is written to the SQLite annotation store.

Registration is explicit — only ``src/cli.py`` registers this sink in this
milestone. Eval runs and web/WhatsApp (which don't call ``ingest_one()``)
do not register it.

Write failures are best-effort but logged — the sink never breaks the
ingest path, but a failed write is always observable via
``logger.exception``.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from src.observability.event_sink import EventSink
from src.runtime.events import DomainEvent, EventType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# "Interesting failure" rules — deterministic, no judge model
# ---------------------------------------------------------------------------


def _is_interesting(payload: dict[str, Any]) -> bool:
    """Return True if this run should be captured as an annotation candidate."""
    status = payload.get("status", "")
    scanner_count = payload.get("scanner_count", 0)
    vision_count = payload.get("vision_count", 0)
    recovery_attempted = payload.get("recovery_attempted", False)
    recovery_labels_resolved = payload.get("recovery_labels_resolved", 0)

    # Error or needs-retry — always interesting.
    if status in ("failed", "needs_retry"):
        return True

    # Missing barcodes — user asked for a better photo.
    if status == "needs_user_input":
        return True

    # Scanner/vision mismatch — scanner missed something or vision hallucinated.
    if scanner_count != vision_count:
        return True

    # Recovery was attempted but resolved nothing.
    if recovery_attempted and recovery_labels_resolved == 0:
        return True

    return False


# ---------------------------------------------------------------------------
# Event → candidate conversion
# ---------------------------------------------------------------------------


def _event_to_candidate(event: DomainEvent) -> Any:
    """Convert an INGEST_COMPLETED event to an AnnotationCandidate."""
    from src.evals.annotation_store import AnnotationCandidate

    payload = event.payload
    return AnnotationCandidate(
        id=str(uuid4()),
        run_id=event.run_id,
        session_id=event.session_id,
        source=payload.get("source", "unknown"),
        image_ref=payload.get("image_ref"),
        status=payload.get("status", "unknown"),
        scanner_count=payload.get("scanner_count", 0),
        vision_count=payload.get("vision_count", 0),
        recovery_attempted=payload.get("recovery_attempted", False),
        recovery_labels_resolved=payload.get("recovery_labels_resolved", 0),
        issues=payload.get("issues", []),
    )


# ---------------------------------------------------------------------------
# Sink
# ---------------------------------------------------------------------------


class AnnotationCandidateSink(EventSink):
    """Event sink that captures interesting failures as annotation candidates.

    Best-effort: write failures are logged but never propagate to the
    ingest path. Candidate creation is idempotent via ``UNIQUE(run_id)``
    in the SQLite store.
    """

    def emit(self, event: DomainEvent) -> None:
        if event.type != EventType.INGEST_COMPLETED.value:
            return
        if not _is_interesting(event.payload):
            return

        candidate = _event_to_candidate(event)
        try:
            from src.evals.annotation_store import create_candidate

            created = create_candidate(candidate)
            if created:
                logger.info(
                    "annotation_candidate_created run_id=%s status=%s",
                    event.run_id, candidate.status,
                )
        except Exception:
            logger.exception(
                "annotation_candidate_write_failed run_id=%s", event.run_id
            )


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------


_annotation_sink: AnnotationCandidateSink | None = None


def register_annotation_sink() -> AnnotationCandidateSink:
    """Register the annotation sink. Idempotent — returns the existing instance."""
    global _annotation_sink
    if _annotation_sink is not None:
        return _annotation_sink
    _annotation_sink = AnnotationCandidateSink()
    from src.observability.event_sink import register_sink

    register_sink(_annotation_sink)
    logger.info("AnnotationCandidateSink registered")
    return _annotation_sink

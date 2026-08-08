"""Online evaluation — deterministic rule-based feedback for production runs.

Unlike offline eval (which compares against ground truth), online eval
generates cheap, deterministic feedback from the ``IngestResult`` itself.
This feedback is pushed to LangSmith as scores on the production trace,
enabling dashboards and alerts without a judge model.

Feedback keys:
    complete              — 1 if status == COMPLETE, else 0
    scanner_vision_match  — 1 if scanner_count == vision_count, else 0
    recovery_required     — 1 if recovery was attempted, else 0
    recovery_success      — 1 if recovery resolved any labels, else 0
    user_retry_required   — 1 if status == NEEDS_USER_INPUT, else 0
    latency_ok            — 1 if elapsed_ms <= threshold, else 0
"""

from __future__ import annotations

from typing import Any

from src.ingest.models import IngestResult, IngestStatus

# Latency threshold for online feedback (ms). Matches offline eval default.
LATENCY_THRESHOLD_MS = 10_000


def evaluate_production_run(result: IngestResult) -> list[dict[str, Any]]:
    """Generate deterministic feedback for a production ``IngestResult``.

    Returns a list of feedback dicts, each with ``key``, ``score``,
    and ``comment``. Push these to LangSmith as scores on the run.
    """
    m = result.metrics
    return [
        {
            "key": "complete",
            "score": 1 if result.status == IngestStatus.COMPLETE else 0,
            "comment": f"status={result.status.value}",
        },
        {
            "key": "scanner_vision_match",
            "score": 1 if m.scanner_count == m.vision_count else 0,
            "comment": f"scanner={m.scanner_count} vision={m.vision_count}",
        },
        {
            "key": "recovery_required",
            "score": 1 if m.recovery_attempted else 0,
            "comment": f"attempted={m.recovery_attempted}",
        },
        {
            "key": "recovery_success",
            "score": 1 if m.recovery_labels_resolved > 0 else 0,
            "comment": f"resolved={m.recovery_labels_resolved}",
        },
        {
            "key": "user_retry_required",
            "score": 1 if result.status == IngestStatus.NEEDS_USER_INPUT else 0,
            "comment": f"status={result.status.value}",
        },
        {
            "key": "latency_ok",
            "score": 1 if m.elapsed_ms <= LATENCY_THRESHOLD_MS else 0,
            "comment": f"elapsed_ms={m.elapsed_ms}",
        },
    ]

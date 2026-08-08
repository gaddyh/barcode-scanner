"""Tests for online eval — deterministic feedback from IngestResult.

No LangSmith or Gemini calls. Stubs IngestResult and verifies feedback
keys and scores.
"""

from __future__ import annotations

from src.evals.online import evaluate_production_run
from src.ingest.models import (
    DetectedItem,
    IngestResult,
    IngestStatus,
    RunMetrics,
)


def _result(
    status: IngestStatus = IngestStatus.COMPLETE,
    *,
    scanner_count: int = 6,
    vision_count: int = 6,
    recovery_attempted: bool = False,
    recovery_labels_resolved: int = 0,
    elapsed_ms: int = 3000,
) -> IngestResult:
    return IngestResult(
        status=status,
        items=[DetectedItem(barcode_value="x") for _ in range(scanner_count)],
        metrics=RunMetrics(
            elapsed_ms=elapsed_ms,
            scanner_count=scanner_count,
            vision_count=vision_count,
            recovery_attempted=recovery_attempted,
            recovery_labels_resolved=recovery_labels_resolved,
        ),
    )


def _score(feedback: list[dict], key: str) -> int:
    for fb in feedback:
        if fb["key"] == key:
            return fb["score"]
    raise KeyError(key)


def test_complete_status_scores_one() -> None:
    fb = evaluate_production_run(_result(IngestStatus.COMPLETE))
    assert _score(fb, "complete") == 1
    assert _score(fb, "user_retry_required") == 0


def test_needs_user_input_scores_zero_complete() -> None:
    fb = evaluate_production_run(_result(IngestStatus.NEEDS_USER_INPUT))
    assert _score(fb, "complete") == 0
    assert _score(fb, "user_retry_required") == 1


def test_scanner_vision_match() -> None:
    fb = evaluate_production_run(_result(scanner_count=6, vision_count=6))
    assert _score(fb, "scanner_vision_match") == 1


def test_scanner_vision_mismatch() -> None:
    fb = evaluate_production_run(_result(scanner_count=11, vision_count=12))
    assert _score(fb, "scanner_vision_match") == 0


def test_recovery_not_attempted() -> None:
    fb = evaluate_production_run(_result(recovery_attempted=False))
    assert _score(fb, "recovery_required") == 0
    assert _score(fb, "recovery_success") == 0


def test_recovery_attempted_but_failed() -> None:
    fb = evaluate_production_run(
        _result(recovery_attempted=True, recovery_labels_resolved=0)
    )
    assert _score(fb, "recovery_required") == 1
    assert _score(fb, "recovery_success") == 0


def test_recovery_attempted_and_succeeded() -> None:
    fb = evaluate_production_run(
        _result(recovery_attempted=True, recovery_labels_resolved=2)
    )
    assert _score(fb, "recovery_required") == 1
    assert _score(fb, "recovery_success") == 1


def test_latency_under_threshold() -> None:
    fb = evaluate_production_run(_result(elapsed_ms=3000))
    assert _score(fb, "latency_ok") == 1


def test_latency_over_threshold() -> None:
    fb = evaluate_production_run(_result(elapsed_ms=20_000))
    assert _score(fb, "latency_ok") == 0


def test_feedback_has_all_expected_keys() -> None:
    fb = evaluate_production_run(_result())
    keys = {f["key"] for f in fb}
    assert keys == {
        "complete",
        "scanner_vision_match",
        "recovery_required",
        "recovery_success",
        "user_retry_required",
        "latency_ok",
    }


def test_feedback_has_comments() -> None:
    fb = evaluate_production_run(_result())
    for f in fb:
        assert "comment" in f
        assert isinstance(f["comment"], str)

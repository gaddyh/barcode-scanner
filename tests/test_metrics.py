"""Tests for compute_metrics() — pure aggregation over run-like objects.

No LangSmith, no network. All aggregation logic tested here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.evals.metrics import (
    compute_metrics,
    unavailable_response,
)


@dataclass
class StubRun:
    """Minimal run-like object with a metadata dict."""
    metadata: dict[str, Any]


def _run(
    final_status: str = "complete",
    source: str = "cli",
    found_count: int = 6,
    recovery_attempted: bool = False,
    recovery_labels_tried: int = 0,
    recovery_labels_resolved: int = 0,
    latency_ms: int = 3000,
    scanner_count: int = 6,
    vision_count: int = 6,
    missing_count: int = 0,
    unassigned_count: int = 0,
    scanner_vision_match: bool = True,
    count_delta: int = 0,
    recovery_succeeded: bool = False,
    primary_issue: str = "none",
    pipeline_version: str = "ingest-v1",
    scanner_version: str = "scanner-0.8",
    recovery_version: str = "recovery-v1",
    vision_model: str = "gemini-3.5-flash-lite",
) -> StubRun:
    return StubRun(metadata={
        "final_status": final_status,
        "source": source,
        "found_count": found_count,
        "recovery_attempted": recovery_attempted,
        "recovery_labels_tried": recovery_labels_tried,
        "recovery_labels_resolved": recovery_labels_resolved,
        "latency_ms": latency_ms,
        "scanner_count": scanner_count,
        "vision_count": vision_count,
        "missing_count": missing_count,
        "unassigned_count": unassigned_count,
        "scanner_vision_match": scanner_vision_match,
        "count_delta": count_delta,
        "recovery_succeeded": recovery_succeeded,
        "primary_issue": primary_issue,
        "pipeline_version": pipeline_version,
        "scanner_version": scanner_version,
        "recovery_version": recovery_version,
        "vision_model": vision_model,
    })


# ---------------------------------------------------------------------------
# Empty / zero cases
# ---------------------------------------------------------------------------


def test_zero_runs_returns_zeros() -> None:
    resp = compute_metrics([])
    assert resp.images_processed == 0
    assert resp.boxes_processed == 0
    assert resp.first_pass_complete_pct == 0.0
    assert resp.final_complete_pct == 0.0
    assert resp.recovery_attempted_pct == 0.0
    assert resp.recovery_success_pct == 0.0
    assert resp.user_retry_required_pct == 0.0
    assert resp.p95_latency_ms == 0.0
    assert resp.scanner_vision_match_pct == 0.0
    assert resp.avg_count_delta == 0.0
    assert resp.avg_missing_count == 0.0
    assert resp.avg_unassigned_count == 0.0
    assert resp.recovered_complete_pct == 0.0
    assert resp.still_incomplete_pct == 0.0
    assert resp.avg_labels_tried == 0.0
    assert resp.avg_labels_resolved == 0.0
    assert resp.primary_issue_counts == {}
    assert resp.source == "langsmith"
    assert resp.truncated is False


def test_unavailable_response() -> None:
    resp = unavailable_response(time_window_hours=12)
    assert resp.source == "unavailable"
    assert resp.time_window_hours == 12
    assert resp.images_processed == 0


# ---------------------------------------------------------------------------
# Operations metrics
# ---------------------------------------------------------------------------


def test_one_complete_no_recovery() -> None:
    resp = compute_metrics([_run(final_status="complete", recovery_attempted=False)])
    assert resp.images_processed == 1
    assert resp.boxes_processed == 6
    assert resp.first_pass_complete_pct == 100.0
    assert resp.final_complete_pct == 100.0
    assert resp.recovery_attempted_pct == 0.0
    assert resp.recovery_success_pct == 0.0
    assert resp.user_retry_required_pct == 0.0
    assert resp.p95_latency_ms == 3000.0


def test_one_complete_with_recovery() -> None:
    resp = compute_metrics([
        _run(
            final_status="complete",
            recovery_attempted=True,
            recovery_labels_resolved=1,
            recovery_succeeded=True,
        ),
    ])
    assert resp.first_pass_complete_pct == 0.0
    assert resp.final_complete_pct == 100.0
    assert resp.recovery_attempted_pct == 100.0
    assert resp.recovery_success_pct == 100.0
    assert resp.recovered_complete_pct == 100.0


def test_one_needs_user_input() -> None:
    resp = compute_metrics([_run(final_status="needs_user_input")])
    assert resp.user_retry_required_pct == 100.0
    assert resp.final_complete_pct == 0.0
    assert resp.still_incomplete_pct == 100.0


def test_mixed_complete_and_retry() -> None:
    runs = [
        _run(final_status="complete", recovery_attempted=False),
        _run(final_status="complete", recovery_attempted=False),
        _run(
            final_status="complete",
            recovery_attempted=True,
            recovery_labels_resolved=1,
            recovery_succeeded=True,
        ),
        _run(
            final_status="needs_user_input",
            recovery_attempted=True,
            recovery_labels_resolved=0,
            recovery_succeeded=False,
        ),
    ]
    resp = compute_metrics(runs)
    assert resp.images_processed == 4
    assert resp.first_pass_complete_pct == 50.0
    assert resp.final_complete_pct == 75.0
    assert resp.recovery_attempted_pct == 50.0
    assert resp.recovery_success_pct == 50.0
    assert resp.user_retry_required_pct == 25.0
    assert resp.recovered_complete_pct == 25.0
    assert resp.still_incomplete_pct == 25.0


def test_recovery_success_pct_zero_when_no_recovery_attempted() -> None:
    runs = [_run(final_status="complete", recovery_attempted=False)]
    resp = compute_metrics(runs)
    assert resp.recovery_success_pct == 0.0


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------


def test_scanner_vision_match_pct() -> None:
    runs = [
        _run(scanner_vision_match=True),
        _run(scanner_vision_match=True),
        _run(scanner_vision_match=False, scanner_count=5, vision_count=6, count_delta=1),
    ]
    resp = compute_metrics(runs)
    assert resp.scanner_vision_match_pct == 66.7


def test_avg_count_delta() -> None:
    runs = [
        _run(count_delta=0),
        _run(count_delta=1, scanner_count=5, vision_count=6),
        _run(count_delta=-1, scanner_count=7, vision_count=6),
    ]
    resp = compute_metrics(runs)
    assert resp.avg_count_delta == 0.0


def test_avg_missing_and_unassigned() -> None:
    runs = [
        _run(missing_count=0, unassigned_count=0),
        _run(missing_count=2, unassigned_count=1),
        _run(missing_count=1, unassigned_count=3),
    ]
    resp = compute_metrics(runs)
    assert resp.avg_missing_count == 1.0
    assert resp.avg_unassigned_count == 1.3


# ---------------------------------------------------------------------------
# Recovery detail
# ---------------------------------------------------------------------------


def test_avg_labels_tried_and_resolved() -> None:
    runs = [
        _run(
            recovery_attempted=True,
            recovery_labels_tried=2,
            recovery_labels_resolved=1,
        ),
        _run(
            recovery_attempted=True,
            recovery_labels_tried=3,
            recovery_labels_resolved=0,
        ),
        _run(recovery_attempted=False),  # not counted in recovery averages
    ]
    resp = compute_metrics(runs)
    assert resp.avg_labels_tried == 2.5  # (2+3)/2
    assert resp.avg_labels_resolved == 0.5  # (1+0)/2


def test_avg_labels_zero_when_no_recovery() -> None:
    runs = [_run(recovery_attempted=False)]
    resp = compute_metrics(runs)
    assert resp.avg_labels_tried == 0.0
    assert resp.avg_labels_resolved == 0.0


# ---------------------------------------------------------------------------
# Primary issue counts
# ---------------------------------------------------------------------------


def test_primary_issue_counts() -> None:
    runs = [
        _run(primary_issue="none"),
        _run(primary_issue="BARCODE_MISSING"),
        _run(primary_issue="BARCODE_MISSING"),
        _run(primary_issue="VISION_SCANNER_MISMATCH"),
    ]
    resp = compute_metrics(runs)
    assert resp.primary_issue_counts == {
        "none": 1,
        "BARCODE_MISSING": 2,
        "VISION_SCANNER_MISMATCH": 1,
    }


def test_primary_issue_counts_defaults_to_none() -> None:
    """Runs without primary_issue metadata default to 'none'."""
    runs = [StubRun(metadata={"final_status": "complete"})]
    resp = compute_metrics(runs)
    assert resp.primary_issue_counts == {"none": 1}


# ---------------------------------------------------------------------------
# P95 latency
# ---------------------------------------------------------------------------


def test_p95_known_distribution() -> None:
    runs = [_run(latency_ms=100 + i * 100) for i in range(20)]
    resp = compute_metrics(runs)
    assert resp.p95_latency_ms == 1900.0


def test_p95_single_value() -> None:
    resp = compute_metrics([_run(latency_ms=5000)])
    assert resp.p95_latency_ms == 5000.0


def test_p95_ignores_zero_latency() -> None:
    runs = [_run(latency_ms=0), _run(latency_ms=1000), _run(latency_ms=2000)]
    resp = compute_metrics(runs)
    assert resp.p95_latency_ms == 2000.0


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


def test_truncation_flag() -> None:
    runs = [_run() for _ in range(100)]
    resp = compute_metrics(runs, truncated=True)
    assert resp.truncated is True
    assert resp.images_processed == 100


def test_not_truncated_under_limit() -> None:
    runs = [_run() for _ in range(99)]
    resp = compute_metrics(runs, truncated=False)
    assert resp.truncated is False


# ---------------------------------------------------------------------------
# Boxes processed
# ---------------------------------------------------------------------------


def test_boxes_processed_sums_found_count() -> None:
    runs = [
        _run(found_count=6),
        _run(found_count=12),
        _run(found_count=0),
    ]
    resp = compute_metrics(runs)
    assert resp.boxes_processed == 18


# ---------------------------------------------------------------------------
# Metadata robustness
# ---------------------------------------------------------------------------


def test_string_bool_recovery_attempted() -> None:
    runs = [StubRun(metadata={
        "final_status": "complete",
        "found_count": "6",
        "recovery_attempted": "True",
        "recovery_labels_resolved": "1",
        "latency_ms": "3000",
        "scanner_vision_match": "True",
        "count_delta": "0",
        "recovery_succeeded": "True",
        "primary_issue": "none",
    })]
    resp = compute_metrics(runs)
    assert resp.recovery_attempted_pct == 100.0
    assert resp.recovery_success_pct == 100.0
    assert resp.boxes_processed == 6
    assert resp.p95_latency_ms == 3000.0
    assert resp.scanner_vision_match_pct == 100.0


def test_missing_metadata_keys() -> None:
    runs = [StubRun(metadata={})]
    resp = compute_metrics(runs)
    assert resp.images_processed == 1
    assert resp.boxes_processed == 0
    assert resp.final_complete_pct == 0.0
    assert resp.scanner_vision_match_pct == 0.0
    assert resp.primary_issue_counts == {"none": 1}


# ---------------------------------------------------------------------------
# Version breakdown rates
# ---------------------------------------------------------------------------


def test_completion_by_pipeline_version() -> None:
    """True completion % per pipeline version, not raw counts."""
    runs = [
        _run(final_status="complete", pipeline_version="ingest-v1"),
        _run(final_status="complete", pipeline_version="ingest-v1"),
        _run(final_status="needs_user_input", pipeline_version="ingest-v1"),
        _run(final_status="complete", pipeline_version="ingest-v2"),
        _run(final_status="needs_user_input", pipeline_version="ingest-v2"),
        _run(final_status="needs_user_input", pipeline_version="ingest-v2"),
        _run(final_status="needs_user_input", pipeline_version="ingest-v2"),
    ]
    resp = compute_metrics(runs)
    rates = {r.version: r for r in resp.completion_by_pipeline_version}
    assert rates["ingest-v1"].total == 3
    assert rates["ingest-v1"].rate_pct == 66.7
    assert rates["ingest-v2"].total == 4
    assert rates["ingest-v2"].rate_pct == 25.0


def test_mismatch_by_scanner_version() -> None:
    """True mismatch % per scanner version."""
    runs = [
        _run(scanner_vision_match=True, scanner_version="scanner-0.8"),
        _run(scanner_vision_match=False, scanner_version="scanner-0.8"),
        _run(scanner_vision_match=True, scanner_version="scanner-0.9"),
        _run(scanner_vision_match=True, scanner_version="scanner-0.9"),
        _run(scanner_vision_match=True, scanner_version="scanner-0.9"),
    ]
    resp = compute_metrics(runs)
    rates = {r.version: r for r in resp.mismatch_by_scanner_version}
    assert rates["scanner-0.8"].total == 2
    assert rates["scanner-0.8"].rate_pct == 50.0
    assert rates["scanner-0.9"].total == 3
    assert rates["scanner-0.9"].rate_pct == 0.0


def test_recovery_success_by_recovery_version() -> None:
    """True recovery success % per recovery version. Only recovery_attempted runs counted."""
    runs = [
        _run(recovery_attempted=True, recovery_succeeded=True, recovery_version="recovery-v1"),
        _run(recovery_attempted=True, recovery_succeeded=False, recovery_version="recovery-v1"),
        _run(recovery_attempted=True, recovery_succeeded=True, recovery_version="recovery-v2"),
        _run(recovery_attempted=True, recovery_succeeded=True, recovery_version="recovery-v2"),
        _run(recovery_attempted=True, recovery_succeeded=True, recovery_version="recovery-v2"),
        # This run has no recovery — should NOT appear in recovery version breakdown
        _run(recovery_attempted=False, recovery_version="recovery-v1"),
    ]
    resp = compute_metrics(runs)
    rates = {r.version: r for r in resp.recovery_success_by_recovery_version}
    assert rates["recovery-v1"].total == 2  # only the 2 attempted runs
    assert rates["recovery-v1"].rate_pct == 50.0
    assert rates["recovery-v2"].total == 3
    assert rates["recovery-v2"].rate_pct == 100.0


def test_retry_by_vision_model() -> None:
    """True retry % per vision model."""
    runs = [
        _run(final_status="complete", vision_model="gemini-a"),
        _run(final_status="needs_user_input", vision_model="gemini-a"),
        _run(final_status="complete", vision_model="gemini-b"),
        _run(final_status="complete", vision_model="gemini-b"),
        _run(final_status="complete", vision_model="gemini-b"),
    ]
    resp = compute_metrics(runs)
    rates = {r.version: r for r in resp.retry_by_vision_model}
    assert rates["gemini-a"].total == 2
    assert rates["gemini-a"].rate_pct == 50.0
    assert rates["gemini-b"].total == 3
    assert rates["gemini-b"].rate_pct == 0.0


def test_version_rates_empty() -> None:
    resp = compute_metrics([])
    assert resp.completion_by_pipeline_version == []
    assert resp.mismatch_by_scanner_version == []
    assert resp.recovery_success_by_recovery_version == []
    assert resp.retry_by_vision_model == []


def test_version_rates_sorted() -> None:
    """Version rates are sorted by version string."""
    runs = [
        _run(pipeline_version="ingest-v3"),
        _run(pipeline_version="ingest-v1"),
        _run(pipeline_version="ingest-v2"),
    ]
    resp = compute_metrics(runs)
    versions = [r.version for r in resp.completion_by_pipeline_version]
    assert versions == ["ingest-v1", "ingest-v2", "ingest-v3"]


def test_version_rates_missing_version_key() -> None:
    """Runs without the version key default to 'unknown'."""
    runs = [StubRun(metadata={"final_status": "complete"})]
    resp = compute_metrics(runs)
    assert len(resp.completion_by_pipeline_version) == 1
    assert resp.completion_by_pipeline_version[0].version == "unknown"
    assert resp.completion_by_pipeline_version[0].rate_pct == 100.0

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
    recovery_labels_resolved: int = 0,
    latency_ms: int = 3000,
) -> StubRun:
    return StubRun(metadata={
        "final_status": final_status,
        "source": source,
        "found_count": found_count,
        "recovery_attempted": recovery_attempted,
        "recovery_labels_resolved": recovery_labels_resolved,
        "latency_ms": latency_ms,
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
    assert resp.source == "langsmith"
    assert resp.truncated is False


def test_unavailable_response() -> None:
    resp = unavailable_response(time_window_hours=12)
    assert resp.source == "unavailable"
    assert resp.time_window_hours == 12
    assert resp.images_processed == 0


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------


def test_one_complete_no_recovery() -> None:
    resp = compute_metrics([_run(final_status="complete", recovery_attempted=False)])
    assert resp.images_processed == 1
    assert resp.boxes_processed == 6
    assert resp.first_pass_complete_pct == 100.0
    assert resp.final_complete_pct == 100.0
    assert resp.recovery_attempted_pct == 0.0
    assert resp.recovery_success_pct == 0.0  # no recovery attempted
    assert resp.user_retry_required_pct == 0.0
    assert resp.p95_latency_ms == 3000.0


def test_one_complete_with_recovery() -> None:
    """complete + recovery_attempted → first_pass_complete is 0%, final_complete is 100%."""
    resp = compute_metrics([
        _run(final_status="complete", recovery_attempted=True, recovery_labels_resolved=1),
    ])
    assert resp.first_pass_complete_pct == 0.0  # completed WITH recovery
    assert resp.final_complete_pct == 100.0
    assert resp.recovery_attempted_pct == 100.0
    assert resp.recovery_success_pct == 100.0


def test_one_needs_user_input() -> None:
    resp = compute_metrics([_run(final_status="needs_user_input")])
    assert resp.user_retry_required_pct == 100.0
    assert resp.final_complete_pct == 0.0


# ---------------------------------------------------------------------------
# Mixed runs
# ---------------------------------------------------------------------------


def test_mixed_complete_and_retry() -> None:
    runs = [
        _run(final_status="complete", recovery_attempted=False),
        _run(final_status="complete", recovery_attempted=False),
        _run(final_status="complete", recovery_attempted=True, recovery_labels_resolved=1),
        _run(final_status="needs_user_input", recovery_attempted=True, recovery_labels_resolved=0),
    ]
    resp = compute_metrics(runs)
    assert resp.images_processed == 4
    assert resp.first_pass_complete_pct == 50.0  # 2/4 completed without recovery
    assert resp.final_complete_pct == 75.0  # 3/4 completed
    assert resp.recovery_attempted_pct == 50.0  # 2/4
    assert resp.recovery_success_pct == 50.0  # 1/2 of recovery attempts
    assert resp.user_retry_required_pct == 25.0  # 1/4


# ---------------------------------------------------------------------------
# Recovery denominator = 0
# ---------------------------------------------------------------------------


def test_recovery_success_pct_zero_when_no_recovery_attempted() -> None:
    """recovery_success_pct is 0.0 (not NaN) when no recovery was attempted."""
    runs = [_run(final_status="complete", recovery_attempted=False)]
    resp = compute_metrics(runs)
    assert resp.recovery_success_pct == 0.0


# ---------------------------------------------------------------------------
# P95 latency
# ---------------------------------------------------------------------------


def test_p95_known_distribution() -> None:
    """20 runs with latencies 100..2000 — P95 is the 19th value (1900)."""
    runs = [_run(latency_ms=100 + i * 100) for i in range(20)]
    resp = compute_metrics(runs)
    # nearest-rank: ceil(0.95 * 20) = 19, index 18 → 1900
    assert resp.p95_latency_ms == 1900.0


def test_p95_single_value() -> None:
    resp = compute_metrics([_run(latency_ms=5000)])
    assert resp.p95_latency_ms == 5000.0


def test_p95_ignores_zero_latency() -> None:
    """Runs with latency_ms=0 are excluded from P95."""
    runs = [_run(latency_ms=0), _run(latency_ms=1000), _run(latency_ms=2000)]
    resp = compute_metrics(runs)
    # Only 2 non-zero values: P95 = ceil(0.95 * 2) = 2, index 1 → 2000
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
        _run(found_count=0),  # failed run
    ]
    resp = compute_metrics(runs)
    assert resp.boxes_processed == 18


# ---------------------------------------------------------------------------
# Metadata robustness
# ---------------------------------------------------------------------------


def test_string_bool_recovery_attempted() -> None:
    """LangSmith may serialize bools as strings."""
    runs = [StubRun(metadata={
        "final_status": "complete",
        "found_count": "6",
        "recovery_attempted": "True",
        "recovery_labels_resolved": "1",
        "latency_ms": "3000",
    })]
    resp = compute_metrics(runs)
    assert resp.recovery_attempted_pct == 100.0
    assert resp.recovery_success_pct == 100.0
    assert resp.boxes_processed == 6
    assert resp.p95_latency_ms == 3000.0


def test_missing_metadata_keys() -> None:
    """Runs with missing metadata keys don't crash."""
    runs = [StubRun(metadata={})]
    resp = compute_metrics(runs)
    assert resp.images_processed == 1
    assert resp.boxes_processed == 0
    assert resp.final_complete_pct == 0.0

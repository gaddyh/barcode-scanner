"""Snapshot regression test for the warehouse benchmark.

Replays captured audit candidates (both Gemini audits per image) through the
current scanner + ``select_best_spatial_audit`` + ``reconcile_with_recovery``.
Deterministic — runs in normal CI without charged Gemini calls.

Skips cleanly when snapshots are absent. Run::

    python -m tests.benchmark_warehouse.runner --capture-snapshots --runs 1

once with ``GEMINI_API_KEY`` set to populate them.
"""

from __future__ import annotations

import pytest

from tests.benchmark_warehouse.runner import (
    SNAPSHOTS_PATH,
    load_dataset,
    run_snapshot_benchmark,
)


def test_warehouse_snapshot_baseline() -> None:
    """Soft-threshold regression on the frozen snapshot baseline.

    Exercises the current scanner, dual-audit selection, reconciliation, and
    recovery against frozen Gemini inputs. Must pass in normal CI.
    """
    if not SNAPSHOTS_PATH.exists():
        pytest.skip(
            f"Snapshot file absent: {SNAPSHOTS_PATH}. Run "
            f"`python -m tests.benchmark_warehouse.runner --capture-snapshots --runs 1` "
            f"once with GEMINI_API_KEY set to populate it."
        )

    result = run_snapshot_benchmark()
    agg = result.aggregate
    dataset = load_dataset()

    # Dataset-derived totals.
    assert agg.image_count == len(dataset.images)

    # Soft thresholds — protect the frozen baseline without being brittle.
    # Snapshot replay freezes one Gemini response per image; recovery crop
    # regions depend on the specific label boxes in that response, so
    # snapshot-vs-live variance of ±1-2 recoveries is expected.
    assert agg.label_count_accuracy >= 0.85
    assert agg.baseline_recall >= 0.95
    assert agg.final_recall >= 0.90
    assert abs(agg.recovery_uplift - agg.expected_recovery_uplift) <= 2
    assert agg.false_positive_violation_count <= 1
    assert agg.failure_reason_mismatch_count <= 2

    # Per-image failure_reason matches expected for most images. Snapshot-vs-live
    # variance can flip reasons (e.g. recovery_failed → all_matched when the
    # frozen Gemini labels happen to all match). Allow up to 2 mismatches.
    mismatched = sum(
        1
        for image, runs in result.run_metrics_by_image.items()
        if runs and runs[0].failure_reason != _expected_reason(dataset, image)
    )
    assert mismatched <= 2


def _expected_reason(dataset, image: str) -> str:
    for gt in dataset.images:
        if gt.image == image:
            return gt.expected_failure_reason
    return ""

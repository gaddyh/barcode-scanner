"""Live consistency test for the warehouse benchmark.

Runs the real pipeline (scanner + dual Gemini audit + selection +
reconciliation + recovery) multiple times per image. Charged, slow,
network-dependent.

Gated by ``RUN_LIVE_GEMINI=1`` so plain ``pytest`` never makes charged API
calls. Run explicitly::

    RUN_LIVE_GEMINI=1 pytest -m live_gemini tests/benchmark_warehouse/test_live.py

First-PR behavior (diagnostic-first):
- Hard gate: p95 latency must be under the configured threshold.
- Diagnostic: first-run exact match is recorded and printed but NOT asserted.
  Gemini non-determinism (label count drift, recovery crop variance) can cause
  1-2 images to mismatch on any given run. Once variance is measured and
  proves stable, promote the commented-out ``assert agg.all_first_runs_match_expected``
  to a hard gate.
- Diagnostic: across-run consistency is recorded and printed but NOT asserted.
  Once variance is measured and proves stable, promote the commented-out
  ``assert result.all_runs_consistent`` to a hard gate.
"""

from __future__ import annotations

import os

import pytest

from tests.benchmark_warehouse.runner import (
    load_dataset,
    print_variance_report,
    run_live_benchmark,
)


@pytest.mark.live_gemini
@pytest.mark.skipif(
    os.getenv("RUN_LIVE_GEMINI") != "1",
    reason="Set RUN_LIVE_GEMINI=1 to run charged Gemini tests.",
)
def test_live_warehouse_consistency() -> None:
    """Live warehouse benchmark: latency gate, diagnostic first-run + consistency."""
    dataset = load_dataset()
    runs_per_image = dataset.consistency.runs_per_image
    latency_threshold = dataset.consistency.latency_p95_threshold_seconds

    result = run_live_benchmark(runs_per_image=runs_per_image)

    # Print the variance report so we can observe Gemini's current drift.
    print_variance_report(result)

    agg = result.aggregate

    # Hard gate — latency.
    assert agg.latency_p95 <= latency_threshold, (
        f"p95 latency {agg.latency_p95:.2f}s exceeds threshold "
        f"{latency_threshold:.2f}s"
    )

    # Diagnostic during the first PR — recorded and printed, NOT asserted.
    # Once variance is measured and proves stable, promote to:
    #   assert agg.all_first_runs_match_expected
    print(
        f"\nall_first_runs_match_expected={agg.all_first_runs_match_expected} "
        f"(diagnostic — not asserted in first PR)\n"
        f"baseline_recall={agg.baseline_recall:.2%} "
        f"final_recall={agg.final_recall:.2%} "
        f"recovery_uplift={agg.recovery_uplift} "
        f"reason_mismatches={agg.failure_reason_mismatch_count} "
        f"fp_violations={agg.false_positive_violation_count}",
        flush=True,
    )

    # Diagnostic during the first PR — recorded and printed, NOT asserted.
    # Once variance is measured and proves stable, promote to:
    #   assert result.all_runs_consistent
    if agg.all_runs_consistent is not None:
        print(
            f"all_runs_consistent={agg.all_runs_consistent} "
            "(diagnostic — not asserted in first PR)",
            flush=True,
        )

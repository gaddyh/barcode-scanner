"""Regression guard for the deterministic scanner baseline.

Freezes the current scanner behavior against the ground-truth benchmark dataset
so that any change which improves one image while silently breaking another is
caught. Run with:

    pytest tests/benchmark/test_baseline.py
"""

from __future__ import annotations

import pytest

from tests.benchmark.runner import run_bench


def test_baseline_recall_and_no_false_positives() -> None:
    results = run_bench(runs_per_image=3)

    agg = results.aggregate

    assert agg.expected_barcode_symbol_count == 20
    assert agg.expected_decoded == 19
    assert agg.exact_matches == 19
    assert agg.expected_unique_values == 14
    assert agg.unique_values_found == 14
    assert agg.false_positives == 0
    assert agg.mismatches == 0
    assert agg.bonuses == 0
    assert agg.passed is True


def test_runs_per_image_must_be_at_least_two() -> None:
    with pytest.raises(ValueError, match="runs_per_image"):
        run_bench(runs_per_image=1)

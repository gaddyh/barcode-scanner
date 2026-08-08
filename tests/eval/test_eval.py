"""Deterministic tests for the eval harness — no LangSmith or Gemini calls.

Exercises the dataset loader and the evaluators with stub predictions so
the scoring logic is covered without network access.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.evals.datasets import load_dataset
from src.evals.evaluators import (
    aggregate_thresholds,
    count_exact,
    latency,
    outcome_correct,
    recovery_gain,
    value_precision,
    value_recall,
)

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def test_load_dataset_returns_examples() -> None:
    examples = load_dataset()
    # The committed dataset has 3 images (marny, multi_clear_6, multi_12_clean).
    assert len(examples) >= 1
    first = examples[0]
    assert "image_name" in first
    assert "image_path" in first
    assert "expected_values" in first
    assert "expected_decoded_count" in first
    assert first["expected_outcome"] == "complete"
    # All expected values are non-empty strings.
    for v in first["expected_values"]:
        assert isinstance(v, str) and v


def test_load_dataset_skips_missing_images(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Point SAMPLES_DIR at an empty dir so every image is skipped.
    import src.evals.datasets as datasets_mod
    monkeypatch.setattr(datasets_mod, "SAMPLES_DIR", tmp_path)
    examples = load_dataset()
    assert examples == []


# ---------------------------------------------------------------------------
# Evaluators — use a fake example namespace
# ---------------------------------------------------------------------------


class _FakeExample:
    def __init__(self, inputs: dict) -> None:
        self.inputs = inputs


def _example(expected_values=None, expected_decoded_count=0, expected_outcome="complete"):
    return _FakeExample({
        "expected_values": expected_values or [],
        "expected_decoded_count": expected_decoded_count,
        "expected_outcome": expected_outcome,
    })


def _run(found_values=None, status="complete", found_count=None,
         elapsed_ms=1000, recovery_attempted=False, recovery_labels_resolved=0):
    """Build a stub IngestResult-shaped dict for evaluator tests."""
    items = [{"barcode_value": v} for v in (found_values or [])]
    return {
        "outputs": {
            "status": status,
            "items": items,
            "metrics": {
                "elapsed_ms": elapsed_ms,
                "scanner_count": len(items),
                "vision_count": len(items),
                "recovery_attempted": recovery_attempted,
                "recovery_labels_tried": 1 if recovery_attempted else 0,
                "recovery_barcodes_found": 0,
                "recovery_labels_resolved": recovery_labels_resolved,
            },
        }
    }


def test_value_recall_all_found() -> None:
    run = _run(found_values=["111", "222"])
    ex = _example(expected_values=["111", "222"])
    result = value_recall(run, ex)
    assert result["score"] == 1.0


def test_value_recall_partial() -> None:
    run = _run(found_values=["111"])
    ex = _example(expected_values=["111", "222"])
    result = value_recall(run, ex)
    assert result["score"] == 0.5


def test_value_recall_none_expected() -> None:
    run = _run(found_values=[])
    ex = _example(expected_values=[])
    result = value_recall(run, ex)
    assert result["score"] == 1.0


def test_value_precision_all_match() -> None:
    run = _run(found_values=["111", "222"])
    ex = _example(expected_values=["111", "222", "333"])
    result = value_precision(run, ex)
    assert result["score"] == 1.0


def test_value_precision_partial_mismatch() -> None:
    run = _run(found_values=["111", "999"])
    ex = _example(expected_values=["111", "222"])
    result = value_precision(run, ex)
    assert result["score"] == 0.5


def test_value_precision_no_found() -> None:
    run = _run(found_values=[])
    ex = _example(expected_values=["111"])
    result = value_precision(run, ex)
    assert result["score"] == 1.0


def test_outcome_correct_match() -> None:
    run = _run(status="complete")
    ex = _example(expected_outcome="complete")
    assert outcome_correct(run, ex)["score"] == 1.0


def test_outcome_correct_mismatch() -> None:
    run = _run(status="needs_user_input")
    ex = _example(expected_outcome="complete")
    assert outcome_correct(run, ex)["score"] == 0.0


def test_count_exact_match() -> None:
    run = _run(found_values=["1", "2", "3", "4", "5", "6"])
    ex = _example(expected_decoded_count=6)
    assert count_exact(run, ex)["score"] == 1.0


def test_count_exact_mismatch() -> None:
    run = _run(found_values=["1", "2", "3", "4", "5"])
    ex = _example(expected_decoded_count=6)
    assert count_exact(run, ex)["score"] == 0.0


# ---------------------------------------------------------------------------
# New evaluators — recovery_gain and latency
# ---------------------------------------------------------------------------


def test_recovery_gain_not_needed() -> None:
    run = _run(recovery_attempted=False)
    ex = _example()
    assert recovery_gain(run, ex)["score"] == 1.0


def test_recovery_gain_succeeded() -> None:
    run = _run(recovery_attempted=True, recovery_labels_resolved=2)
    ex = _example()
    assert recovery_gain(run, ex)["score"] == 1.0


def test_recovery_gain_failed() -> None:
    run = _run(recovery_attempted=True, recovery_labels_resolved=0)
    ex = _example()
    assert recovery_gain(run, ex)["score"] == 0.0


def test_latency_under_threshold() -> None:
    run = _run(elapsed_ms=3000)
    ex = _example()
    assert latency(run, ex)["score"] == 1.0


def test_latency_over_threshold() -> None:
    run = _run(elapsed_ms=20_000)
    ex = _example()
    assert latency(run, ex)["score"] == 0.0


# ---------------------------------------------------------------------------
# Aggregate thresholds
# ---------------------------------------------------------------------------


def test_aggregate_thresholds_pass() -> None:
    runs = [
        {
            "value_recall": {"score": 1.0},
            "value_precision": {"score": 1.0},
            "outcome_correct": {"score": 1.0},
        },
        {
            "value_recall": {"score": 0.9},
            "value_precision": {"score": 0.95},
            "outcome_correct": {"score": 0.8},
        },
    ]
    result = aggregate_thresholds(runs)
    assert result["score"] == 1.0


def test_aggregate_thresholds_fail_low_recall() -> None:
    runs = [
        {
            "value_recall": {"score": 0.5},
            "value_precision": {"score": 1.0},
            "outcome_correct": {"score": 1.0},
        },
    ]
    result = aggregate_thresholds(runs)
    assert result["score"] == 0.0


def test_aggregate_thresholds_fail_low_outcome() -> None:
    runs = [
        {
            "value_recall": {"score": 1.0},
            "value_precision": {"score": 1.0},
            "outcome_correct": {"score": 0.5},
        },
    ]
    result = aggregate_thresholds(runs)
    assert result["score"] == 0.0

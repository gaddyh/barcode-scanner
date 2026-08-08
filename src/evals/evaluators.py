"""Evaluators for offline barcode-scanner evaluation.

Each evaluator receives a LangSmith run (a ``Run`` object with ``.outputs``
containing an ``IngestResult`` serialized as a dict) and an example, and
returns ``{"score": float, "comment": str}``.

Existing evaluators (moved from ``tests/eval/runner.py``):
    value_recall, value_precision, outcome_correct, count_exact

New evaluators:
    recovery_gain — did recovery increase the match count?
    latency — score = 1.0 if under threshold, else 0.0
"""

from __future__ import annotations

from typing import Any

# Latency threshold in milliseconds. Runs slower than this score 0.
LATENCY_THRESHOLD_MS = 10_000


def _outputs(run: Any) -> dict[str, Any]:
    """Extract the outputs dict from a LangSmith Run or a plain dict.

    LangSmith passes ``Run`` objects (pydantic models with ``.outputs``),
    while tests pass plain dicts with ``"outputs"`` key.
    """
    if hasattr(run, "outputs"):
        return run.outputs or {}
    if isinstance(run, dict):
        return run.get("outputs", run)
    return {}


def _found_values(prediction: dict[str, Any]) -> list[str]:
    """Extract barcode values from the prediction.

    Reads ``items`` (IngestResult shape) with fallback to ``found``
    (legacy dict shape) for backward compatibility.
    """
    items = prediction.get("items", prediction.get("found", []))
    return [f.get("barcode_value", "") for f in items]


def value_recall(run: Any, example: Any) -> dict[str, float | str]:
    """Fraction of expected decoded barcodes found by the pipeline."""
    inputs = example.inputs if hasattr(example, "inputs") else example["inputs"]
    expected = set(inputs["expected_values"])
    if not expected:
        return {"score": 1.0, "comment": "no expected decoded barcodes"}
    found = set(_found_values(_outputs(run)))
    hits = expected & found
    score = len(hits) / len(expected)
    return {
        "score": score,
        "comment": (
            f"found {len(hits)}/{len(expected)} expected values: "
            f"missing={sorted(expected - found)}"
        ),
    }


def value_precision(run: Any, example: Any) -> dict[str, float | str]:
    """Fraction of found barcodes that match an expected value."""
    inputs = example.inputs if hasattr(example, "inputs") else example["inputs"]
    expected = set(inputs["expected_values"])
    found = _found_values(_outputs(run))
    if not found:
        return {"score": 1.0, "comment": "no found barcodes (vacuously precise)"}
    matches = sum(1 for v in found if v in expected)
    score = matches / len(found)
    return {
        "score": score,
        "comment": f"{matches}/{len(found)} found values match expected",
    }


def outcome_correct(run: Any, example: Any) -> dict[str, float | str]:
    """Did the pipeline report the right outcome?"""
    inputs = example.inputs if hasattr(example, "inputs") else example["inputs"]
    expected_outcome = inputs["expected_outcome"]
    actual_outcome = _outputs(run).get("status", "failed")
    score = 1.0 if actual_outcome == expected_outcome else 0.0
    return {
        "score": score,
        "comment": f"expected={expected_outcome} actual={actual_outcome}",
    }


def count_exact(run: Any, example: Any) -> dict[str, float | str]:
    """found_count == expected decoded count."""
    inputs = example.inputs if hasattr(example, "inputs") else example["inputs"]
    expected_count = inputs["expected_decoded_count"]
    actual_count = len(_outputs(run).get("items", []))
    score = 1.0 if actual_count == expected_count else 0.0
    return {
        "score": score,
        "comment": f"expected_found={expected_count} actual_found={actual_count}",
    }


def recovery_gain(run: Any, example: Any) -> dict[str, float | str]:
    """Did recovery increase the match count?

    Measures recovery value: score = 1.0 if recovery was attempted AND
    resolved at least one label, 0.0 otherwise. When recovery was not
    attempted, the score is 1.0 (vacuously — no recovery needed).
    """
    metrics = _outputs(run).get("metrics", {})
    attempted = metrics.get("recovery_attempted", False)
    resolved = metrics.get("recovery_labels_resolved", 0)

    if not attempted:
        return {"score": 1.0, "comment": "recovery not needed"}
    if resolved > 0:
        return {"score": 1.0, "comment": f"recovery resolved {resolved} label(s)"}
    return {"score": 0.0, "comment": "recovery attempted but resolved 0 labels"}


def latency(run: Any, example: Any) -> dict[str, float | str]:
    """Score = 1.0 if latency under threshold, else 0.0.

    Measures speed regression. The threshold is configurable via
    ``LATENCY_THRESHOLD_MS``.
    """
    metrics = _outputs(run).get("metrics", {})
    elapsed_ms = metrics.get("elapsed_ms", 0)

    if elapsed_ms <= LATENCY_THRESHOLD_MS:
        return {"score": 1.0, "comment": f"latency={elapsed_ms}ms (under {LATENCY_THRESHOLD_MS}ms)"}
    return {"score": 0.0, "comment": f"latency={elapsed_ms}ms (over {LATENCY_THRESHOLD_MS}ms)"}


# ---------------------------------------------------------------------------
# Summary evaluator — aggregate pass/fail thresholds
# ---------------------------------------------------------------------------


def aggregate_thresholds(runs: list[dict[str, Any]]) -> dict[str, float | str]:
    """Soft aggregate thresholds: mean recall >= 0.90, mean precision >= 0.95."""
    recalls = [r["value_recall"]["score"] for r in runs if "value_recall" in r]
    precisions = [r["value_precision"]["score"] for r in runs if "value_precision" in r]
    outcomes = [r["outcome_correct"]["score"] for r in runs if "outcome_correct" in r]

    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
    mean_precision = sum(precisions) / len(precisions) if precisions else 0.0
    outcome_accuracy = sum(outcomes) / len(outcomes) if outcomes else 0.0

    passed = mean_recall >= 0.90 and mean_precision >= 0.95 and outcome_accuracy >= 0.80

    return {
        "score": 1.0 if passed else 0.0,
        "comment": (
            f"mean_recall={mean_recall:.3f} mean_precision={mean_precision:.3f} "
            f"outcome_accuracy={outcome_accuracy:.3f} passed={passed}"
        ),
    }

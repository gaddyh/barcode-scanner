"""Offline evaluation harness for the barcode-scanner pipeline.

Runs ``analyze_image()`` on the ground-truth dataset (``dataset.json``) and
scores each result with LangSmith ``evaluate()``:

- **value_recall**    — fraction of expected decoded barcodes found by the pipeline.
- **value_precision** — fraction of found barcodes that match an expected value.
- **outcome_correct** — did the pipeline report the right outcome?
  ``complete`` when every expected decoded barcode was found, else
  ``needs_better_photo``.
- **count_exact**     — found_count == expected decoded count.

The harness is **offline** for the scanner (no Gemini calls when
``GEMINI_API_KEY`` is absent — the audit fails and the outcome is
``retryable_error``, which the evaluators score against the expected outcome).
For a live run that exercises Gemini, set ``GEMINI_API_KEY`` and run:

    python -m tests.eval.runner

Results are uploaded to LangSmith as an experiment under
``LANGSMITH_PROJECT``. Set ``LANGSMITH_TRACING=true`` and
``LANGSMITH_API_KEY`` to enable upload.

Usage:

    python -m tests.eval.runner                 # live (charged, needs GEMINI_API_KEY)
    python -m tests.eval.runner --scanner-only  # scanner-only, no Gemini
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from langsmith import evaluate

from app.services.analyze import analyze_image

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

DATASET_PATH = Path(__file__).resolve().parent / "dataset.json"
SAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "samples"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset() -> list[dict[str, Any]]:
    """Load the ground-truth dataset and resolve each image to an absolute path."""
    with DATASET_PATH.open() as f:
        data = json.load(f)

    examples: list[dict[str, Any]] = []
    for img_entry in data["images"]:
        image_name = img_entry["image"]
        image_path = SAMPLES_DIR / image_name
        if not image_path.exists():
            logger.warning("Sample image missing, skipping: %s", image_path)
            continue

        decoded_boxes = [b for b in img_entry["boxes"] if b.get("status") == "decoded"]
        expected_values = [b["value"] for b in decoded_boxes if b.get("value")]
        expected_unique = sorted(set(expected_values))

        examples.append({
            "image_name": image_name,
            "image_path": str(image_path),
            "expected_barcode_symbol_count": img_entry["expected_barcode_symbol_count"],
            "expected_decoded_count": len(decoded_boxes),
            "expected_values": expected_values,
            "expected_unique_values": expected_unique,
            "expected_unique_count": len(expected_unique),
            "expected_outcome": "complete",
        })
    return examples


# ---------------------------------------------------------------------------
# Target — the function LangSmith evaluate() calls per example
# ---------------------------------------------------------------------------


def _target(inputs: dict[str, Any]) -> dict[str, Any]:
    """Run analyze_image on one example and return the product result."""
    image_path = inputs["image_path"]
    logger.info("Analyzing %s", inputs["image_name"])
    try:
        result = analyze_image(image_path)
    except Exception as exc:
        logger.exception("analyze_image failed for %s", inputs["image_name"])
        return {
            "outcome": "retryable_error",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "found": [],
            "missing": [],
            "unassigned": [],
            "summary": {
                "visible_label_count": 0,
                "found_count": 0,
                "missing_count": 0,
                "unassigned_count": 0,
                "all_found": False,
            },
        }
    return {
        "outcome": result.get("outcome", "retryable_error"),
        "found": result.get("found", []),
        "missing": result.get("missing", []),
        "unassigned": result.get("unassigned", []),
        "summary": result.get("summary", {}),
    }


# ---------------------------------------------------------------------------
# Evaluators — each returns {"score": float, "comment": str}
# ---------------------------------------------------------------------------


def _found_values(prediction: dict[str, Any]) -> list[str]:
    return [f.get("barcode_value", "") for f in prediction.get("found", [])]


def value_recall(run: dict[str, Any], example: Any) -> dict[str, float | str]:
    """Fraction of expected decoded barcodes found by the pipeline."""
    inputs = example.inputs if hasattr(example, "inputs") else example["inputs"]
    expected = set(inputs["expected_values"])
    if not expected:
        return {"score": 1.0, "comment": "no expected decoded barcodes"}
    found = set(_found_values(run.get("outputs", run)))
    hits = expected & found
    score = len(hits) / len(expected)
    return {
        "score": score,
        "comment": (
            f"found {len(hits)}/{len(expected)} expected values: "
            f"missing={sorted(expected - found)}"
        ),
    }


def value_precision(run: dict[str, Any], example: Any) -> dict[str, float | str]:
    """Fraction of found barcodes that match an expected value."""
    inputs = example.inputs if hasattr(example, "inputs") else example["inputs"]
    expected = set(inputs["expected_values"])
    found = _found_values(run.get("outputs", run))
    if not found:
        return {"score": 1.0, "comment": "no found barcodes (vacuously precise)"}
    matches = sum(1 for v in found if v in expected)
    score = matches / len(found)
    return {
        "score": score,
        "comment": f"{matches}/{len(found)} found values match expected",
    }


def outcome_correct(run: dict[str, Any], example: Any) -> dict[str, float | str]:
    """Did the pipeline report the right outcome?"""
    inputs = example.inputs if hasattr(example, "inputs") else example["inputs"]
    expected_outcome = inputs["expected_outcome"]
    actual_outcome = run.get("outputs", run).get("outcome", "retryable_error")
    score = 1.0 if actual_outcome == expected_outcome else 0.0
    return {
        "score": score,
        "comment": f"expected={expected_outcome} actual={actual_outcome}",
    }


def count_exact(run: dict[str, Any], example: Any) -> dict[str, float | str]:
    """found_count == expected decoded count."""
    inputs = example.inputs if hasattr(example, "inputs") else example["inputs"]
    expected_count = inputs["expected_decoded_count"]
    actual_count = run.get("outputs", run).get("summary", {}).get("found_count", 0)
    score = 1.0 if actual_count == expected_count else 0.0
    return {
        "score": score,
        "comment": f"expected_found={expected_count} actual_found={actual_count}",
    }


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


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_eval(*, scanner_only: bool = False, experiment_prefix: str = "barcode-scanner") -> Any:
    """Run the LangSmith evaluate() harness."""
    examples = load_dataset()
    if not examples:
        print("No eval examples — samples/ missing or dataset.json empty.", file=sys.stderr)
        return None

    if scanner_only:
        # Disable Gemini so the audit fails fast and only the scanner is scored.
        os.environ["GEMINI_API_KEY"] = os.environ.get("GEMINI_API_KEY", "")
        # The audit will raise ValueError without a key; that's expected.

    print(
        f"Running eval on {len(examples)} image(s) "
        f"({'scanner-only' if scanner_only else 'live Gemini'})",
        file=sys.stderr,
    )

    # Build LangSmith examples from our dataset.
    ls_examples = [
        {
            "inputs": ex,
            "outputs": {"expected_outcome": ex["expected_outcome"]},
        }
        for ex in examples
    ]

    results = evaluate(
        _target,
        data=ls_examples,
        evaluators=[value_recall, value_precision, outcome_correct, count_exact],
        summary_evaluators=[aggregate_thresholds],
        experiment_prefix=experiment_prefix,
        description="Barcode-scanner happy-path offline evaluation",
        max_concurrency=1,  # scanner is CPU-bound; avoid oversubscription
        blocking=True,
    )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tests.eval.runner",
        description="LangSmith offline evaluation for the barcode-scanner pipeline.",
    )
    parser.add_argument(
        "--scanner-only",
        action="store_true",
        help="Run without Gemini (audit fails → retryable_error for every image).",
    )
    parser.add_argument(
        "--experiment-prefix",
        default="barcode-scanner",
        help="LangSmith experiment name prefix.",
    )
    args = parser.parse_args(argv)

    run_eval(scanner_only=args.scanner_only, experiment_prefix=args.experiment_prefix)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

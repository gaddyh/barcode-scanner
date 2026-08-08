"""Offline evaluation runner — calls ingest_one() and scores with evaluators.

This is the new home for the eval harness. It calls ``ingest_one()``
(the typed boundary from Milestone 1) instead of ``analyze_image()``
(the legacy dict-returning function). The evaluators read the typed
``IngestResult`` fields directly (``items``, ``status``, ``metrics``)
instead of the old dict shape (``found``, ``outcome``, ``summary``).

Usage::

    python -m src.evals.runner                 # live (charged, needs GEMINI_API_KEY)
    python -m src.evals.runner --scanner-only  # scanner-only, no Gemini

Results upload to LangSmith as an experiment under ``LANGSMITH_PROJECT``.
Set ``LANGSMITH_TRACING=true`` and ``LANGSMITH_API_KEY`` to enable upload.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any
from uuid import uuid4

from langsmith import Client, evaluate

from app.models.upload import generate_upload_id
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
from src.ingest import ingest_one
from src.runtime import RunContext

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


# ---------------------------------------------------------------------------
# Target — the function LangSmith evaluate() calls per example
# ---------------------------------------------------------------------------


def _target(inputs: dict[str, Any]) -> dict[str, Any]:
    """Run ingest_one on one example and return the serialized IngestResult."""
    image_path = inputs["image_path"]
    logger.info("Analyzing %s", inputs["image_name"])

    ctx = RunContext(
        run_id=str(uuid4()),
        session_id=generate_upload_id(),
        user_id=None,
        source="eval",
        metadata={"filename": inputs["image_name"]},
    )

    try:
        result = ingest_one(image_path, ctx)
    except Exception as exc:
        logger.exception("ingest_one failed for %s", inputs["image_name"])
        # Return a minimal failed-result dict so evaluators can still score it.
        return {
            "status": "failed",
            "items": [],
            "missing": [],
            "unassigned": [],
            "issues": [],
            "metrics": {"elapsed_ms": 0, "scanner_count": 0, "vision_count": 0,
                        "recovery_attempted": False, "recovery_labels_tried": 0,
                        "recovery_barcodes_found": 0, "recovery_labels_resolved": 0},
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }

    return result.model_dump(mode="json")


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

    # Upload examples to a LangSmith dataset. The dataset is created once
    # and reused across runs so experiment history is preserved. Examples
    # are refreshed each run to pick up dataset.json changes without
    # accumulating duplicates.
    dataset_name = "barcode-scanner-eval"
    client = Client()
    try:
        client.read_dataset(dataset_name=dataset_name)
    except Exception:
        client.create_dataset(
            dataset_name=dataset_name,
            description="Barcode-scanner ground-truth dataset (from tests/eval/dataset.json)",
        )

    # Clear stale examples so re-running eval doesn't accumulate duplicates.
    existing = list(client.list_examples(dataset_name=dataset_name))
    if existing:
        client.delete_examples(example_ids=[e.id for e in existing])

    for ex in examples:
        client.create_example(
            inputs=ex,
            outputs={"expected_outcome": ex["expected_outcome"]},
            dataset_name=dataset_name,
        )

    results = evaluate(
        _target,
        data=dataset_name,
        evaluators=[
            value_recall,
            value_precision,
            outcome_correct,
            count_exact,
            recovery_gain,
            latency,
        ],
        summary_evaluators=[aggregate_thresholds],
        experiment_prefix=experiment_prefix,
        description="Barcode-scanner offline evaluation (ingest_one)",
        max_concurrency=1,  # scanner is CPU-bound; avoid oversubscription
        blocking=True,
    )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="src.evals.runner",
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

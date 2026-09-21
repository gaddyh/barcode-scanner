"""Offline regression gate — scanner-only, no network, no Gemini, no LangSmith.

Runs ``BarcodeScanner().scan_bytes()`` on each image in the canonical
dataset, scores with multiset evaluators, and compares against a frozen
baseline. Exits non-zero on quality regression.

Usage::

    python -m src.evals.regression              # compare against baseline
    python -m src.evals.regression --write-baseline  # freeze current results

The frozen baseline lives at ``tests/eval/baseline_frozen.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from src.evals.datasets import load_dataset
from src.evals.evaluators import barcode_accuracy, occurrence_precision, occurrence_recall
from src.ingest.scanner import BarcodeScanner

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BASELINE_PATH = _REPO_ROOT / "tests" / "eval" / "baseline_frozen.json"


def _run_scanner(image_path: str) -> dict[str, Any]:
    """Run the scanner on one image and return a prediction dict."""
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    scanner = BarcodeScanner()
    t0 = time.perf_counter()
    detections = scanner.scan_bytes(image_bytes)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    return {
        "items": [{"barcode_value": d.value} for d in detections],
        "status": "complete" if detections else "needs_user_input",
        "metrics": {
            "elapsed_ms": elapsed_ms,
            "scanner_count": len(detections),
        },
    }


def run_regression() -> list[dict[str, Any]]:
    """Run the scanner on every case and return per-case results."""
    examples = load_dataset()
    if not examples:
        print("No eval examples — samples/ missing or dataset empty.", file=sys.stderr)
        return []

    results: list[dict[str, Any]] = []
    for ex in examples:
        prediction = _run_scanner(ex["image_path"])
        # Build a fake example for the evaluators (they read .inputs or ["inputs"])
        fake_example = {"inputs": ex}

        recall = occurrence_recall(prediction, fake_example)
        precision = occurrence_precision(prediction, fake_example)
        accuracy = barcode_accuracy(prediction, fake_example)

        results.append({
            "id": ex["id"],
            "image_name": ex["image_name"],
            "expected_count": ex["expected_count"],
            "found_count": len(prediction["items"]),
            "occurrence_recall": recall["score"],
            "occurrence_precision": precision["score"],
            "barcode_accuracy": accuracy["score"],
            "elapsed_ms": prediction["metrics"]["elapsed_ms"],
            "comment_recall": recall["comment"],
            "comment_precision": precision["comment"],
        })
    return results


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate metrics from per-case results."""
    n = len(results)
    if n == 0:
        return {"n": 0}

    latencies = sorted(r["elapsed_ms"] for r in results)
    p50_idx = n // 2
    p95_idx = min(n - 1, max(0, int(0.95 * n) - 1))

    return {
        "n": n,
        "mean_occurrence_recall": sum(r["occurrence_recall"] for r in results) / n,
        "mean_occurrence_precision": sum(r["occurrence_precision"] for r in results) / n,
        "barcode_accuracy": sum(r["barcode_accuracy"] for r in results) / n,
        "total_matched_occurrences": sum(
            r["occurrence_recall"] * r["expected_count"] for r in results
        ),
        "total_expected_occurrences": sum(r["expected_count"] for r in results),
        "total_found_occurrences": sum(r["found_count"] for r in results),
        "p50_latency_ms": latencies[p50_idx],
        "p95_latency_ms": latencies[p95_idx],
    }


def _format_results(results: list[dict[str, Any]], agg: dict[str, Any]) -> str:
    """Format results as a human-readable report."""
    lines = [
        "",
        "=== Barcode Scanner Regression Report ===",
        f"Cases: {agg['n']}",
        f"Mean occurrence recall:    {agg['mean_occurrence_recall']:.3f}",
        f"Mean occurrence precision: {agg['mean_occurrence_precision']:.3f}",
        f"Barcode accuracy (strict): {agg['barcode_accuracy']:.3f}",
        f"Matched occurrences:        {agg['total_matched_occurrences']:.0f}/{agg['total_expected_occurrences']}",
        f"Found occurrences:         {agg['total_found_occurrences']}",
        f"P50 latency: {agg['p50_latency_ms']}ms  P95 latency: {agg['p95_latency_ms']}ms",
        "",
        "Per-case:",
    ]
    for r in results:
        status = "PASS" if r["barcode_accuracy"] == 1.0 else "FAIL"
        lines.append(
            f"  [{status}] {r['id']:<25} "
            f"recall={r['occurrence_recall']:.2f} prec={r['occurrence_precision']:.2f} "
            f"found={r['found_count']}/{r['expected_count']} "
            f"latency={r['elapsed_ms']}ms"
        )
    lines.append("")
    return "\n".join(lines)


def compare_against_baseline(
    results: list[dict[str, Any]],
    agg: dict[str, Any],
    *,
    baseline_path: Path = BASELINE_PATH,
) -> int:
    """Compare current results against the frozen baseline.

    Returns exit code: 0 = no regression, 1 = regression detected.
    """
    if not baseline_path.exists():
        print(
            f"WARNING: No baseline at {baseline_path}. "
            f"Run with --write-baseline to freeze current results.",
            file=sys.stderr,
        )
        return 0

    with baseline_path.open() as f:
        baseline = json.load(f)

    baseline_agg = baseline["aggregate"]
    errors: list[str] = []

    # Quality regressions (hard gate)
    if agg["mean_occurrence_recall"] < baseline_agg["mean_occurrence_recall"]:
        errors.append(
            f"REGRESSION: mean_occurrence_recall "
            f"{agg['mean_occurrence_recall']:.3f} < "
            f"{baseline_agg['mean_occurrence_recall']:.3f}"
        )
    if agg["mean_occurrence_precision"] < baseline_agg["mean_occurrence_precision"]:
        errors.append(
            f"REGRESSION: mean_occurrence_precision "
            f"{agg['mean_occurrence_precision']:.3f} < "
            f"{baseline_agg['mean_occurrence_precision']:.3f}"
        )

    # Per-case regression (hard gate)
    baseline_cases = {c["id"]: c for c in baseline["cases"]}
    for r in results:
        bc = baseline_cases.get(r["id"])
        if bc is None:
            continue
        if r["barcode_accuracy"] < bc["barcode_accuracy"]:
            errors.append(
                f"REGRESSION: case '{r['id']}' barcode_accuracy "
                f"{r['barcode_accuracy']:.2f} < {bc['barcode_accuracy']:.2f}"
            )

    # Latency (informational only — not gated)
    if agg["p95_latency_ms"] > baseline_agg["p95_latency_ms"] * 1.5:
        print(
            f"INFO: P95 latency regression: "
            f"{agg['p95_latency_ms']}ms > {baseline_agg['p95_latency_ms']}ms (1.5x threshold)",
            file=sys.stderr,
        )

    if errors:
        print("\n=== REGRESSION DETECTED ===", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1

    print("\n=== No regression — quality matches baseline ===", file=sys.stderr)
    return 0


def write_baseline(results: list[dict[str, Any]], agg: dict[str, Any]) -> None:
    """Write the current results as the frozen baseline."""
    baseline = {
        "aggregate": agg,
        "cases": [
            {
                "id": r["id"],
                "expected_count": r["expected_count"],
                "found_count": r["found_count"],
                "occurrence_recall": r["occurrence_recall"],
                "occurrence_precision": r["occurrence_precision"],
                "barcode_accuracy": r["barcode_accuracy"],
                "elapsed_ms": r["elapsed_ms"],
            }
            for r in results
        ],
    }
    with BASELINE_PATH.open("w") as f:
        json.dump(baseline, f, indent=2)
        f.write("\n")
    print(f"Baseline written to {BASELINE_PATH}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="src.evals.regression",
        description="Offline scanner-only regression gate (no Gemini, no LangSmith, no network).",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Freeze current results as the baseline (overwrites baseline_frozen.json).",
    )
    args = parser.parse_args(argv)

    results = run_regression()
    if not results:
        return 1

    agg = _aggregate(results)
    report = _format_results(results, agg)
    print(report)

    if args.write_baseline:
        write_baseline(results, agg)
        return 0

    return compare_against_baseline(results, agg)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

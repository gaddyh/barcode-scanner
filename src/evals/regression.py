"""Offline regression gate — scanner-only or full-pipeline.

Scanner-only mode runs ``BarcodeScanner().scan_bytes()`` on each image
in the canonical dataset. No Gemini, no LangSmith, no network.

Full-pipeline mode runs ``analyze_image()`` (scanner + Gemini audit +
reconciliation + Gemini-guided recovery). Requires ``GEMINI_API_KEY``.
Gemini is nondeterministic, so the full-pipeline baseline is
observational — use it to track improvements, not as a hard gate.

Both modes score with multiset evaluators and compare against a frozen
baseline. Scanner-only exits non-zero on quality regression.

Usage::

    python -m src.evals.regression                       # scanner-only, gate
    python -m src.evals.regression --write-baseline       # freeze scanner-only
    python -m src.evals.regression --full-pipeline         # full pipeline, observe
    python -m src.evals.regression --full-pipeline --write-baseline  # freeze full

Baselines:
    Scanner-only:      tests/eval/baseline_frozen.json
    Full-pipeline:     tests/eval/baseline_full_pipeline.json
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
BASELINE_SCANNER_PATH = _REPO_ROOT / "tests" / "eval" / "baseline_frozen.json"
BASELINE_FULL_PIPELINE_PATH = _REPO_ROOT / "tests" / "eval" / "baseline_full_pipeline.json"


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


def _run_full_pipeline(image_path: str) -> dict[str, Any]:
    """Run the full pipeline (scanner + Gemini + recovery) on one image."""
    from src.ingest.analyze import analyze_image

    t0 = time.perf_counter()
    result = analyze_image(image_path)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    found = result.get("found", [])
    # Extract Gemini audit latency from the pipeline summary (if available).
    audit_latency_ms = result.get("summary", {}).get("audit_latency_ms", 0)
    return {
        "items": [{"barcode_value": f.get("barcode_value", "")} for f in found],
        "status": result.get("outcome", "failed"),
        "metrics": {
            "elapsed_ms": elapsed_ms,
            "scanner_count": len(found),
            "audit_latency_ms": audit_latency_ms,
        },
    }


def _setup_gemini_cache(args: argparse.Namespace) -> None:
    """Configure Gemini audit cache mode in the graph module."""
    from src.evals.gemini_cache import GeminiAuditCache
    from src.ingest import graph

    if not args.full_pipeline:
        return

    if args.replay_gemini:
        cache = GeminiAuditCache()
        graph.set_audit_cache_mode("replay", cache)
        print(f"Replay mode: using {len(cache)} cached Gemini audits", file=sys.stderr)
    elif args.cache_gemini:
        cache = GeminiAuditCache()
        graph.set_audit_cache_mode("capture", cache)
        print(f"Capture mode: recording Gemini audits to {cache.cache_path}", file=sys.stderr)


def _save_gemini_cache(args: argparse.Namespace) -> None:
    """Save the Gemini audit cache if in capture mode."""
    if not args.full_pipeline or not args.cache_gemini:
        return

    from src.ingest import graph

    if graph._audit_cache_store is not None:
        graph._audit_cache_store.save()


def run_regression(*, full_pipeline: bool = False) -> list[dict[str, Any]]:
    """Run the scanner (or full pipeline) on every case and return per-case results."""
    examples = load_dataset()
    if not examples:
        print("No eval examples — samples/ missing or dataset empty.", file=sys.stderr)
        return []

    runner = _run_full_pipeline if full_pipeline else _run_scanner

    results: list[dict[str, Any]] = []
    for ex in examples:
        prediction = runner(ex["image_path"])
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
            "audit_latency_ms": prediction["metrics"].get("audit_latency_ms", 0),
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

    audit_latencies = sorted(r.get("audit_latency_ms", 0) for r in results)
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
        "p50_audit_latency_ms": audit_latencies[p50_idx],
        "p95_audit_latency_ms": audit_latencies[p95_idx],
    }


def _format_results(results: list[dict[str, Any]], agg: dict[str, Any]) -> str:
    """Format results as a human-readable report."""
    has_audit = "audit_latency_ms" in results[0] if results else False
    lines = [
        "",
        "=== Barcode Scanner Regression Report ===",
        f"Cases: {agg['n']}",
    ]
    if has_audit:
        from src.ingest.vision import VISION_PROMPT_VERSION
        lines.append(f"Vision prompt: {VISION_PROMPT_VERSION}")
    lines += [
        f"Mean occurrence recall:    {agg['mean_occurrence_recall']:.3f}",
        f"Mean occurrence precision: {agg['mean_occurrence_precision']:.3f}",
        f"Barcode accuracy (strict): {agg['barcode_accuracy']:.3f}",
        f"Matched occurrences:        {agg['total_matched_occurrences']:.0f}/{agg['total_expected_occurrences']}",
        f"Found occurrences:         {agg['total_found_occurrences']}",
        f"P50 latency: {agg['p50_latency_ms']}ms  P95 latency: {agg['p95_latency_ms']}ms",
    ]
    if has_audit:
        lines.append(
            f"P50 audit: {agg['p50_audit_latency_ms']}ms  "
            f"P95 audit: {agg['p95_audit_latency_ms']}ms"
        )
    lines += ["", "Per-case:"]
    for r in results:
        status = "PASS" if r["barcode_accuracy"] == 1.0 else "FAIL"
        audit_str = ""
        if has_audit:
            audit_str = f" audit={r.get('audit_latency_ms', 0)}ms"
        lines.append(
            f"  [{status}] {r['id']:<25} "
            f"recall={r['occurrence_recall']:.2f} prec={r['occurrence_precision']:.2f} "
            f"found={r['found_count']}/{r['expected_count']} "
            f"latency={r['elapsed_ms']}ms{audit_str}"
        )
    lines.append("")
    return "\n".join(lines)


def compare_against_baseline(
    results: list[dict[str, Any]],
    agg: dict[str, Any],
    *,
    baseline_path: Path = BASELINE_SCANNER_PATH,
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


def write_baseline(
    results: list[dict[str, Any]],
    agg: dict[str, Any],
    *,
    baseline_path: Path = BASELINE_SCANNER_PATH,
) -> None:
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
                "audit_latency_ms": r.get("audit_latency_ms", 0),
            }
            for r in results
        ],
    }
    with baseline_path.open("w") as f:
        json.dump(baseline, f, indent=2)
        f.write("\n")
    print(f"Baseline written to {baseline_path}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="src.evals.regression",
        description="Offline regression gate (scanner-only or full-pipeline).",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Freeze current results as the baseline.",
    )
    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Run the full pipeline (scanner + Gemini + recovery). "
        "Requires GEMINI_API_KEY (unless --replay-gemini). "
        "Observational — not a hard gate.",
    )
    parser.add_argument(
        "--cache-gemini",
        action="store_true",
        help="Capture Gemini audit results to cache for deterministic replay. "
        "Only meaningful with --full-pipeline.",
    )
    parser.add_argument(
        "--replay-gemini",
        action="store_true",
        help="Replay cached Gemini audit results instead of calling Gemini. "
        "Makes full-pipeline eval deterministic. "
        "Only meaningful with --full-pipeline.",
    )
    args = parser.parse_args(argv)

    baseline_path = (
        BASELINE_FULL_PIPELINE_PATH if args.full_pipeline else BASELINE_SCANNER_PATH
    )

    _setup_gemini_cache(args)

    results = run_regression(full_pipeline=args.full_pipeline)
    if not results:
        return 1

    _save_gemini_cache(args)

    agg = _aggregate(results)
    report = _format_results(results, agg)
    print(report)

    if args.write_baseline:
        write_baseline(results, agg, baseline_path=baseline_path)
        return 0

    return compare_against_baseline(results, agg, baseline_path=baseline_path)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

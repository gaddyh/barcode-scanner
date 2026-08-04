"""Benchmark runner for the full warehouse pipeline.

Two modes:

- ``run_live_benchmark`` — runs the real ``pipeline_path()`` (scanner + dual
  Gemini audit + selection + reconciliation + recovery) for each image,
  ``runs_per_image`` times. Slow, charged, network-dependent. Optionally
  captures both audit candidates into ``snapshots/pipeline_responses.json`` for
  offline replay.

- ``run_snapshot_benchmark`` — replays captured audit candidates through the
  current scanner + ``select_best_spatial_audit`` + ``reconcile_with_recovery``.
  Deterministic. Exercises scanner, selection, reconciliation, and recovery
  code changes against frozen Gemini inputs.

The CLI is::

    python -m tests.benchmark_warehouse.runner                 # live, 5 runs
    python -m tests.benchmark_warehouse.runner --runs 3
    python -m tests.benchmark_warehouse.runner --capture-snapshots
    python -m tests.benchmark_warehouse.runner --model gemini-2.5-flash

Exit 0 if first-run expectations pass; 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.services.barcode_scanner import BarcodeScanner
from app.services.gemini_box_audit import DEFAULT_MODEL
from app.services.pipeline import (
    pipeline_path,
    reconcile_with_recovery,
    select_best_spatial_audit,
)
from tests.benchmark_warehouse.metrics import (
    aggregate_runs,
    compute_run_metrics,
)
from tests.benchmark_warehouse.models import (
    BenchResult,
    RunMetrics,
    WarehouseDataset,
)

DATASET_PATH = Path(__file__).resolve().parent / "dataset.json"
SNAPSHOTS_PATH = Path(__file__).resolve().parent / "snapshots" / "pipeline_responses.json"
SAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "samples"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset(path: Path = DATASET_PATH) -> WarehouseDataset:
    raw = json.loads(path.read_text())
    return WarehouseDataset.model_validate(raw)


def _samples_path(image: str) -> Path:
    return SAMPLES_DIR / image


# ---------------------------------------------------------------------------
# Snapshot capture / replay
# ---------------------------------------------------------------------------


def _load_snapshots() -> dict[str, Any]:
    if not SNAPSHOTS_PATH.exists():
        raise FileNotFoundError(
            f"Snapshot file not found: {SNAPSHOTS_PATH}. Run "
            f"`python -m tests.benchmark_warehouse.runner --capture-snapshots` "
            f"once with GEMINI_API_KEY set to populate it."
        )
    return json.loads(SNAPSHOTS_PATH.read_text())


def _save_snapshots(store: dict[str, Any]) -> None:
    SNAPSHOTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOTS_PATH.write_text(json.dumps(store, indent=2))


def _audit_candidate_to_snapshot(candidate: dict[str, Any]) -> dict[str, Any]:
    """Serialize an audit result dict for snapshot storage.

    Only the ``spatial`` (labels + image dims) and ``status`` are kept —
    everything else is transient.
    """
    return {
        "status": candidate.get("status"),
        "spatial": candidate.get("spatial"),
    }


def _snapshot_to_audit_candidate(entry: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct an audit result dict from a snapshot entry."""
    return {
        "status": entry.get("status", "ok"),
        "spatial": entry.get("spatial", {}),
    }


# ---------------------------------------------------------------------------
# Live benchmark
# ---------------------------------------------------------------------------


def run_live_benchmark(
    *,
    runs_per_image: int = 5,
    capture_snapshots: bool = False,
    model: str | None = None,
) -> BenchResult:
    """Run the real pipeline for each image, multiple times.

    Shared by the CLI and the marked pytest test so behavior cannot diverge.
    """
    if runs_per_image < 1:
        raise ValueError("runs_per_image must be at least 1")

    dataset = load_dataset()
    scanner = BarcodeScanner()
    resolved_model = model

    snapshots: dict[str, Any] = {
        "version": 1,
        "captured_at": datetime.now(UTC).isoformat(),
        "model": resolved_model or DEFAULT_MODEL,
        "images": {},
    }
    captured_any = False

    run_metrics_by_image: dict[str, list[RunMetrics]] = {}

    for gt in dataset.images:
        image_path = _samples_path(gt.image)
        if not image_path.exists():
            print(
                f"WARNING: source image not found, skipping {gt.image}: {image_path}",
                file=sys.stderr,
            )
            continue

        runs: list[RunMetrics] = []

        for run_idx in range(runs_per_image):
            t0 = time.perf_counter()
            summary = pipeline_path(
                image_path,
                scanner,
                model=resolved_model,
                max_retries=3,
                retry_delay_seconds=1.0,
                dual_audit=True,
                return_audit_candidates=(capture_snapshots and run_idx == 0),
            )
            elapsed = time.perf_counter() - t0

            metrics = compute_run_metrics(gt.image, summary, gt, latency=elapsed)
            runs.append(metrics)

            if capture_snapshots and run_idx == 0 and gt.image not in snapshots["images"]:
                candidates = summary.get("audit_candidates", [])
                spatial = candidates[0].get("spatial", {}) if candidates else {}
                snapshots["images"][gt.image] = {
                    "image_width": spatial.get("image_width"),
                    "image_height": spatial.get("image_height"),
                    "audit_candidates": [
                        _audit_candidate_to_snapshot(c) for c in candidates
                    ],
                    "captured_scanner_detections": summary.get("scanner_detections", []),
                }
                captured_any = True

        run_metrics_by_image[gt.image] = runs

    if capture_snapshots and captured_any:
        _save_snapshots(snapshots)
        print(
            f"Saved snapshots for {len(snapshots['images'])} image(s) to "
            f"{SNAPSHOTS_PATH}",
            file=sys.stderr,
        )

    agg = aggregate_runs(run_metrics_by_image, dataset.images)
    return BenchResult(run_metrics_by_image=run_metrics_by_image, aggregate=agg)


# ---------------------------------------------------------------------------
# Snapshot benchmark
# ---------------------------------------------------------------------------


def run_snapshot_benchmark() -> BenchResult:
    """Replay captured audit candidates through the current scanner + selection
    + reconciliation + recovery. Deterministic.
    """
    dataset = load_dataset()
    store = _load_snapshots()
    snapshot_images: dict[str, Any] = store.get("images", {})

    scanner = BarcodeScanner()
    run_metrics_by_image: dict[str, list[RunMetrics]] = {}

    for gt in dataset.images:
        if gt.image not in snapshot_images:
            print(
                f"WARNING: no snapshot for {gt.image}, skipping",
                file=sys.stderr,
            )
            continue

        image_path = _samples_path(gt.image)
        if not image_path.exists():
            print(
                f"WARNING: source image not found, skipping {gt.image}: {image_path}",
                file=sys.stderr,
            )
            continue

        snap = snapshot_images[gt.image]
        image_width = snap["image_width"]
        image_height = snap["image_height"]
        candidates = [
            _snapshot_to_audit_candidate(c) for c in snap.get("audit_candidates", [])
        ]

        # Run the current scanner.
        scan_detections = scanner.scan_bytes(image_path.read_bytes())
        detection_dicts = [asdict(d) for d in scan_detections]

        # Build a scan_result-like dict for compute_run_metrics.
        values = [d["value"] for d in detection_dicts]
        scan_result = {
            "status": "found" if detection_dicts else "not_found",
            "count": len(detection_dicts),
            "barcodes": detection_dicts,
        }

        # Select the best audit candidate using the real selection algorithm.
        if candidates:
            audit_result = select_best_spatial_audit(detection_dicts, candidates)
        else:
            audit_result = {"status": "error", "spatial": {}}

        # Build the summary dict (same shape as pipeline_path).
        summary: dict[str, Any] = {
            "path": str(image_path),
            "scan_status": scan_result["status"],
            "audit_status": audit_result.get("status"),
        }

        scan_ok = scan_result["status"] in ("found", "not_found")
        audit_ok = audit_result.get("status") == "ok"

        barcodes: list[dict] = []
        if scan_ok:
            barcodes = detection_dicts
            summary["decoded_count"] = len(detection_dicts)
            summary["unique_values"] = sorted(set(values))
            summary["unique_value_count"] = len(set(values))
            summary["scanner_detections"] = barcodes

        if audit_ok:
            spatial = audit_result.get("spatial", {})
            labels = spatial.get("labels", [])
            summary["visible_labels"] = len(labels)
            summary["clear_labels"] = sum(
                1 for label in labels if label.get("status") == "clear"
            )
            summary["gemini_labels"] = labels

        if scan_ok and audit_ok:
            spatial = audit_result.get("spatial", {})
            labels = spatial.get("labels", [])

            reconciliation, recon_summary = reconcile_with_recovery(
                barcodes,
                labels,
                image_width=image_width,
                image_height=image_height,
                scanner=scanner,
                image_path=image_path,
            )
            summary.update(recon_summary)

            decoded = summary.get("decoded_count", 0)
            visible = summary.get("visible_labels", 0)
            summary["decoded_vs_visible"] = {
                "decoded": decoded,
                "visible": visible,
                "match": decoded == visible,
                "difference": decoded - visible,
                "matched_labels": reconciliation.matched_label_count,
                "all_labels_matched": reconciliation.all_labels_matched,
            }

        summary["ok"] = scan_ok and audit_ok

        metrics = compute_run_metrics(gt.image, summary, gt, latency=None)
        run_metrics_by_image[gt.image] = [metrics]

    agg = aggregate_runs(run_metrics_by_image, dataset.images)
    return BenchResult(run_metrics_by_image=run_metrics_by_image, aggregate=agg)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    return f"{seconds:.2f}s"


def print_report(result: BenchResult) -> None:
    """Print a per-image summary table + aggregate totals."""
    header = (
        f"{'Image':<30} {'Scan':>5} {'Visible':>8} {'Baseline':>9} "
        f"{'Final':>6} {'Rec':>4} {'Unasgn':>7} {'Reason':>22} {'Latency':>8}"
    )
    print(header)
    print("-" * len(header))

    for _image, runs in result.run_metrics_by_image.items():
        if not runs:
            continue
        r = runs[0]
        print(
            f"{r.image:<30} {r.scanner_detection_count:>5} "
            f"{r.visible_label_count or 0:>8} "
            f"{r.baseline_matched_labels or 0:>9} "
            f"{r.final_matched_labels or 0:>6} {r.recovered_label_count:>4} "
            f"{r.unassigned_scanner_detections:>7} {r.failure_reason:>22} "
            f"{_fmt_seconds(r.latency_seconds):>8}"
        )

    print("-" * len(header))
    agg = result.aggregate
    print(
        f"{'TOTAL':<30} {agg.actual_scanner_detections:>5} "
        f"{agg.actual_visible_labels:>8} {agg.actual_baseline_matched:>9} "
        f"{agg.actual_final_matched:>6} {agg.actual_recovered:>4} "
        f"{agg.unassigned_scanner_detection_count:>7} "
        f"{'':>22} {_fmt_seconds(agg.median_latency):>8}"
    )
    print()
    print(f"baseline_recall={agg.baseline_recall:.2%} "
          f"final_recall={agg.final_recall:.2%} "
          f"recovery_uplift={agg.recovery_uplift} "
          f"fp_violations={agg.false_positive_violation_count} "
          f"reason_mismatches={agg.failure_reason_mismatch_count}")
    print(f"first_runs_match_expected={agg.all_first_runs_match_expected}")
    if agg.all_runs_consistent is not None:
        print(f"all_runs_consistent={agg.all_runs_consistent}")
    # Show both pass criteria so the CLI output is clear in either mode.
    print(f"PASS(exact)={agg.passed} PASS(snapshot_soft)={agg.snapshot_passed}")


def print_variance_report(result: BenchResult) -> None:
    """Print per-image across-run drift table (live runs only)."""
    print("\n=== Variance Report ===", file=sys.stderr)
    header = f"{'Image':<30} {'Consistent':>11} {'Drift'}"
    print(header, file=sys.stderr)
    print("-" * len(header), file=sys.stderr)
    for report in result.aggregate.consistency_reports:
        flag = "Y" if report.consistent else "N"
        print(
            f"{report.image:<30} {flag:>11} {report.variance_summary}",
            file=sys.stderr,
        )
    print("-" * len(header), file=sys.stderr)
    if result.aggregate.all_runs_consistent is not None:
        print(
            f"{'ALL':<30} "
            f"{'Y' if result.aggregate.all_runs_consistent else 'N':>11}",
            file=sys.stderr,
        )
    print(file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="benchmark_warehouse",
        description="Full warehouse pipeline benchmark (scanner + Gemini + recovery).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Number of live runs per image. Defaults to dataset consistency.runs_per_image.",
    )
    parser.add_argument(
        "--capture-snapshots",
        action="store_true",
        help="Save both audit candidates to snapshots/pipeline_responses.json.",
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Run snapshot replay benchmark instead of live benchmark.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Gemini model name. Defaults to GEMINI_MODEL or {DEFAULT_MODEL!r}.",
    )
    args = parser.parse_args()

    if args.snapshot:
        result = run_snapshot_benchmark()
    else:
        dataset = load_dataset()
        runs = args.runs if args.runs is not None else dataset.consistency.runs_per_image
        result = run_live_benchmark(
            runs_per_image=runs,
            capture_snapshots=args.capture_snapshots,
            model=args.model,
        )
        print_variance_report(result)

    print_report(result)
    if args.snapshot:
        return 0 if result.aggregate.snapshot_passed else 1
    return 0 if result.aggregate.passed else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

"""Benchmark runner for the Gemini spatial pipeline.

Two modes:

- ``run_live_benchmark`` — runs the real scanner + real Gemini ``audit_shoebox_labels``
  for each image, ``runs_per_image`` times. Slow, charged, network-dependent.
  Optionally captures the first run's Gemini response into
  ``snapshots/gemini_responses.json`` for offline replay.
- ``run_snapshot_benchmark`` — replays a previously captured
  ``snapshots/gemini_responses.json``: runs the real scanner against
  ``samples/<source>`` and feeds the stored Gemini labels through the
  reconciliation + metrics. Deterministic except for the scanner (which is
  deterministic).

The CLI is::

    python -m tests.benchmark_spatial.runner                 # live, 5 runs
    python -m tests.benchmark_spatial.runner --runs 3
    python -m tests.benchmark_spatial.runner --capture-snapshots
    python -m tests.benchmark_spatial.runner --model gemini-2.5-flash

Exit 0 if image-level rules pass; 1 otherwise. Latency is reported only, not
asserted, in this first version.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.services.barcode_scanner import BarcodeScanner
from app.services.gemini_box_audit import (
    DEFAULT_MODEL,
    audit_shoebox_labels,
)
from app.services.spatial_reconciliation import match_scanner_to_labels
from tests.benchmark_spatial.metrics import (
    aggregate_image_metrics,
    compute_image_metrics,
)
from tests.benchmark_spatial.models import (
    BenchResult,
    GroundTruthImage,
    ImageMetrics,
    SpatialDataset,
)

DATASET_PATH = Path(__file__).resolve().parent / "dataset.json"
SNAPSHOTS_PATH = Path(__file__).resolve().parent / "snapshots" / "gemini_responses.json"
SAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "samples"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset(path: Path = DATASET_PATH) -> SpatialDataset:
    raw = json.loads(path.read_text())
    return SpatialDataset.model_validate(raw)


def _samples_path(source: str) -> Path:
    return SAMPLES_DIR / source


# ---------------------------------------------------------------------------
# Snapshot capture / replay
# ---------------------------------------------------------------------------


def _snapshot_store_path() -> Path:
    return SNAPSHOTS_PATH


def _load_snapshots() -> dict[str, Any]:
    path = _snapshot_store_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Snapshot file not found: {path}. Run "
            f"`python -m tests.benchmark_spatial.runner --capture-snapshots` "
            f"once with GEMINI_API_KEY set to populate it."
        )
    return json.loads(path.read_text())


def _save_snapshots(store: dict[str, Any]) -> None:
    path = _snapshot_store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, indent=2))


def _spatial_to_snapshot(spatial: Any) -> dict[str, Any]:
    """Serialize a SpatialLabelAuditPixels to the snapshot store format."""
    return spatial.model_dump(mode="json")


def _snapshot_to_labels(entry: dict[str, Any]) -> tuple[list[dict], int, int]:
    """Return (labels, image_width, image_height) from a snapshot entry."""
    return (
        entry["labels"],
        entry["image_width"],
        entry["image_height"],
    )


# ---------------------------------------------------------------------------
# Per-run metric computation
# ---------------------------------------------------------------------------


def _metrics_from_run(
    gt_image: GroundTruthImage,
    scanner_detections: list[dict],
    gemini_labels: list[dict],
    image_width: int,
    image_height: int,
    latency: float | None = None,
) -> ImageMetrics:
    reconciliation = match_scanner_to_labels(
        scanner_detections,
        gemini_labels,
        image_width=image_width,
        image_height=image_height,
    )
    clear_count = sum(1 for lab in gemini_labels if lab.get("status") == "clear")
    return compute_image_metrics(
        gt_image,
        actual_visible_label_count=len(gemini_labels),
        actual_clear_label_count=clear_count,
        actual_scanner_symbol_count=len(scanner_detections),
        actual_unmatched_label_count=len(reconciliation.unmatched_labels),
        actual_unassigned_scanner_detection_count=(
            len(reconciliation.unassigned_scanner_detections)
        ),
        actual_all_labels_matched=reconciliation.all_labels_matched,
        gemini_labels=gemini_labels,
        image_width=image_width,
        image_height=image_height,
        latency=latency,
    )


# ---------------------------------------------------------------------------
# Live benchmark
# ---------------------------------------------------------------------------


def run_live_benchmark(
    *,
    runs_per_image: int = 5,
    capture_snapshots: bool = False,
    model: str | None = None,
) -> BenchResult:
    """Run the real scanner + real Gemini for each image, multiple times.

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

    all_run_metrics: list[list[ImageMetrics]] = []
    latencies_by_image: dict[str, list[float]] = {}

    for gt in dataset.images:
        image_path = _samples_path(gt.source)
        if not image_path.exists():
            print(
                f"WARNING: source image not found, skipping {gt.image}: {image_path}",
                file=sys.stderr,
            )
            continue

        image_bytes = image_path.read_bytes()
        run_metrics: list[ImageMetrics] = []
        image_latencies: list[float] = []

        for run_idx in range(runs_per_image):
            t0 = time.perf_counter()
            detections = scanner.scan_bytes(image_bytes)
            spatial = audit_shoebox_labels(image_path, model=resolved_model)
            elapsed = time.perf_counter() - t0

            metrics = _metrics_from_run(
                gt,
                detections,
                spatial.labels,
                image_width=spatial.image_width,
                image_height=spatial.image_height,
                latency=elapsed,
            )
            run_metrics.append(metrics)
            image_latencies.append(elapsed)

            if capture_snapshots and run_idx == 0 and gt.image not in snapshots["images"]:
                snapshots["images"][gt.image] = _spatial_to_snapshot(spatial)
                captured_any = True

        all_run_metrics.append(run_metrics)
        latencies_by_image[gt.image] = image_latencies

    if capture_snapshots and captured_any:
        _save_snapshots(snapshots)
        print(
            f"Saved snapshots for {len(snapshots['images'])} image(s) to "
            f"{_snapshot_store_path()}",
            file=sys.stderr,
        )

    # Aggregate: use the first run's metrics per image for the aggregate, but
    # collect latency stats across all runs.
    first_run_metrics = [runs[0] for runs in all_run_metrics if runs]
    agg = aggregate_image_metrics(first_run_metrics)
    # Replace latencies with the full per-image set for median/p90 reporting.
    agg.latencies = [
        lat for lats in latencies_by_image.values() for lat in lats
    ]

    return BenchResult(image_metrics=first_run_metrics, aggregate=agg)


# ---------------------------------------------------------------------------
# Snapshot benchmark
# ---------------------------------------------------------------------------


def run_snapshot_benchmark() -> BenchResult:
    """Replay captured Gemini snapshots through the real scanner + metrics."""
    dataset = load_dataset()
    store = _load_snapshots()
    snapshot_images: dict[str, Any] = store.get("images", {})

    scanner = BarcodeScanner()
    image_metrics: list[ImageMetrics] = []

    for gt in dataset.images:
        if gt.image not in snapshot_images:
            print(
                f"WARNING: no snapshot for {gt.image}, skipping",
                file=sys.stderr,
            )
            continue

        image_path = _samples_path(gt.source)
        if not image_path.exists():
            print(
                f"WARNING: source image not found, skipping {gt.image}: {image_path}",
                file=sys.stderr,
            )
            continue

        labels, image_width, image_height = _snapshot_to_labels(
            snapshot_images[gt.image]
        )
        detections = scanner.scan_bytes(image_path.read_bytes())
        metrics = _metrics_from_run(
            gt,
            detections,
            labels,
            image_width=image_width,
            image_height=image_height,
        )
        image_metrics.append(metrics)

    agg = aggregate_image_metrics(image_metrics)
    return BenchResult(image_metrics=image_metrics, aggregate=agg)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_seconds(seconds: float) -> str:
    return f"{seconds:.2f}s"


def _fmt_percent(value: float | None) -> str:
    if value is None:
        return "  n/a"
    return f"{value * 100:.0f}%"


def print_report(result: BenchResult) -> None:
    header = (
        f"{'Image':<28} {'Count OK':>9} {'Spatial recall':>15} "
        f"{'Missing OK':>11} {'Median':>8}"
    )
    print(header)
    print("-" * len(header))

    for m in result.image_metrics:
        count_ok = "Y" if m.label_count_correct else "N"
        if m.unmatched_count_correct is None:
            missing_ok = "n/a"
        elif m.unmatched_count_correct:
            missing_ok = "Y"
        else:
            missing_ok = "N"
        print(
            f"{m.image:<28} {count_ok:>9} {_fmt_percent(m.spatial_label_recall):>15} "
            f"{str(missing_ok):>11} {_fmt_seconds(m.latency or 0.0):>8}"
        )

    print("-" * len(header))
    agg = result.aggregate
    print(
        f"{'TOTAL':<28} "
        f"{agg.label_count_correct_images}/{agg.image_count:>6} "
        f"{_fmt_percent(agg.spatial_label_recall):>15} "
        f"{agg.correct_unmatched_labels}/{agg.expected_unmatched_labels or 0:>7} "
        f"{_fmt_seconds(agg.median_latency):>8}"
    )
    print()
    print(f"PASS={agg.passed}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="benchmark_spatial",
        description="Live Gemini spatial pipeline benchmark.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of live runs per image. Default: 5.",
    )
    parser.add_argument(
        "--capture-snapshots",
        action="store_true",
        help="Save the first run's Gemini response to snapshots/gemini_responses.json.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Gemini model name. Defaults to GEMINI_MODEL or {DEFAULT_MODEL!r}.",
    )
    args = parser.parse_args()

    result = run_live_benchmark(
        runs_per_image=args.runs,
        capture_snapshots=args.capture_snapshots,
        model=args.model,
    )
    print_report(result)
    return 0 if result.aggregate.passed else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

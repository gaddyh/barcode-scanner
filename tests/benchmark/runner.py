"""Benchmark runner for the deterministic barcode scanner.

Loads a ground-truth dataset, runs the scanner against each image, matches
detections to expected boxes by center-in-box with minimum-absolute-padding
expansion and global distance-based assignment, and reports exact-match recall,
unique-value recall, false positives, mismatches, bonuses, and latency
percentiles.

Run with:

    python -m tests.benchmark.runner
    make bench
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.services.barcode_scanner import BarcodeScanner, BoundingBox, DetectedBarcode

DATASET_PATH = Path(__file__).resolve().parent / "dataset.json"
SAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "samples"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpectedBox:
    status: str  # "decoded" | "unreadable"
    value: str | None
    format: str | None
    bounding_box: BoundingBox
    location: dict[str, Any] | None = None
    visible_metadata: dict[str, Any] | None = None
    reason: str | None = None


@dataclass(frozen=True)
class ExpectedImage:
    image: str
    expected_barcode_symbol_count: int
    boxes: list[ExpectedBox]

    @property
    def decoded_boxes(self) -> list[ExpectedBox]:
        return [b for b in self.boxes if b.status == "decoded"]

    @property
    def expected_decoded(self) -> int:
        return len(self.decoded_boxes)

    @property
    def expected_unique_values(self) -> int:
        return len({b.value for b in self.decoded_boxes})


@dataclass
class MatchMetrics:
    exact_matches: int = 0
    mismatches: int = 0
    misses: int = 0
    false_positives: int = 0
    bonuses: int = 0


@dataclass
class ImageResult:
    image: str
    expected_barcode_symbol_count: int
    expected_decoded: int
    expected_unique_values: int
    found: int  # total scanner detections returned (len(detections))
    metrics: MatchMetrics
    unique_values_found: int
    first_latency: float
    warm_latencies: list[float]

    @property
    def median_latency(self) -> float:
        return statistics.median(self.warm_latencies) if self.warm_latencies else 0.0

    @property
    def p95_latency(self) -> float:
        return percentile_nearest_rank(self.warm_latencies, 0.95) if self.warm_latencies else 0.0


@dataclass
class AggregateResult:
    expected_barcode_symbol_count: int = 0
    expected_decoded: int = 0
    found: int = 0
    exact_matches: int = 0
    mismatches: int = 0
    misses: int = 0
    false_positives: int = 0
    bonuses: int = 0
    expected_unique_values: int = 0
    unique_values_found: int = 0
    first_latency: float = 0.0
    warm_latencies: list[float] = field(default_factory=list)

    @property
    def median_latency(self) -> float:
        return statistics.median(self.warm_latencies) if self.warm_latencies else 0.0

    @property
    def p95_latency(self) -> float:
        return percentile_nearest_rank(self.warm_latencies, 0.95) if self.warm_latencies else 0.0

    @property
    def passed(self) -> bool:
        return (
            self.exact_matches == self.expected_decoded
            and self.false_positives == 0
            and self.mismatches == 0
            and self.unique_values_found == self.expected_unique_values
        )


@dataclass
class BenchResult:
    images: list[ImageResult]
    aggregate: AggregateResult


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset(path: Path = DATASET_PATH) -> list[ExpectedImage]:
    raw = json.loads(path.read_text())
    images: list[ExpectedImage] = []
    for entry in raw["images"]:
        boxes = [
            ExpectedBox(
                status=b["status"],
                value=b.get("value"),
                format=b.get("format"),
                bounding_box=BoundingBox(
                    x1=b["bounding_box"]["x1"],
                    y1=b["bounding_box"]["y1"],
                    x2=b["bounding_box"]["x2"],
                    y2=b["bounding_box"]["y2"],
                ),
                location=b.get("location"),
                visible_metadata=b.get("visible_metadata"),
                reason=b.get("reason"),
            )
            for b in entry["boxes"]
        ]
        images.append(
            ExpectedImage(
                image=entry["image"],
                expected_barcode_symbol_count=entry["expected_barcode_symbol_count"],
                boxes=boxes,
            )
        )
    return images


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def box_center(box: BoundingBox) -> tuple[float, float]:
    return ((box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0)


def expand_box(box: BoundingBox) -> BoundingBox:
    """Expand a box by 20% on each side, with a minimum absolute padding of 25px.

    Thin ZXing boxes (e.g. x1 == x2) get real horizontal tolerance instead of
    ~zero from a purely proportional expansion.
    """
    width = max(1, box.x2 - box.x1)
    height = max(1, box.y2 - box.y1)
    padding_x = max(25, round(width * 0.20))
    padding_y = max(25, round(height * 0.20))
    return BoundingBox(
        x1=box.x1 - padding_x,
        y1=box.y1 - padding_y,
        x2=box.x2 + padding_x,
        y2=box.y2 + padding_y,
    )


def point_inside(point: tuple[float, float], box: BoundingBox) -> bool:
    px, py = point
    return box.x1 <= px <= box.x2 and box.y1 <= py <= box.y2


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def percentile_nearest_rank(values: list[float], percentile: float) -> float:
    """Nearest-rank percentile. Deterministic, no library-convention ambiguity."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def match_detections(
    detections: list[DetectedBarcode],
    expected_boxes: list[ExpectedBox],
) -> MatchMetrics:
    """Match scanner detections to expected boxes.

    Global distance-based assignment: generate every valid (expected, detection)
    candidate pair where the detection center falls inside the expanded expected
    box, sort all candidates by center distance, and greedily assign the nearest
    pair whose expected box and detection are both still unmatched.

    Classification:
      - expected decoded + scanner value matches  -> exact match
      - expected decoded + scanner value differs  -> mismatch (miss + false positive)
      - expected unreadable + scanner found value  -> bonus (not FP, not success criterion)
      - unmatched scanner detection                -> false positive
      - unmatched expected decoded box             -> miss
    """
    metrics = MatchMetrics()

    candidates: list[tuple[float, int, int]] = []
    for ei, expected in enumerate(expected_boxes):
        expanded = expand_box(expected.bounding_box)
        ec = box_center(expected.bounding_box)
        for di, detection in enumerate(detections):
            dc = box_center(detection.bounding_box)
            if point_inside(dc, expanded):
                candidates.append((distance(ec, dc), ei, di))

    assigned_expected: set[int] = set()
    assigned_detections: set[int] = set()
    matched_pairs: list[tuple[int, int]] = []

    for _, ei, di in sorted(candidates):
        if ei in assigned_expected or di in assigned_detections:
            continue
        assigned_expected.add(ei)
        assigned_detections.add(di)
        matched_pairs.append((ei, di))

    for ei, di in matched_pairs:
        expected = expected_boxes[ei]
        detection = detections[di]
        if expected.status == "unreadable":
            metrics.bonuses += 1
        elif detection.value == expected.value:
            metrics.exact_matches += 1
        else:
            metrics.mismatches += 1

    unmatched_expected_decoded = sum(
        1
        for ei, expected in enumerate(expected_boxes)
        if ei not in assigned_expected and expected.status == "decoded"
    )
    unmatched_detections = sum(
        1 for di in range(len(detections)) if di not in assigned_detections
    )

    metrics.misses = unmatched_expected_decoded + metrics.mismatches
    metrics.false_positives = unmatched_detections + metrics.mismatches
    return metrics


def unique_values_found(
    detections: list[DetectedBarcode],
    expected_boxes: list[ExpectedBox],
) -> int:
    """Count how many unique expected-decoded values appear in scanner output."""
    expected_values = {b.value for b in expected_boxes if b.status == "decoded"}
    found_values = {d.value for d in detections}
    return len(expected_values & found_values)


# ---------------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------------


def run_bench(runs_per_image: int = 10) -> BenchResult:
    if runs_per_image < 2:
        raise ValueError("runs_per_image must be at least 2")

    dataset = load_dataset()
    image_results: list[ImageResult] = []

    for expected in dataset:
        image_path = SAMPLES_DIR / expected.image
        image_bytes = image_path.read_bytes()

        # First run: fresh scanner construction + first scan (first-scan latency).
        scanner = BarcodeScanner()
        t0 = time.perf_counter()
        detections = scanner.scan_bytes(image_bytes)
        first_latency = time.perf_counter() - t0

        # Warm runs: reuse the scanner.
        warm_latencies: list[float] = []
        for _ in range(runs_per_image - 1):
            t0 = time.perf_counter()
            detections = scanner.scan_bytes(image_bytes)
            warm_latencies.append(time.perf_counter() - t0)

        metrics = match_detections(detections, expected.boxes)
        uvf = unique_values_found(detections, expected.boxes)

        image_results.append(
            ImageResult(
                image=expected.image,
                expected_barcode_symbol_count=expected.expected_barcode_symbol_count,
                expected_decoded=expected.expected_decoded,
                expected_unique_values=expected.expected_unique_values,
                found=len(detections),
                metrics=metrics,
                unique_values_found=uvf,
                first_latency=first_latency,
                warm_latencies=warm_latencies,
            )
        )

    aggregate = AggregateResult(
        expected_barcode_symbol_count=sum(r.expected_barcode_symbol_count for r in image_results),
        expected_decoded=sum(r.expected_decoded for r in image_results),
        found=sum(r.found for r in image_results),
        exact_matches=sum(r.metrics.exact_matches for r in image_results),
        mismatches=sum(r.metrics.mismatches for r in image_results),
        misses=sum(r.metrics.misses for r in image_results),
        false_positives=sum(r.metrics.false_positives for r in image_results),
        bonuses=sum(r.metrics.bonuses for r in image_results),
        expected_unique_values=sum(r.expected_unique_values for r in image_results),
        unique_values_found=sum(r.unique_values_found for r in image_results),
        first_latency=sum(r.first_latency for r in image_results),
        warm_latencies=[lat for r in image_results for lat in r.warm_latencies],
    )

    return BenchResult(images=image_results, aggregate=aggregate)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_seconds(seconds: float) -> str:
    return f"{seconds:.2f}s"


def print_report(result: BenchResult) -> None:
    header = (
        f"{'Image':<28} {'Symbols':>7} {'Decoded':>7} {'Found':>5} "
        f"{'Exact':>7} {'UniqueCases':>11} {'FP':>3} {'Bonus':>5} "
        f"{'Mismatch':>8} {'Median':>7} {'P95':>7} {'First':>7}"
    )
    print(header)
    print("-" * len(header))

    for r in result.images:
        m = r.metrics
        exact = f"{m.exact_matches}/{r.expected_decoded}"
        unique = f"{r.unique_values_found}/{r.expected_unique_values}"
        print(
            f"{r.image:<28} {r.expected_barcode_symbol_count:>7} {r.expected_decoded:>7} "
            f"{r.found:>5} {exact:>7} {unique:>11} {m.false_positives:>3} {m.bonuses:>5} "
            f"{m.mismatches:>8} {_fmt_seconds(r.median_latency):>7} "
            f"{_fmt_seconds(r.p95_latency):>7} {_fmt_seconds(r.first_latency):>7}"
        )

    print("-" * len(header))
    agg = result.aggregate
    exact = f"{agg.exact_matches}/{agg.expected_decoded}"
    unique = f"{agg.unique_values_found}/{agg.expected_unique_values}"
    print(
        f"{'TOTAL':<28} {agg.expected_barcode_symbol_count:>7} {agg.expected_decoded:>7} "
        f"{agg.found:>5} {exact:>7} {unique:>11} {agg.false_positives:>3} {agg.bonuses:>5} "
        f"{agg.mismatches:>8} {_fmt_seconds(agg.median_latency):>7} "
        f"{_fmt_seconds(agg.p95_latency):>7} {_fmt_seconds(agg.first_latency):>7}"
    )
    print()
    print(f"PASS={agg.passed}")


def main() -> int:
    result = run_bench()
    print_report(result)
    return 0 if result.aggregate.passed else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

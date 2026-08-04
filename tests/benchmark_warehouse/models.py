"""Pydantic models for the warehouse benchmark dataset and result records.

The dataset (``dataset.json``) describes per-image ground truth for the full
end-to-end pipeline: scanner counts, Gemini label counts, baseline and final
matched-label counts, recovery uplift, false-positive expectations, and the
expected failure-reason category.

Result records (``RunMetrics`` / ``AggregateMetrics`` / ``ConsistencyReport``)
are plain dataclasses mirroring the deterministic and spatial benchmark styles.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field

from tests.benchmark_spatial.models import COORDINATE_SPACE

# ---------------------------------------------------------------------------
# Ground-truth dataset models
# ---------------------------------------------------------------------------


class ConsistencyConfig(BaseModel):
    """Configuration for the live consistency test."""

    model_config = ConfigDict(extra="forbid")

    runs_per_image: int = Field(ge=1)
    latency_p95_threshold_seconds: float = Field(gt=0)


class GroundTruthImage(BaseModel):
    """One benchmark image and its per-step expected metrics.

    All ``expected_*`` fields are required — this benchmark is the
    production-grade bar, not the per-label-deferred spatial one.
    """

    model_config = ConfigDict(extra="forbid")

    image: str = Field(description="Filename inside samples/.")
    allow_extra_scanner_detections: bool = Field(
        description=(
            "True when one physical label may legitimately carry multiple "
            "decoded barcode symbols. Extra unassigned detections are not "
            "violations in that case."
        ),
    )
    expected_visible_label_count: int = Field(ge=0)
    expected_clear_label_count: int = Field(ge=0)
    expected_scanner_detection_count: int = Field(ge=0)
    expected_unique_value_count: int = Field(ge=0)
    expected_baseline_matched_labels: int = Field(ge=0)
    expected_final_matched_labels: int = Field(ge=0)
    expected_recovered_label_count: int = Field(ge=0)
    expected_unassigned_scanner_detections: int = Field(ge=0)
    expected_extra_labels: int = Field(ge=0)
    expected_failure_reason: str = Field(
        description=(
            "Expected failure-reason category: scan_error, audit_error, "
            "gemini_label_count_mismatch, scanner_miss, recovery_partial, "
            "recovery_failed, false_positive, or all_matched."
        ),
    )
    expected_all_labels_matched: bool


class WarehouseDataset(BaseModel):
    """Top-level warehouse benchmark dataset."""

    model_config = ConfigDict(extra="forbid")

    version: int = Field(ge=1)
    coordinate_space: str = Field(default=COORDINATE_SPACE)
    consistency: ConsistencyConfig
    images: list[GroundTruthImage]


# ---------------------------------------------------------------------------
# Result records (dataclasses)
# ---------------------------------------------------------------------------


@dataclass
class RunMetrics:
    """Metrics for one image, for one pipeline run."""

    image: str
    # Status
    scan_status: str | None
    audit_status: str | None
    ok: bool
    # Step 1: scanner baseline
    scanner_detection_count: int
    unique_value_count: int
    baseline_matched_labels: int | None
    # Step 2: Gemini labels
    visible_label_count: int | None
    clear_label_count: int | None
    # Step 3: recovery uplift
    recovered_label_count: int
    final_matched_labels: int | None
    all_labels_matched: bool | None
    # Step 4: false positives (raw)
    unassigned_scanner_detections: int
    extra_gemini_labels: int
    # Step 5: latency
    latency_seconds: float | None
    # Step 6: failure reason
    failure_reason: str
    # Whether the recovery path executed (regardless of whether it found anything).
    recovery_fired: bool = False
    # Errors
    scan_error: dict | None = None
    audit_error: dict | None = None
    recovery_error: dict | None = None


@dataclass
class ImageConsistency:
    """Across-run consistency report for one image."""

    image: str
    visible_label_counts: list[int | None]
    final_matched_counts: list[int | None]
    recovered_counts: list[int]
    unassigned_counts: list[int]
    consistent: bool

    @property
    def variance_summary(self) -> str:
        """One-line summary of observed values for diagnostic printing."""
        return (
            f"visible={self.visible_label_counts} "
            f"final={self.final_matched_counts} "
            f"recovered={self.recovered_counts} "
            f"unassigned={self.unassigned_counts}"
        )


@dataclass
class AggregateMetrics:
    """Aggregate metrics across all images for one run-set (first runs)."""

    image_count: int = 0

    # Step 1: scanner baseline
    expected_scanner_detections: int = 0
    actual_scanner_detections: int = 0
    expected_baseline_matched: int = 0
    actual_baseline_matched: int = 0

    # Step 2: Gemini labels
    expected_visible_labels: int = 0
    actual_visible_labels: int = 0
    expected_clear_labels: int = 0
    actual_clear_labels: int = 0

    # Step 3: recovery uplift
    expected_recovered: int = 0
    actual_recovered: int = 0
    expected_final_matched: int = 0
    actual_final_matched: int = 0

    # Step 4: false positives (raw + derived)
    unassigned_scanner_detection_count: int = 0
    extra_gemini_label_count: int = 0
    false_positive_violation_count: int = 0

    # Step 6: failure reasons
    failure_reason_mismatch_count: int = 0

    # Latency
    latencies: list[float] = field(default_factory=list)

    # Consistency (live runs only)
    consistency_reports: list[ImageConsistency] = field(default_factory=list)
    all_runs_consistent: bool | None = None

    # First-run exact match
    all_first_runs_match_expected: bool = False

    @property
    def label_count_accuracy(self) -> float:
        if self.expected_visible_labels == 0:
            return 1.0
        correct = sum(
            1 for r in self._first_run_metrics
            if r.visible_label_count is not None
            and r.visible_label_count == self._expected_visible_by_image.get(r.image)
        )
        return correct / max(1, self.image_count)

    @property
    def baseline_recall(self) -> float:
        if self.expected_baseline_matched == 0:
            return 1.0
        return self.actual_baseline_matched / self.expected_baseline_matched

    @property
    def final_recall(self) -> float:
        if self.expected_final_matched == 0:
            return 1.0
        return self.actual_final_matched / self.expected_final_matched

    @property
    def recovery_uplift(self) -> int:
        return self.actual_final_matched - self.actual_baseline_matched

    @property
    def expected_recovery_uplift(self) -> int:
        return self.expected_final_matched - self.expected_baseline_matched

    @property
    def median_latency(self) -> float:
        if not self.latencies:
            return 0.0
        return sorted(self.latencies)[len(self.latencies) // 2]

    @property
    def latency_p95(self) -> float:
        if not self.latencies:
            return 0.0
        ordered = sorted(self.latencies)
        import math
        index = max(0, math.ceil(0.95 * len(ordered)) - 1)
        return ordered[index]

    @property
    def passed(self) -> bool:
        """True iff every image's first run meets every expected field exactly."""
        return self.all_first_runs_match_expected

    @property
    def snapshot_passed(self) -> bool:
        """Soft-threshold pass for snapshot replay.

        Snapshot-vs-live variance is expected because the frozen Gemini labels
        lead to different recovery crop regions. Use soft aggregate thresholds
        matching ``test_snapshot.py``.
        """
        return (
            self.label_count_accuracy >= 0.85
            and self.baseline_recall >= 0.95
            and self.final_recall >= 0.90
            and abs(self.recovery_uplift - self.expected_recovery_uplift) <= 2
            and self.false_positive_violation_count <= 1
            and self.failure_reason_mismatch_count <= 2
        )

    # Internal: set by aggregate_runs() for label_count_accuracy.
    _first_run_metrics: list[RunMetrics] = field(default_factory=list, repr=False)
    _expected_visible_by_image: dict[str, int] = field(default_factory=dict, repr=False)


@dataclass
class BenchResult:
    """Full benchmark result: per-image run metrics + aggregate."""

    # Per-image: list of runs (first run is the primary one).
    run_metrics_by_image: dict[str, list[RunMetrics]]
    aggregate: AggregateMetrics

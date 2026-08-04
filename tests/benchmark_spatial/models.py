"""Pydantic models for the spatial Gemini benchmark dataset and result records.

The dataset (``dataset.json``) describes image-level ground truth plus optional
per-label ground truth. In this first version only image-level expectations are
populated; ``labels`` is empty until manually reviewed annotations are frozen.

Result records (``ImageMetrics`` / ``AggregateMetrics``) are plain dataclasses
mirroring the deterministic benchmark runner style.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.services.spatial_geometry import PixelBoundingBox

# Coordinate space used by every record in the dataset. Stored on the dataset
# and on each annotation file so the ``freeze`` step can refuse mismatched
# coordinate systems.
COORDINATE_SPACE = "original_exif_normalized_pixels"


# ---------------------------------------------------------------------------
# Ground-truth dataset models
# ---------------------------------------------------------------------------


class GroundTruthLabel(BaseModel):
    """One physical product label with manually reviewed ground-truth boxes.

    All bounding-box fields default to ``None`` so image-level-only dataset
    entries (``labels: []``) remain valid. Per-label spatial assertions become
    active only once these boxes are frozen from reviewed annotations.
    """

    model_config = ConfigDict(extra="forbid")

    label_id: str = Field(description="Stable id, e.g. 'r4c1' for row 4 column 1.")
    row: int = Field(ge=1)
    column: int = Field(ge=1)
    label_bbox: PixelBoundingBox | None = None
    barcode_bbox: PixelBoundingBox | None = None
    expected_scanner_status: str = Field(
        default="decoded",
        description="Either 'decoded' or 'unreadable'.",
    )
    expected_unmatched: bool = False
    visible_metadata: dict[str, Any] | None = None


class GroundTruthImage(BaseModel):
    """One benchmark image and its image-level expectations.

    ``source`` maps the benchmark name to an existing ``samples/`` filename —
    images are never duplicated into the benchmark tree.

    Any ``expected_*`` field set to ``None`` means "do not assert". The
    aggregation code must exclude ``None`` expectations from its denominators.
    """

    model_config = ConfigDict(extra="forbid")

    image: str = Field(description="Benchmark name, e.g. 'clean_12_labels.jpeg'.")
    source: str = Field(description="Filename inside samples/ to load at runtime.")
    expected_visible_label_count: int = Field(ge=0)
    expected_clear_label_count: int | None = Field(
        default=None,
        description="None means do not assert the clear count.",
    )
    expected_scanner_symbol_count: int | None = Field(
        default=None,
        description="None means do not assert the decoded symbol count.",
    )
    expected_unmatched_label_count: int | None = Field(
        default=None,
        description=(
            "None means do not assert reconciliation. Used for fuzzy images "
            "where zero scanner detections make every label unmatched by "
            "construction, which adds no information beyond count checks."
        ),
    )
    expected_unassigned_scanner_detection_count: int | None = Field(
        default=None,
        description="None means do not assert the unassigned detection count.",
    )
    expected_all_labels_matched: bool | None = Field(
        default=None,
        description="None means do not assert the all_labels_matched flag.",
    )
    allow_extra_scanner_detections: bool = Field(
        default=False,
        description=(
            "True when one physical label may legitimately carry multiple "
            "decoded barcode symbols (e.g. the single Marny label with two "
            "barcodes). Extra unassigned detections are not failures in that "
            "case."
        ),
    )
    expected_unmatched_label_ids: list[str] = Field(
        default_factory=list,
        description="Specific label ids expected to be unmatched. Active only "
        "when per-label ground truth is frozen.",
    )
    labels: list[GroundTruthLabel] = Field(
        default_factory=list,
        description="Per-label ground truth. Empty until annotations are "
        "manually reviewed and frozen.",
    )


class SpatialDataset(BaseModel):
    """Top-level spatial benchmark dataset."""

    model_config = ConfigDict(extra="forbid")

    version: int = Field(ge=1)
    coordinate_space: str = Field(default=COORDINATE_SPACE)
    images: list[GroundTruthImage]


# ---------------------------------------------------------------------------
# Result records (dataclasses — mirrors the deterministic benchmark style)
# ---------------------------------------------------------------------------


@dataclass
class LabelMatchRecord:
    """Result of matching one Gemini label to one ground-truth label.

    ``gemini_label_index`` is ``None`` when no Gemini label matched this
    ground-truth label. ``skipped`` is True when no per-label ground truth was
    available for this image (``labels: []``).
    """

    gt_label_id: str | None
    gemini_label_index: int | None
    center_distance: float | None
    center_inside_gt: bool | None
    skipped: bool = False


@dataclass
class BarcodeLocalizationRecord:
    """Barcode-region localization quality for one matched label.

    ``skipped`` is True when either the ground-truth or Gemini barcode box is
    absent, or when no per-label ground truth exists.
    """

    gt_label_id: str | None
    gemini_label_index: int | None
    center_distance: float | None
    iou: float | None
    center_inside_gt: bool | None
    skipped: bool = False


@dataclass
class ImageMetrics:
    """Metrics for one image, for one run."""

    image: str
    # Image-level counts (always measurable).
    expected_visible_label_count: int
    actual_visible_label_count: int
    label_count_correct: bool
    extra_labels: int

    expected_clear_label_count: int | None
    actual_clear_label_count: int | None
    clear_count_correct: bool | None

    expected_scanner_symbol_count: int | None
    actual_scanner_symbol_count: int | None
    scanner_symbol_count_correct: bool | None

    expected_unmatched_label_count: int | None
    actual_unmatched_label_count: int | None
    unmatched_count_correct: bool | None

    expected_unassigned_scanner_detection_count: int | None
    actual_unassigned_scanner_detection_count: int | None
    unassigned_detection_count_correct: bool | None

    expected_all_labels_matched: bool | None
    actual_all_labels_matched: bool | None
    all_labels_matched_correct: bool | None

    # Per-label spatial metrics. ``None`` while ground-truth boxes are absent.
    label_match_records: list[LabelMatchRecord] = field(default_factory=list)
    barcode_localization_records: list[BarcodeLocalizationRecord] = field(
        default_factory=list
    )
    spatial_label_recall: float | None = None

    # Latency (seconds). Populated by the live runner; None for snapshot runs.
    latency: float | None = None

    @property
    def passed(self) -> bool:
        """True only when every active image-level expectation is met.

        Per-label spatial metrics (``spatial_label_recall``,
        ``barcode_localization_records``) are intentionally excluded — they are
        not asserted until per-label ground truth is frozen.
        """
        checks: list[bool] = [self.label_count_correct, self.extra_labels == 0]
        if self.clear_count_correct is not None:
            checks.append(self.clear_count_correct)
        if self.scanner_symbol_count_correct is not None:
            checks.append(self.scanner_symbol_count_correct)
        if self.unmatched_count_correct is not None:
            checks.append(self.unmatched_count_correct)
        if self.unassigned_detection_count_correct is not None:
            checks.append(self.unassigned_detection_count_correct)
        if self.all_labels_matched_correct is not None:
            checks.append(self.all_labels_matched_correct)
        return all(checks)


@dataclass
class AggregateMetrics:
    """Aggregate metrics across all images for one run-set."""

    image_count: int = 0
    expected_visible_labels: int = 0
    actual_visible_labels: int = 0
    extra_labels: int = 0
    label_count_correct_images: int = 0
    label_count_accuracy: float = 0.0

    expected_unmatched_labels: int = 0
    correct_unmatched_labels: int = 0
    wrong_unmatched_labels: int = 0
    unmatched_label_accuracy: float = 0.0

    # Per-label spatial aggregates — None while ground truth is absent.
    spatial_label_recall: float | None = None

    # Latency (seconds). Populated by the live runner.
    latencies: list[float] = field(default_factory=list)

    @property
    def median_latency(self) -> float:
        import statistics

        return statistics.median(self.latencies) if self.latencies else 0.0

    @property
    def p90_latency(self) -> float:
        if not self.latencies:
            return 0.0
        ordered = sorted(self.latencies)
        import math

        index = max(0, math.ceil(0.90 * len(ordered)) - 1)
        return ordered[index]

    @property
    def passed(self) -> bool:
        """True only when active image-level expectations are met across images.

        Per-label spatial metrics are excluded (see ``ImageMetrics.passed``).
        """
        return (
            self.label_count_accuracy == 1.0
            and self.extra_labels == 0
            and self.unmatched_label_accuracy == 1.0
        )


@dataclass
class BenchResult:
    """Full benchmark result: per-image metrics + aggregate."""

    image_metrics: list[ImageMetrics]
    aggregate: AggregateMetrics

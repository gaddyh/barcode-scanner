"""Deterministic unit tests for tests.benchmark_spatial.metrics.

No Gemini, no scanner, no network. All inputs are synthetic boxes. These are
the active guards in the first PR — they verify the metric calculations
themselves, not Gemini's behavior.
"""

from __future__ import annotations

from app.services.spatial_geometry import PixelBoundingBox
from tests.benchmark_spatial.metrics import (
    aggregate_image_metrics,
    barcode_localization,
    center_inside,
    compute_image_metrics,
    extra_label_count,
    iou,
    label_count_correct,
    match_gemini_labels_to_ground_truth,
    reconciliation_correct,
    spatial_label_recall,
)
from tests.benchmark_spatial.models import GroundTruthImage, GroundTruthLabel

IMAGE_W = 1000
IMAGE_H = 1000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gt_label(
    label_id: str,
    row: int,
    column: int,
    *,
    label_bbox: tuple[int, int, int, int] | None = None,
    barcode_bbox: tuple[int, int, int, int] | None = None,
    expected_unmatched: bool = False,
) -> GroundTruthLabel:
    return GroundTruthLabel(
        label_id=label_id,
        row=row,
        column=column,
        label_bbox=(
            PixelBoundingBox(
                x1=label_bbox[0], y1=label_bbox[1],
                x2=label_bbox[2], y2=label_bbox[3],
            )
            if label_bbox is not None
            else None
        ),
        barcode_bbox=(
            PixelBoundingBox(
                x1=barcode_bbox[0], y1=barcode_bbox[1],
                x2=barcode_bbox[2], y2=barcode_bbox[3],
            )
            if barcode_bbox is not None
            else None
        ),
        expected_unmatched=expected_unmatched,
    )


def _gemini_label(
    label_index: int,
    label_bbox: tuple[int, int, int, int],
    *,
    barcode_bbox: tuple[int, int, int, int] | None = None,
    status: str = "clear",
) -> dict:
    return {
        "label_index": label_index,
        "label_bbox": {
            "x1": label_bbox[0],
            "y1": label_bbox[1],
            "x2": label_bbox[2],
            "y2": label_bbox[3],
        },
        "barcode_bbox": (
            None
            if barcode_bbox is None
            else {
                "x1": barcode_bbox[0],
                "y1": barcode_bbox[1],
                "x2": barcode_bbox[2],
                "y2": barcode_bbox[3],
            }
        ),
        "status": status,
    }


# ---------------------------------------------------------------------------
# 1. Label-count accuracy
# ---------------------------------------------------------------------------


def test_label_count_correct_exact_match() -> None:
    assert label_count_correct(12, 12) is True


def test_label_count_correct_undercount() -> None:
    assert label_count_correct(11, 12) is False


def test_label_count_correct_overcount() -> None:
    assert label_count_correct(13, 12) is False


def test_extra_label_count_zero_when_under() -> None:
    assert extra_label_count(11, 12) == 0


def test_extra_label_count_zero_when_exact() -> None:
    assert extra_label_count(12, 12) == 0


def test_extra_label_count_positive_when_over() -> None:
    assert extra_label_count(15, 12) == 3


# ---------------------------------------------------------------------------
# 2. Spatial label recall — center containment matching
# ---------------------------------------------------------------------------


def test_match_all_labels_when_centers_inside_padded_gt() -> None:
    gt = [
        _gt_label("r1c1", 1, 1, label_bbox=(100, 100, 200, 200)),
        _gt_label("r1c2", 1, 2, label_bbox=(400, 100, 500, 200)),
    ]
    gemini = [
        _gemini_label(1, (110, 110, 190, 190)),
        _gemini_label(2, (410, 110, 490, 190)),
    ]
    records = match_gemini_labels_to_ground_truth(
        gemini, gt, image_width=IMAGE_W, image_height=IMAGE_H
    )
    assert len(records) == 2
    assert all(not r.skipped for r in records)
    assert all(r.gemini_label_index is not None for r in records)
    assert spatial_label_recall(records) == 1.0


def test_match_misses_when_gemini_center_outside_padded_gt() -> None:
    gt = [_gt_label("r1c1", 1, 1, label_bbox=(100, 100, 200, 200))]
    # Gemini box centered far away from the GT label.
    gemini = [_gemini_label(1, (800, 800, 900, 900))]
    records = match_gemini_labels_to_ground_truth(
        gemini, gt, image_width=IMAGE_W, image_height=IMAGE_H
    )
    assert records[0].gemini_label_index is None
    assert records[0].center_inside_gt is False
    assert spatial_label_recall(records) == 0.0


def test_match_tolerates_small_offset_via_padding() -> None:
    gt = [_gt_label("r1c1", 1, 1, label_bbox=(100, 100, 200, 200))]
    # Gemini center at (215, 150) — 15px outside the GT box but inside the
    # 20%-padded (min 25px) expansion.
    gemini = [_gemini_label(1, (200, 140, 230, 160))]
    records = match_gemini_labels_to_ground_truth(
        gemini, gt, image_width=IMAGE_W, image_height=IMAGE_H
    )
    assert records[0].gemini_label_index == 0
    assert records[0].center_inside_gt is True


def test_match_assigns_nearest_first_globally() -> None:
    gt = [
        _gt_label("r1c1", 1, 1, label_bbox=(100, 100, 200, 200)),
        _gt_label("r1c2", 1, 2, label_bbox=(400, 100, 500, 200)),
    ]
    # Both Gemini boxes could match both GT labels under heavy padding, but the
    # nearest pair should win each label.
    gemini = [
        _gemini_label(1, (110, 110, 190, 190)),  # near r1c1
        _gemini_label(2, (410, 110, 490, 190)),  # near r1c2
    ]
    records = match_gemini_labels_to_ground_truth(
        gemini, gt, image_width=IMAGE_W, image_height=IMAGE_H
    )
    by_id = {r.gt_label_id: r for r in records}
    assert by_id["r1c1"].gemini_label_index == 0
    assert by_id["r1c2"].gemini_label_index == 1


def test_match_returns_skipped_when_gt_labels_empty() -> None:
    records = match_gemini_labels_to_ground_truth(
        [_gemini_label(1, (100, 100, 200, 200))],
        [],
        image_width=IMAGE_W,
        image_height=IMAGE_H,
    )
    assert len(records) == 1
    assert records[0].skipped is True
    assert spatial_label_recall(records) is None


# ---------------------------------------------------------------------------
# 3. Barcode localization — IoU and center-inside
# ---------------------------------------------------------------------------


def _box(x1: int, y1: int, x2: int, y2: int) -> PixelBoundingBox:
    return PixelBoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def test_iou_identical_boxes() -> None:
    a = _box(100, 100, 200, 200)
    b = _box(100, 100, 200, 200)
    assert iou(a, b) == 1.0


def test_iou_disjoint_boxes() -> None:
    a = _box(100, 100, 200, 200)
    b = _box(300, 300, 400, 400)
    assert iou(a, b) == 0.0


def test_iou_partial_overlap() -> None:
    a = _box(100, 100, 200, 200)
    b = _box(150, 150, 250, 250)
    # inter = 50*50 = 2500; area_a = area_b = 10000; union = 17500
    assert abs(iou(a, b) - 2500 / 17500) < 1e-9


def test_iou_thin_rectangles() -> None:
    a = _box(100, 100, 110, 500)
    b = _box(105, 100, 115, 500)
    # inter = 5*400 = 2000; area_a = area_b = 10*400 = 4000; union = 6000
    assert abs(iou(a, b) - 2000 / 6000) < 1e-9


def test_center_inside_within_padding() -> None:
    target = _box(100, 100, 200, 200)
    assert center_inside((225, 150), target, pad_x=25, pad_y=25) is True


def test_center_inside_outside_padding() -> None:
    target = _box(100, 100, 200, 200)
    assert center_inside((300, 150), target, pad_x=25, pad_y=25) is False


def test_barcode_localization_skipped_when_gt_labels_empty() -> None:
    records = barcode_localization(
        [_gemini_label(1, (100, 100, 200, 200), barcode_bbox=(110, 110, 120, 190))],
        [],
        [],
        image_width=IMAGE_W,
        image_height=IMAGE_H,
    )
    assert len(records) == 1
    assert records[0].skipped is True


def test_barcode_localization_skipped_when_gt_barcode_none() -> None:
    gt = [_gt_label("r1c1", 1, 1, label_bbox=(100, 100, 200, 200))]
    gemini = [_gemini_label(1, (110, 110, 190, 190), barcode_bbox=(120, 120, 130, 180))]
    matches = match_gemini_labels_to_ground_truth(
        gemini, gt, image_width=IMAGE_W, image_height=IMAGE_H
    )
    records = barcode_localization(
        gemini, gt, matches, image_width=IMAGE_W, image_height=IMAGE_H
    )
    assert records[0].skipped is True
    assert records[0].iou is None


def test_barcode_localization_computes_iou_and_center_inside() -> None:
    gt = [
        _gt_label(
            "r1c1", 1, 1,
            label_bbox=(100, 100, 300, 300),
            barcode_bbox=(110, 110, 120, 290),
        )
    ]
    gemini = [
        _gemini_label(
            1, (100, 100, 300, 300), barcode_bbox=(112, 112, 122, 288)
        )
    ]
    matches = match_gemini_labels_to_ground_truth(
        gemini, gt, image_width=IMAGE_W, image_height=IMAGE_H
    )
    records = barcode_localization(
        gemini, gt, matches, image_width=IMAGE_W, image_height=IMAGE_H
    )
    assert records[0].skipped is False
    assert records[0].iou is not None and records[0].iou > 0.0
    assert records[0].center_inside_gt is True


# ---------------------------------------------------------------------------
# 4. Reconciliation correctness
# ---------------------------------------------------------------------------


def test_reconciliation_correct_match() -> None:
    assert reconciliation_correct(1, 1) is True


def test_reconciliation_correct_mismatch() -> None:
    assert reconciliation_correct(2, 1) is False


def test_reconciliation_correct_none_expectation_returns_none() -> None:
    assert reconciliation_correct(17, None) is None


# ---------------------------------------------------------------------------
# 5. Aggregation — None expectations excluded from denominators
# ---------------------------------------------------------------------------


def _image_metrics(
    image: str,
    *,
    expected_visible: int,
    actual_visible: int,
    expected_unmatched: int | None,
    actual_unmatched: int,
) -> object:
    """Build a minimal ImageMetrics for aggregation tests."""
    from tests.benchmark_spatial.models import ImageMetrics

    return ImageMetrics(
        image=image,
        expected_visible_label_count=expected_visible,
        actual_visible_label_count=actual_visible,
        label_count_correct=actual_visible == expected_visible,
        extra_labels=max(0, actual_visible - expected_visible),
        expected_clear_label_count=None,
        actual_clear_label_count=None,
        clear_count_correct=None,
        expected_scanner_symbol_count=None,
        actual_scanner_symbol_count=0,
        scanner_symbol_count_correct=None,
        expected_unmatched_label_count=expected_unmatched,
        actual_unmatched_label_count=actual_unmatched,
        unmatched_count_correct=(
            actual_unmatched == expected_unmatched
            if expected_unmatched is not None
            else None
        ),
        expected_unassigned_scanner_detection_count=None,
        actual_unassigned_scanner_detection_count=0,
        unassigned_detection_count_correct=None,
        expected_all_labels_matched=None,
        actual_all_labels_matched=False,
        all_labels_matched_correct=None,
    )


def test_aggregate_excludes_none_unmatched_from_denominator() -> None:
    # clean_12: expected unmatched = 1, actual = 1 (correct)
    # fuzzy_17: expected unmatched = None, actual = 17 (must NOT count)
    metrics = [
        _image_metrics(
            "clean_12", expected_visible=12, actual_visible=12,
            expected_unmatched=1, actual_unmatched=1,
        ),
        _image_metrics(
            "fuzzy_17", expected_visible=17, actual_visible=17,
            expected_unmatched=None, actual_unmatched=17,
        ),
    ]
    agg = aggregate_image_metrics(metrics)
    assert agg.image_count == 2
    assert agg.expected_visible_labels == 29
    assert agg.expected_unmatched_labels == 1  # only clean_12 counted
    assert agg.correct_unmatched_labels == 1
    assert agg.wrong_unmatched_labels == 0
    assert agg.unmatched_label_accuracy == 1.0
    assert agg.label_count_accuracy == 1.0
    assert agg.extra_labels == 0
    assert agg.passed is True


def test_aggregate_fails_when_unmatched_wrong() -> None:
    metrics = [
        _image_metrics(
            "clean_12", expected_visible=12, actual_visible=12,
            expected_unmatched=1, actual_unmatched=2,
        ),
    ]
    agg = aggregate_image_metrics(metrics)
    assert agg.correct_unmatched_labels == 0
    assert agg.wrong_unmatched_labels == 1
    assert agg.unmatched_label_accuracy == 0.0
    assert agg.passed is False


def test_aggregate_fails_when_extra_labels() -> None:
    metrics = [
        _image_metrics(
            "clean_12", expected_visible=12, actual_visible=14,
            expected_unmatched=1, actual_unmatched=1,
        ),
    ]
    agg = aggregate_image_metrics(metrics)
    assert agg.extra_labels == 2
    assert agg.label_count_accuracy == 0.0
    assert agg.passed is False


def test_aggregate_spatial_recall_none_when_any_image_lacks_gt() -> None:
    metrics = [
        _image_metrics(
            "clean_12", expected_visible=12, actual_visible=12,
            expected_unmatched=1, actual_unmatched=1,
        ),
    ]
    agg = aggregate_image_metrics(metrics)
    assert agg.spatial_label_recall is None


def test_aggregate_empty_image_list() -> None:
    agg = aggregate_image_metrics([])
    assert agg.image_count == 0
    assert agg.label_count_accuracy == 1.0  # vacuously
    assert agg.unmatched_label_accuracy == 1.0  # vacuously
    assert agg.passed is True


# ---------------------------------------------------------------------------
# compute_image_metrics — integration of all metric groups
# ---------------------------------------------------------------------------


def test_compute_image_metrics_image_level_only_when_labels_empty() -> None:
    gt_image = GroundTruthImage(
        image="clean_12",
        source="multi_12_clean.jpeg",
        expected_visible_label_count=12,
        expected_clear_label_count=12,
        expected_scanner_symbol_count=11,
        expected_unmatched_label_count=1,
        expected_all_labels_matched=False,
        labels=[],
    )
    gemini_labels = [_gemini_label(i, (i * 100, 100, i * 100 + 50, 200)) for i in range(1, 13)]
    m = compute_image_metrics(
        gt_image,
        actual_visible_label_count=12,
        actual_clear_label_count=12,
        actual_scanner_symbol_count=11,
        actual_unmatched_label_count=1,
        actual_unassigned_scanner_detection_count=0,
        actual_all_labels_matched=False,
        gemini_labels=gemini_labels,
        image_width=IMAGE_W,
        image_height=IMAGE_H,
    )
    assert m.label_count_correct is True
    assert m.extra_labels == 0
    assert m.clear_count_correct is True
    assert m.scanner_symbol_count_correct is True
    assert m.unmatched_count_correct is True
    assert m.all_labels_matched_correct is True
    assert m.spatial_label_recall is None  # no per-label GT
    assert m.passed is True


def test_compute_image_metrics_with_per_label_gt() -> None:
    gt_image = GroundTruthImage(
        image="test",
        source="test.jpeg",
        expected_visible_label_count=2,
        labels=[
            _gt_label(
                "r1c1", 1, 1,
                label_bbox=(100, 100, 200, 200),
                barcode_bbox=(110, 110, 120, 190),
            ),
            _gt_label(
                "r1c2", 1, 2,
                label_bbox=(400, 100, 500, 200),
                barcode_bbox=(410, 110, 420, 190),
            ),
        ],
    )
    gemini_labels = [
        _gemini_label(1, (110, 110, 190, 190), barcode_bbox=(112, 112, 122, 188)),
        _gemini_label(2, (410, 110, 490, 190), barcode_bbox=(412, 112, 422, 188)),
    ]
    m = compute_image_metrics(
        gt_image,
        actual_visible_label_count=2,
        actual_clear_label_count=2,
        actual_scanner_symbol_count=2,
        actual_unmatched_label_count=0,
        actual_unassigned_scanner_detection_count=0,
        actual_all_labels_matched=True,
        gemini_labels=gemini_labels,
        image_width=IMAGE_W,
        image_height=IMAGE_H,
    )
    assert m.spatial_label_recall == 1.0
    assert len(m.label_match_records) == 2
    assert all(r.gemini_label_index is not None for r in m.label_match_records)
    assert len(m.barcode_localization_records) == 2
    assert all(not r.skipped for r in m.barcode_localization_records)

"""Pure metric functions for the spatial Gemini benchmark.

This module knows nothing about Gemini, ZXing, or the CLI. It operates on plain
dicts and ``PixelBoundingBox`` values and reuses the generic helpers in
``app.services.spatial_geometry``. It is safe to unit-test deterministically.

Five metric groups are implemented:

1. Label-count accuracy      — ``label_count_correct`` / ``extra_label_count``
2. Spatial label recall      — ``match_gemini_labels_to_ground_truth``
3. Barcode localization      — ``barcode_localization`` (center distance, IoU,
                                center-inside-ground-truth)
4. Reconciliation correctness — ``reconciliation_correct``
5. Aggregation               — ``aggregate_image_metrics``

In this first version only image-level metrics (groups 1, 4, 5) affect
``passed``. Per-label spatial metrics (groups 2, 3) are computed when
ground-truth boxes exist and otherwise returned as skipped/``None``.
"""

from __future__ import annotations

from typing import Any

from app.services.spatial_geometry import (
    PixelBoundingBox,
    bbox_center,
    normalized_center_distance,
    padded_bbox,
    point_inside_bbox,
)
from tests.benchmark_spatial.models import (
    AggregateMetrics,
    BarcodeLocalizationRecord,
    GroundTruthImage,
    GroundTruthLabel,
    ImageMetrics,
    LabelMatchRecord,
)

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _bbox_from_dict(d: dict[str, int]) -> PixelBoundingBox:
    return PixelBoundingBox(
        x1=d["x1"],
        y1=d["y1"],
        x2=d["x2"],
        y2=d["y2"],
    )


def _gt_label_padding(box: PixelBoundingBox) -> tuple[int, int]:
    """Padding proportional to label size, with an absolute minimum of 25px.

    Mirrors the deterministic benchmark's ``expand_box`` tolerance so a Gemini
    label box counts as a hit when its center falls inside the padded
    ground-truth label region.
    """
    width = max(1, box.width)
    height = max(1, box.height)
    return max(25, round(width * 0.20)), max(25, round(height * 0.20))


def _barcode_padding(box: PixelBoundingBox) -> tuple[int, int]:
    """Padding for barcode-region center-inside checks.

    Barcode rectangles are thin; a smaller proportional padding with the same
    20px absolute floor as the reconciliation matcher is used.
    """
    width = max(1, box.width)
    height = max(1, box.height)
    return max(20, round(width * 0.15)), max(20, round(height * 0.15))


def iou(a: PixelBoundingBox, b: PixelBoundingBox) -> float:
    """Intersection-over-union of two pixel bounding boxes."""
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0, a.width) * max(0, a.height)
    area_b = max(0, b.width) * max(0, b.height)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def center_inside(
    center: tuple[float, float],
    target: PixelBoundingBox,
    pad_x: int,
    pad_y: int,
) -> bool:
    """True when ``center`` lies inside ``target`` expanded by the given padding."""
    expanded = padded_bbox(target, pad_x, pad_y)
    return point_inside_bbox(center, expanded)


# ---------------------------------------------------------------------------
# 1. Label-count accuracy
# ---------------------------------------------------------------------------


def label_count_correct(actual_visible: int, expected_visible: int) -> bool:
    """True when Gemini returned exactly the expected number of visible labels."""
    return actual_visible == expected_visible


def extra_label_count(actual_visible: int, expected_visible: int) -> int:
    """Image-level label-count overflow. Never negative.

    This is NOT ``false_label_regions``. Region-level false positives cannot be
    measured without per-label ground truth and are deferred until annotations
    are frozen.
    """
    return max(0, actual_visible - expected_visible)


# ---------------------------------------------------------------------------
# 2. Spatial label recall
# ---------------------------------------------------------------------------


def match_gemini_labels_to_ground_truth(
    gemini_labels: list[dict[str, Any]],
    gt_labels: list[GroundTruthLabel],
    *,
    image_width: int,
    image_height: int,
) -> list[LabelMatchRecord]:
    """Match each ground-truth label to the nearest Gemini label by center.

    A Gemini label is a candidate for a ground-truth label when the Gemini
    label center falls inside the padded ground-truth ``label_bbox``. Candidates
    are sorted globally by normalized center distance and assigned greedily;
    each Gemini label is used at most once.

    Returns one ``LabelMatchRecord`` per ground-truth label. When
    ``gt_labels`` is empty, returns a single skipped record so callers can
    distinguish "no ground truth" from "all matched".
    """
    if not gt_labels:
        return [LabelMatchRecord(
            gt_label_id=None,
            gemini_label_index=None,
            center_distance=None,
            center_inside_gt=None,
            skipped=True,
        )]

    candidates: list[tuple[float, int, int]] = []  # (dist, gt_idx, gemini_idx)
    for gt_idx, gt in enumerate(gt_labels):
        if gt.label_bbox is None:
            continue
        gt_box = gt.label_bbox
        gt_center = bbox_center(gt_box)
        pad_x, pad_y = _gt_label_padding(gt_box)
        expanded = padded_bbox(gt_box, pad_x, pad_y)
        for g_idx, g_label in enumerate(gemini_labels):
            g_box = _bbox_from_dict(g_label["label_bbox"])
            g_center = bbox_center(g_box)
            if point_inside_bbox(g_center, expanded):
                dist = normalized_center_distance(
                    gt_center, g_center, image_width, image_height
                )
                candidates.append((dist, gt_idx, g_idx))

    assigned_gt: set[int] = set()
    assigned_gemini: set[int] = set()
    matches: dict[int, tuple[int, float]] = {}  # gt_idx -> (g_idx, dist)

    for dist, gt_idx, g_idx in sorted(candidates):
        if gt_idx in assigned_gt or g_idx in assigned_gemini:
            continue
        assigned_gt.add(gt_idx)
        assigned_gemini.add(g_idx)
        matches[gt_idx] = (g_idx, dist)

    records: list[LabelMatchRecord] = []
    for gt_idx, gt in enumerate(gt_labels):
        if gt.label_bbox is None:
            records.append(LabelMatchRecord(
                gt_label_id=gt.label_id,
                gemini_label_index=None,
                center_distance=None,
                center_inside_gt=None,
                skipped=True,
            ))
            continue
        if gt_idx in matches:
            g_idx, dist = matches[gt_idx]
            records.append(LabelMatchRecord(
                gt_label_id=gt.label_id,
                gemini_label_index=g_idx,
                center_distance=dist,
                center_inside_gt=True,
            ))
        else:
            records.append(LabelMatchRecord(
                gt_label_id=gt.label_id,
                gemini_label_index=None,
                center_distance=None,
                center_inside_gt=False,
            ))
    return records


def spatial_label_recall(records: list[LabelMatchRecord]) -> float | None:
    """Fraction of ground-truth labels matched to a Gemini label.

    Returns ``None`` when all records are skipped (no per-label ground truth).
    """
    active = [r for r in records if not r.skipped]
    if not active:
        return None
    matched = sum(1 for r in active if r.gemini_label_index is not None)
    return matched / len(active)


# ---------------------------------------------------------------------------
# 3. Barcode-region localization
# ---------------------------------------------------------------------------


def barcode_localization(
    gemini_labels: list[dict[str, Any]],
    gt_labels: list[GroundTruthLabel],
    match_records: list[LabelMatchRecord],
    *,
    image_width: int,
    image_height: int,
) -> list[BarcodeLocalizationRecord]:
    """For each matched label, compare Gemini's barcode box to the GT barcode box.

    Reports center distance, IoU, and center-inside-ground-truth. Skipped when
    either side has no barcode box, or when no per-label ground truth exists.
    """
    if not gt_labels:
        return [BarcodeLocalizationRecord(
            gt_label_id=None,
            gemini_label_index=None,
            center_distance=None,
            iou=None,
            center_inside_gt=None,
            skipped=True,
        )]

    records: list[BarcodeLocalizationRecord] = []

    for match in match_records:
        if match.skipped or match.gemini_label_index is None:
            records.append(BarcodeLocalizationRecord(
                gt_label_id=match.gt_label_id,
                gemini_label_index=match.gemini_label_index,
                center_distance=None,
                iou=None,
                center_inside_gt=None,
                skipped=True,
            ))
            continue

        # Find the ground-truth label by id to recover its barcode box.
        gt = next(
            (g for g in gt_labels if g.label_id == match.gt_label_id),
            None,
        )
        if gt is None or gt.barcode_bbox is None:
            records.append(BarcodeLocalizationRecord(
                gt_label_id=match.gt_label_id,
                gemini_label_index=match.gemini_label_index,
                center_distance=None,
                iou=None,
                center_inside_gt=None,
                skipped=True,
            ))
            continue

        g_label = gemini_labels[match.gemini_label_index]
        g_barcode_dict = g_label.get("barcode_bbox")
        if g_barcode_dict is None:
            records.append(BarcodeLocalizationRecord(
                gt_label_id=match.gt_label_id,
                gemini_label_index=match.gemini_label_index,
                center_distance=None,
                iou=None,
                center_inside_gt=None,
                skipped=True,
            ))
            continue

        g_box = _bbox_from_dict(g_barcode_dict)
        gt_box = gt.barcode_bbox
        g_center = bbox_center(g_box)
        gt_center = bbox_center(gt_box)
        pad_x, pad_y = _barcode_padding(gt_box)

        records.append(BarcodeLocalizationRecord(
            gt_label_id=match.gt_label_id,
            gemini_label_index=match.gemini_label_index,
            center_distance=normalized_center_distance(
                gt_center, g_center, image_width, image_height
            ),
            iou=iou(gt_box, g_box),
            center_inside_gt=center_inside(g_center, gt_box, pad_x, pad_y),
            skipped=False,
        ))
    return records


# ---------------------------------------------------------------------------
# 4. Reconciliation correctness
# ---------------------------------------------------------------------------


def reconciliation_correct(
    actual_unmatched_count: int,
    expected_unmatched_count: int | None,
) -> bool | None:
    """True when the unmatched count matches the expectation.

    Returns ``None`` when ``expected_unmatched_count`` is ``None`` (do not
    assert). Callers must exclude ``None`` from accuracy denominators.
    """
    if expected_unmatched_count is None:
        return None
    return actual_unmatched_count == expected_unmatched_count


# ---------------------------------------------------------------------------
# 5. Aggregation
# ---------------------------------------------------------------------------


def _safe_accuracy(correct: int, total: int) -> float:
    return correct / total if total > 0 else 1.0


def compute_image_metrics(
    gt_image: GroundTruthImage,
    *,
    actual_visible_label_count: int,
    actual_clear_label_count: int | None,
    actual_scanner_symbol_count: int,
    actual_unmatched_label_count: int,
    actual_unassigned_scanner_detection_count: int,
    actual_all_labels_matched: bool,
    gemini_labels: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    latency: float | None = None,
) -> ImageMetrics:
    """Compute the full ``ImageMetrics`` record for one image run.

    Per-label spatial metrics are computed only when ``gt_image.labels`` is
    non-empty; otherwise they are left as defaults (empty / ``None``).
    """
    label_match_records = match_gemini_labels_to_ground_truth(
        gemini_labels, gt_image.labels,
        image_width=image_width, image_height=image_height,
    )
    recall = spatial_label_recall(label_match_records)
    barcode_records = barcode_localization(
        gemini_labels, gt_image.labels, label_match_records,
        image_width=image_width, image_height=image_height,
    )

    return ImageMetrics(
        image=gt_image.image,
        expected_visible_label_count=gt_image.expected_visible_label_count,
        actual_visible_label_count=actual_visible_label_count,
        label_count_correct=label_count_correct(
            actual_visible_label_count, gt_image.expected_visible_label_count
        ),
        extra_labels=extra_label_count(
            actual_visible_label_count, gt_image.expected_visible_label_count
        ),
        expected_clear_label_count=gt_image.expected_clear_label_count,
        actual_clear_label_count=actual_clear_label_count,
        clear_count_correct=(
            actual_clear_label_count == gt_image.expected_clear_label_count
            if gt_image.expected_clear_label_count is not None
            else None
        ),
        expected_scanner_symbol_count=gt_image.expected_scanner_symbol_count,
        actual_scanner_symbol_count=actual_scanner_symbol_count,
        scanner_symbol_count_correct=(
            actual_scanner_symbol_count == gt_image.expected_scanner_symbol_count
            if gt_image.expected_scanner_symbol_count is not None
            else None
        ),
        expected_unmatched_label_count=gt_image.expected_unmatched_label_count,
        actual_unmatched_label_count=actual_unmatched_label_count,
        unmatched_count_correct=reconciliation_correct(
            actual_unmatched_label_count, gt_image.expected_unmatched_label_count
        ),
        expected_unassigned_scanner_detection_count=(
            gt_image.expected_unassigned_scanner_detection_count
        ),
        actual_unassigned_scanner_detection_count=(
            actual_unassigned_scanner_detection_count
        ),
        unassigned_detection_count_correct=(
            actual_unassigned_scanner_detection_count
            == gt_image.expected_unassigned_scanner_detection_count
            if gt_image.expected_unassigned_scanner_detection_count is not None
            else None
        ),
        expected_all_labels_matched=gt_image.expected_all_labels_matched,
        actual_all_labels_matched=actual_all_labels_matched,
        all_labels_matched_correct=(
            actual_all_labels_matched == gt_image.expected_all_labels_matched
            if gt_image.expected_all_labels_matched is not None
            else None
        ),
        label_match_records=label_match_records,
        barcode_localization_records=barcode_records,
        spatial_label_recall=recall,
        latency=latency,
    )


def aggregate_image_metrics(image_metrics: list[ImageMetrics]) -> AggregateMetrics:
    """Aggregate per-image metrics into a single ``AggregateMetrics``.

    ``None`` expectations are excluded from denominators. Per-label spatial
    aggregates (``spatial_label_recall``) are ``None`` while any image lacks
    ground-truth boxes.
    """
    agg = AggregateMetrics(image_count=len(image_metrics))

    agg.expected_visible_labels = sum(
        m.expected_visible_label_count for m in image_metrics
    )
    agg.actual_visible_labels = sum(
        m.actual_visible_label_count for m in image_metrics
    )
    agg.extra_labels = sum(m.extra_labels for m in image_metrics)
    agg.label_count_correct_images = sum(
        1 for m in image_metrics if m.label_count_correct
    )
    agg.label_count_accuracy = _safe_accuracy(
        agg.label_count_correct_images, agg.image_count
    )

    # Unmatched-label accuracy — only over images with a non-None expectation.
    unmatched_expected = [
        m for m in image_metrics if m.expected_unmatched_label_count is not None
    ]
    agg.expected_unmatched_labels = sum(
        m.expected_unmatched_label_count or 0 for m in unmatched_expected
    )
    agg.correct_unmatched_labels = sum(
        1 for m in unmatched_expected if m.unmatched_count_correct
    )
    agg.wrong_unmatched_labels = sum(
        1 for m in unmatched_expected if not m.unmatched_count_correct
    )
    agg.unmatched_label_accuracy = _safe_accuracy(
        agg.correct_unmatched_labels, len(unmatched_expected)
    )

    # Per-label spatial recall — None while any image lacks ground truth.
    recalls = [m.spatial_label_recall for m in image_metrics]
    if all(r is not None for r in recalls) and recalls:
        agg.spatial_label_recall = sum(recalls) / len(recalls)
    else:
        agg.spatial_label_recall = None

    agg.latencies = [m.latency for m in image_metrics if m.latency is not None]

    return agg

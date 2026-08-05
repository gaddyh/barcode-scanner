"""
spatial_reconciliation.py

Match deterministic scanner detections to Gemini product-label observations
using padded center-in-box containment with global nearest-first assignment.

Dependency direction::

    spatial_reconciliation.py ──→ spatial_geometry.py

This module imports only ``spatial_geometry``. It does not import Gemini or
scanner modules; it receives both as plain dicts.

Matching rule (first version)
-----------------------------

For each Gemini label, the *target* box is:

    target = label["barcode_bbox"] if present else label["label_bbox"]

A scanner detection becomes a candidate when its center falls inside the
padded target box:

    padding_x = max(20, round(target_width * 0.15))
    padding_y = max(20, round(target_height * 0.15))

When ``barcode_bbox`` is present but no detection falls inside it, the label
stays unmatched — there is **no fallback** to ``label_bbox``. This prevents an
unrelated barcode elsewhere on the same label from being accepted via the
larger label region.

All candidate pairs are sorted globally by normalized center distance and
assigned greedily: each label and each detection is used at most once. This
avoids order-dependent failures where an earlier label "steals" a detection
that a later label could only match.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.services.spatial_geometry import (
    PixelBoundingBox,
    bbox_center,
    clamp_bbox,
    normalized_center_distance,
    padded_bbox,
    point_inside_bbox,
)

MatchBasis = Literal["barcode_bbox", "label_bbox"]


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class LabelMatch(BaseModel):
    """A successful assignment of one scanner detection to one Gemini label."""

    label_index: int
    scanner_detection_index: int
    barcode_value: str
    match_basis: MatchBasis
    center_distance: float = Field(
        description="Normalized (0..1) center distance between target and detection."
    )


class UnmatchedLabel(BaseModel):
    """A Gemini product label with no assigned scanner detection."""

    label_index: int
    label_bbox: dict
    barcode_bbox: dict | None
    status: str


class SpatialReconciliation(BaseModel):
    """
    Full reconciliation result.

    ``all_labels_matched`` means only "every Gemini product label has at least
    one scanner match." It does **not** mean every barcode was decoded, every
    value is correct, or the order is ready for Priority.
    """

    matches: list[LabelMatch]
    unmatched_labels: list[UnmatchedLabel]
    unassigned_scanner_detections: list[dict] = Field(
        description=(
            "Scanner detections not assigned to any Gemini label. Not "
            "necessarily false positives — one physical label can legitimately "
            "carry multiple barcodes."
        )
    )
    matched_label_count: int
    visible_label_count: int
    all_labels_matched: bool


class RecoveredLabel(BaseModel):
    """A label confirmed as recovered by the final reconciliation."""

    label_index: int
    barcode_value: str
    scanner_detection_index: int = Field(
        description="Index into the merged detection list used for final reconciliation."
    )
    crop_basis: MatchBasis
    crop_box: dict = Field(description="The padded crop box that produced the detection.")


class RecoveryResult(BaseModel):
    """Result of Gemini-guided targeted crop recovery.

    A label is only counted as recovered when the final reconciliation assigns
    a recovered detection to that attempted label.  This prevents false
    recovery counts when a crop accidentally finds a nearby barcode belonging
    to a different label.
    """

    attempted_label_count: int
    attempted_label_indexes: list[int]
    recovered_labels: list[RecoveredLabel]
    recovered_label_count: int = Field(
        description="Distinct labels confirmed by final reconciliation."
    )
    recovered_detection_count: int = Field(
        description=(
            "Total detections that originated from recovery crops.  One label "
            "may produce multiple barcode symbols."
        )
    )
    still_unmatched_labels: list[UnmatchedLabel]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _bbox_from_dict(d: dict) -> PixelBoundingBox:
    return PixelBoundingBox(
        x1=d["x1"],
        y1=d["y1"],
        x2=d["x2"],
        y2=d["y2"],
    )


def _target_padding(target: PixelBoundingBox) -> tuple[int, int]:
    """Padding proportional to target size, with an absolute minimum of 20px."""
    padding_x = max(20, round(target.width * 0.15))
    padding_y = max(20, round(target.height * 0.15))
    return padding_x, padding_y


def build_candidate_pairs(
    scanner_detections: list[dict],
    gemini_labels: list[dict],
    *,
    image_width: int,
    image_height: int,
) -> list[tuple[float, int, int, MatchBasis]]:
    """
    Generate every plausible (distance, label_index, detection_index, basis) pair.

    A pair is plausible when the detection center falls inside the padded
    target box. The target is ``barcode_bbox`` when present, otherwise
    ``label_bbox`` — but when ``barcode_bbox`` is present, only it is used
    (no fallback to ``label_bbox``).
    """
    candidates: list[tuple[float, int, int, MatchBasis]] = []

    for li, label in enumerate(gemini_labels):
        barcode_dict = label.get("barcode_bbox")
        label_dict = label["label_bbox"]

        if barcode_dict is not None:
            target = _bbox_from_dict(barcode_dict)
            basis: MatchBasis = "barcode_bbox"
        else:
            target = _bbox_from_dict(label_dict)
            basis = "label_bbox"

        # Clamp the target to the image frame so padding doesn't produce
        # nonsensical negative coordinates for matching purposes.
        target = clamp_bbox(target, image_width, image_height)
        target_center = bbox_center(target)
        pad_x, pad_y = _target_padding(target)
        expanded = padded_bbox(target, pad_x, pad_y)

        for di, detection in enumerate(scanner_detections):
            det_box = _bbox_from_dict(detection["bounding_box"])
            det_center = bbox_center(det_box)

            if point_inside_bbox(det_center, expanded):
                dist = normalized_center_distance(
                    target_center, det_center, image_width, image_height
                )
                candidates.append((dist, li, di, basis))

    return candidates


def assign_matches(
    candidates: list[tuple[float, int, int, MatchBasis]],
    num_labels: int,
    num_detections: int,
) -> list[tuple[int, int, MatchBasis, float]]:
    """
    Greedy global nearest-first assignment.

    Sort all candidates by distance, then assign each pair whose label and
    detection are both still unassigned. Each label and detection is used at
    most once.
    """
    assigned_labels: set[int] = set()
    assigned_detections: set[int] = set()
    matches: list[tuple[int, int, MatchBasis, float]] = []

    for dist, li, di, basis in sorted(candidates):
        if li in assigned_labels or di in assigned_detections:
            continue
        assigned_labels.add(li)
        assigned_detections.add(di)
        matches.append((li, di, basis, dist))

    return matches


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def match_scanner_to_labels(
    scanner_detections: list[dict],
    gemini_labels: list[dict],
    *,
    image_width: int,
    image_height: int,
) -> SpatialReconciliation:
    """
    Match scanner detections to Gemini product labels.

    Args:
        scanner_detections:
            List of detection dicts, each with ``"bounding_box"`` (``x1, y1,
            x2, y2``) and ``"value"``.
        gemini_labels:
            List of Gemini label dicts, each with ``label_index``,
            ``label_bbox`` (pixel ``x1, y1, x2, y2``), ``barcode_bbox`` (pixel
            or ``None``), and ``status``.
        image_width, image_height:
            Dimensions of the normalized image both branches analyzed.

    Returns:
        A ``SpatialReconciliation`` with matches, unmatched labels, and
        unassigned scanner detections.
    """
    candidates = build_candidate_pairs(
        scanner_detections,
        gemini_labels,
        image_width=image_width,
        image_height=image_height,
    )
    raw_matches = assign_matches(
        candidates,
        num_labels=len(gemini_labels),
        num_detections=len(scanner_detections),
    )

    matched_label_indices: set[int] = set()
    matched_detection_indices: set[int] = set()

    matches: list[LabelMatch] = []
    for li, di, basis, dist in raw_matches:
        # Defensive: enforce one-to-one invariant. This should never trigger
        # because assign_matches uses assigned_labels/assigned_detections sets,
        # but if it does, we have a critical bug to catch early.
        assert li not in matched_label_indices, (
            f"Label {li} assigned twice — one-to-one invariant violated"
        )
        assert di not in matched_detection_indices, (
            f"Detection {di} assigned twice — one-to-one invariant violated"
        )
        matched_label_indices.add(li)
        matched_detection_indices.add(di)
        matches.append(
            LabelMatch(
                label_index=gemini_labels[li]["label_index"],
                scanner_detection_index=di,
                barcode_value=scanner_detections[di]["value"],
                match_basis=basis,
                center_distance=dist,
            )
        )

    unmatched_labels: list[UnmatchedLabel] = []
    for li, label in enumerate(gemini_labels):
        if li in matched_label_indices:
            continue
        unmatched_labels.append(
            UnmatchedLabel(
                label_index=label["label_index"],
                label_bbox=label["label_bbox"],
                barcode_bbox=label.get("barcode_bbox"),
                status=label["status"],
            )
        )

    unassigned: list[dict] = []
    for di, detection in enumerate(scanner_detections):
        if di in matched_detection_indices:
            continue
        unassigned.append(detection)

    visible_label_count = len(gemini_labels)
    matched_label_count = len(matched_label_indices)

    return SpatialReconciliation(
        matches=matches,
        unmatched_labels=unmatched_labels,
        unassigned_scanner_detections=unassigned,
        matched_label_count=matched_label_count,
        visible_label_count=visible_label_count,
        all_labels_matched=matched_label_count == visible_label_count,
    )

"""Tests for spatial_reconciliation: scanner detection ↔ Gemini label matching."""

from __future__ import annotations

from src.ingest.reconciliation import (
    LabelMatch,
    SpatialReconciliation,
    UnmatchedLabel,
    match_scanner_to_labels,
)


def _detection(value: str, x1: int, y1: int, x2: int, y2: int) -> dict:
    return {
        "value": value,
        "format": "Code128",
        "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
    }


def _label(
    index: int,
    *,
    label_box: tuple[int, int, int, int],
    barcode_box: tuple[int, int, int, int] | None = None,
    status: str = "clear",
) -> dict:
    return {
        "label_index": index,
        "label_bbox": {
            "x1": label_box[0], "y1": label_box[1],
            "x2": label_box[2], "y2": label_box[3],
        },
        "barcode_bbox": (
            {
                "x1": barcode_box[0], "y1": barcode_box[1],
                "x2": barcode_box[2], "y2": barcode_box[3],
            }
            if barcode_box is not None
            else None
        ),
        "status": status,
    }


def test_all_labels_matched() -> None:
    detections = [
        _detection("111", 110, 110, 190, 290),
        _detection("222", 510, 110, 590, 290),
    ]
    labels = [
        _label(1, label_box=(50, 50, 250, 350), barcode_box=(100, 100, 200, 300)),
        _label(2, label_box=(450, 50, 650, 350), barcode_box=(500, 100, 600, 300)),
    ]

    recon = match_scanner_to_labels(
        detections, labels, image_width=1000, image_height=1000,
    )

    assert recon.matched_label_count == 2
    assert recon.all_labels_matched is True
    assert len(recon.matches) == 2
    assert len(recon.unmatched_labels) == 0
    assert len(recon.unassigned_scanner_detections) == 0
    assert {m.barcode_value for m in recon.matches} == {"111", "222"}


def test_one_label_unmatched() -> None:
    detections = [_detection("111", 110, 110, 190, 290)]
    labels = [
        _label(1, label_box=(50, 50, 250, 350), barcode_box=(100, 100, 200, 300)),
        _label(2, label_box=(450, 50, 650, 350), barcode_box=(500, 100, 600, 300)),
    ]

    recon = match_scanner_to_labels(
        detections, labels, image_width=1000, image_height=1000,
    )

    assert recon.matched_label_count == 1
    assert recon.all_labels_matched is False
    assert len(recon.unmatched_labels) == 1
    assert recon.unmatched_labels[0].label_index == 2


def test_unassigned_scanner_detection() -> None:
    detections = [
        _detection("111", 110, 110, 190, 290),
        _detection("999", 800, 800, 900, 900),  # far from any label
    ]
    labels = [
        _label(1, label_box=(50, 50, 250, 350), barcode_box=(100, 100, 200, 300)),
    ]

    recon = match_scanner_to_labels(
        detections, labels, image_width=1000, image_height=1000,
    )

    assert recon.matched_label_count == 1
    assert len(recon.unassigned_scanner_detections) == 1
    assert recon.unassigned_scanner_detections[0]["value"] == "999"


def test_no_detections_all_unmatched() -> None:
    labels = [
        _label(1, label_box=(50, 50, 250, 350), barcode_box=(100, 100, 200, 300)),
    ]

    recon = match_scanner_to_labels(
        [], labels, image_width=1000, image_height=1000,
    )

    assert recon.matched_label_count == 0
    assert recon.all_labels_matched is False
    assert len(recon.unmatched_labels) == 1


def test_no_labels_all_unassigned() -> None:
    detections = [_detection("111", 110, 110, 190, 290)]

    recon = match_scanner_to_labels(
        detections, [], image_width=1000, image_height=1000,
    )

    assert recon.matched_label_count == 0
    assert len(recon.unassigned_scanner_detections) == 1


def test_strict_barcode_bbox_no_label_fallback() -> None:
    """When barcode_bbox is present but no detection falls inside it, the label
    stays unmatched — there is no fallback to the larger label_bbox."""
    # Detection is inside the label_bbox but outside the barcode_bbox.
    detections = [_detection("111", 60, 60, 90, 90)]
    labels = [
        _label(
            1,
            label_box=(50, 50, 250, 350),
            barcode_box=(100, 100, 200, 300),
        ),
    ]

    recon = match_scanner_to_labels(
        detections, labels, image_width=1000, image_height=1000,
    )

    assert recon.matched_label_count == 0
    assert len(recon.unmatched_labels) == 1
    # The detection is unassigned because no label matched it.
    assert len(recon.unassigned_scanner_detections) == 1


def test_label_bbox_used_when_no_barcode_bbox() -> None:
    """When barcode_bbox is None, the label_bbox is the target."""
    detections = [_detection("111", 110, 110, 190, 290)]
    labels = [
        _label(1, label_box=(50, 50, 250, 350), barcode_box=None),
    ]

    recon = match_scanner_to_labels(
        detections, labels, image_width=1000, image_height=1000,
    )

    assert recon.matched_label_count == 1
    assert recon.matches[0].match_basis == "label_bbox"


def test_nearest_first_assignment() -> None:
    """Two detections both inside one label's padded box — nearest wins."""
    detections = [
        _detection("near", 110, 110, 190, 290),   # center ~ (150, 200)
        _detection("far", 180, 180, 260, 360),    # center ~ (220, 270)
    ]
    labels = [
        _label(1, label_box=(50, 50, 350, 450), barcode_box=(100, 100, 200, 300)),
    ]

    recon = match_scanner_to_labels(
        detections, labels, image_width=1000, image_height=1000,
    )

    assert recon.matched_label_count == 1
    assert recon.matches[0].barcode_value == "near"
    assert len(recon.unassigned_scanner_detections) == 1
    assert recon.unassigned_scanner_detections[0]["value"] == "far"


def test_one_to_one_invariant() -> None:
    """A detection is never assigned to two labels."""
    detections = [_detection("111", 110, 110, 190, 290)]
    labels = [
        _label(1, label_box=(50, 50, 250, 350), barcode_box=(100, 100, 200, 300)),
        _label(2, label_box=(50, 50, 250, 350), barcode_box=(100, 100, 200, 300)),
    ]

    recon = match_scanner_to_labels(
        detections, labels, image_width=1000, image_height=1000,
    )

    assert recon.matched_label_count == 1
    assigned = {m.label_index for m in recon.matches}
    assert len(assigned) == 1


def test_models_are_pydantic() -> None:
    """The output models serialize cleanly."""
    m = LabelMatch(
        label_index=1,
        scanner_detection_index=0,
        barcode_value="123",
        match_basis="barcode_bbox",
        center_distance=0.1,
    )
    d = m.model_dump()
    assert d["label_index"] == 1
    assert d["match_basis"] == "barcode_bbox"

    u = UnmatchedLabel(
        label_index=2,
        label_bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10},
        barcode_bbox=None,
        status="blurred",
    )
    assert u.model_dump()["status"] == "blurred"

    r = SpatialReconciliation(
        matches=[m],
        unmatched_labels=[u],
        unassigned_scanner_detections=[],
        matched_label_count=1,
        visible_label_count=2,
        all_labels_matched=False,
    )
    dumped = r.model_dump(mode="json")
    assert dumped["matched_label_count"] == 1

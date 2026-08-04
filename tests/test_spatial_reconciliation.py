"""Tests for app.services.spatial_reconciliation — scanner↔label matching.

These are the most important behavioral tests for the spatial pipeline. They
cover the matching rules from the plan:

- thin scanner bbox inside larger Gemini barcode bbox → match, basis barcode_bbox
- slight Gemini offset → still matches via padding
- two labels + two detections in different orders → correct global assignment
- one unmatched Gemini label → unmatched_labels, all_labels_matched=False
- extra scanner detection → unassigned_scanner_detections
- two detections competing for one label → only one assigned
- scanner detection inside label_bbox when barcode_bbox is null → match, basis label_bbox
- detection inside full label_bbox but outside a non-null barcode_bbox → does NOT match
  (strict target selection; no fallback to label_bbox when barcode_bbox is present)
"""

from __future__ import annotations

from app.services.spatial_reconciliation import match_scanner_to_labels

# All tests use a fixed 1000 x 1000 image so normalized distance is intuitive.
IMAGE_WIDTH = 1000
IMAGE_HEIGHT = 1000


def _detection(x1: int, y1: int, x2: int, y2: int, value: str = "V") -> dict:
    return {
        "value": value,
        "format": "Code128",
        "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
    }


def _label(
    label_index: int,
    label_bbox: dict,
    *,
    barcode_bbox: dict | None = None,
    status: str = "clear",
) -> dict:
    return {
        "label_index": label_index,
        "label_bbox": label_bbox,
        "barcode_bbox": barcode_bbox,
        "status": status,
    }


# ---------------------------------------------------------------------------
# 1. Clean match — thin scanner bbox inside larger Gemini barcode bbox
# ---------------------------------------------------------------------------


def test_thin_scanner_bbox_inside_larger_gemini_barcode_bbox() -> None:
    """ZXing returns a thin rectangle around the barcode bars; Gemini's
    barcode_bbox is larger. The detection center should fall inside the
    padded barcode region and match with basis barcode_bbox."""
    labels = [
        _label(
            1,
            label_bbox={"x1": 100, "y1": 100, "x2": 400, "y2": 300},
            barcode_bbox={"x1": 200, "y1": 150, "x2": 380, "y2": 280},
        )
    ]
    detections = [
        # Thin box centered inside the barcode region.
        _detection(240, 200, 360, 210, "7297501154117"),
    ]

    result = match_scanner_to_labels(
        detections, labels, image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT
    )

    assert result.matched_label_count == 1
    assert result.visible_label_count == 1
    assert result.all_labels_matched is True
    assert len(result.matches) == 1
    m = result.matches[0]
    assert m.label_index == 1
    assert m.scanner_detection_index == 0
    assert m.barcode_value == "7297501154117"
    assert m.match_basis == "barcode_bbox"
    assert result.unmatched_labels == []
    assert result.unassigned_scanner_detections == []


# ---------------------------------------------------------------------------
# 2. Slight Gemini offset → still matches via padding
# ---------------------------------------------------------------------------


def test_gemini_bbox_shifted_slightly_still_matches_via_padding() -> None:
    """Gemini localization is approximate. A detection center 15px outside the
    barcode_bbox should still match thanks to 15%-of-target (min 20px) padding."""
    labels = [
        _label(
            1,
            label_bbox={"x1": 100, "y1": 100, "x2": 400, "y2": 300},
            barcode_bbox={"x1": 200, "y1": 150, "x2": 300, "y2": 250},
        )
    ]
    # Detection center at (315, 200) — 15px past the right edge of barcode_bbox
    # (x2=300). padding_x = max(20, round(100*0.15)) = 20, so expanded right
    # edge is 320 > 315. Should match.
    detections = [_detection(305, 195, 325, 205, "V1")]

    result = match_scanner_to_labels(
        detections, labels, image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT
    )

    assert result.matched_label_count == 1
    assert result.matches[0].match_basis == "barcode_bbox"


# ---------------------------------------------------------------------------
# 3. Two labels + two detections in different orders → correct assignment
# ---------------------------------------------------------------------------


def test_two_labels_two_detections_different_orders() -> None:
    """Labels are given L1 (top), L2 (bottom); detections are given D2 (bottom
    first), D1 (top). Global nearest-first should assign correctly regardless
    of input order."""
    labels = [
        _label(
            1,
            label_bbox={"x1": 100, "y1": 100, "x2": 300, "y2": 200},
            barcode_bbox={"x1": 150, "y1": 120, "x2": 280, "y2": 190},
        ),
        _label(
            2,
            label_bbox={"x1": 100, "y1": 800, "x2": 300, "y2": 900},
            barcode_bbox={"x1": 150, "y1": 820, "x2": 280, "y2": 890},
        ),
    ]
    # Detections given in reverse spatial order (bottom first, then top).
    detections = [
        _detection(160, 830, 270, 840, "BOTTOM"),
        _detection(160, 130, 270, 140, "TOP"),
    ]

    result = match_scanner_to_labels(
        detections, labels, image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT
    )

    assert result.matched_label_count == 2
    assert result.all_labels_matched is True
    by_label = {m.label_index: m for m in result.matches}
    assert by_label[1].barcode_value == "TOP"
    assert by_label[1].scanner_detection_index == 1
    assert by_label[2].barcode_value == "BOTTOM"
    assert by_label[2].scanner_detection_index == 0


# ---------------------------------------------------------------------------
# 4. One unmatched Gemini label
# ---------------------------------------------------------------------------


def test_one_unmatched_gemini_label() -> None:
    labels = [
        _label(
            1,
            label_bbox={"x1": 100, "y1": 100, "x2": 300, "y2": 200},
            barcode_bbox={"x1": 150, "y1": 120, "x2": 280, "y2": 190},
        ),
        _label(
            2,
            label_bbox={"x1": 700, "y1": 700, "x2": 900, "y2": 850},
            barcode_bbox={"x1": 750, "y1": 720, "x2": 880, "y2": 840},
        ),
    ]
    # Only one detection, matching label 1. Label 2 has no detection.
    detections = [_detection(160, 130, 270, 140, "V1")]

    result = match_scanner_to_labels(
        detections, labels, image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT
    )

    assert result.matched_label_count == 1
    assert result.visible_label_count == 2
    assert result.all_labels_matched is False
    assert len(result.unmatched_labels) == 1
    assert result.unmatched_labels[0].label_index == 2
    assert result.unmatched_labels[0].status == "clear"
    assert result.unmatched_labels[0].barcode_bbox is not None
    assert result.unassigned_scanner_detections == []


# ---------------------------------------------------------------------------
# 5. Extra scanner detection → unassigned_scanner_detections
# ---------------------------------------------------------------------------


def test_extra_scanner_detection_is_unassigned() -> None:
    """One label, two detections. The second detection is not a false positive
    necessarily — one label can carry multiple barcodes. It goes to
    unassigned_scanner_detections, not unmatched_labels."""
    labels = [
        _label(
            1,
            label_bbox={"x1": 100, "y1": 100, "x2": 400, "y2": 300},
            barcode_bbox={"x1": 200, "y1": 150, "x2": 380, "y2": 280},
        )
    ]
    detections = [
        _detection(240, 200, 360, 210, "EAN"),
        # Extra detection far away, not inside any label.
        _detection(800, 800, 850, 810, "EXTRA"),
    ]

    result = match_scanner_to_labels(
        detections, labels, image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT
    )

    assert result.matched_label_count == 1
    assert result.all_labels_matched is True
    assert len(result.unassigned_scanner_detections) == 1
    assert result.unassigned_scanner_detections[0]["value"] == "EXTRA"


# ---------------------------------------------------------------------------
# 6. Two detections competing for one label → only one assigned
# ---------------------------------------------------------------------------


def test_two_detections_competing_for_one_label() -> None:
    """Both detections fall inside the same label's barcode region. Global
    nearest-first assigns the closer one; the other becomes unassigned."""
    labels = [
        _label(
            1,
            label_bbox={"x1": 100, "y1": 100, "x2": 400, "y2": 300},
            barcode_bbox={"x1": 150, "y1": 120, "x2": 380, "y2": 280},
        )
    ]
    # Two detections, both inside the barcode region. The first is closer to
    # the barcode center (265, 200).
    detections = [
        _detection(250, 195, 270, 205, "CLOSER"),  # center ~ (260, 200)
        _detection(350, 260, 370, 270, "FARTHER"),  # center ~ (360, 265)
    ]

    result = match_scanner_to_labels(
        detections, labels, image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT
    )

    assert result.matched_label_count == 1
    assert len(result.matches) == 1
    assert result.matches[0].barcode_value == "CLOSER"
    assert len(result.unassigned_scanner_detections) == 1
    assert result.unassigned_scanner_detections[0]["value"] == "FARTHER"


# ---------------------------------------------------------------------------
# 7. Scanner detection inside label_bbox when barcode_bbox is null
# ---------------------------------------------------------------------------


def test_match_via_label_bbox_when_barcode_bbox_is_null() -> None:
    """Gemini could see the label but not isolate the barcode region. The
    detection should match against the larger label_bbox with basis label_bbox."""
    labels = [
        _label(1, label_bbox={"x1": 100, "y1": 100, "x2": 400, "y2": 300}),
    ]
    detections = [_detection(200, 180, 320, 190, "V1")]

    result = match_scanner_to_labels(
        detections, labels, image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT
    )

    assert result.matched_label_count == 1
    assert result.matches[0].match_basis == "label_bbox"


# ---------------------------------------------------------------------------
# 8. Detection inside label_bbox but outside a non-null barcode_bbox → NO match
# ---------------------------------------------------------------------------


def test_detection_inside_label_but_outside_barcode_bbox_does_not_match() -> None:
    """Strict target selection: when barcode_bbox is present, only it is used.
    A detection elsewhere on the label (inside label_bbox but outside the
    barcode region) must NOT be accepted via the larger label_bbox. The label
    stays unmatched rather than accepting an unrelated barcode."""
    labels = [
        _label(
            1,
            # Label spans x=100..400, y=100..300.
            label_bbox={"x1": 100, "y1": 100, "x2": 400, "y2": 300},
            # Barcode is in the right half of the label.
            barcode_bbox={"x1": 250, "y1": 150, "x2": 380, "y2": 280},
        )
    ]
    # Detection is in the LEFT half of the label — inside label_bbox but well
    # outside barcode_bbox (and outside its padding).
    detections = [_detection(120, 180, 180, 190, "UNRELATED")]

    result = match_scanner_to_labels(
        detections, labels, image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT
    )

    assert result.matched_label_count == 0
    assert result.all_labels_matched is False
    assert len(result.unmatched_labels) == 1
    assert result.unmatched_labels[0].label_index == 1
    # The detection is not assigned to any label.
    assert len(result.unassigned_scanner_detections) == 1
    assert result.unassigned_scanner_detections[0]["value"] == "UNRELATED"


# ---------------------------------------------------------------------------
# 9. Global nearest-first avoids order-dependent stealing
# ---------------------------------------------------------------------------


def test_global_nearest_first_avoids_order_dependent_stealing() -> None:
    """L1 can match D1 or D2; L2 can only match D1. If L1 grabbed D1 first, L2
    would be unmatched. Global nearest-first should give D1 to L2 (closer) and
    D2 to L1, matching both labels."""
    labels = [
        _label(
            1,
            label_bbox={"x1": 100, "y1": 100, "x2": 500, "y2": 300},
            barcode_bbox={"x1": 150, "y1": 120, "x2": 480, "y2": 280},
        ),
        _label(
            2,
            label_bbox={"x1": 180, "y1": 350, "x2": 320, "y2": 450},
            barcode_bbox={"x1": 190, "y1": 360, "x2": 310, "y2": 440},
        ),
    ]
    # D1 is near L2 (center ~250, 400); D2 is near L1 (center ~400, 200).
    # L1's barcode region is large and contains both D1 and D2 centers.
    # L2's barcode region contains only D1.
    # Greedy nearest-first: L2↔D1 (small distance) wins over L1↔D1.
    detections = [
        _detection(230, 390, 270, 410, "D1"),  # center (250, 400) — near L2
        _detection(380, 190, 420, 210, "D2"),  # center (400, 200) — near L1
    ]

    result = match_scanner_to_labels(
        detections, labels, image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT
    )

    assert result.matched_label_count == 2
    assert result.all_labels_matched is True
    by_label = {m.label_index: m for m in result.matches}
    assert by_label[2].barcode_value == "D1"
    assert by_label[1].barcode_value == "D2"


# ---------------------------------------------------------------------------
# 10. Empty inputs
# ---------------------------------------------------------------------------


def test_empty_detections_and_empty_labels() -> None:
    result = match_scanner_to_labels(
        [], [], image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT
    )
    assert result.matched_label_count == 0
    assert result.visible_label_count == 0
    assert result.all_labels_matched is True  # 0 == 0
    assert result.matches == []
    assert result.unmatched_labels == []
    assert result.unassigned_scanner_detections == []


def test_detections_but_no_labels() -> None:
    detections = [_detection(100, 100, 200, 110, "V1")]
    result = match_scanner_to_labels(
        detections, [], image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT
    )
    assert result.matched_label_count == 0
    assert result.visible_label_count == 0
    assert result.all_labels_matched is True
    assert len(result.unassigned_scanner_detections) == 1


def test_labels_but_no_detections() -> None:
    labels = [_label(1, label_bbox={"x1": 100, "y1": 100, "x2": 300, "y2": 200})]
    result = match_scanner_to_labels(
        [], labels, image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT
    )
    assert result.matched_label_count == 0
    assert result.visible_label_count == 1
    assert result.all_labels_matched is False
    assert len(result.unmatched_labels) == 1

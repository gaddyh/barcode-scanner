"""Tests for BarcodeScanner.

The scanner calls zxingcpp.read_barcodes directly, so we monkeypatch that
function to inject deterministic results without needing real barcode images.
"""

from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from app.services.barcode_scanner import (
    BarcodeScanner,
    BoundingBox,
    DetectedBarcode,
    LabelCropRequest,
    Point,
)


def make_png(width: int = 200, height: int = 100) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def _fake_position(x1: int, y1: int, x2: int, y2: int) -> SimpleNamespace:
    """A zxing-style position with four named corners."""
    return SimpleNamespace(
        top_left=SimpleNamespace(x=x1, y=y1),
        top_right=SimpleNamespace(x=x2, y=y1),
        bottom_right=SimpleNamespace(x=x2, y=y2),
        bottom_left=SimpleNamespace(x=x1, y=y2),
    )


def _fake_result(
    value: str,
    fmt: str = "Code 128",
    position: object | None = None,
) -> SimpleNamespace:
    if position is None:
        position = _fake_position(10, 5, 90, 40)
    return SimpleNamespace(
        text=value,
        format=fmt,
        content_type="Text",
        orientation=0,
        position=position,
    )


def _patch_read_barcodes(monkeypatch, results_by_call: list[list[object]]) -> None:
    """Feed different result lists for successive read_barcodes calls.

    The scanner calls read_barcodes once per region/variant. This returns the
    next pre-built list for each call, cycling to [] when exhausted.
    """
    call_index = {"i": 0}

    def fake_read_barcodes(image, **kwargs):
        i = call_index["i"]
        call_index["i"] += 1
        if i < len(results_by_call):
            return results_by_call[i]
        return []

    monkeypatch.setattr("app.services.barcode_scanner.zxingcpp.read_barcodes", fake_read_barcodes)


def test_scan_bytes_returns_detected_barcode(monkeypatch) -> None:
    result = _fake_result("7290001234567", "EAN13")
    _patch_read_barcodes(monkeypatch, [[result]])

    barcodes = BarcodeScanner().scan_bytes(make_png(width=200, height=100))

    assert len(barcodes) == 1
    detected = barcodes[0]
    assert detected.value == "7290001234567"
    assert detected.format == "EAN13"
    assert detected.content_type == "Text"
    assert detected.orientation == 0
    assert len(detected.position) == 4
    assert detected.bounding_box == BoundingBox(x1=10, y1=5, x2=90, y2=40)


def test_scan_bytes_returns_empty_list_when_nothing_found(monkeypatch) -> None:
    _patch_read_barcodes(monkeypatch, [[]])

    barcodes = BarcodeScanner().scan_bytes(make_png())
    assert barcodes == []


def test_deduplicate_merges_same_physical_barcode(monkeypatch) -> None:
    """Same value + format + overlapping position → one detection (largest box kept)."""
    small = _fake_result("12345", position=_fake_position(10, 5, 50, 40))
    large = _fake_result("12345", position=_fake_position(5, 0, 95, 50))

    # First call (full image) finds small, second call (tile) finds large.
    _patch_read_barcodes(monkeypatch, [[small], [large]])

    barcodes = BarcodeScanner(tile_rows=1, tile_columns=1).scan_bytes(make_png())

    assert len(barcodes) == 1
    assert barcodes[0].value == "12345"
    # The larger bounding box should be kept.
    assert barcodes[0].bounding_box == BoundingBox(x1=5, y1=0, x2=95, y2=50)


def test_deduplicate_keeps_same_value_at_different_positions(monkeypatch) -> None:
    """Same value but far-apart positions → two separate detections."""
    top = _fake_result("12345", position=_fake_position(10, 5, 90, 40))
    bottom = _fake_result("12345", position=_fake_position(10, 500, 90, 540))

    _patch_read_barcodes(monkeypatch, [[top, bottom]])

    barcodes = BarcodeScanner(tile_rows=1, tile_columns=1).scan_bytes(
        make_png(width=200, height=600)
    )

    assert len(barcodes) == 2
    values = [b.value for b in barcodes]
    assert values == ["12345", "12345"]


def test_deduplicate_results_sorted_top_to_bottom(monkeypatch) -> None:
    bottom = _fake_result("bbb", position=_fake_position(10, 500, 90, 540))
    top = _fake_result("aaa", position=_fake_position(10, 5, 90, 40))

    _patch_read_barcodes(monkeypatch, [[bottom, top]])

    barcodes = BarcodeScanner(tile_rows=1, tile_columns=1).scan_bytes(
        make_png(width=200, height=600)
    )

    assert [b.value for b in barcodes] == ["aaa", "bbb"]


def test_tile_coordinate_mapping_applies_offset(monkeypatch) -> None:
    """Barcode found in a tile should have coordinates mapped back to full image."""
    # The full-image scan finds nothing.
    # A tile scan finds a barcode at local position (5, 10)-(50, 40).
    tile_result = _fake_result("999", position=_fake_position(5, 10, 50, 40))

    _patch_read_barcodes(monkeypatch, [[], [tile_result]])

    scanner = BarcodeScanner(tile_rows=1, tile_columns=1)
    barcodes = scanner.scan_bytes(make_png(width=200, height=100))

    assert len(barcodes) == 1
    detected = barcodes[0]
    # With 1x1 tiles and 0.2 overlap, the tile covers the full image, so
    # offset is (0, 0) and scale is 1:1.
    assert detected.position[0] == Point(x=5, y=10)
    assert detected.position[2] == Point(x=50, y=40)


def test_rejects_invalid_tile_rows() -> None:
    with pytest.raises(ValueError, match="tile_rows"):
        BarcodeScanner(tile_rows=0)


def test_rejects_invalid_tile_overlap() -> None:
    with pytest.raises(ValueError, match="tile_overlap"):
        BarcodeScanner(tile_overlap=1.0)


def test_scan_bytes_rejects_invalid_image() -> None:
    from PIL import UnidentifiedImageError

    with pytest.raises((UnidentifiedImageError, OSError, ValueError)):
        BarcodeScanner().scan_bytes(b"not an image")


# ---------------------------------------------------------------------------
# scan_label_crops — Gemini-guided recovery
# ---------------------------------------------------------------------------


def _det(value: str, x1: int, y1: int, x2: int, y2: int) -> DetectedBarcode:
    return DetectedBarcode(
        value=value,
        format="Code128",
        content_type="Text",
        orientation=0,
        position=(
            Point(x=x1, y=y1),
            Point(x=x2, y=y1),
            Point(x=x2, y=y2),
            Point(x=x1, y=y2),
        ),
        bounding_box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
    )


def test_scan_label_crops_finds_barcode_in_crop(monkeypatch) -> None:
    """Barcode crop decodes a barcode; coordinates use padded crop origin."""
    # The crop starts at (100, 200) in full-image space.
    # First attempt uses scale=2.0, so ZXing sees a 2x upscaled image.
    # Local coords (10,20)-(50,60) in crop → ZXing sees (20,40)-(100,120).
    result = _fake_result(
        "7290001234567",
        position=_fake_position(20, 40, 100, 120),
    )
    _patch_read_barcodes(monkeypatch, [[result]])

    image = Image.new("RGB", (500, 500), "white")
    req = LabelCropRequest(
        label_index=10,
        barcode_crop=BoundingBox(x1=100, y1=200, x2=200, y2=300),
        exact_barcode_crop=BoundingBox(x1=110, y1=210, x2=190, y2=290),
        label_crop=BoundingBox(x1=50, y1=150, x2=250, y2=350),
    )

    recovered = BarcodeScanner().scan_label_crops(image, [req])

    assert len(recovered) == 1
    rec = recovered[0]
    assert rec.label_index == 10
    assert rec.crop_basis == "barcode_bbox"
    # ZXing (20,40)-(100,120) in 2x → scaled back (10,20)-(50,60) → +offset (100,200)
    assert rec.detection.bounding_box == BoundingBox(
        x1=110, y1=220, x2=150, y2=260
    )


def test_scan_label_crops_skips_crops_with_existing_detection(monkeypatch) -> None:
    """A crop that already contains a primary barcode is skipped."""
    _patch_read_barcodes(monkeypatch, [[_fake_result("should_not_appear")]])

    image = Image.new("RGB", (500, 500), "white")
    req = LabelCropRequest(
        label_index=10,
        barcode_crop=BoundingBox(x1=100, y1=200, x2=200, y2=300),
        exact_barcode_crop=BoundingBox(x1=110, y1=210, x2=190, y2=290),
        label_crop=None,
    )
    existing = [_det("7290001234567", 120, 220, 180, 280)]

    recovered = BarcodeScanner().scan_label_crops(
        image, [req], existing_detections=existing
    )

    assert recovered == []


def test_scan_label_crops_deduplicates(monkeypatch) -> None:
    """Two overlapping crops that find the same barcode return one recovery."""
    result = _fake_result(
        "7290001234567",
        position=_fake_position(10, 10, 50, 40),
    )
    _patch_read_barcodes(monkeypatch, [[result], [result]])

    image = Image.new("RGB", (500, 500), "white")
    req1 = LabelCropRequest(
        label_index=10,
        barcode_crop=BoundingBox(x1=100, y1=100, x2=200, y2=200),
        exact_barcode_crop=BoundingBox(x1=110, y1=110, x2=190, y2=190),
        label_crop=None,
    )
    req2 = LabelCropRequest(
        label_index=11,
        barcode_crop=BoundingBox(x1=100, y1=100, x2=200, y2=200),
        exact_barcode_crop=BoundingBox(x1=110, y1=110, x2=190, y2=190),
        label_crop=None,
    )

    recovered = BarcodeScanner().scan_label_crops(image, [req1, req2])

    # Both crops find the same barcode at the same full-image coordinates.
    # Dedup should retain only one RecoveredDetection.
    assert len(recovered) == 1


def test_scan_label_crops_falls_back_to_label_bbox(monkeypatch) -> None:
    """Barcode crop finds nothing, label crop succeeds → crop_basis='label_bbox'."""
    # Barcode crop: 8 attempts all return empty (no primary → runs all 8).
    # Label crop: first attempt finds the barcode.
    # Label crop uses scale=2.0; local (10,10)-(50,40) → ZXing sees (20,20)-(100,80)
    label_hit = [_fake_result(
        "7290001234567",
        position=_fake_position(20, 20, 100, 80),
    )]
    _patch_read_barcodes(monkeypatch, [[], [], [], [], [], [], [], [], label_hit])

    image = Image.new("RGB", (500, 500), "white")
    req = LabelCropRequest(
        label_index=10,
        barcode_crop=BoundingBox(x1=100, y1=100, x2=150, y2=150),
        exact_barcode_crop=BoundingBox(x1=110, y1=110, x2=140, y2=140),
        label_crop=BoundingBox(x1=50, y1=50, x2=250, y2=250),
    )

    recovered = BarcodeScanner().scan_label_crops(image, [req])

    assert len(recovered) == 1
    assert recovered[0].crop_basis == "label_bbox"
    # ZXing (20,20)-(100,80) in 2x → scaled back (10,10)-(50,40) → +offset (50,50)
    assert recovered[0].detection.bounding_box == BoundingBox(
        x1=60, y1=60, x2=100, y2=90
    )


def test_scan_label_crops_returns_two_symbols_one_label(monkeypatch) -> None:
    """One label crop produces two detections → two RecoveredDetection, same label_index."""
    det1 = _fake_result("7290001111111", position=_fake_position(10, 10, 50, 40))
    det2 = _fake_result("7290002222222", position=_fake_position(60, 10, 90, 40))
    _patch_read_barcodes(monkeypatch, [[det1, det2]])

    image = Image.new("RGB", (500, 500), "white")
    req = LabelCropRequest(
        label_index=5,
        barcode_crop=BoundingBox(x1=100, y1=100, x2=300, y2=200),
        exact_barcode_crop=BoundingBox(x1=110, y1=110, x2=290, y2=190),
        label_crop=None,
    )

    recovered = BarcodeScanner().scan_label_crops(image, [req])

    assert len(recovered) == 2
    assert all(r.label_index == 5 for r in recovered)
    assert all(r.crop_basis == "barcode_bbox" for r in recovered)
    values = {r.detection.value for r in recovered}
    assert values == {"7290001111111", "7290002222222"}


def test_decode_crop_variants_preserves_offset_coordinates(monkeypatch) -> None:
    """Explicit test: coordinates map through padded crop origin."""
    # Scale=2.0: local (5,10)-(95,50) → ZXing sees (10,20)-(190,100)
    result = _fake_result(
        "7290009999999",
        position=_fake_position(10, 20, 190, 100),
    )
    _patch_read_barcodes(monkeypatch, [[result]])

    image = Image.new("RGB", (1000, 1000), "white")
    scanner = BarcodeScanner()

    # Crop origin at (300, 400).
    crop = image.crop((300, 400, 500, 600))
    detections = scanner._decode_crop_variants(
        crop,
        offset_x=300,
        offset_y=400,
    )

    assert len(detections) == 1
    det = detections[0]
    # ZXing (10,20)-(190,100) in 2x → scaled back (5,10)-(95,50) → +offset (300,400)
    assert det.bounding_box == BoundingBox(x1=305, y1=410, x2=395, y2=450)
    assert det.position[0] == Point(x=305, y=410)
    assert det.position[2] == Point(x=395, y=450)


def test_scan_label_crops_high_scale_recovers_small_barcode(monkeypatch) -> None:
    """When standard scales fail, the high-scale exact-crop attempt recovers
    a very small barcode that only decodes at 8x."""
    # Standard attempts (2x, 3x) return nothing.
    # High-scale attempt at 8x finds the barcode.
    # _decode_crop_variants runs 8 attempts, then _decode_crop_high_scale
    # runs 4 attempts (6, 8, 10, 12). The 8x one finds it.
    # ZXing at 8x: local (5,10)-(25,40) → ZXing sees (40,80)-(200,320)
    high_scale_hit = [_fake_result(
        "7297501195257",
        position=_fake_position(40, 80, 200, 320),
    )]
    # 8 empty results for _decode_crop_variants (barcode crop),
    # 8 empty for label crop,
    # 1 empty for high-scale 6x, then 1 hit for high-scale 8x.
    _patch_read_barcodes(
        monkeypatch,
        [[], [], [], [], [], [], [], []] +  # barcode crop (8 variants)
        [[], [], [], [], [], [], [], []] +  # label crop (8 variants)
        [[], high_scale_hit],  # high-scale: 6x fails, 8x succeeds
    )

    image = Image.new("RGB", (500, 500), "white")
    req = LabelCropRequest(
        label_index=10,
        barcode_crop=BoundingBox(x1=100, y1=100, x2=150, y2=200),
        exact_barcode_crop=BoundingBox(x1=110, y1=110, x2=140, y2=190),
        label_crop=BoundingBox(x1=50, y1=50, x2=250, y2=250),
    )

    recovered = BarcodeScanner().scan_label_crops(image, [req])

    assert len(recovered) == 1
    rec = recovered[0]
    assert rec.label_index == 10
    assert rec.crop_basis == "barcode_bbox"
    assert rec.detection.value == "7297501195257"
    # 8x scale: ZXing (40,80)-(200,320) → scaled back (5,10)-(25,40)
    # → +offset (110,110) = (115,120)-(135,150)
    assert rec.detection.bounding_box == BoundingBox(
        x1=115, y1=120, x2=135, y2=150
    )

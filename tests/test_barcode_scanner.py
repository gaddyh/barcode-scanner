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

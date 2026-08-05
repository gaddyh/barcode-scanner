"""Tests for the deterministic BarcodeScanner.

Mocks ``zxingcpp.read_barcodes`` so no real barcode images are required.

The scanner calls ``read_barcodes`` many times (full image, then grid tiles,
shifted tiles, local fallbacks, label-candidate crops). The mocks below return
detections only on the first call (the full-image scan) and empty lists on
every subsequent call, so the scanner's deduplication and tile logic is
exercised without producing duplicate detections from every tile.
"""

from __future__ import annotations

import io
from typing import Any

import pytest
import zxingcpp
from PIL import Image, UnidentifiedImageError

from app.services.barcode_scanner import BarcodeScanner
from tests._zxing_fake import FakeReadResult, make_read_result


def _png_bytes(width: int = 800, height: int = 600) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


def _first_call_mock(results: list[FakeReadResult]) -> Any:
    """Return a mock that yields ``results`` on the first call, then ``[]``."""
    calls = {"n": 0}

    def _mock(_img, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return results
        return []

    return _mock


def test_scan_bytes_returns_detections(monkeypatch: pytest.MonkeyPatch) -> None:
    dets = [
        make_read_result("1234567890123", x1=100, y1=100, x2=200, y2=300),
        make_read_result("9876543210987", x1=500, y1=100, x2=600, y2=300),
    ]
    monkeypatch.setattr(zxingcpp, "read_barcodes", _first_call_mock(dets))

    scanner = BarcodeScanner()
    result = scanner.scan_bytes(_png_bytes())

    values = [d.value for d in result]
    assert "1234567890123" in values
    assert "9876543210987" in values


def test_scan_bytes_no_barcodes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(zxingcpp, "read_barcodes", lambda _img, **kwargs: [])
    scanner = BarcodeScanner()
    assert scanner.scan_bytes(_png_bytes()) == []


def test_scan_bytes_invalid_image_raises() -> None:
    scanner = BarcodeScanner()
    with pytest.raises(UnidentifiedImageError):
        scanner.scan_bytes(b"not an image")


def test_scan_bytes_deduplicates_same_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    det = make_read_result("1111111111111")
    monkeypatch.setattr(zxingcpp, "read_barcodes", _first_call_mock([det, det]))
    scanner = BarcodeScanner()
    result = scanner.scan_bytes(_png_bytes())
    assert len(result) == 1


def test_scan_bytes_keeps_same_value_different_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two physical barcodes with the same value at different positions are kept."""
    det_a = make_read_result("1111111111111", x1=100, y1=100, x2=200, y2=300)
    det_b = make_read_result("1111111111111", x1=500, y1=500, x2=600, y2=700)
    monkeypatch.setattr(
        zxingcpp, "read_barcodes", _first_call_mock([det_a, det_b])
    )
    scanner = BarcodeScanner()
    result = scanner.scan_bytes(_png_bytes())
    assert len(result) == 2


def test_scanner_rejects_invalid_tile_rows() -> None:
    with pytest.raises(ValueError):
        BarcodeScanner(tile_rows=0)


def test_scanner_rejects_invalid_tile_columns() -> None:
    with pytest.raises(ValueError):
        BarcodeScanner(tile_columns=0)


def test_scanner_rejects_invalid_tile_overlap() -> None:
    with pytest.raises(ValueError):
        BarcodeScanner(tile_overlap=1.5)


def test_scanner_rejects_invalid_label_fallback_threshold() -> None:
    with pytest.raises(ValueError):
        BarcodeScanner(label_fallback_threshold=-1)

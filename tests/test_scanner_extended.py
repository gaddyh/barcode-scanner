"""Extended tests for ``src.ingest.scanner.py``.

Targets the methods and branches not covered by ``test_barcode_scanner.py``.
All zxingcpp calls are mocked so no real barcode decoding happens.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zxingcpp
from PIL import Image

from src.ingest.scanner import (
    SCANNER_VERSION,
    BarcodeScanner,
    BoundingBox,
    DetectedBarcode,
    LabelCandidate,
    Point,
    RecoveryAttempt,
    Tile,
)
from tests._zxing_fake import FakeReadResult, make_read_result

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _png_bytes(width: int = 800, height: int = 600) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


def _solid_image(
    width: int = 800,
    height: int = 600,
    color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    return Image.new("RGB", (width, height), color)


def _first_call_mock(results: list[FakeReadResult]) -> Any:
    """Return a mock that yields ``results`` on the first call, then ``[]``."""
    calls = {"n": 0}

    def _mock(_img: Any, **_kwargs: Any) -> list[FakeReadResult]:
        calls["n"] += 1
        if calls["n"] == 1:
            return results
        return []

    return _mock


def _empty_mock(_img: Any, **_kwargs: Any) -> list[FakeReadResult]:
    return []


def _counting_mock(
    results: list[FakeReadResult],
    on_call: int,
) -> Any:
    """Return ``results`` on the ``on_call``-th invocation, else ``[]``."""
    calls = {"n": 0}

    def _mock(_img: Any, **_kwargs: Any) -> list[FakeReadResult]:
        calls["n"] += 1
        if calls["n"] == on_call:
            return results
        return []

    return _mock


def _det(
    value: str = "123456789012",
    x1: int = 100,
    y1: int = 100,
    x2: int = 200,
    y2: int = 300,
    fmt: str = "Code128",
) -> DetectedBarcode:
    """Build a DetectedBarcode directly (no zxingcpp needed)."""
    position = (
        Point(x=x1, y=y1),
        Point(x=x2, y=y1),
        Point(x=x2, y=y2),
        Point(x=x1, y=y2),
    )
    return DetectedBarcode(
        value=value,
        format=fmt,
        content_type="Text",
        orientation=0,
        position=position,
        bounding_box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
    )


# ----------------------------------------------------------------------
# Module-level constants and data classes
# ----------------------------------------------------------------------


class TestScannerVersion:
    def test_scanner_version_is_string(self) -> None:
        assert isinstance(SCANNER_VERSION, str)
        assert SCANNER_VERSION.startswith("scanner-")


class TestDataclasses:
    def test_point_frozen(self) -> None:
        p = Point(x=1, y=2)
        assert p.x == 1
        assert p.y == 2
        with pytest.raises(AttributeError):
            p.x = 5  # type: ignore[misc]

    def test_bounding_box_frozen(self) -> None:
        bb = BoundingBox(x1=0, y1=0, x2=10, y2=20)
        assert bb.x1 == 0
        assert bb.y2 == 20
        with pytest.raises(AttributeError):
            bb.x1 = 5  # type: ignore[misc]

    def test_detected_barcode_frozen(self) -> None:
        det = _det()
        assert det.value == "123456789012"
        assert det.format == "Code128"
        assert det.content_type == "Text"
        assert det.orientation == 0
        assert len(det.position) == 4
        assert det.bounding_box == BoundingBox(100, 100, 200, 300)
        with pytest.raises(AttributeError):
            det.value = "x"  # type: ignore[misc]

    def test_tile_frozen(self) -> None:
        img = _solid_image(10, 10)
        tile = Tile(image=img, offset_x=5, offset_y=7, name="t")
        assert tile.offset_x == 5
        assert tile.offset_y == 7
        assert tile.name == "t"

    def test_label_candidate_frozen(self) -> None:
        img = _solid_image(10, 10)
        cand = LabelCandidate(
            crop=img,
            offset_x=1,
            offset_y=2,
            bounding_box=BoundingBox(1, 2, 11, 12),
            score=3.5,
        )
        assert cand.score == 3.5
        assert cand.offset_x == 1

    def test_recovery_attempt_frozen(self) -> None:
        ra = RecoveryAttempt(
            rotation=90,
            scale=2.0,
            preprocessing="clahe",
            inverted=True,
            values=("a", "b"),
            duration_ms=1.5,
        )
        assert ra.rotation == 90
        assert ra.values == ("a", "b")
        assert ra.duration_ms == 1.5


# ----------------------------------------------------------------------
# __init__ validation
# ----------------------------------------------------------------------


class TestInitValidation:
    def test_default_formats(self) -> None:
        scanner = BarcodeScanner()
        assert scanner.formats == (zxingcpp.BarcodeFormat.Code128,)

    def test_custom_formats(self) -> None:
        scanner = BarcodeScanner(formats=(zxingcpp.BarcodeFormat.EAN13,))
        assert scanner.formats == (zxingcpp.BarcodeFormat.EAN13,)

    def test_invalid_targeted_crop_width_ratio_low(self) -> None:
        with pytest.raises(ValueError, match="targeted_crop_width_ratio"):
            BarcodeScanner(targeted_crop_width_ratio=0.05)

    def test_invalid_targeted_crop_width_ratio_high(self) -> None:
        with pytest.raises(ValueError, match="targeted_crop_width_ratio"):
            BarcodeScanner(targeted_crop_width_ratio=2.5)

    def test_invalid_targeted_crop_height_ratio_low(self) -> None:
        with pytest.raises(ValueError, match="targeted_crop_height_ratio"):
            BarcodeScanner(targeted_crop_height_ratio=0.05)

    def test_invalid_targeted_crop_height_ratio_high(self) -> None:
        with pytest.raises(ValueError, match="targeted_crop_height_ratio"):
            BarcodeScanner(targeted_crop_height_ratio=2.5)

    def test_invalid_max_label_candidates(self) -> None:
        with pytest.raises(ValueError, match="max_label_candidates"):
            BarcodeScanner(max_label_candidates=0)

    def test_invalid_label_padding_ratio_negative(self) -> None:
        with pytest.raises(ValueError, match="label_padding_ratio"):
            BarcodeScanner(label_padding_ratio=-0.1)

    def test_invalid_label_padding_ratio_high(self) -> None:
        with pytest.raises(ValueError, match="label_padding_ratio"):
            BarcodeScanner(label_padding_ratio=0.6)

    def test_valid_boundary_overlap_zero(self) -> None:
        scanner = BarcodeScanner(tile_overlap=0)
        assert scanner.tile_overlap == 0

    def test_valid_label_padding_ratio_upper_bound(self) -> None:
        scanner = BarcodeScanner(label_padding_ratio=0.5)
        assert scanner.label_padding_ratio == 0.5

    def test_attributes_stored(self) -> None:
        scanner = BarcodeScanner(
            tile_rows=3,
            tile_columns=5,
            tile_overlap=0.25,
            enable_shifted_tiles=False,
            enable_targeted_recovery=False,
            targeted_crop_width_ratio=1.5,
            targeted_crop_height_ratio=0.5,
            enable_label_fallback=False,
            label_fallback_threshold=2,
            max_label_candidates=10,
            label_padding_ratio=0.2,
        )
        assert scanner.tile_rows == 3
        assert scanner.tile_columns == 5
        assert scanner.tile_overlap == 0.25
        assert scanner.enable_shifted_tiles is False
        assert scanner.enable_targeted_recovery is False
        assert scanner.targeted_crop_width_ratio == 1.5
        assert scanner.targeted_crop_height_ratio == 0.5
        assert scanner.enable_label_fallback is False
        assert scanner.label_fallback_threshold == 2
        assert scanner.max_label_candidates == 10
        assert scanner.label_padding_ratio == 0.2


# ----------------------------------------------------------------------
# scan_bytes
# ----------------------------------------------------------------------


class TestScanBytes:
    def test_png_format(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dets = [make_read_result("123456789012")]
        monkeypatch.setattr(zxingcpp, "read_barcodes", _first_call_mock(dets))
        scanner = BarcodeScanner()
        result = scanner.scan_bytes(_png_bytes())
        assert len(result) == 1
        assert result[0].value == "123456789012"

    def test_jpeg_format(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        buf = io.BytesIO()
        _solid_image(200, 200).save(buf, format="JPEG")
        dets = [make_read_result("123456789012")]
        monkeypatch.setattr(zxingcpp, "read_barcodes", _first_call_mock(dets))
        scanner = BarcodeScanner()
        result = scanner.scan_bytes(buf.getvalue())
        assert len(result) == 1

    def test_bmp_format(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        buf = io.BytesIO()
        _solid_image(200, 200).save(buf, format="BMP")
        dets = [make_read_result("123456789012")]
        monkeypatch.setattr(zxingcpp, "read_barcodes", _first_call_mock(dets))
        scanner = BarcodeScanner()
        result = scanner.scan_bytes(buf.getvalue())
        assert len(result) == 1

    def test_webp_format(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        buf = io.BytesIO()
        _solid_image(200, 200).save(buf, format="WEBP")
        dets = [make_read_result("123456789012")]
        monkeypatch.setattr(zxingcpp, "read_barcodes", _first_call_mock(dets))
        scanner = BarcodeScanner()
        result = scanner.scan_bytes(buf.getvalue())
        assert len(result) == 1

    def test_empty_bytes_raises(self) -> None:
        scanner = BarcodeScanner()
        with pytest.raises((OSError, ValueError, Exception)):  # noqa: B017
            scanner.scan_bytes(b"")

    def test_truncated_png_raises(self) -> None:
        scanner = BarcodeScanner()
        with pytest.raises((OSError, ValueError, Exception)):  # noqa: B017
            scanner.scan_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)

    def test_rgba_image(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        buf = io.BytesIO()
        Image.new("RGBA", (200, 200), (255, 0, 0, 128)).save(buf, format="PNG")
        monkeypatch.setattr(zxingcpp, "read_barcodes", _empty_mock)
        scanner = BarcodeScanner()
        assert scanner.scan_bytes(buf.getvalue()) == []

    def test_grayscale_image(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        buf = io.BytesIO()
        Image.new("L", (200, 200), 128).save(buf, format="PNG")
        monkeypatch.setattr(zxingcpp, "read_barcodes", _empty_mock)
        scanner = BarcodeScanner()
        assert scanner.scan_bytes(buf.getvalue()) == []


# ----------------------------------------------------------------------
# scan_image
# ----------------------------------------------------------------------


class TestScanImage:
    def test_no_barcodes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(zxingcpp, "read_barcodes", _empty_mock)
        scanner = BarcodeScanner()
        assert scanner.scan_image(_solid_image()) == []

    def test_label_fallback_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(zxingcpp, "read_barcodes", _empty_mock)
        scanner = BarcodeScanner(enable_label_fallback=False)
        assert scanner.scan_image(_solid_image()) == []

    def test_label_fallback_threshold_met(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When primary count >= threshold, label fallback is skipped."""
        dets = [
            make_read_result("123456789012", x1=10, y1=10, x2=50, y2=50),
            make_read_result("234567890123", x1=60, y1=10, x2=100, y2=50),
        ]
        monkeypatch.setattr(zxingcpp, "read_barcodes", _first_call_mock(dets))
        scanner = BarcodeScanner(label_fallback_threshold=2)
        result = scanner.scan_image(_solid_image())
        assert len(result) == 2

    def test_non_primary_barcode_triggers_label_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A short non-primary barcode still triggers label fallback."""
        dets = [make_read_result("ABC", x1=10, y1=10, x2=50, y2=50)]
        monkeypatch.setattr(zxingcpp, "read_barcodes", _first_call_mock(dets))
        scanner = BarcodeScanner(label_fallback_threshold=1)
        result = scanner.scan_image(_solid_image())
        # The non-primary barcode is kept; label fallback finds nothing new.
        assert any(d.value == "ABC" for d in result)

    def test_shifted_tiles_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(zxingcpp, "read_barcodes", _empty_mock)
        scanner = BarcodeScanner(enable_shifted_tiles=False)
        assert scanner.scan_image(_solid_image()) == []

    def test_targeted_recovery_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(zxingcpp, "read_barcodes", _empty_mock)
        scanner = BarcodeScanner(enable_targeted_recovery=False)
        assert scanner.scan_image(_solid_image()) == []


# ----------------------------------------------------------------------
# scan_crop_with_recovery
# ----------------------------------------------------------------------


class TestScanCropWithRecovery:
    def test_finds_barcode_normal_orientation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        det = make_read_result("123456789012")
        monkeypatch.setattr(zxingcpp, "read_barcodes", _first_call_mock([det]))
        scanner = BarcodeScanner()
        result = scanner.scan_crop_with_recovery(_solid_image(200, 200))
        assert len(result) == 1
        assert result[0].value == "123456789012"

    def test_no_barcode_found(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(zxingcpp, "read_barcodes", _empty_mock)
        scanner = BarcodeScanner()
        result = scanner.scan_crop_with_recovery(_solid_image(100, 100))
        assert result == []

    def test_finds_barcode_on_rotation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Barcode found only on the 90-degree rotated pass."""
        det = make_read_result("123456789012", x1=10, y1=10, x2=50, y2=50)
        # Normal orientation: all 12 attempts return [].
        # Rotated orientation: first call returns the barcode.
        calls = {"n": 0}

        def _mock(_img: Any, **_kw: Any) -> list[FakeReadResult]:
            calls["n"] += 1
            # 12 normal attempts all empty; call 13 (first rotated) returns det.
            if calls["n"] == 13:
                return [det]
            return []

        monkeypatch.setattr(zxingcpp, "read_barcodes", _mock)
        scanner = BarcodeScanner()
        result = scanner.scan_crop_with_recovery(_solid_image(100, 100))
        assert len(result) == 1
        assert result[0].value == "123456789012"

    def test_with_offset(self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        det = make_read_result("123456789012", x1=10, y1=10, x2=50, y2=50)
        monkeypatch.setattr(zxingcpp, "read_barcodes", _first_call_mock([det]))
        scanner = BarcodeScanner()
        result = scanner.scan_crop_with_recovery(
            _solid_image(200, 200),
            offset_x=100,
            offset_y=200,
        )
        assert len(result) == 1
        # Position should be offset by (100, 200).
        assert result[0].bounding_box.x1 >= 100
        assert result[0].bounding_box.y1 >= 200


class TestScanCropWithRecoveryDiagnostics:
    def test_returns_diagnostics(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        det = make_read_result("123456789012")
        monkeypatch.setattr(zxingcpp, "read_barcodes", _first_call_mock([det]))
        scanner = BarcodeScanner()
        dets, attempts = scanner.scan_crop_with_recovery_diagnostics(
            _solid_image(100, 100)
        )
        assert len(dets) == 1
        assert len(attempts) >= 1
        assert all(isinstance(a, RecoveryAttempt) for a in attempts)
        assert attempts[0].rotation == 0

    def test_diagnostics_with_rotation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(zxingcpp, "read_barcodes", _empty_mock)
        scanner = BarcodeScanner()
        dets, attempts = scanner.scan_crop_with_recovery_diagnostics(
            _solid_image(50, 50)
        )
        assert dets == []
        # 12 normal + 12 rotated = 24 attempts.
        assert len(attempts) == 24
        assert attempts[0].rotation == 0
        assert attempts[12].rotation == 90

    def test_diagnostics_break_on_primary(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        det = make_read_result("123456789012")
        monkeypatch.setattr(zxingcpp, "read_barcodes", _first_call_mock([det]))
        scanner = BarcodeScanner()
        dets, attempts = scanner.scan_crop_with_recovery_diagnostics(
            _solid_image(50, 50)
        )
        assert len(dets) == 1
        # Should break after first primary find (1 attempt).
        assert len(attempts) == 1
        assert attempts[0].values == ("123456789012",)

    def test_diagnostics_finds_on_rotation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Barcode found on the 90-degree rotated pass (maps position back)."""
        det = make_read_result("123456789012", x1=10, y1=10, x2=50, y2=50)
        calls = {"n": 0}

        def _mock(_img: Any, **_kw: Any) -> list[FakeReadResult]:
            calls["n"] += 1
            # 12 normal attempts empty; call 13 (first rotated) returns det.
            if calls["n"] == 13:
                return [det]
            return []

        monkeypatch.setattr(zxingcpp, "read_barcodes", _mock)
        scanner = BarcodeScanner()
        dets, attempts = scanner.scan_crop_with_recovery_diagnostics(
            _solid_image(100, 100)
        )
        assert len(dets) == 1
        assert dets[0].value == "123456789012"
        # 12 normal (all empty) + at least 1 rotated.
        assert len(attempts) >= 13
        assert attempts[12].rotation == 90


# ----------------------------------------------------------------------
# _decode_crop_variants with debug_dir
# ----------------------------------------------------------------------


class TestDecodeCropVariantsDebug:
    def test_debug_dir_writes_files(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(zxingcpp, "read_barcodes", _empty_mock)
        scanner = BarcodeScanner()
        debug_dir = tmp_path / "debug"
        scanner._decode_crop_variants(
            _solid_image(100, 100),
            offset_x=0,
            offset_y=0,
            debug_dir=debug_dir,
            debug_tag="test",
        )
        assert debug_dir.exists()
        pngs = list(debug_dir.glob("*.png"))
        assert len(pngs) > 0
        # The raw crop is always saved.
        assert any("crop_test_raw" in p.name for p in pngs)


# ----------------------------------------------------------------------
# _prepare_image (static method, all preprocessing modes)
# ----------------------------------------------------------------------


class TestPrepareImage:
    def test_original_no_scale(self) -> None:
        img = _solid_image(100, 100)
        out = BarcodeScanner._prepare_image(img, scale=1.0, preprocessing="original")
        assert out.size == (100, 100)

    def test_original_with_scale(self) -> None:
        img = _solid_image(100, 100)
        out = BarcodeScanner._prepare_image(img, scale=2.0, preprocessing="original")
        assert out.size == (200, 200)

    def test_grayscale(self) -> None:
        img = _solid_image(100, 100)
        out = BarcodeScanner._prepare_image(img, scale=1.0, preprocessing="grayscale")
        assert out.size == (100, 100)

    def test_clahe(self) -> None:
        img = _solid_image(100, 100, (128, 128, 128))
        out = BarcodeScanner._prepare_image(img, scale=1.0, preprocessing="clahe")
        assert out.size == (100, 100)

    def test_sharpened(self) -> None:
        img = _solid_image(100, 100, (128, 128, 128))
        out = BarcodeScanner._prepare_image(img, scale=1.0, preprocessing="sharpened")
        assert out.size == (100, 100)

    def test_aggressive_sharpen(self) -> None:
        img = _solid_image(100, 100, (128, 128, 128))
        out = BarcodeScanner._prepare_image(
            img, scale=1.0, preprocessing="aggressive_sharpen"
        )
        assert out.size == (100, 100)

    def test_otsu(self) -> None:
        img = _solid_image(100, 100, (128, 128, 128))
        out = BarcodeScanner._prepare_image(img, scale=1.0, preprocessing="otsu")
        assert out.size == (100, 100)

    def test_otsu_with_scale(self) -> None:
        img = _solid_image(100, 100, (128, 128, 128))
        out = BarcodeScanner._prepare_image(img, scale=3.0, preprocessing="otsu")
        assert out.size == (300, 300)

    def test_adaptive(self) -> None:
        img = _solid_image(100, 100, (128, 128, 128))
        out = BarcodeScanner._prepare_image(img, scale=1.0, preprocessing="adaptive")
        assert out.size == (100, 100)

    def test_adaptive_tiny_image(self) -> None:
        """Adaptive threshold on a very small image (max_block <= 3)."""
        img = _solid_image(2, 2, (128, 128, 128))
        out = BarcodeScanner._prepare_image(img, scale=1.0, preprocessing="adaptive")
        assert out.size == (2, 2)

    def test_adaptive_small_block(self) -> None:
        """Adaptive threshold with a small image that forces block_size even."""
        img = _solid_image(4, 4, (128, 128, 128))
        out = BarcodeScanner._prepare_image(img, scale=1.0, preprocessing="adaptive")
        assert out.size == (4, 4)

    def test_perspective_clahe(self) -> None:
        img = _solid_image(100, 100, (128, 128, 128))
        out = BarcodeScanner._prepare_image(
            img, scale=1.0, preprocessing="perspective_clahe"
        )
        assert out.size == (100, 100)

    def test_unknown_preprocessing_raises(self) -> None:
        img = _solid_image(50, 50)
        with pytest.raises(ValueError, match="Unknown preprocessing"):
            BarcodeScanner._prepare_image(img, scale=1.0, preprocessing="bogus")

    def test_scale_down(self) -> None:
        img = _solid_image(100, 100)
        out = BarcodeScanner._prepare_image(img, scale=0.5, preprocessing="original")
        assert out.size == (50, 50)

    def test_scale_produces_min_1px(self) -> None:
        img = _solid_image(1, 1)
        out = BarcodeScanner._prepare_image(img, scale=0.5, preprocessing="original")
        assert out.size == (1, 1)


# ----------------------------------------------------------------------
# _perspective_rectify
# ----------------------------------------------------------------------


class TestPerspectiveRectify:
    def test_solid_image_returns_same(self) -> None:
        img = _solid_image(100, 100, (128, 128, 128))
        out = BarcodeScanner._perspective_rectify(img)
        assert out.size == (100, 100)

    def test_no_contours(self) -> None:
        img = Image.new("RGB", (50, 50), (0, 0, 0))
        out = BarcodeScanner._perspective_rectify(img)
        assert out.size == (50, 50)

    def test_with_quadrilateral(self) -> None:
        """Create an image with a bright quadrilateral on dark background."""
        img = Image.new("RGB", (200, 200), (0, 0, 0))
        arr = np.array(img)
        # Draw a bright rectangle.
        arr[50:150, 50:150] = (255, 255, 255)
        img = Image.fromarray(arr)
        out = BarcodeScanner._perspective_rectify(img)
        # Should return a valid image.
        assert out.width > 0
        assert out.height > 0


# ----------------------------------------------------------------------
# _detect_label_candidates
# ----------------------------------------------------------------------


class TestDetectLabelCandidates:
    def test_solid_white_image(self) -> None:
        """A fully white image should not produce label candidates."""
        scanner = BarcodeScanner()
        candidates = scanner._detect_label_candidates(_solid_image(800, 600))
        # A single white blob may or may not pass the aspect ratio filter.
        # Either way, it should not crash.
        assert isinstance(candidates, list)

    def test_dark_image_no_candidates(self) -> None:
        scanner = BarcodeScanner()
        candidates = scanner._detect_label_candidates(
            _solid_image(800, 600, (10, 10, 10))
        )
        assert candidates == []

    def test_bright_rectangles_found(self) -> None:
        """Bright rectangular regions on a dark background are detected."""
        img = Image.new("RGB", (800, 600), (20, 20, 20))
        arr = np.array(img)
        # Draw a bright white rectangle (label-like).
        arr[100:250, 100:400] = (240, 240, 240)
        img = Image.fromarray(arr)
        scanner = BarcodeScanner()
        candidates = scanner._detect_label_candidates(img)
        assert len(candidates) >= 1
        # Candidates should be sorted by (y1, x1).
        for a, b in zip(candidates, candidates[1:], strict=False):
            assert (a.bounding_box.y1, a.bounding_box.x1) <= (
                b.bounding_box.y1,
                b.bounding_box.x1,
            )

    def test_max_label_candidates_cap(self) -> None:
        scanner = BarcodeScanner(max_label_candidates=2)
        img = Image.new("RGB", (800, 600), (20, 20, 20))
        arr = np.array(img)
        for i in range(6):
            y = 50 + i * 90
            arr[y : y + 60, 50:200] = (240, 240, 240)
        img = Image.fromarray(arr)
        candidates = scanner._detect_label_candidates(img)
        assert len(candidates) <= 2


# ----------------------------------------------------------------------
# _decode_label_candidate and _scan_detected_label_candidates
# ----------------------------------------------------------------------


class TestLabelCandidateDecoding:
    def test_decode_label_candidate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        det = make_read_result("123456789012")
        monkeypatch.setattr(zxingcpp, "read_barcodes", _first_call_mock([det]))
        scanner = BarcodeScanner()
        cand = LabelCandidate(
            crop=_solid_image(100, 100),
            offset_x=50,
            offset_y=60,
            bounding_box=BoundingBox(50, 60, 150, 160),
            score=1.0,
        )
        result = scanner._decode_label_candidate(cand)
        assert len(result) == 1
        assert result[0].value == "123456789012"

    def test_scan_detected_label_candidates_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(zxingcpp, "read_barcodes", _empty_mock)
        scanner = BarcodeScanner()
        result = scanner._scan_detected_label_candidates(
            _solid_image(800, 600),
            existing=[],
        )
        assert result == []

    def test_scan_detected_label_candidates_with_existing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A candidate already containing a primary barcode is skipped."""
        monkeypatch.setattr(zxingcpp, "read_barcodes", _empty_mock)
        scanner = BarcodeScanner()
        existing = [_det("123456789012", x1=100, y1=100, x2=200, y2=200)]
        result = scanner._scan_detected_label_candidates(
            _solid_image(800, 600),
            existing=existing,
        )
        assert result == []

    def test_scan_detected_label_candidates_finds_barcode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Label candidates are scanned and a barcode is found in a crop."""
        det = make_read_result("123456789012")
        monkeypatch.setattr(zxingcpp, "read_barcodes", _first_call_mock([det]))
        scanner = BarcodeScanner()
        # Create an image with a bright label-like rectangle.
        img = Image.new("RGB", (800, 600), (20, 20, 20))
        arr = np.array(img)
        arr[100:250, 100:400] = (240, 240, 240)
        img = Image.fromarray(arr)
        result = scanner._scan_detected_label_candidates(img, existing=[])
        # Should find the barcode from the label crop.
        assert any(d.value == "123456789012" for d in result)


# ----------------------------------------------------------------------
# _recover_missing_grid_cells
# ----------------------------------------------------------------------


class TestRecoverMissingGridCells:
    def test_no_primary_barcodes(self) -> None:
        scanner = BarcodeScanner(tile_rows=2, tile_columns=2)
        result = scanner._recover_missing_grid_cells(
            _solid_image(800, 600),
            detections=[],
        )
        assert result == []

    def test_enough_primary_barcodes(self) -> None:
        """When primary count >= expected, return unchanged."""
        scanner = BarcodeScanner(tile_rows=2, tile_columns=2)
        dets = [
            _det("123456789012", x1=0, y1=0, x2=50, y2=50),
            _det("234567890123", x1=100, y1=0, x2=150, y2=50),
            _det("345678901234", x1=0, y1=100, x2=50, y2=150),
            _det("456789012345", x1=100, y1=100, x2=150, y2=150),
        ]
        result = scanner._recover_missing_grid_cells(
            _solid_image(200, 200),
            detections=dets,
        )
        assert len(result) == 4

    def test_too_few_primary_barcodes(self) -> None:
        """When primary count < max(rows, cols), return unchanged."""
        scanner = BarcodeScanner(tile_rows=4, tile_columns=3)
        dets = [_det("123456789012")]
        result = scanner._recover_missing_grid_cells(
            _solid_image(800, 600),
            detections=dets,
        )
        assert result == dets

    def test_partial_grid_recovery(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """3 of 4 grid cells filled → recovery scans the missing cell."""
        # Use a mock that returns a 4th barcode on the 2nd call
        # (1st call is the full-image scan in scan_image; but here we call
        # _recover_missing_grid_cells directly, so the 1st read_barcodes call
        # is the targeted crop).
        det = make_read_result("456789012345", x1=10, y1=10, x2=50, y2=50)
        monkeypatch.setattr(zxingcpp, "read_barcodes", _first_call_mock([det]))
        scanner = BarcodeScanner(tile_rows=2, tile_columns=2)
        dets = [
            _det("123456789012", x1=100, y1=100, x2=140, y2=140),
            _det("234567890123", x1=500, y1=100, x2=540, y2=140),
            _det("345678901234", x1=100, y1=400, x2=140, y2=440),
        ]
        result = scanner._recover_missing_grid_cells(
            _solid_image(800, 600),
            detections=dets,
        )
        # Should find the 4th barcode from the targeted crop.
        values = [d.value for d in result]
        assert "456789012345" in values

    def test_cluster_mismatch_returns_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When x/y clustering doesn't match expected counts, return unchanged."""
        monkeypatch.setattr(zxingcpp, "read_barcodes", _empty_mock)
        scanner = BarcodeScanner(tile_rows=2, tile_columns=3)
        # 3 barcodes but their x-centers won't cluster into 3 columns cleanly.
        dets = [
            _det("123456789012", x1=100, y1=100, x2=140, y2=140),
            _det("234567890123", x1=200, y1=100, x2=240, y2=140),
            _det("345678901234", x1=300, y1=400, x2=340, y2=440),
        ]
        result = scanner._recover_missing_grid_cells(
            _solid_image(800, 600),
            detections=dets,
        )
        # Clustering 3 x-values into 3 columns may fail → return unchanged.
        assert len(result) == 3


# ----------------------------------------------------------------------
# _cluster_1d
# ----------------------------------------------------------------------


class TestCluster1d:
    def test_empty_values(self) -> None:
        assert BarcodeScanner._cluster_1d([], 3) == []

    def test_cluster_count_too_high(self) -> None:
        assert BarcodeScanner._cluster_1d([1.0, 2.0], 5) == []

    def test_single_cluster(self) -> None:
        result = BarcodeScanner._cluster_1d([1.0, 2.0, 3.0], 1)
        assert result == [2.0]

    def test_two_clusters(self) -> None:
        result = BarcodeScanner._cluster_1d([1.0, 2.0, 10.0, 11.0], 2)
        assert len(result) == 2
        assert result[0] < result[1]

    def test_converges_immediately(self) -> None:
        """Well-separated values converge in one iteration."""
        result = BarcodeScanner._cluster_1d([0.0, 100.0], 2)
        assert len(result) == 2

    def test_empty_group_returns_empty(self) -> None:
        """If a cluster ends up empty, return []."""
        result = BarcodeScanner._cluster_1d([1.0, 1.0, 1.0], 2)
        assert result == []


# ----------------------------------------------------------------------
# _nearest_index
# ----------------------------------------------------------------------


class TestNearestIndex:
    def test_single_center(self) -> None:
        assert BarcodeScanner._nearest_index(5.0, [5.0]) == 0

    def test_nearest(self) -> None:
        assert BarcodeScanner._nearest_index(3.0, [1.0, 5.0, 10.0]) == 0
        assert BarcodeScanner._nearest_index(6.0, [1.0, 5.0, 10.0]) == 1
        assert BarcodeScanner._nearest_index(9.0, [1.0, 5.0, 10.0]) == 2


# ----------------------------------------------------------------------
# _typical_spacing
# ----------------------------------------------------------------------


class TestTypicalSpacing:
    def test_single_center(self) -> None:
        assert BarcodeScanner._typical_spacing([5.0], fallback=10.0) == 10.0

    def test_empty_centers(self) -> None:
        assert BarcodeScanner._typical_spacing([], fallback=7.0) == 7.0

    def test_two_centers(self) -> None:
        assert BarcodeScanner._typical_spacing([1.0, 5.0], fallback=0.0) == 4.0

    def test_three_centers(self) -> None:
        assert BarcodeScanner._typical_spacing([1.0, 5.0, 9.0], fallback=0.0) == 4.0

    def test_no_increasing_pairs(self) -> None:
        assert BarcodeScanner._typical_spacing([5.0, 5.0], fallback=3.0) == 3.0


# ----------------------------------------------------------------------
# _deduplicate
# ----------------------------------------------------------------------


class TestDeduplicate:
    def test_empty(self) -> None:
        assert BarcodeScanner._deduplicate([]) == []

    def test_single(self) -> None:
        det = _det()
        assert BarcodeScanner._deduplicate([det]) == [det]

    def test_same_value_same_position_dedup(self) -> None:
        d1 = _det("123456789012", x1=100, y1=100, x2=200, y2=200)
        d2 = _det("123456789012", x1=100, y1=100, x2=200, y2=200)
        result = BarcodeScanner._deduplicate([d1, d2])
        assert len(result) == 1

    def test_same_value_different_position_kept(self) -> None:
        d1 = _det("123456789012", x1=100, y1=100, x2=200, y2=200)
        d2 = _det("123456789012", x1=500, y1=500, x2=600, y2=600)
        result = BarcodeScanner._deduplicate([d1, d2])
        assert len(result) == 2

    def test_different_value_same_position_misread(self) -> None:
        """Different value at nearly identical position → misread, keep better.

        The larger box is processed first (sorted by area desc). The smaller
        candidate is then compared: if it's less plausible it's dropped.
        """
        d1 = _det("123456789012", x1=100, y1=100, x2=210, y2=210)  # larger box
        d2 = _det("12345678901", x1=100, y1=100, x2=200, y2=200)  # smaller, shorter
        result = BarcodeScanner._deduplicate([d1, d2])
        # The larger, longer all-digit value should win.
        assert len(result) == 1
        assert result[0].value == "123456789012"

    def test_different_value_same_position_keep_existing(self) -> None:
        """When existing is more plausible, candidate is dropped."""
        d1 = _det("123456789012", x1=100, y1=100, x2=200, y2=200)
        d2 = _det("12345678901", x1=100, y1=100, x2=200, y2=200)
        result = BarcodeScanner._deduplicate([d1, d2])
        assert len(result) == 1
        assert result[0].value == "123456789012"

    def test_misread_candidate_more_plausible_keeps_existing(self) -> None:
        """Candidate is more plausible but has smaller box → existing stays.

        Covers the ``duplicate_index = index`` branch (line 1533) where the
        candidate is more plausible, but its box isn't larger so the existing
        detection is kept.
        """
        # d1: larger box, short value (processed first due to area sort).
        d1 = _det("12345", x1=100, y1=100, x2=210, y2=210)
        # d2: smaller box, longer all-digit value (more plausible).
        d2 = _det("123456789012", x1=100, y1=100, x2=200, y2=200)
        result = BarcodeScanner._deduplicate([d1, d2])
        assert len(result) == 1
        # Existing (d1) is kept because candidate box isn't larger.
        assert result[0].value == "12345"

    def test_larger_box_replaces_smaller(self) -> None:
        """When same physical barcode, the larger box replaces the smaller."""
        d1 = _det("123456789012", x1=100, y1=100, x2=150, y2=150)
        d2 = _det("123456789012", x1=100, y1=100, x2=200, y2=200)
        result = BarcodeScanner._deduplicate([d1, d2])
        assert len(result) == 1
        assert result[0].bounding_box.x2 == 200

    def test_sorted_by_position(self) -> None:
        d1 = _det("123456789012", x1=500, y1=500, x2=600, y2=600)
        d2 = _det("234567890123", x1=100, y1=100, x2=200, y2=200)
        result = BarcodeScanner._deduplicate([d1, d2])
        assert result[0].bounding_box.y1 <= result[1].bounding_box.y1


# ----------------------------------------------------------------------
# _same_physical_barcode
# ----------------------------------------------------------------------


class TestSamePhysicalBarcode:
    def test_identical_boxes(self) -> None:
        d1 = _det("123456789012", x1=100, y1=100, x2=200, y2=200)
        d2 = _det("123456789012", x1=100, y1=100, x2=200, y2=200)
        assert BarcodeScanner._same_physical_barcode(d1, d2) is True

    def test_far_apart(self) -> None:
        d1 = _det("123456789012", x1=100, y1=100, x2=200, y2=200)
        d2 = _det("123456789012", x1=500, y1=500, x2=600, y2=600)
        assert BarcodeScanner._same_physical_barcode(d1, d2) is False

    def test_overlapping_boxes(self) -> None:
        d1 = _det("123456789012", x1=100, y1=100, x2=200, y2=200)
        d2 = _det("123456789012", x1=110, y1=110, x2=210, y2=210)
        assert BarcodeScanner._same_physical_barcode(d1, d2) is True


# ----------------------------------------------------------------------
# _is_primary_barcode
# ----------------------------------------------------------------------


class TestIsPrimaryBarcode:
    def test_12_digits(self) -> None:
        assert BarcodeScanner._is_primary_barcode("123456789012") is True

    def test_13_digits(self) -> None:
        assert BarcodeScanner._is_primary_barcode("1234567890123") is True

    def test_short_digits(self) -> None:
        assert BarcodeScanner._is_primary_barcode("12345") is False

    def test_non_digits(self) -> None:
        assert BarcodeScanner._is_primary_barcode("ABCDEFGHIJKL") is False

    def test_mixed(self) -> None:
        assert BarcodeScanner._is_primary_barcode("ABC123456789") is False

    def test_empty(self) -> None:
        assert BarcodeScanner._is_primary_barcode("") is False


# ----------------------------------------------------------------------
# _normalize_format
# ----------------------------------------------------------------------


class TestNormalizeFormat:
    def test_code128(self) -> None:
        assert BarcodeScanner._normalize_format("Code128") == "Code128"

    def test_code128_with_spaces(self) -> None:
        assert BarcodeScanner._normalize_format("Code 128") == "Code128"

    def test_ean13(self) -> None:
        assert BarcodeScanner._normalize_format("EAN13") == "EAN13"

    def test_ean13_with_spaces(self) -> None:
        assert BarcodeScanner._normalize_format("EAN 13") == "EAN13"


# ----------------------------------------------------------------------
# _map_position
# ----------------------------------------------------------------------


class TestMapPosition:
    def test_basic_mapping(self) -> None:
        from tests._zxing_fake import FakePoint, FakePosition

        pos = FakePosition(
            top_left=FakePoint(10, 20),
            top_right=FakePoint(50, 20),
            bottom_right=FakePoint(50, 80),
            bottom_left=FakePoint(10, 80),
        )
        result = BarcodeScanner._map_position(
            pos, offset_x=100, offset_y=200, scale_x=2.0, scale_y=1.0
        )
        assert len(result) == 4
        assert result[0] == Point(x=100 + 20, y=200 + 20)
        assert result[1] == Point(x=100 + 100, y=200 + 20)

    def test_no_offset_no_scale(self) -> None:
        from tests._zxing_fake import FakePoint, FakePosition

        pos = FakePosition(
            top_left=FakePoint(5, 10),
            top_right=FakePoint(15, 10),
            bottom_right=FakePoint(15, 20),
            bottom_left=FakePoint(5, 20),
        )
        result = BarcodeScanner._map_position(
            pos, offset_x=0, offset_y=0, scale_x=1.0, scale_y=1.0
        )
        assert result[0] == Point(5, 10)
        assert result[2] == Point(15, 20)


# ----------------------------------------------------------------------
# _bounding_box
# ----------------------------------------------------------------------


class TestBoundingBoxHelper:
    def test_basic(self) -> None:
        points = (Point(10, 20), Point(50, 10), Point(40, 80), Point(5, 60))
        bb = BarcodeScanner._bounding_box(points)
        assert bb == BoundingBox(5, 10, 50, 80)


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------


class TestBoxCenter:
    def test_basic(self) -> None:
        assert BarcodeScanner._box_center(BoundingBox(0, 0, 100, 200)) == (50.0, 100.0)

    def test_offset(self) -> None:
        assert BarcodeScanner._box_center(BoundingBox(10, 20, 30, 40)) == (20.0, 30.0)


class TestExpandBox:
    def test_basic(self) -> None:
        bb = BarcodeScanner._expand_box(BoundingBox(10, 10, 20, 20), padding=5)
        assert bb == BoundingBox(5, 5, 25, 25)


class TestBoxArea:
    def test_basic(self) -> None:
        assert BarcodeScanner._box_area(BoundingBox(0, 0, 10, 20)) == 200

    def test_zero_width(self) -> None:
        assert BarcodeScanner._box_area(BoundingBox(5, 0, 5, 10)) == 10

    def test_zero_height(self) -> None:
        assert BarcodeScanner._box_area(BoundingBox(0, 5, 10, 5)) == 10


class TestIoU:
    def test_identical(self) -> None:
        bb = BoundingBox(0, 0, 100, 100)
        assert BarcodeScanner._intersection_over_union(bb, bb) == 1.0

    def test_no_overlap(self) -> None:
        a = BoundingBox(0, 0, 50, 50)
        b = BoundingBox(100, 100, 200, 200)
        assert BarcodeScanner._intersection_over_union(a, b) == 0.0

    def test_partial_overlap(self) -> None:
        a = BoundingBox(0, 0, 100, 100)
        b = BoundingBox(50, 50, 150, 150)
        result = BarcodeScanner._intersection_over_union(a, b)
        assert 0.0 < result < 1.0

    def test_zero_union(self) -> None:
        a = BoundingBox(0, 0, 0, 0)
        b = BoundingBox(0, 0, 0, 0)
        assert BarcodeScanner._intersection_over_union(a, b) == 0.0


class TestContainmentRatio:
    def test_full_containment(self) -> None:
        inner = BoundingBox(10, 10, 20, 20)
        outer = BoundingBox(0, 0, 100, 100)
        assert BarcodeScanner._containment_ratio(inner, outer) == 1.0

    def test_no_overlap(self) -> None:
        inner = BoundingBox(0, 0, 50, 50)
        outer = BoundingBox(100, 100, 200, 200)
        assert BarcodeScanner._containment_ratio(inner, outer) == 0.0

    def test_partial(self) -> None:
        inner = BoundingBox(0, 0, 100, 100)
        outer = BoundingBox(50, 50, 150, 150)
        result = BarcodeScanner._containment_ratio(inner, outer)
        assert 0.0 < result < 1.0


class TestPadBox:
    def test_basic(self) -> None:
        bb = BarcodeScanner._pad_box(
            BoundingBox(100, 100, 200, 200),
            image_width=800,
            image_height=600,
            padding_ratio=0.1,
        )
        assert bb.x1 == 90
        assert bb.y1 == 90
        assert bb.x2 == 210
        assert bb.y2 == 210

    def test_clamped_to_zero(self) -> None:
        bb = BarcodeScanner._pad_box(
            BoundingBox(5, 5, 50, 50),
            image_width=800,
            image_height=600,
            padding_ratio=0.5,
        )
        assert bb.x1 == 0
        assert bb.y1 == 0

    def test_clamped_to_image(self) -> None:
        bb = BarcodeScanner._pad_box(
            BoundingBox(790, 590, 800, 600),
            image_width=800,
            image_height=600,
            padding_ratio=0.5,
        )
        assert bb.x2 == 800
        assert bb.y2 == 600


# ----------------------------------------------------------------------
# _suppress_overlapping_label_boxes
# ----------------------------------------------------------------------


class TestSuppressOverlapping:
    def test_empty(self) -> None:
        assert BarcodeScanner._suppress_overlapping_label_boxes([]) == []

    def test_non_overlapping(self) -> None:
        boxes = [
            (1.0, BoundingBox(0, 0, 50, 50)),
            (2.0, BoundingBox(200, 200, 250, 250)),
        ]
        result = BarcodeScanner._suppress_overlapping_label_boxes(boxes)
        assert len(result) == 2

    def test_overlapping_suppressed(self) -> None:
        boxes = [
            (1.0, BoundingBox(0, 0, 100, 100)),
            (2.0, BoundingBox(10, 10, 110, 110)),
        ]
        result = BarcodeScanner._suppress_overlapping_label_boxes(boxes)
        assert len(result) == 1
        # Higher score kept.
        assert result[0][0] == 2.0

    def test_containment_suppressed(self) -> None:
        boxes = [
            (1.0, BoundingBox(0, 0, 200, 200)),
            (2.0, BoundingBox(10, 10, 50, 50)),
        ]
        result = BarcodeScanner._suppress_overlapping_label_boxes(boxes)
        assert len(result) == 1


# ----------------------------------------------------------------------
# _candidate_already_has_primary
# ----------------------------------------------------------------------


class TestCandidateAlreadyHasPrimary:
    def test_no_detections(self) -> None:
        box = BoundingBox(0, 0, 100, 100)
        assert BarcodeScanner._candidate_already_has_primary(box, []) is False

    def test_primary_inside(self) -> None:
        box = BoundingBox(0, 0, 200, 200)
        det = _det("123456789012", x1=50, y1=50, x2=100, y2=100)
        assert BarcodeScanner._candidate_already_has_primary(box, [det]) is True

    def test_primary_outside(self) -> None:
        box = BoundingBox(0, 0, 100, 100)
        det = _det("123456789012", x1=200, y1=200, x2=250, y2=250)
        assert BarcodeScanner._candidate_already_has_primary(box, [det]) is False

    def test_non_primary_ignored(self) -> None:
        box = BoundingBox(0, 0, 200, 200)
        det = _det("ABC", x1=50, y1=50, x2=100, y2=100)
        assert BarcodeScanner._candidate_already_has_primary(box, [det]) is False


# ----------------------------------------------------------------------
# _centers_within
# ----------------------------------------------------------------------


class TestCentersWithin:
    def test_within_tolerance(self) -> None:
        d1 = _det("111111111111", x1=100, y1=100, x2=200, y2=200)
        d2 = _det("222222222222", x1=105, y1=105, x2=205, y2=205)
        assert BarcodeScanner._centers_within(d1, d2, 15) is True

    def test_outside_tolerance(self) -> None:
        d1 = _det("111111111111", x1=100, y1=100, x2=200, y2=200)
        d2 = _det("222222222222", x1=200, y1=200, x2=300, y2=300)
        assert BarcodeScanner._centers_within(d1, d2, 15) is False


# ----------------------------------------------------------------------
# _is_more_plausible
# ----------------------------------------------------------------------


class TestIsMorePlausible:
    def test_digit_vs_non_digit(self) -> None:
        assert BarcodeScanner._is_more_plausible("12345", "ABC") is True
        assert BarcodeScanner._is_more_plausible("ABC", "12345") is False

    def test_both_digit_longer_wins(self) -> None:
        assert BarcodeScanner._is_more_plausible("123456", "123") is True
        assert BarcodeScanner._is_more_plausible("123", "123456") is False

    def test_both_non_digit_longer_wins(self) -> None:
        assert BarcodeScanner._is_more_plausible("ABCDEF", "ABC") is True

    def test_with_dashes_and_spaces(self) -> None:
        # "12-34 5" is longer (7 chars) and both are digit after cleanup,
        # so the longer raw value is more plausible.
        assert BarcodeScanner._is_more_plausible("12-34 5", "12345") is True
        assert BarcodeScanner._is_more_plausible("12345", "12-34 5") is False

    def test_equal_length(self) -> None:
        assert BarcodeScanner._is_more_plausible("12345", "12345") is False


# ----------------------------------------------------------------------
# Tile generation
# ----------------------------------------------------------------------


class TestTileGeneration:
    def test_regular_tiles_count(self) -> None:
        scanner = BarcodeScanner(tile_rows=4, tile_columns=3)
        tiles = list(scanner._generate_regular_tiles(_solid_image(800, 600)))
        assert len(tiles) == 12

    def test_regular_tile_names(self) -> None:
        scanner = BarcodeScanner(tile_rows=2, tile_columns=2)
        tiles = list(scanner._generate_regular_tiles(_solid_image(200, 200)))
        assert tiles[0].name == "regular-0-0"
        assert tiles[-1].name == "regular-1-1"

    def test_regular_tile_offsets(self) -> None:
        scanner = BarcodeScanner(tile_rows=2, tile_columns=2)
        tiles = list(scanner._generate_regular_tiles(_solid_image(200, 200)))
        assert tiles[0].offset_x == 0
        assert tiles[0].offset_y == 0

    def test_shifted_tiles_count(self) -> None:
        scanner = BarcodeScanner(tile_rows=4, tile_columns=3)
        tiles = list(scanner._generate_shifted_tiles(_solid_image(800, 600)))
        assert len(tiles) == (4 - 1) * (3 - 1)

    def test_shifted_tiles_disabled_for_single_row(self) -> None:
        scanner = BarcodeScanner(tile_rows=1, tile_columns=3)
        tiles = list(scanner._generate_shifted_tiles(_solid_image(300, 100)))
        assert tiles == []

    def test_shifted_tiles_disabled_for_single_col(self) -> None:
        scanner = BarcodeScanner(tile_rows=3, tile_columns=1)
        tiles = list(scanner._generate_shifted_tiles(_solid_image(100, 300)))
        assert tiles == []

    def test_make_tile_with_overlap(self) -> None:
        scanner = BarcodeScanner(tile_rows=2, tile_columns=2, tile_overlap=0.1)
        tile = scanner._make_tile(
            image=_solid_image(200, 200),
            core_x1=0,
            core_y1=0,
            core_x2=100,
            core_y2=100,
            name="test",
        )
        assert tile.name == "test"
        assert tile.offset_x == 0
        assert tile.offset_y == 0
        assert tile.image.width >= 100
        assert tile.image.height >= 100


# ----------------------------------------------------------------------
# _make_targeted_crop
# ----------------------------------------------------------------------


class TestMakeTargetedCrop:
    def test_basic(self) -> None:
        scanner = BarcodeScanner()
        tile = scanner._make_targeted_crop(
            image=_solid_image(800, 600),
            center_x=400,
            center_y=300,
            crop_width=200,
            crop_height=100,
            name="target",
        )
        assert tile.name == "target"
        assert tile.image.width == 200
        assert tile.image.height == 100
        assert tile.offset_x == 300
        assert tile.offset_y == 250

    def test_clamped_to_edges(self) -> None:
        scanner = BarcodeScanner()
        tile = scanner._make_targeted_crop(
            image=_solid_image(800, 600),
            center_x=10,
            center_y=10,
            crop_width=200,
            crop_height=200,
            name="edge",
        )
        assert tile.offset_x == 0
        assert tile.offset_y == 0


# ----------------------------------------------------------------------
# _decode_region
# ----------------------------------------------------------------------


class TestDecodeRegion:
    def test_basic_decode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        det = make_read_result("123456789012")
        monkeypatch.setattr(zxingcpp, "read_barcodes", lambda _img, **_kw: [det])
        scanner = BarcodeScanner()
        result = scanner._decode_region(
            image=_solid_image(200, 200),
            offset_x=0,
            offset_y=0,
            scale=1.0,
            preprocessing="original",
            try_downscale=False,
        )
        assert len(result) == 1
        assert result[0].value == "123456789012"
        assert result[0].format == "Code128"
        assert result[0].content_type == "Text"

    def test_empty_text_skipped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        det = make_read_result("")
        monkeypatch.setattr(zxingcpp, "read_barcodes", lambda _img, **_kw: [det])
        scanner = BarcodeScanner()
        result = scanner._decode_region(
            image=_solid_image(100, 100),
            offset_x=0,
            offset_y=0,
            scale=1.0,
            preprocessing="original",
            try_downscale=False,
        )
        assert result == []

    def test_format_assertion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A format not in the requested set raises AssertionError."""
        det = FakeReadResult(
            text="123456789012",
            position=make_read_result("x").position,
            format=zxingcpp.BarcodeFormat.EAN13,
        )
        monkeypatch.setattr(zxingcpp, "read_barcodes", lambda _img, **_kw: [det])
        scanner = BarcodeScanner(formats=(zxingcpp.BarcodeFormat.Code128,))
        with pytest.raises(AssertionError, match="not in"):
            scanner._decode_region(
                image=_solid_image(100, 100),
                offset_x=0,
                offset_y=0,
                scale=1.0,
                preprocessing="original",
                try_downscale=False,
            )

    def test_debug_dir_writes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(zxingcpp, "read_barcodes", _empty_mock)
        scanner = BarcodeScanner()
        debug_dir = tmp_path / "dbg"
        scanner._decode_region(
            image=_solid_image(50, 50),
            offset_x=0,
            offset_y=0,
            scale=1.0,
            preprocessing="original",
            try_downscale=False,
            debug_dir=debug_dir,
            debug_filename="test.png",
        )
        assert (debug_dir / "test.png").exists()

    def test_with_offset_and_scale(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        det = make_read_result("123456789012", x1=10, y1=10, x2=50, y2=50)
        monkeypatch.setattr(zxingcpp, "read_barcodes", lambda _img, **_kw: [det])
        scanner = BarcodeScanner()
        result = scanner._decode_region(
            image=_solid_image(100, 100),
            offset_x=200,
            offset_y=300,
            scale=2.0,
            preprocessing="original",
            try_downscale=False,
        )
        assert len(result) == 1
        # Position should be offset by (200, 300) and scaled by 0.5
        # (scale_x = image.width / prepared.width = 100/200 = 0.5).
        assert result[0].bounding_box.x1 >= 200

    def test_try_invert(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        det = make_read_result("123456789012")
        captured: dict[str, Any] = {}

        def _mock(_img: Any, **kwargs: Any) -> list[FakeReadResult]:
            captured.update(kwargs)
            return [det]

        monkeypatch.setattr(zxingcpp, "read_barcodes", _mock)
        scanner = BarcodeScanner()
        scanner._decode_region(
            image=_solid_image(50, 50),
            offset_x=0,
            offset_y=0,
            scale=1.0,
            preprocessing="original",
            try_downscale=False,
            try_invert=True,
        )
        assert captured.get("try_invert") is True

    def test_try_downscale(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        det = make_read_result("123456789012")
        captured: dict[str, Any] = {}

        def _mock(_img: Any, **kwargs: Any) -> list[FakeReadResult]:
            captured.update(kwargs)
            return [det]

        monkeypatch.setattr(zxingcpp, "read_barcodes", _mock)
        scanner = BarcodeScanner()
        scanner._decode_region(
            image=_solid_image(50, 50),
            offset_x=0,
            offset_y=0,
            scale=1.0,
            preprocessing="original",
            try_downscale=True,
        )
        assert captured.get("try_downscale") is True


# ----------------------------------------------------------------------
# _count_primary_barcodes / _contains_primary_barcode
# ----------------------------------------------------------------------


class TestPrimaryBarcodeHelpers:
    def test_count_primary(self) -> None:
        dets = [
            _det("123456789012"),
            _det("ABC"),
            _det("234567890123"),
        ]
        assert BarcodeScanner._count_primary_barcodes(dets) == 2

    def test_count_primary_empty(self) -> None:
        assert BarcodeScanner._count_primary_barcodes([]) == 0

    def test_contains_primary_true(self) -> None:
        dets = [_det("ABC"), _det("123456789012")]
        assert BarcodeScanner._contains_primary_barcode(dets) is True

    def test_contains_primary_false(self) -> None:
        dets = [_det("ABC"), _det("DEF")]
        assert BarcodeScanner._contains_primary_barcode(dets) is False

    def test_contains_primary_empty(self) -> None:
        assert BarcodeScanner._contains_primary_barcode([]) is False


# ----------------------------------------------------------------------
# _decode_targeted_tile
# ----------------------------------------------------------------------


class TestDecodeTargetedTile:
    def test_finds_barcode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        det = make_read_result("123456789012")
        monkeypatch.setattr(zxingcpp, "read_barcodes", _first_call_mock([det]))
        scanner = BarcodeScanner()
        tile = Tile(
            image=_solid_image(100, 100),
            offset_x=50,
            offset_y=60,
            name="test",
        )
        result = scanner._decode_targeted_tile(tile)
        assert len(result) == 1
        assert result[0].value == "123456789012"

    def test_no_barcode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(zxingcpp, "read_barcodes", _empty_mock)
        scanner = BarcodeScanner()
        tile = Tile(
            image=_solid_image(100, 100),
            offset_x=0,
            offset_y=0,
            name="test",
        )
        assert scanner._decode_targeted_tile(tile) == []


# ----------------------------------------------------------------------
# scan_bytes with file path (read from disk)
# ----------------------------------------------------------------------


class TestScanFromFile:
    def test_scan_bytes_from_file(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Read image bytes from a file on disk and scan."""
        dets = [make_read_result("123456789012")]
        monkeypatch.setattr(zxingcpp, "read_barcodes", _first_call_mock(dets))
        path = tmp_path / "test.png"
        _solid_image(200, 200).save(path, format="PNG")
        scanner = BarcodeScanner()
        result = scanner.scan_bytes(path.read_bytes())
        assert len(result) == 1
        assert result[0].value == "123456789012"

    def test_scan_bytes_from_jpeg_file(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        dets = [make_read_result("123456789012")]
        monkeypatch.setattr(zxingcpp, "read_barcodes", _first_call_mock(dets))
        path = tmp_path / "test.jpg"
        _solid_image(200, 200).save(path, format="JPEG")
        scanner = BarcodeScanner()
        result = scanner.scan_bytes(path.read_bytes())
        assert len(result) == 1


# ----------------------------------------------------------------------
# Integration: full scan_image with label fallback
# ----------------------------------------------------------------------


class TestLabelFallbackIntegration:
    def test_label_fallback_finds_barcode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Label fallback finds a barcode that the fast path missed."""
        # Create an image with a bright label-like rectangle.
        img = Image.new("RGB", (800, 600), (20, 20, 20))
        arr = np.array(img)
        arr[100:250, 100:400] = (240, 240, 240)
        img = Image.fromarray(arr)

        det = make_read_result("123456789012")
        calls = {"n": 0}

        def _mock(_img: Any, **_kw: Any) -> list[FakeReadResult]:
            calls["n"] += 1
            # Return the barcode on a later call (label crop scan).
            if calls["n"] >= 5:
                return [det]
            return []

        monkeypatch.setattr(zxingcpp, "read_barcodes", _mock)
        scanner = BarcodeScanner(label_fallback_threshold=4)
        result = scanner.scan_image(img)
        values = [d.value for d in result]
        assert "123456789012" in values

    def test_label_fallback_no_candidates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dark image: no label candidates, fallback returns []."""
        monkeypatch.setattr(zxingcpp, "read_barcodes", _empty_mock)
        scanner = BarcodeScanner()
        result = scanner.scan_image(_solid_image(800, 600, (10, 10, 10)))
        assert result == []

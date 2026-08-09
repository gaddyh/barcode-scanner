"""Tests for the product API (analyze_image) — the clean happy path.

Mocks ``BarcodeScanner`` and ``pipeline._traced_audit`` so no real barcode
images or Gemini API calls are required.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from src.ingest.analyze import analyze_image
from src.ingest.scanner import (
    BoundingBox,
    DetectedBarcode,
    Point,
)
from src.ingest.vision import (
    AuditConfidence,
    SpatialLabelAuditPixels,
    SpatialLabelObservationPixels,
    SpatialLabelStatus,
)
from src.ingest.geometry import PixelBoundingBox

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _png_path(tmp_path: Path, width: int = 800, height: int = 600) -> Path:
    p = tmp_path / "img.png"
    Image.new("RGB", (width, height), (255, 255, 255)).save(p, format="PNG")
    return p


def _detection(
    value: str,
    *,
    x1: int = 110,
    y1: int = 110,
    x2: int = 190,
    y2: int = 290,
    fmt: str = "Code128",
) -> DetectedBarcode:
    return DetectedBarcode(
        value=value,
        format=fmt,
        content_type="text",
        orientation=0,
        position=(Point(x=x1, y=y1), Point(x=x2, y=y1)),
        bounding_box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
    )


def _detection_dict(
    value: str,
    *,
    x1: int = 110,
    y1: int = 110,
    x2: int = 190,
    y2: int = 290,
    fmt: str = "Code128",
) -> dict:
    d = _detection(value, x1=x1, y1=y1, x2=x2, y2=y2, fmt=fmt)
    from dataclasses import asdict
    return asdict(d)


def _label_pixels(
    index: int,
    *,
    label_box: tuple[int, int, int, int],
    barcode_box: tuple[int, int, int, int] | None = None,
    status: SpatialLabelStatus = SpatialLabelStatus.CLEAR,
) -> SpatialLabelObservationPixels:
    return SpatialLabelObservationPixels(
        label_index=index,
        label_bbox=PixelBoundingBox(
            x1=label_box[0], y1=label_box[1], x2=label_box[2], y2=label_box[3]
        ),
        barcode_bbox=(
            PixelBoundingBox(
                x1=barcode_box[0], y1=barcode_box[1],
                x2=barcode_box[2], y2=barcode_box[3],
            )
            if barcode_box is not None
            else None
        ),
        status=status,
        confidence=AuditConfidence.HIGH,
    )


def _spatial(
    labels: list[SpatialLabelObservationPixels],
    *,
    image_width: int = 800,
    image_height: int = 600,
) -> SpatialLabelAuditPixels:
    return SpatialLabelAuditPixels(
        image_width=image_width,
        image_height=image_height,
        labels=labels,
    )


def _mock_scan(detections: list[DetectedBarcode]) -> object:
    """Return a fake BarcodeScanner whose scan_bytes returns ``detections``."""
    class _FakeScanner:
        def scan_bytes(self, image_bytes: bytes) -> list[DetectedBarcode]:
            return detections

        def scan_crop_with_recovery(self, crop, *, offset_x=0, offset_y=0):
            return []
    return _FakeScanner()


def _patch_audit_ok(spatial: SpatialLabelAuditPixels) -> object:
    """Patch pipeline._traced_audit to return a successful audit result."""
    def _fake(path, *, model, max_retries, retry_delay_seconds):
        return {"status": "ok", "spatial": spatial.model_dump(mode="json")}
    return patch("src.ingest.pipeline._traced_audit", side_effect=_fake)


def _patch_audit_error(error: dict) -> object:
    def _fake(path, *, model, max_retries, retry_delay_seconds):
        return {"status": "error", "error": error}
    return patch("src.ingest.pipeline._traced_audit", side_effect=_fake)


# ---------------------------------------------------------------------------
# Happy path: complete
# ---------------------------------------------------------------------------


def test_complete_all_labels_found(tmp_path: Path) -> None:
    img = _png_path(tmp_path)
    detections = [
        _detection("111", x1=110, y1=110, x2=190, y2=290),
        _detection("222", x1=510, y1=110, x2=590, y2=290),
    ]
    spatial = _spatial([
        _label_pixels(1, label_box=(50, 50, 250, 350), barcode_box=(100, 100, 200, 300)),
        _label_pixels(2, label_box=(450, 50, 650, 350), barcode_box=(500, 100, 600, 300)),
    ])

    scanner = _mock_scan(detections)
    with _patch_audit_ok(spatial):
        result = analyze_image(img, scanner=scanner)

    assert result["outcome"] == "complete"
    assert result["ok"] is True
    assert result["audit_available"] is True
    assert result["summary"]["visible_label_count"] == 2
    assert result["summary"]["found_count"] == 2
    assert result["summary"]["missing_count"] == 0
    assert result["summary"]["all_found"] is True
    assert {f["barcode_value"] for f in result["found"]} == {"111", "222"}
    assert "annotated_image_b64" not in result
    assert "error" not in result


# ---------------------------------------------------------------------------
# needs_better_photo
# ---------------------------------------------------------------------------


def test_needs_better_photo_one_label_missing(tmp_path: Path) -> None:
    img = _png_path(tmp_path)
    detections = [_detection("111", x1=110, y1=110, x2=190, y2=290)]
    spatial = _spatial([
        _label_pixels(1, label_box=(50, 50, 250, 350), barcode_box=(100, 100, 200, 300)),
        _label_pixels(2, label_box=(450, 50, 650, 350), barcode_box=(500, 100, 600, 300)),
    ])

    scanner = _mock_scan(detections)
    with _patch_audit_ok(spatial):
        result = analyze_image(img, scanner=scanner)

    assert result["outcome"] == "needs_better_photo"
    assert result["summary"]["found_count"] == 1
    assert result["summary"]["missing_count"] == 1
    assert result["summary"]["all_found"] is False
    assert result["missing"][0]["label_index"] == 2
    assert "annotated_image_b64" in result
    assert "message" in result
    # The annotated image is a decodable PNG.
    png_bytes = base64.b64decode(result["annotated_image_b64"])
    Image.open(io.BytesIO(png_bytes)).verify()


def test_needs_better_photo_zero_labels(tmp_path: Path) -> None:
    img = _png_path(tmp_path)
    detections = [_detection("111")]
    spatial = _spatial([])

    scanner = _mock_scan(detections)
    with _patch_audit_ok(spatial):
        result = analyze_image(img, scanner=scanner)

    assert result["outcome"] == "needs_better_photo"
    assert result["summary"]["visible_label_count"] == 0
    assert result["summary"]["found_count"] == 0
    assert "annotated_image_b64" in result
    assert "No barcode labels" in result["message"]


# ---------------------------------------------------------------------------
# retryable_error
# ---------------------------------------------------------------------------


def test_retryable_error_on_audit_failure(tmp_path: Path) -> None:
    img = _png_path(tmp_path)
    detections = [_detection("111")]
    scanner = _mock_scan(detections)
    with _patch_audit_error({"type": "ShoeboxAuditError", "message": "boom"}):
        result = analyze_image(img, scanner=scanner)

    assert result["outcome"] == "retryable_error"
    assert result["audit_available"] is False
    assert result["error"] == {"type": "ShoeboxAuditError", "message": "boom"}
    # Scanner detections become unassigned.
    assert len(result["unassigned"]) == 1
    assert result["unassigned"][0]["barcode_value"] == "111"
    assert result["summary"]["found_count"] == 0


def test_retryable_error_on_scan_error(tmp_path: Path) -> None:
    img = _png_path(tmp_path)

    class _BrokenScanner:
        def scan_bytes(self, image_bytes: bytes):
            raise ValueError("bad image")

        def scan_crop_with_recovery(self, crop, *, offset_x=0, offset_y=0):
            raise ValueError("bad image")

    spatial = _spatial([_label_pixels(1, label_box=(50, 50, 250, 350))])
    with _patch_audit_ok(spatial):
        result = analyze_image(img, scanner=_BrokenScanner())

    assert result["outcome"] == "retryable_error"
    assert result["ok"] is False
    assert result["audit_available"] is False
    assert result["error"]["code"] == "invalid_image"


# ---------------------------------------------------------------------------
# Bytes input
# ---------------------------------------------------------------------------


def test_analyze_image_accepts_bytes(tmp_path: Path) -> None:
    buf = io.BytesIO()
    Image.new("RGB", (800, 600), (255, 255, 255)).save(buf, format="PNG")
    image_bytes = buf.getvalue()

    detections = [_detection("111")]
    spatial = _spatial([
        _label_pixels(1, label_box=(50, 50, 250, 350), barcode_box=(100, 100, 200, 300)),
    ])

    scanner = _mock_scan(detections)
    with _patch_audit_ok(spatial):
        result = analyze_image(image_bytes, scanner=scanner)

    assert result["outcome"] == "complete"
    assert result["image_width"] == 800
    assert result["image_height"] == 600


# ---------------------------------------------------------------------------
# Unassigned detections
# ---------------------------------------------------------------------------


def test_unassigned_detection_reported(tmp_path: Path) -> None:
    img = _png_path(tmp_path)
    detections = [
        _detection("111", x1=110, y1=110, x2=190, y2=290),
        _detection("999", x1=700, y1=500, x2=780, y2=580),
    ]
    spatial = _spatial([
        _label_pixels(1, label_box=(50, 50, 250, 350), barcode_box=(100, 100, 200, 300)),
    ])

    scanner = _mock_scan(detections)
    with _patch_audit_ok(spatial):
        result = analyze_image(img, scanner=scanner)

    # One label found, one unassigned detection.
    assert result["summary"]["found_count"] == 1
    assert result["summary"]["unassigned_count"] == 1
    assert result["unassigned"][0]["barcode_value"] == "999"

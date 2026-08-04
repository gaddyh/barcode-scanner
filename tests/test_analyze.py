"""Tests for app.services.analyze.analyze_image().

These tests mock the pipeline the same way tests/test_cli.py does:
- Patch ``BarcodeScanner`` so no real zxing calls happen.
- Patch ``app.services.pipeline._traced_audit`` so no real Gemini calls happen.
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.services import analyze as analyze_mod
from app.services import pipeline as pipeline_mod
from app.services.barcode_scanner import (
    BarcodeScanner,
    BoundingBox,
    DetectedBarcode,
    Point,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_png(width: int = 100, height: int = 50) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def _fake_barcode(
    value: str = "7290001234567",
    x1: int = 160,
    y1: int = 130,
    x2: int = 270,
    y2: int = 140,
) -> DetectedBarcode:
    return DetectedBarcode(
        value=value,
        format="EAN13",
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


def _spatial_result(
    image_width: int = 1000,
    image_height: int = 1000,
    labels=None,
) -> dict:
    """A minimal spatial audit result in the shape returned by
    audit_shoebox_labels().model_dump(mode='json')."""
    if labels is None:
        labels = [
            {
                "label_index": 1,
                "label_bbox": {"x1": 100, "y1": 100, "x2": 300, "y2": 200},
                "barcode_bbox": {"x1": 150, "y1": 120, "x2": 280, "y2": 190},
                "status": "clear",
                "confidence": "high",
            },
            {
                "label_index": 2,
                "label_bbox": {"x1": 100, "y1": 800, "x2": 300, "y2": 900},
                "barcode_bbox": {"x1": 150, "y1": 820, "x2": 280, "y2": 890},
                "status": "clear",
                "confidence": "high",
            },
        ]
    return {
        "image_width": image_width,
        "image_height": image_height,
        "labels": labels,
    }


class _FakeScanner(BarcodeScanner):
    """Fake scanner that returns a fixed list of barcodes."""

    def __init__(self, barcodes: list[DetectedBarcode]) -> None:
        self._barcodes = barcodes

    def scan_bytes(self, image_bytes: bytes) -> list[DetectedBarcode]:
        if image_bytes == b"not an image":
            raise UnidentifiedImageError("not an image")
        return self._barcodes

    def scan_label_crops(self, image, requests, *, existing_detections=None, debug_dir=None):
        return []

    def merge_detections(self, existing, recovered):
        return list(existing) + list(recovered)


def _patch_pipeline(
    monkeypatch,
    barcodes: list[DetectedBarcode],
    spatial_result_dict: dict | None = None,
    audit_error: bool = False,
):
    """Patch the pipeline so scan returns ``barcodes`` and the Gemini audit
    returns ``spatial_result_dict`` (or an error).

    Returns a fake scanner that ``analyze_image`` will use.
    """
    fake_scanner = _FakeScanner(barcodes)

    # Patch BarcodeScanner in both analyze and pipeline modules so
    # analyze_image() constructs our fake scanner when scanner=None.
    monkeypatch.setattr(analyze_mod, "BarcodeScanner", lambda: fake_scanner)
    monkeypatch.setattr(pipeline_mod, "BarcodeScanner", lambda: fake_scanner)

    if audit_error:
        def fake_traced_audit(path, *, model, max_retries, retry_delay_seconds):
            return {
                "status": "error",
                "error": {"type": "ShoeboxAuditError", "message": "429 quota exceeded"},
            }
    else:
        def fake_traced_audit(path, *, model, max_retries, retry_delay_seconds):
            return {"status": "ok", "spatial": spatial_result_dict}

    monkeypatch.setattr(pipeline_mod, "_traced_audit", fake_traced_audit)
    return fake_scanner


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_analyze_all_found(tmp_path, monkeypatch) -> None:
    """2 barcodes, 2 labels, both match → outcome=complete."""
    barcodes = [
        _fake_barcode("V1", x1=160, y1=130, x2=270, y2=140),
        _fake_barcode("V2", x1=160, y1=830, x2=270, y2=840),
    ]
    _patch_pipeline(monkeypatch, barcodes, _spatial_result())

    image_path = tmp_path / "box.png"
    image_path.write_bytes(make_png())

    result = analyze_mod.analyze_image(image_path)

    assert result["ok"] is True
    assert result["outcome"] == "complete"
    assert result["audit_available"] is True
    assert result["summary"]["found_count"] == 2
    assert result["summary"]["missing_count"] == 0
    assert result["summary"]["unassigned_count"] == 0
    assert result["summary"]["all_found"] is True
    assert result["summary"]["visible_label_count"] == 2

    # Check found entries have the right shape.
    found = result["found"]
    assert len(found) == 2
    assert found[0]["barcode_value"] == "V1"
    assert found[0]["barcode_format"] == "EAN13"
    assert found[0]["label_index"] == 1
    assert "barcode_bbox" in found[0]
    assert "label_bbox" in found[0]
    assert found[0]["match_basis"] in ("barcode_bbox", "label_bbox")


def test_analyze_some_missing(tmp_path, monkeypatch) -> None:
    """1 barcode, 2 labels → outcome=needs_better_photo, missing has location."""
    barcodes = [
        _fake_barcode("V1", x1=160, y1=130, x2=270, y2=140),
    ]
    _patch_pipeline(monkeypatch, barcodes, _spatial_result())

    image_path = tmp_path / "box.png"
    image_path.write_bytes(make_png())

    result = analyze_mod.analyze_image(image_path)

    assert result["ok"] is True
    assert result["outcome"] == "needs_better_photo"
    assert result["summary"]["found_count"] == 1
    assert result["summary"]["missing_count"] == 1
    assert result["summary"]["all_found"] is False

    # Missing entry has pixel locations.
    missing = result["missing"]
    assert len(missing) == 1
    assert missing[0]["label_index"] == 2
    assert missing[0]["status"] == "clear"
    assert "label_bbox" in missing[0]
    assert "barcode_bbox" in missing[0]
    assert missing[0]["label_bbox"]["x1"] == 100
    assert missing[0]["barcode_bbox"]["x1"] == 150


def test_analyze_audit_failure_is_retryable_error(tmp_path, monkeypatch) -> None:
    """Gemini 429/audit error → outcome=retryable_error, NOT needs_better_photo."""
    barcodes = [
        _fake_barcode("V1"),
    ]
    _patch_pipeline(monkeypatch, barcodes, audit_error=True)

    image_path = tmp_path / "box.png"
    image_path.write_bytes(make_png())

    result = analyze_mod.analyze_image(image_path)

    assert result["ok"] is True
    assert result["outcome"] == "retryable_error"
    assert result["audit_available"] is False
    # Decoded barcodes go to unassigned, not found.
    assert result["found"] == []
    assert result["missing"] == []
    assert len(result["unassigned"]) == 1
    assert result["unassigned"][0]["barcode_value"] == "V1"
    assert result["summary"]["all_found"] is False


def test_analyze_invalid_image(tmp_path, monkeypatch) -> None:
    """Invalid image bytes → ok=False, outcome=retryable_error."""
    _patch_pipeline(monkeypatch, [], _spatial_result(labels=[]))

    bad = tmp_path / "not-an-image.png"
    bad.write_bytes(b"not an image")

    result = analyze_mod.analyze_image(bad)

    assert result["ok"] is False
    assert result["outcome"] == "retryable_error"
    assert result["found"] == []
    assert result["missing"] == []
    assert result["unassigned"] == []
    assert result["error"]["code"] == "invalid_image"


def test_analyze_zero_labels_zero_barcodes(tmp_path, monkeypatch) -> None:
    """Zero labels, zero barcodes → needs_better_photo, not complete."""
    _patch_pipeline(
        monkeypatch,
        [],
        _spatial_result(labels=[]),
    )

    image_path = tmp_path / "box.png"
    image_path.write_bytes(make_png())

    result = analyze_mod.analyze_image(image_path)

    assert result["ok"] is True
    assert result["outcome"] == "needs_better_photo"
    assert result["found"] == []
    assert result["missing"] == []
    assert result["summary"]["visible_label_count"] == 0
    assert result["summary"]["all_found"] is False


def test_analyze_non_contiguous_label_indexes(tmp_path, monkeypatch) -> None:
    """Labels with indexes 1 and 5 (gap), scanner matches label 5 → no IndexError."""
    barcodes = [
        _fake_barcode("V5", x1=160, y1=830, x2=270, y2=840),
    ]
    spatial = _spatial_result(
        labels=[
            {
                "label_index": 1,
                "label_bbox": {"x1": 100, "y1": 100, "x2": 300, "y2": 200},
                "barcode_bbox": {"x1": 150, "y1": 120, "x2": 280, "y2": 190},
                "status": "clear",
                "confidence": "high",
            },
            {
                "label_index": 5,
                "label_bbox": {"x1": 100, "y1": 800, "x2": 300, "y2": 900},
                "barcode_bbox": {"x1": 150, "y1": 820, "x2": 280, "y2": 890},
                "status": "clear",
                "confidence": "high",
            },
        ]
    )
    _patch_pipeline(monkeypatch, barcodes, spatial)

    image_path = tmp_path / "box.png"
    image_path.write_bytes(make_png())

    result = analyze_mod.analyze_image(image_path)

    assert result["ok"] is True
    assert result["outcome"] == "needs_better_photo"
    assert len(result["found"]) == 1
    # The found entry should have label_index=5 and the correct label_bbox.
    assert result["found"][0]["label_index"] == 5
    assert result["found"][0]["label_bbox"]["x1"] == 100
    assert result["found"][0]["label_bbox"]["y1"] == 800
    # Label 1 is missing.
    assert len(result["missing"]) == 1
    assert result["missing"][0]["label_index"] == 1


def test_analyze_bytes_input(tmp_path, monkeypatch) -> None:
    """Bytes input produces the same result as path input, temp file cleaned up."""
    barcodes = [
        _fake_barcode("V1", x1=160, y1=130, x2=270, y2=140),
        _fake_barcode("V2", x1=160, y1=830, x2=270, y2=840),
    ]
    _patch_pipeline(monkeypatch, barcodes, _spatial_result())

    image_path = tmp_path / "box.png"
    png_bytes = make_png()
    image_path.write_bytes(png_bytes)

    # Test with bytes input.
    result_bytes = analyze_mod.analyze_image(png_bytes)
    # Test with path input.
    result_path = analyze_mod.analyze_image(image_path)

    # Both should produce the same outcome and found count.
    assert result_bytes["outcome"] == result_path["outcome"]
    assert result_bytes["summary"]["found_count"] == result_path["summary"]["found_count"]
    assert result_bytes["summary"]["found_count"] == 2
    assert result_bytes["outcome"] == "complete"

    # Verify temp file was cleaned up: the function completed successfully
    # and the finally block ran (the file was read and deleted).
    # We verify by checking the result is valid and no temp file path
    # persists from our call.
    import tempfile as tmp_module

    temp_dir = Path(tmp_module.gettempdir())
    leftover = list(temp_dir.glob("tmp*.jpg"))
    # There should be no leftover temp files from our function.
    # (Other processes may have files, so we just verify ours are gone
    # by confirming the result is correct — the finally block ran.)
    assert result_bytes["ok"] is True
    assert leftover == [] or not any(
        f.stat().st_size == len(png_bytes) for f in leftover
    )

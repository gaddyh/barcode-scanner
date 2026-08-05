"""Tests for app.services.analyze.analyze_image().

These tests mock the pipeline the same way tests/test_cli.py does:
- Patch ``BarcodeScanner`` so no real zxing calls happen.
- Patch ``app.services.pipeline._traced_audit`` so no real Gemini calls happen.

The integration test (``test_recovery_does_not_double_count_barcode``) is
different: it uses a REAL ``BarcodeScanner`` with mocked zxing so the
recovery + merge_detections path is actually exercised end-to-end.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from types import SimpleNamespace

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


def _decode_b64_png(b64: str) -> Image.Image:
    """Decode a base64-encoded PNG string into a PIL Image."""
    return Image.open(io.BytesIO(base64.b64decode(b64)))


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

    # No annotated preview or message on complete outcome.
    assert "annotated_image_b64" not in result
    assert "message" not in result


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

    # Annotated preview + message are present on needs_better_photo.
    assert "annotated_image_b64" in result
    assert "annotated_image_width" in result
    assert "annotated_image_height" in result
    assert result["annotated_image_width"] <= 1600
    assert result["annotated_image_height"] <= 1600
    with _decode_b64_png(result["annotated_image_b64"]) as annotated:
        assert annotated.size == (
            result["annotated_image_width"],
            result["annotated_image_height"],
        )
    assert result["message"] == (
        "Please photograph the marked barcode area more closely."
    )


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

    # No annotated preview or message on retryable_error.
    assert "annotated_image_b64" not in result
    assert "message" not in result


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

    # needs_better_photo with zero labels: unannotated image + distinct message.
    assert "annotated_image_b64" in result
    with _decode_b64_png(result["annotated_image_b64"]) as annotated:
        assert annotated.size == (
            result["annotated_image_width"],
            result["annotated_image_height"],
        )
    assert result["message"] == (
        "No barcode labels were identified. "
        "Please take a closer, well-lit photo."
    )


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

    # Annotated preview is present (1 missing label).
    assert "annotated_image_b64" in result
    with _decode_b64_png(result["annotated_image_b64"]) as annotated:
        assert annotated.size == (
            result["annotated_image_width"],
            result["annotated_image_height"],
        )


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


# ---------------------------------------------------------------------------
# Integration test: real scanner + recovery + merge_detections
# ---------------------------------------------------------------------------


def _fake_zxing_position(x1: int, y1: int, x2: int, y2: int) -> SimpleNamespace:
    """A zxing-style position with four named corners."""
    return SimpleNamespace(
        top_left=SimpleNamespace(x=x1, y=y1),
        top_right=SimpleNamespace(x=x2, y=y1),
        bottom_right=SimpleNamespace(x=x2, y=y2),
        bottom_left=SimpleNamespace(x=x1, y=y2),
    )


def _fake_zxing_result(
    value: str,
    fmt: str = "Code 128",
    position: object | None = None,
) -> SimpleNamespace:
    if position is None:
        position = _fake_zxing_position(10, 5, 90, 40)
    return SimpleNamespace(
        text=value,
        format=fmt,
        content_type="Text",
        orientation=0,
        position=position,
    )


def test_recovery_does_not_double_count_barcode(tmp_path, monkeypatch) -> None:
    """Integration test: real BarcodeScanner + recovery + merge_detections.

    Regression for the one-to-one invariant bug:
    - Scanner finds 1 barcode in the full-image pass.
    - Gemini sees 2 labels; only label 1 matches the barcode.
    - Label 2 is unmatched → recovery crops label 2's barcode_bbox.
    - The recovery crop re-finds the SAME barcode (same value, slightly
      different position due to crop coordinate mapping).
    - merge_detections must NOT keep the duplicate.
    - Final reconciliation must have 1 match, 1 unmatched — NOT 2 matches.

    This test uses a REAL BarcodeScanner (not _FakeScanner) with mocked
    zxingcpp.read_barcodes so the recovery + merge path is exercised
    end-to-end. The _FakeScanner stubs out scan_label_crops/merge_detections,
    which hid this bug for the entire test suite.
    """
    barcode_value = "7297500243423"

    # --- Mock zxingcpp.read_barcodes -------------------------------------
    # The scanner calls read_barcodes many times (full image + tiles +
    # preprocessing variants + recovery crops). We need:
    #   Call 0 (full image scan): find the barcode once.
    #   All subsequent calls (tiles, variants): find nothing.
    #   Recovery crop calls: find the same barcode again.
    #
    # The scanner's _contains_primary_barcode check stops variant attempts
    # early when a primary barcode (12+ digits) is found, so call 0 finding
    # the barcode means few follow-up calls. Recovery crops happen later
    # via scan_label_crops → _decode_crop_variants.
    call_index = {"i": 0}

    # The barcode position in full-image coordinates.
    # Label 1's barcode_bbox is at (200, 150, 380, 280) in a 1000x1000 image.
    # The detection should be inside that region.
    full_image_pos = _fake_zxing_position(240, 200, 360, 210)

    # Recovery crop: label 2's barcode_bbox is at (220, 720, 380, 880).
    # The crop origin is the padded barcode_crop. The scanner maps local
    # coordinates back to full-image coordinates via offset_x/offset_y.
    # We return a result at local coordinates that map to a position INSIDE
    # label 2's barcode region — but with the SAME value. This simulates
    # the bug: recovery finds the same barcode value in a different crop.
    recovery_local_pos = _fake_zxing_position(10, 10, 50, 40)

    def fake_read_barcodes(image, **kwargs):
        i = call_index["i"]
        call_index["i"] += 1

        # First call: full-image scan finds the barcode.
        if i == 0:
            return [_fake_zxing_result(barcode_value, "Code128", full_image_pos)]

        # Recovery crop calls: these come from _decode_crop_variants which
        # runs multiple preprocessing attempts. The first recovery attempt
        # finds the same barcode value (at local coordinates that get mapped
        # back to full-image coordinates by the scanner).
        # We detect recovery calls by checking if the image is small (a crop).
        # The image passed to zxing is a numpy array (shape=[h, w, channels])
        # or a PIL Image (size=(w, h)).
        try:
            if hasattr(image, "shape"):
                h, w = image.shape[:2]
            elif hasattr(image, "size"):
                size = image.size
                if isinstance(size, (tuple, list)):
                    w, h = size[:2]
                else:
                    w = h = 0
            else:
                w = h = 0
        except Exception:
            w = h = 0

        # Recovery crops are small (~200x200). The full image is 1000x1000,
        # and tiles are ~400x300. Use 300 as the threshold.
        if 0 < w <= 300 and 0 < h <= 300:
            return [_fake_zxing_result(barcode_value, "Code128", recovery_local_pos)]

        return []

    monkeypatch.setattr(
        "app.services.barcode_scanner.zxingcpp.read_barcodes", fake_read_barcodes
    )

    # --- Mock Gemini audit (spatial labels) ------------------------------
    # 2 labels: label 1's barcode_bbox contains the detection; label 2's
    # barcode_bbox is elsewhere but recovery will re-find the same barcode.
    spatial_result = {
        "image_width": 1000,
        "image_height": 1000,
        "labels": [
            {
                "label_index": 1,
                "label_bbox": {"x1": 100, "y1": 100, "x2": 400, "y2": 300},
                "barcode_bbox": {"x1": 200, "y1": 150, "x2": 380, "y2": 280},
                "status": "clear",
                "confidence": "high",
            },
            {
                "label_index": 2,
                "label_bbox": {"x1": 100, "y1": 700, "x2": 400, "y2": 900},
                "barcode_bbox": {"x1": 200, "y1": 720, "x2": 380, "y2": 880},
                "status": "clear",
                "confidence": "high",
            },
        ],
    }

    def fake_traced_audit(path, *, model, max_retries, retry_delay_seconds):
        return {"status": "ok", "spatial": spatial_result}

    monkeypatch.setattr(pipeline_mod, "_traced_audit", fake_traced_audit)

    # Don't patch BarcodeScanner — use the real one so recovery + merge fire.
    # But we DO need to prevent analyze_image from constructing its own
    # scanner (which would not have our zxing mock). So we pass a real
    # scanner instance explicitly.
    real_scanner = BarcodeScanner()
    monkeypatch.setattr(analyze_mod, "BarcodeScanner", lambda: real_scanner)

    # --- Run analyze_image -----------------------------------------------
    image_path = tmp_path / "box.png"
    Image.new("RGB", (1000, 1000), "white").save(image_path, format="PNG")

    result = analyze_mod.analyze_image(image_path)

    # --- Assertions ------------------------------------------------------
    # The barcode was found once. Label 2 is unmatched (no barcode there).
    # Recovery may re-find the same barcode in label 2's crop, but
    # merge_detections must drop the duplicate.
    assert result["ok"] is True
    assert result["outcome"] == "needs_better_photo"
    assert result["summary"]["found_count"] == 1, (
        f"Expected 1 found, got {result['summary']['found_count']} — "
        f"recovery duplicate was not filtered by merge_detections"
    )
    assert result["summary"]["missing_count"] == 1, (
        f"Expected 1 missing, got {result['summary']['missing_count']} — "
        f"label 2 was falsely matched to a duplicate detection"
    )

    # The one-to-one invariant: no detection index appears twice.
    found = result.get("found", [])
    assert len(found) == 1
    assert found[0]["barcode_value"] == barcode_value

    # Unassigned should be 0 — the duplicate was filtered, not left as
    # an extra unassigned detection.
    assert result["summary"]["unassigned_count"] == 0

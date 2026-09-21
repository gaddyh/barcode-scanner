"""Tests for src/ingest/vision.py — pure functions, validators, and mocked Gemini paths.

No real Gemini API calls are made. All genai.Client interactions are mocked.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from src.ingest.vision import (
    AuditConfidence,
    BoxAuditCounts,
    ImageQuality,
    NormalizedBoundingBox,
    ShoeboxAuditError,
    ShoeboxImageAudit,
    SpatialLabelAudit,
    SpatialLabelAuditPixels,
    SpatialLabelObservation,
    SpatialLabelStatus,
    _convert_spatial_audit_to_pixels,
    _detect_mime_type,
    _gemini_compatible_schema,
    _load_image,
    audit_shoebox_counts,
    audit_shoebox_image,
    audit_shoebox_labels,
    audit_shoebox_labels_async,
    load_normalized_image,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _png_path(tmp_path: Path, name: str = "img.png", w: int = 800, h: int = 600) -> Path:
    p = tmp_path / name
    Image.new("RGB", (w, h), (255, 255, 255)).save(p, format="PNG")
    return p


def _jpeg_path(tmp_path: Path, name: str = "img.jpg", w: int = 800, h: int = 600) -> Path:
    p = tmp_path / name
    Image.new("RGB", (w, h), (255, 255, 255)).save(p, format="JPEG", quality=85)
    return p


def _normalized_bbox(
    *, top: int = 100, left: int = 100, bottom: int = 500, right: int = 500
) -> NormalizedBoundingBox:
    return NormalizedBoundingBox(top=top, left=left, bottom=bottom, right=right)


def _spatial_audit(
    *, labels: list[SpatialLabelObservation] | None = None,
) -> SpatialLabelAudit:
    if labels is None:
        labels = [
            SpatialLabelObservation(
                label_index=1,
                label_bbox=_normalized_bbox(),
                barcode_bbox=_normalized_bbox(top=200, left=200, bottom=400, right=400),
                status=SpatialLabelStatus.CLEAR,
                confidence=AuditConfidence.HIGH,
            )
        ]
    return SpatialLabelAudit(labels=labels)


def _mock_response(*, parsed=None, text: str | None = None):
    """Mock a Gemini generate_content response."""
    resp = MagicMock()
    resp.parsed = parsed
    resp.text = text
    return resp


def _mock_client(responses):
    """Build a mock genai.Client whose generate_content yields responses in order."""
    client = MagicMock()
    # responses can be a list or a single item
    if not isinstance(responses, list):
        responses = [responses]
    client.models.generate_content = MagicMock(side_effect=responses)
    client.aio.models.generate_content = AsyncMock(side_effect=responses)
    return client


# ===========================================================================
# NormalizedBoundingBox validator
# ===========================================================================


class TestNormalizedBoundingBox:
    def test_valid(self):
        bb = NormalizedBoundingBox(top=10, left=20, bottom=30, right=40)
        assert bb.top == 10 and bb.bottom == 30

    def test_bottom_le_top_raises(self):
        with pytest.raises(ValueError, match="bottom must be greater than top"):
            NormalizedBoundingBox(top=50, left=10, bottom=50, right=40)

    def test_bottom_lt_top_raises(self):
        with pytest.raises(ValueError, match="bottom must be greater than top"):
            NormalizedBoundingBox(top=50, left=10, bottom=40, right=60)

    def test_right_le_left_raises(self):
        with pytest.raises(ValueError, match="right must be greater than left"):
            NormalizedBoundingBox(top=10, left=50, bottom=30, right=50)

    def test_right_lt_left_raises(self):
        with pytest.raises(ValueError, match="right must be greater than left"):
            NormalizedBoundingBox(top=10, left=50, bottom=30, right=40)


# ===========================================================================
# ShoeboxImageAudit validator
# ===========================================================================


class TestShoeboxImageAuditValidator:
    def _base_kwargs(self, **overrides):
        kwargs = dict(
            physical_box_count=3,
            visible_product_barcode_label_count=2,
            clear_product_barcode_label_count=1,
            boxes_without_visible_product_barcode=1,
            partially_obscured_product_barcode_count=0,
            blurred_product_barcode_count=0,
            cropped_product_barcode_count=0,
            estimated_unique_product_label_groups=2,
            possible_non_product_barcode_count=0,
            duplicate_view_risk=False,
            overlapping_boxes_risk=False,
            image_quality=ImageQuality.GOOD,
            overall_confidence=AuditConfidence.HIGH,
            suitable_for_automatic_draft=True,
            observations=[],
            visible_label_text=[],
            warnings=[],
            notes=[],
        )
        kwargs.update(overrides)
        return kwargs

    def test_valid(self):
        audit = ShoeboxImageAudit(**self._base_kwargs())
        assert audit.physical_box_count == 3

    def test_visible_exceeds_physical_raises(self):
        with pytest.raises(ValueError, match="visible_product_barcode_label_count cannot exceed"):
            ShoeboxImageAudit(**self._base_kwargs(
                physical_box_count=1,
                visible_product_barcode_label_count=2,
                clear_product_barcode_label_count=1,
                boxes_without_visible_product_barcode=0,
            ))

    def test_clear_exceeds_visible_raises(self):
        with pytest.raises(ValueError, match="clear_product_barcode_label_count cannot exceed"):
            ShoeboxImageAudit(**self._base_kwargs(
                physical_box_count=3,
                visible_product_barcode_label_count=1,
                clear_product_barcode_label_count=2,
            ))

    def test_boxes_without_exceeds_physical_raises(self):
        with pytest.raises(ValueError, match="boxes_without_visible_product_barcode cannot exceed"):
            ShoeboxImageAudit(**self._base_kwargs(
                physical_box_count=1,
                visible_product_barcode_label_count=1,
                clear_product_barcode_label_count=0,
                boxes_without_visible_product_barcode=2,
            ))


# ===========================================================================
# _gemini_compatible_schema
# ===========================================================================


class TestGeminiCompatibleSchema:
    def test_strips_additional_properties(self):
        schema = _gemini_compatible_schema(ShoeboxImageAudit)
        assert "additionalProperties" not in schema
        # Recursively stripped from nested dicts
        s = str(schema)
        assert "additionalProperties" not in s

    def test_strips_from_list_items(self):
        schema = _gemini_compatible_schema(SpatialLabelAudit)
        assert "additionalProperties" not in str(schema)

    def test_preserves_required_fields(self):
        schema = _gemini_compatible_schema(BoxAuditCounts)
        assert "properties" in schema
        assert "visible_product_barcode_label_count" in schema.get("properties", {})


# ===========================================================================
# _detect_mime_type
# ===========================================================================


class TestDetectMimeType:
    def test_png_by_extension(self, tmp_path):
        p = _png_path(tmp_path, "x.png")
        assert _detect_mime_type(p) == "image/png"

    def test_jpeg_by_extension(self, tmp_path):
        p = _jpeg_path(tmp_path, "x.jpg")
        assert _detect_mime_type(p) == "image/jpeg"

    def test_jpeg_by_signature(self, tmp_path):
        p = tmp_path / "unknown.dat"
        p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 32)
        assert _detect_mime_type(p) == "image/jpeg"

    def test_png_by_signature(self, tmp_path):
        p = tmp_path / "unknown.dat"
        p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
        assert _detect_mime_type(p) == "image/png"

    def test_webp_by_signature(self, tmp_path):
        p = tmp_path / "unknown.dat"
        p.write_bytes(b"RIFF" + b"\x00" * 4 + b"WEBP" + b"\x00" * 16)
        assert _detect_mime_type(p) == "image/webp"

    def test_unrecognized_raises(self, tmp_path):
        p = tmp_path / "unknown.dat"
        p.write_bytes(b"\x00" * 32)
        with pytest.raises(ValueError, match="Unsupported or unrecognized"):
            _detect_mime_type(p)

    def test_unsupported_extension_raises(self, tmp_path):
        p = tmp_path / "x.gif"
        p.write_bytes(b"GIF89a" + b"\x00" * 32)
        with pytest.raises(ValueError, match="Unsupported or unrecognized"):
            _detect_mime_type(p)


# ===========================================================================
# load_normalized_image
# ===========================================================================


class TestLoadNormalizedImage:
    def test_loads_png(self, tmp_path):
        p = _png_path(tmp_path, w=400, h=300)
        img = load_normalized_image(p)
        assert img.mime_type == "image/jpeg"
        assert img.width <= 400
        assert img.height <= 300
        assert img.original_width == 400
        assert img.original_height == 300
        assert len(img.data) > 0

    def test_loads_jpeg(self, tmp_path):
        p = _jpeg_path(tmp_path, w=400, h=300)
        img = load_normalized_image(p)
        assert img.mime_type == "image/jpeg"
        assert img.original_width == 400

    def test_resizes_large_image(self, tmp_path):
        p = _png_path(tmp_path, w=3200, h=2400)
        img = load_normalized_image(p, max_dimension=1600)
        assert img.width <= 1600
        assert img.height <= 1600
        assert img.original_width == 3200
        assert img.original_height == 2400

    def test_no_resize_small_image(self, tmp_path):
        p = _png_path(tmp_path, w=200, h=100)
        img = load_normalized_image(p, max_dimension=1600)
        assert img.width == 200
        assert img.height == 100

    def test_invalid_max_dimension_zero(self, tmp_path):
        p = _png_path(tmp_path)
        with pytest.raises(ValueError, match="max_dimension must be positive"):
            load_normalized_image(p, max_dimension=0)

    def test_invalid_max_dimension_negative(self, tmp_path):
        p = _png_path(tmp_path)
        with pytest.raises(ValueError, match="max_dimension must be positive"):
            load_normalized_image(p, max_dimension=-1)

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Image does not exist"):
            load_normalized_image(tmp_path / "missing.png")

    def test_empty_file_raises(self, tmp_path):
        p = tmp_path / "empty.png"
        p.write_bytes(b"")
        with pytest.raises(ValueError, match="Image file is empty"):
            load_normalized_image(p)

    def test_directory_raises(self, tmp_path):
        with pytest.raises(ValueError, match="not a regular file"):
            load_normalized_image(tmp_path)

    def test_rgba_converted_to_rgb(self, tmp_path):
        p = tmp_path / "rgba.png"
        Image.new("RGBA", (100, 100), (255, 0, 0, 128)).save(p, format="PNG")
        img = load_normalized_image(p)
        assert img.mime_type == "image/jpeg"
        assert img.width == 100

    def test_grayscale_converted_to_rgb(self, tmp_path):
        p = tmp_path / "gray.png"
        Image.new("L", (100, 100), 128).save(p, format="PNG")
        img = load_normalized_image(p)
        assert img.mime_type == "image/jpeg"


# ===========================================================================
# _load_image
# ===========================================================================


class TestLoadImage:
    def test_returns_path_bytes_mime(self, tmp_path):
        p = _png_path(tmp_path)
        path, data, mime = _load_image(p)
        assert path == p.resolve()
        assert len(data) > 0
        assert mime == "image/jpeg"


# ===========================================================================
# _convert_spatial_audit_to_pixels
# ===========================================================================


class TestConvertSpatialAuditToPixels:
    def test_converts_with_barcode(self):
        audit = _spatial_audit()
        result = _convert_spatial_audit_to_pixels(
            audit, original_width=800, original_height=600
        )
        assert isinstance(result, SpatialLabelAuditPixels)
        assert result.image_width == 800
        assert result.image_height == 600
        assert len(result.labels) == 1
        label = result.labels[0]
        assert label.label_index == 1
        assert label.barcode_bbox is not None
        # Normalized 200..400 → pixels: 200*800/1000=160, 400*800/1000=320
        assert label.barcode_bbox.x1 == 160
        assert label.barcode_bbox.x2 == 320

    def test_converts_without_barcode(self):
        audit = SpatialLabelAudit(labels=[
            SpatialLabelObservation(
                label_index=2,
                label_bbox=_normalized_bbox(),
                barcode_bbox=None,
                status=SpatialLabelStatus.CROPPED,
                confidence=AuditConfidence.LOW,
            )
        ])
        result = _convert_spatial_audit_to_pixels(
            audit, original_width=1000, original_height=1000
        )
        assert len(result.labels) == 1
        assert result.labels[0].barcode_bbox is None
        assert result.labels[0].status == SpatialLabelStatus.CROPPED

    def test_clamps_oversized_bbox(self):
        audit = SpatialLabelAudit(labels=[
            SpatialLabelObservation(
                label_index=1,
                label_bbox=NormalizedBoundingBox(top=0, left=0, bottom=1000, right=1000),
                barcode_bbox=NormalizedBoundingBox(top=950, left=950, bottom=1000, right=1000),
                status=SpatialLabelStatus.CLEAR,
                confidence=AuditConfidence.HIGH,
            )
        ])
        result = _convert_spatial_audit_to_pixels(
            audit, original_width=200, original_height=200
        )
        label = result.labels[0]
        # 1000*200/1000 = 200, clamped to width-1=199
        assert label.label_bbox.x2 <= 200
        assert label.label_bbox.y2 <= 200

    def test_multiple_labels_preserve_order(self):
        labels = [
            SpatialLabelObservation(
                label_index=i,
                label_bbox=_normalized_bbox(
                    top=i * 10, left=i * 10,
                    bottom=i * 10 + 100, right=i * 10 + 100,
                ),
                barcode_bbox=None,
                status=SpatialLabelStatus.CLEAR,
                confidence=AuditConfidence.HIGH,
            )
            for i in range(1, 5)
        ]
        audit = SpatialLabelAudit(labels=labels)
        result = _convert_spatial_audit_to_pixels(
            audit, original_width=1000, original_height=1000
        )
        assert [lbl.label_index for lbl in result.labels] == [1, 2, 3, 4]


# ===========================================================================
# audit_shoebox_image — mocked Gemini
# ===========================================================================


class TestAuditShoeboxImage:
    def _valid_audit_dict(self):
        return {
            "physical_box_count": 2,
            "visible_product_barcode_label_count": 2,
            "clear_product_barcode_label_count": 2,
            "boxes_without_visible_product_barcode": 0,
            "partially_obscured_product_barcode_count": 0,
            "blurred_product_barcode_count": 0,
            "cropped_product_barcode_count": 0,
            "estimated_unique_product_label_groups": 1,
            "possible_non_product_barcode_count": 0,
            "duplicate_view_risk": False,
            "overlapping_boxes_risk": False,
            "image_quality": "good",
            "overall_confidence": "high",
            "suitable_for_automatic_draft": True,
            "observations": [],
            "visible_label_text": [],
            "warnings": [],
            "notes": [],
        }

    def test_negative_max_retries_raises(self, tmp_path):
        p = _png_path(tmp_path)
        with pytest.raises(ValueError, match="max_retries must be >= 0"):
            audit_shoebox_image(p, api_key="k", max_retries=-1)

    def test_negative_retry_delay_raises(self, tmp_path):
        p = _png_path(tmp_path)
        with pytest.raises(ValueError, match="retry_delay_seconds must be >= 0"):
            audit_shoebox_image(p, api_key="k", retry_delay_seconds=-1.0)

    def test_missing_api_key_raises(self, tmp_path, monkeypatch):
        p = _png_path(tmp_path)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setattr("src.ingest.vision.load_dotenv", lambda: None)
        with pytest.raises(ValueError, match="Missing Gemini API key"):
            audit_shoebox_image(p, api_key=None)

    def test_returns_parsed_object(self, tmp_path):
        p = _png_path(tmp_path)
        parsed = ShoeboxImageAudit(**self._valid_audit_dict())
        client = _mock_client(_mock_response(parsed=parsed))
        with patch("src.ingest.vision.genai.Client", return_value=client):
            result = audit_shoebox_image(p, api_key="k", max_retries=0)
        assert isinstance(result, ShoeboxImageAudit)
        assert result.physical_box_count == 2

    def test_returns_parsed_dict_validates(self, tmp_path):
        p = _png_path(tmp_path)
        client = _mock_client(_mock_response(parsed=self._valid_audit_dict()))
        with patch("src.ingest.vision.genai.Client", return_value=client):
            result = audit_shoebox_image(p, api_key="k", max_retries=0)
        assert isinstance(result, ShoeboxImageAudit)

    def test_returns_from_text(self, tmp_path):
        import json
        p = _png_path(tmp_path)
        client = _mock_client(_mock_response(
            parsed=None, text=json.dumps(self._valid_audit_dict())
        ))
        with patch("src.ingest.vision.genai.Client", return_value=client):
            result = audit_shoebox_image(p, api_key="k", max_retries=0)
        assert isinstance(result, ShoeboxImageAudit)

    def test_no_parsed_no_text_raises(self, tmp_path):
        p = _png_path(tmp_path)
        client = _mock_client(_mock_response(parsed=None, text=None))
        with patch("src.ingest.vision.genai.Client", return_value=client):
            with pytest.raises(ShoeboxAuditError, match="neither parsed output nor text"):
                audit_shoebox_image(p, api_key="k", max_retries=0)

    def test_retry_then_success(self, tmp_path):
        p = _png_path(tmp_path)
        parsed = ShoeboxImageAudit(**self._valid_audit_dict())
        err_resp = _mock_response(parsed=None, text=None)
        ok_resp = _mock_response(parsed=parsed)
        client = _mock_client([err_resp, ok_resp])
        with patch("src.ingest.vision.genai.Client", return_value=client):
            with patch("src.ingest.vision.time.sleep") as _:
                result = audit_shoebox_image(
                    p, api_key="k", max_retries=1, retry_delay_seconds=0.01
                )
        assert isinstance(result, ShoeboxImageAudit)

    def test_all_retries_fail_raises(self, tmp_path):
        p = _png_path(tmp_path)
        client = _mock_client(_mock_response(parsed=None, text=None))
        with patch("src.ingest.vision.genai.Client", return_value=client):
            with patch("src.ingest.vision.time.sleep") as _:
                with pytest.raises(ShoeboxAuditError, match="after 2 attempt"):
                    audit_shoebox_image(p, api_key="k", max_retries=1, retry_delay_seconds=0.01)

    def test_exception_during_call_retries(self, tmp_path):
        p = _png_path(tmp_path)
        parsed = ShoeboxImageAudit(**self._valid_audit_dict())
        client = MagicMock()
        client.models.generate_content = MagicMock(
            side_effect=[RuntimeError("net err"), _mock_response(parsed=parsed)]
        )
        with patch("src.ingest.vision.genai.Client", return_value=client):
            with patch("src.ingest.vision.time.sleep") as _:
                result = audit_shoebox_image(
                    p, api_key="k", max_retries=1, retry_delay_seconds=0.01
                )
        assert isinstance(result, ShoeboxImageAudit)


# ===========================================================================
# audit_shoebox_counts — mocked Gemini
# ===========================================================================


class TestAuditShoeboxCounts:
    def _valid_counts_dict(self):
        return {
            "visible_product_barcode_label_count": 2,
            "clear_product_barcode_label_count": 1,
            "boxes_without_visible_product_barcode": 1,
            "partially_obscured_product_barcode_count": 0,
        }

    def test_negative_max_retries_raises(self, tmp_path):
        p = _png_path(tmp_path)
        with pytest.raises(ValueError, match="max_retries must be >= 0"):
            audit_shoebox_counts(p, api_key="k", max_retries=-1)

    def test_missing_api_key_raises(self, tmp_path, monkeypatch):
        p = _png_path(tmp_path)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setattr("src.ingest.vision.load_dotenv", lambda: None)
        with pytest.raises(ValueError, match="Missing Gemini API key"):
            audit_shoebox_counts(p, api_key=None)

    def test_returns_parsed_object(self, tmp_path):
        p = _png_path(tmp_path)
        parsed = BoxAuditCounts(**self._valid_counts_dict())
        client = _mock_client(_mock_response(parsed=parsed))
        with patch("src.ingest.vision.genai.Client", return_value=client):
            result = audit_shoebox_counts(p, api_key="k", max_retries=0)
        assert isinstance(result, BoxAuditCounts)
        assert result.visible_product_barcode_label_count == 2

    def test_returns_from_text(self, tmp_path):
        import json
        p = _png_path(tmp_path)
        client = _mock_client(_mock_response(
            parsed=None, text=json.dumps(self._valid_counts_dict())
        ))
        with patch("src.ingest.vision.genai.Client", return_value=client):
            result = audit_shoebox_counts(p, api_key="k", max_retries=0)
        assert isinstance(result, BoxAuditCounts)

    def test_no_parsed_no_text_raises(self, tmp_path):
        p = _png_path(tmp_path)
        client = _mock_client(_mock_response(parsed=None, text=None))
        with patch("src.ingest.vision.genai.Client", return_value=client):
            with pytest.raises(ShoeboxAuditError, match="neither parsed output nor text"):
                audit_shoebox_counts(p, api_key="k", max_retries=0)

    def test_all_retries_fail_raises(self, tmp_path):
        p = _png_path(tmp_path)
        client = _mock_client(_mock_response(parsed=None, text=None))
        with patch("src.ingest.vision.genai.Client", return_value=client):
            with patch("src.ingest.vision.time.sleep") as _:
                with pytest.raises(ShoeboxAuditError, match="after 2 attempt"):
                    audit_shoebox_counts(p, api_key="k", max_retries=1, retry_delay_seconds=0.01)

    def test_lite_model_skips_thinking_config(self, tmp_path):
        p = _png_path(tmp_path)
        parsed = BoxAuditCounts(**self._valid_counts_dict())
        client = _mock_client(_mock_response(parsed=parsed))
        with patch("src.ingest.vision.genai.Client", return_value=client):
            result = audit_shoebox_counts(
                p, api_key="k", model="gemini-2.5-flash-lite", max_retries=0
            )
        assert isinstance(result, BoxAuditCounts)
        # Verify thinking_config was NOT set for lite model
        call_kwargs = client.models.generate_content.call_args.kwargs
        assert "thinking_config" not in call_kwargs.get(
            "config", MagicMock()
        ).__dict__.get("_kwargs", {})


# ===========================================================================
# audit_shoebox_labels — mocked Gemini
# ===========================================================================


class TestAuditShoeboxLabels:
    def test_negative_max_retries_raises(self, tmp_path):
        p = _png_path(tmp_path)
        with pytest.raises(ValueError, match="max_retries must be >= 0"):
            audit_shoebox_labels(p, api_key="k", max_retries=-1)

    def test_missing_api_key_raises(self, tmp_path, monkeypatch):
        p = _png_path(tmp_path)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setattr("src.ingest.vision.load_dotenv", lambda: None)
        with pytest.raises(ValueError, match="Missing Gemini API key"):
            audit_shoebox_labels(p, api_key=None)

    def test_returns_pixels(self, tmp_path):
        p = _png_path(tmp_path, w=800, h=600)
        audit = _spatial_audit()
        client = _mock_client(_mock_response(parsed=audit))
        with patch("src.ingest.vision.genai.Client", return_value=client):
            result = audit_shoebox_labels(p, api_key="k", max_retries=0)
        assert isinstance(result, SpatialLabelAuditPixels)
        assert result.image_width == 800
        assert result.image_height == 600
        assert len(result.labels) == 1

    def test_returns_from_text(self, tmp_path):
        p = _png_path(tmp_path, w=800, h=600)
        audit = _spatial_audit()
        client = _mock_client(_mock_response(parsed=None, text=audit.model_dump_json()))
        with patch("src.ingest.vision.genai.Client", return_value=client):
            result = audit_shoebox_labels(p, api_key="k", max_retries=0)
        assert isinstance(result, SpatialLabelAuditPixels)

    def test_no_parsed_no_text_raises(self, tmp_path):
        p = _png_path(tmp_path)
        client = _mock_client(_mock_response(parsed=None, text=None))
        with patch("src.ingest.vision.genai.Client", return_value=client):
            with pytest.raises(ShoeboxAuditError, match="neither parsed output nor text"):
                audit_shoebox_labels(p, api_key="k", max_retries=0)

    def test_all_retries_fail_raises(self, tmp_path):
        p = _png_path(tmp_path)
        client = _mock_client(_mock_response(parsed=None, text=None))
        with patch("src.ingest.vision.genai.Client", return_value=client):
            with patch("src.ingest.vision.time.sleep") as _:
                with pytest.raises(ShoeboxAuditError, match="after 2 attempt"):
                    audit_shoebox_labels(p, api_key="k", max_retries=1, retry_delay_seconds=0.01)

    def test_exception_retries_then_succeeds(self, tmp_path):
        p = _png_path(tmp_path, w=800, h=600)
        audit = _spatial_audit()
        client = MagicMock()
        client.models.generate_content = MagicMock(
            side_effect=[RuntimeError("net"), _mock_response(parsed=audit)]
        )
        with patch("src.ingest.vision.genai.Client", return_value=client):
            with patch("src.ingest.vision.time.sleep") as _:
                result = audit_shoebox_labels(
                    p, api_key="k", max_retries=1, retry_delay_seconds=0.01
                )
        assert isinstance(result, SpatialLabelAuditPixels)


# ===========================================================================
# audit_shoebox_labels_async — mocked Gemini
# ===========================================================================


class TestAuditShoeboxLabelsAsync:
    @pytest.mark.asyncio
    async def test_negative_max_retries_raises(self, tmp_path):
        p = _png_path(tmp_path)
        with pytest.raises(ValueError, match="max_retries must be >= 0"):
            await audit_shoebox_labels_async(p, api_key="k", max_retries=-1)

    @pytest.mark.asyncio
    async def test_missing_api_key_raises(self, tmp_path, monkeypatch):
        p = _png_path(tmp_path)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setattr("src.ingest.vision.load_dotenv", lambda: None)
        with pytest.raises(ValueError, match="Missing Gemini API key"):
            await audit_shoebox_labels_async(p, api_key=None)

    @pytest.mark.asyncio
    async def test_returns_pixels(self, tmp_path):
        p = _png_path(tmp_path, w=800, h=600)
        audit = _spatial_audit()
        client = _mock_client(_mock_response(parsed=audit))
        with patch("src.ingest.vision.genai.Client", return_value=client):
            result = await audit_shoebox_labels_async(p, api_key="k", max_retries=0)
        assert isinstance(result, SpatialLabelAuditPixels)
        assert result.image_width == 800

    @pytest.mark.asyncio
    async def test_returns_from_text(self, tmp_path):
        p = _png_path(tmp_path, w=800, h=600)
        audit = _spatial_audit()
        client = _mock_client(_mock_response(parsed=None, text=audit.model_dump_json()))
        with patch("src.ingest.vision.genai.Client", return_value=client):
            result = await audit_shoebox_labels_async(p, api_key="k", max_retries=0)
        assert isinstance(result, SpatialLabelAuditPixels)

    @pytest.mark.asyncio
    async def test_no_parsed_no_text_raises(self, tmp_path):
        p = _png_path(tmp_path)
        client = _mock_client(_mock_response(parsed=None, text=None))
        with patch("src.ingest.vision.genai.Client", return_value=client):
            with pytest.raises(ShoeboxAuditError, match="neither parsed output nor text"):
                await audit_shoebox_labels_async(p, api_key="k", max_retries=0)

    @pytest.mark.asyncio
    async def test_all_retries_fail_raises(self, tmp_path):
        p = _png_path(tmp_path)
        client = _mock_client(_mock_response(parsed=None, text=None))
        with patch("src.ingest.vision.genai.Client", return_value=client):
            with patch("asyncio.sleep") as _:
                with pytest.raises(ShoeboxAuditError, match="after 2 attempt"):
                    await audit_shoebox_labels_async(
                        p, api_key="k", max_retries=1, retry_delay_seconds=0.01
                    )

    @pytest.mark.asyncio
    async def test_exception_retries_then_succeeds(self, tmp_path):
        p = _png_path(tmp_path, w=800, h=600)
        audit = _spatial_audit()
        client = MagicMock()
        client.aio.models.generate_content = AsyncMock(
            side_effect=[RuntimeError("net"), _mock_response(parsed=audit)]
        )
        with patch("src.ingest.vision.genai.Client", return_value=client):
            with patch("asyncio.sleep") as _:
                result = await audit_shoebox_labels_async(
                    p, api_key="k", max_retries=1, retry_delay_seconds=0.01
                )
        assert isinstance(result, SpatialLabelAuditPixels)


# ===========================================================================
# main() — CLI entry point
# ===========================================================================


class TestVisionMain:
    def test_main_no_args_errors(self, monkeypatch):
        from src.ingest.vision import main
        monkeypatch.setattr("sys.argv", ["vision"])
        with pytest.raises(SystemExit):
            main()

    def test_main_missing_file_errors(self, tmp_path, monkeypatch):
        from src.ingest.vision import main
        monkeypatch.setattr(
            "sys.argv",
            ["vision", str(tmp_path / "missing.png")],
        )
        monkeypatch.setattr("src.ingest.vision.load_dotenv", lambda: None)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(SystemExit):
            main()

"""Tests for app.services.gemini_box_audit — schema, EXIF normalization, and
audit_shoebox_labels pixel conversion.

No test makes a real Gemini request. The genai.Client is mocked.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from app.services.gemini_box_audit import (
    AuditConfidence,
    NormalizedBoundingBox,
    NormalizedImage,
    ShoeboxAuditError,
    SpatialLabelAudit,
    SpatialLabelObservation,
    SpatialLabelStatus,
    load_normalized_image,
)
from app.services.spatial_geometry import PixelBoundingBox

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_png(width: int = 100, height: int = 50, color="white") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def _make_jpeg(width: int = 100, height: int = 50, color="white") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _write_image(tmp_path: Path, data: bytes, name: str = "img.jpg") -> Path:
    p = tmp_path / name
    p.write_bytes(data)
    return p


def _make_exif_rotated_image(
    tmp_path: Path,
    base_width: int = 200,
    base_height: int = 100,
    orientation: int = 6,
) -> Path:
    """Create a JPEG with EXIF orientation tag set.

    Orientation 6 means "rotate 90° clockwise for display". The stored pixel
    matrix is base_width x base_height, but after exif_transpose the displayed
    image is base_height x base_width.
    """
    img = Image.new("RGB", (base_width, base_height), "red")
    buf = io.BytesIO()
    # Save with EXIF orientation tag.
    exif = img.getexif()
    exif[0x0112] = orientation  # Orientation tag
    img.save(buf, format="JPEG", quality=95, exif=exif.tobytes())
    p = tmp_path / f"rotated_{orientation}.jpg"
    p.write_bytes(buf.getvalue())
    return p


# ---------------------------------------------------------------------------
# Schema validation — SpatialLabelAudit
# ---------------------------------------------------------------------------


class TestSpatialLabelAuditSchema:
    def test_valid_spatial_label_audit(self) -> None:
        audit = SpatialLabelAudit(
            labels=[
                SpatialLabelObservation(
                    label_index=1,
                    label_bbox=NormalizedBoundingBox(
                        top=100, left=200, bottom=500, right=800
                    ),
                    barcode_bbox=NormalizedBoundingBox(
                        top=150, left=250, bottom=480, right=780
                    ),
                    status=SpatialLabelStatus.CLEAR,
                    confidence=AuditConfidence.HIGH,
                ),
                SpatialLabelObservation(
                    label_index=2,
                    label_bbox=NormalizedBoundingBox(
                        top=600, left=200, bottom=900, right=800
                    ),
                    barcode_bbox=None,
                    status=SpatialLabelStatus.PARTIALLY_OBSCURED,
                    confidence=AuditConfidence.MEDIUM,
                ),
            ]
        )
        assert len(audit.labels) == 2
        assert audit.labels[0].label_index == 1
        assert audit.labels[1].barcode_bbox is None

    def test_invalid_coordinate_order_bottom_le_top(self) -> None:
        with pytest.raises(ValueError, match="bottom must be greater than top"):
            NormalizedBoundingBox(top=500, left=200, bottom=100, right=800)

    def test_invalid_coordinate_order_right_le_left(self) -> None:
        with pytest.raises(ValueError, match="right must be greater than left"):
            NormalizedBoundingBox(top=100, left=800, bottom=500, right=200)

    def test_coordinate_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            NormalizedBoundingBox(top=-1, left=200, bottom=500, right=800)
        with pytest.raises(ValueError):
            NormalizedBoundingBox(top=100, left=200, bottom=500, right=1001)

    def test_spatial_label_audit_rejects_extra_fields(self) -> None:
        with pytest.raises(ValueError):
            SpatialLabelAudit.model_validate(
                {
                    "labels": [],
                    "visible_count": 0,  # extra — counts are derived
                }
            )

    def test_spatial_label_observation_rejects_extra_fields(self) -> None:
        with pytest.raises(ValueError):
            SpatialLabelObservation.model_validate(
                {
                    "label_index": 1,
                    "label_bbox": {
                        "top": 100, "left": 200, "bottom": 500, "right": 800
                    },
                    "barcode_bbox": None,
                    "status": "clear",
                    "confidence": "high",
                    "extra_field": "bad",
                }
            )

    def test_label_index_must_be_ge_1(self) -> None:
        with pytest.raises(ValueError):
            SpatialLabelObservation.model_validate(
                {
                    "label_index": 0,
                    "label_bbox": {
                        "top": 100, "left": 200, "bottom": 500, "right": 800
                    },
                    "barcode_bbox": None,
                    "status": "clear",
                    "confidence": "high",
                }
            )

    def test_invalid_status_value(self) -> None:
        with pytest.raises(ValueError):
            SpatialLabelObservation.model_validate(
                {
                    "label_index": 1,
                    "label_bbox": {
                        "top": 100, "left": 200, "bottom": 500, "right": 800
                    },
                    "barcode_bbox": None,
                    "status": "totally_bogus",
                    "confidence": "high",
                }
            )


# ---------------------------------------------------------------------------
# load_normalized_image — EXIF normalization
# ---------------------------------------------------------------------------


class TestLoadNormalizedImage:
    def test_no_exif_returns_original_dimensions(self, tmp_path: Path) -> None:
        p = _write_image(tmp_path, _make_jpeg(200, 100), "plain.jpg")
        norm = load_normalized_image(p)
        assert norm.width == 200
        assert norm.height == 100
        assert norm.original_width == 200
        assert norm.original_height == 100
        assert norm.mime_type == "image/jpeg"
        # Data is valid JPEG.
        with Image.open(io.BytesIO(norm.data)) as img:
            assert img.size == (200, 100)
            assert img.mode == "RGB"

    def test_png_converted_to_rgb_jpeg(self, tmp_path: Path) -> None:
        p = _write_image(tmp_path, _make_png(200, 100), "plain.png")
        norm = load_normalized_image(p)
        assert norm.mime_type == "image/jpeg"
        assert norm.width == 200
        assert norm.height == 100
        # Data is JPEG, not PNG.
        assert norm.data[:2] == b"\xff\xd8"

    def test_orientation_6_transposes_dimensions(self, tmp_path: Path) -> None:
        """Orientation 6 = rotate 90° CW. Stored 200x100 → displayed 100x200."""
        p = _make_exif_rotated_image(
            tmp_path, base_width=200, base_height=100, orientation=6
        )
        norm = load_normalized_image(p)
        # After exif_transpose, dimensions swap.
        assert norm.width == 100
        assert norm.height == 200
        assert norm.original_width == 100
        assert norm.original_height == 200
        with Image.open(io.BytesIO(norm.data)) as img:
            assert img.size == (100, 200)

    def test_orientation_1_keeps_dimensions(self, tmp_path: Path) -> None:
        """Orientation 1 = normal (no rotation)."""
        p = _make_exif_rotated_image(
            tmp_path, base_width=200, base_height=100, orientation=1
        )
        norm = load_normalized_image(p)
        assert norm.width == 200
        assert norm.height == 100

    def test_nonexistent_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_normalized_image(tmp_path / "nope.jpg")

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.jpg"
        p.write_bytes(b"")
        with pytest.raises(ValueError, match="empty"):
            load_normalized_image(p)

    def test_invalid_max_dimension_raises(self, tmp_path: Path) -> None:
        p = _write_image(tmp_path, _make_jpeg(200, 100), "plain.jpg")
        with pytest.raises(ValueError, match="max_dimension"):
            load_normalized_image(p, max_dimension=0)

    def test_large_image_is_resized(self, tmp_path: Path) -> None:
        """Image larger than max_dimension is downscaled; original dims preserved."""
        p = _write_image(tmp_path, _make_jpeg(3200, 2400), "big.jpg")
        norm = load_normalized_image(p, max_dimension=1600)
        # Resized to fit within 1600x1600, aspect ratio preserved.
        assert norm.width == 1600
        assert norm.height == 1200
        # Original dimensions recorded for coordinate scaling.
        assert norm.original_width == 3200
        assert norm.original_height == 2400
        with Image.open(io.BytesIO(norm.data)) as img:
            assert img.size == (1600, 1200)

    def test_small_image_not_enlarged(self, tmp_path: Path) -> None:
        """Image smaller than max_dimension is not resized."""
        p = _write_image(tmp_path, _make_jpeg(800, 600), "small.jpg")
        norm = load_normalized_image(p, max_dimension=1600)
        assert norm.width == 800
        assert norm.height == 600
        assert norm.original_width == 800
        assert norm.original_height == 600

    def test_normalized_image_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        norm = NormalizedImage(
            data=b"x",
            mime_type="image/jpeg",
            width=10,
            height=10,
            original_width=10,
            original_height=10,
        )
        with pytest.raises(FrozenInstanceError):
            norm.width = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# audit_shoebox_labels — pixel conversion with mocked Gemini client
# ---------------------------------------------------------------------------


class TestAuditShoeboxLabels:
    def _mock_response(self, audit_dict: dict) -> MagicMock:
        """Build a mock Gemini response carrying parsed structured output."""
        response = MagicMock()
        response.parsed = audit_dict
        response.text = json.dumps(audit_dict)
        return response

    def _patch_client(self, response: MagicMock) -> str:
        """Patch genai.Client to return ``response`` from generate_content.

        Returns the dotted patch path for use with patch().
        """
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = response
        return mock_client

    def test_returns_pixel_coordinates(self, tmp_path: Path) -> None:
        """Gemini returns normalized 0..1000 boxes; the function returns pixel
        boxes converted and clamped to the image frame."""
        p = _write_image(tmp_path, _make_jpeg(2000, 1000), "box.jpg")

        audit_dict = {
            "labels": [
                {
                    "label_index": 1,
                    "label_bbox": {
                        "top": 100, "left": 250, "bottom": 500, "right": 750
                    },
                    "barcode_bbox": {
                        "top": 150, "left": 300, "bottom": 480, "right": 720
                    },
                    "status": "clear",
                    "confidence": "high",
                }
            ]
        }
        response = self._mock_response(audit_dict)
        mock_client = self._patch_client(response)

        with patch("app.services.gemini_box_audit.genai.Client", return_value=mock_client):
            from app.services.gemini_box_audit import audit_shoebox_labels

            result = audit_shoebox_labels(p, api_key="fake-key")

        assert result.image_width == 2000
        assert result.image_height == 1000
        assert len(result.labels) == 1
        label = result.labels[0]
        assert label.label_index == 1
        # normalized top=100,left=250,bottom=500,right=750 on 2000x1000
        # → x1=500, y1=100, x2=1500, y2=500
        assert label.label_bbox == PixelBoundingBox(
            x1=500, y1=100, x2=1500, y2=500
        )
        # barcode: top=150,left=300,bottom=480,right=720
        # → x1=600, y1=150, x2=1440, y2=480
        assert label.barcode_bbox == PixelBoundingBox(
            x1=600, y1=150, x2=1440, y2=480
        )
        assert label.status == SpatialLabelStatus.CLEAR
        assert label.confidence == AuditConfidence.HIGH

    def test_clamps_boxes_to_image_bounds(self, tmp_path: Path) -> None:
        """Gemini returns a box that would convert to pixels beyond the image
        frame; the result is clamped."""
        p = _write_image(tmp_path, _make_jpeg(500, 500), "small.jpg")

        audit_dict = {
            "labels": [
                {
                    "label_index": 1,
                    "label_bbox": {
                        "top": 950, "left": 950, "bottom": 1000, "right": 1000
                    },
                    "barcode_bbox": None,
                    "status": "uncertain",
                    "confidence": "low",
                }
            ]
        }
        response = self._mock_response(audit_dict)
        mock_client = self._patch_client(response)

        with patch("app.services.gemini_box_audit.genai.Client", return_value=mock_client):
            from app.services.gemini_box_audit import audit_shoebox_labels

            result = audit_shoebox_labels(p, api_key="fake-key")

        label = result.labels[0]
        # 950 * 500 / 1000 = 475, 1000 * 500 / 1000 = 500 → clamped to 500.
        assert label.label_bbox == PixelBoundingBox(
            x1=475, y1=475, x2=500, y2=500
        )
        assert label.barcode_bbox is None

    def test_derived_counts(self, tmp_path: Path) -> None:
        p = _write_image(tmp_path, _make_jpeg(1000, 1000), "box.jpg")
        audit_dict = {
            "labels": [
                {
                    "label_index": 1,
                    "label_bbox": {
                        "top": 100, "left": 100, "bottom": 300, "right": 300
                    },
                    "barcode_bbox": None,
                    "status": "clear",
                    "confidence": "high",
                },
                {
                    "label_index": 2,
                    "label_bbox": {
                        "top": 400, "left": 100, "bottom": 600, "right": 300
                    },
                    "barcode_bbox": None,
                    "status": "partially_obscured",
                    "confidence": "medium",
                },
                {
                    "label_index": 3,
                    "label_bbox": {
                        "top": 700, "left": 100, "bottom": 900, "right": 300
                    },
                    "barcode_bbox": None,
                    "status": "clear",
                    "confidence": "high",
                },
            ]
        }
        response = self._mock_response(audit_dict)
        mock_client = self._patch_client(response)

        with patch("app.services.gemini_box_audit.genai.Client", return_value=mock_client):
            from app.services.gemini_box_audit import audit_shoebox_labels

            result = audit_shoebox_labels(p, api_key="fake-key")

        assert result.visible_count == 3
        assert result.clear_count == 2  # labels 1 and 3

    def test_missing_api_key_raises(self, tmp_path: Path) -> None:
        p = _write_image(tmp_path, _make_jpeg(100, 100), "box.jpg")
        # Clear env AND neutralize load_dotenv so the repo .env doesn't
        # re-populate GEMINI_API_KEY during the call.
        with patch.dict("os.environ", {}, clear=True), \
             patch("app.services.gemini_box_audit.load_dotenv", lambda: None):
            from app.services.gemini_box_audit import audit_shoebox_labels

            with pytest.raises(ValueError, match="Missing Gemini API key"):
                audit_shoebox_labels(p, api_key=None)

    def test_gemini_failure_raises_shoebox_audit_error(self, tmp_path: Path) -> None:
        p = _write_image(tmp_path, _make_jpeg(100, 100), "box.jpg")
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("boom")

        with patch("app.services.gemini_box_audit.genai.Client", return_value=mock_client):
            from app.services.gemini_box_audit import audit_shoebox_labels

            with pytest.raises(ShoeboxAuditError, match="spatial label audit failed"):
                audit_shoebox_labels(
                    p, api_key="fake-key", max_retries=0, retry_delay_seconds=0
                )

    def test_nonexistent_file_raises(self, tmp_path: Path) -> None:
        from app.services.gemini_box_audit import audit_shoebox_labels

        with pytest.raises(FileNotFoundError):
            audit_shoebox_labels(
                tmp_path / "nope.jpg", api_key="fake-key"
            )

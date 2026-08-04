"""Tests for app.services.spatial_geometry — pure coordinate mathematics."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from app.services.spatial_geometry import (
    PixelBoundingBox,
    bbox_center,
    clamp_bbox,
    normalized_center_distance,
    normalized_to_pixels,
    padded_bbox,
    point_inside_bbox,
)

# ---------------------------------------------------------------------------
# PixelBoundingBox validation
# ---------------------------------------------------------------------------


def test_pixel_bbox_allows_thin_box() -> None:
    """x1 == x2 is permitted (a thin barcode detection can be degenerate)."""
    box = PixelBoundingBox(x1=100, y1=50, x2=100, y2=200)
    assert box.width == 0
    assert box.height == 150


def test_pixel_bbox_allows_flat_box() -> None:
    box = PixelBoundingBox(x1=100, y1=50, x2=300, y2=50)
    assert box.width == 200
    assert box.height == 0


def test_pixel_bbox_rejects_x2_less_than_x1() -> None:
    with pytest.raises(ValidationError, match="x2 must be greater than or equal to x1"):
        PixelBoundingBox(x1=300, y1=50, x2=100, y2=200)


def test_pixel_bbox_rejects_y2_less_than_y1() -> None:
    with pytest.raises(ValidationError, match="y2 must be greater than or equal to y1"):
        PixelBoundingBox(x1=100, y1=200, x2=300, y2=50)


def test_pixel_bbox_allows_negative_coordinates() -> None:
    """Negative coordinates are permitted (padded_bbox can produce them
    before clamping). Use clamp_bbox to enforce image-frame bounds."""
    box = PixelBoundingBox(x1=-1, y1=-5, x2=10, y2=10)
    assert box.x1 == -1
    assert box.y1 == -5


def test_pixel_bbox_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        PixelBoundingBox(x1=0, y1=0, x2=10, y2=10, extra=5)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# normalized_to_pixels
# ---------------------------------------------------------------------------


def test_normalized_to_pixels_basic_conversion() -> None:
    """
    normalized top=100, left=250, bottom=500, right=750
    image 2000 x 1000
    -> x1=500, y1=100, x2=1500, y2=500
    """
    result = normalized_to_pixels(
        top=100,
        left=250,
        bottom=500,
        right=750,
        image_width=2000,
        image_height=1000,
    )
    assert result == PixelBoundingBox(x1=500, y1=100, x2=1500, y2=500)


def test_normalized_to_pixels_zero_origin() -> None:
    result = normalized_to_pixels(
        top=0,
        left=0,
        bottom=1000,
        right=1000,
        image_width=4032,
        image_height=3024,
    )
    assert result == PixelBoundingBox(x1=0, y1=0, x2=4032, y2=3024)


def test_normalized_to_pixels_rounding() -> None:
    """round() is used, so 0.5 rounds to nearest even (banker's rounding)."""
    result = normalized_to_pixels(
        top=1,
        left=1,
        bottom=2,
        right=2,
        image_width=1000,
        image_height=1000,
    )
    # 1 * 1000 / 1000 = 1.0, 2 * 1000 / 1000 = 2.0
    assert result == PixelBoundingBox(x1=1, y1=1, x2=2, y2=2)


def test_normalized_to_pixels_rejects_zero_width() -> None:
    with pytest.raises(ValueError, match="image_width must be positive"):
        normalized_to_pixels(
            top=0, left=0, bottom=100, right=100,
            image_width=0, image_height=1000,
        )


def test_normalized_to_pixels_rejects_zero_height() -> None:
    with pytest.raises(ValueError, match="image_height must be positive"):
        normalized_to_pixels(
            top=0, left=0, bottom=100, right=100,
            image_width=1000, image_height=0,
        )


# ---------------------------------------------------------------------------
# bbox_center
# ---------------------------------------------------------------------------


def test_bbox_center_basic() -> None:
    box = PixelBoundingBox(x1=100, y1=50, x2=300, y2=250)
    assert bbox_center(box) == (200.0, 150.0)


def test_bbox_center_thin_box() -> None:
    box = PixelBoundingBox(x1=100, y1=50, x2=100, y2=250)
    assert bbox_center(box) == (100.0, 150.0)


# ---------------------------------------------------------------------------
# point_inside_bbox
# ---------------------------------------------------------------------------


def test_point_inside_center() -> None:
    box = PixelBoundingBox(x1=100, y1=50, x2=300, y2=250)
    assert point_inside_bbox((200, 150), box) is True


def test_point_inside_on_edge() -> None:
    box = PixelBoundingBox(x1=100, y1=50, x2=300, y2=250)
    assert point_inside_bbox((100, 50), box) is True
    assert point_inside_bbox((300, 250), box) is True


def test_point_outside() -> None:
    box = PixelBoundingBox(x1=100, y1=50, x2=300, y2=250)
    assert point_inside_bbox((301, 150), box) is False
    assert point_inside_bbox((200, 251), box) is False


# ---------------------------------------------------------------------------
# padded_bbox
# ---------------------------------------------------------------------------


def test_padded_bbox_expands() -> None:
    box = PixelBoundingBox(x1=100, y1=50, x2=300, y2=250)
    result = padded_bbox(box, padding_x=20, padding_y=10)
    assert result == PixelBoundingBox(x1=80, y1=40, x2=320, y2=260)


def test_padded_bbox_can_go_negative() -> None:
    """padded_bbox does not clamp; use clamp_bbox for that."""
    box = PixelBoundingBox(x1=5, y1=5, x2=10, y2=10)
    result = padded_bbox(box, padding_x=20, padding_y=20)
    assert result.x1 == -15
    assert result.y1 == -15


# ---------------------------------------------------------------------------
# clamp_bbox
# ---------------------------------------------------------------------------


def test_clamp_bbox_within_bounds() -> None:
    box = PixelBoundingBox(x1=100, y1=50, x2=300, y2=250)
    result = clamp_bbox(box, image_width=1000, image_height=1000)
    assert result == box


def test_clamp_bbox_clamps_negative() -> None:
    box = PixelBoundingBox(x1=-50, y1=-30, x2=300, y2=250)
    result = clamp_bbox(box, image_width=1000, image_height=1000)
    assert result == PixelBoundingBox(x1=0, y1=0, x2=300, y2=250)


def test_clamp_bbox_clamps_overflow() -> None:
    box = PixelBoundingBox(x1=100, y1=50, x2=1200, y2=1100)
    result = clamp_bbox(box, image_width=1000, image_height=1000)
    assert result == PixelBoundingBox(x1=100, y1=50, x2=1000, y2=1000)


# ---------------------------------------------------------------------------
# normalized_center_distance
# ---------------------------------------------------------------------------


def test_normalized_center_distance_zero() -> None:
    d = normalized_center_distance((100, 100), (100, 100), 1000, 1000)
    assert d == 0.0


def test_normalized_center_distance_symmetric() -> None:
    d1 = normalized_center_distance((100, 100), (200, 300), 1000, 1000)
    d2 = normalized_center_distance((200, 300), (100, 100), 1000, 1000)
    assert d1 == d2


def test_normalized_center_distance_scale_correct() -> None:
    """
    On a 4000 x 1000 image, 100 px horizontally is 2.5% of width,
    100 px vertically is 10% of height. They should not contribute equally.
    """
    d_horizontal = normalized_center_distance((0, 0), (100, 0), 4000, 1000)
    d_vertical = normalized_center_distance((0, 0), (0, 100), 4000, 1000)

    # Vertical distance should be larger because 100/1000 > 100/4000.
    assert d_vertical > d_horizontal
    assert math.isclose(d_horizontal, 0.025)
    assert math.isclose(d_vertical, 0.10)


def test_normalized_center_distance_rejects_zero_dimensions() -> None:
    with pytest.raises(ValueError, match="image_width must be positive"):
        normalized_center_distance((0, 0), (10, 10), 0, 1000)
    with pytest.raises(ValueError, match="image_height must be positive"):
        normalized_center_distance((0, 0), (10, 10), 1000, 0)

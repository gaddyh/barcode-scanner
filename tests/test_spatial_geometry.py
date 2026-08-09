"""Tests for spatial_geometry coordinate math."""

from __future__ import annotations

import pytest

from src.ingest.geometry import (
    PixelBoundingBox,
    bbox_center,
    clamp_bbox,
    normalized_center_distance,
    normalized_to_pixels,
    padded_bbox,
    point_inside_bbox,
)


def test_pixel_bounding_box_width_height() -> None:
    box = PixelBoundingBox(x1=10, y1=20, x2=110, y2=220)
    assert box.width == 100
    assert box.height == 200


def test_pixel_bounding_box_rejects_inverted_x() -> None:
    with pytest.raises(ValueError):
        PixelBoundingBox(x1=100, y1=0, x2=10, y2=10)


def test_pixel_bounding_box_rejects_inverted_y() -> None:
    with pytest.raises(ValueError):
        PixelBoundingBox(x1=0, y1=100, x2=10, y2=10)


def test_pixel_bounding_box_allows_degenerate_axis() -> None:
    box = PixelBoundingBox(x1=10, y1=20, x2=10, y2=20)
    assert box.width == 0
    assert box.height == 0


def test_normalized_to_pixels_scales_correctly() -> None:
    box = normalized_to_pixels(
        top=0, left=0, bottom=1000, right=1000,
        image_width=2000, image_height=1000,
    )
    assert box.x1 == 0
    assert box.y1 == 0
    assert box.x2 == 2000
    assert box.y2 == 1000


def test_normalized_to_pixels_partial() -> None:
    box = normalized_to_pixels(
        top=250, left=500, bottom=750, right=500,
        image_width=1000, image_height=1000,
    )
    assert box.x1 == 500
    assert box.y1 == 250
    assert box.x2 == 500
    assert box.y2 == 750


def test_normalized_to_pixels_rejects_zero_dimensions() -> None:
    with pytest.raises(ValueError):
        normalized_to_pixels(
            top=0, left=0, bottom=10, right=10,
            image_width=0, image_height=100,
        )


def test_bbox_center() -> None:
    center = bbox_center(PixelBoundingBox(x1=0, y1=0, x2=100, y2=200))
    assert center == (50.0, 100.0)


def test_point_inside_bbox_inside() -> None:
    box = PixelBoundingBox(x1=10, y1=10, x2=100, y2=100)
    assert point_inside_bbox((50, 50), box) is True


def test_point_inside_bbox_on_boundary() -> None:
    box = PixelBoundingBox(x1=10, y1=10, x2=100, y2=100)
    assert point_inside_bbox((10, 10), box) is True
    assert point_inside_bbox((100, 100), box) is True


def test_point_inside_bbox_outside() -> None:
    box = PixelBoundingBox(x1=10, y1=10, x2=100, y2=100)
    assert point_inside_bbox((5, 50), box) is False
    assert point_inside_bbox((150, 50), box) is False


def test_padded_bbox_expands() -> None:
    box = PixelBoundingBox(x1=100, y1=100, x2=200, y2=200)
    padded = padded_bbox(box, padding_x=10, padding_y=20)
    assert padded.x1 == 90
    assert padded.y1 == 80
    assert padded.x2 == 210
    assert padded.y2 == 220


def test_clamp_bbox() -> None:
    box = PixelBoundingBox(x1=-50, y1=-50, x2=2000, y2=2000)
    clamped = clamp_bbox(box, image_width=1000, image_height=500)
    assert clamped.x1 == 0
    assert clamped.y1 == 0
    assert clamped.x2 == 1000
    assert clamped.y2 == 500


def test_normalized_center_distance() -> None:
    dist = normalized_center_distance(
        (0, 0), (500, 0), image_width=1000, image_height=1000,
    )
    assert dist == pytest.approx(0.5)


def test_normalized_center_distance_non_square_image() -> None:
    # 100px horizontally on 1000px-wide image = 0.1
    # 100px vertically on 4000px-tall image = 0.025
    dist = normalized_center_distance(
        (0, 0), (100, 100), image_width=1000, image_height=4000,
    )
    assert dist == pytest.approx((0.1**2 + 0.025**2) ** 0.5)

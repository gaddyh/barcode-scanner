"""
spatial_geometry.py

Generic coordinate mathematics for the spatial reconciliation pipeline.

This module knows nothing about Gemini, ZXing, barcodes, shoeboxes, or labels.
It operates only on plain numbers and ``PixelBoundingBox`` values.

Dependency direction::

    gemini_box_audit.py ──→ spatial_geometry.py ←── spatial_reconciliation.py

``spatial_geometry`` imports neither of the two domain modules above.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, model_validator


class PixelBoundingBox(BaseModel):
    """
    Bounding box in absolute pixel coordinates.

    ``x1 == x2`` or ``y1 == y2`` is permitted for defensive parsing (a thin
    barcode detection can be degenerate in one axis), but ``x2 < x1`` and
    ``y2 < y1`` are rejected.
    """

    model_config = ConfigDict(extra="forbid")

    x1: int
    y1: int
    x2: int
    y2: int

    @model_validator(mode="after")
    def validate_order(self) -> PixelBoundingBox:
        if self.x2 < self.x1:
            raise ValueError("x2 must be greater than or equal to x1")
        if self.y2 < self.y1:
            raise ValueError("y2 must be greater than or equal to y1")
        return self

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


def normalized_to_pixels(
    *,
    top: int,
    left: int,
    bottom: int,
    right: int,
    image_width: int,
    image_height: int,
) -> PixelBoundingBox:
    """
    Convert a Gemini-native normalized 0..1000 bounding box to pixel space.

    Gemini orders normalized coordinates as ``top, left, bottom, right``.
    The returned ``PixelBoundingBox`` uses ``x1, y1, x2, y2`` where
    ``x1 = left``, ``y1 = top``, ``x2 = right``, ``y2 = bottom``.

    Raises:
        ValueError: if ``image_width`` or ``image_height`` is not positive.
    """
    if image_width <= 0:
        raise ValueError("image_width must be positive")
    if image_height <= 0:
        raise ValueError("image_height must be positive")

    return PixelBoundingBox(
        x1=round(left * image_width / 1000),
        y1=round(top * image_height / 1000),
        x2=round(right * image_width / 1000),
        y2=round(bottom * image_height / 1000),
    )


def bbox_center(box: PixelBoundingBox) -> tuple[float, float]:
    """Return the (x, y) center of ``box`` as floats."""
    return ((box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0)


def point_inside_bbox(point: tuple[float, float], box: PixelBoundingBox) -> bool:
    """True when ``point`` lies inside (inclusive) ``box``."""
    px, py = point
    return box.x1 <= px <= box.x2 and box.y1 <= py <= box.y2


def padded_bbox(
    box: PixelBoundingBox,
    padding_x: int,
    padding_y: int,
) -> PixelBoundingBox:
    """
    Expand ``box`` by ``padding_x`` / ``padding_y`` on each side.

    Negative coordinates are allowed here; callers that need clamping to the
    image frame should pass the result through ``clamp_bbox``.
    """
    return PixelBoundingBox(
        x1=box.x1 - padding_x,
        y1=box.y1 - padding_y,
        x2=box.x2 + padding_x,
        y2=box.y2 + padding_y,
    )


def clamp_bbox(
    box: PixelBoundingBox,
    image_width: int,
    image_height: int,
) -> PixelBoundingBox:
    """
    Clamp ``box`` to the image frame ``[0, image_width] x [0, image_height]``.

    ``image_width`` / ``image_height`` are inclusive upper bounds (a pixel
    coordinate equal to the width/height is considered on-frame).
    """
    if image_width < 0 or image_height < 0:
        raise ValueError("image dimensions must be non-negative")

    return PixelBoundingBox(
        x1=min(max(box.x1, 0), image_width),
        y1=min(max(box.y1, 0), image_height),
        x2=min(max(box.x2, 0), image_width),
        y2=min(max(box.y2, 0), image_height),
    )


def normalized_center_distance(
    center_a: tuple[float, float],
    center_b: tuple[float, float],
    image_width: int,
    image_height: int,
) -> float:
    """
    Euclidean distance between two pixel centers, normalized by image size.

    Pixel distance alone behaves badly on non-square images: 100 px
    horizontally is a much larger fraction of a 1000 px-wide image than 100 px
    vertically is of a 4000 px-tall image. Dividing each axis by the image
    dimension yields a dimensionless, image-scale-independent score.

    Raises:
        ValueError: if ``image_width`` or ``image_height`` is not positive.
    """
    if image_width <= 0:
        raise ValueError("image_width must be positive")
    if image_height <= 0:
        raise ValueError("image_height must be positive")

    dx = (center_a[0] - center_b[0]) / image_width
    dy = (center_a[1] - center_b[1]) / image_height
    return math.hypot(dx, dy)

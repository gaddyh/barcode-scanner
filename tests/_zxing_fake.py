"""Shared test helpers for mocking zxingcpp read results.

``zxingcpp.read_barcodes`` returns objects with ``.text``, ``.position``,
``.format``, ``.content_type``, ``.orientation`` attributes — not
``DetectedBarcode`` dataclasses. The scanner's ``_map_position`` accesses
``position.top_left`` / ``.top_right`` / ``.bottom_right`` / ``.bottom_left``
(a zxingcpp ``Position``), each of which has ``.x`` / ``.y``. These helpers
build lightweight fakes that the scanner's result-mapping loop accepts.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FakePoint:
    x: int
    y: int


@dataclass(frozen=True)
class FakePosition:
    top_left: FakePoint
    top_right: FakePoint
    bottom_right: FakePoint
    bottom_left: FakePoint


@dataclass(frozen=True)
class FakeReadResult:
    """Mimics zxingcpp.ReadResult for scanner tests."""

    text: str
    position: FakePosition
    format: int = 0  # zxingcpp.BarcodeFormat.Code128
    content_type: str = "Text"
    orientation: int = 0


def make_read_result(
    value: str,
    *,
    x1: int = 100,
    y1: int = 100,
    x2: int = 200,
    y2: int = 300,
) -> FakeReadResult:
    """Build a FakeReadResult with a 4-corner position polygon."""
    return FakeReadResult(
        text=value,
        position=FakePosition(
            top_left=FakePoint(x=x1, y=y1),
            top_right=FakePoint(x=x2, y=y1),
            bottom_right=FakePoint(x=x2, y=y2),
            bottom_left=FakePoint(x=x1, y=y2),
        ),
    )

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from io import BytesIO
from statistics import median

import numpy as np
import zxingcpp
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


@dataclass(frozen=True)
class Point:
    x: int
    y: int


@dataclass(frozen=True)
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class DetectedBarcode:
    value: str
    format: str
    content_type: str
    orientation: int
    position: tuple[Point, ...]
    bounding_box: BoundingBox


@dataclass(frozen=True)
class Tile:
    image: Image.Image
    offset_x: int
    offset_y: int
    name: str


class BarcodeScanner:
    """
    Progressive Code 128 scanner for structured warehouse/product photos.

    Search strategy:
      1. Full-image scan.
      2. Regular overlapping grid at native resolution.
      3. Half-cell-shifted interior tiles at native resolution.
      4. Progressive fallback only for unresolved tiles.
      5. Targeted missing-region recovery:
         - infer row and column barcode-center clusters,
         - identify missing grid cells,
         - scan only small crops around those expected barcode locations.

    Identical values at different positions remain separate physical detections.
    """

    def __init__(
        self,
        *,
        tile_rows: int = 4,
        tile_columns: int = 3,
        tile_overlap: float = 0.12,
        enable_shifted_tiles: bool = True,
        enable_targeted_recovery: bool = True,
        targeted_crop_width_ratio: float = 1.00,
        targeted_crop_height_ratio: float = 1.00,
    ) -> None:
        if tile_rows < 1:
            raise ValueError("tile_rows must be at least 1")

        if tile_columns < 1:
            raise ValueError("tile_columns must be at least 1")

        if not 0 <= tile_overlap < 1:
            raise ValueError("tile_overlap must be between 0 and 1")

        if not 0.1 <= targeted_crop_width_ratio <= 2.0:
            raise ValueError(
                "targeted_crop_width_ratio must be between 0.1 and 2.0"
            )

        if not 0.1 <= targeted_crop_height_ratio <= 2.0:
            raise ValueError(
                "targeted_crop_height_ratio must be between 0.1 and 2.0"
            )

        self.tile_rows = tile_rows
        self.tile_columns = tile_columns
        self.tile_overlap = tile_overlap
        self.enable_shifted_tiles = enable_shifted_tiles
        self.enable_targeted_recovery = enable_targeted_recovery
        self.targeted_crop_width_ratio = targeted_crop_width_ratio
        self.targeted_crop_height_ratio = targeted_crop_height_ratio

    def scan_bytes(self, image_bytes: bytes) -> list[DetectedBarcode]:
        image = Image.open(BytesIO(image_bytes))
        image = ImageOps.exif_transpose(image).convert("RGB")
        return self.scan_image(image)

    def scan_image(self, image: Image.Image) -> list[DetectedBarcode]:
        detections: list[DetectedBarcode] = []

        # Pass 1: cheap full-image discovery.
        detections.extend(
            self._decode_region(
                image=image,
                offset_x=0,
                offset_y=0,
                scale=1.0,
                preprocessing="original",
                try_downscale=True,
            )
        )

        regular_tiles = list(self._generate_regular_tiles(image))
        shifted_tiles = (
            list(self._generate_shifted_tiles(image))
            if self.enable_shifted_tiles
            else []
        )

        unresolved_regular = self._scan_native_tiles(
            regular_tiles,
            detections,
        )
        detections = self._deduplicate(detections)

        unresolved_shifted = self._scan_native_tiles(
            shifted_tiles,
            detections,
        )
        detections = self._deduplicate(detections)

        unresolved = unresolved_regular + unresolved_shifted

        # General fallbacks remain progressive and local to unresolved tiles.
        unresolved, detections = self._run_fallback_pass(
            unresolved,
            detections,
            scale=2.0,
            preprocessing="original",
        )

        unresolved, detections = self._run_fallback_pass(
            unresolved,
            detections,
            scale=2.0,
            preprocessing="grayscale",
        )

        unresolved, detections = self._run_fallback_pass(
            unresolved,
            detections,
            scale=2.0,
            preprocessing="sharpened",
        )

        if unresolved:
            _, detections = self._run_fallback_pass(
                unresolved,
                detections,
                scale=3.0,
                preprocessing="original",
            )

        detections = self._deduplicate(detections)

        # Final focused recovery for structured row/column photos.
        if self.enable_targeted_recovery:
            detections = self._recover_missing_grid_cells(
                image,
                detections,
            )

        return self._deduplicate(detections)

    def _recover_missing_grid_cells(
        self,
        image: Image.Image,
        detections: list[DetectedBarcode],
    ) -> list[DetectedBarcode]:
        primary = [
            detection
            for detection in detections
            if self._is_primary_barcode(detection.value)
        ]

        expected_count = self.tile_rows * self.tile_columns

        if len(primary) >= expected_count:
            return detections

        if len(primary) < max(self.tile_rows, self.tile_columns):
            return detections

        centers = [
            self._box_center(detection.bounding_box)
            for detection in primary
        ]

        x_centers = self._cluster_1d(
            [x for x, _ in centers],
            self.tile_columns,
        )
        y_centers = self._cluster_1d(
            [y for _, y in centers],
            self.tile_rows,
        )

        if len(x_centers) != self.tile_columns:
            return detections

        if len(y_centers) != self.tile_rows:
            return detections

        occupied: set[tuple[int, int]] = set()

        for x, y in centers:
            column = self._nearest_index(x, x_centers)
            row = self._nearest_index(y, y_centers)
            occupied.add((row, column))

        missing_cells = [
            (row, column)
            for row in range(self.tile_rows)
            for column in range(self.tile_columns)
            if (row, column) not in occupied
        ]

        if not missing_cells:
            return detections

        x_spacing = self._typical_spacing(
            x_centers,
            fallback=image.width / self.tile_columns,
        )
        y_spacing = self._typical_spacing(
            y_centers,
            fallback=image.height / self.tile_rows,
        )

        updated = list(detections)

        for row, column in missing_cells:
            crop = self._make_targeted_crop(
                image=image,
                center_x=x_centers[column],
                center_y=y_centers[row],
                crop_width=x_spacing * self.targeted_crop_width_ratio,
                crop_height=y_spacing * self.targeted_crop_height_ratio,
                name=f"targeted-{row}-{column}",
            )

            found = self._decode_targeted_tile(crop)
            updated.extend(found)

        return self._deduplicate(updated)

    def _decode_targeted_tile(
        self,
        tile: Tile,
    ) -> list[DetectedBarcode]:
        """
        Targeted crops are small, so several attempts remain inexpensive.
        Stop as soon as a primary barcode is found.
        """
        attempts = (
            # Cheap attempts first.
            (2.0, "original", False),
            (2.0, "grayscale", False),
            (2.0, "sharpened", False),

            # Reproduce the stronger parts of the old exhaustive search,
            # but only for inferred missing barcode cells.
            (3.0, "original", False),
            (3.0, "grayscale", False),
            (3.0, "sharpened", False),

            # Final rare attempt for unusual light/dark or glare conditions.
            (3.0, "sharpened", True),
        )

        collected: list[DetectedBarcode] = []

        for scale, preprocessing, try_invert in attempts:
            found = self._decode_region(
                image=tile.image,
                offset_x=tile.offset_x,
                offset_y=tile.offset_y,
                scale=scale,
                preprocessing=preprocessing,
                try_downscale=False,
                try_invert=try_invert,
            )
            collected.extend(found)

            if self._contains_primary_barcode(found):
                break

        return self._deduplicate(collected)

    def _make_targeted_crop(
        self,
        *,
        image: Image.Image,
        center_x: float,
        center_y: float,
        crop_width: float,
        crop_height: float,
        name: str,
    ) -> Tile:
        half_width = crop_width / 2.0
        half_height = crop_height / 2.0

        x1 = max(0, round(center_x - half_width))
        y1 = max(0, round(center_y - half_height))
        x2 = min(image.width, round(center_x + half_width))
        y2 = min(image.height, round(center_y + half_height))

        return Tile(
            image=image.crop((x1, y1, x2, y2)),
            offset_x=x1,
            offset_y=y1,
            name=name,
        )

    @staticmethod
    def _cluster_1d(
        values: list[float],
        cluster_count: int,
        *,
        max_iterations: int = 30,
    ) -> list[float]:
        """
        Small deterministic 1D k-means implementation.

        Quantile-like initialization works well for ordered warehouse layouts
        and avoids adding another dependency.
        """
        if cluster_count < 1 or len(values) < cluster_count:
            return []

        ordered = sorted(values)

        if cluster_count == 1:
            return [sum(ordered) / len(ordered)]

        centers = [
            ordered[
                round(
                    index * (len(ordered) - 1) / (cluster_count - 1)
                )
            ]
            for index in range(cluster_count)
        ]

        for _ in range(max_iterations):
            groups: list[list[float]] = [
                [] for _ in range(cluster_count)
            ]

            for value in ordered:
                groups[
                    BarcodeScanner._nearest_index(value, centers)
                ].append(value)

            # An empty cluster means the evidence is insufficient.
            if any(not group for group in groups):
                return []

            new_centers = [
                sum(group) / len(group)
                for group in groups
            ]

            if all(
                abs(old - new) < 0.5
                for old, new in zip(centers, new_centers)
            ):
                centers = new_centers
                break

            centers = new_centers

        return sorted(centers)

    @staticmethod
    def _nearest_index(
        value: float,
        centers: list[float],
    ) -> int:
        return min(
            range(len(centers)),
            key=lambda index: abs(value - centers[index]),
        )

    @staticmethod
    def _typical_spacing(
        centers: list[float],
        *,
        fallback: float,
    ) -> float:
        if len(centers) < 2:
            return fallback

        differences = [
            right - left
            for left, right in zip(centers, centers[1:])
            if right > left
        ]

        if not differences:
            return fallback

        return float(median(differences))

    def _scan_native_tiles(
        self,
        tiles: list[Tile],
        detections: list[DetectedBarcode],
    ) -> list[Tile]:
        unresolved: list[Tile] = []

        for tile in tiles:
            found = self._decode_tile(
                tile,
                scale=1.0,
                preprocessing="original",
            )
            detections.extend(found)

            if not self._contains_primary_barcode(found):
                unresolved.append(tile)

        return unresolved

    def _run_fallback_pass(
        self,
        tiles: list[Tile],
        detections: list[DetectedBarcode],
        *,
        scale: float,
        preprocessing: str,
    ) -> tuple[list[Tile], list[DetectedBarcode]]:
        if not tiles:
            return [], detections

        updated = list(detections)
        still_unresolved: list[Tile] = []

        for tile in tiles:
            found = self._decode_tile(
                tile,
                scale=scale,
                preprocessing=preprocessing,
            )
            updated.extend(found)

            if not self._contains_primary_barcode(found):
                still_unresolved.append(tile)

        return still_unresolved, self._deduplicate(updated)

    def _decode_tile(
        self,
        tile: Tile,
        *,
        scale: float,
        preprocessing: str,
    ) -> list[DetectedBarcode]:
        return self._decode_region(
            image=tile.image,
            offset_x=tile.offset_x,
            offset_y=tile.offset_y,
            scale=scale,
            preprocessing=preprocessing,
            try_downscale=False,
        )

    def _decode_region(
        self,
        *,
        image: Image.Image,
        offset_x: int,
        offset_y: int,
        scale: float,
        preprocessing: str,
        try_downscale: bool,
        try_invert: bool = False,
    ) -> list[DetectedBarcode]:
        prepared = self._prepare_image(
            image,
            scale=scale,
            preprocessing=preprocessing,
        )

        scale_x = image.width / prepared.width
        scale_y = image.height / prepared.height

        results = zxingcpp.read_barcodes(
            np.asarray(prepared),
            formats=zxingcpp.BarcodeFormat.Code128,
            try_rotate=True,
            try_downscale=try_downscale,
            try_invert=try_invert,
            return_errors=False,
        )

        detections: list[DetectedBarcode] = []

        for result in results:
            if not result.text:
                continue

            position = self._map_position(
                result.position,
                offset_x=offset_x,
                offset_y=offset_y,
                scale_x=scale_x,
                scale_y=scale_y,
            )

            detections.append(
                DetectedBarcode(
                    value=result.text,
                    format=self._normalize_format(result.format),
                    content_type=str(result.content_type),
                    orientation=int(result.orientation),
                    position=position,
                    bounding_box=self._bounding_box(position),
                )
            )

        return detections

    @staticmethod
    def _prepare_image(
        image: Image.Image,
        *,
        scale: float,
        preprocessing: str,
    ) -> Image.Image:
        prepared = image

        if scale != 1.0:
            prepared = prepared.resize(
                (
                    max(1, round(prepared.width * scale)),
                    max(1, round(prepared.height * scale)),
                ),
                Image.Resampling.LANCZOS,
            )

        if preprocessing == "original":
            return prepared

        grayscale = ImageOps.grayscale(prepared)

        if preprocessing == "grayscale":
            return grayscale

        if preprocessing == "sharpened":
            contrasted = ImageEnhance.Contrast(grayscale).enhance(1.8)

            return contrasted.filter(
                ImageFilter.UnsharpMask(
                    radius=1.5,
                    percent=180,
                    threshold=2,
                )
            )

        raise ValueError(f"Unknown preprocessing mode: {preprocessing}")

    def _generate_regular_tiles(
        self,
        image: Image.Image,
    ) -> Iterable[Tile]:
        cell_width = image.width / self.tile_columns
        cell_height = image.height / self.tile_rows

        for row in range(self.tile_rows):
            for column in range(self.tile_columns):
                yield self._make_tile(
                    image=image,
                    core_x1=column * cell_width,
                    core_y1=row * cell_height,
                    core_x2=(column + 1) * cell_width,
                    core_y2=(row + 1) * cell_height,
                    name=f"regular-{row}-{column}",
                )

    def _generate_shifted_tiles(
        self,
        image: Image.Image,
    ) -> Iterable[Tile]:
        """
        For a 4x3 regular grid, create 3x2 = 6 interior shifted tiles.
        Each shifted tile straddles regular-grid boundaries.
        """
        if self.tile_rows < 2 or self.tile_columns < 2:
            return

        cell_width = image.width / self.tile_columns
        cell_height = image.height / self.tile_rows

        for row in range(self.tile_rows - 1):
            for column in range(self.tile_columns - 1):
                core_x1 = (column + 0.5) * cell_width
                core_y1 = (row + 0.5) * cell_height

                yield self._make_tile(
                    image=image,
                    core_x1=core_x1,
                    core_y1=core_y1,
                    core_x2=core_x1 + cell_width,
                    core_y2=core_y1 + cell_height,
                    name=f"shifted-{row}-{column}",
                )

    def _make_tile(
        self,
        *,
        image: Image.Image,
        core_x1: float,
        core_y1: float,
        core_x2: float,
        core_y2: float,
        name: str,
    ) -> Tile:
        cell_width = core_x2 - core_x1
        cell_height = core_y2 - core_y1

        overlap_x = cell_width * self.tile_overlap
        overlap_y = cell_height * self.tile_overlap

        crop_x1 = max(0, round(core_x1 - overlap_x))
        crop_y1 = max(0, round(core_y1 - overlap_y))
        crop_x2 = min(image.width, round(core_x2 + overlap_x))
        crop_y2 = min(image.height, round(core_y2 + overlap_y))

        return Tile(
            image=image.crop((crop_x1, crop_y1, crop_x2, crop_y2)),
            offset_x=crop_x1,
            offset_y=crop_y1,
            name=name,
        )

    @classmethod
    def _contains_primary_barcode(
        cls,
        detections: list[DetectedBarcode],
    ) -> bool:
        return any(
            cls._is_primary_barcode(detection.value)
            for detection in detections
        )

    @staticmethod
    def _is_primary_barcode(value: str) -> bool:
        return value.isdigit() and len(value) >= 12

    @staticmethod
    def _normalize_format(value: object) -> str:
        normalized = str(value).replace(" ", "").lower()

        if normalized == "code128":
            return "Code128"

        return str(value).replace(" ", "")

    @staticmethod
    def _map_position(
        position: object,
        *,
        offset_x: int,
        offset_y: int,
        scale_x: float,
        scale_y: float,
    ) -> tuple[Point, ...]:
        raw_points = (
            position.top_left,
            position.top_right,
            position.bottom_right,
            position.bottom_left,
        )

        return tuple(
            Point(
                x=offset_x + round(point.x * scale_x),
                y=offset_y + round(point.y * scale_y),
            )
            for point in raw_points
        )

    @staticmethod
    def _bounding_box(
        position: tuple[Point, ...],
    ) -> BoundingBox:
        xs = [point.x for point in position]
        ys = [point.y for point in position]

        return BoundingBox(
            x1=min(xs),
            y1=min(ys),
            x2=max(xs),
            y2=max(ys),
        )

    @classmethod
    def _deduplicate(
        cls,
        detections: list[DetectedBarcode],
    ) -> list[DetectedBarcode]:
        sorted_detections = sorted(
            detections,
            key=lambda detection: cls._box_area(detection.bounding_box),
            reverse=True,
        )

        unique: list[DetectedBarcode] = []

        for candidate in sorted_detections:
            duplicate_index: int | None = None

            for index, existing in enumerate(unique):
                if candidate.value != existing.value:
                    continue

                if candidate.format != existing.format:
                    continue

                if cls._same_physical_barcode(candidate, existing):
                    duplicate_index = index
                    break

            if duplicate_index is None:
                unique.append(candidate)
                continue

            existing = unique[duplicate_index]

            if cls._box_area(candidate.bounding_box) > cls._box_area(
                existing.bounding_box
            ):
                unique[duplicate_index] = candidate

        return sorted(
            unique,
            key=lambda detection: (
                detection.bounding_box.y1,
                detection.bounding_box.x1,
            ),
        )

    @classmethod
    def _same_physical_barcode(
        cls,
        first: DetectedBarcode,
        second: DetectedBarcode,
    ) -> bool:
        first_box = first.bounding_box
        second_box = second.bounding_box

        first_center_x, first_center_y = cls._box_center(first_box)
        second_center_x, second_center_y = cls._box_center(second_box)

        center_distance_x = abs(first_center_x - second_center_x)
        center_distance_y = abs(first_center_y - second_center_y)

        first_width = max(1, first_box.x2 - first_box.x1)
        first_height = max(1, first_box.y2 - first_box.y1)

        second_width = max(1, second_box.x2 - second_box.x1)
        second_height = max(1, second_box.y2 - second_box.y1)

        longest_dimension = max(
            first_width,
            first_height,
            second_width,
            second_height,
        )

        position_tolerance = max(40.0, longest_dimension * 0.35)

        if (
            center_distance_x <= position_tolerance
            and center_distance_y <= position_tolerance
        ):
            return True

        return cls._intersection_over_union(
            cls._expand_box(first_box, padding=20),
            cls._expand_box(second_box, padding=20),
        ) >= 0.20

    @staticmethod
    def _box_center(
        box: BoundingBox,
    ) -> tuple[float, float]:
        return (
            (box.x1 + box.x2) / 2.0,
            (box.y1 + box.y2) / 2.0,
        )

    @staticmethod
    def _expand_box(
        box: BoundingBox,
        *,
        padding: int,
    ) -> BoundingBox:
        return BoundingBox(
            x1=box.x1 - padding,
            y1=box.y1 - padding,
            x2=box.x2 + padding,
            y2=box.y2 + padding,
        )

    @staticmethod
    def _intersection_over_union(
        first: BoundingBox,
        second: BoundingBox,
    ) -> float:
        intersection_x1 = max(first.x1, second.x1)
        intersection_y1 = max(first.y1, second.y1)
        intersection_x2 = min(first.x2, second.x2)
        intersection_y2 = min(first.y2, second.y2)

        intersection_width = max(0, intersection_x2 - intersection_x1)
        intersection_height = max(0, intersection_y2 - intersection_y1)

        intersection_area = intersection_width * intersection_height

        if intersection_area == 0:
            return 0.0

        first_area = BarcodeScanner._box_area(first)
        second_area = BarcodeScanner._box_area(second)
        union_area = first_area + second_area - intersection_area

        if union_area <= 0:
            return 0.0

        return intersection_area / union_area

    @staticmethod
    def _box_area(
        box: BoundingBox,
    ) -> int:
        width = max(1, box.x2 - box.x1)
        height = max(1, box.y2 - box.y1)
        return width * height
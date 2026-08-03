from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from io import BytesIO
from statistics import median

import cv2
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


@dataclass(frozen=True)
class LabelCandidate:
    crop: Image.Image
    offset_x: int
    offset_y: int
    bounding_box: BoundingBox
    score: float


class BarcodeScanner:
    """
    Progressive Code 128 scanner for warehouse/product photos.

    Fast path:
      1. Full-image scan.
      2. Regular overlapping grid.
      3. Half-cell-shifted interior tiles.
      4. Progressive local fallbacks.
      5. Targeted recovery for structured row/column photos.

    Difficult-photo fallback:
      6. If very few primary barcodes were found, detect bright rectangular
         product labels with OpenCV and scan each label crop independently.
         Label crops use CLAHE, Otsu, adaptive thresholding, NEAREST binary
         upscaling, stronger sharpening, and a limited inversion attempt.

    The label fallback runs only when the fast path finds fewer than
    ``label_fallback_threshold`` primary barcodes, preserving the latency and
    recall of the existing clean-image pipeline.
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
        enable_label_fallback: bool = True,
        label_fallback_threshold: int = 4,
        max_label_candidates: int = 24,
        label_padding_ratio: float = 0.12,
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

        if label_fallback_threshold < 0:
            raise ValueError("label_fallback_threshold cannot be negative")

        if max_label_candidates < 1:
            raise ValueError("max_label_candidates must be at least 1")

        if not 0 <= label_padding_ratio <= 0.5:
            raise ValueError("label_padding_ratio must be between 0 and 0.5")

        self.tile_rows = tile_rows
        self.tile_columns = tile_columns
        self.tile_overlap = tile_overlap
        self.enable_shifted_tiles = enable_shifted_tiles
        self.enable_targeted_recovery = enable_targeted_recovery
        self.targeted_crop_width_ratio = targeted_crop_width_ratio
        self.targeted_crop_height_ratio = targeted_crop_height_ratio

        self.enable_label_fallback = enable_label_fallback
        self.label_fallback_threshold = label_fallback_threshold
        self.max_label_candidates = max_label_candidates
        self.label_padding_ratio = label_padding_ratio

    def scan_bytes(self, image_bytes: bytes) -> list[DetectedBarcode]:
        image = Image.open(BytesIO(image_bytes))
        image = ImageOps.exif_transpose(image).convert("RGB")
        return self.scan_image(image)

    def scan_image(self, image: Image.Image) -> list[DetectedBarcode]:
        image = ImageOps.exif_transpose(image).convert("RGB")
        detections = self._scan_fast_path(image)
        detections = self._deduplicate(detections)

        primary_count = self._count_primary_barcodes(detections)

        if (
            self.enable_label_fallback
            and primary_count < self.label_fallback_threshold
        ):
            detections.extend(
                self._scan_detected_label_candidates(
                    image,
                    existing=detections,
                )
            )

        return self._deduplicate(detections)

    def _scan_fast_path(
        self,
        image: Image.Image,
    ) -> list[DetectedBarcode]:
        detections: list[DetectedBarcode] = []

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

        if self.enable_targeted_recovery:
            detections = self._recover_missing_grid_cells(
                image,
                detections,
            )

        return self._deduplicate(detections)

    # ------------------------------------------------------------------
    # OpenCV label-candidate fallback
    # ------------------------------------------------------------------

    def _scan_detected_label_candidates(
        self,
        image: Image.Image,
        *,
        existing: list[DetectedBarcode],
    ) -> list[DetectedBarcode]:
        candidates = self._detect_label_candidates(image)

        if not candidates:
            return []

        detections: list[DetectedBarcode] = []

        for candidate in candidates:
            if self._candidate_already_has_primary(
                candidate.bounding_box,
                existing + detections,
            ):
                continue

            found = self._decode_label_candidate(candidate)
            detections.extend(found)

        return self._deduplicate(detections)

    def _detect_label_candidates(
        self,
        image: Image.Image,
    ) -> list[LabelCandidate]:
        """
        Propose bright, low-saturation, rectangular label regions.

        False positives are acceptable because ZXing validates each crop.
        Candidates are capped to keep latency bounded.
        """
        rgb = np.asarray(image)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        # White/grey paper labels: relatively bright and not highly saturated.
        mask = np.where(
            (value >= 135) & (saturation <= 95),
            255,
            0,
        ).astype(np.uint8)

        # Retain moderately bright low-contrast labels in blurry images.
        gray_mask = cv2.threshold(
            gray,
            145,
            255,
            cv2.THRESH_BINARY,
        )[1]
        mask = cv2.bitwise_or(mask, gray_mask)

        # Join fragmented text/barcode/white-paper patches into label blobs.
        image_scale = max(image.width, image.height) / 1600.0
        close_w = max(9, round(23 * image_scale))
        close_h = max(7, round(15 * image_scale))
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (close_w | 1, close_h | 1),
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            close_kernel,
            iterations=2,
        )

        open_size = max(3, round(5 * image_scale)) | 1
        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (open_size, open_size),
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            open_kernel,
            iterations=1,
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        image_area = image.width * image.height
        min_area = image_area * 0.003
        max_area = image_area * 0.10

        raw: list[tuple[float, BoundingBox]] = []

        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            area = width * height

            if area < min_area or area > max_area:
                continue

            short_side = min(width, height)
            long_side = max(width, height)

            if short_side < max(35, round(min(image.size) * 0.035)):
                continue

            aspect_ratio = long_side / max(1, short_side)

            # Labels in these samples vary from near-square to elongated.
            if not 1.05 <= aspect_ratio <= 4.5:
                continue

            contour_area = cv2.contourArea(contour)
            rectangularity = contour_area / max(1.0, area)

            if rectangularity < 0.35:
                continue

            roi_mask = mask[y : y + height, x : x + width]
            white_ratio = float(np.count_nonzero(roi_mask)) / max(1, area)

            if white_ratio < 0.32:
                continue

            # Prefer label-sized, bright, rectangular proposals.
            normalized_area = area / image_area
            score = (
                rectangularity * 2.0
                + white_ratio
                + min(normalized_area / 0.02, 1.5)
            )

            raw.append(
                (
                    score,
                    BoundingBox(
                        x1=x,
                        y1=y,
                        x2=x + width,
                        y2=y + height,
                    ),
                )
            )

        boxes = self._suppress_overlapping_label_boxes(raw)

        candidates: list[LabelCandidate] = []

        for score, box in boxes[: self.max_label_candidates]:
            padded = self._pad_box(
                box,
                image_width=image.width,
                image_height=image.height,
                padding_ratio=self.label_padding_ratio,
            )

            crop = image.crop(
                (
                    padded.x1,
                    padded.y1,
                    padded.x2,
                    padded.y2,
                )
            )

            candidates.append(
                LabelCandidate(
                    crop=crop,
                    offset_x=padded.x1,
                    offset_y=padded.y1,
                    bounding_box=padded,
                    score=score,
                )
            )

        # Stable reading order helps debugging and repeatability.
        return sorted(
            candidates,
            key=lambda candidate: (
                candidate.bounding_box.y1,
                candidate.bounding_box.x1,
            ),
        )

    def _decode_label_candidate(
        self,
        candidate: LabelCandidate,
    ) -> list[DetectedBarcode]:
        """
        Strong but bounded label-crop search.

        The crop is already small, so these variants remain far cheaper than
        applying the same work to the whole image.
        """
        attempts = (
            # Normal image variants.
            (2.0, "original", False),
            (2.0, "clahe", False),
            (2.0, "sharpened", False),

            # Hard binary variants. NEAREST scaling is selected internally.
            (3.0, "otsu", False),
            (3.0, "adaptive", False),

            # Stronger final attempts for severe blur.
            (3.0, "aggressive_sharpen", False),
            (0.80, "original", False),
            (3.0, "otsu", True),
        )

        collected: list[DetectedBarcode] = []

        for scale, preprocessing, try_invert in attempts:
            found = self._decode_region(
                image=candidate.crop,
                offset_x=candidate.offset_x,
                offset_y=candidate.offset_y,
                scale=scale,
                preprocessing=preprocessing,
                try_downscale=False,
                try_invert=try_invert,
            )
            collected.extend(found)

            if self._contains_primary_barcode(found):
                break

        return self._deduplicate(collected)

    @classmethod
    def _candidate_already_has_primary(
        cls,
        candidate_box: BoundingBox,
        detections: list[DetectedBarcode],
    ) -> bool:
        for detection in detections:
            if not cls._is_primary_barcode(detection.value):
                continue

            center_x, center_y = cls._box_center(
                detection.bounding_box
            )

            if (
                candidate_box.x1 <= center_x <= candidate_box.x2
                and candidate_box.y1 <= center_y <= candidate_box.y2
            ):
                return True

        return False

    @classmethod
    def _suppress_overlapping_label_boxes(
        cls,
        scored_boxes: list[tuple[float, BoundingBox]],
    ) -> list[tuple[float, BoundingBox]]:
        kept: list[tuple[float, BoundingBox]] = []

        for score, candidate in sorted(
            scored_boxes,
            key=lambda item: item[0],
            reverse=True,
        ):
            if any(
                cls._intersection_over_union(candidate, existing) >= 0.35
                or cls._containment_ratio(candidate, existing) >= 0.75
                or cls._containment_ratio(existing, candidate) >= 0.75
                for _, existing in kept
            ):
                continue

            kept.append((score, candidate))

        return kept

    @staticmethod
    def _containment_ratio(
        inner: BoundingBox,
        outer: BoundingBox,
    ) -> float:
        intersection_x1 = max(inner.x1, outer.x1)
        intersection_y1 = max(inner.y1, outer.y1)
        intersection_x2 = min(inner.x2, outer.x2)
        intersection_y2 = min(inner.y2, outer.y2)

        intersection_width = max(0, intersection_x2 - intersection_x1)
        intersection_height = max(0, intersection_y2 - intersection_y1)
        intersection_area = intersection_width * intersection_height

        inner_area = BarcodeScanner._box_area(inner)

        return intersection_area / max(1, inner_area)

    @staticmethod
    def _pad_box(
        box: BoundingBox,
        *,
        image_width: int,
        image_height: int,
        padding_ratio: float,
    ) -> BoundingBox:
        width = box.x2 - box.x1
        height = box.y2 - box.y1

        padding_x = round(width * padding_ratio)
        padding_y = round(height * padding_ratio)

        return BoundingBox(
            x1=max(0, box.x1 - padding_x),
            y1=max(0, box.y1 - padding_y),
            x2=min(image_width, box.x2 + padding_x),
            y2=min(image_height, box.y2 + padding_y),
        )

    # ------------------------------------------------------------------
    # Structured-grid targeted recovery
    # ------------------------------------------------------------------

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

            updated.extend(self._decode_targeted_tile(crop))

        return self._deduplicate(updated)

    def _decode_targeted_tile(
        self,
        tile: Tile,
    ) -> list[DetectedBarcode]:
        attempts = (
            (2.0, "original", False),
            (2.0, "grayscale", False),
            (2.0, "sharpened", False),
            (3.0, "original", False),
            (3.0, "grayscale", False),
            (3.0, "sharpened", False),
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

    # ------------------------------------------------------------------
    # Fast tiling pipeline
    # ------------------------------------------------------------------

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
        binary_mode = preprocessing in {"otsu", "adaptive"}

        prepared = image

        # For binary images, threshold first and resize with NEAREST.
        if binary_mode:
            grayscale = ImageOps.grayscale(prepared)
            gray_array = np.asarray(grayscale)

            if preprocessing == "otsu":
                processed = cv2.threshold(
                    gray_array,
                    0,
                    255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                )[1]
            else:
                block_size = 31

                # Block size must be odd and fit the image.
                max_block = min(gray_array.shape[:2])
                if max_block <= 3:
                    processed = gray_array
                else:
                    block_size = min(block_size, max_block)
                    if block_size % 2 == 0:
                        block_size -= 1
                    block_size = max(3, block_size)

                    processed = cv2.adaptiveThreshold(
                        gray_array,
                        255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY,
                        block_size,
                        7,
                    )

            prepared = Image.fromarray(processed)

            if scale != 1.0:
                prepared = prepared.resize(
                    (
                        max(1, round(prepared.width * scale)),
                        max(1, round(prepared.height * scale)),
                    ),
                    Image.Resampling.NEAREST,
                )

            return prepared

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

        if preprocessing == "clahe":
            gray_array = np.asarray(grayscale)
            clahe = cv2.createCLAHE(
                clipLimit=2.5,
                tileGridSize=(8, 8),
            )
            return Image.fromarray(clahe.apply(gray_array))

        if preprocessing == "sharpened":
            contrasted = ImageEnhance.Contrast(grayscale).enhance(1.8)

            return contrasted.filter(
                ImageFilter.UnsharpMask(
                    radius=1.5,
                    percent=180,
                    threshold=2,
                )
            )

        if preprocessing == "aggressive_sharpen":
            contrasted = ImageEnhance.Contrast(grayscale).enhance(2.0)

            return contrasted.filter(
                ImageFilter.UnsharpMask(
                    radius=2.75,
                    percent=300,
                    threshold=1,
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

    # ------------------------------------------------------------------
    # Common helpers
    # ------------------------------------------------------------------

    @classmethod
    def _count_primary_barcodes(
        cls,
        detections: list[DetectedBarcode],
    ) -> int:
        return sum(
            1
            for detection in detections
            if cls._is_primary_barcode(detection.value)
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

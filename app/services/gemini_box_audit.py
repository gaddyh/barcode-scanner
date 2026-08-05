"""
gemini_box_audit.py

Independent visual audit for a shoebox image.

The module does NOT treat Gemini as the authoritative barcode decoder.
It asks Gemini to:
- count physical shoeboxes;
- count visible product-barcode labels;
- identify boxes with missing/obscured/blurred labels;
- estimate distinct product-label groups;
- report scene-quality risks;
- optionally return approximate normalized bounding boxes.

Install:
    pip install google-genai pydantic

Environment:
    export GEMINI_API_KEY="..."
    export GEMINI_MODEL="gemini-2.5-flash"   # optional

CLI:
    python gemini_box_audit.py /path/to/image.jpg

Import:
    from gemini_box_audit import audit_shoebox_image

    result = audit_shoebox_image("/path/to/image.jpg")
    print(result.model_dump_json(indent=2))
"""

from __future__ import annotations

import argparse
import mimetypes
import os
import time
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.services.spatial_geometry import (
    PixelBoundingBox,
    clamp_bbox,
    normalized_to_pixels,
)

DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_COUNTS_MODEL = "gemini-3.5-flash-lite"
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_DELAY_SECONDS = 1.0

SUPPORTED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/heic",
    "image/heif",
}


class ImageQuality(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNUSABLE = "unusable"


class AuditConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class BoxLabelStatus(str, Enum):
    CLEAR = "clear"
    PARTIALLY_OBSCURED = "partially_obscured"
    BLURRED = "blurred"
    CROPPED = "cropped"
    NOT_VISIBLE = "not_visible"
    UNCERTAIN = "uncertain"


NormalizedCoordinate = Annotated[int, Field(ge=0, le=1000)]


class BoxAuditCounts(BaseModel):
    """
    Minimal fast audit: only the four count fields needed to decide whether
    the deterministic scanner's output is consistent with the visual scene.
    """

    model_config = ConfigDict(extra="forbid")

    visible_product_barcode_label_count: int = Field(
        ge=0,
        description="Physical shoebox product labels with a visible barcode region.",
    )
    clear_product_barcode_label_count: int = Field(
        ge=0,
        description="Product barcode labels that appear clear enough to scan.",
    )
    boxes_without_visible_product_barcode: int = Field(
        ge=0,
        description="Physical boxes for which no product barcode is visible.",
    )
    partially_obscured_product_barcode_count: int = Field(
        ge=0,
        description="Product barcode regions that are partially obscured.",
    )


COUNTS_PROMPT = """
Count product-barcode labels visible on shoeboxes in this image.

Return only these four integers:
- visible_product_barcode_label_count: physical shoebox product labels with a
  visible barcode region (whether or not digits are readable).
- clear_product_barcode_label_count: product barcode labels that appear clear
  enough for a dedicated barcode decoder.
- boxes_without_visible_product_barcode: physical boxes with no visible product
  barcode.
- partially_obscured_product_barcode_count: product barcode regions that are
  partially obscured.

Do not guess barcode digits. Count only what is visibly present.
""".strip()


class NormalizedBoundingBox(BaseModel):
    """
    Approximate bounding box using Gemini's normalized 0..1000 coordinate space.

    Coordinates are ordered as:
        top, left, bottom, right
    """

    model_config = ConfigDict(extra="forbid")

    top: NormalizedCoordinate
    left: NormalizedCoordinate
    bottom: NormalizedCoordinate
    right: NormalizedCoordinate

    @model_validator(mode="after")
    def validate_order(self) -> NormalizedBoundingBox:
        if self.bottom <= self.top:
            raise ValueError("bottom must be greater than top")
        if self.right <= self.left:
            raise ValueError("right must be greater than left")
        return self


# ---------------------------------------------------------------------------
# Spatial label audit — Gemini-facing (normalized 0..1000) and pipeline-facing
# (pixel) models. The normalized models stay internal to this module; downstream
# code only receives ``SpatialLabelAuditPixels``.
# ---------------------------------------------------------------------------


class SpatialLabelStatus(str, Enum):
    CLEAR = "clear"
    PARTIALLY_OBSCURED = "partially_obscured"
    BLURRED = "blurred"
    CROPPED = "cropped"
    UNCERTAIN = "uncertain"


class SpatialLabelObservation(BaseModel):
    """
    One visible shoebox product label, Gemini-facing (normalized 0..1000).

    ``label_bbox`` covers the complete printed product label.
    ``barcode_bbox`` tightly covers the barcode bars when they are visible,
    or ``None`` when the label is visible but the barcode region cannot be
    localized.
    """

    model_config = ConfigDict(extra="forbid")

    label_index: int = Field(ge=1)
    label_bbox: NormalizedBoundingBox
    barcode_bbox: NormalizedBoundingBox | None = None
    status: SpatialLabelStatus
    confidence: AuditConfidence


class SpatialLabelAudit(BaseModel):
    """
    Gemini-facing spatial audit result.

    No separate count fields — ``visible_count = len(labels)`` and
    ``clear_count = sum(label.status == CLEAR)`` are derived by the caller.
    The ``labels`` array is authoritative.
    """

    model_config = ConfigDict(extra="forbid")

    labels: list[SpatialLabelObservation]


class SpatialLabelObservationPixels(BaseModel):
    """
    One visible shoebox product label, pipeline-facing (pixel coordinates).

    This is what downstream code (reconciliation, cropping, visualization)
    consumes. The Gemini-native 0..1000 coordinates have been converted and
    clamped to the image frame.
    """

    model_config = ConfigDict(extra="forbid")

    label_index: int = Field(ge=1)
    label_bbox: PixelBoundingBox
    barcode_bbox: PixelBoundingBox | None = None
    status: SpatialLabelStatus
    confidence: AuditConfidence


class SpatialLabelAuditPixels(BaseModel):
    """
    Pipeline-facing spatial audit result with pixel coordinates.

    ``image_width`` / ``image_height`` are the dimensions of the normalized
    image that both Gemini and the scanner analyzed, so downstream code can
    compute normalized distances consistently.
    """

    model_config = ConfigDict(extra="forbid")

    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    labels: list[SpatialLabelObservationPixels]

    @property
    def visible_count(self) -> int:
        return len(self.labels)

    @property
    def clear_count(self) -> int:
        return sum(
            label.status == SpatialLabelStatus.CLEAR for label in self.labels
        )


SPATIAL_LABEL_PROMPT = """
Locate every visible shoebox product label that contains, or appears intended
to contain, the product barcode used to identify the sellable item.

Return one observation per physical product label.

For every observation:
- label_bbox must cover the complete printed product label.
- barcode_bbox must tightly cover the barcode bars — the striped black-and-white
  pattern of alternating bars and spaces.  This is NOT the human-readable digits
  printed below the bars, NOT the product name or brand text, and NOT any other
  printed text on the label.  Point only at the machine-readable bar pattern
  itself.
- The barcode bars are typically a narrow horizontal or vertical strip of
  alternating dark and light bars, usually much wider than it is tall (or taller
  than it is wide if rotated).  The bounding box should match that aspect ratio.
- If the label is visible but the barcode region cannot be localized,
  return barcode_bbox as null.
- Do not return shipping labels, warehouse labels, carton labels, handwritten
  stickers, or unrelated barcodes.
- Do not read, guess, or return barcode values.
- Use normalized coordinates from 0 to 1000 in this order:
  top, left, bottom, right.
- Assign label indexes top-to-bottom, then left-to-right.
- Do not return count fields separately; the labels array is authoritative.
""".strip()


class VisibleLabelText(BaseModel):
    """
    Human-readable text that appears on a product label.

    This is advisory OCR only. It must not replace the deterministic barcode
    decoder or the Priority item lookup.
    """

    model_config = ConfigDict(extra="forbid")

    box_index: int = Field(ge=1)
    model_or_style: str | None = None
    color: str | None = None
    size: str | None = None
    width: str | None = None
    other_text: list[str] = Field(default_factory=list)
    confidence: AuditConfidence = AuditConfidence.LOW


class BoxObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    box_index: int = Field(
        ge=1,
        description="Stable 1-based index assigned to one physical shoebox.",
    )
    box_bbox: NormalizedBoundingBox | None = None
    product_label_bbox: NormalizedBoundingBox | None = None
    label_status: BoxLabelStatus
    appears_to_have_product_barcode: bool
    possible_extra_non_product_barcode: bool = Field(
        description=(
            "True when the box also appears to show a shipping, logistics, "
            "supplier, or other barcode that may not identify the sellable SKU."
        )
    )
    note: str | None = None


class ShoeboxImageAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    physical_box_count: int = Field(
        ge=0,
        description="Estimated number of distinct physical shoeboxes visible.",
    )
    visible_product_barcode_label_count: int = Field(
        ge=0,
        description=(
            "Number of physical product labels on which a product barcode "
            "region is visibly present, whether or not its digits are readable."
        ),
    )
    clear_product_barcode_label_count: int = Field(
        ge=0,
        description="Number of product barcode labels that look clear enough to scan.",
    )
    boxes_without_visible_product_barcode: int = Field(
        ge=0,
        description="Physical boxes for which no product barcode is visible.",
    )
    partially_obscured_product_barcode_count: int = Field(ge=0)
    blurred_product_barcode_count: int = Field(ge=0)
    cropped_product_barcode_count: int = Field(ge=0)

    estimated_unique_product_label_groups: int = Field(
        ge=0,
        description=(
            "Visual estimate of distinct product-label groups based on readable "
            "model, size, color, and label appearance. This is not a count of "
            "decoded unique barcode values."
        ),
    )

    possible_non_product_barcode_count: int = Field(
        ge=0,
        description=(
            "Estimated count of visible shipping/logistics/supplier barcodes "
            "that may not represent a sellable SKU."
        ),
    )
    duplicate_view_risk: bool = Field(
        description=(
            "True if mirrors, repeated screens, reflections, or unusual framing "
            "could make the same physical box or label appear more than once."
        ),
    )
    overlapping_boxes_risk: bool
    image_quality: ImageQuality
    overall_confidence: AuditConfidence
    suitable_for_automatic_draft: bool = Field(
        description=(
            "Visual opinion only: true when every visible box appears to have "
            "one clear product label and no major scene-quality issue exists. "
            "The caller must still reconcile this with deterministic decoding."
        ),
    )

    observations: list[BoxObservation] = Field(default_factory=list)
    visible_label_text: list[VisibleLabelText] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_internal_consistency(self) -> ShoeboxImageAudit:
        if self.visible_product_barcode_label_count > self.physical_box_count:
            # More than one label may occasionally be visible on a box, but the
            # requested metric is one product-label presence per physical box.
            raise ValueError(
                "visible_product_barcode_label_count cannot exceed physical_box_count"
            )

        if self.clear_product_barcode_label_count > self.visible_product_barcode_label_count:
            raise ValueError(
                "clear_product_barcode_label_count cannot exceed "
                "visible_product_barcode_label_count"
            )

        if self.boxes_without_visible_product_barcode > self.physical_box_count:
            raise ValueError(
                "boxes_without_visible_product_barcode cannot exceed physical_box_count"
            )

        return self


AUDIT_PROMPT = """
Act as an independent visual auditor for a system that converts photographs of
shoeboxes into draft ERP order lines.

Inspect the image carefully and assess the physical scene. Do not act as the
authoritative barcode decoder, and do not guess barcode digits.

Count and distinguish all of the following:

1. Distinct physical shoeboxes.
2. Physical shoebox product labels with a visible barcode region.
3. Product barcode labels that appear clear enough for a dedicated barcode
   decoder.
4. Boxes whose product barcode is not visible.
5. Product barcode regions that are obscured, blurred, or cropped.
6. Visually distinct product-label groups, using visible model/style, color,
   size, width, and label appearance. This is only a visual estimate and is
   not the number of decoded unique barcode values.
7. Possible non-product barcodes, such as shipping, logistics, warehouse,
   supplier, or carton barcodes.
8. Risks caused by overlap, reflections, duplicate views, perspective,
   low resolution, glare, or cropping.

Assign one observation to every distinct physical shoebox you can identify.
Use stable 1-based box indexes. When reasonably possible, include approximate
bounding boxes using normalized coordinates from 0 to 1000 in this order:
top, left, bottom, right.

Read human-visible model/style, color, size, and width text only when it is
actually legible. Never manufacture missing text. Do not return or infer
barcode numbers.

Set suitable_for_automatic_draft to true only when the visual scene appears to
show exactly one clear product-barcode label for every physical box and there
are no meaningful quality, overlap, duplicate-view, or non-product-barcode
ambiguities. This flag is advisory; deterministic barcode results will be
compared separately.
""".strip()


class ShoeboxAuditError(RuntimeError):
    """Raised when the Gemini visual audit cannot be completed safely."""


def _gemini_compatible_schema(model: type[BaseModel]) -> dict:
    """
    Return a JSON schema for ``model`` that Gemini's response_schema accepts.

    Pydantic emits ``additionalProperties: false`` whenever a model uses
    ``ConfigDict(extra="forbid")``. Gemini's structured-output endpoint rejects
    that field with ``INVALID_ARGUMENT``. We keep ``extra="forbid"`` on the
    Pydantic side (so we still validate strictly on the client) and strip the
    field from the schema we send to Gemini.
    """
    schema = model.model_json_schema()

    def _strip(node: object) -> None:
        if isinstance(node, dict):
            node.pop("additionalProperties", None)
            for value in node.values():
                _strip(value)
        elif isinstance(node, list):
            for item in node:
                _strip(item)

    _strip(schema)
    return schema


def _detect_mime_type(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(image_path.name)
    if mime_type in SUPPORTED_MIME_TYPES:
        return mime_type

    # Small signature fallback for common image formats.
    header = image_path.read_bytes()[:16]

    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return "image/webp"
    if len(header) >= 12 and header[4:12] in {
        b"ftypheic",
        b"ftypheix",
        b"ftyphevc",
        b"ftyphevx",
        b"ftypmif1",
    }:
        return "image/heic"

    raise ValueError(
        f"Unsupported or unrecognized image type for '{image_path}'. "
        f"Supported MIME types: {sorted(SUPPORTED_MIME_TYPES)}"
    )


@dataclass(frozen=True)
class NormalizedImage:
    """
    In-memory image normalized for Gemini analysis.

    The raw file is never modified. Normalization happens entirely in memory:

    - EXIF orientation is applied to the pixel matrix (``ImageOps.exif_transpose``)
      so a phone photo stored sideways is rotated to its display orientation.
    - The image is converted to RGB.
    - If either dimension exceeds ``max_dimension``, it is resized (aspect
      ratio preserved via ``thumbnail``). Images already smaller than
      ``max_dimension`` are left untouched. Gemini only needs enough resolution
      to locate labels, not decode thin barcode lines.
    - It is re-encoded as JPEG bytes with ``image/jpeg`` declared as the MIME
      type.

    ``width``/``height`` are the dimensions Gemini actually sees (possibly
    resized). ``original_width``/``original_height`` are the full-resolution
    dimensions. Gemini's normalized 0..1000 coordinates are converted directly
    to the original frame, so the resize does not affect the coordinate system
    downstream code receives.
    """

    data: bytes
    mime_type: str
    width: int
    height: int
    original_width: int
    original_height: int


# Default max dimension for the Gemini copy. The scanner keeps full resolution;
# Gemini only needs enough to locate product labels, not decode barcode bars.
DEFAULT_GEMINI_MAX_DIMENSION = 1600


def load_normalized_image(
    image_path: str | os.PathLike[str],
    *,
    max_dimension: int = DEFAULT_GEMINI_MAX_DIMENSION,
) -> NormalizedImage:
    """
    Load ``image_path`` and return an EXIF-normalized, resized RGB JPEG.

    The image is resized so neither dimension exceeds ``max_dimension`` (aspect
    ratio preserved). ``original_width``/``original_height`` record the
    full-resolution dimensions so callers can scale Gemini bounding boxes back
    to the scanner's coordinate space.

    Raises:
        FileNotFoundError: the path does not exist.
        ValueError: the path is not a regular file or is empty, or
            ``max_dimension`` is not positive.
        PIL.UnidentifiedImageError: Pillow cannot decode the image.
    """
    if max_dimension <= 0:
        raise ValueError("max_dimension must be positive")

    path = Path(image_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Image does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Image path is not a regular file: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Image file is empty: {path}")

    with Image.open(path) as source:
        normalized = ImageOps.exif_transpose(source)
        if normalized.mode != "RGB":
            normalized = normalized.convert("RGB")

        original_width, original_height = normalized.size

        # Resize for Gemini — thumbnail preserves aspect ratio and never
        # enlarges. The scanner keeps full resolution independently.
        if original_width > max_dimension or original_height > max_dimension:
            normalized.thumbnail(
                (max_dimension, max_dimension),
                Image.Resampling.LANCZOS,
            )

        width, height = normalized.size
        buffer = BytesIO()
        normalized.save(buffer, format="JPEG", quality=85, optimize=True)

    return NormalizedImage(
        data=buffer.getvalue(),
        mime_type="image/jpeg",
        width=width,
        height=height,
        original_width=original_width,
        original_height=original_height,
    )


def _load_image(image_path: str | os.PathLike[str]) -> tuple[Path, bytes, str]:
    """
    Backward-compatible loader: returns (path, normalized_jpeg_bytes, mime_type).

    All Gemini audit functions consume EXIF-normalized RGB JPEG bytes. The
    image is resized for Gemini efficiency; coordinate conversion back to
    full-resolution scanner space is handled in ``audit_shoebox_labels``.
    """
    path = Path(image_path).expanduser().resolve()
    normalized = load_normalized_image(path)
    return path, normalized.data, normalized.mime_type


def audit_shoebox_image(
    image_path: str | os.PathLike[str],
    *,
    api_key: str | None = None,
    model: str | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
) -> ShoeboxImageAudit:
    """
    Audit a local shoebox image with Gemini and return validated structured data.

    Args:
        image_path:
            Path to a JPEG, PNG, WebP, HEIC, or HEIF image.
        api_key:
            Gemini API key. Defaults to GEMINI_API_KEY.
        model:
            Gemini model name. Defaults to GEMINI_MODEL, then
            "gemini-2.5-flash".
        max_retries:
            Number of retries after the initial request.
        retry_delay_seconds:
            Initial delay between retries. Exponential backoff is applied.

    Returns:
        A validated ShoeboxImageAudit Pydantic object.

    Raises:
        FileNotFoundError:
            The image path does not exist.
        ValueError:
            The input or configuration is invalid.
        ShoeboxAuditError:
            Gemini did not return a valid structured result.
    """
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    if retry_delay_seconds < 0:
        raise ValueError("retry_delay_seconds must be >= 0")

    # Best-effort .env load so a CLI run from the repo root picks up
    # GEMINI_API_KEY / GEMINI_MODEL without an explicit `source .env`.
    # No-op when no .env is present; explicit api_key/env vars still win.
    if api_key is None:
        load_dotenv()

    resolved_api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not resolved_api_key:
        raise ValueError(
            "Missing Gemini API key. Pass api_key=... or set GEMINI_API_KEY."
        )

    resolved_model = model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    path, image_bytes, mime_type = _load_image(image_path)

    client = genai.Client(api_key=resolved_api_key)
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=resolved_model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    AUDIT_PROMPT,
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=_gemini_compatible_schema(ShoeboxImageAudit),
                    temperature=0,
                ),
            )

            if response.parsed is not None:
                if isinstance(response.parsed, ShoeboxImageAudit):
                    return response.parsed
                return ShoeboxImageAudit.model_validate(response.parsed)

            if response.text:
                return ShoeboxImageAudit.model_validate_json(response.text)

            raise ShoeboxAuditError(
                f"Gemini returned neither parsed output nor text for '{path.name}'."
            )

        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break

            delay = retry_delay_seconds * (2**attempt)
            if delay > 0:
                time.sleep(delay)

    raise ShoeboxAuditError(
        f"Gemini visual audit failed for '{path.name}' after "
        f"{max_retries + 1} attempt(s): {last_error}"
    ) from last_error


def audit_shoebox_counts(
    image_path: str | os.PathLike[str],
    *,
    api_key: str | None = None,
    model: str | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
) -> BoxAuditCounts:
    """
    Fast minimal audit: return only the four barcode-label counts.

    Much faster than ``audit_shoebox_image`` because the schema is tiny and the
    prompt does not ask for bounding boxes, OCR text, or per-box observations.
    """
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    if retry_delay_seconds < 0:
        raise ValueError("retry_delay_seconds must be >= 0")

    if api_key is None:
        load_dotenv()

    resolved_api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not resolved_api_key:
        raise ValueError(
            "Missing Gemini API key. Pass api_key=... or set GEMINI_API_KEY."
        )

    resolved_model = model or os.getenv("GEMINI_MODEL", DEFAULT_COUNTS_MODEL)
    path, image_bytes, mime_type = _load_image(image_path)

    client = genai.Client(api_key=resolved_api_key)
    last_error: Exception | None = None

    # flash-lite does not support thinking_budget=0; only disable thinking on
    # full thinking models (flash, pro).
    config_kwargs: dict[str, object] = dict(
        response_mime_type="application/json",
        response_schema=_gemini_compatible_schema(BoxAuditCounts),
        temperature=0,
    )
    if "lite" not in resolved_model:
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=resolved_model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    COUNTS_PROMPT,
                ],
                config=types.GenerateContentConfig(**config_kwargs),
            )

            if response.parsed is not None:
                if isinstance(response.parsed, BoxAuditCounts):
                    return response.parsed
                return BoxAuditCounts.model_validate(response.parsed)

            if response.text:
                return BoxAuditCounts.model_validate_json(response.text)

            raise ShoeboxAuditError(
                f"Gemini returned neither parsed output nor text for '{path.name}'."
            )

        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break

            delay = retry_delay_seconds * (2**attempt)
            if delay > 0:
                time.sleep(delay)

    raise ShoeboxAuditError(
        f"Gemini counts audit failed for '{path.name}' after "
        f"{max_retries + 1} attempt(s): {last_error}"
    ) from last_error


def _convert_spatial_audit_to_pixels(
    audit: SpatialLabelAudit,
    *,
    original_width: int,
    original_height: int,
) -> SpatialLabelAuditPixels:
    """
    Convert a Gemini-facing ``SpatialLabelAudit`` (normalized 0..1000) directly
    to original full-resolution pixel coordinates.

    Gemini's normalized coordinates are resolution-independent — they represent
    relative positions in the image, not pixels in any particular raster. So
    regardless of what resize was applied to the Gemini copy, we convert
    directly to the original frame:

        x = round(normalized * original_dimension / 1000)

    This avoids the unnecessary intermediate step through resized-image pixels
    and reduces rounding error.
    """
    pixel_labels: list[SpatialLabelObservationPixels] = []
    for obs in audit.labels:
        label_px = clamp_bbox(
            normalized_to_pixels(
                top=obs.label_bbox.top,
                left=obs.label_bbox.left,
                bottom=obs.label_bbox.bottom,
                right=obs.label_bbox.right,
                image_width=original_width,
                image_height=original_height,
            ),
            image_width=original_width,
            image_height=original_height,
        )
        barcode_px: PixelBoundingBox | None = None
        if obs.barcode_bbox is not None:
            barcode_px = clamp_bbox(
                normalized_to_pixels(
                    top=obs.barcode_bbox.top,
                    left=obs.barcode_bbox.left,
                    bottom=obs.barcode_bbox.bottom,
                    right=obs.barcode_bbox.right,
                    image_width=original_width,
                    image_height=original_height,
                ),
                image_width=original_width,
                image_height=original_height,
            )
        pixel_labels.append(
            SpatialLabelObservationPixels(
                label_index=obs.label_index,
                label_bbox=label_px,
                barcode_bbox=barcode_px,
                status=obs.status,
                confidence=obs.confidence,
            )
        )
    return SpatialLabelAuditPixels(
        image_width=original_width,
        image_height=original_height,
        labels=pixel_labels,
    )


def audit_shoebox_labels(
    image_path: str | os.PathLike[str],
    *,
    api_key: str | None = None,
    model: str | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
) -> SpatialLabelAuditPixels:
    """
    Spatial label audit: locate every visible product label and its barcode
    region, return pixel-space bounding boxes.

    This is the middle Gemini mode — faster than ``audit_shoebox_image`` (no
    OCR text, no per-box observations, no quality analysis) but more useful
    than ``audit_shoebox_counts`` because it returns spatial regions that can
    be matched to scanner detections.

    Gemini's normalized 0..1000 coordinates are resolution-independent, so they
    are converted directly to the original full-resolution pixel frame and
    clamped. Downstream code never sees the normalized form — all pixel boxes
    are in the scanner's coordinate space.

    The Gemini copy is resized to ``DEFAULT_GEMINI_MAX_DIMENSION`` (1600px) only
    if the original exceeds that, to reduce encoding time, request size, and
    Gemini image-processing work on large phone photos. The scanner keeps full
    resolution independently. The resize does not affect the coordinate system
    because Gemini's normalized coordinates are converted directly to original
    dimensions.

    Returns:
        A ``SpatialLabelAuditPixels`` with pixel-space label/barcode boxes in
        the original full-resolution coordinate frame.

    Raises:
        FileNotFoundError: the image path does not exist.
        ValueError: the input or configuration is invalid.
        ShoeboxAuditError: Gemini did not return a valid structured result.
    """
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    if retry_delay_seconds < 0:
        raise ValueError("retry_delay_seconds must be >= 0")

    if api_key is None:
        load_dotenv()

    resolved_api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not resolved_api_key:
        raise ValueError(
            "Missing Gemini API key. Pass api_key=... or set GEMINI_API_KEY."
        )

    resolved_model = model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    path = Path(image_path).expanduser().resolve()
    normalized = load_normalized_image(path)
    image_bytes = normalized.data
    mime_type = normalized.mime_type

    client = genai.Client(api_key=resolved_api_key)
    last_error: Exception | None = None

    config_kwargs: dict[str, object] = dict(
        response_mime_type="application/json",
        response_schema=_gemini_compatible_schema(SpatialLabelAudit),
        temperature=0,
    )
    if "lite" not in resolved_model:
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=resolved_model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    SPATIAL_LABEL_PROMPT,
                ],
                config=types.GenerateContentConfig(**config_kwargs),
            )

            if response.parsed is not None:
                if isinstance(response.parsed, SpatialLabelAudit):
                    audit = response.parsed
                else:
                    audit = SpatialLabelAudit.model_validate(response.parsed)
            elif response.text:
                audit = SpatialLabelAudit.model_validate_json(response.text)
            else:
                raise ShoeboxAuditError(
                    f"Gemini returned neither parsed output nor text for '{path.name}'."
                )

            return _convert_spatial_audit_to_pixels(
                audit,
                original_width=normalized.original_width,
                original_height=normalized.original_height,
            )

        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break

            delay = retry_delay_seconds * (2**attempt)
            if delay > 0:
                time.sleep(delay)

    raise ShoeboxAuditError(
        f"Gemini spatial label audit failed for '{path.name}' after "
        f"{max_retries + 1} attempt(s): {last_error}"
    ) from last_error


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a structured Gemini visual audit on a shoebox image."
    )
    parser.add_argument("image_path", help="Path to the image to audit.")
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Gemini model name. Defaults to GEMINI_MODEL or "
            f"{DEFAULT_MODEL!r}."
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Retries after the first request. Default: {DEFAULT_MAX_RETRIES}.",
    )
    args = parser.parse_args()

    try:
        result = audit_shoebox_image(
            args.image_path,
            model=args.model,
            max_retries=args.max_retries,
        )
    except (FileNotFoundError, ValueError, ShoeboxAuditError) as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")

    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

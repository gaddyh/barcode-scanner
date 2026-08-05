"""
analyze.py

Product API for the barcode-scanning pipeline — the clean happy path.

``analyze_image()`` runs two branches in parallel on one image:

1. **Deterministic scanner** — decodes barcode values with pixel bounding boxes.
2. **Gemini Flash spatial audit** — locates every visible product label and its
   barcode region.

The two are joined with containment-based reconciliation and returned as a
product-shaped JSON dict with an explicit ``outcome`` and three arrays:

- ``found``      — barcodes matched to a Gemini product label.
- ``missing``    — Gemini product labels with no decoded barcode, each with pixel
  locations (``label_bbox`` / ``barcode_bbox``) so the client can ask the user
  to re-photograph those specific regions.
- ``unassigned`` — barcodes the scanner decoded but Gemini did not match to any
  label (diagnostic / audit purposes).

The client switches on ``outcome``:

    result = analyze_image(photo)

    if result["outcome"] == "complete":
        # every label has a decoded barcode — create the draft order
    elif result["outcome"] == "needs_better_photo":
        # show the missing regions to the user — ``annotated_image_b64``
        # is a PNG with red circles around the regions to re-photograph,
        # and ``message`` is a human-readable prompt.
    else:  # "retryable_error"
        # retry the service (Gemini failure, not the user's fault)
"""

from __future__ import annotations

import base64
import io
import logging
import os
import tempfile
from pathlib import Path

from PIL import (
    Image,
    ImageDraw,
    ImageFont,
    ImageOps,
    UnidentifiedImageError,
)

from app.services.barcode_scanner import BarcodeScanner
from app.services.gemini_box_audit import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY_SECONDS,
)
from app.services.pipeline import pipeline_path

logger = logging.getLogger(__name__)

ImageInput = bytes | Path | str

# Annotated preview images are display-only; cap the longest side so the
# base64 payload stays reasonable for HTTP / WhatsApp transport.
ANNOTATION_MAX_DIMENSION = 1600


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_image(
    image: ImageInput,
    *,
    scanner: BarcodeScanner | None = None,
    model: str | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
) -> dict[str, object]:
    """Run the pipeline on one image and return a product-shaped result.

    Args:
        image: Raw image bytes, or a path (``Path`` / ``str``) to an image file.
        scanner: Optional pre-constructed ``BarcodeScanner``. A new one is
            created if ``None``.
        model: Gemini model name. Defaults to ``GEMINI_MODEL`` env var or the
            module default.
        max_retries: Gemini retry count after the first attempt.
        retry_delay_seconds: Base delay between retries (exponential backoff).

    Returns:
        A JSON-serializable dict with ``outcome``, ``found``, ``missing``,
        ``unassigned``, and ``summary``. On ``needs_better_photo`` it also
        includes ``annotated_image_b64`` (base64 PNG with red circles around
        the missing regions), ``annotated_image_width`` /
        ``annotated_image_height`` (dimensions of the possibly-downscaled
        annotated PNG), and ``message`` (human-readable prompt). See module
        docstring for the full schema.
    """
    own_scanner = scanner is None
    if own_scanner:
        scanner = BarcodeScanner()

    # Resolve input to a path. Bytes are written to a temp file because
    # audit_shoebox_labels needs to re-open the image by path.
    cleanup_path: str | None = None
    try:
        if isinstance(image, (bytes, bytearray)):
            path = _write_temp_image(bytes(image))
            cleanup_path = str(path)
        else:
            path = Path(image).expanduser().resolve()

        # Read image dimensions early (before pipeline_path) so they are
        # available even when the audit fails or the image is invalid.
        try:
            image_width, image_height = _image_dimensions(path)
        except (OSError, UnidentifiedImageError, ValueError):
            image_width, image_height = 0, 0

        summary = pipeline_path(
            path,
            scanner,
            model=model,
            max_retries=max_retries,
            retry_delay_seconds=retry_delay_seconds,
        )
        result = _reshape(summary, image_width, image_height, image_path=path)
        logger.info(
            "analyze_image result: outcome=%s found=%d missing=%d unassigned=%d "
            "image=%dx%d",
            result.get("outcome"),
            result.get("summary", {}).get("found_count", 0),
            result.get("summary", {}).get("missing_count", 0),
            result.get("summary", {}).get("unassigned_count", 0),
            image_width,
            image_height,
        )
        for item in result.get("found", []):
            logger.info(
                "  found: label=%s value=%s format=%s",
                item.get("label_index"),
                item.get("barcode_value"),
                item.get("barcode_format"),
            )
        return result
    finally:
        if cleanup_path is not None:
            try:
                os.unlink(cleanup_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _write_temp_image(data: bytes) -> Path:
    """Write image bytes to a temporary file and return its path.

    Uses ``delete=False`` because ``audit_shoebox_labels`` re-opens the file
    by path. The caller is responsible for cleaning up.
    """
    suffix = ".jpg"
    tmp = tempfile.NamedTemporaryFile(
        suffix=suffix, delete=False, mode="wb"
    )
    try:
        tmp.write(data)
    finally:
        tmp.close()
    return Path(tmp.name)


def _image_dimensions(path: Path) -> tuple[int, int]:
    """Read image width/height from the header (cheap, no full decode)."""
    with Image.open(path) as img:
        return img.size


def _reshape(
    summary: dict[str, object],
    image_width: int = 0,
    image_height: int = 0,
    *,
    image_path: Path | None = None,
) -> dict[str, object]:
    """Reshape a pipeline_path summary into the product schema."""
    scan_status = summary.get("scan_status")
    audit_status = summary.get("audit_status")
    scan_ok = scan_status in ("found", "not_found")
    audit_ok = audit_status == "ok"

    # --- Error / retryable cases -------------------------------------------
    if not scan_ok:
        scan_error = summary.get("scan_error", {})
        return {
            "ok": False,
            "outcome": "retryable_error",
            "audit_available": False,
            "image_width": image_width,
            "image_height": image_height,
            "found": [],
            "missing": [],
            "unassigned": [],
            "summary": {
                "visible_label_count": 0,
                "found_count": 0,
                "missing_count": 0,
                "unassigned_count": 0,
                "all_found": False,
            },
            "error": scan_error,
        }

    # --- Audit unavailable: all decoded barcodes go to unassigned ----------
    if not audit_ok:
        scanner_detections = summary.get("scanner_detections", [])
        unassigned = [
            _detection_to_unassigned(d) for d in scanner_detections
        ]
        return {
            "ok": True,
            "outcome": "retryable_error",
            "audit_available": False,
            "image_width": image_width,
            "image_height": image_height,
            "found": [],
            "missing": [],
            "unassigned": unassigned,
            "summary": {
                "visible_label_count": 0,
                "found_count": 0,
                "missing_count": 0,
                "unassigned_count": len(unassigned),
                "all_found": False,
            },
            "error": summary.get("audit_error", {}),
        }

    # --- Audit available: build found / missing / unassigned ---------------
    gemini_labels = summary.get("gemini_labels", [])
    visible_label_count = len(gemini_labels)
    scanner_detections = summary.get("scanner_detections", [])
    reconciliation = summary.get("reconciliation", {})

    # Build lookup dictionaries (do not assume contiguous label indexes).
    labels_by_index: dict[int, dict] = {
        label["label_index"]: label for label in gemini_labels
    }

    matches = reconciliation.get("matches", [])
    unmatched_labels = reconciliation.get("unmatched_labels", [])
    unassigned_detections = reconciliation.get(
        "unassigned_scanner_detections", []
    )

    # --- found ---
    found: list[dict[str, object]] = []
    for match in matches:
        label_index = match["label_index"]
        det_index = match["scanner_detection_index"]
        barcode_value = match["barcode_value"]
        match_basis = match["match_basis"]

        # Validate detection index exists.
        if det_index < 0 or det_index >= len(scanner_detections):
            continue
        detection = scanner_detections[det_index]
        barcode_format = detection.get("format")
        barcode_bbox = detection.get("bounding_box")

        # Validate label exists.
        label = labels_by_index.get(label_index)
        if label is None:
            continue
        label_bbox = label.get("label_bbox")

        found.append(
            {
                "label_index": label_index,
                "barcode_value": barcode_value,
                "barcode_format": barcode_format,
                "barcode_bbox": barcode_bbox,
                "label_bbox": label_bbox,
                "match_basis": match_basis,
            }
        )

    # --- missing ---
    missing: list[dict[str, object]] = []
    for ul in unmatched_labels:
        missing.append(
            {
                "label_index": ul["label_index"],
                "status": ul["status"],
                "label_bbox": ul.get("label_bbox"),
                "barcode_bbox": ul.get("barcode_bbox"),
            }
        )

    # --- unassigned ---
    unassigned: list[dict[str, object]] = [
        _detection_to_unassigned(d) for d in unassigned_detections
    ]

    # --- outcome ---
    missing_count = len(missing)
    found_count = len(found)
    unassigned_count = len(unassigned)
    all_found = (
        audit_ok
        and visible_label_count > 0
        and missing_count == 0
    )

    if missing_count > 0 or visible_label_count == 0:
        outcome = "needs_better_photo"
    else:
        outcome = "complete"

    result: dict[str, object] = {
        "ok": True,
        "outcome": outcome,
        "audit_available": True,
        "image_width": image_width,
        "image_height": image_height,
        "found": found,
        "missing": missing,
        "unassigned": unassigned,
        "summary": {
            "visible_label_count": visible_label_count,
            "found_count": found_count,
            "missing_count": missing_count,
            "unassigned_count": unassigned_count,
            "all_found": all_found,
        },
    }

    # --- annotated preview for needs_better_photo -------------------------
    # Draw red circles around the missing regions so the user can see where
    # to re-photograph without interpreting pixel coordinates. Render failure
    # never changes the outcome — the field is just omitted.
    if outcome == "needs_better_photo":
        if visible_label_count == 0:
            result["message"] = (
                "No barcode labels were identified. "
                "Please take a closer, well-lit photo."
            )
        else:
            result["message"] = (
                "Please photograph the marked barcode area more closely."
            )
        if image_path is not None:
            try:
                b64, aw, ah = _render_missing_annotation(image_path, missing)
                result["annotated_image_b64"] = b64
                result["annotated_image_width"] = aw
                result["annotated_image_height"] = ah
            except Exception:
                logger.exception(
                    "Failed to render missing-region annotation for %s",
                    image_path,
                )

    return result


def _detection_to_unassigned(detection: dict) -> dict[str, object]:
    """Convert a scanner detection dict to the unassigned product shape."""
    return {
        "barcode_value": detection.get("value"),
        "barcode_format": detection.get("format"),
        "barcode_bbox": detection.get("bounding_box"),
    }


def _render_missing_annotation(
    image_path: Path,
    missing: list[dict[str, object]],
) -> tuple[str, int, int]:
    """Render an annotated preview PNG marking the missing regions.

    The pipeline's bounding boxes are in EXIF-normalized pixel space, so the
    image is opened with ``ImageOps.exif_transpose`` before drawing — otherwise
    circles would land in the wrong place on rotated phone photos.

    A red circle is drawn around each missing label's ``barcode_bbox``
    (falling back to ``label_bbox`` when no barcode box is present) with the
    label index number beside it. The result is downscaled to
    ``ANNOTATION_MAX_DIMENSION`` on its longest side for transport.

    Returns a ``(base64_png_str, width, height)`` tuple where width/height
    are the dimensions of the possibly-downscaled annotated image.
    """
    with Image.open(image_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    padding = max(8, round(image.width * 0.02))
    stroke_width = max(3, image.width // 400)

    for item in missing:
        box = item.get("barcode_bbox") or item.get("label_bbox")
        if not box:
            continue

        center_x = (box["x1"] + box["x2"]) / 2
        center_y = (box["y1"] + box["y2"]) / 2
        radius = max(
            (box["x2"] - box["x1"]) / 2,
            (box["y2"] - box["y1"]) / 2,
        ) + padding
        circle = (
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
        )
        draw.ellipse(circle, outline="red", width=stroke_width)

        # Label index beside/above the circle so it doesn't cover the barcode.
        idx_text = str(item.get("label_index", "?"))
        text_x = max(0, int(circle[0]))
        text_y = max(0, int(circle[1] - stroke_width * 6))
        draw.text((text_x, text_y), idx_text, fill="red", font=font)

    # Downscale for transport (display-only — original pixel fidelity not needed).
    image.thumbnail(
        (ANNOTATION_MAX_DIMENSION, ANNOTATION_MAX_DIMENSION),
        Image.Resampling.LANCZOS,
    )

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    return b64, image.width, image.height

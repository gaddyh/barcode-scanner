"""
analyze.py

Product API for the barcode-scanning pipeline.

``analyze_image()`` runs the full pipeline (deterministic scan + dual Gemini
audit + reconciliation + Gemini-guided recovery) on one image and returns a
clean product-shaped JSON dict with an explicit ``outcome`` and three arrays:

- ``found``    — barcodes matched to a Gemini product label.
- ``missing``  — Gemini product labels with no decoded barcode, each with pixel
  locations (``label_bbox`` / ``barcode_bbox``) so the client can ask the user
  to re-photograph those specific regions.
- ``unassigned`` — barcodes the scanner decoded but Gemini did not match to any
  label (diagnostic / audit purposes).

The client switches on ``outcome``:

    result = analyze_image(photo)

    if result["outcome"] == "complete":
        # every label has a decoded barcode — create the draft order
    elif result["outcome"] == "needs_better_photo":
        # show the missing regions to the user
    else:  # "retryable_error"
        # retry the service (Gemini failure, not the user's fault)

This is the hot path: a user uploads an image and either gets all barcodes
back, is asked to re-photograph specific missing locations, or the client
retries the service.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.services.barcode_scanner import BarcodeScanner
from app.services.gemini_box_audit import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY_SECONDS,
)
from app.services.pipeline import pipeline_path

ImageInput = bytes | Path | str


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
    dual_audit: bool = True,
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
        dual_audit: Run two Gemini audits in parallel and pick the better one.

    Returns:
        A JSON-serializable dict with ``outcome``, ``found``, ``missing``,
        ``unassigned``, and ``summary``. See module docstring for the full
        schema.
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
            recovery_debug_dir=None,
            dual_audit=dual_audit,
        )
        return _reshape(summary, image_width, image_height)
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

    return {
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


def _detection_to_unassigned(detection: dict) -> dict[str, object]:
    """Convert a scanner detection dict to the unassigned product shape."""
    return {
        "barcode_value": detection.get("value"),
        "barcode_format": detection.get("format"),
        "barcode_bbox": detection.get("bounding_box"),
    }

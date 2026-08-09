"""
pipeline.py

Clean happy-path orchestration for the barcode-scanning pipeline.

Runs two independent branches **in parallel** on one image:

1. **Deterministic scanner** (zxing-cpp + OpenCV label fallback) — decodes
   barcode values with full-resolution pixel bounding boxes.
2. **Gemini Flash spatial audit** — locates every visible product label and its
   barcode region, returns pixel-space bounding boxes.

The two results are joined with a single containment-based reconciliation
(``match_scanner_to_labels``) that assigns each scanner detection to the Gemini
label whose barcode region contains it. No dual-audit, no recovery, no crop
retries — just one scan + one audit + one match.

The summary returned by ``pipeline_path`` is consumed by ``analyze_image`` to
produce the product-shaped response (``complete`` / ``needs_better_photo`` /
``retryable_error``).

Service layer used by:
- ``src.cli_app`` (CLI presentation)
- ``src.ingest.analyze`` (product API)
- ``tests.eval.runner`` (LangSmith offline evaluation)

No argument parsing or CLI presentation code lives here.
"""

from __future__ import annotations

import contextvars
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, is_dataclass
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image, ImageOps, UnidentifiedImageError

from src.ingest.scanner import SCANNER_VERSION, BarcodeScanner
from src.ingest.vision import (
    DEFAULT_MODEL,
    VISION_PROMPT_VERSION,
    ShoeboxAuditError,
    audit_shoebox_labels,
)
from src.ingest.reconciliation import match_scanner_to_labels
from src.observability.tracing import emit_pipeline_event
from src.runtime.events import EventType

# Load .env early so LANGSMITH_* vars are set before langsmith is imported.
load_dotenv()

logger = logging.getLogger(__name__)

# langsmith is optional — tracing is enabled when LANGSMITH_TRACING=true.
_TRACING = os.getenv("LANGSMITH_TRACING", "").lower() in ("true", "1", "yes")
if _TRACING:
    import langsmith as ls
    from langsmith import traceable
else:
    # no-op decorator fallback when tracing is disabled.
    def traceable(*args, **kwargs):  # type: ignore[misc]
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def _wrap(fn):
            return fn

        return _wrap

    class _NoRun:
        metadata: dict = {}

    def get_current_run_tree():  # type: ignore[misc]
        return None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _to_jsonable(value: object) -> object:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------


@traceable(run_type="tool", name="barcode_scan")
def scan_path(path: Path, scanner: BarcodeScanner) -> dict[str, object]:
    """Deterministic barcode scan of one image file."""
    try:
        image_bytes = path.read_bytes()
    except OSError as exc:
        return {
            "path": str(path),
            "status": "error",
            "error": {"code": "unreadable_file", "message": str(exc)},
        }

    try:
        barcodes = scanner.scan_bytes(image_bytes)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return {
            "path": str(path),
            "status": "error",
            "error": {"code": "invalid_image", "message": str(exc)},
        }

    return {
        "path": str(path),
        "status": "found" if barcodes else "not_found",
        "count": len(barcodes),
        "barcodes": [_to_jsonable(b) for b in barcodes],
    }


# ---------------------------------------------------------------------------
# Traced audit wrapper
# ---------------------------------------------------------------------------


@traceable(run_type="tool", name="gemini_audit")
def _traced_audit(
    path: Path,
    *,
    model: str | None,
    max_retries: int,
    retry_delay_seconds: float,
) -> dict[str, object]:
    """Traced wrapper for the Gemini spatial label audit."""
    try:
        spatial = audit_shoebox_labels(
            path,
            model=model,
            max_retries=max_retries,
            retry_delay_seconds=retry_delay_seconds,
        )
    except FileNotFoundError as exc:
        return {
            "status": "error",
            "error": {"type": "FileNotFoundError", "message": str(exc)},
        }
    except (ValueError, ShoeboxAuditError) as exc:
        return {
            "status": "error",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
    return {"status": "ok", "spatial": spatial.model_dump(mode="json")}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@traceable(run_type="chain", name="pipeline")
def pipeline_path(
    path: Path,
    scanner: BarcodeScanner,
    *,
    model: str | None,
    max_retries: int,
    retry_delay_seconds: float,
) -> dict[str, object]:
    """Run deterministic scan and Gemini audit in parallel, return combined summary.

    Both branches run concurrently in a thread pool. When both succeed, scanner
    detections are matched to Gemini labels with containment-based
    reconciliation. The summary is reshaped by ``analyze_image`` into the
    product response.
    """
    # Stamp component versions on the pipeline span.
    _run = ls.get_current_run_tree() if _TRACING else None
    if _run is not None:
        _run.metadata.update(
            {
                "scanner_version": SCANNER_VERSION,
                "vision_prompt_version": VISION_PROMPT_VERSION,
                "vision_model": model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
                "recovery_version": RECOVERY_VERSION,
            }
        )

    def _do_scan() -> dict[str, object]:
        return scan_path(path, scanner)

    def _do_audit() -> dict[str, object]:
        return _traced_audit(
            path,
            model=model,
            max_retries=max_retries,
            retry_delay_seconds=retry_delay_seconds,
        )

    # Copy the current context so langsmith trace context propagates to threads.
    scan_ctx = contextvars.copy_context()
    audit_ctx = contextvars.copy_context()

    with ThreadPoolExecutor(max_workers=2) as pool:
        scan_future = pool.submit(scan_ctx.run, _do_scan)
        audit_future = pool.submit(audit_ctx.run, _do_audit)
        scan_result = scan_future.result()
        audit_result = audit_future.result()

    # Log scanner detections for debugging (values + formats).
    scan_barcodes = scan_result.get("barcodes", [])
    if scan_barcodes:
        scan_values = [
            (b.get("value"), b.get("format")) for b in scan_barcodes  # type: ignore[union-attr]
        ]
        logger.info(
            "Scanner detected %d barcode(s): %s",
            len(scan_values),
            scan_values,
        )
    else:
        logger.info("Scanner detected 0 barcodes (status=%s)", scan_result.get("status"))

    logger.info("Gemini audit status: %s", audit_result.get("status"))

    scan_ok = scan_result.get("status") in ("found", "not_found")
    audit_ok = audit_result.get("status") == "ok"

    # Emit structured events for each completed phase.
    emit_pipeline_event(
        EventType.SCAN_COMPLETED,
        scanner_count=scan_result.get("count", 0),
        scanner_status=scan_result.get("status"),
    )
    emit_pipeline_event(
        EventType.AUDIT_COMPLETED,
        vision_count=len(audit_result.get("spatial", {}).get("labels", [])) if audit_ok else 0,
        audit_status=audit_result.get("status"),
    )

    summary: dict[str, object] = {
        "path": str(path),
        "scan_status": scan_result.get("status"),
        "audit_status": audit_result.get("status"),
    }

    barcodes: list[dict] = []
    if scan_ok:
        barcodes = scan_result.get("barcodes", [])  # type: ignore[assignment]
        values = [b["value"] for b in barcodes]  # type: ignore[index]
        summary["decoded_count"] = scan_result.get("count", 0)
        summary["unique_values"] = sorted(set(values))
        summary["unique_value_count"] = len(set(values))
        summary["scanner_detections"] = barcodes

    if audit_ok:
        spatial = audit_result.get("spatial", {})
        labels = spatial.get("labels", [])
        summary["visible_labels"] = len(labels)
        summary["clear_labels"] = sum(
            1 for label in labels if label.get("status") == "clear"
        )
        summary["gemini_labels"] = labels

    if scan_ok and audit_ok:
        spatial = audit_result.get("spatial", {})
        labels = spatial.get("labels", [])
        image_width = spatial["image_width"]
        image_height = spatial["image_height"]

        reconciliation = match_scanner_to_labels(
            barcodes,
            labels,
            image_width=image_width,
            image_height=image_height,
        )

        emit_pipeline_event(
            EventType.RECONCILIATION_COMPLETED,
            matched_count=reconciliation.matched_label_count,
            unmatched_count=len(reconciliation.unmatched_labels),
        )

        # --- Gemini-guided recovery ---------------------------------------
        # When labels remain unmatched after reconciliation, crop each
        # missing label's barcode region from the full-resolution image and
        # scan it aggressively (including a 90° rotation attempt). Any newly
        # decoded barcodes are merged into the scanner detections and
        # reconciliation is re-run.
        recovery_attempted = False
        recovery_labels_tried = 0
        recovery_barcodes_found = 0
        recovery_labels_resolved = 0

        if reconciliation.unmatched_labels:
            recovery_attempted = True
            recovery_labels_tried = len(reconciliation.unmatched_labels)
            matched_before = reconciliation.matched_label_count

            emit_pipeline_event(
                EventType.RECOVERY_STARTED,
                labels_to_try=recovery_labels_tried,
            )

            recovery_detections = _gemini_guided_recovery(
                path,
                scanner,
                reconciliation.unmatched_labels,
                image_width,
                image_height,
            )
            recovery_barcodes_found = len(recovery_detections)

            if recovery_detections:
                logger.info(
                    "Recovery decoded %d additional barcode(s)",
                    len(recovery_detections),
                )
                barcodes.extend(recovery_detections)
                summary["scanner_detections"] = barcodes
                summary["decoded_count"] = len(barcodes)
                values = [b["value"] for b in barcodes]  # type: ignore[index]
                summary["unique_values"] = sorted(set(values))
                summary["unique_value_count"] = len(set(values))

                # Re-run reconciliation with the augmented detections.
                reconciliation = match_scanner_to_labels(
                    barcodes,
                    labels,
                    image_width=image_width,
                    image_height=image_height,
                )
                recovery_labels_resolved = (
                    reconciliation.matched_label_count - matched_before
                )
                logger.info(
                    "Reconciliation after recovery: %d matched, %d unmatched",
                    len(reconciliation.matches),
                    len(reconciliation.unmatched_labels),
                )

        # Emit recovery metrics to LangSmith trace metadata and pipeline summary.
        run = ls.get_current_run_tree()
        if run is not None:
            run.metadata.update(
                {
                    "recovery_attempted": recovery_attempted,
                    "recovery_labels_tried": recovery_labels_tried,
                    "recovery_barcodes_found": recovery_barcodes_found,
                    "recovery_labels_resolved": recovery_labels_resolved,
                }
            )

        if recovery_attempted:
            emit_pipeline_event(
                EventType.RECOVERY_COMPLETED,
                barcodes_found=recovery_barcodes_found,
                labels_resolved=recovery_labels_resolved,
            )

        summary["recovery"] = {
            "attempted": recovery_attempted,
            "labels_tried": recovery_labels_tried,
            "barcodes_found": recovery_barcodes_found,
            "labels_resolved": recovery_labels_resolved,
        }

        summary["reconciliation"] = reconciliation.model_dump(mode="json")

        matches = reconciliation.matches
        unmatched = reconciliation.unmatched_labels
        logger.info(
            "Reconciliation: %d matched, %d unmatched",
            len(matches),
            len(unmatched),
        )
        for m in matches:
            logger.info(
                "  match: label=%s detection=%s barcode=%s basis=%s",
                m.label_index,
                m.scanner_detection_index,
                m.barcode_value,
                m.match_basis,
            )
        for u in unmatched:
            logger.info(
                "  unmatched: label=%s status=%s",
                u.label_index,
                u.status,
            )

        decoded = summary.get("decoded_count", 0)
        visible = summary.get("visible_labels", 0)
        summary["decoded_vs_visible"] = {
            "decoded": decoded,
            "visible": visible,
            "match": decoded == visible,
            "difference": decoded - visible,
            "matched_labels": reconciliation.matched_label_count,
            "all_labels_matched": reconciliation.all_labels_matched,
        }

    if not scan_ok:
        summary["scan_error"] = scan_result.get("error", {})
    if not audit_ok:
        summary["audit_error"] = audit_result.get("error", {})

    summary["ok"] = scan_ok and audit_ok
    return summary


# ---------------------------------------------------------------------------
# Gemini-guided recovery
# ---------------------------------------------------------------------------

# Version of the recovery logic (crop padding, scan_crop_with_recovery
# variants, rotation attempt). Bumped when the recovery algorithm changes.
RECOVERY_VERSION = "recovery-v1"

# Padding ratio for recovery crops — wider than the label fallback (0.12) to
# give the scanner more context around the barcode region.
_RECOVERY_PADDING_RATIO = 0.20


@traceable(run_type="chain", name="recovery")
def _gemini_guided_recovery(
    image_path: Path,
    scanner: BarcodeScanner,
    unmatched_labels: list,
    image_width: int,
    image_height: int,
) -> list[dict]:
    """Crop and scan unmatched label regions from the full-resolution image.

    For each unmatched Gemini label, extracts the ``barcode_bbox`` (or
    ``label_bbox`` as fallback) region with increased padding, and runs the
    aggressive crop-variant scanner including a 90° rotation attempt.

    Returns a list of detection dicts (same shape as ``scanner_detections``)
    suitable for merging into the existing detections list and re-running
    reconciliation.
    """
    try:
        source = Image.open(image_path)
        source = ImageOps.exif_transpose(source).convert("RGB")
    except Exception as exc:
        logger.warning("Recovery: could not open image %s: %s", image_path, exc)
        return []

    recovery_detections: list[dict] = []

    for ul in unmatched_labels:
        # Prefer barcode_bbox, fall back to label_bbox.
        bbox = ul.barcode_bbox or ul.label_bbox
        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            continue

        pad_x = max(20, round(w * _RECOVERY_PADDING_RATIO))
        pad_y = max(20, round(h * _RECOVERY_PADDING_RATIO))

        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(image_width, x2 + pad_x)
        cy2 = min(image_height, y2 + pad_y)

        crop = source.crop((cx1, cy1, cx2, cy2))

        logger.info(
            "Recovery: cropping label=%s bbox=(%d,%d,%d,%d) padded=(%d,%d,%d,%d) "
            "crop_size=%dx%d",
            ul.label_index,
            x1, y1, x2, y2,
            cx1, cy1, cx2, cy2,
            crop.width, crop.height,
        )

        detections = scanner.scan_crop_with_recovery(
            crop,
            offset_x=cx1,
            offset_y=cy1,
        )

        for det in detections:
            recovery_detections.append(
                {
                    "value": det.value,
                    "format": det.format,
                    "content_type": det.content_type,
                    "orientation": det.orientation,
                    "position": [
                        {"x": p.x, "y": p.y} for p in det.position
                    ],
                    "bounding_box": {
                        "x1": det.bounding_box.x1,
                        "y1": det.bounding_box.y1,
                        "x2": det.bounding_box.x2,
                        "y2": det.bounding_box.y2,
                    },
                }
            )

        if detections:
            logger.info(
                "Recovery: label=%s decoded %d barcode(s): %s",
                ul.label_index,
                len(detections),
                [d.value for d in detections],
            )

    return recovery_detections

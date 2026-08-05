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
- ``app.cli`` (CLI presentation)
- ``app.services.analyze`` (product API)
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
from PIL import UnidentifiedImageError

from app.services.barcode_scanner import BarcodeScanner
from app.services.gemini_box_audit import (
    ShoeboxAuditError,
    audit_shoebox_labels,
)
from app.services.spatial_reconciliation import match_scanner_to_labels

# Load .env early so LANGSMITH_* vars are set before langsmith is imported.
load_dotenv()

logger = logging.getLogger(__name__)

# langsmith is optional — tracing is enabled when LANGSMITH_TRACING=true.
_TRACING = os.getenv("LANGSMITH_TRACING", "").lower() in ("true", "1", "yes")
if _TRACING:
    from langsmith import traceable
else:
    # no-op decorator fallback when tracing is disabled.
    def traceable(*args, **kwargs):  # type: ignore[misc]
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def _wrap(fn):
            return fn

        return _wrap


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

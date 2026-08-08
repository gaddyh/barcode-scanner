"""ingest_one() — the canonical typed boundary for barcode ingest.

This is the strangler-pattern entry point. It calls the existing
``analyze_image()`` (which returns an untyped dict) and converts the
result into a typed ``IngestResult``. No existing code is modified —
this is a thin adapter that establishes the new boundary.

Later milestones will move the actual scanner/vision/reconciliation
implementation behind this interface, but the contract stays the same.

Usage::

    from src.ingest.service import ingest_one
    from src.runtime.context import RunContext

    ctx = RunContext(
        run_id=str(uuid4()),
        session_id=generate_upload_id(),
        user_id=None,
        source="cli",
    )
    result = ingest_one("./samples/photo.jpg", ctx)
    print(result.status, len(result.items))
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from app.services.analyze import analyze_image
from app.services.barcode_scanner import BarcodeScanner
from app.services.gemini_box_audit import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY_SECONDS,
)
from src.ingest.models import (
    DetectedItem,
    IngestResult,
    IngestStatus,
    Issue,
    RunMetrics,
)
from src.observability.tracing import emit_event, emit_metadata
from src.runtime.context import RunContext
from src.runtime.events import DomainEvent

logger = logging.getLogger(__name__)

ImageInput = bytes | Path | str


def ingest_one(
    image: ImageInput,
    context: RunContext,
    *,
    scanner: BarcodeScanner | None = None,
    model: str | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
) -> IngestResult:
    """Run the full ingest pipeline on one image and return a typed result.

    This is a thin adapter over the existing ``analyze_image()``. It calls
    the current implementation and converts the dict result into an
    ``IngestResult``. No behavioral change from calling ``analyze_image()``
    directly — just a typed boundary.

    Args:
        image: Raw image bytes, or a path (``Path`` / ``str``) to an image file.
        context: The ``RunContext`` for this execution (correlation, source, etc.).
        scanner: Optional pre-constructed ``BarcodeScanner``.
        model: Gemini model name override.
        max_retries: Gemini retry count after the first attempt.
        retry_delay_seconds: Base delay between retries (exponential backoff).

    Returns:
        A typed ``IngestResult`` with status, items, missing, unassigned,
        issues, and metrics.
    """
    t0 = time.perf_counter()

    # Emit IMAGE_RECEIVED event — always logged, appended to trace when tracing.
    emit_event(DomainEvent(
        type="IMAGE_RECEIVED",
        run_id=context.run_id,
        session_id=context.session_id,
        payload={"source": context.source, "image_type": type(image).__name__},
    ))

    raw: dict[str, Any] = analyze_image(
        image,
        scanner=scanner,
        model=model,
        max_retries=max_retries,
        retry_delay_seconds=retry_delay_seconds,
    )

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    result = _dict_to_ingest_result(raw, elapsed_ms, context)

    # Emit result metadata to the current LangSmith trace.
    emit_metadata(
        context,
        final_status=result.status.value,
        scanner_count=result.metrics.scanner_count,
        vision_count=result.metrics.vision_count,
        found_count=len(result.items),
        missing_count=len(result.missing),
        unassigned_count=len(result.unassigned),
        recovery_attempted=result.metrics.recovery_attempted,
        recovery_labels_tried=result.metrics.recovery_labels_tried,
        recovery_barcodes_found=result.metrics.recovery_barcodes_found,
        recovery_labels_resolved=result.metrics.recovery_labels_resolved,
        latency_ms=elapsed_ms,
    )

    # Emit INGEST_COMPLETED event.
    emit_event(DomainEvent(
        type="INGEST_COMPLETED",
        run_id=context.run_id,
        session_id=context.session_id,
        payload={
            "status": result.status.value,
            "found_count": len(result.items),
            "missing_count": len(result.missing),
            "elapsed_ms": elapsed_ms,
        },
    ))

    return result


# ---------------------------------------------------------------------------
# Dict → IngestResult conversion
# ---------------------------------------------------------------------------


def _dict_to_ingest_result(
    raw: dict[str, Any],
    elapsed_ms: int,
    context: RunContext,
) -> IngestResult:
    """Convert the legacy ``analyze_image()`` dict into a typed ``IngestResult``."""

    outcome = raw.get("outcome", "retryable_error")
    ok = raw.get("ok", True)
    audit_available = raw.get("audit_available", False)

    # --- Status mapping ---------------------------------------------------
    if outcome == "complete":
        status = IngestStatus.COMPLETE
    elif outcome == "needs_better_photo":
        status = IngestStatus.NEEDS_USER_INPUT
    elif ok:
        status = IngestStatus.NEEDS_RETRY
    else:
        status = IngestStatus.FAILED

    # --- Items (found) ----------------------------------------------------
    found = raw.get("found", [])
    items = [
        DetectedItem(
            label_index=f.get("label_index"),
            barcode_value=f.get("barcode_value", ""),
            barcode_format=f.get("barcode_format"),
            barcode_bbox=f.get("barcode_bbox"),
            label_bbox=f.get("label_bbox"),
            match_basis=f.get("match_basis"),
        )
        for f in found
    ]

    # --- Missing / unassigned ---------------------------------------------
    missing = raw.get("missing", [])
    unassigned = raw.get("unassigned", [])

    # --- Metrics ----------------------------------------------------------
    summary = raw.get("summary", {})
    recovery = summary.get("recovery", {})
    metrics = RunMetrics(
        elapsed_ms=elapsed_ms,
        scanner_count=summary.get("found_count", 0) + summary.get("unassigned_count", 0),
        vision_count=summary.get("visible_label_count", 0),
        recovery_attempted=recovery.get("attempted", False),
        recovery_labels_tried=recovery.get("labels_tried", 0),
        recovery_barcodes_found=recovery.get("barcodes_found", 0),
        recovery_labels_resolved=recovery.get("labels_resolved", 0),
    )

    # --- Issues (derived) -------------------------------------------------
    issues: list[Issue] = []

    if not ok:
        error = raw.get("error", {})
        issues.append(
            Issue(
                code="BAD_IMAGE" if outcome == "retryable_error" and not audit_available
                else "PIPELINE_ERROR",
                severity="error",
                message=str(error.get("message", "Unknown error")),
                evidence={"error": error},
            )
        )

    if audit_available and len(missing) > 0:
        issues.append(
            Issue(
                code="BARCODE_MISSING",
                severity="warning",
                message=f"{len(missing)} label(s) without a decoded barcode",
                evidence={"missing_count": len(missing), "label_indices": [
                    m.get("label_index") for m in missing
                ]},
            )
        )

    if audit_available and metrics.scanner_count != metrics.vision_count:
        issues.append(
            Issue(
                code="VISION_SCANNER_MISMATCH",
                severity="info",
                message=(
                    f"Scanner decoded {metrics.scanner_count} barcode(s) "
                    f"but vision found {metrics.vision_count} label(s)"
                ),
                evidence={
                    "scanner_count": metrics.scanner_count,
                    "vision_count": metrics.vision_count,
                },
            )
        )

    if metrics.recovery_attempted and metrics.recovery_labels_resolved == 0:
        issues.append(
            Issue(
                code="RECOVERY_FAILED",
                severity="info",
                message=(
                    f"Recovery tried {metrics.recovery_labels_tried} label(s) "
                    f"but resolved 0"
                ),
                evidence={
                    "labels_tried": metrics.recovery_labels_tried,
                    "barcodes_found": metrics.recovery_barcodes_found,
                },
            )
        )

    result = IngestResult(
        status=status,
        items=items,
        missing=missing,
        unassigned=unassigned,
        issues=issues,
        metrics=metrics,
        image_width=raw.get("image_width", 0),
        image_height=raw.get("image_height", 0),
        audit_available=audit_available,
        annotated_image_b64=raw.get("annotated_image_b64"),
        annotated_image_width=raw.get("annotated_image_width"),
        annotated_image_height=raw.get("annotated_image_height"),
        message=raw.get("message"),
        error=raw.get("error"),
    )

    logger.info(
        "ingest_one run_id=%s source=%s status=%s items=%d missing=%d "
        "unassigned=%d issues=%d elapsed_ms=%d",
        context.run_id,
        context.source,
        result.status,
        len(result.items),
        len(result.missing),
        len(result.unassigned),
        len(result.issues),
        elapsed_ms,
    )

    return result

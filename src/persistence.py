"""Persistence helper for the write path.

The caller (route handler, WhatsApp processor, CLI) owns the persistence
lifecycle. This module provides a simple async helper that wraps the
repository calls:

    await persist_run(repo, run_id, source, endpoint, result_dict, ...)

It converts the ``analyze_image()`` dict to an ``IngestResult`` and calls
``repo.complete_run()``. On failure, calls ``repo.fail_run()``.

``ingest_one()`` stays pure — it never imports this module.
"""

from __future__ import annotations

import logging
from typing import Any

from src.ingest.models import (
    DetectedItem,
    IngestResult,
    IngestStatus,
    Issue,
    RunMetrics,
)
from src.repository import NewRun, RunRepository

logger = logging.getLogger(__name__)


def dict_to_ingest_result(raw: dict[str, Any], elapsed_ms: int) -> IngestResult:
    """Convert an ``analyze_image()`` dict to a typed ``IngestResult``.

    This is the public version of ``_dict_to_ingest_result`` in
    ``src.ingest.service`` — no RunContext required, suitable for
    the web/WhatsApp paths that call ``analyze_image()`` directly.
    """
    outcome = raw.get("outcome", "retryable_error")
    ok = raw.get("ok", True)
    audit_available = raw.get("audit_available", False)

    if outcome == "complete":
        status = IngestStatus.COMPLETE
    elif outcome == "needs_better_photo":
        status = IngestStatus.NEEDS_USER_INPUT
    elif ok:
        status = IngestStatus.NEEDS_RETRY
    else:
        status = IngestStatus.FAILED

    items = [
        DetectedItem(
            label_index=f.get("label_index"),
            barcode_value=f.get("barcode_value", ""),
            barcode_format=f.get("barcode_format"),
            barcode_bbox=f.get("barcode_bbox"),
            label_bbox=f.get("label_bbox"),
            match_basis=f.get("match_basis"),
        )
        for f in raw.get("found", [])
    ]

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

    issues: list[Issue] = []
    if not ok:
        error = raw.get("error", {})
        issues.append(
            Issue(
                code="BAD_IMAGE" if outcome == "retryable_error" and not audit_available
                else error.get("code", "UNKNOWN_ERROR"),
                severity="error",
                message=error.get("message", "Unknown error"),
                evidence=error,
            )
        )
    if audit_available and metrics.scanner_count != metrics.vision_count:
        issues.append(
            Issue(
                code="VISION_SCANNER_MISMATCH",
                severity="info",
                message=f"Scanner decoded {metrics.scanner_count} barcode(s) "
                        f"but vision found {metrics.vision_count} label(s)",
                evidence={
                    "scanner_count": metrics.scanner_count,
                    "vision_count": metrics.vision_count,
                },
            )
        )
    if summary.get("missing_count", 0) > 0:
        issues.append(
            Issue(
                code="BARCODE_MISSING",
                severity="warning",
                message=f"{summary['missing_count']} label(s) without a decoded barcode",
                evidence={
                    "missing_count": summary["missing_count"],
                    "missing_labels": [
                        m.get("label_index") for m in raw.get("missing", [])
                    ],
                },
            )
        )
    if metrics.recovery_attempted and metrics.recovery_labels_resolved == 0:
        issues.append(
            Issue(
                code="RECOVERY_FAILED",
                severity="info",
                message=f"Recovery tried {metrics.recovery_labels_tried} label(s) "
                        f"but resolved {metrics.recovery_labels_resolved}",
                evidence={
                    "barcodes_found": metrics.recovery_barcodes_found,
                    "labels_tried": metrics.recovery_labels_tried,
                },
            )
        )

    return IngestResult(
        status=status,
        items=items,
        missing=raw.get("missing", []),
        unassigned=raw.get("unassigned", []),
        issues=issues,
        metrics=metrics,
        image_width=raw.get("image_width", 0),
        image_height=raw.get("image_height", 0),
        audit_available=audit_available,
        error=raw.get("error"),
    )


async def create_run(
    repo: RunRepository,
    *,
    run_id: str,
    source: str,
    endpoint: str | None = None,
    trace_id: str | None = None,
    session_id: str | None = None,
    filename: str | None = None,
    image_ref: str | None = None,
    upload_bytes: int | None = None,
    image_width: int | None = None,
    image_height: int | None = None,
    provider_message_id: str | None = None,
    sender: str | None = None,
) -> None:
    """Create a run row with status=pending."""
    await repo.create_run(NewRun(
        id=run_id,
        session_id=session_id,
        trace_id=trace_id,
        source=source,
        endpoint=endpoint,
        filename=filename,
        image_ref=image_ref,
        upload_bytes=upload_bytes,
        image_width=image_width,
        image_height=image_height,
        provider_message_id=provider_message_id,
        sender=sender,
    ))


async def complete_run_from_dict(
    repo: RunRepository,
    run_id: str,
    result: dict[str, Any],
    elapsed_ms: int,
    *,
    versions: dict[str, str] | None = None,
) -> None:
    """Convert dict result to IngestResult and persist atomically."""
    ingest_result = dict_to_ingest_result(result, elapsed_ms)
    await repo.complete_run(run_id, ingest_result, versions=versions)


async def fail_run(
    repo: RunRepository,
    run_id: str,
    error: Exception,
) -> None:
    """Mark a run as failed."""
    await repo.fail_run(run_id, error)

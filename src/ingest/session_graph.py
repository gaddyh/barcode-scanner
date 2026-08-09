"""SessionGraph — multi-image ingest session orchestration.

The SessionGraph wraps the ScanGraph (one-image pipeline) in a higher-level
state machine that accumulates results across multiple photos:

    Photo #1 → 12 visible, 11 found, 1 missing
    → session persists, status=active, "send photo of box 7"
    Photo #2 → scans the missing box
    → merge by barcode_value → 12/12 → session complete

Key design:

- **ScanGraph is a subgraph** — each image runs through the full scan + audit
  + reconcile + recovery pipeline. The SessionGraph calls it and merges the
  result.
- **Merge by barcode_value** — items found in multiple photos are deduplicated.
  Already-known barcodes are ignored; new barcodes resolve missing entries.
- **Expected count from first image** — the first audit's visible_label_count
  sets the target. Subsequent photos don't change it.
- **Checkpointing** — the session state is checkpointed to Postgres via
  thread_id=session_id. When the user sends photo #2, we load the existing
  session state and continue.

The SessionGraph is NOT a LangGraph StateGraph itself (yet) — it's a simpler
orchestration layer that calls ScanGraph + merge logic. The complexity is in
the merge, not in conditional routing. If we later need more complex session
logic (e.g. priority ordering, HITL approval), we can promote it to a full
StateGraph.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.ingest.scanner import BarcodeScanner
from src.ingest.session_models import (
    ImageResult,
    MissingItem,
    SessionItem,
    SessionResult,
    SessionStatus,
)
from src.session_repository import NoOpSessionRepository, SessionRepository

logger = logging.getLogger(__name__)


async def run_session_graph(
    session_id: str,
    image: bytes | Path | str,
    *,
    repo: SessionRepository | NoOpSessionRepository,
    scanner: BarcodeScanner | None = None,
    model: str | None = None,
    max_retries: int = 0,
    retry_delay_seconds: float = 0.0,
    source: str | None = None,
) -> SessionResult:
    """Process one image within an ingest session.

    This is the canonical entry point for multi-image ingest. It:

    1. Loads (or creates) the session from the repository.
    2. Runs ScanGraph on the image (via ``analyze_image``).
    3. Merges the result into accumulated session state.
    4. Persists the updated state.
    5. Returns a ``SessionResult`` — either ``complete`` or ``active``
       with information about what's still missing.

    The caller (API route, CLI) checks ``result.status``:

    - ``SessionStatus.COMPLETE`` — all expected boxes have barcodes.
    - ``SessionStatus.ACTIVE`` — still missing boxes, ask for more photos.
    - ``SessionStatus.FAILED`` — scan/audit error on this image.

    Args:
        session_id: Unique session identifier (e.g. upload_id from the API).
        image: Raw image bytes or path to an image file.
        repo: Session repository (Postgres or NoOp).
        scanner: Optional pre-constructed BarcodeScanner.
        model: Gemini model name override.
        max_retries: Gemini retry count.
        retry_delay_seconds: Base delay between retries.
        source: Source label (web, whatsapp, cli) for the session.

    Returns:
        ``SessionResult`` with accumulated items, missing items, and status.
    """
    from src.ingest.analyze import analyze_image_async

    if scanner is None:
        scanner = BarcodeScanner()

    # Load or create the session.
    state = await repo.load_session_state(session_id)
    is_new_session = state is None

    if is_new_session:
        await repo.create_session(session_id, source=source)
        image_index = 0
        existing_items: list[SessionItem] = []
        existing_missing: list[MissingItem] = []
        expected_count = 0
    else:
        s = state["session"]
        image_index = s.get("image_count", 0)
        existing_items = state["items"]
        existing_missing = [m for m in state["missing"] if not m.resolved]
        expected_count = s.get("expected_count", 0)

    # Run ScanGraph on this image (async — stays in the caller's event loop).
    raw = await analyze_image_async(
        image,
        scanner=scanner,
        model=model,
        max_retries=max_retries,
        retry_delay_seconds=retry_delay_seconds,
    )

    image_result = _dict_to_image_result(raw, image_index)

    # If this image's audit failed, don't update the session — return the error.
    if not image_result.audit_available and image_result.status == "retryable_error":
        result = SessionResult(
            session_id=session_id,
            status=SessionStatus.FAILED,
            expected_count=expected_count,
            found_count=len(existing_items),
            missing_count=len(existing_missing),
            items=existing_items,
            missing=existing_missing,
            image_count=image_index,
            latest_image=image_result,
            message=(
                image_result.error.get("message", "Audit failed")
                if image_result.error
                else "Audit failed"
            ),
        )
        return result

    # First image sets the expected count.
    if is_new_session:
        expected_count = image_result.visible_label_count

    # Merge: add new items, resolve missing items.
    # Label indices are per-image (from each photo's Gemini audit), NOT global
    # across the session. So we can't match by label_index across photos.
    # Instead: each new unique barcode resolves one unresolved missing item
    # (the user sent a photo of the missing box, we decoded its barcode).
    new_items: list[SessionItem] = []
    for found in image_result.found:
        # Check if this barcode is already known.
        already_known = any(i.barcode_value == found.barcode_value for i in existing_items)
        if not already_known:
            # New barcode — add to session.
            item = SessionItem(
                barcode_value=found.barcode_value,
                barcode_format=found.barcode_format,
                barcode_bbox=found.barcode_bbox,
                label_bbox=found.label_bbox,
                label_index=found.label_index,
                match_basis=found.match_basis,
                source_image=image_index,
            )
            inserted = await repo.add_item(session_id, item)
            if inserted:
                new_items.append(item)
                existing_items.append(item)

                # Resolve one unresolved missing item (FIFO).
                for m in existing_missing:
                    if not m.resolved:
                        await repo.resolve_missing(session_id, m.label_index, image_index)
                        m.resolved = True
                        logger.info(
                            "Session %s: missing label %d resolved by image %d (barcode=%s)",
                            session_id, m.label_index, image_index, found.barcode_value,
                        )
                        break

    # Update missing items: only from the first image (or if we had none before).
    # Subsequent images' missing entries are less reliable (close-ups show fewer boxes).
    if is_new_session:
        for m in image_result.missing:
            missing_item = MissingItem(
                label_index=m.label_index,
                label_bbox=m.label_bbox,
                barcode_bbox=m.barcode_bbox,
                status=m.status,
                source_image=image_index,
            )
            await repo.add_missing(session_id, missing_item)
            existing_missing.append(missing_item)

    # Recompute counts.
    unresolved_missing = [m for m in existing_missing if not m.resolved]
    found_count = len(existing_items)
    missing_count = len(unresolved_missing)
    image_count = image_index + 1

    # Determine session status.
    if expected_count > 0 and found_count >= expected_count:
        session_status = SessionStatus.COMPLETE
        message = None
    elif missing_count == 0 and found_count > 0:
        # No missing items but found_count < expected_count — shouldn't happen
        # normally, but treat as complete if we have items and no missing.
        session_status = SessionStatus.COMPLETE
        message = None
    else:
        session_status = SessionStatus.ACTIVE
        if unresolved_missing:
            labels = [m.label_index for m in unresolved_missing if m.label_index is not None]
            message = f"Found {found_count}/{expected_count}. Send a photo of box(es): {labels}"
        else:
            message = f"Found {found_count}/{expected_count}. Send another photo."

    # Persist session metadata.
    await repo.update_session(
        session_id,
        status=session_status,
        expected_count=expected_count,
        found_count=found_count,
        missing_count=missing_count,
        image_count=image_count,
        message=message,
    )

    result = SessionResult(
        session_id=session_id,
        status=session_status,
        expected_count=expected_count,
        found_count=found_count,
        missing_count=missing_count,
        items=existing_items,
        missing=unresolved_missing,
        image_count=image_count,
        message=message,
        latest_image=image_result,
    )

    logger.info(
        "Session %s: image %d processed — status=%s found=%d/%d missing=%d",
        session_id, image_index, session_status.value, found_count, expected_count, missing_count,
    )

    return result


def _dict_to_image_result(raw: dict[str, Any], image_index: int) -> ImageResult:
    """Convert an analyze_image() dict result into an ImageResult."""
    found_items = []
    for f in raw.get("found", []):
        found_items.append(SessionItem(
            barcode_value=f["barcode_value"],
            barcode_format=f.get("barcode_format"),
            barcode_bbox=f.get("barcode_bbox"),
            label_bbox=f.get("label_bbox"),
            label_index=f.get("label_index"),
            match_basis=f.get("match_basis"),
            source_image=image_index,
        ))

    missing_items = []
    for m in raw.get("missing", []):
        missing_items.append(MissingItem(
            label_index=m.get("label_index"),
            label_bbox=m.get("label_bbox"),
            barcode_bbox=m.get("barcode_bbox"),
            status=m.get("status", "not_visible"),
            source_image=image_index,
        ))

    summary = raw.get("summary", {})

    return ImageResult(
        image_index=image_index,
        status=raw.get("outcome", "retryable_error"),
        found=found_items,
        missing=missing_items,
        unassigned=raw.get("unassigned", []),
        visible_label_count=summary.get("visible_label_count", 0),
        found_count=summary.get("found_count", 0),
        missing_count=summary.get("missing_count", 0),
        elapsed_ms=raw.get("elapsed_ms", 0),
        audit_available=raw.get("audit_available", False),
        error=raw.get("error"),
    )

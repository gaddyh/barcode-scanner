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

import json
import logging
from datetime import UTC, datetime, timedelta
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

# Sessions that haven't received a photo in this long are lazily expired
# on the next access. 30 minutes is generous for a user photographing
# boxes on a shelf.
SESSION_INACTIVITY_TTL = timedelta(minutes=30)


async def run_session_graph(
    image: bytes | Path | str,
    *,
    repo: SessionRepository | NoOpSessionRepository,
    channel: str,
    participant_id: str,
    scanner: BarcodeScanner | None = None,
    model: str | None = None,
    max_retries: int = 0,
    retry_delay_seconds: float = 0.0,
    source: str | None = None,
    customer_id: str | None = None,
    branch_id: str | None = None,
    action: str | None = None,
) -> SessionResult:
    """Process one image within an ingest session.

    This is the canonical entry point for multi-image ingest. It:

    1. Resolves the session by ``participant_id`` (same mechanism for
       web and WhatsApp).
    2. Loads (or creates) the session from the repository.
    3. Runs ScanGraph on the image (via ``analyze_image_async``).
    4. Merges the result into accumulated session state.
    5. Persists the updated state.
    6. Returns a ``SessionResult`` — either ``complete`` or ``active``
       with information about what's still missing.

    **Session resolution (unified for both channels):**

    - ``participant_id`` identifies the user across requests.
      - Web: a UUID generated in the browser, stored in localStorage.
      - WhatsApp: the sender's phone number.
    - The server looks up the active session for this participant.
    - If found and still active → reuse it.
    - If not found, or complete/expired/closed → create a new session.

    The client never sends a ``session_id``. The server creates it
    internally and returns it in the response (for debugging/admin).

    Args:
        image: Raw image bytes or path to an image file.
        repo: Session repository (Postgres or NoOp).
        channel: 'web' or 'whatsapp'.
        participant_id: Stable user identity. Web: localStorage UUID.
            WhatsApp: sender phone number.
        scanner: Optional pre-constructed BarcodeScanner.
        model: Gemini model name override.
        max_retries: Gemini retry count.
        retry_delay_seconds: Base delay between retries.
        source: Source label (web, whatsapp, cli) for the session.

    Returns:
        ``SessionResult`` with accumulated items, missing items, and status.
    """
    from src.models.upload import generate_upload_id

    # --- Session resolution (unified for both channels) ---
    # find_active_by_participant only returns status='active' sessions.
    # If the session is complete/expired/closed, it won't be found, and a
    # new session is created automatically — the user just sends another
    # photo and gets a fresh session.
    existing = await repo.find_active_by_participant(channel, participant_id)

    session_id: str | None = None
    if existing is not None:
        # Check lazy expiry before reusing.
        last_activity = existing.get("last_activity_at")
        if last_activity:
            if isinstance(last_activity, str):
                last_activity = datetime.fromisoformat(last_activity)
            if last_activity.tzinfo is None:
                last_activity = last_activity.replace(tzinfo=UTC)
            age = datetime.now(UTC) - last_activity
            if age > SESSION_INACTIVITY_TTL:
                # Session expired — mark it and let a new session be created.
                await repo.expire_session(existing["id"])
            else:
                session_id = existing["id"]

    if session_id is None:
        session_id = generate_upload_id()

    from src.ingest.analyze import analyze_image_async

    if scanner is None:
        scanner = BarcodeScanner()

    # Load or create the session.
    state = await repo.load_session_state(session_id)
    is_new_session = state is None

    if is_new_session:
        # Don't create the session row yet — wait until the scan succeeds.
        # If the first image fails, no session is created, and the next
        # image starts fresh (find_active_by_participant won't find it).
        image_index = 0
        existing_items: list[SessionItem] = []
        existing_missing: list[MissingItem] = []
        expected_count = 0
    else:
        s = state["session"]
        customer_id = s.get("customer_id", customer_id)
        branch_id = s.get("branch_id", branch_id)
        action = s.get("action", action)
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

    # If this image's audit failed, don't create/update the session — return the error.
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
            customer_id=customer_id,
            branch_id=branch_id,
            action=action,
        )
        return result

    # Scan succeeded — create the session row if this is the first image.
    if is_new_session:
        await repo.create_session(
            session_id,
            source=source,
            channel=channel,
            participant_id=participant_id,
            customer_id=customer_id,
            branch_id=branch_id,
            action=action,
        )
        expected_count = image_result.visible_label_count
    else:
        # Update expected_count if this photo shows more visible labels than
        # the first (e.g. first photo was at an angle, missed some boxes).
        # Also handle the edge case where expected_count was never set.
        if image_result.visible_label_count > expected_count:
            old_expected = expected_count
            expected_count = image_result.visible_label_count
            logger.info(
                "Session %s: expected_count updated %d → %d (photo %d saw more labels)",
                session_id, old_expected, expected_count, image_index,
            )

    # Merge: add new items, resolve missing items.
    #
    # First image: add ALL found items (one per label). Set expected_count.
    #
    # Subsequent images — deterministic resolution:
    #   Filter out already-known barcodes (neighbors). The remaining are
    #   "new" candidates that could resolve missing labels.
    #   - new_unique == missing_count → perfect match, add all, resolve all
    #   - new_unique < missing_count  → add all new (partial resolution)
    #   - new_unique > missing_count  → AMBIGUOUS, ask user to pick
    #   - new_unique == 0             → nothing new, ask for better photo
    #
    # We dedup new barcodes by value first — same product on 2 boxes in the
    # photo counts as 1 unique new barcode, not 2.
    unresolved_before = [m for m in existing_missing if not m.resolved]
    missing_before_count = len(unresolved_before)

    # Filter to only NEW barcodes (not already in session), then dedup by value.
    known_barcodes = {i.barcode_value for i in existing_items}
    new_found = [f for f in image_result.found if f.barcode_value not in known_barcodes]
    # Dedup by barcode_value — keep first occurrence per value.
    seen_values: set[str] = set()
    new_found_unique: list = []
    for f in new_found:
        if f.barcode_value not in seen_values:
            seen_values.add(f.barcode_value)
            new_found_unique.append(f)
    new_count = len(new_found_unique)

    candidates: list[SessionItem] = []
    needs_selection = False

    if is_new_session:
        # First image — add every found label.
        for found in image_result.found:
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
                existing_items.append(item)
    elif new_count == 0:
        # Nothing new — ask for a better photo.
        pass
    elif new_count <= missing_before_count:
        # Exact or fewer — accept all new items, resolve missing labels.
        for found in new_found_unique:
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
                existing_items.append(item)
                # Resolve one unresolved missing item (FIFO).
                for m in existing_missing:
                    if not m.resolved:
                        await repo.resolve_missing(
                            session_id, m.label_index, image_index
                        )
                        m.resolved = True
                        logger.info(
                            "Session %s: missing label %d resolved by image %d "
                            "(barcode=%s)",
                            session_id, m.label_index, image_index,
                            found.barcode_value,
                        )
                        break
    else:
        # More new than missing — ambiguous. Don't add anything.
        # Return candidates for user selection.
        needs_selection = True
        for found in new_found_unique:
            candidates.append(SessionItem(
                barcode_value=found.barcode_value,
                barcode_format=found.barcode_format,
                barcode_bbox=found.barcode_bbox,
                label_bbox=found.label_bbox,
                label_index=found.label_index,
                match_basis=found.match_basis,
                source_image=image_index,
            ))

    # Update missing items.
    # First image: record all missing labels.
    # Subsequent images: if expected_count increased (photo saw more labels),
    # add the new missing labels from this photo. Label indices are per-image,
    # so we can't dedup across photos — we add all missing from this photo.
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
    elif image_result.visible_label_count > 0 and image_result.missing:
        # expected_count grew — add new missing labels from this photo.
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
            logger.info(
                "Session %s: new missing label %d added from image %d "
                "(expected_count grew)",
                session_id, m.label_index, image_index,
            )

    # Recompute counts.
    unresolved_missing = [m for m in existing_missing if not m.resolved]
    found_count = len(existing_items)
    missing_count = len(unresolved_missing)
    image_count = image_index + 1

    # Determine session status.
    if needs_selection:
        session_status = SessionStatus.NEEDS_USER_SELECTION
        barcodes = [c.barcode_value for c in candidates]
        message = (
            f"Found {new_count} new barcodes but only {missing_before_count} "
            f"missing. Which one(s) to add? {barcodes}"
        )
    elif expected_count > 0 and found_count >= expected_count:
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
    candidate_dicts = (
        [c.model_dump(mode="json") for c in candidates] if needs_selection else []
    )
    await repo.update_session(
        session_id,
        status=session_status,
        expected_count=expected_count,
        found_count=found_count,
        missing_count=missing_count,
        image_count=image_count,
        message=message,
        candidates=candidate_dicts if needs_selection else [],
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
        candidates=candidates if needs_selection else [],
        customer_id=customer_id,
        branch_id=branch_id,
        action=action,
    )

    logger.info(
        "Session %s: image %d processed — status=%s found=%d/%d missing=%d",
        session_id, image_index, session_status.value, found_count, expected_count, missing_count,
    )

    return result


async def select_candidate(
    session_id: str,
    barcode_value: str,
    *,
    repo: SessionRepository | NoOpSessionRepository,
) -> SessionResult:
    """Resolve a user selection from the candidates list.

    Called when a session is in ``NEEDS_USER_SELECTION`` status and the
    user has chosen which barcode to add. This:

    1. Loads the session and its persisted candidates.
    2. Finds the candidate matching ``barcode_value``.
    3. Adds it as a session item, resolves one missing label.
    4. Clears the candidates list.
    5. Returns the updated ``SessionResult``.

    If the barcode is not in the candidates list, raises ``ValueError``.
    If the session is not in ``needs_user_selection`` status, raises ``ValueError``.
    """
    state = await repo.load_session_state(session_id)
    if state is None:
        raise ValueError(f"Session {session_id} not found")

    s = state["session"]
    customer_id = s.get("customer_id")
    branch_id = s.get("branch_id")
    action = s.get("action")
    current_status = s.get("status", "active")
    if current_status != "needs_user_selection":
        raise ValueError(
            f"Session {session_id} is not awaiting selection (status={current_status})"
        )

    # Load persisted candidates.
    raw_candidates = s.get("candidates") or []
    if isinstance(raw_candidates, str):
        raw_candidates = json.loads(raw_candidates)

    # Find the matching candidate.
    matched = None
    for c in raw_candidates:
        if c.get("barcode_value") == barcode_value:
            matched = c
            break

    if matched is None:
        available = [c.get("barcode_value") for c in raw_candidates]
        raise ValueError(
            f"Barcode {barcode_value} not in candidates. Available: {available}"
        )

    # Add the item.
    item = SessionItem(
        barcode_value=matched["barcode_value"],
        barcode_format=matched.get("barcode_format"),
        barcode_bbox=matched.get("barcode_bbox"),
        label_bbox=matched.get("label_bbox"),
        label_index=matched.get("label_index"),
        match_basis=matched.get("match_basis"),
        source_image=matched.get("source_image", 0),
    )
    inserted = await repo.add_item(session_id, item)
    existing_items = state["items"]
    if inserted:
        existing_items.append(item)

    # Resolve one unresolved missing item (FIFO).
    existing_missing = [m for m in state["missing"] if not m.resolved]
    for m in existing_missing:
        if not m.resolved:
            await repo.resolve_missing(session_id, m.label_index, item.source_image)
            m.resolved = True
            logger.info(
                "Session %s: missing label %d resolved by user selection (barcode=%s)",
                session_id, m.label_index, barcode_value,
            )
            break

    # Recompute counts.
    unresolved_missing = [m for m in existing_missing if not m.resolved]
    found_count = len(existing_items)
    missing_count = len(unresolved_missing)
    expected_count = s.get("expected_count", 0)
    image_count = s.get("image_count", 0)

    # Determine new status.
    if expected_count > 0 and found_count >= expected_count:
        session_status = SessionStatus.COMPLETE
        message = None
    else:
        session_status = SessionStatus.ACTIVE
        if unresolved_missing:
            labels = [m.label_index for m in unresolved_missing if m.label_index is not None]
            message = f"Found {found_count}/{expected_count}. Send a photo of box(es): {labels}"
        else:
            message = f"Found {found_count}/{expected_count}. Send another photo."

    # Persist — clear candidates, update status.
    await repo.update_session(
        session_id,
        status=session_status,
        found_count=found_count,
        missing_count=missing_count,
        message=message,
        candidates=[],
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
        latest_image=None,
        candidates=[],
        customer_id=customer_id,
        branch_id=branch_id,
        action=action,
    )

    logger.info(
        "Session %s: user selected %s — status=%s found=%d/%d missing=%d",
        session_id, barcode_value, session_status.value, found_count, expected_count, missing_count,
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

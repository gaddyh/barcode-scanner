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
# boxes on a shelf. Within the TTL, COMPLETE sessions are reused for
# aggregation (the user can add more boxes to an already-complete
# session). After the TTL, a new session is created.
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
       web).
    2. Loads (or creates) the session from the repository.
    3. Runs ScanGraph on the image (via ``analyze_image_async``).
    4. Merges the result into accumulated session state.
    5. Persists the updated state.
    6. Returns a ``SessionResult`` — either ``complete`` or ``active``
       with information about what's still missing.

    **Session resolution (unified for both channels):**

    - ``participant_id`` identifies the user across requests.
      - Web: a UUID generated in the browser, per page load.
      - Web: the participant UUID (per page load).
    - The server looks up the active, needs_user_selection, or complete
      session for this participant.
    - If found → reuse it (including COMPLETE sessions, so the user can
      aggregate more boxes).
    - If not found → create a new session.
    - A session only stops accepting images when it enters the
      submission state machine (SUBMITTING/SUBMITTED) or the user starts
      a new session (page refresh → new participant_id).

    The client never sends a ``session_id``. The server creates it
    internally and returns it in the response (for debugging/admin).

    Args:
        image: Raw image bytes or path to an image file.
        repo: Session repository (Postgres or NoOp).
        channel: 'web' or 'cli'.
        participant_id: Stable user identity. Web: localStorage UUID.
            Web: participant UUID from localStorage.
        scanner: Optional pre-constructed BarcodeScanner.
        model: Gemini model name override.
        max_retries: Gemini retry count.
        retry_delay_seconds: Base delay between retries.
        source: Source label (web, cli) for the session.

    Returns:
        ``SessionResult`` with accumulated items, missing items, and status.
    """
    from src.models.upload import generate_upload_id

    # --- Session resolution (unified for both channels) ---
    # find_active_by_participant returns active, needs_user_selection,
    # or complete sessions. A COMPLETE session is included so the user
    # can aggregate additional boxes into an already-complete session.
    # Sessions are lazily expired after SESSION_INACTIVITY_TTL of
    # inactivity; after that, a new session is created.
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
        else:
            session_id = existing["id"]

    if session_id is None:
        session_id = generate_upload_id()

    from src.ingest.analyze import analyze_image_async

    if scanner is None:
        scanner = BarcodeScanner()

    # Load or create the session.
    state = await repo.load_session_state(session_id)
    # Treat a session with no items as a first image — the receiving API
    # creates the session row before the first photo, so the row exists
    # but has no accumulated items yet.
    is_new_session = state is None or not state.get("items")

    if is_new_session:
        # Don't create the session row yet — wait until the scan succeeds.
        # If the first image fails, no session is created, and the next
        # image starts fresh (find_active_by_participant won't find it).
        image_index: int = 0
        existing_items: list[SessionItem] = []
        existing_missing: list[MissingItem] = []
        expected_count = 0
    else:
        s = state["session"]  # type: ignore[index]
        customer_id = s.get("customer_id", customer_id)
        branch_id = s.get("branch_id", branch_id)
        action = s.get("action", action)
        image_index = int(s.get("image_count", 0))
        existing_items = state["items"]  # type: ignore[index]
        existing_missing = [m for m in state["missing"] if not m.resolved]  # type: ignore[index]
        expected_count = s.get("expected_count", 0)

    # Capture expected_count before any update, so the missing-items
    # logic can tell whether this photo genuinely grew the target.
    _expected_count_at_photo_start = expected_count

    # Run ScanGraph on this image (async — stays in the caller's event loop).
    raw = await analyze_image_async(
        image,
        scanner=scanner,
        model=model,
        max_retries=max_retries,
        retry_delay_seconds=retry_delay_seconds,
    )

    image_result = _dict_to_image_result(raw, image_index)

    logger.info(
        "Session %s image %d analyzed: status=%s found=%d missing=%d "
        "visible=%d audit=%s outcome=%s annotated=%s",
        session_id, image_index, image_result.status,
        image_result.found_count, image_result.missing_count,
        image_result.visible_label_count, image_result.audit_available,
        image_result.status, bool(image_result.annotated_image_b64),
    )
    if image_result.found:
        logger.info(
            "Session %s image %d found barcodes: %s",
            session_id, image_index,
            [f.barcode_value for f in image_result.found],
        )
    if image_result.missing:
        logger.info(
            "Session %s image %d missing labels: %s",
            session_id, image_index,
            [m.label_index for m in image_result.missing],
        )

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

    # Track whether expected_count genuinely grew on this photo (for the
    # missing-items update below). Computed BEFORE the expected_count update
    # above would have already happened — so we check against the value that
    # was current at the start of this photo.
    expected_count_grew = (
        not is_new_session
        and image_result.visible_label_count > 0
        and image_result.visible_label_count > _expected_count_at_photo_start
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

    # Targeted-retry occurrence contract (PR A):
    #
    # A subsequent photo is a TARGETED RETRY — "photograph only the missing
    # box(es)". Under this contract:
    #
    # - If the session is COMPLETE (0 missing) and the photo has new barcodes:
    #   treat as AGGREGATION — accept all new items, grow expected_count.
    # - If the photo has ≤ missing slots barcodes: accept ALL as candidates,
    #   even if some match existing barcode values. Duplicate EANs represent
    #   separate physical occurrences — the missing box may have the same
    #   product code as an already-found box.
    # - If the photo has > missing slots barcodes: filter out already-known
    #   values (they're likely neighbors from a wider frame). If the
    #   remaining candidates still exceed missing slots → AMBIGUOUS, return
    #   candidates for user selection. If 0 remain after filtering (all
    #   duplicates), ask for a better photo — the user needs to photograph
    #   only the missing boxes, not a wider frame with neighbors.
    # - Do NOT dedup candidates by barcode value.
    all_found = list(image_result.found)
    all_count = len(all_found)
    is_aggregation = False

    if not is_new_session and missing_before_count == 0 and all_count > 0:
        # Session is COMPLETE — treat as aggregation. Accept all NEW
        # barcodes (not in known set). Grow expected_count to match.
        known_barcodes = {i.barcode_value for i in existing_items}
        new_found = [f for f in all_found if f.barcode_value not in known_barcodes]
        is_aggregation = bool(new_found)
    elif not is_new_session and all_count > missing_before_count:
        # More barcodes than missing slots — filter known (neighbors).
        known_barcodes = {i.barcode_value for i in existing_items}
        new_found = [f for f in all_found if f.barcode_value not in known_barcodes]
        # If 0 remain after filtering (all duplicates), do NOT accept —
        # ask for a better photo with only the missing boxes.
    else:
        # ≤ missing slots — accept all (even duplicates of known values).
        new_found = all_found
    new_count = len(new_found)

    existing_barcodes = sorted({i.barcode_value for i in existing_items})
    logger.info(
        "Session %s merge: image_index=%d is_new=%s "
        "image_found=%d image_missing=%d image_visible=%d "
        "existing_items=%d existing_missing=%d unresolved=%d "
        "existing_barcodes=%s new_found=%d new_count=%d "
        "expected_count=%d expected_count_grew=%s",
        session_id, image_index, is_new_session,
        image_result.found_count, image_result.missing_count,
        image_result.visible_label_count,
        len(existing_items), len(existing_missing), missing_before_count,
        existing_barcodes, len(new_found), new_count,
        expected_count, expected_count_grew,
    )
    if new_found:
        logger.info(
            "Session %s new_found barcodes: %s",
            session_id,
            [f.barcode_value for f in new_found],
        )

    candidates: list[SessionItem] = []
    needs_selection = False

    if is_new_session:
        # First image — add every found label.
        logger.info(
            "Session %s: first image, adding all %d found items",
            session_id, len(image_result.found),
        )
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
    elif is_aggregation:
        # Session was COMPLETE — aggregation mode. Add all new items and
        # grow expected_count to account for the additional boxes.
        logger.info(
            "Session %s: aggregation mode, adding %d new items",
            session_id, new_count,
        )
        added = 0
        for found in new_found:
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
                added += 1
        # Grow expected_count by the number of new items added.
        expected_count += added
        logger.info(
            "Session %s: expected_count grew %d → %d (aggregation)",
            session_id, _expected_count_at_photo_start, expected_count,
        )
    elif new_count == 0:
        # Nothing new — ask for a better photo.
        logger.info(
            "Session %s: no new barcodes found (all %d already known)",
            session_id, len(image_result.found),
        )
        pass
    elif new_count <= missing_before_count:
        # Exact or fewer — accept all new occurrences, resolve missing labels.
        logger.info(
            "Session %s: accepting %d new barcodes to resolve %d missing slots",
            session_id, new_count, missing_before_count,
        )
        for found in new_found:
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
                            session_id, m.label_index or 0, image_index
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
        # More candidates than missing slots — ambiguous. Don't add anything.
        # Return candidates for user selection.
        logger.info(
            "Session %s: AMBIGUOUS — %d new barcodes > %d missing slots, asking user to pick",
            session_id, new_count, missing_before_count,
        )
        needs_selection = True
        for found in new_found:
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
    # Subsequent images (targeted-retry mode): keep the target fixed.
    # The expected_count was established by the first photo (or the audit).
    # A retry photo that sees "missing" labels is just seeing the same
    # missing boxes from a different angle — we do NOT add new missing
    # slots, because that would let every retry grow the target.
    # The expected_count_grew check below only fires when the audit
    # genuinely sees more labels than before (tracked explicitly).
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
    elif expected_count_grew and image_result.missing:
        # expected_count genuinely grew (audit sees more labels than the
        # previous expected_count) — add the new missing labels from this
        # photo. This is NOT a retry; it's a photo that reveals more boxes
        # than previously known.
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
                "(expected_count grew from %d to %d)",
                session_id, m.label_index, image_index,
                _expected_count_at_photo_start, expected_count,
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
        annotated_image_b64=image_result.annotated_image_b64,
        annotated_image_width=image_result.annotated_image_width,
        annotated_image_height=image_result.annotated_image_height,
        candidates=candidates if needs_selection else [],
        customer_id=customer_id,
        branch_id=branch_id,
        action=action,
    )

    logger.info(
        "Session %s: image %d processed — status=%s found=%d/%d missing=%d "
        "items=%d unresolved_missing=%d",
        session_id, image_index, session_status.value, found_count, expected_count, missing_count,
        len(existing_items), len(unresolved_missing),
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
            await repo.resolve_missing(session_id, m.label_index or 0, item.source_image or 0)
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
        annotated_image_b64=raw.get("annotated_image_b64"),
        annotated_image_width=raw.get("annotated_image_width"),
        annotated_image_height=raw.get("annotated_image_height"),
    )

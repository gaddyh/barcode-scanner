"""Receiving flow API — the end-to-end vertical slice for Visual Receiving.

This router wires one safe end-to-end flow:

    POST /receiving/sessions          — create session (customer + branch + action)
    POST /receiving/sessions/{id}/images — upload photo, run analyze_image(), append boxes
    POST /receiving/sessions/{id}/submit   — freeze, create draft order via runtime
    GET  /receiving/sessions/{id}        — inspect session

The submit endpoint goes THROUGH ``runtime.execute(policy=EXTERNAL_WRITE,
idempotency_key=f"priority:draft:{session_id}")`` in the SERVICE layer
(NOT inside the gateway — per PR #4 layering). Double-submit returns
the cached order ID; an indeterminate outcome transitions to
``SUBMISSION_UNKNOWN`` and replays on retry.

Persistence (PR A): sessions are stored in Postgres via
``ReceivingSessionStore``. Submission transitions are compare-and-set
UPDATEs guarded by the expected ``submission_status`` — two concurrent
submit clicks have exactly one winner. The frozen order payload is
persisted verbatim on ACTIVE → SUBMITTING and reused on every retry;
it is never rebuilt from ``session_items``.

One unresolved receiving session per participant: if a participant has
an open (active or submission_unknown) session, creating a new session
returns the existing one with 409.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from src.domain.receiving import (
    ReceivingSessionStatus,
)
from src.integrations.priority.models import (
    CreateDraftOrderRequest,
    OrderLineItem,
)
from src.runtime.context import RunContext
from src.runtime.errors import (
    IndeterminateError,
    PermanentError,
    RetryableError,
)
from src.runtime.executor import execute
from src.runtime.idempotency import InMemoryIdempotencyStore
from src.runtime.policy import EXTERNAL_WRITE
from src.session_repository import ReceivingSessionStore

router = APIRouter(prefix="/receiving", tags=["receiving"])

# In-memory idempotency store for tests/local dev only.
# In production (DATABASE_URL set), _get_idempotency_store() returns
# PostgresIdempotencyStore. If Receiving is enabled in a deployed
# environment and there is no DB, submit returns 503 rather than
# pretending it is safely idempotent.
_idempotency_store = InMemoryIdempotencyStore[dict[str, Any]]()


def _get_receiving_store() -> ReceivingSessionStore:
    """Get the Postgres-backed ReceivingSessionStore.

    Raises HTTPException 503 if no DB pool is configured — Receiving
    requires durable persistence. In-memory fallback is test/local only
    via dependency override, never in a deployed app.
    """
    from src.main import _db_pool

    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "persistence_required",
                "message": "Receiving requires a configured database.",
            },
        )
    return ReceivingSessionStore(_db_pool)


def _get_idempotency_store() -> Any:
    """Get the idempotency store for the draft-order write.

    Returns PostgresIdempotencyStore when a DB pool is configured.
    Falls back to the module-level InMemoryIdempotencyStore for tests
    and local dev. In a deployed app without a DB, submit returns 503
    via _get_receiving_store() before reaching this point.
    """
    from src.main import _db_pool

    if _db_pool is not None:
        from src.runtime.postgres_idempotency import PostgresIdempotencyStore

        return PostgresIdempotencyStore(_db_pool)
    return _idempotency_store


def _get_priority_gateway() -> Any:
    """Get the Priority gateway singleton."""
    from src.api.routes import _get_priority_repo

    return _get_priority_repo()


async def _validate_customer_and_branch(
    customer_id: str, branch_id: str
) -> None:
    """Validate that the customer exists, the branch exists, and the branch
    belongs to the selected customer. Raises HTTPException on failure.
    """
    gateway = _get_priority_gateway()
    try:
        customers = await gateway.customers()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "code": "priority_unavailable",
                "message": f"Could not validate customer: {exc}",
            },
        ) from exc

    customer_ids = {c["id"] for c in customers}
    if customer_id not in customer_ids:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "unknown_customer",
                "message": f"Customer '{customer_id}' does not exist.",
            },
        )

    try:
        branches = await gateway.branches(customer_id)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "code": "priority_unavailable",
                "message": f"Could not validate branch: {exc}",
            },
        ) from exc

    branch_ids = {b["id"] for b in branches}
    if branch_id not in branch_ids:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "unknown_or_mismatched_branch",
                "message": (
                    f"Branch '{branch_id}' does not exist or does not "
                    f"belong to customer '{customer_id}'."
                ),
            },
        )


# ---------------------------------------------------------------------------
# POST /receiving/sessions — create session
# ---------------------------------------------------------------------------


@router.post("/sessions", status_code=status.HTTP_201_CREATED)
async def create_session(
    customer_id: str = Form(..., description="Priority customer ID"),
    branch_id: str = Form(..., description="Priority branch ID"),
    action: str = Form(
        ..., description="create_order or verify_order_before_shipment"
    ),
    participant_id: str = Form(
        "", description="Operator/participant ID (one unresolved session per participant)"
    ),
) -> dict[str, Any]:
    """Create a new receiving session. Session starts ACTIVE.

    One unresolved receiving session per participant: if the participant
    already has an active or submission_unknown session, returns 409
    with the existing session ID.
    """
    if not customer_id.strip() or not branch_id.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "order_context_required",
                "message": "customer_id and branch_id are required.",
            },
        )
    if action not in ("create_order", "verify_order_before_shipment"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "invalid_action", "message": "Unsupported action."},
        )

    # Validate customer + branch existence and relationship.
    await _validate_customer_and_branch(customer_id, branch_id)

    store = _get_receiving_store()

    # Enforce one unresolved receiving session per participant.
    if participant_id.strip():
        existing = await store.find_open_submission_by_participant(participant_id)
        if existing is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "open_session_exists",
                    "message": (
                        "An unresolved receiving session already exists for "
                        "this participant. Retry or resolve it before creating "
                        "a new session."
                    ),
                    "existing_session_id": existing.session_id,
                    "existing_status": existing.status.value,
                },
            )

    session_id = str(uuid.uuid4())
    await store.create_receiving_session(
        session_id,
        customer_id=customer_id,
        branch_id=branch_id,
        action=action,
        participant_id=participant_id.strip() or None,
    )

    return {
        "session_id": session_id,
        "status": ReceivingSessionStatus.ACTIVE.value,
        "customer_id": customer_id,
        "branch_id": branch_id,
        "action": action,
        "box_count": 0,
    }


# ---------------------------------------------------------------------------
# POST /receiving/sessions/{id}/images — upload photo, append boxes
# ---------------------------------------------------------------------------


@router.post("/sessions/{session_id}/images")
async def upload_image(
    session_id: str,
    file: UploadFile = File(..., description="JPEG, PNG, or WebP product photo"),
) -> dict[str, Any]:
    """Upload a photo, run the persisted scan-graph accumulation, append boxes.

    The session must be ACTIVE. Once SUBMITTING, images cannot be added.
    Uses ``run_session_graph`` for cross-image occurrence-based accumulation
    (targeted-retry contract, multiset semantics for duplicate EANs).
    """
    store = _get_receiving_store()
    session = await store.get_receiving_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "session_not_found", "message": "Session not found."},
        )

    if session.status != ReceivingSessionStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "session_not_editable",
                "message": (
                    f"Session is {session.status.value}; "
                    f"cannot add images."
                ),
            },
        )

    image_bytes = await file.read()

    # Use the persisted SessionRepository for cross-image accumulation.
    # run_session_graph reads/writes session_items and session_missing
    # rows, applies the targeted-retry occurrence contract, and updates
    # expected_count/found_count/missing_count.
    from src.ingest.session_graph import run_session_graph
    from src.main import _db_pool
    from src.session_repository import SessionRepository

    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "persistence_unavailable",
                "message": "Postgres is not configured; image upload is disabled.",
            },
        )
    repo = SessionRepository(_db_pool)

    # Use the receiving session_id as the ingest participant_id so the
    # ingest session is 1:1 with the receiving session. The ingest
    # session accumulates boxes; the receiving session tracks submission
    # state. They share the same session_id row in the sessions table.
    ingest_participant_id = session_id

    result = await run_session_graph(
        image_bytes,
        repo=repo,
        source="web",
        channel="web",
        participant_id=ingest_participant_id,
        customer_id=session.customer_id,
        branch_id=session.branch_id,
        action=session.action,
    )

    if result.status.value == "failed":
        latest = result.latest_image
        if latest and latest.error:
            msg = latest.error.get("message", "Analysis failed.")
        else:
            msg = "Analysis failed."
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "code": "analysis_failed",
                "message": msg,
            },
        )

    return {
        "session_id": session_id,
        "status": "active",  # receiving session stays active until submit
        "outcome": result.status.value,
        "boxes_added": result.found_count,
        "total_boxes": result.found_count,
        "expected_count": result.expected_count,
        "missing_count": result.missing_count,
        "discrepancy": {
            "expected": result.expected_count,
            "found": result.found_count,
            "missing": result.missing_count,
            "is_complete": result.missing_count == 0 and result.expected_count > 0,
        },
        "candidates": [c.model_dump(mode="json") for c in result.candidates],
        "message": result.message,
    }


# ---------------------------------------------------------------------------
# POST /receiving/sessions/{id}/submit — freeze + create draft order
# ---------------------------------------------------------------------------


async def _create_draft_order_operation(
    input_: Any,
    context: RunContext,
    **kwargs: Any,
) -> dict[str, Any]:
    """The operation executed by the runtime.

    Calls the Priority gateway to create a draft order. The gateway
    classifies errors at the integration boundary; the executor
    preserves and persists what the gateway raises.
    """
    gateway = _get_priority_gateway()
    request: CreateDraftOrderRequest = input_

    result = await gateway.create_draft_order(request)
    return {
        "order_id": result.order_id,
        "session_id": result.session_id,
        "status": result.status,
    }


@router.post("/sessions/{session_id}/submit")
async def submit_session(session_id: str) -> dict[str, Any]:
    """Freeze the session and create a draft order via the runtime.

    Idempotency: same session → same key → double-submit returns the
    cached order ID, never creates a duplicate.

    State transitions (persisted via CAS UPDATE):
        ACTIVE → SUBMITTING → SUBMITTED (success)
        ACTIVE → SUBMITTING → SUBMISSION_UNKNOWN (indeterminate)
        ACTIVE (stays) on pre-submit failure
    """
    store = _get_receiving_store()
    session = await store.get_receiving_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "session_not_found", "message": "Session not found."},
        )

    # Already submitted — return the cached result.
    if session.status == ReceivingSessionStatus.SUBMITTED:
        return {
            "session_id": session_id,
            "status": session.status.value,
            "order_id": session.external_order_id,
            "idempotent": True,
        }

    if session.status == ReceivingSessionStatus.SUBMITTING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "submission_in_progress",
                "message": "Submission is already in progress.",
            },
        )

    if (
        session.status != ReceivingSessionStatus.ACTIVE
        and session.status != ReceivingSessionStatus.SUBMISSION_UNKNOWN
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "invalid_state",
                "message": f"Session is {session.status.value}; cannot submit.",
            },
        )

    # Reject empty orders.
    if not session.boxes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "empty_order",
                "message": "Cannot submit an empty draft order.",
            },
        )

    # Build the frozen order payload from the current session contents.
    # This exact payload is persisted and reused on every retry.
    quantities = session.aggregate_quantities()
    request = CreateDraftOrderRequest(
        session_id=session.session_id,
        customer_id=session.customer_id,
        branch_id=session.branch_id,
        action=session.action,
        items=[
            OrderLineItem(
                barcode_value=value,
                barcode_format=fmt,
                quantity=qty,
            )
            for value, fmt, qty in quantities
        ],
    )
    frozen_payload: dict[str, Any] = {
        "session_id": session.session_id,
        "customer_id": session.customer_id,
        "branch_id": session.branch_id,
        "action": session.action,
        "items": request.items_as_dicts(),
    }

    # Freeze: ACTIVE → SUBMITTING (CAS). On retry from SUBMISSION_UNKNOWN,
    # the frozen payload is already persisted — reuse it.
    if session.status == ReceivingSessionStatus.ACTIVE:
        ok = await store.freeze_submission(session_id, frozen_payload)
        if not ok:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "submission_in_progress",
                    "message": "Another submission is already in progress.",
                },
            )
    else:
        # Retry from SUBMISSION_UNKNOWN — use the already-frozen payload.
        frozen = await store.get_frozen_payload(session_id)
        if frozen is not None:
            frozen_payload = frozen

    context = RunContext(
        run_id=str(uuid.uuid4()),
        session_id=session.session_id,
        source="web",
    )

    try:
        result = await execute(
            _create_draft_order_operation,
            request,
            context,
            name="priority_create_draft_order",
            policy=EXTERNAL_WRITE,
            idempotency_key=f"priority:draft:{session_id}",
            idempotency_store=_get_idempotency_store(),
        )
    except IndeterminateError as exc:
        await store.mark_submission_unknown(session_id)
        return {
            "session_id": session_id,
            "status": ReceivingSessionStatus.SUBMISSION_UNKNOWN.value,
            "error": {"code": "submission_unknown", "message": str(exc)},
            "retry_recommended": True,
        }
    except RetryableError as exc:
        # Pre-submit failure — revert to ACTIVE so the user can retry/edit.
        await store.revert_to_active(session_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "code": "submission_failed",
                "message": str(exc),
            },
        ) from exc
    except PermanentError as exc:
        await store.revert_to_active(session_id)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "submission_permanent_error",
                "message": str(exc),
            },
        ) from exc

    order_id = result["order_id"]
    await store.mark_submitted(session_id, order_id)

    return {
        "session_id": session_id,
        "status": ReceivingSessionStatus.SUBMITTED.value,
        "order_id": order_id,
        "items": [
            {"barcode_value": v, "barcode_format": f, "quantity": q}
            for v, f, q in quantities
        ],
    }


# ---------------------------------------------------------------------------
# GET /receiving/sessions/{id} — inspect session
# ---------------------------------------------------------------------------


@router.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict[str, Any]:
    """Get the current state of a receiving session."""
    store = _get_receiving_store()
    session = await store.get_receiving_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "session_not_found", "message": "Session not found."},
        )
    quantities = session.aggregate_quantities()
    return {
        "session_id": session_id,
        "status": session.status.value,
        "customer_id": session.customer_id,
        "branch_id": session.branch_id,
        "action": session.action,
        "box_count": len(session.boxes),
        "expected_count": session.expected_count,
        "external_order_id": session.external_order_id,
        "frozen": session.frozen,
        "items": [
            {"barcode_value": v, "barcode_format": f, "quantity": q}
            for v, f, q in quantities
        ],
        "discrepancy": {
            "expected": session.discrepancy.expected,
            "found": session.discrepancy.found,
            "missing": session.discrepancy.missing,
            "is_complete": session.discrepancy.is_complete,
        },
    }

"""Receiving flow API — the end-to-end vertical slice for Visual Receiving.

This router wires one safe end-to-end flow:

    POST /receiving/sessions          — create session (customer + branch + action)
    POST /receiving/sessions/{id}/images — upload photo, run analyze_image(), append boxes
    POST /receiving/sessions/{id}/submit   — freeze, create draft order via runtime

The submit endpoint goes THROUGH ``runtime.execute(policy=EXTERNAL_WRITE,
idempotency_key=f"priority:draft:{session_id}")`` in the SERVICE layer
(NOT inside the gateway — per PR #4 layering). Double-submit returns
the cached order ID; an indeterminate outcome transitions to
``SUBMISSION_UNKNOWN`` and replays on retry.

Session immutability after SUBMITTING: once a session enters SUBMITTING,
its contents cannot be edited. This makes ``session_id`` a safe
logical idempotency key.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from src.domain.receiving import (
    PhysicalBox,
    ReceivingSession,
    ReceivingSessionStatus,
)
from src.ingest.analyze import analyze_image
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

router = APIRouter(prefix="/receiving", tags=["receiving"])

# In-memory session store (PR #5 vertical slice). PR #6 will persist this
# in Postgres. Keyed by session_id.
_sessions: dict[str, ReceivingSession] = {}

# In-memory idempotency store for the draft-order write. In production
# this would be PostgresIdempotencyStore; for the vertical slice we use
# the in-memory store so tests don't need a live DB.
_idempotency_store = InMemoryIdempotencyStore[dict[str, Any]]()


def _get_session(session_id: str) -> ReceivingSession:
    """Get a session by ID or raise 404."""
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "session_not_found", "message": "Session not found."},
        )
    return session


def _get_priority_gateway():
    """Get the Priority gateway singleton."""
    from src.api.routes import _get_priority_repo

    return _get_priority_repo()


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
) -> dict[str, Any]:
    """Create a new receiving session. Session starts ACTIVE."""
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

    session_id = str(uuid.uuid4())
    session = ReceivingSession(
        session_id=session_id,
        customer_id=customer_id,
        branch_id=branch_id,
        action=action,
    )
    _sessions[session_id] = session

    return {
        "session_id": session_id,
        "status": session.status.value,
        "customer_id": session.customer_id,
        "branch_id": session.branch_id,
        "action": session.action,
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
    """Upload a photo, run analyze_image(), append decoded boxes to the session.

    The session must be ACTIVE. Once SUBMITTING, images cannot be added.
    """
    session = _get_session(session_id)

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

    result = analyze_image(image_bytes)

    if result["outcome"] == "retryable_error":
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "code": "analysis_failed",
                "message": result.get("error", {}).get("message", "Analysis failed."),
            },
        )

    # Append found boxes as physical occurrences.
    boxes: list[PhysicalBox] = []
    for item in result.get("found", []):
        boxes.append(
            PhysicalBox(
                barcode_value=item["barcode_value"],
                barcode_format=item.get("barcode_format", ""),
                label_index=item.get("label_index"),
            )
        )
    session.add_boxes(boxes)

    # Update expected_count from the audit (only if audit was available
    # and we don't already have a count).
    if result.get("audit_available") and session.expected_count == 0:
        session.expected_count = result.get("summary", {}).get(
            "visible_label_count", 0
        )

    return {
        "session_id": session_id,
        "status": session.status.value,
        "outcome": result["outcome"],
        "boxes_added": len(boxes),
        "total_boxes": len(session.boxes),
        "expected_count": session.expected_count,
        "discrepancy": {
            "expected": session.discrepancy.expected,
            "found": session.discrepancy.found,
            "missing": session.discrepancy.missing,
            "is_complete": session.discrepancy.is_complete,
        },
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

    # Use the new gateway API if available, fall back to the shim.
    if hasattr(gateway, "create_draft_order"):
        result = await gateway.create_draft_order(request)
        return {
            "order_id": result.order_id,
            "session_id": result.session_id,
            "status": result.status,
        }

    # Backward-compatible shim path (PriorityRepository.create_order).
    order_id = await gateway.create_order(
        session_id=request.session_id,
        customer_id=request.customer_id,
        branch_id=request.branch_id,
        action=request.action,
        items=request.items_as_dicts(),
    )
    return {"order_id": order_id, "session_id": request.session_id, "status": "draft"}


@router.post("/sessions/{session_id}/submit")
async def submit_session(session_id: str) -> dict[str, Any]:
    """Freeze the session and create a draft order via the runtime.

    Idempotency: same session → same key → double-submit returns the
    cached order ID, never creates a duplicate.

    State transitions:
        ACTIVE → SUBMITTING → SUBMITTED (success)
        ACTIVE → SUBMITTING → SUBMISSION_UNKNOWN (indeterminate)
        ACTIVE (stays) on pre-submit failure
    """
    session = _get_session(session_id)

    # Already submitted — return the cached result.
    if session.status == ReceivingSessionStatus.SUBMITTED:
        return {
            "session_id": session_id,
            "status": session.status.value,
            "order_id": session.external_order_id,
            "idempotent": True,
        }

    # Indeterminate — retry with the same key; the idempotency store
    # replays the outcome.
    if session.status == ReceivingSessionStatus.SUBMISSION_UNKNOWN:
        # Fall through to retry path below.
        pass
    elif session.status != ReceivingSessionStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "invalid_state",
                "message": f"Session is {session.status.value}; cannot submit.",
            },
        )

    # Freeze the session (ACTIVE → SUBMITTING).
    if session.status == ReceivingSessionStatus.ACTIVE:
        session.freeze()

    # Build the request from the frozen payload.
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
            idempotency_key=session.idempotency_key,
            idempotency_store=_idempotency_store,
        )
    except IndeterminateError as exc:
        session.mark_submission_unknown()
        return {
            "session_id": session_id,
            "status": session.status.value,
            "error": {"code": "submission_unknown", "message": str(exc)},
            "retry_recommended": True,
        }
    except RetryableError as exc:
        # Pre-submit failure — revert to ACTIVE so the user can retry/edit.
        session.revert_to_active()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "code": "submission_failed",
                "message": str(exc),
            },
        ) from exc
    except PermanentError as exc:
        session.revert_to_active()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "submission_permanent_error",
                "message": str(exc),
            },
        ) from exc

    order_id = result["order_id"]
    session.mark_submitted(order_id)

    return {
        "session_id": session_id,
        "status": session.status.value,
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
    session = _get_session(session_id)
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

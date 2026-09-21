"""Receiving flow API — the end-to-end vertical slice for Visual Receiving.

This router wires one safe end-to-end flow:

    POST /receiving/sessions             — create/resume session (no business
                                            context required; scan first)
    POST /receiving/sessions/{id}/images  — upload photo, run analyze_image(),
                                            append boxes
    POST /receiving/sessions/{id}/context — attach customer + branch + action
                                            (after scanning, before submit)
    POST /receiving/sessions/{id}/submit  — freeze, create draft order via runtime
    GET  /receiving/sessions/{id}         — inspect session

Scan-first flow: a session is created when the user starts scanning,
BEFORE they have chosen a customer/branch/action. Business context is
attached later via ``POST /context`` and validated there (and again at
submit). All three context fields are ``None`` until attached — never
empty strings — so "not yet chosen" is unambiguous.

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
it is never rebuilt from ``session_items``. The request passed to
``execute`` is rebuilt FROM the frozen payload on retry, so the same
idempotency key always pairs with the exact same payload.

One unresolved receiving session per participant: if a participant has
an ACTIVE session, creating a new session resumes the existing one
(200). A SUBMISSION_UNKNOWN session blocks new creation (409) until the
user retries/reconciles it.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Response,
    UploadFile,
    status,
)

from src.domain.receiving import (
    ALLOWED_ACTIONS,
    ReceivingSession,
    ReceivingSessionStatus,
)
from src.integrations.priority.models import (
    CreateDraftOrderRequest,
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

    customer_ids = {c.id for c in customers}
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

    branch_ids = {b.id for b in branches}
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


def _session_response(session: ReceivingSession) -> dict[str, Any]:
    """Build the standard session response dict from a ReceivingSession.

    Used by create/get/context endpoints so the response shape is
    consistent. ``customer_id`` / ``branch_id`` / ``action`` are ``None``
    until attached (never empty strings).
    """
    quantities = session.aggregate_quantities()
    return {
        "session_id": session.session_id,
        "status": session.status.value,
        "customer_id": session.customer_id,
        "branch_id": session.branch_id,
        "action": session.action,
        "participant_id": session.participant_id,
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


# ---------------------------------------------------------------------------
# POST /receiving/sessions — create or resume session (no context required)
# ---------------------------------------------------------------------------


@router.post("/sessions")
async def create_session(
    response: Response,
    participant_id: str = Form(
        "", description="Operator/participant ID (one unresolved session per participant)"
    ),
    customer_id: str = Form(
        "", description="Optional Priority customer ID (attach later via /context)"
    ),
    branch_id: str = Form(
        "", description="Optional Priority branch ID (attach later via /context)"
    ),
    action: str = Form(
        "",
        description="Optional action (create_order or verify_order_before_shipment; "
        "attach later via /context)",
    ),
) -> dict[str, Any]:
    """Create a new receiving session, or resume an existing ACTIVE one.

    Scan-first: business context (customer/branch/action) is OPTIONAL at
    creation. The session starts ACTIVE with no context; the user scans
    first, then attaches context via ``POST /context`` before submitting.

    One unresolved receiving session per participant:
    - If the participant has an ACTIVE session, resume it (200) — the
      caller treats 200 and 201 identically.
    - If the participant has a SUBMISSION_UNKNOWN session, return 409
      (must retry/resolve before creating a new one).

    Returns 201 when a new session is created, 200 when an existing
    ACTIVE session is resumed.
    """
    store = _get_receiving_store()

    # Resume an existing ACTIVE session for this participant, or block
    # on a SUBMISSION_UNKNOWN one. An empty participant_id skips the
    # lookup (each call gets a fresh session — used by tests/CLI).
    if participant_id.strip():
        existing = await store.find_open_submission_by_participant(participant_id)
        if existing is not None:
            if existing.status == ReceivingSessionStatus.SUBMISSION_UNKNOWN:
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
            # ACTIVE — resume it.
            resp = _session_response(existing)
            # 200, not 201 — the session already existed.
            response.status_code = status.HTTP_200_OK
            return resp

    session_id = str(uuid.uuid4())
    # Use session_id as participant_id when none is provided, so the
    # session graph can find the session row by participant_id.
    effective_participant_id = participant_id.strip() or session_id

    # Context is optional at creation. If provided, validate it; if not,
    # store None (the DB columns are nullable).
    ctx_customer = customer_id.strip() or None
    ctx_branch = branch_id.strip() or None
    ctx_action = action.strip() or None
    if ctx_action is not None and ctx_action not in ALLOWED_ACTIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "invalid_action", "message": "Unsupported action."},
        )
    if ctx_customer is not None or ctx_branch is not None:
        # If any context field is provided, require all three and validate.
        if ctx_customer is None or ctx_branch is None or ctx_action is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": "partial_context",
                    "message": (
                        "Provide all of customer_id, branch_id, and action, "
                        "or none (attach later via /context)."
                    ),
                },
            )
        await _validate_customer_and_branch(ctx_customer, ctx_branch)

    await store.create_receiving_session(
        session_id,
        customer_id=ctx_customer,
        branch_id=ctx_branch,
        action=ctx_action,
        participant_id=effective_participant_id,
    )

    session = await store.get_receiving_session(session_id)
    assert session is not None  # just created
    response.status_code = status.HTTP_201_CREATED
    return _session_response(session)


# ---------------------------------------------------------------------------
# POST /receiving/sessions/{id}/context — attach business context
# ---------------------------------------------------------------------------


@router.post("/sessions/{session_id}/context")
async def attach_context(
    session_id: str,
    customer_id: str = Form(..., description="Priority customer ID"),
    branch_id: str = Form(..., description="Priority branch ID"),
    action: str = Form(
        ..., description="create_order or verify_order_before_shipment"
    ),
) -> dict[str, Any]:
    """Attach business context (customer + branch + action) to an ACTIVE
    session, after scanning and before submitting to Priority.

    Validates that the customer exists, the branch exists, and the
    branch belongs to the customer. Rejects if the session is not
    ACTIVE (409) — context cannot be changed once submission has started.
    """
    if not customer_id.strip() or not branch_id.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "order_context_required",
                "message": "customer_id and branch_id are required.",
            },
        )
    if action not in ALLOWED_ACTIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "invalid_action", "message": "Unsupported action."},
        )

    await _validate_customer_and_branch(customer_id, branch_id)

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
                    f"cannot attach context."
                ),
            },
        )

    ok = await store.update_session_context(
        session_id,
        customer_id=customer_id,
        branch_id=branch_id,
        action=action,
    )
    if not ok:
        # Lost the race — another caller moved the session out of ACTIVE.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "session_not_editable",
                "message": "Session is no longer active.",
            },
        )

    session = await store.get_receiving_session(session_id)
    assert session is not None
    return _session_response(session)


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

    # Use the receiving session's participant_id so the ingest session
    # is 1:1 with the receiving session. The ingest session accumulates
    # boxes; the receiving session tracks submission state. They share
    # the same session_id row in the sessions table.
    ingest_participant_id = session.participant_id or session_id

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

    # Require business context before submitting. The scan-first flow
    # creates sessions without context; it must be attached via
    # POST /context before the draft order can be created.
    if not session.has_context:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "order_context_required",
                "message": (
                    "Choose a customer, branch, and action before submitting."
                ),
            },
        )

    # Build the frozen order payload from the current session contents
    # (ACTIVE path). On retry from SUBMISSION_UNKNOWN, the frozen
    # payload is already persisted — reuse it verbatim. The request
    # passed to execute() is then rebuilt FROM the final payload, so
    # the same idempotency key always pairs with the exact same payload.
    if session.status == ReceivingSessionStatus.ACTIVE:
        quantities = session.aggregate_quantities()
        frozen_payload: dict[str, Any] = {
            "session_id": session.session_id,
            "customer_id": session.customer_id,
            "branch_id": session.branch_id,
            "action": session.action,
            "items": [
                {
                    "barcode_value": value,
                    "barcode_format": fmt,
                    "quantity": qty,
                }
                for value, fmt, qty in quantities
            ],
        }
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
        if frozen is None:
            # Defensive: SUBMISSION_UNKNOWN without a frozen payload
            # shouldn't happen (freeze_submission persists it before the
            # indeterminate transition). Treat as a server error.
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "code": "frozen_payload_missing",
                    "message": "No frozen payload found for retry.",
                },
            )
        frozen_payload = frozen

    # Rebuild the request from the final payload — the same dict that
    # is persisted and paired with the idempotency key. This restores
    # the invariant: same key → exact same payload, even on retry.
    request = CreateDraftOrderRequest.from_payload(frozen_payload)

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
            {
                "barcode_value": item.barcode_value,
                "barcode_format": item.barcode_format,
                "quantity": item.quantity,
            }
            for item in request.items
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
    return _session_response(session)

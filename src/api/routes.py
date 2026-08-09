import logging
import time
from io import BytesIO as _BytesIO
from typing import Any
from uuid import uuid4

import langsmith as ls
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from langsmith.schemas import Attachment
from langsmith.utils import LangSmithError
from PIL import Image, UnidentifiedImageError

from src.config import Settings, get_settings
from src.models.barcode import ScanResponse, ScanStatus
from src.models.feedback import FeedbackRequest, FeedbackResponse
from src.models.upload import generate_upload_id
from src.ingest.analyze import analyze_image_async
from src.ingest.scanner import BarcodeScanner
from src.api.feedback import submit_upload_feedback
from src.persistence import create_run as persist_create_run, complete_run_from_dict, fail_run as persist_fail_run
from src.observability.versions import collect_versions

logger = logging.getLogger(__name__)

router = APIRouter()


def get_scanner() -> BarcodeScanner:
    return BarcodeScanner()


def _get_run_repo():
    """Get the module-level run_repo from src.main (set by lifespan)."""
    from src.main import run_repo
    return run_repo


def _sanitize_scan_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Keep the filename and content type out of the traced raw bytes."""
    file = inputs.get("file")
    return {
        "filename": getattr(file, "filename", None),
        "content_type": getattr(file, "content_type", None),
    }


def _attach_image_to_run(image_bytes: bytes, mime_type: str) -> None:
    """Attach the uploaded image to the current LangSmith run as a viewable attachment."""
    run = ls.get_current_run_tree()
    if run is not None:
        run.attachments = {
            "uploaded_image": Attachment(mime_type=mime_type, data=image_bytes)
        }


@router.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    tags=["feedback"],
    summary="Record Correct/Incorrect feedback on a LangSmith trace",
)
def create_feedback(payload: FeedbackRequest) -> FeedbackResponse:
    try:
        score = submit_upload_feedback(
            trace_id=payload.trace_id,
            correct=payload.correct,
            comment=payload.comment,
        )
    except LangSmithError as exc:
        raise HTTPException(
            status_code=502,
            detail="Could not submit feedback to LangSmith",
        ) from exc

    return FeedbackResponse(
        status="recorded",
        trace_id=payload.trace_id,
        score=score,
    )


# ---------------------------------------------------------------------------
# Traced pipeline functions — accept langsmith_extra at call time so the
# route handler can assign an explicit run_id (trace_id) and metadata.
# ---------------------------------------------------------------------------


@ls.traceable(
    name="web_scan_barcode",
    run_type="chain",
    tags=["barcode-scanner", "web", "scanner-only"],
    metadata={
        "channel": "web",
        "endpoint": "/barcode/scan",
        "app_version": "v1",
    },
    process_inputs=_sanitize_scan_inputs,
)
async def _traced_scan(
    *,
    file: UploadFile,
    image_bytes: bytes,
    upload_id: str,
    trace_id: str,
    source: str,
    settings: Settings,
    scanner: BarcodeScanner,
) -> ScanResponse:
    """Traced scanner-only path. Called with langsmith_extra to set run_id."""
    filename = file.filename or "unknown"
    upload_bytes = len(image_bytes)

    # Attach the uploaded image to the LangSmith trace so it's viewable in the UI.
    _attach_image_to_run(image_bytes, (file.content_type or "image/jpeg"))

    # Persist run row (status=pending) before processing
    repo = _get_run_repo()
    await persist_create_run(
        repo,
        run_id=upload_id,
        source=source,
        endpoint="/barcode/scan",
        trace_id=trace_id,
        session_id=upload_id,
        filename=filename,
        upload_bytes=upload_bytes,
    )

    try:
        t0 = time.perf_counter()
        barcodes = scanner.scan_bytes(image_bytes)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "invalid_image", "message": str(exc)},
        ) from exc

    with Image.open(_BytesIO(image_bytes)) as img:
        image_width, image_height = img.size

    logger.info(
        "scan upload_id=%s trace_id=%s filename=%s upload_bytes=%d "
        "dims=%dx%d count=%d elapsed_ms=%d",
        upload_id, trace_id, filename, upload_bytes,
        image_width, image_height, len(barcodes), elapsed_ms,
    )

    run = ls.get_current_run_tree()
    if run is not None:
        run.metadata.update(
            {
                "upload_id": upload_id,
                "source": source,
                "filename": filename,
                "upload_bytes": upload_bytes,
                "image_width": image_width,
                "image_height": image_height,
                "barcode_count": len(barcodes),
                "scan_status": "found" if barcodes else "not_found",
                "elapsed_ms": elapsed_ms,
                "barcode_values": [b.value for b in barcodes],
            }
        )

    # Persist completed scan run
    try:
        from src.ingest.models import IngestResult, IngestStatus, DetectedItem, RunMetrics
        scan_result = IngestResult(
            status=IngestStatus.COMPLETE if barcodes else IngestStatus.NEEDS_USER_INPUT,
            items=[
                DetectedItem(
                    barcode_value=b.value,
                    barcode_format=b.format,
                    barcode_bbox={
                        "x1": b.bounding_box.x1, "y1": b.bounding_box.y1,
                        "x2": b.bounding_box.x2, "y2": b.bounding_box.y2,
                    } if b.bounding_box else None,
                )
                for b in barcodes
            ],
            metrics=RunMetrics(
                elapsed_ms=elapsed_ms,
                scanner_count=len(barcodes),
            ),
            image_width=image_width,
            image_height=image_height,
        )
        await repo.complete_run(upload_id, scan_result)
    except Exception:
        logger.warning("Failed to persist scan run %s to DB", upload_id, exc_info=True)

    return ScanResponse(
        upload_id=upload_id,
        trace_id=trace_id,
        source=source,
        status=ScanStatus.FOUND if barcodes else ScanStatus.NOT_FOUND,
        count=len(barcodes),
        image_width=image_width,
        image_height=image_height,
        filename=filename,
        upload_bytes=upload_bytes,
        elapsed_ms=elapsed_ms,
        barcodes=barcodes,
    )


@ls.traceable(
    name="web_analyze_barcode",
    run_type="chain",
    tags=["barcode-scanner", "web", "pipeline", "gemini"],
    metadata={
        "channel": "web",
        "endpoint": "/barcode/analyze",
        "app_version": "v1",
    },
    process_inputs=_sanitize_scan_inputs,
)
async def _traced_analyze(
    *,
    file: UploadFile,
    image_bytes: bytes,
    upload_id: str,
    trace_id: str,
    source: str,
) -> dict:
    """Traced full-pipeline path. Called with langsmith_extra to set run_id."""
    filename = file.filename or "unknown"
    upload_bytes = len(image_bytes)

    # Attach the uploaded image to the LangSmith trace so it's viewable in the UI.
    _attach_image_to_run(image_bytes, (file.content_type or "image/jpeg"))

    # Persist run row (status=pending) before processing
    repo = _get_run_repo()
    await persist_create_run(
        repo,
        run_id=upload_id,
        source=source,
        endpoint="/barcode/analyze",
        trace_id=trace_id,
        session_id=upload_id,
        filename=filename,
        upload_bytes=upload_bytes,
    )

    t0 = time.perf_counter()
    try:
        # Call analyze_image_async directly — stays in FastAPI's event loop
        # so the Postgres checkpointer's asyncio.Lock works (M16C).
        # Pass upload_id as thread_id for checkpoint persistence.
        result = await analyze_image_async(
            image_bytes, thread_id=upload_id
        )
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        await persist_fail_run(repo, upload_id, exc)
        raise
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    result["upload_id"] = upload_id
    result["trace_id"] = trace_id
    result["source"] = source
    result["filename"] = filename
    result["upload_bytes"] = upload_bytes
    result["elapsed_ms"] = elapsed_ms

    logger.info(
        "analyze upload_id=%s trace_id=%s filename=%s upload_bytes=%d "
        "dims=%dx%d outcome=%s found=%d missing=%d unassigned=%d elapsed_ms=%d",
        upload_id, trace_id, filename, upload_bytes,
        result.get("image_width", 0), result.get("image_height", 0),
        result.get("outcome"),
        result.get("summary", {}).get("found_count", 0),
        result.get("summary", {}).get("missing_count", 0),
        result.get("summary", {}).get("unassigned_count", 0),
        elapsed_ms,
    )

    summary = result.get("summary", {})
    recovery_info = summary.get("recovery", {})
    scanner_count = len(result.get("found", [])) + summary.get("missing_count", 0)
    vision_count = summary.get("visible_label_count", 0)
    recovery_attempted = recovery_info.get("attempted", False)
    recovery_labels_resolved = recovery_info.get("labels_resolved", 0)
    outcome = result.get("outcome", "retryable_error")
    # Normalize outcome to the canonical final_status values used by ingest_one.
    final_status = "needs_user_input" if outcome == "needs_better_photo" else outcome

    run = ls.get_current_run_tree()
    if run is not None:
        run.metadata.update(
            {
                "upload_id": upload_id,
                "source": source,
                "filename": filename,
                "upload_bytes": upload_bytes,
                "image_width": result.get("image_width", 0),
                "image_height": result.get("image_height", 0),
                "outcome": outcome,
                "final_status": final_status,
                "audit_available": result.get("audit_available"),
                "visible_label_count": vision_count,
                "found_count": summary.get("found_count", 0),
                "missing_count": summary.get("missing_count", 0),
                "unassigned_count": summary.get("unassigned_count", 0),
                "all_found": summary.get("all_found", False),
                "elapsed_ms": elapsed_ms,
                "latency_ms": elapsed_ms,
                "scanner_count": scanner_count,
                "vision_count": vision_count,
                "found_values": [
                    f.get("barcode_value") for f in result.get("found", [])
                ],
                "missing_labels": [
                    m.get("label_index") for m in result.get("missing", [])
                ],
                "has_annotated_image": bool(result.get("annotated_image_b64")),
                "recovery_attempted": recovery_attempted,
                "recovery_labels_tried": recovery_info.get("labels_tried", 0),
                "recovery_barcodes_found": recovery_info.get("barcodes_found", 0),
                "recovery_labels_resolved": recovery_labels_resolved,
                # Derived fields for dashboard grouping
                "scanner_vision_match": scanner_count == vision_count,
                "count_delta": vision_count - scanner_count,
                "recovery_succeeded": recovery_labels_resolved > 0,
            }
        )

    # Persist completed run with versions
    try:
        versions = collect_versions().model_dump()
        await complete_run_from_dict(repo, upload_id, result, elapsed_ms, versions=versions)
    except Exception:
        logger.warning("Failed to persist run %s to DB", upload_id, exc_info=True)

    outcome = result.get("outcome", "retryable_error")
    if outcome == "complete":
        return await _send_complete_reply(result)
    elif outcome == "needs_better_photo":
        return await _send_needs_better_photo_reply(result)
    else:
        return result


@ls.traceable(
    name="send_complete_reply",
    run_type="chain",
    tags=["barcode-scanner", "web", "reply"],
)
async def _send_complete_reply(result: dict[str, Any]) -> dict[str, Any]:
    """Build the web response for a complete outcome.

    Mirrors the WhatsApp ``send_complete_reply`` — the web equivalent of
    sending the barcode list is returning the JSON response with found barcodes.
    """
    found = result.get("found", [])
    unassigned = result.get("unassigned", [])
    reply_run = ls.get_current_run_tree()
    if reply_run is not None:
        reply_run.metadata.update(
            {
                "total_barcodes": len(found) + len(unassigned),
                "found_count": len(found),
                "unassigned_count": len(unassigned),
            }
        )
    return result


@ls.traceable(
    name="send_needs_better_photo_reply",
    run_type="chain",
    tags=["barcode-scanner", "web", "reply"],
)
async def _send_needs_better_photo_reply(
    result: dict[str, Any],
) -> dict[str, Any]:
    """Build the web response for a needs_better_photo outcome.

    Mirrors the WhatsApp ``send_needs_better_photo_reply`` — the web equivalent
    of sending the annotated image is returning the JSON response with
    ``annotated_image_b64`` and ``missing`` labels.
    """
    reply_run = ls.get_current_run_tree()
    if reply_run is not None:
        reply_run.metadata.update(
            {
                "has_annotated_image": bool(result.get("annotated_image_b64")),
                "missing_count": len(result.get("missing", [])),
                "message": result.get("message", ""),
            }
        )
    return result


# ---------------------------------------------------------------------------
# Route handlers — thin HTTP layer. Generate upload_id + trace_id, read the
# file, validate, then call the traced function with langsmith_extra.
# ---------------------------------------------------------------------------


@router.post(
    "/barcode/scan",
    response_model=ScanResponse,
    tags=["barcode"],
    summary="Scan all barcodes visible in one product photo (scanner only)",
)
async def scan_barcode(
    file: UploadFile = File(..., description="JPEG, PNG, or WebP product photo"),
    settings: Settings = Depends(get_settings),
    scanner: BarcodeScanner = Depends(get_scanner),
) -> ScanResponse:
    content_type = (file.content_type or "").lower()
    if content_type not in settings.allowed_content_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={
                "code": "unsupported_image_type",
                "message": f"Unsupported content type: {content_type or 'unknown'}",
                "allowed": sorted(settings.allowed_content_types),
            },
        )

    image_bytes = await file.read(settings.max_upload_bytes + 1)
    await file.close()

    if len(image_bytes) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "code": "image_too_large",
                "message": f"Maximum upload size is {settings.max_upload_bytes} bytes.",
            },
        )

    upload_id = generate_upload_id()
    trace_id = str(uuid4())
    source = "web"

    return await _traced_scan(
        file=file,
        image_bytes=image_bytes,
        upload_id=upload_id,
        trace_id=trace_id,
        source=source,
        settings=settings,
        scanner=scanner,
        langsmith_extra={
            "run_id": trace_id,
            "metadata": {
                "upload_id": upload_id,
                "source": source,
            },
            "tags": [f"source:{source}", "image-upload"],
        },
    )


@router.post(
    "/barcode/analyze",
    tags=["barcode"],
    summary="Full pipeline: scanner + Gemini audit + annotated image on missing",
)
async def analyze_barcode(
    file: UploadFile = File(..., description="JPEG, PNG, or WebP product photo"),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Run the full analyze_image() pipeline (scanner + Gemini spatial audit).

    Returns the product-shaped result with ``outcome`` (complete /
    needs_better_photo / retryable_error), ``found``, ``missing``,
    ``unassigned``, and — on needs_better_photo — ``annotated_image_b64``
    (base64 PNG with red circles around missing regions) plus ``message``.

    Requires ``GEMINI_API_KEY`` in the environment.
    """
    content_type = (file.content_type or "").lower()
    if content_type not in settings.allowed_content_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={
                "code": "unsupported_image_type",
                "message": f"Unsupported content type: {content_type or 'unknown'}",
                "allowed": sorted(settings.allowed_content_types),
            },
        )

    image_bytes = await file.read(settings.max_upload_bytes + 1)
    await file.close()

    if len(image_bytes) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "code": "image_too_large",
                "message": f"Maximum upload size is {settings.max_upload_bytes} bytes.",
            },
        )

    upload_id = generate_upload_id()
    trace_id = str(uuid4())
    source = "web"

    return await _traced_analyze(
        file=file,
        image_bytes=image_bytes,
        upload_id=upload_id,
        trace_id=trace_id,
        source=source,
        langsmith_extra={
            "run_id": trace_id,
            "metadata": {
                "upload_id": upload_id,
                "source": source,
            },
            "tags": [f"source:{source}", "image-upload"],
        },
    )


# ---------------------------------------------------------------------------
# Multi-image session endpoint (M16C)
# ---------------------------------------------------------------------------


def _get_session_repo():
    """Get the session repository singleton (Postgres or NoOp)."""
    from src.main import _db_pool

    if _db_pool is not None:
        from src.session_repository import SessionRepository

        return SessionRepository(_db_pool)
    from src.session_repository import NoOpSessionRepository

    return NoOpSessionRepository()


@router.post(
    "/barcode/session",
    tags=["barcode"],
    summary="Multi-image session: submit a photo to an ingest session",
)
async def session_ingest(
    file: UploadFile = File(..., description="JPEG, PNG, or WebP product photo"),
    participant_id: str = Form(..., description="Stable client ID (UUID in localStorage for web, phone number for WhatsApp)"),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Submit an image to a multi-image ingest session.

    The ``participant_id`` identifies the user across requests. For web,
    the browser generates a UUID on first visit and stores it in
    localStorage. For WhatsApp, it's the sender's phone number.

    The server resolves the active session for this participant
    automatically. If none exists (or the previous one is
    complete/expired/closed), a new session is created.

    Returns a ``SessionResult`` with:

    - ``session_id``: internal session ID (for debugging/admin)
    - ``status``: ``active`` (need more photos) or ``complete`` (all boxes found)
    - ``expected_count``: total visible boxes (from first photo's audit)
    - ``found_count``: unique barcodes confirmed so far
    - ``missing_count``: boxes still without barcodes
    - ``items``: confirmed barcodes
    - ``missing``: unresolved boxes with label metadata
    - ``message``: human-readable prompt for the next photo
    - ``image_count``: how many photos submitted so far

    Requires ``GEMINI_API_KEY`` in the environment.
    """
    content_type = (file.content_type or "").lower()
    if content_type not in settings.allowed_content_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={
                "code": "unsupported_image_type",
                "message": f"Unsupported content type: {content_type or 'unknown'}",
                "allowed": sorted(settings.allowed_content_types),
            },
        )

    image_bytes = await file.read(settings.max_upload_bytes + 1)
    await file.close()

    if len(image_bytes) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "code": "image_too_large",
                "message": f"Maximum upload size is {settings.max_upload_bytes} bytes.",
            },
        )

    repo = _get_session_repo()

    from src.ingest.session_graph import run_session_graph

    result = await run_session_graph(
        image_bytes,
        repo=repo,
        source="web",
        channel="web",
        participant_id=participant_id,
    )

    return result.model_dump(mode="json")


@router.get(
    "/barcode/session/{session_id}",
    tags=["barcode"],
    summary="Get session state without submitting a new image",
)
async def get_session(session_id: str) -> dict:
    """Get the current state of an ingest session.

    Returns the same ``SessionResult`` shape as the POST endpoint, but
    without running a new image. Useful for polling or resuming a session
    after a page refresh.
    """
    repo = _get_session_repo()
    result = await repo.to_result(session_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "session_not_found",
                "message": f"Session {session_id} not found.",
            },
        )
    return result.model_dump(mode="json")


@router.delete(
    "/barcode/session/{session_id}",
    tags=["barcode"],
    summary="Close an ingest session",
)
async def close_session(session_id: str) -> dict:
    """Explicitly close a session.

    The session is marked ``closed`` and can no longer accept new images.
    The client starts a fresh session by calling ``POST /barcode/session``
    without a ``session_id``.

    Returns the final session state. Returns 404 if the session doesn't exist.
    """
    repo = _get_session_repo()
    result = await repo.to_result(session_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "session_not_found",
                "message": f"Session {session_id} not found.",
            },
        )
    await repo.close_session(session_id)
    result = await repo.to_result(session_id)
    if result:
        return result.model_dump(mode="json")
    return {"session_id": session_id, "status": "closed"}


@router.post(
    "/barcode/session/select",
    tags=["barcode"],
    summary="Select a candidate barcode to resolve a missing label",
)
async def select_session_candidate(
    participant_id: str = Form(..., description="Client participant ID"),
    barcode_value: str = Form(..., description="Barcode value from the candidates list"),
) -> dict:
    """Resolve a user selection from the candidates list.

    When a session is in ``needs_user_selection`` status, the client shows
    the candidate barcodes. The user picks one, and the client calls this
    endpoint with the selected ``barcode_value``.

    The server resolves one missing label with the selected barcode and
    returns the updated session state. If all labels are resolved, the
    session becomes ``complete``.

    Returns 400 if the session is not awaiting selection or the barcode
    is not in the candidates list.
    """
    repo = _get_session_repo()

    # Find the needs_user_selection session for this participant.
    session = await repo.find_active_by_participant("web", participant_id)
    if session is None or session.get("status") != "needs_user_selection":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "no_selection_pending",
                "message": "No session awaiting selection for this participant.",
            },
        )
    session_id = session["id"]

    from src.ingest.session_graph import select_candidate

    try:
        result = await select_candidate(session_id, barcode_value, repo=repo)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "selection_error", "message": str(exc)},
        )

    return result.model_dump(mode="json")

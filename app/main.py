"""FastAPI application: 360dialog WhatsApp webhook + barcode image analysis.

Webhook flow:
    WhatsApp message → /webhook/360dialog
        → deduplicate message ID
        → schedule background processing (BackgroundTasks)
        → return HTTP 200 immediately

Background processing (per message type):
    image  → download → analyze_image() →
        complete:           send barcode list as text
        needs_better_photo: send annotated JPEG with caption
        retryable_error:    send temporary-failure text (never blames photo)
    audio  → transcribe → send "please send a photo" reply
    text   → send "please send a photo" reply

NOTE: BackgroundTasks is not durable — a process restart after the webhook
acknowledgment can lose the job. Redis/Temporal/a queue is the production
upgrade.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import langsmith as ls
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from langsmith.schemas import Attachment
from PIL import Image

from app.api.admin import router as admin_router
from app.api.routes import router
from app.config import settings
from src.models.upload import generate_upload_id
from src.ingest.analyze import analyze_image
from app.services.dialog360 import Dialog360Client, iter_incoming_messages
from app.services.transcribe import (
    Transcriber,
    handle_360dialog_audio_message,
)
from app.services.transcription_factory import get_transcriber

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO)
)

for noisy in ("httpx", "httpcore", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

wa_client: Dialog360Client | None = None
transcriber: Transcriber | None = None

_processed_msg_ids: set[str] = set()
_PROCESSED_MSG_ID_MAX = 5000

# 360dialog supports outgoing images up to 5 MB.
_D360_MAX_IMAGE_BYTES = 5_000_000
# Inbound image MIME types we accept for analysis.
_SUPPORTED_IMAGE_MIMES = {"image/jpeg", "image/png", "image/webp"}
# Inline image MIME → file suffix map (kept local to avoid coupling to the
# audio-specific suffix_from_mime in transcribe.py).
_IMAGE_SUFFIX_BY_MIME = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global wa_client, transcriber

    wa_client = Dialog360Client(settings)
    transcriber = get_transcriber(settings)

    yield

    if transcriber is not None:
        await transcriber.close()


app = FastAPI(lifespan=lifespan)

# Experiment-only CORS: allow any origin so the local Vite dev server and
# ngrok tunnels can call the API from the browser. Safe because this app
# uses no credentials/cookies. Tighten before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(admin_router)

# NOTE: The static files mount at "/" must be added AFTER all API routes
# (including the @app.post/@app.get routes defined below). Starlette matches
# routes in registration order, and a Mount("/") catch-all would shadow any
# route added after it. The mount is therefore registered at the very end of
# this module — see the bottom of the file.


def sanitize_process_message_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Prevent the sender phone number and raw provider payload from being traced."""
    msg = inputs.get("msg", {})

    return {
        "msg": {
            "id": msg.get("id"),
            "type": msg.get("type"),
            "has_text": bool(msg.get("text")),
            "has_media_id": bool(msg.get("media_id")),
            "mime_type": msg.get("mime_type"),
            "has_caption": bool(msg.get("caption")),
        }
    }


@ls.traceable(
    name="process_whatsapp_message",
    run_type="chain",
    tags=["barcode-scanner", "whatsapp", "image-analysis"],
    metadata={
        "channel": "whatsapp",
        "provider": "360dialog",
        "app_version": "v1",
    },
    process_inputs=sanitize_process_message_inputs,
)
async def _traced_process_message(
    msg: dict[str, Any],
    *,
    trace_id: str,
) -> None:
    """Traced WhatsApp message processor. Called with langsmith_extra to set run_id."""
    sender = msg["from"]
    msg_id = msg["id"]
    msg_type = msg["type"]

    run = ls.get_current_run_tree()

    if run is not None:
        run.metadata.update(
            {
                "environment": settings.app_env,
                "message_type": msg_type,
                "provider_message_id": msg_id,
                "trace_id": trace_id,
            }
        )

    # --- Deduplication ---------------------------------------------------
    # In-memory set is sufficient for local/dev single-process. For production
    # use Redis: SET message:{wamid} processing NX EX 3600, then mark complete
    # after replying. Webhook retries can re-deliver the same wamid.
    if msg_id in _processed_msg_ids:
        logger.info(
            "Skipping duplicate message id=%s from=%s",
            msg_id,
            sender,
        )
        if run is not None:
            run.metadata["processing_status"] = "duplicate"
        return

    _processed_msg_ids.add(msg_id)
    if len(_processed_msg_ids) > _PROCESSED_MSG_ID_MAX:
        _processed_msg_ids.clear()
        _processed_msg_ids.add(msg_id)

    try:
        # --- Typing indicator (best-effort) -----------------------------
        if wa_client is not None:
            try:
                await wa_client.send_typing_indicator(msg_id)
            except Exception:
                logger.warning(
                    "Failed to send typing indicator for msg=%s",
                    msg_id,
                )

        if msg_type == "image":
            await process_image_message(msg, sender, run, trace_id=trace_id)
        elif msg_type == "document":
            # Documents with image MIME types are treated as image uploads.
            # Non-image documents (e.g. PDF) get a "please send a photo" reply.
            mime = (msg.get("mime_type") or "").lower()
            if mime in _SUPPORTED_IMAGE_MIMES:
                await process_image_message(msg, sender, run, trace_id=trace_id)
            else:
                await send_photo_instruction(sender)
                if run is not None:
                    run.metadata["processing_status"] = "unsupported_document_type"
        elif msg_type == "audio":
            await process_audio_message(msg, sender, run)
        elif msg_type == "text":
            await send_photo_instruction(sender)
            if run is not None:
                run.metadata["processing_status"] = "completed"
        else:
            if run is not None:
                run.metadata["processing_status"] = "unsupported_message_type"
            return

    except Exception as exc:
        logger.exception(
            "Failed to process message trace_id=%s from=%s type=%s msg_id=%s",
            trace_id,
            sender,
            msg_type,
            msg_id,
        )
        if run is not None:
            run.metadata.update(
                {
                    "processing_status": "failed",
                    "error_type": type(exc).__name__,
                }
            )
        # Swallow — background-task failures must not crash the server.
        return


async def process_message(msg: dict[str, Any]) -> None:
    """Thin dispatcher — generates trace_id, calls the traced function.

    This is the entry point for BackgroundTasks. It generates a stable
    trace_id (UUID4) and passes it to _traced_process_message via
    langsmith_extra so the LangSmith root run ID is explicitly set.
    """
    trace_id = str(uuid4())
    await _traced_process_message(
        msg,
        trace_id=trace_id,
        langsmith_extra={
            "run_id": trace_id,
            "metadata": {
                "source": "whatsapp",
            },
            "tags": ["source:whatsapp", "image-upload"],
        },
    )


# ---------------------------------------------------------------------------
# Per-type handlers
# ---------------------------------------------------------------------------


@ls.traceable(
    name="process_image_message",
    run_type="chain",
    tags=["barcode-scanner", "whatsapp", "image"],
)
async def process_image_message(
    msg: dict[str, Any],
    sender: str,
    run: Any,
    *,
    trace_id: str = "",
) -> None:
    """Download, analyze, and reply to an inbound WhatsApp image.

    Handles both ``image`` messages (camera/gallery — no filename) and
    ``document`` messages with image MIME types (preserves the original
    filename in ``msg["filename"]``).
    """
    msg_id = msg.get("id", "")
    media_id = msg.get("media_id", "")
    mime_type = (msg.get("mime_type") or "").lower()
    filename = msg.get("filename", "") or ""

    sub_run = ls.get_current_run_tree()
    if sub_run is not None:
        sub_run.metadata.update(
            {
                "sender": sender,
                "media_id": media_id,
                "mime_type": mime_type,
                "trace_id": trace_id,
                "filename": filename or None,
                "whatsapp_msg_type": msg.get("type", ""),
            }
        )

    if not media_id:
        logger.error("Image message missing media_id from=%s", sender)
        if run is not None:
            run.metadata["processing_status"] = "missing_media_id"
        await _safe_send_text(sender, "Could not read the image. Please send it again.")
        return

    if mime_type not in _SUPPORTED_IMAGE_MIMES:
        logger.info("Unsupported image mime=%s from=%s", mime_type, sender)
        if run is not None:
            run.metadata["processing_status"] = "unsupported_image_mime"
        await _safe_send_text(
            sender, "Please send the photo as a JPEG, PNG, or WebP image."
        )
        return

    # --- Download media --------------------------------------------------
    suffix = _IMAGE_SUFFIX_BY_MIME.get(mime_type, ".bin")
    temp_path: Path | None = None
    try:
        try:
            logger.info("Downloading media media_id=%s from=%s", media_id, sender)
            temp_path = await wa_client.download_media_to_tempfile(  # type: ignore[union-attr]
                media_id=media_id,
                suffix=suffix,
            )
            file_size = temp_path.stat().st_size
            logger.info(
                "Downloaded media to %s (%d bytes) from=%s",
                temp_path,
                file_size,
                sender,
            )
            if sub_run is not None:
                sub_run.metadata.update(
                    {"downloaded_bytes": file_size, "temp_path": str(temp_path)}
                )
                # Attach the downloaded image to the LangSmith trace.
                try:
                    image_data = temp_path.read_bytes()
                    sub_run.attachments = {
                        "uploaded_image": Attachment(
                            mime_type=mime_type or "image/jpeg",
                            data=image_data,
                        )
                    }
                except Exception:
                    logger.warning(
                        "Failed to attach image to LangSmith trace from=%s",
                        sender,
                    )
        except Exception:
            logger.exception(
                "image media_download failed msg_id=%s sender=%s media_id=%s",
                msg_id,
                sender,
                media_id,
            )
            if sub_run is not None:
                sub_run.metadata["stage"] = "media_download_failed"
            await _safe_send_text(
                sender, "Could not download the image. Please try sending it again."
            )
            return

        # --- Analyze -----------------------------------------------------
        upload_id = generate_upload_id()
        result: dict[str, Any]
        try:
            logger.info(
                "Starting analyze_image upload_id=%s trace_id=%s for %s from=%s",
                upload_id, trace_id, temp_path, sender,
            )
            result = await asyncio.to_thread(analyze_image, temp_path)
            result["upload_id"] = upload_id
            result["trace_id"] = trace_id
            result["source"] = "whatsapp"
            result["filename"] = filename or "unknown"
            outcome = result.get("outcome", "retryable_error")
            summary = result.get("summary", {})
            logger.info(
                "analyze_image complete upload_id=%s trace_id=%s filename=%s "
                "outcome=%s found=%s missing=%s from=%s",
                upload_id,
                trace_id,
                filename or "unknown",
                outcome,
                summary.get("found_count", 0),
                summary.get("missing_count", 0),
                sender,
            )
            if sub_run is not None:
                sub_run.metadata.update(
                    {
                        "upload_id": upload_id,
                        "trace_id": trace_id,
                        "source": "whatsapp",
                        "filename": filename or None,
                        "outcome": outcome,
                        "found_count": summary.get("found_count", 0),
                        "missing_count": summary.get("missing_count", 0),
                        "unassigned_count": summary.get("unassigned_count", 0),
                        "visible_label_count": summary.get("visible_label_count", 0),
                        "audit_available": result.get("audit_available", False),
                        "recovery_attempted": summary.get("recovery", {}).get(
                            "attempted", False
                        ),
                        "recovery_labels_tried": summary.get("recovery", {}).get(
                            "labels_tried", 0
                        ),
                        "recovery_barcodes_found": summary.get("recovery", {}).get(
                            "barcodes_found", 0
                        ),
                        "recovery_labels_resolved": summary.get("recovery", {}).get(
                            "labels_resolved", 0
                        ),
                    }
                )
        except Exception as exc:
            logger.exception(
                "image analyze failed upload_id=%s trace_id=%s msg_id=%s "
                "sender=%s media_id=%s",
                upload_id,
                trace_id,
                msg_id,
                sender,
                media_id,
            )
            if sub_run is not None:
                sub_run.metadata.update(
                    {"stage": "analyze_failed", "error_type": type(exc).__name__}
                )
            await _safe_send_text(
                sender,
                "The barcode service is temporarily unavailable. "
                "Please try sending the photo again.",
            )
            return

        outcome = result.get("outcome", "retryable_error")
        summary = result.get("summary", {})
        final_status = "needs_user_input" if outcome == "needs_better_photo" else outcome

        if run is not None:
            recovery_info = summary.get("recovery", {})
            scanner_count = summary.get("found_count", 0) + summary.get("missing_count", 0)
            vision_count = summary.get("visible_label_count", 0)
            recovery_labels_resolved = recovery_info.get("labels_resolved", 0)
            run.metadata.update(
                {
                    "upload_id": upload_id,
                    "trace_id": trace_id,
                    "source": "whatsapp",
                    "filename": filename or None,
                    "outcome": outcome,
                    "final_status": final_status,
                    "found_count": summary.get("found_count", 0),
                    "missing_count": summary.get("missing_count", 0),
                    "unassigned_count": summary.get("unassigned_count", 0),
                    "scanner_count": scanner_count,
                    "vision_count": vision_count,
                    "input_source": "image",
                    "recovery_attempted": recovery_info.get("attempted", False),
                    "recovery_labels_tried": recovery_info.get("labels_tried", 0),
                    "recovery_barcodes_found": recovery_info.get("barcodes_found", 0),
                    "recovery_labels_resolved": recovery_labels_resolved,
                    # Derived fields for dashboard grouping
                    "scanner_vision_match": scanner_count == vision_count,
                    "count_delta": vision_count - scanner_count,
                    "recovery_succeeded": recovery_labels_resolved > 0,
                }
            )

        # --- Reply by outcome -------------------------------------------
        if outcome == "complete":
            await _send_complete_reply(sender, result)
            if run is not None:
                run.metadata["processing_status"] = "completed"

        elif outcome == "needs_better_photo":
            await _send_needs_better_photo_reply(sender, result)
            if run is not None:
                run.metadata["processing_status"] = "needs_better_photo"

        else:  # retryable_error
            error = result.get("error", {})
            logger.warning(
                "analyze_image returned retryable_error from=%s error=%s",
                sender,
                error,
            )
            if sub_run is not None:
                sub_run.metadata["error"] = error
            await _safe_send_text(
                sender,
                "The barcode service is temporarily unavailable. "
                "Please try sending the photo again.",
            )
            if run is not None:
                run.metadata["processing_status"] = "retryable_error"

    finally:
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


async def process_audio_message(
    msg: dict[str, Any],
    sender: str,
    run: Any,
) -> None:
    """Transcribe an inbound voice/audio note, then ask for a photo.

    Transcription is kept wired so the transcript is logged (and available
    for future voice commands), but the product behavior in v1 is simply to
    ask for a barcode photo.
    """
    if transcriber is None:
        logger.error(
            "No transcriber available — skipping audio message from=%s",
            sender,
        )
        if run is not None:
            run.metadata["processing_status"] = "transcriber_unavailable"
        await _safe_send_text(sender, "Please send a clear photo of the box labels.")
        return

    media_id = msg.get("media_id", "")
    mime_type = msg.get("mime_type", "")

    if not media_id:
        logger.error("Audio message missing media_id from=%s", sender)
        if run is not None:
            run.metadata["processing_status"] = "missing_media_id"
        await _safe_send_text(sender, "Please send a clear photo of the box labels.")
        return

    try:
        user_text = await handle_360dialog_audio_message(
            wa=wa_client,  # type: ignore[arg-type]
            transcriber=transcriber,
            media_id=media_id,
            mime_type=mime_type,
        )
    except Exception:
        logger.exception(
            "audio transcription failed from=%s media_id=%s",
            sender,
            media_id,
        )
        if run is not None:
            run.metadata["processing_status"] = "transcription_failed"
        await _safe_send_text(sender, "Please send a clear photo of the box labels.")
        return

    if not user_text.strip():
        logger.info("Empty transcript from=%s — skipping", sender)
        if run is not None:
            run.metadata["processing_status"] = "empty_transcript"
        await _safe_send_text(sender, "Please send a clear photo of the box labels.")
        return

    if run is not None:
        run.metadata["transcript_length"] = len(user_text)

    await send_photo_instruction(sender)
    if run is not None:
        run.metadata["processing_status"] = "completed"


async def send_photo_instruction(sender: str) -> None:
    """Canned reply for text / transcribed audio: ask for a barcode photo."""
    await _safe_send_text(sender, "Please send a clear photo of the box labels.")


# ---------------------------------------------------------------------------
# Outcome reply helpers
# ---------------------------------------------------------------------------


@ls.traceable(
    name="send_complete_reply",
    run_type="chain",
    tags=["barcode-scanner", "whatsapp", "reply"],
)
async def _send_complete_reply(sender: str, result: dict[str, Any]) -> None:
    """Send the barcode list as a WhatsApp text message.

    Includes ALL decoded barcodes: Gemini-matched ``found`` entries first
    (with their label index), then any ``unassigned`` scanner detections
    that Gemini did not match to a label. This ensures every decoded
    barcode value is reported even when Gemini's label count differs from
    the scanner's detection count.
    """
    found = result.get("found", [])
    unassigned = result.get("unassigned", [])
    total = len(found) + len(unassigned)
    upload_id = result.get("upload_id", "")
    trace_id = result.get("trace_id", "")

    lines = [f"Found {total} barcodes:"]
    for item in found:
        label_index = item.get("label_index", "?")
        value = item.get("barcode_value", "")
        fmt = item.get("barcode_format", "")
        lines.append(f"{label_index}. {value} ({fmt})")
    for item in unassigned:
        value = item.get("barcode_value", "")
        fmt = item.get("barcode_format", "")
        lines.append(f"-. {value} ({fmt})")
    if upload_id:
        lines.append(f"\nID: {upload_id}")
    if trace_id:
        lines.append(f"Trace: {trace_id}")

    body = "\n".join(lines)
    logger.info(
        "Sending complete reply upload_id=%s trace_id=%s to=%s barcodes=%d "
        "(found=%d unassigned=%d)",
        upload_id,
        trace_id,
        sender,
        total,
        len(found),
        len(unassigned),
    )
    reply_run = ls.get_current_run_tree()
    if reply_run is not None:
        reply_run.metadata.update(
            {
                "total_barcodes": total,
                "found_count": len(found),
                "unassigned_count": len(unassigned),
                "reply_length": len(body),
            }
        )
    await _safe_send_text(sender, body)


@ls.traceable(
    name="send_needs_better_photo_reply",
    run_type="chain",
    tags=["barcode-scanner", "whatsapp", "reply"],
)
async def _send_needs_better_photo_reply(
    sender: str,
    result: dict[str, Any],
) -> None:
    """Send the annotated image as a WhatsApp JPEG with a caption.

    analyze_image() returns a base64-encoded PNG (≤1600px longest side).
    We re-encode to JPEG (quality 88, optimize) because phone photographs
    compress far smaller as JPEG and 360dialog's outgoing image limit is
    5 MB. If the JPEG still exceeds the limit, we downscale to 1280px and
    retry once. If that still fails, we fall back to a text message.
    """
    b64 = result.get("annotated_image_b64")
    summary = result.get("summary", {})
    found_count = summary.get("found_count", 0)
    visible_label_count = summary.get("visible_label_count", 0)
    upload_id = result.get("upload_id", "")
    trace_id = result.get("trace_id", "")

    caption = (
        f"I read {found_count} of {visible_label_count} labels. "
        "Please take a closer photo of the marked barcode."
    )
    if upload_id:
        caption = f"{caption}\nID: {upload_id}"
    if trace_id:
        caption = f"{caption}\nTrace: {trace_id}"

    if not b64:
        # Render failure in analyze.py — fall back to text.
        await _safe_send_text(sender, result.get("message", caption))
        return

    try:
        png_bytes = base64.b64decode(b64)
        image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    except Exception:
        logger.exception("Failed to decode annotated image for sender=%s", sender)
        await _safe_send_text(sender, result.get("message", caption))
        return

    jpeg_bytes, used_image = _encode_transport_jpeg(image)

    if len(jpeg_bytes) > _D360_MAX_IMAGE_BYTES:
        # Downscale and retry once.
        downscaled = image.copy()
        downscaled.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
        jpeg_bytes, used_image = _encode_transport_jpeg(downscaled)

    if len(jpeg_bytes) > _D360_MAX_IMAGE_BYTES:
        logger.error(
            "Annotated JPEG still too large (%d bytes) for sender=%s — "
            "falling back to text",
            len(jpeg_bytes),
            sender,
        )
        await _safe_send_text(sender, result.get("message", caption))
        return

    try:
        if wa_client is not None:
            await wa_client.send_image(
                to=sender,
                image_bytes=jpeg_bytes,
                mime_type="image/jpeg",
                caption=caption,
                filename="annotated.jpg",
            )
    except Exception:
        logger.exception(
            "send_annotated failed sender=%s — falling back to text",
            sender,
        )
        await _safe_send_text(sender, result.get("message", caption))


def _encode_transport_jpeg(image: Image.Image) -> tuple[bytes, Image.Image]:
    """Encode an image to JPEG (quality 88, optimize) for WhatsApp transport."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88, optimize=True)
    return buffer.getvalue(), image


# ---------------------------------------------------------------------------
# Safe send helper
# ---------------------------------------------------------------------------


async def _safe_send_text(sender: str, body: str) -> None:
    """Send a text message, swallowing transport errors (best-effort reply)."""
    if wa_client is None:
        return
    try:
        await wa_client.send_text(to=sender, body=body)
    except Exception:
        logger.exception("Failed to send text reply to=%s", sender)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post("/webhook/360dialog")
async def webhook(
    request: Request,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    """Receive 360dialog webhook, schedule background processing, return 200."""
    try:
        payload = await request.json()
    except Exception:
        logger.error("Failed to parse webhook payload as JSON")
        return JSONResponse(status_code=200, content={"status": "ok"})

    messages = list(iter_incoming_messages(payload))

    if not messages:
        logger.info("Webhook received but no messages in payload")
        return JSONResponse(status_code=200, content={"status": "ok"})

    logger.info(
        "Webhook received %d message(s): %s",
        len(messages),
        [m.get("type") for m in messages],
    )

    for msg in messages:
        background_tasks.add_task(process_message, msg)

    return JSONResponse(status_code=200, content={"status": "ok"})


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Static frontend — must be mounted LAST so it doesn't shadow API routes.
# ---------------------------------------------------------------------------

# Serve the built React frontend from web/dist/ when it exists (Docker/prod).
# In local dev, Vite serves the frontend separately on :5173 and this dir is
# absent, so we skip the mount.
_static_dir = Path(__file__).resolve().parent.parent / "web" / "dist"
if _static_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="frontend")

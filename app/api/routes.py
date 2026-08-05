import logging
import time
from io import BytesIO as _BytesIO
from typing import Any

import langsmith as ls
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

from app.core.config import Settings, get_settings
from app.models.barcode import ScanResponse, ScanStatus
from app.services.analyze import analyze_image
from app.services.barcode_scanner import BarcodeScanner

logger = logging.getLogger(__name__)

router = APIRouter()


def get_scanner() -> BarcodeScanner:
    return BarcodeScanner()


def _sanitize_scan_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Keep the filename and content type out of the traced raw bytes."""
    file = inputs.get("file")
    return {
        "filename": getattr(file, "filename", None),
        "content_type": getattr(file, "content_type", None),
    }


@router.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post(
    "/barcode/scan",
    response_model=ScanResponse,
    tags=["barcode"],
    summary="Scan all barcodes visible in one product photo (scanner only)",
)
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

    filename = file.filename or "unknown"

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

    upload_bytes = len(image_bytes)

    try:
        t0 = time.perf_counter()
        barcodes = scanner.scan_bytes(image_bytes)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "invalid_image", "message": str(exc)},
        ) from exc

    # Read just the header for dimensions — PIL.Image.open is lazy and does
    # not decode pixel data, so this is cheap.
    with Image.open(_BytesIO(image_bytes)) as img:
        image_width, image_height = img.size

    logger.info(
        "scan filename=%s upload_bytes=%d dims=%dx%d count=%d elapsed_ms=%d",
        filename,
        upload_bytes,
        image_width,
        image_height,
        len(barcodes),
        elapsed_ms,
    )

    run = ls.get_current_run_tree()
    if run is not None:
        run.metadata.update(
            {
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

    return ScanResponse(
        status=ScanStatus.FOUND if barcodes else ScanStatus.NOT_FOUND,
        count=len(barcodes),
        image_width=image_width,
        image_height=image_height,
        filename=filename,
        upload_bytes=upload_bytes,
        elapsed_ms=elapsed_ms,
        barcodes=barcodes,
    )


@router.post(
    "/barcode/analyze",
    tags=["barcode"],
    summary="Full pipeline: scanner + Gemini audit + annotated image on missing",
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

    filename = file.filename or "unknown"
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

    upload_bytes = len(image_bytes)

    t0 = time.perf_counter()
    result = analyze_image(image_bytes)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    # Merge upload metadata into the result.
    result["filename"] = filename
    result["upload_bytes"] = upload_bytes
    result["elapsed_ms"] = elapsed_ms

    logger.info(
        "analyze filename=%s upload_bytes=%d dims=%dx%d outcome=%s "
        "found=%d missing=%d unassigned=%d elapsed_ms=%d",
        filename,
        upload_bytes,
        result.get("image_width", 0),
        result.get("image_height", 0),
        result.get("outcome"),
        result.get("summary", {}).get("found_count", 0),
        result.get("summary", {}).get("missing_count", 0),
        result.get("summary", {}).get("unassigned_count", 0),
        elapsed_ms,
    )

    summary = result.get("summary", {})
    run = ls.get_current_run_tree()
    if run is not None:
        run.metadata.update(
            {
                "filename": filename,
                "upload_bytes": upload_bytes,
                "image_width": result.get("image_width", 0),
                "image_height": result.get("image_height", 0),
                "outcome": result.get("outcome"),
                "audit_available": result.get("audit_available"),
                "visible_label_count": summary.get("visible_label_count", 0),
                "found_count": summary.get("found_count", 0),
                "missing_count": summary.get("missing_count", 0),
                "unassigned_count": summary.get("unassigned_count", 0),
                "all_found": summary.get("all_found", False),
                "elapsed_ms": elapsed_ms,
                "found_values": [
                    f.get("barcode_value") for f in result.get("found", [])
                ],
                "missing_labels": [
                    m.get("label_index") for m in result.get("missing", [])
                ],
                "has_annotated_image": bool(result.get("annotated_image_b64")),
            }
        )

    return result

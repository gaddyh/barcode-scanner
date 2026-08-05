import logging
import time
from io import BytesIO as _BytesIO

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

from app.core.config import Settings, get_settings
from app.models.barcode import ScanResponse, ScanStatus
from app.services.barcode_scanner import BarcodeScanner

logger = logging.getLogger(__name__)

router = APIRouter()


def get_scanner() -> BarcodeScanner:
    return BarcodeScanner()


@router.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post(
    "/barcode/scan",
    response_model=ScanResponse,
    tags=["barcode"],
    summary="Scan all barcodes visible in one product photo",
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

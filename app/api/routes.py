from io import BytesIO as _BytesIO

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

from app.core.config import Settings, get_settings
from app.models.barcode import ScanResponse, ScanStatus
from app.services.barcode_scanner import BarcodeScanner

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

    try:
        barcodes = scanner.scan_bytes(image_bytes)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "invalid_image", "message": str(exc)},
        ) from exc

    # Read just the header for dimensions — PIL.Image.open is lazy and does
    # not decode pixel data, so this is cheap.
    with Image.open(_BytesIO(image_bytes)) as img:
        image_width, image_height = img.size

    return ScanResponse(
        status=ScanStatus.FOUND if barcodes else ScanStatus.NOT_FOUND,
        count=len(barcodes),
        image_width=image_width,
        image_height=image_height,
        barcodes=barcodes,
    )

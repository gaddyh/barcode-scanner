from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from app.services.barcode_scanner import DetectedBarcode


class ScanStatus(StrEnum):
    FOUND = "found"
    NOT_FOUND = "not_found"


class ScanResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    status: ScanStatus
    count: int
    image_width: int
    image_height: int
    filename: str
    upload_bytes: int
    elapsed_ms: int
    barcodes: list[DetectedBarcode]

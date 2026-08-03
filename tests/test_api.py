from io import BytesIO
from unittest.mock import Mock

from fastapi.testclient import TestClient
from PIL import Image

from app.services.barcode_scanner import BoundingBox, DetectedBarcode, Point


def _make_png(width: int = 100, height: int = 50) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_scan_endpoint(client: TestClient, override_scanner: dict[str, object]) -> None:
    scanner = Mock()
    scanner.scan_bytes.return_value = [
        DetectedBarcode(
            value="7290001234567",
            format="EAN13",
            content_type="Text",
            orientation=0,
            position=(
                Point(x=10, y=5),
                Point(x=90, y=5),
                Point(x=90, y=40),
                Point(x=10, y=40),
            ),
            bounding_box=BoundingBox(x1=10, y1=5, x2=90, y2=40),
        )
    ]
    override_scanner["scanner"] = scanner

    response = client.post(
        "/barcode/scan",
        files={"file": ("product.png", _make_png(100, 50), "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "found"
    assert body["count"] == 1
    assert body["image_width"] == 100
    assert body["image_height"] == 50
    assert body["barcodes"][0]["value"] == "7290001234567"
    scanner.scan_bytes.assert_called_once()


def test_scan_endpoint_not_found(client: TestClient, override_scanner: dict[str, object]) -> None:
    scanner = Mock()
    scanner.scan_bytes.return_value = []
    override_scanner["scanner"] = scanner

    response = client.post(
        "/barcode/scan",
        files={"file": ("product.png", _make_png(), "image/png")},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "not_found"


def test_rejects_unsupported_content_type(
    client: TestClient, override_scanner: dict[str, object]
) -> None:
    override_scanner["scanner"] = Mock()

    response = client.post(
        "/barcode/scan",
        files={"file": ("product.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 415
    assert response.json()["detail"]["code"] == "unsupported_image_type"

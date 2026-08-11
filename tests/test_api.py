"""Tests for the FastAPI /barcode/scan endpoint (deterministic scanner only)."""

from __future__ import annotations

import io

import pytest
import zxingcpp
from PIL import Image

from src.ingest.scanner import BarcodeScanner
from tests._zxing_fake import make_read_result


def _first_call_mock(results):
    """Return a mock that yields ``results`` on the first call, then ``[]``."""
    calls = {"n": 0}

    def _mock(_img, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return results
        return []

    return _mock


def _png_bytes(width: int = 100, height: int = 100) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


def test_scan_endpoint_found(
    client: pytest.fixture,
    override_scanner: pytest.fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        zxingcpp, "read_barcodes",
        _first_call_mock([make_read_result("1234567890123")]),
    )
    override_scanner["scanner"] = BarcodeScanner()

    response = client.post(
        "/barcode/scan",
        files={"file": ("img.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "found"
    assert body["count"] == 1
    assert body["barcodes"][0]["value"] == "1234567890123"
    assert body["image_width"] == 100
    assert body["image_height"] == 100


def test_scan_endpoint_not_found(
    client: pytest.fixture,
    override_scanner: pytest.fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(zxingcpp, "read_barcodes", lambda _img, **kwargs: [])
    override_scanner["scanner"] = BarcodeScanner()

    response = client.post(
        "/barcode/scan",
        files={"file": ("img.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "not_found"


def test_scan_endpoint_unsupported_type(
    client: pytest.fixture,
) -> None:
    response = client.post(
        "/barcode/scan",
        files={"file": ("img.gif", b"GIF89a", "image/gif")},
    )
    assert response.status_code == 415


def test_health_endpoint(client: pytest.fixture) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_customers_endpoint(monkeypatch: pytest.MonkeyPatch, client: pytest.fixture) -> None:
    async def customers(_self):
        return [{"id": "C1", "name": "Acme"}]

    class FakePriorityRepository:
        async def customers(self):
            return await customers(self)

    monkeypatch.setattr("src.api.routes._get_priority_repo", lambda: FakePriorityRepository())
    response = client.get("/customers")
    assert response.status_code == 200
    assert response.json() == {"items": [{"id": "C1", "name": "Acme"}]}


def test_branches_endpoint_is_customer_scoped(
    monkeypatch: pytest.MonkeyPatch, client: pytest.fixture
) -> None:
    async def branches(_self, customer_id):
        return [{"id": "B1", "name": f"Branch for {customer_id}"}]

    class FakePriorityRepository:
        async def branches(self, customer_id):
            return await branches(self, customer_id)

    monkeypatch.setattr("src.api.routes._get_priority_repo", lambda: FakePriorityRepository())
    response = client.get("/customers/C1/branches")
    assert response.status_code == 200
    assert response.json() == {"items": [{"id": "B1", "name": "Branch for C1"}]}


def test_session_requires_order_context(client: pytest.fixture) -> None:
    response = client.post(
        "/barcode/session",
        files={"file": ("img.png", _png_bytes(), "image/png")},
        data={"participant_id": "p1"},
    )
    assert response.status_code == 422

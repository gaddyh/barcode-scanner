"""Shared pytest fixtures for the barcode-scanner test suite.

Tests mock ``zxingcpp.read_barcodes`` and ``genai.Client`` so no real barcode
images or Gemini API calls are required.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from src.api.routes import get_scanner
from src.main import app


@pytest.fixture
def override_scanner() -> Iterator[dict[str, object]]:
    holder: dict[str, object] = {}

    def dependency() -> object:
        return holder["scanner"]

    app.dependency_overrides[get_scanner] = dependency
    yield holder
    app.dependency_overrides.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)

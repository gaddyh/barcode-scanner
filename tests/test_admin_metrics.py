"""Tests for the /admin/metrics endpoint — LangSmith query + graceful degradation.

Uses FastAPI TestClient with a fake LangSmith Client.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from fastapi.testclient import TestClient


@dataclass
class StubRun:
    metadata: dict[str, Any]


def _run(
    final_status: str = "complete",
    found_count: int = 6,
    recovery_attempted: bool = False,
    recovery_labels_resolved: int = 0,
    latency_ms: int = 3000,
) -> StubRun:
    return StubRun(metadata={
        "final_status": final_status,
        "source": "cli",
        "found_count": found_count,
        "recovery_attempted": recovery_attempted,
        "recovery_labels_resolved": recovery_labels_resolved,
        "latency_ms": latency_ms,
    })


class FakeClient:
    """Fake LangSmith Client that returns stub runs."""

    def __init__(self, runs: list[StubRun] | None = None, exc: Exception | None = None):
        self._runs = runs or []
        self._exc = exc

    def list_runs(self, **kwargs):
        if self._exc:
            raise self._exc
        return iter(self._runs)


@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)


def _patch_client(monkeypatch: pytest.MonkeyPatch, fake: FakeClient) -> None:
    """Patch the Client constructor in admin.py to return our fake."""
    monkeypatch.setattr("langsmith.Client", lambda: fake)
    monkeypatch.setenv("LANGSMITH_PROJECT", "test-project")


def test_metrics_successful_query(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Successful LangSmith query → source=langsmith with real metrics."""
    runs = [_run(), _run(final_status="needs_user_input")]
    _patch_client(monkeypatch, FakeClient(runs=runs))

    resp = client.get("/admin/metrics?hours=24")
    assert resp.status_code == 200
    data = resp.json()
    assert data["source"] == "langsmith"
    assert data["images_processed"] == 2
    assert data["boxes_processed"] == 12
    assert data["final_complete_pct"] == 50.0
    assert data["user_retry_required_pct"] == 50.0
    assert data["truncated"] is False


def test_metrics_empty_query(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty LangSmith query → source=langsmith with zeros."""
    _patch_client(monkeypatch, FakeClient(runs=[]))

    resp = client.get("/admin/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert data["source"] == "langsmith"
    assert data["images_processed"] == 0
    assert data["final_complete_pct"] == 0.0


def test_metrics_langsmith_unavailable(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """LangSmith exception → source=unavailable, zeros, no HTTP error."""
    _patch_client(monkeypatch, FakeClient(exc=Exception("LangSmith API down")))

    resp = client.get("/admin/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert data["source"] == "unavailable"
    assert data["images_processed"] == 0
    assert data["final_complete_pct"] == 0.0


def test_metrics_truncation(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """100 runs → truncated=true (LangSmith max page size)."""
    runs = [_run() for _ in range(100)]
    _patch_client(monkeypatch, FakeClient(runs=runs))

    resp = client.get("/admin/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert data["truncated"] is True
    assert data["images_processed"] == 100

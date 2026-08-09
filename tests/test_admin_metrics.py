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
    source: str = "cli",
    pipeline_version: str = "ingest-v1",
    scanner_version: str = "scanner-0.8",
    recovery_version: str = "recovery-v1",
    vision_model: str = "gemini-3.5-flash-lite",
    vision_prompt_version: str = "label-audit-v1",
    scanner_vision_match: bool = True,
) -> StubRun:
    return StubRun(metadata={
        "final_status": final_status,
        "source": source,
        "found_count": found_count,
        "recovery_attempted": recovery_attempted,
        "recovery_labels_resolved": recovery_labels_resolved,
        "latency_ms": latency_ms,
        "pipeline_version": pipeline_version,
        "scanner_version": scanner_version,
        "recovery_version": recovery_version,
        "vision_model": vision_model,
        "vision_prompt_version": vision_prompt_version,
        "scanner_vision_match": scanner_vision_match,
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
    from src.main import app
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


# ---------------------------------------------------------------------------
# group_by — version comparisons as true rates
# ---------------------------------------------------------------------------


def test_group_by_pipeline_version(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """group_by=pipeline_version → one MetricsResponse per version."""
    runs = [
        _run(final_status="complete", pipeline_version="ingest-v1"),
        _run(final_status="complete", pipeline_version="ingest-v1"),
        _run(final_status="needs_user_input", pipeline_version="ingest-v1"),
        _run(final_status="complete", pipeline_version="ingest-v2"),
        _run(final_status="needs_user_input", pipeline_version="ingest-v2"),
    ]
    _patch_client(monkeypatch, FakeClient(runs=runs))

    resp = client.get("/admin/metrics?group_by=pipeline_version")
    assert resp.status_code == 200
    data = resp.json()
    assert data["group_by"] == "pipeline_version"
    assert data["source"] == "langsmith"
    groups = data["groups"]
    assert set(groups.keys()) == {"ingest-v1", "ingest-v2"}
    assert groups["ingest-v1"]["images_processed"] == 3
    assert groups["ingest-v1"]["final_complete_pct"] == 66.7
    assert groups["ingest-v2"]["images_processed"] == 2
    assert groups["ingest-v2"]["final_complete_pct"] == 50.0


def test_group_by_scanner_version(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """group_by=scanner_version → mismatch rate per scanner version."""
    runs = [
        _run(scanner_vision_match=True, scanner_version="scanner-0.8"),
        _run(scanner_vision_match=False, scanner_version="scanner-0.8"),
        _run(scanner_vision_match=True, scanner_version="scanner-0.9"),
        _run(scanner_vision_match=True, scanner_version="scanner-0.9"),
    ]
    _patch_client(monkeypatch, FakeClient(runs=runs))

    resp = client.get("/admin/metrics?group_by=scanner_version")
    assert resp.status_code == 200
    groups = resp.json()["groups"]
    assert groups["scanner-0.8"]["scanner_vision_match_pct"] == 50.0
    assert groups["scanner-0.9"]["scanner_vision_match_pct"] == 100.0


def test_group_by_source(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """group_by=source → metrics per source (cli, web, whatsapp)."""
    runs = [
        _run(final_status="complete", source="cli"),
        _run(final_status="needs_user_input", source="web"),
    ]
    _patch_client(monkeypatch, FakeClient(runs=runs))

    resp = client.get("/admin/metrics?group_by=source")
    assert resp.status_code == 200
    groups = resp.json()["groups"]
    assert set(groups.keys()) == {"cli", "web"}
    assert groups["cli"]["final_complete_pct"] == 100.0
    assert groups["web"]["user_retry_required_pct"] == 100.0


def test_group_by_invalid_returns_400(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid group_by value → 400 with valid options."""
    _patch_client(monkeypatch, FakeClient(runs=[]))

    resp = client.get("/admin/metrics?group_by=nonexistent")
    assert resp.status_code == 400
    assert "nonexistent" in resp.json()["detail"]


def test_group_by_empty_runs(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty runs with group_by → empty groups dict."""
    _patch_client(monkeypatch, FakeClient(runs=[]))

    resp = client.get("/admin/metrics?group_by=pipeline_version")
    assert resp.status_code == 200
    data = resp.json()
    assert data["groups"] == {}
    assert data["source"] == "langsmith"


def test_group_by_langsmith_unavailable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LangSmith down + group_by → source=unavailable, empty groups."""
    _patch_client(monkeypatch, FakeClient(exc=Exception("API down")))

    resp = client.get("/admin/metrics?group_by=pipeline_version")
    assert resp.status_code == 200
    data = resp.json()
    assert data["source"] == "unavailable"
    assert data["groups"] == {}


def test_group_by_missing_version_defaults_to_unknown(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runs without the version key → grouped under 'unknown'."""
    runs = [
        StubRun(metadata={"final_status": "complete", "source": "cli"}),
        _run(final_status="complete", pipeline_version="ingest-v1"),
    ]
    _patch_client(monkeypatch, FakeClient(runs=runs))

    resp = client.get("/admin/metrics?group_by=pipeline_version")
    assert resp.status_code == 200
    groups = resp.json()["groups"]
    assert set(groups.keys()) == {"ingest-v1", "unknown"}
    assert groups["unknown"]["images_processed"] == 1


# ---------------------------------------------------------------------------
# DB-first metrics — Postgres source
# ---------------------------------------------------------------------------


class FakeDbRepo:
    """Fake PostgresRunRepository that returns DB rows with metadata dicts."""

    def __init__(self, rows: list[dict[str, Any]] | None = None, exc: Exception | None = None):
        self._rows = rows or []
        self._exc = exc

    async def query_runs(self, *, hours: int, limit: int = 500) -> list[dict[str, Any]]:
        if self._exc:
            raise self._exc
        return self._rows


def _db_row(
    final_status: str = "complete",
    found_count: int = 6,
    recovery_attempted: bool = False,
    recovery_labels_resolved: int = 0,
    latency_ms: int = 3000,
    source: str = "web",
    pipeline_version: str = "ingest-v1",
    scanner_version: str = "scanner-0.8",
    scanner_vision_match: bool = True,
    count_delta: int = 0,
    missing_count: int = 0,
    unassigned_count: int = 0,
) -> dict[str, Any]:
    """Build a DB row dict with metadata sub-dict matching compute_metrics keys."""
    return {
        "id": "01JTEST",
        "source": source,
        "status": final_status,
        "metadata": {
            "final_status": final_status,
            "source": source,
            "found_count": found_count,
            "recovery_attempted": recovery_attempted,
            "recovery_labels_resolved": recovery_labels_resolved,
            "latency_ms": latency_ms,
            "pipeline_version": pipeline_version,
            "scanner_version": scanner_version,
            "scanner_vision_match": scanner_vision_match,
            "count_delta": count_delta,
            "missing_count": missing_count,
            "unassigned_count": unassigned_count,
        },
    }


def _patch_db_repo(monkeypatch: pytest.MonkeyPatch, repo: FakeDbRepo) -> None:
    """Patch src.main.run_repo to use our fake DB repo."""
    import src.main
    monkeypatch.setattr(src.main, "run_repo", repo)


def test_db_metrics_successful(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """DB available → source=postgres with real metrics."""
    rows = [_db_row(), _db_row(final_status="needs_user_input")]
    _patch_db_repo(monkeypatch, FakeDbRepo(rows=rows))

    resp = client.get("/admin/metrics?hours=24")
    assert resp.status_code == 200
    data = resp.json()
    assert data["source"] == "postgres"
    assert data["images_processed"] == 2
    assert data["boxes_processed"] == 12
    assert data["final_complete_pct"] == 50.0


def test_db_metrics_empty(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """DB accessible but empty → source=postgres with zeros (no LangSmith fallback)."""
    _patch_db_repo(monkeypatch, FakeDbRepo(rows=[]))

    resp = client.get("/admin/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert data["source"] == "postgres"
    assert data["images_processed"] == 0


def test_db_metrics_fallback_to_langsmith_on_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DB query fails → fall back to LangSmith."""
    _patch_db_repo(monkeypatch, FakeDbRepo(exc=Exception("DB connection lost")))
    runs = [_run(), _run(final_status="needs_user_input")]
    _patch_client(monkeypatch, FakeClient(runs=runs))

    resp = client.get("/admin/metrics?hours=24")
    assert resp.status_code == 200
    data = resp.json()
    assert data["source"] == "langsmith"
    assert data["images_processed"] == 2


def test_db_metrics_grouped(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """DB + group_by → source=postgres with per-group metrics."""
    rows = [
        _db_row(final_status="complete", pipeline_version="ingest-v1"),
        _db_row(final_status="complete", pipeline_version="ingest-v1"),
        _db_row(final_status="needs_user_input", pipeline_version="ingest-v1"),
        _db_row(final_status="complete", pipeline_version="ingest-v2"),
    ]
    _patch_db_repo(monkeypatch, FakeDbRepo(rows=rows))

    resp = client.get("/admin/metrics?group_by=pipeline_version")
    assert resp.status_code == 200
    data = resp.json()
    assert data["source"] == "postgres"
    assert data["group_by"] == "pipeline_version"
    groups = data["groups"]
    assert set(groups.keys()) == {"ingest-v1", "ingest-v2"}
    assert groups["ingest-v1"]["images_processed"] == 3
    assert groups["ingest-v1"]["final_complete_pct"] == 66.7
    assert groups["ingest-v2"]["images_processed"] == 1

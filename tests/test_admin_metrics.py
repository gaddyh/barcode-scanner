"""Tests for the /admin/metrics endpoint — DB-only metrics.

The admin endpoint reads from Postgres only. LangSmith is not queried.
These tests use a FakeDbRepo to stub the repository.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from src.main import app
    return TestClient(app)


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
    recovery_version: str = "recovery-v1",
    vision_model: str = "gemini-3.5-flash-lite",
    vision_prompt_version: str = "label-audit-v1",
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
            "recovery_version": recovery_version,
            "vision_model": vision_model,
            "vision_prompt_version": vision_prompt_version,
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


# ---------------------------------------------------------------------------
# Flat metrics
# ---------------------------------------------------------------------------


def test_metrics_successful(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """DB query → source=postgres with real metrics."""
    rows = [_db_row(), _db_row(final_status="needs_user_input")]
    _patch_db_repo(monkeypatch, FakeDbRepo(rows=rows))

    resp = client.get("/admin/metrics?hours=24")
    assert resp.status_code == 200
    data = resp.json()
    assert data["source"] == "postgres"
    assert data["images_processed"] == 2
    assert data["boxes_processed"] == 12
    assert data["final_complete_pct"] == 50.0
    assert data["user_retry_required_pct"] == 50.0
    assert data["truncated"] is False


def test_metrics_empty(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """DB accessible but empty → source=postgres with zeros."""
    _patch_db_repo(monkeypatch, FakeDbRepo(rows=[]))

    resp = client.get("/admin/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert data["source"] == "postgres"
    assert data["images_processed"] == 0
    assert data["final_complete_pct"] == 0.0


def test_metrics_db_unavailable(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """DB exception → source=unavailable, zeros, no HTTP error."""
    _patch_db_repo(monkeypatch, FakeDbRepo(exc=Exception("DB connection lost")))

    resp = client.get("/admin/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert data["source"] == "unavailable"
    assert data["images_processed"] == 0
    assert data["final_complete_pct"] == 0.0


def test_metrics_truncation(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """2000 runs → truncated=true (DB limit)."""
    rows = [_db_row() for _ in range(2000)]
    _patch_db_repo(monkeypatch, FakeDbRepo(rows=rows))

    resp = client.get("/admin/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert data["truncated"] is True
    assert data["images_processed"] == 2000


# ---------------------------------------------------------------------------
# group_by — version comparisons as true rates
# ---------------------------------------------------------------------------


def test_group_by_pipeline_version(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """group_by=pipeline_version → one MetricsResponse per version."""
    rows = [
        _db_row(final_status="complete", pipeline_version="ingest-v1"),
        _db_row(final_status="complete", pipeline_version="ingest-v1"),
        _db_row(final_status="needs_user_input", pipeline_version="ingest-v1"),
        _db_row(final_status="complete", pipeline_version="ingest-v2"),
        _db_row(final_status="needs_user_input", pipeline_version="ingest-v2"),
    ]
    _patch_db_repo(monkeypatch, FakeDbRepo(rows=rows))

    resp = client.get("/admin/metrics?group_by=pipeline_version")
    assert resp.status_code == 200
    data = resp.json()
    assert data["group_by"] == "pipeline_version"
    assert data["source"] == "postgres"
    groups = data["groups"]
    assert set(groups.keys()) == {"ingest-v1", "ingest-v2"}
    assert groups["ingest-v1"]["images_processed"] == 3
    assert groups["ingest-v1"]["final_complete_pct"] == 66.7
    assert groups["ingest-v2"]["images_processed"] == 2
    assert groups["ingest-v2"]["final_complete_pct"] == 50.0


def test_group_by_scanner_version(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """group_by=scanner_version → mismatch rate per scanner version."""
    rows = [
        _db_row(scanner_vision_match=True, scanner_version="scanner-0.8"),
        _db_row(scanner_vision_match=False, scanner_version="scanner-0.8"),
        _db_row(scanner_vision_match=True, scanner_version="scanner-0.9"),
        _db_row(scanner_vision_match=True, scanner_version="scanner-0.9"),
    ]
    _patch_db_repo(monkeypatch, FakeDbRepo(rows=rows))

    resp = client.get("/admin/metrics?group_by=scanner_version")
    assert resp.status_code == 200
    groups = resp.json()["groups"]
    assert groups["scanner-0.8"]["scanner_vision_match_pct"] == 50.0
    assert groups["scanner-0.9"]["scanner_vision_match_pct"] == 100.0


def test_group_by_source(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """group_by=source → metrics per source."""
    rows = [
        _db_row(final_status="complete", source="cli"),
        _db_row(final_status="needs_user_input", source="web"),
    ]
    _patch_db_repo(monkeypatch, FakeDbRepo(rows=rows))

    resp = client.get("/admin/metrics?group_by=source")
    assert resp.status_code == 200
    groups = resp.json()["groups"]
    assert set(groups.keys()) == {"cli", "web"}
    assert groups["cli"]["final_complete_pct"] == 100.0
    assert groups["web"]["user_retry_required_pct"] == 100.0


def test_group_by_invalid_returns_400(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid group_by value → 400 with valid options."""
    _patch_db_repo(monkeypatch, FakeDbRepo(rows=[]))

    resp = client.get("/admin/metrics?group_by=nonexistent")
    assert resp.status_code == 400
    assert "nonexistent" in resp.json()["detail"]


def test_group_by_empty_runs(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty runs with group_by → empty groups dict."""
    _patch_db_repo(monkeypatch, FakeDbRepo(rows=[]))

    resp = client.get("/admin/metrics?group_by=pipeline_version")
    assert resp.status_code == 200
    data = resp.json()
    assert data["groups"] == {}
    assert data["source"] == "postgres"


def test_group_by_db_unavailable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DB down + group_by → source=unavailable, empty groups."""
    _patch_db_repo(monkeypatch, FakeDbRepo(exc=Exception("DB down")))

    resp = client.get("/admin/metrics?group_by=pipeline_version")
    assert resp.status_code == 200
    data = resp.json()
    assert data["source"] == "unavailable"
    assert data["groups"] == {}


def test_group_by_missing_version_defaults_to_unknown(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runs without the version key → grouped under 'unknown'."""
    rows = [
        _db_row(final_status="complete", pipeline_version="ingest-v1"),
        _db_row(final_status="complete", pipeline_version="ingest-v1"),
    ]
    # Remove the key entirely from the first row to simulate missing version
    del rows[0]["metadata"]["pipeline_version"]
    _patch_db_repo(monkeypatch, FakeDbRepo(rows=rows))

    resp = client.get("/admin/metrics?group_by=pipeline_version")
    assert resp.status_code == 200
    groups = resp.json()["groups"]
    assert set(groups.keys()) == {"ingest-v1", "unknown"}
    assert groups["unknown"]["images_processed"] == 1

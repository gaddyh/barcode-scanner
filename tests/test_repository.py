"""Tests for src/repository.py — run & annotation repositories."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import asyncpg
import pytest

from src.ingest.models import (
    DetectedItem,
    IngestResult,
    IngestStatus,
    Issue,
    RunMetrics,
)
from src.repository import (
    GROUPABLE_FIELDS,
    NewRun,
    NoOpAnnotationRepository,
    NoOpRunRepository,
    PostgresAnnotationRepository,
    PostgresRunRepository,
    get_annotation_repository,
    get_run_repository,
)

# ---------------------------------------------------------------------------
# Fake asyncpg pool / connection helpers
# ---------------------------------------------------------------------------


class FakeAcquire:
    """Async context manager that yields ``conn``."""

    def __init__(self, conn: MagicMock) -> None:
        self._conn = conn

    async def __aenter__(self) -> MagicMock:
        return self._conn

    async def __aexit__(self, *exc: object) -> bool:
        return False


class FakeTransaction:
    """Async context manager that simulates conn.transaction()."""

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *exc: object) -> bool:
        return False


def _make_pool(
    *,
    execute_return: str = "INSERT 0 1",
    fetch_rows: list[dict] | None = None,
    fetchrow: dict | None = None,
) -> tuple[MagicMock, MagicMock]:
    """Build a fake asyncpg pool and its underlying connection."""
    conn = MagicMock()
    conn.execute = AsyncMock(return_value=execute_return)
    conn.fetch = AsyncMock(return_value=fetch_rows or [])
    conn.fetchrow = AsyncMock(return_value=fetchrow)
    conn.transaction = MagicMock(return_value=FakeTransaction())

    pool = MagicMock()
    pool.acquire = MagicMock(return_value=FakeAcquire(conn))
    return pool, conn


def _make_pool_raising(exc: Exception) -> tuple[MagicMock, MagicMock]:
    """Build a fake pool whose conn methods raise ``exc``."""
    conn = MagicMock()
    conn.execute = AsyncMock(side_effect=exc)
    conn.fetch = AsyncMock(side_effect=exc)
    conn.fetchrow = AsyncMock(side_effect=exc)
    conn.transaction = MagicMock(return_value=FakeTransaction())

    pool = MagicMock()
    pool.acquire = MagicMock(return_value=FakeAcquire(conn))
    return pool, conn


def _make_ingest_result(
    *,
    status: IngestStatus = IngestStatus.COMPLETE,
    items: list[DetectedItem] | None = None,
    missing: list[dict] | None = None,
    issues: list[Issue] | None = None,
    metrics: RunMetrics | None = None,
    error: dict | None = None,
    image_width: int = 800,
    image_height: int = 600,
) -> IngestResult:
    return IngestResult(
        status=status,
        items=items or [],
        missing=missing or [],
        unassigned=[],
        issues=issues or [],
        metrics=metrics or RunMetrics(
            elapsed_ms=42,
            scanner_count=2,
            vision_count=2,
        ),
        image_width=image_width,
        image_height=image_height,
        error=error,
    )


# ---------------------------------------------------------------------------
# NewRun DTO
# ---------------------------------------------------------------------------


def test_new_run_defaults():
    run = NewRun(id="r1", source="web")
    assert run.id == "r1"
    assert run.source == "web"
    assert run.session_id is None
    assert run.trace_id is None
    assert run.endpoint is None
    assert run.filename is None
    assert run.image_ref is None
    assert run.upload_bytes is None
    assert run.image_width is None
    assert run.image_height is None
    assert run.provider_message_id is None
    assert run.sender is None


def test_new_run_all_fields():
    run = NewRun(
        id="r1",
        source="web",
        session_id="s1",
        trace_id="t1",
        endpoint="/scan",
        filename="f.jpg",
        image_ref="/tmp/f.jpg",
        upload_bytes=100,
        image_width=800,
        image_height=600,
        provider_message_id="m1",
        sender="u1",
    )
    assert run.session_id == "s1"
    assert run.trace_id == "t1"
    assert run.endpoint == "/scan"
    assert run.filename == "f.jpg"
    assert run.image_ref == "/tmp/f.jpg"
    assert run.upload_bytes == 100
    assert run.image_width == 800
    assert run.image_height == 600
    assert run.provider_message_id == "m1"
    assert run.sender == "u1"


# ---------------------------------------------------------------------------
# NoOpRunRepository
# ---------------------------------------------------------------------------


async def test_noop_run_create_run():
    repo = NoOpRunRepository()
    await repo.create_run(NewRun(id="r1", source="web"))


async def test_noop_run_mark_processing():
    repo = NoOpRunRepository()
    await repo.mark_processing("r1")


async def test_noop_run_complete_run():
    repo = NoOpRunRepository()
    result = _make_ingest_result()
    await repo.complete_run("r1", result)


async def test_noop_run_complete_run_with_versions():
    repo = NoOpRunRepository()
    result = _make_ingest_result()
    await repo.complete_run(
        "r1", result, versions={"pipeline_version": "v1"},
    )


async def test_noop_run_fail_run():
    repo = NoOpRunRepository()
    await repo.fail_run("r1", ValueError("boom"))


async def test_noop_run_query_metrics():
    repo = NoOpRunRepository()
    result = await repo.query_metrics(hours=24)
    assert result == {"source": "noop", "runs": []}


async def test_noop_run_query_metrics_with_group_by():
    repo = NoOpRunRepository()
    result = await repo.query_metrics(hours=24, group_by="source")
    assert result == {"source": "noop", "runs": []}


async def test_noop_run_query_runs():
    repo = NoOpRunRepository()
    result = await repo.query_runs(hours=24)
    assert result == []


async def test_noop_run_query_runs_with_limit():
    repo = NoOpRunRepository()
    result = await repo.query_runs(hours=24, limit=10)
    assert result == []


# ---------------------------------------------------------------------------
# NoOpAnnotationRepository
# ---------------------------------------------------------------------------


async def test_noop_annotation_create():
    repo = NoOpAnnotationRepository()
    await repo.create_annotation("r1")


async def test_noop_annotation_list_pending():
    repo = NoOpAnnotationRepository()
    result = await repo.list_pending()
    assert result == []


async def test_noop_annotation_list_pending_with_limit():
    repo = NoOpAnnotationRepository()
    result = await repo.list_pending(limit=10)
    assert result == []


async def test_noop_annotation_review():
    repo = NoOpAnnotationRepository()
    await repo.review("r1")


async def test_noop_annotation_review_all_params():
    repo = NoOpAnnotationRepository()
    await repo.review(
        "r1",
        expected_barcodes=["BC1"],
        expected_outcome="complete",
        reviewed_by="tester",
    )


# ---------------------------------------------------------------------------
# PostgresRunRepository — create_run
# ---------------------------------------------------------------------------


async def test_pg_create_run_calls_execute():
    pool, conn = _make_pool()
    repo = PostgresRunRepository(pool)
    run = NewRun(
        id="r1", source="web", session_id="s1", trace_id="t1",
        endpoint="/scan", filename="f.jpg", image_ref="/tmp/f.jpg",
        upload_bytes=100, image_width=800, image_height=600,
        provider_message_id="m1", sender="u1",
    )
    await repo.create_run(run)
    conn.execute.assert_called_once()
    args = conn.execute.call_args.args
    assert args[1] == "r1"
    assert args[2] == "s1"
    assert args[3] == "t1"
    assert args[4] == "web"
    assert args[5] == "/scan"
    assert args[6] == "f.jpg"
    assert args[7] == "/tmp/f.jpg"
    assert args[8] == 100
    assert args[9] == 800
    assert args[10] == 600
    assert args[11] == "m1"
    assert args[12] == "u1"


async def test_pg_create_run_minimal_fields():
    pool, conn = _make_pool()
    repo = PostgresRunRepository(pool)
    run = NewRun(id="r1", source="web")
    await repo.create_run(run)
    conn.execute.assert_called_once()
    args = conn.execute.call_args.args
    assert args[1] == "r1"
    assert args[2] is None  # session_id
    assert args[4] == "web"


async def test_pg_create_run_error_propagates():
    pool, _ = _make_pool_raising(asyncpg.PostgresError("boom"))
    repo = PostgresRunRepository(pool)
    with pytest.raises(asyncpg.PostgresError, match="boom"):
        await repo.create_run(NewRun(id="r1", source="web"))


# ---------------------------------------------------------------------------
# PostgresRunRepository — mark_processing
# ---------------------------------------------------------------------------


async def test_pg_mark_processing_calls_execute():
    pool, conn = _make_pool()
    repo = PostgresRunRepository(pool)
    await repo.mark_processing("r1")
    conn.execute.assert_called_once()
    args = conn.execute.call_args.args
    assert "processing" in args[0]
    assert args[1] == "r1"


async def test_pg_mark_processing_error_propagates():
    pool, _ = _make_pool_raising(OSError("conn refused"))
    repo = PostgresRunRepository(pool)
    with pytest.raises(OSError, match="conn refused"):
        await repo.mark_processing("r1")


# ---------------------------------------------------------------------------
# PostgresRunRepository — complete_run
# ---------------------------------------------------------------------------


async def test_pg_complete_run_updates_and_inserts_items():
    pool, conn = _make_pool()
    repo = PostgresRunRepository(pool)
    item = DetectedItem(
        label_index=0, barcode_value="BC1", barcode_format="EAN_13",
        barcode_bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10},
        label_bbox={"x1": 0, "y1": 0, "x2": 20, "y2": 20},
        match_basis="containment",
    )
    result = _make_ingest_result(items=[item])
    await repo.complete_run("r1", result)
    # 1 UPDATE + 1 INSERT per item
    assert conn.execute.call_count == 2
    # First call is the UPDATE
    first_sql = conn.execute.call_args_list[0].args[0]
    assert "UPDATE runs SET" in first_sql


async def test_pg_complete_run_with_missing_items():
    pool, conn = _make_pool()
    repo = PostgresRunRepository(pool)
    missing = [
        {"label_index": 5, "barcode_bbox": {"x1": 0, "y1": 0}, "label_bbox": {}},
    ]
    result = _make_ingest_result(missing=missing)
    await repo.complete_run("r1", result)
    # 1 UPDATE + 1 INSERT per missing
    assert conn.execute.call_count == 2


async def test_pg_complete_run_with_versions():
    pool, conn = _make_pool()
    repo = PostgresRunRepository(pool)
    result = _make_ingest_result()
    await repo.complete_run(
        "r1", result,
        versions={
            "pipeline_version": "v1", "scanner_version": "v2",
            "vision_prompt_version": "v3", "vision_model": "gemini",
            "recovery_version": "v4",
        },
    )
    conn.execute.assert_called_once()
    args = conn.execute.call_args.args
    # versions are the last 5 args
    assert args[-5] == "v1"
    assert args[-4] == "v2"
    assert args[-3] == "v3"
    assert args[-2] == "gemini"
    assert args[-1] == "v4"


async def test_pg_complete_run_with_issues():
    pool, conn = _make_pool()
    repo = PostgresRunRepository(pool)
    issue = Issue(code="BARCODE_MISSING", severity="warning", message="miss")
    result = _make_ingest_result(issues=[issue])
    await repo.complete_run("r1", result)
    conn.execute.assert_called_once()
    args = conn.execute.call_args.args
    # primary_issue is at a specific position; check it's in the args
    assert "BARCODE_MISSING" in args


async def test_pg_complete_run_with_error():
    pool, conn = _make_pool()
    repo = PostgresRunRepository(pool)
    result = _make_ingest_result(
        status=IngestStatus.FAILED,
        error={"type": "SCAN_ERROR", "message": "scanner failed"},
    )
    await repo.complete_run("r1", result)
    conn.execute.assert_called_once()
    args = conn.execute.call_args.args
    assert "SCAN_ERROR" in args
    assert "scanner failed" in args


async def test_pg_complete_run_error_with_code_key():
    pool, conn = _make_pool()
    repo = PostgresRunRepository(pool)
    result = _make_ingest_result(
        status=IngestStatus.FAILED,
        error={"code": "CUSTOM", "message": "custom err"},
    )
    await repo.complete_run("r1", result)
    args = conn.execute.call_args.args
    assert "CUSTOM" in args
    assert "custom err" in args


async def test_pg_complete_run_no_items_no_missing():
    pool, conn = _make_pool()
    repo = PostgresRunRepository(pool)
    result = _make_ingest_result()
    await repo.complete_run("r1", result)
    # Only the UPDATE, no INSERTs
    assert conn.execute.call_count == 1


async def test_pg_complete_run_uses_transaction():
    pool, conn = _make_pool()
    repo = PostgresRunRepository(pool)
    result = _make_ingest_result()
    await repo.complete_run("r1", result)
    conn.transaction.assert_called_once()


async def test_pg_complete_run_error_propagates():
    pool, _ = _make_pool_raising(asyncpg.PostgresError("boom"))
    repo = PostgresRunRepository(pool)
    result = _make_ingest_result()
    with pytest.raises(asyncpg.PostgresError, match="boom"):
        await repo.complete_run("r1", result)


async def test_pg_complete_run_item_without_bbox():
    pool, conn = _make_pool()
    repo = PostgresRunRepository(pool)
    item = DetectedItem(label_index=0, barcode_value="BC1")
    result = _make_ingest_result(items=[item])
    await repo.complete_run("r1", result)
    assert conn.execute.call_count == 2


async def test_pg_complete_run_missing_without_bbox():
    pool, conn = _make_pool()
    repo = PostgresRunRepository(pool)
    missing = [{"label_index": 3}]
    result = _make_ingest_result(missing=missing)
    await repo.complete_run("r1", result)
    assert conn.execute.call_count == 2


# ---------------------------------------------------------------------------
# PostgresRunRepository — fail_run
# ---------------------------------------------------------------------------


async def test_pg_fail_run_calls_execute():
    pool, conn = _make_pool()
    repo = PostgresRunRepository(pool)
    await repo.fail_run("r1", ValueError("something broke"))
    conn.execute.assert_called_once()
    args = conn.execute.call_args.args
    assert args[1] == "r1"
    assert args[2] == "ValueError"
    assert "something broke" in args[3]


async def test_pg_fail_run_truncates_long_message():
    pool, conn = _make_pool()
    repo = PostgresRunRepository(pool)
    long_msg = "x" * 600
    await repo.fail_run("r1", ValueError(long_msg))
    args = conn.execute.call_args.args
    assert len(args[3]) == 500


async def test_pg_fail_run_error_propagates():
    pool, _ = _make_pool_raising(OSError("conn refused"))
    repo = PostgresRunRepository(pool)
    with pytest.raises(OSError, match="conn refused"):
        await repo.fail_run("r1", ValueError("boom"))


# ---------------------------------------------------------------------------
# PostgresRunRepository — query_metrics
# ---------------------------------------------------------------------------


async def test_pg_query_metrics_no_group_by():
    rows = [{"id": "r1", "status": "complete"}, {"id": "r2", "status": "failed"}]
    pool, conn = _make_pool(fetch_rows=rows)
    repo = PostgresRunRepository(pool)
    result = await repo.query_metrics(hours=24)
    assert result["source"] == "postgres"
    assert result["grouped"] is False
    assert len(result["rows"]) == 2
    conn.fetch.assert_called_once()


async def test_pg_query_metrics_empty_rows():
    pool, conn = _make_pool(fetch_rows=[])
    repo = PostgresRunRepository(pool)
    result = await repo.query_metrics(hours=24)
    assert result["rows"] == []


async def test_pg_query_metrics_with_group_by():
    rows = [{"group_key": "web", "total": 10, "complete": 8}]
    pool, conn = _make_pool(fetch_rows=rows)
    repo = PostgresRunRepository(pool)
    result = await repo.query_metrics(hours=24, group_by="source")
    assert result["source"] == "postgres"
    assert result["grouped"] is True
    assert len(result["groups"]) == 1
    assert result["groups"][0]["group_key"] == "web"


async def test_pg_query_metrics_invalid_group_by():
    pool, _ = _make_pool()
    repo = PostgresRunRepository(pool)
    with pytest.raises(ValueError, match="Invalid group_by"):
        await repo.query_metrics(hours=24, group_by="malicious_field")


async def test_pg_query_metrics_all_groupable_fields():
    for field in GROUPABLE_FIELDS:
        pool, _ = _make_pool(fetch_rows=[])
        repo = PostgresRunRepository(pool)
        result = await repo.query_metrics(hours=1, group_by=field)
        assert result["grouped"] is True


async def test_pg_query_metrics_error_propagates():
    pool, _ = _make_pool_raising(asyncpg.PostgresError("boom"))
    repo = PostgresRunRepository(pool)
    with pytest.raises(asyncpg.PostgresError, match="boom"):
        await repo.query_metrics(hours=24)


# ---------------------------------------------------------------------------
# PostgresRunRepository — query_runs
# ---------------------------------------------------------------------------


async def test_pg_query_runs_returns_rows_with_metadata():
    rows = [
        {
            "id": "r1", "source": "web", "status": "complete",
            "filename": "f.jpg", "scanner_count": 2, "vision_count": 2,
            "scanner_vision_match": True, "count_delta": 0,
            "found_count": 2, "missing_count": 0, "unassigned_count": 0,
            "recovery_attempted": False, "recovery_labels_tried": 0,
            "recovery_barcodes_found": 0, "recovery_labels_resolved": 0,
            "recovery_succeeded": False, "latency_ms": 42,
            "primary_issue": None, "issue_count": 0,
            "pipeline_version": "v1", "scanner_version": "v2",
            "vision_prompt_version": "v3", "vision_model": "gemini",
            "recovery_version": "v4",
            "created_at": "2024-01-01", "processed_at": "2024-01-01",
        },
    ]
    pool, conn = _make_pool(fetch_rows=rows)
    repo = PostgresRunRepository(pool)
    result = await repo.query_runs(hours=24, limit=100)
    assert len(result) == 1
    d = result[0]
    assert d["id"] == "r1"
    assert d["metadata"]["final_status"] == "complete"
    assert d["metadata"]["found_count"] == 2
    assert d["metadata"]["latency_ms"] == 42
    assert d["metadata"]["scanner_vision_match"] is True
    assert d["metadata"]["pipeline_version"] == "v1"
    assert d["metadata"]["source"] == "web"


async def test_pg_query_runs_empty():
    pool, conn = _make_pool(fetch_rows=[])
    repo = PostgresRunRepository(pool)
    result = await repo.query_runs(hours=24)
    assert result == []


async def test_pg_query_runs_default_limit():
    pool, conn = _make_pool(fetch_rows=[])
    repo = PostgresRunRepository(pool)
    await repo.query_runs(hours=24)
    args = conn.fetch.call_args.args
    assert args[2] == 500


async def test_pg_query_runs_custom_limit():
    pool, conn = _make_pool(fetch_rows=[])
    repo = PostgresRunRepository(pool)
    await repo.query_runs(hours=24, limit=10)
    args = conn.fetch.call_args.args
    assert args[2] == 10


async def test_pg_query_runs_error_propagates():
    pool, _ = _make_pool_raising(OSError("conn refused"))
    repo = PostgresRunRepository(pool)
    with pytest.raises(OSError, match="conn refused"):
        await repo.query_runs(hours=24)


async def test_pg_query_runs_metadata_defaults_for_missing_keys():
    rows = [{"id": "r1", "source": "web", "status": "complete"}]
    pool, _ = _make_pool(fetch_rows=rows)
    repo = PostgresRunRepository(pool)
    result = await repo.query_runs(hours=24)
    d = result[0]
    assert d["metadata"]["found_count"] == 0
    assert d["metadata"]["scanner_count"] == 0
    assert d["metadata"]["scanner_vision_match"] is False
    assert d["metadata"]["recovery_attempted"] is False
    assert d["metadata"]["latency_ms"] == 0


# ---------------------------------------------------------------------------
# PostgresAnnotationRepository
# ---------------------------------------------------------------------------


async def test_pg_annotation_create():
    pool, conn = _make_pool()
    repo = PostgresAnnotationRepository(pool)
    await repo.create_annotation("r1")
    conn.execute.assert_called_once()
    args = conn.execute.call_args.args
    assert args[1] == "r1"


async def test_pg_annotation_create_error_propagates():
    pool, _ = _make_pool_raising(asyncpg.PostgresError("boom"))
    repo = PostgresAnnotationRepository(pool)
    with pytest.raises(asyncpg.PostgresError, match="boom"):
        await repo.create_annotation("r1")


async def test_pg_annotation_list_pending():
    rows = [{"run_id": "r1", "status": "pending"}]
    pool, conn = _make_pool(fetch_rows=rows)
    repo = PostgresAnnotationRepository(pool)
    result = await repo.list_pending()
    assert len(result) == 1
    assert result[0]["run_id"] == "r1"


async def test_pg_annotation_list_pending_empty():
    pool, _ = _make_pool(fetch_rows=[])
    repo = PostgresAnnotationRepository(pool)
    result = await repo.list_pending()
    assert result == []


async def test_pg_annotation_list_pending_with_limit():
    pool, conn = _make_pool(fetch_rows=[])
    repo = PostgresAnnotationRepository(pool)
    await repo.list_pending(limit=10)
    args = conn.fetch.call_args.args
    assert args[1] == 10


async def test_pg_annotation_list_pending_error_propagates():
    pool, _ = _make_pool_raising(OSError("conn refused"))
    repo = PostgresAnnotationRepository(pool)
    with pytest.raises(OSError, match="conn refused"):
        await repo.list_pending()


async def test_pg_annotation_review():
    pool, conn = _make_pool()
    repo = PostgresAnnotationRepository(pool)
    await repo.review("r1")
    conn.execute.assert_called_once()
    args = conn.execute.call_args.args
    assert args[1] == "r1"
    assert args[2] is None  # reviewed_by
    assert args[3] is None  # expected_json
    assert args[4] is None  # expected_outcome


async def test_pg_annotation_review_all_params():
    pool, conn = _make_pool()
    repo = PostgresAnnotationRepository(pool)
    await repo.review(
        "r1",
        expected_barcodes=["BC1", "BC2"],
        expected_outcome="complete",
        reviewed_by="tester",
    )
    args = conn.execute.call_args.args
    assert args[1] == "r1"
    assert args[2] == "tester"
    assert json.loads(args[3]) == ["BC1", "BC2"]
    assert args[4] == "complete"


async def test_pg_annotation_review_error_propagates():
    pool, _ = _make_pool_raising(asyncpg.PostgresError("boom"))
    repo = PostgresAnnotationRepository(pool)
    with pytest.raises(asyncpg.PostgresError, match="boom"):
        await repo.review("r1")


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def test_get_run_repository_noop():
    repo = get_run_repository(None)
    assert isinstance(repo, NoOpRunRepository)


def test_get_run_repository_empty_string():
    repo = get_run_repository("")
    assert isinstance(repo, NoOpRunRepository)


def test_get_run_repository_with_url_raises():
    with pytest.raises(RuntimeError, match="Use create_repositories"):
        get_run_repository("postgres://localhost/db")


def test_get_annotation_repository_noop():
    repo = get_annotation_repository(None)
    assert isinstance(repo, NoOpAnnotationRepository)


def test_get_annotation_repository_empty_string():
    repo = get_annotation_repository("")
    assert isinstance(repo, NoOpAnnotationRepository)


def test_get_annotation_repository_with_url_raises():
    with pytest.raises(RuntimeError, match="Use create_repositories"):
        get_annotation_repository("postgres://localhost/db")


# ---------------------------------------------------------------------------
# GROUPABLE_FIELDS
# ---------------------------------------------------------------------------


def test_groupable_fields_contains_expected_keys():
    assert "source" in GROUPABLE_FIELDS
    assert "pipeline_version" in GROUPABLE_FIELDS
    assert "scanner_version" in GROUPABLE_FIELDS
    assert "vision_prompt_version" in GROUPABLE_FIELDS
    assert "vision_model" in GROUPABLE_FIELDS
    assert "recovery_version" in GROUPABLE_FIELDS


def test_groupable_fields_is_frozen():
    with pytest.raises(AttributeError):
        GROUPABLE_FIELDS.add("evil")  # type: ignore[attr-defined]

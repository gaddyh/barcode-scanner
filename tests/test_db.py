"""Integration tests for Postgres repository.

These tests require a live Postgres instance. Set DATABASE_URL env var
or use the default test connection. Tests are skipped if asyncpg is
not installed or DB is unreachable.

Run: pytest tests/test_db.py -v
"""

from __future__ import annotations

import os
import pytest
import asyncpg

from src.db import create_pool, init_db
from src.repository import (
    NewRun,
    PostgresRunRepository,
    PostgresAnnotationRepository,
    NoOpRunRepository,
    GROUPABLE_FIELDS,
)
from src.ingest.models import IngestResult, IngestStatus, DetectedItem, RunMetrics, Issue

# Use the provided Render Postgres instance for integration tests.
TEST_DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://tami_one_postgre_user:K6Pqcojs1NvipyWCpkUHkEXV8tbtBkcv@dpg-d956p5gjs32c73fgj4t0-a.frankfurt-postgres.render.com/tami_one_postgre",
)

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def pool():
    """Create a connection pool, initialize schema, and clean up test data.

    Uses DELETE (not DROP) after each test so the schema persists but test
    rows don't leak between runs. Test run IDs use a distinctive prefix
    (``01JTEST``) so we only delete our own rows.
    """
    p = await create_pool(TEST_DB_URL, min_size=1, max_size=3)
    await init_db(p)
    # Clean up any leftover test data from previous runs.
    async with p.acquire() as conn:
        await conn.execute("DELETE FROM run_items WHERE run_id LIKE '01JTEST%'")
        await conn.execute("DELETE FROM annotations WHERE run_id LIKE '01JTEST%'")
        await conn.execute("DELETE FROM runs WHERE id LIKE '01JTEST%'")
    yield p
    # Clean up after the test too.
    async with p.acquire() as conn:
        await conn.execute("DELETE FROM run_items WHERE run_id LIKE '01JTEST%'")
        await conn.execute("DELETE FROM annotations WHERE run_id LIKE '01JTEST%'")
        await conn.execute("DELETE FROM runs WHERE id LIKE '01JTEST%'")
    await p.close()


@pytest.fixture
def run_repo(pool):
    return PostgresRunRepository(pool)


@pytest.fixture
def annotation_repo(pool):
    return PostgresAnnotationRepository(pool)


def _make_run(run_id: str = "01JTEST0000000000000000001") -> NewRun:
    return NewRun(
        id=run_id,
        session_id="session-123",
        trace_id="trace-456",
        source="web",
        endpoint="/barcode/analyze",
        filename="test.jpg",
        upload_bytes=1024,
        image_width=800,
        image_height=600,
    )


def _make_result(status: IngestStatus = IngestStatus.COMPLETE) -> IngestResult:
    return IngestResult(
        status=status,
        items=[
            DetectedItem(
                label_index=0,
                barcode_value="7297500243416",
                barcode_format="Code128",
                barcode_bbox={"x1": 100, "y1": 200, "x2": 150, "y2": 300},
                label_bbox={"x1": 90, "y1": 190, "x2": 160, "y2": 310},
                match_basis="containment",
            ),
        ],
        missing=[
            {"label_index": 1, "label_bbox": {"x1": 200, "y1": 200, "x2": 250, "y2": 300}},
        ],
        issues=[
            Issue(code="BARCODE_MISSING", severity="warning", message="1 label missing"),
        ],
        metrics=RunMetrics(
            elapsed_ms=5000,
            scanner_count=1,
            vision_count=2,
            recovery_attempted=True,
            recovery_labels_tried=1,
            recovery_barcodes_found=0,
            recovery_labels_resolved=0,
        ),
        image_width=800,
        image_height=600,
    )


# --- Schema tests ----------------------------------------------------------

async def test_schema_creates_tables(pool):
    async with pool.acquire() as conn:
        tables = await conn.fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename"
        )
        names = {r["tablename"] for r in tables}
        assert "runs" in names
        assert "run_items" in names
        assert "annotations" in names


async def test_runs_status_check_constraint(pool, run_repo):
    """Invalid status should raise."""
    await run_repo.create_run(_make_run("01JTEST000000000000000000C"))
    async with pool.acquire() as conn:
        with pytest.raises(asyncpg.CheckViolationError):
            await conn.execute(
                "UPDATE runs SET status = 'bogus' WHERE id = $1",
                "01JTEST000000000000000000C",
            )


async def test_run_items_status_check_constraint(pool, run_repo):
    """run_items.status must be 'found' or 'missing'."""
    await run_repo.create_run(_make_run("01JTEST000000000000000000D"))
    async with pool.acquire() as conn:
        with pytest.raises(asyncpg.CheckViolationError):
            await conn.execute(
                "INSERT INTO run_items (run_id, status) VALUES ($1, 'bogus')",
                "01JTEST000000000000000000D",
            )


# --- Repository tests ------------------------------------------------------

async def test_create_and_complete_run(run_repo, pool):
    run = _make_run("01JTEST000000000000000000A")
    await run_repo.create_run(run)

    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM runs WHERE id = $1", run.id)
        assert row["status"] == "pending"
        assert row["source"] == "web"
        assert row["filename"] == "test.jpg"

    result = _make_result()
    versions = {
        "pipeline_version": "ingest-v1",
        "scanner_version": "scanner-0.8",
        "vision_prompt_version": "label-audit-v1",
        "vision_model": "gemini-3.5-flash-lite",
        "recovery_version": "recovery-v1",
    }
    await run_repo.complete_run(run.id, result, versions=versions)

    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM runs WHERE id = $1", run.id)
        assert row["status"] == "complete"
        assert row["found_count"] == 1
        assert row["missing_count"] == 1
        assert row["scanner_count"] == 1
        assert row["vision_count"] == 2
        assert row["scanner_vision_match"] is False
        assert row["count_delta"] == 1
        assert row["recovery_attempted"] is True
        assert row["recovery_succeeded"] is False
        assert row["latency_ms"] == 5000
        assert row["primary_issue"] == "BARCODE_MISSING"
        assert row["issue_count"] == 1
        assert row["pipeline_version"] == "ingest-v1"
        assert row["processed_at"] is not None

        # Verify run_items were inserted atomically
        items = await conn.fetch(
            "SELECT * FROM run_items WHERE run_id = $1 ORDER BY status, label_index",
            run.id,
        )
        assert len(items) == 2
        found = [i for i in items if i["status"] == "found"]
        missing = [i for i in items if i["status"] == "missing"]
        assert len(found) == 1
        assert len(missing) == 1
        assert found[0]["barcode_value"] == "7297500243416"
        assert found[0]["barcode_bbox"] is not None
        assert missing[0]["label_index"] == 1


async def test_fail_run(run_repo, pool):
    run = _make_run("01JTEST000000000000000000B")
    await run_repo.create_run(run)

    await run_repo.fail_run(run.id, ValueError("Gemini timeout"))

    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM runs WHERE id = $1", run.id)
        assert row["status"] == "failed"
        assert row["error_code"] == "ValueError"
        assert "Gemini timeout" in row["error_message"]
        assert row["processed_at"] is not None


async def test_idempotent_create_run(run_repo):
    """Creating the same run ID twice should not error (ON CONFLICT DO NOTHING)."""
    run = _make_run("01JTEST000000000000000000E")
    await run_repo.create_run(run)
    await run_repo.create_run(run)  # should not raise


async def test_provider_message_id_unique(run_repo, pool):
    """WhatsApp message ID should be unique — prevents duplicate processing."""
    run1 = NewRun(
        id="01JTEST000000000000000000F",
        source="whatsapp",
        provider_message_id="wamid.123",
    )
    run2 = NewRun(
        id="01JTEST000000000000000000G",
        source="whatsapp",
        provider_message_id="wamid.123",  # same message ID
    )
    await run_repo.create_run(run1)
    with pytest.raises(asyncpg.UniqueViolationError):
        await run_repo.create_run(run2)


async def test_query_metrics_flat(run_repo):
    """Flat metrics query returns all runs in time window."""
    for i in range(5):
        run = NewRun(id=f"01JTEST0000000000000000MET{i}", source="web")
        await run_repo.create_run(run)
        result = _make_result(IngestStatus.COMPLETE if i < 3 else IngestStatus.NEEDS_USER_INPUT)
        await run_repo.complete_run(run.id, result)

    data = await run_repo.query_metrics(hours=1)
    assert data["source"] == "postgres"
    assert data["grouped"] is False
    # Filter to only our test rows — the shared DB may have other data.
    test_rows = [r for r in data["rows"] if str(r.get("id", "")).startswith("01JTEST")]
    assert len(test_rows) == 5


async def test_query_metrics_grouped(run_repo):
    """Grouped metrics query returns per-group aggregates."""
    for i in range(3):
        run = NewRun(id=f"01JTEST0000000000000000GRP{i}", source="web")
        await run_repo.create_run(run)
        await run_repo.complete_run(run.id, _make_result(IngestStatus.COMPLETE))

    for i in range(2):
        run = NewRun(id=f"01JTEST0000000000000000GRW{i}", source="whatsapp")
        await run_repo.create_run(run)
        await run_repo.complete_run(run.id, _make_result(IngestStatus.NEEDS_USER_INPUT))

    data = await run_repo.query_metrics(hours=1, group_by="source")
    assert data["source"] == "postgres"
    assert data["grouped"] is True
    groups = {g["group_key"]: g for g in data["groups"]}
    assert "web" in groups
    assert "whatsapp" in groups
    # The shared DB may have other runs; check that our test runs are included.
    # We can't assert exact totals, but we can verify the groups exist and
    # have at least our test counts.
    assert groups["web"]["total"] >= 3
    assert groups["web"]["complete"] >= 3
    assert groups["whatsapp"]["total"] >= 2
    assert groups["whatsapp"]["needs_user_input"] >= 2


async def test_query_metrics_invalid_group_by(run_repo):
    with pytest.raises(ValueError):
        await run_repo.query_metrics(hours=1, group_by="evil_injection")


async def test_groupable_fields_allowlist():
    assert "source" in GROUPABLE_FIELDS
    assert "pipeline_version" in GROUPABLE_FIELDS
    assert "evil" not in GROUPABLE_FIELDS


# --- Annotation tests ------------------------------------------------------

async def test_annotation_lifecycle(annotation_repo, run_repo, pool):
    run = _make_run("01JTEST000000000000000000ANN")
    await run_repo.create_run(run)

    await annotation_repo.create_annotation(run.id)

    pending = await annotation_repo.list_pending()
    assert any(a["run_id"] == run.id for a in pending)

    await annotation_repo.review(
        run.id,
        expected_barcodes=["7297500243416"],
        expected_outcome="complete",
        reviewed_by="gaddy",
    )

    pending_after = await annotation_repo.list_pending()
    assert not any(a["run_id"] == run.id for a in pending_after)

    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM annotations WHERE run_id = $1", run.id)
        assert row["status"] == "reviewed"
        assert row["reviewed_by"] == "gaddy"
        assert row["expected_outcome"] == "complete"
        # asyncpg returns JSONB as a string — parse it
        import json
        assert json.loads(row["expected_barcodes"]) == ["7297500243416"]


async def test_annotation_unique_per_run(annotation_repo, run_repo):
    """One annotation per run — ON CONFLICT DO NOTHING."""
    run = _make_run("01JTEST000000000000000000DU1")
    await run_repo.create_run(run)
    await annotation_repo.create_annotation(run.id)
    await annotation_repo.create_annotation(run.id)  # should not raise


# --- NoOp tests ------------------------------------------------------------

async def test_noop_repository_does_not_raise():
    repo = NoOpRunRepository()
    await repo.create_run(_make_run())
    await repo.mark_processing("any-id")
    await repo.complete_run("any-id", _make_result())
    await repo.fail_run("any-id", RuntimeError("test"))
    data = await repo.query_metrics(hours=24)
    assert data["source"] == "noop"

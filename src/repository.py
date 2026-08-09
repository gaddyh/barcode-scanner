"""Repository boundary for operational persistence.

The application talks to ``RunRepository`` and ``AnnotationRepository``
protocols. It never sees SQL directly.

Implementations:
    - ``PostgresRunRepository`` / ``PostgresAnnotationRepository`` — production
    - ``NoOpRunRepository`` / ``NoOpAnnotationRepository`` — CLI/eval (no DB)

``ingest_one()`` stays pure — it never imports this module. The caller
(route handler, CLI, WhatsApp processor) owns the persistence lifecycle:

    await repo.create_run(run)          # status=pending
    await repo.mark_processing(run_id)  # status=processing
    result = await ingest_one(...)
    await repo.complete_run(run_id, result)  # atomic: UPDATE runs + INSERT items
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

import asyncpg

from src.ingest.models import IngestResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data transfer objects
# ---------------------------------------------------------------------------

class NewRun:
    """Fields needed to create a run row."""

    def __init__(
        self,
        *,
        id: str,
        session_id: str | None = None,
        trace_id: str | None = None,
        source: str,
        endpoint: str | None = None,
        filename: str | None = None,
        image_ref: str | None = None,
        upload_bytes: int | None = None,
        image_width: int | None = None,
        image_height: int | None = None,
        provider_message_id: str | None = None,
        sender: str | None = None,
    ):
        self.id = id
        self.session_id = session_id
        self.trace_id = trace_id
        self.source = source
        self.endpoint = endpoint
        self.filename = filename
        self.image_ref = image_ref
        self.upload_bytes = upload_bytes
        self.image_width = image_width
        self.image_height = image_height
        self.provider_message_id = provider_message_id
        self.sender = sender


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

class RunRepository(Protocol):
    async def create_run(self, run: NewRun) -> None: ...
    async def mark_processing(self, run_id: str) -> None: ...
    async def complete_run(self, run_id: str, result: IngestResult, *, versions: dict[str, str] | None = None) -> None: ...
    async def fail_run(self, run_id: str, error: Exception) -> None: ...
    async def query_metrics(self, *, hours: int, group_by: str | None = None) -> dict[str, Any]: ...


class AnnotationRepository(Protocol):
    async def create_annotation(self, run_id: str) -> None: ...
    async def list_pending(self, limit: int = 50) -> list[dict[str, Any]]: ...
    async def review(self, run_id: str, *, expected_barcodes: list[str] | None = None, expected_outcome: str | None = None, reviewed_by: str | None = None) -> None: ...


# ---------------------------------------------------------------------------
# No-op implementations (CLI / eval — no DB required)
# ---------------------------------------------------------------------------

class NoOpRunRepository:
    """Does nothing. Used when DATABASE_URL is not set."""

    async def create_run(self, run: NewRun) -> None:
        pass

    async def mark_processing(self, run_id: str) -> None:
        pass

    async def complete_run(self, run_id: str, result: IngestResult, *, versions: dict[str, str] | None = None) -> None:
        pass

    async def fail_run(self, run_id: str, error: Exception) -> None:
        pass

    async def query_metrics(self, *, hours: int, group_by: str | None = None) -> dict[str, Any]:
        return {"source": "noop", "runs": []}


class NoOpAnnotationRepository:
    async def create_annotation(self, run_id: str) -> None:
        pass

    async def list_pending(self, limit: int = 50) -> list[dict[str, Any]]:
        return []

    async def review(self, run_id: str, *, expected_barcodes: list[str] | None = None, expected_outcome: str | None = None, reviewed_by: str | None = None) -> None:
        pass


# ---------------------------------------------------------------------------
# Postgres implementations
# ---------------------------------------------------------------------------

# Allowlist for group_by — prevents SQL injection via interpolation.
GROUPABLE_FIELDS = frozenset({
    "source",
    "pipeline_version",
    "scanner_version",
    "vision_prompt_version",
    "vision_model",
    "recovery_version",
})


class PostgresRunRepository:
    """Postgres implementation of RunRepository."""

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    async def create_run(self, run: NewRun) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO runs (
                    id, session_id, trace_id, source, endpoint,
                    filename, image_ref, upload_bytes, image_width, image_height,
                    provider_message_id, sender, status
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, 'pending')
                ON CONFLICT (id) DO NOTHING
                """,
                run.id, run.session_id, run.trace_id, run.source, run.endpoint,
                run.filename, run.image_ref, run.upload_bytes, run.image_width,
                run.image_height, run.provider_message_id, run.sender,
            )

    async def mark_processing(self, run_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE runs SET status = 'processing' WHERE id = $1",
                run_id,
            )

    async def complete_run(
        self,
        run_id: str,
        result: IngestResult,
        *,
        versions: dict[str, str] | None = None,
    ) -> None:
        """Atomically update run status + insert run_items in one transaction."""
        versions = versions or {}
        issues_json = json.dumps([i.model_dump() for i in result.issues])
        primary_issue = result.issues[0].code if result.issues else None
        issue_count = len(result.issues)
        error_code = None
        error_message = None
        if result.error:
            error_code = result.error.get("type") or result.error.get("code")
            error_message = result.error.get("message")

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    UPDATE runs SET
                        status = $2,
                        image_width = COALESCE($3, image_width),
                        image_height = COALESCE($4, image_height),
                        scanner_count = $5,
                        vision_count = $6,
                        scanner_vision_match = $7,
                        count_delta = $8,
                        found_count = $9,
                        missing_count = $10,
                        unassigned_count = $11,
                        recovery_attempted = $12,
                        recovery_labels_tried = $13,
                        recovery_barcodes_found = $14,
                        recovery_labels_resolved = $15,
                        recovery_succeeded = $16,
                        latency_ms = $17,
                        primary_issue = $18,
                        issue_count = $19,
                        issues = $20::jsonb,
                        error_code = $21,
                        error_message = $22,
                        pipeline_version = $23,
                        scanner_version = $24,
                        vision_prompt_version = $25,
                        vision_model = $26,
                        recovery_version = $27,
                        processed_at = NOW()
                    WHERE id = $1
                    """,
                    run_id,
                    result.status.value,
                    result.image_width or None,
                    result.image_height or None,
                    result.metrics.scanner_count,
                    result.metrics.vision_count,
                    result.metrics.scanner_count == result.metrics.vision_count,
                    result.metrics.vision_count - result.metrics.scanner_count,
                    len(result.items),
                    len(result.missing),
                    len(result.unassigned),
                    result.metrics.recovery_attempted,
                    result.metrics.recovery_labels_tried,
                    result.metrics.recovery_barcodes_found,
                    result.metrics.recovery_labels_resolved,
                    result.metrics.recovery_labels_resolved > 0,
                    result.metrics.elapsed_ms,
                    primary_issue,
                    issue_count,
                    issues_json,
                    error_code,
                    error_message,
                    versions.get("pipeline_version"),
                    versions.get("scanner_version"),
                    versions.get("vision_prompt_version"),
                    versions.get("vision_model"),
                    versions.get("recovery_version"),
                )

                # Insert run_items — found items
                for item in result.items:
                    await conn.execute(
                        """
                        INSERT INTO run_items (
                            run_id, label_index, barcode_value, barcode_format,
                            barcode_bbox, label_bbox, match_basis, status
                        ) VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, $7, 'found')
                        """,
                        run_id,
                        item.label_index,
                        item.barcode_value,
                        item.barcode_format,
                        json.dumps(item.barcode_bbox) if item.barcode_bbox else None,
                        json.dumps(item.label_bbox) if item.label_bbox else None,
                        item.match_basis,
                    )

                # Insert run_items — missing labels
                for m in result.missing:
                    await conn.execute(
                        """
                        INSERT INTO run_items (
                            run_id, label_index, barcode_value, barcode_format,
                            barcode_bbox, label_bbox, match_basis, status
                        ) VALUES ($1, $2, NULL, NULL, $3::jsonb, $4::jsonb, NULL, 'missing')
                        """,
                        run_id,
                        m.get("label_index"),
                        json.dumps(m.get("barcode_bbox")) if m.get("barcode_bbox") else None,
                        json.dumps(m.get("label_bbox")) if m.get("label_bbox") else None,
                    )

    async def fail_run(self, run_id: str, error: Exception) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE runs SET
                    status = 'failed',
                    error_code = $2,
                    error_message = $3,
                    processed_at = NOW()
                WHERE id = $1
                """,
                run_id,
                type(error).__name__,
                str(error)[:500],
            )

    async def query_metrics(self, *, hours: int, group_by: str | None = None) -> dict[str, Any]:
        """Query runs table for metrics aggregation."""
        if group_by is not None and group_by not in GROUPABLE_FIELDS:
            raise ValueError(f"Invalid group_by '{group_by}'. Valid: {sorted(GROUPABLE_FIELDS)}")

        async with self._pool.acquire() as conn:
            if group_by is None:
                rows = await conn.fetch(
                    """
                    SELECT * FROM runs
                    WHERE created_at >= NOW() - make_interval(hours => $1)
                    ORDER BY created_at DESC
                    """,
                    hours,
                )
                return {"source": "postgres", "grouped": False, "rows": [dict(r) for r in rows]}
            else:
                rows = await conn.fetch(
                    f"""
                    SELECT {group_by} AS group_key,
                           COUNT(*) AS total,
                           COUNT(*) FILTER (WHERE status = 'complete') AS complete,
                           COUNT(*) FILTER (WHERE status = 'needs_user_input') AS needs_user_input,
                           COUNT(*) FILTER (WHERE status = 'needs_retry') AS needs_retry,
                           COUNT(*) FILTER (WHERE status = 'failed') AS failed,
                           COUNT(*) FILTER (WHERE recovery_attempted) AS recovery_attempted,
                           COUNT(*) FILTER (WHERE recovery_succeeded) AS recovery_succeeded,
                           AVG(latency_ms)::int AS avg_latency,
                           COALESCE(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms), 0)::int AS p95_latency,
                           AVG(missing_count)::float AS avg_missing,
                           AVG(unassigned_count)::float AS avg_unassigned,
                           AVG(count_delta)::float AS avg_count_delta
                    FROM runs
                    WHERE created_at >= NOW() - make_interval(hours => $1)
                      AND {group_by} IS NOT NULL
                    GROUP BY {group_by}
                    ORDER BY total DESC
                    """,
                    hours,
                )
                return {"source": "postgres", "grouped": True, "groups": [dict(r) for r in rows]}


class PostgresAnnotationRepository:
    """Postgres implementation of AnnotationRepository."""

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    async def create_annotation(self, run_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO annotations (run_id) VALUES ($1) ON CONFLICT (run_id) DO NOTHING",
                run_id,
            )

    async def list_pending(self, limit: int = 50) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT a.*, r.source, r.status AS run_status, r.filename,
                       r.scanner_count, r.vision_count, r.primary_issue
                FROM annotations a
                JOIN runs r ON r.id = a.run_id
                WHERE a.status = 'pending'
                ORDER BY a.created_at DESC
                LIMIT $1
                """,
                limit,
            )
            return [dict(r) for r in rows]

    async def review(
        self,
        run_id: str,
        *,
        expected_barcodes: list[str] | None = None,
        expected_outcome: str | None = None,
        reviewed_by: str | None = None,
    ) -> None:
        expected_json = json.dumps(expected_barcodes) if expected_barcodes else None
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE annotations SET
                    status = 'reviewed',
                    reviewed_at = NOW(),
                    reviewed_by = $2,
                    expected_barcodes = $3::jsonb,
                    expected_outcome = $4
                WHERE run_id = $1
                """,
                run_id,
                reviewed_by,
                expected_json,
                expected_outcome,
            )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_run_repository(database_url: str | None) -> RunRepository:
    """Return a Postgres repository if DATABASE_URL is set, else NoOp."""
    if database_url:
        # Pool is created lazily by the app lifespan; this is just a marker.
        # The actual pool is passed to PostgresRunRepository at startup.
        raise RuntimeError("Use create_repositories() for Postgres — needs a pool")
    return NoOpRunRepository()


def get_annotation_repository(database_url: str | None) -> AnnotationRepository:
    if database_url:
        raise RuntimeError("Use create_repositories() for Postgres — needs a pool")
    return NoOpAnnotationRepository()

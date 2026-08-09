"""Postgres connection pool and schema management.

The operational database stores every incoming request (runs), individual
barcode results (run_items), and annotation review state (annotations).

LangSmith remains the observability/evaluation system of record. This DB
is the operational system of record — fast admin metrics, request history,
and annotation workflow.
"""

from __future__ import annotations

import logging

import asyncpg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
-- Idempotent schema creation — safe to call on every startup.
-- Does NOT drop existing tables or data.
CREATE TABLE IF NOT EXISTS runs (
    id                  TEXT PRIMARY KEY,
    session_id          TEXT,
    trace_id            TEXT,
    source              TEXT NOT NULL,
    endpoint            TEXT,
    status              TEXT NOT NULL DEFAULT 'pending'
                        CHECK (status IN (
                            'pending', 'processing',
                            'complete', 'needs_user_input',
                            'needs_retry', 'failed'
                        )),

    filename            TEXT,
    image_ref           TEXT,
    upload_bytes        INTEGER,
    image_width         INTEGER,
    image_height        INTEGER,

    scanner_count       INTEGER DEFAULT 0,
    vision_count        INTEGER DEFAULT 0,
    scanner_vision_match BOOLEAN NOT NULL DEFAULT FALSE,
    count_delta         INTEGER DEFAULT 0,

    found_count         INTEGER DEFAULT 0,
    missing_count       INTEGER DEFAULT 0,
    unassigned_count    INTEGER DEFAULT 0,

    recovery_attempted  BOOLEAN NOT NULL DEFAULT FALSE,
    recovery_labels_tried    INTEGER DEFAULT 0,
    recovery_barcodes_found  INTEGER DEFAULT 0,
    recovery_labels_resolved INTEGER DEFAULT 0,
    recovery_succeeded  BOOLEAN NOT NULL DEFAULT FALSE,

    latency_ms          INTEGER DEFAULT 0,

    primary_issue       TEXT,
    issue_count         INTEGER DEFAULT 0,
    issues              JSONB,

    error_code          TEXT,
    error_message       TEXT,

    pipeline_version    TEXT,
    scanner_version     TEXT,
    vision_prompt_version TEXT,
    vision_model        TEXT,
    recovery_version    TEXT,

    provider_message_id TEXT,
    sender              TEXT,

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed_at        TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at);
CREATE INDEX IF NOT EXISTS idx_runs_status_created ON runs(status, created_at);
CREATE INDEX IF NOT EXISTS idx_runs_source_created ON runs(source, created_at);
CREATE INDEX IF NOT EXISTS idx_runs_scanner_version_created ON runs(scanner_version, created_at);
CREATE INDEX IF NOT EXISTS idx_runs_pipeline_version_created ON runs(pipeline_version, created_at);
CREATE INDEX IF NOT EXISTS idx_runs_trace_id ON runs(trace_id);
CREATE UNIQUE INDEX IF NOT EXISTS uq_runs_provider_message
    ON runs(provider_message_id)
    WHERE provider_message_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS run_items (
    id                  BIGSERIAL PRIMARY KEY,
    run_id              TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    label_index         INTEGER,
    barcode_value       TEXT,
    barcode_format      TEXT,
    barcode_bbox        JSONB,
    label_bbox          JSONB,
    match_basis         TEXT,
    status              TEXT CHECK (status IN ('found', 'missing')),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_run_items_run_id ON run_items(run_id);

CREATE TABLE IF NOT EXISTS annotations (
    id                  BIGSERIAL PRIMARY KEY,
    run_id              TEXT NOT NULL UNIQUE REFERENCES runs(id) ON DELETE CASCADE,
    status              TEXT NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending', 'reviewed', 'exported')),
    reviewed_at         TIMESTAMPTZ,
    reviewed_by         TEXT,
    expected_barcodes   JSONB,
    expected_outcome    TEXT,
    added_to_dataset    BOOLEAN NOT NULL DEFAULT FALSE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_annotations_status ON annotations(status);
"""


async def init_db(pool: asyncpg.Pool) -> None:
    """Create schema if it doesn't exist. Safe to call on every startup.

    Uses ``CREATE TABLE IF NOT EXISTS`` — existing tables and data are
    preserved across restarts and deploys.
    """
    async with pool.acquire() as conn:
        await conn.execute(_SCHEMA_SQL)
    logger.info("Database schema initialized")


async def create_pool(database_url: str, *, min_size: int = 2, max_size: int = 10) -> asyncpg.Pool:
    """Create an asyncpg connection pool."""
    return await asyncpg.create_pool(
        dsn=database_url,
        min_size=min_size,
        max_size=max_size,
        command_timeout=30,
    )

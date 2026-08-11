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

-- ---------------------------------------------------------------------------
-- Local Priority-compatible catalog (temporary ERP stand-in)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS priority_customers (
    id                  TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    active              BOOLEAN NOT NULL DEFAULT TRUE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS priority_branches (
    id                  TEXT PRIMARY KEY,
    customer_id         TEXT NOT NULL REFERENCES priority_customers(id) ON DELETE CASCADE,
    name                TEXT NOT NULL,
    active              BOOLEAN NOT NULL DEFAULT TRUE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_priority_branches_customer
    ON priority_branches(customer_id, active, name);

CREATE TABLE IF NOT EXISTS priority_orders (
    id                  BIGSERIAL PRIMARY KEY,
    session_id          TEXT,
    customer_id         TEXT NOT NULL REFERENCES priority_customers(id),
    branch_id           TEXT NOT NULL REFERENCES priority_branches(id),
    action              TEXT NOT NULL CHECK (
        action IN ('create_order', 'verify_order_before_shipment')
    ),
    status              TEXT NOT NULL DEFAULT 'draft',
    items               JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Development seed data. ON CONFLICT preserves edits made in the database.
INSERT INTO priority_customers (id, name) VALUES
    ('cust-acme', 'Acme Retail'),
    ('cust-northstar', 'Northstar Shoes'),
    ('cust-demo', 'Demo Customer')
ON CONFLICT (id) DO NOTHING;

INSERT INTO priority_branches (id, customer_id, name) VALUES
    ('branch-acme-main', 'cust-acme', 'Acme Main Store'),
    ('branch-acme-outlet', 'cust-acme', 'Acme Outlet'),
    ('branch-northstar-central', 'cust-northstar', 'Northstar Central'),
    ('branch-northstar-west', 'cust-northstar', 'Northstar West'),
    ('branch-demo', 'cust-demo', 'Demo Branch')
ON CONFLICT (id) DO NOTHING;

-- ---------------------------------------------------------------------------
-- Multi-image ingest sessions (M16)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS sessions (
    id                  TEXT PRIMARY KEY,
    status              TEXT NOT NULL DEFAULT 'active'
                        CHECK (status IN ('active', 'complete', 'expired', 'failed', 'closed',
                                          'needs_user_selection')),
    expected_count      INTEGER NOT NULL DEFAULT 0,
    found_count         INTEGER NOT NULL DEFAULT 0,
    missing_count       INTEGER NOT NULL DEFAULT 0,
    image_count         INTEGER NOT NULL DEFAULT 0,
    channel             TEXT,                          -- 'web', 'whatsapp'
    participant_id      TEXT,                          -- WhatsApp sender; null for web
    customer_id         TEXT,
    branch_id           TEXT,
    action              TEXT,
    source              TEXT,                          -- legacy: 'web', 'whatsapp', 'cli'
    message             TEXT,                          -- prompt for next photo
    candidates          JSONB,                         -- pending candidates for user selection
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_activity_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at        TIMESTAMPTZ,
    closed_at           TIMESTAMPTZ
);

-- Additive columns for pre-existing tables (idempotent).
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS last_activity_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS closed_at TIMESTAMPTZ;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS channel TEXT;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS participant_id TEXT;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS customer_id TEXT;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS branch_id TEXT;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS action TEXT;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS candidates JSONB;

-- Drop and recreate the status CHECK constraint to include 'closed'.
-- The original constraint (from CREATE TABLE) only allowed
-- active/complete/expired/failed. We need 'closed' for the DELETE endpoint.
DO $$
BEGIN
    -- Drop any existing check constraint on sessions.status
    EXECUTE (
        SELECT 'ALTER TABLE sessions DROP CONSTRAINT IF EXISTS ' || conname
        FROM pg_constraint
        WHERE conrelid = 'sessions'::regclass
          AND contype = 'c'
          AND pg_get_constraintdef(oid) LIKE '%status%'
        LIMIT 1
    );
EXCEPTION WHEN OTHERS THEN
    NULL;
END $$;

ALTER TABLE sessions ADD CONSTRAINT sessions_status_check
    CHECK (status IN ('active', 'complete', 'expired', 'failed', 'closed',
                      'needs_user_selection'));

CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions(created_at);
CREATE INDEX IF NOT EXISTS idx_sessions_last_activity ON sessions(last_activity_at);
CREATE INDEX IF NOT EXISTS idx_sessions_participant
    ON sessions(participant_id, channel, status)
    WHERE participant_id IS NOT NULL AND status = 'active';

-- Confirmed barcodes accumulated across all images in a session.
-- Deduplicated by barcode_value within a session.
CREATE TABLE IF NOT EXISTS session_items (
    id                  BIGSERIAL PRIMARY KEY,
    session_id          TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    barcode_value       TEXT NOT NULL,
    barcode_format      TEXT,
    barcode_bbox        JSONB,
    label_bbox          JSONB,
    label_index         INTEGER,
    match_basis         TEXT,
    source_image        INTEGER NOT NULL DEFAULT 0,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Drop the old unique constraint (session_id, barcode_value) if it exists,
-- since multiple boxes can share the same barcode value (same product,
-- multiple units). Replace with a unique on (session_id, source_image,
-- label_index) to prevent duplicate labels within the same image.
DO $$
BEGIN
    EXECUTE (
        SELECT 'ALTER TABLE session_items DROP CONSTRAINT IF EXISTS ' || conname
        FROM pg_constraint
        WHERE conrelid = 'session_items'::regclass
          AND contype = 'u'
          AND pg_get_constraintdef(oid) LIKE '%session_id, barcode_value%'
        LIMIT 1
    );
EXCEPTION WHEN OTHERS THEN
    NULL;
END $$;

CREATE UNIQUE INDEX IF NOT EXISTS uq_session_items_label
    ON session_items(session_id, source_image, label_index)
    WHERE label_index IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_session_items_session ON session_items(session_id);

-- Boxes visible but without a decoded barcode, tracked across images.
CREATE TABLE IF NOT EXISTS session_missing (
    id                  BIGSERIAL PRIMARY KEY,
    session_id          TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    label_index         INTEGER,
    label_bbox          JSONB,
    barcode_bbox        JSONB,
    status              TEXT NOT NULL DEFAULT 'not_visible',
    source_image        INTEGER NOT NULL DEFAULT 0,
    resolved            BOOLEAN NOT NULL DEFAULT FALSE,
    resolved_by_image   INTEGER,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_session_missing_session ON session_missing(session_id);
CREATE INDEX IF NOT EXISTS idx_session_missing_unresolved
    ON session_missing(session_id) WHERE resolved = FALSE;
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

"""Admin API — annotation queue management + operational metrics.

Endpoints for reviewing interesting pipeline failures, promoting
reviewed candidates to the eval dataset, and querying operational
metrics.

Metrics source priority:
    1. Postgres (if DATABASE_URL is set and pool is available) — fast, no
       external API calls, no 100-run limit.
    2. LangSmith (fallback if DB is unavailable) — queries trace metadata.

No auth — dev only. Add auth before any real deployment.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime, timedelta
from typing import Any

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.evals.annotation_store import (
    export_to_dataset_json,
    get_candidate,
    get_stats,
    list_pending,
    submit_review,
)
from src.evals.metrics import (
    VALID_GROUP_BY,
    GroupedMetricsResponse,
    MetricsResponse,
    compute_grouped_metrics,
    compute_metrics,
    unavailable_grouped_response,
    unavailable_response,
)
from src.repository import NoOpRunRepository

load_dotenv()

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class CandidateSummary(BaseModel):
    id: str
    run_id: str
    source: str
    status: str
    image_ref: str | None
    scanner_count: int
    vision_count: int
    recovery_attempted: bool
    recovery_labels_resolved: int
    created_at: str
    reviewed_at: str | None = None
    reviewed_by: str | None = None
    added_to_dataset: bool = False


class ReviewRequest(BaseModel):
    expected_barcodes: list[dict[str, Any]]
    expected_outcome: str
    reviewer: str


class ExportResponse(BaseModel):
    exported_count: int


class StatsResponse(BaseModel):
    pending: int
    reviewed: int
    exported: int
    total: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/annotation/pending")
def get_pending(limit: int = 50) -> list[CandidateSummary]:
    """List unreviewed annotation candidates, newest first."""
    candidates = list_pending(limit=limit)
    return [_candidate_to_summary(c) for c in candidates]


@router.get("/annotation/{candidate_id}")
def get_one(candidate_id: str) -> dict[str, Any]:
    """Get one candidate with full details."""
    cand = get_candidate(candidate_id)
    if cand is None:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return cand.model_dump(mode="json")


@router.post("/annotation/{candidate_id}/review")
def post_review(candidate_id: str, req: ReviewRequest) -> dict[str, str]:
    """Submit ground-truth barcodes for a candidate."""
    ok = submit_review(
        candidate_id,
        expected_barcodes=req.expected_barcodes,
        expected_outcome=req.expected_outcome,
        reviewer=req.reviewer,
    )
    if not ok:
        raise HTTPException(
            status_code=404,
            detail="Candidate not found or already reviewed",
        )
    return {"status": "reviewed"}


@router.post("/annotation/export")
def post_export() -> ExportResponse:
    """Export reviewed candidates to dataset.json. Returns count added."""
    count = export_to_dataset_json()
    return ExportResponse(exported_count=count)


@router.get("/annotation/stats")
def get_annotation_stats() -> StatsResponse:
    """Return counts: pending, reviewed, exported, total."""
    stats = get_stats()
    return StatsResponse(**stats)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _candidate_to_summary(cand: Any) -> CandidateSummary:
    return CandidateSummary(
        id=cand.id,
        run_id=cand.run_id,
        source=cand.source,
        status=cand.status,
        image_ref=cand.image_ref,
        scanner_count=cand.scanner_count,
        vision_count=cand.vision_count,
        recovery_attempted=cand.recovery_attempted,
        recovery_labels_resolved=cand.recovery_labels_resolved,
        created_at=cand.created_at.isoformat(),
        reviewed_at=cand.reviewed_at.isoformat() if cand.reviewed_at else None,
        reviewed_by=cand.reviewed_by,
        added_to_dataset=cand.added_to_dataset,
    )


# ---------------------------------------------------------------------------
# Metrics — operational health from LangSmith traces
# ---------------------------------------------------------------------------


_METRICS_LIMIT = 100
_DB_METRICS_LIMIT = 2000


def _get_run_repo():
    """Get the module-level run_repo from src.main (set by lifespan)."""
    from src.main import run_repo
    return run_repo


class _DbRowRunLike:
    """Adapter that wraps a DB row dict as a RunLike for compute_metrics()."""

    def __init__(self, row: dict[str, Any]):
        self._row = row
        self.metadata = row.get("metadata", {})


@router.get("/metrics")
async def get_metrics(
    hours: int = 24,
    group_by: str | None = None,
) -> MetricsResponse | GroupedMetricsResponse:
    """Return operational metrics.

    Source priority:
        1. Postgres (if available) — fast, no external API, no 100-run limit.
        2. LangSmith (fallback if DB unavailable) — queries trace metadata.

    If the DB is accessible but empty, returns zeroed metrics with
    ``source="postgres"`` — no LangSmith fallback for empty data.

    If ``group_by`` is set (e.g. ``pipeline_version``, ``scanner_version``,
    ``recovery_version``, ``vision_prompt_version``, ``vision_model``,
    ``source``), returns a ``GroupedMetricsResponse`` with one
    ``MetricsResponse`` per group — true rates per version, not raw counts.
    """
    if group_by is not None and group_by not in VALID_GROUP_BY:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid group_by '{group_by}'. Valid values: {sorted(VALID_GROUP_BY)}",
        )

    # --- Try Postgres first ---
    repo = _get_run_repo()
    # NoOpRunRepository means no DB configured — skip to LangSmith fallback
    db_available = not isinstance(repo, NoOpRunRepository)

    if db_available:
        try:
            rows = await repo.query_runs(hours=hours, limit=_DB_METRICS_LIMIT)
            runs = [_DbRowRunLike(r) for r in rows]
            truncated = len(runs) >= _DB_METRICS_LIMIT

            if group_by is not None:
                resp = compute_grouped_metrics(
                    runs,
                    group_by=group_by,
                    time_window_hours=hours,
                    truncated=truncated,
                )
                resp.source = "postgres"
                return resp

            resp = compute_metrics(
                runs,
                time_window_hours=hours,
                truncated=truncated,
            )
            resp.source = "postgres"
            return resp
        except Exception:
            logger.warning("DB metrics query failed, falling back to LangSmith", exc_info=True)

    # --- Fallback: LangSmith ---
    start = datetime.now(UTC) - timedelta(hours=hours)
    project = os.getenv("LANGSMITH_PROJECT", "default")

    try:
        from langsmith import Client

        client = Client()
        runs = list(
            client.list_runs(
                project_name=project,
                filter=(
                    'or('
                    'eq(name, "ingest_one"), '
                    'eq(name, "web_analyze_barcode"), '
                    'eq(name, "process_whatsapp_message")'
                    ')'
                ),
                is_root=True,
                start_time=start,
                limit=_METRICS_LIMIT,
            )
        )
    except Exception:
        logger.exception("admin_metrics_query_failed")
        if group_by is not None:
            return unavailable_grouped_response(group_by=group_by, time_window_hours=hours)
        return unavailable_response(time_window_hours=hours)

    truncated = len(runs) >= _METRICS_LIMIT

    if group_by is not None:
        return compute_grouped_metrics(
            runs,
            group_by=group_by,
            time_window_hours=hours,
            truncated=truncated,
        )

    return compute_metrics(
        runs,
        time_window_hours=hours,
        truncated=truncated,
    )

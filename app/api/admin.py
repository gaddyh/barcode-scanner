"""Admin API — annotation queue management.

Endpoints for reviewing interesting pipeline failures and promoting
reviewed candidates to the eval dataset.

No auth — dev only. Add auth before any real deployment.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.evals.annotation_store import (
    export_to_dataset_json,
    get_candidate,
    get_stats,
    list_pending,
    submit_review,
)

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

"""Typed domain models for the ingest pipeline.

These models replace the untyped dict returned by ``analyze_image()``.
``IngestResult`` is the canonical output of ``ingest_one()`` — every
consumer (API, WhatsApp, CLI, eval, online feedback) reads from this
single typed contract.

The mapping from the old dict shape is:

    dict["outcome"] == "complete"          → IngestStatus.COMPLETE
    dict["outcome"] == "needs_better_photo" → IngestStatus.NEEDS_USER_INPUT
    dict["outcome"] == "retryable_error"   → IngestStatus.NEEDS_RETRY (ok=True)
                                             or IngestStatus.FAILED (ok=False)
    dict["found"]                          → items
    dict["missing"]                        → missing
    dict["unassigned"]                     → unassigned
    dict["summary"]                        → metrics
    dict["annotated_image_b64"]            → annotated_image_b64
    dict["message"]                        → message
    (derived)                              → issues
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class IngestStatus(StrEnum):
    """Outcome of one ingest operation."""

    COMPLETE = "complete"
    NEEDS_RETRY = "needs_retry"
    NEEDS_USER_INPUT = "needs_user_input"
    FAILED = "failed"


class Issue(BaseModel):
    """A structured issue discovered during ingest.

    Issues are the common language for monitoring and evaluation.
    Every component can emit issues with stable ``code`` strings.
    """

    code: str
    severity: str = Field(description='"info", "warning", or "error"')
    message: str
    evidence: dict[str, Any] = Field(default_factory=dict)


class DetectedItem(BaseModel):
    """A barcode successfully matched to a product label."""

    label_index: int | None = None
    barcode_value: str
    barcode_format: str | None = None
    barcode_bbox: dict[str, Any] | None = None
    label_bbox: dict[str, Any] | None = None
    match_basis: str | None = None


class RunMetrics(BaseModel):
    """Operational metrics for one ingest run."""

    elapsed_ms: int = 0
    scanner_count: int = 0
    vision_count: int = 0
    recovery_attempted: bool = False
    recovery_labels_tried: int = 0
    recovery_barcodes_found: int = 0
    recovery_labels_resolved: int = 0


class IngestResult(BaseModel):
    """The canonical typed result of ``ingest_one()``.

    This is the single contract that production, offline eval, and online
    eval all consume. It replaces the untyped dict from ``analyze_image()``.
    """

    status: IngestStatus
    items: list[DetectedItem] = Field(default_factory=list)
    missing: list[dict[str, Any]] = Field(default_factory=list)
    unassigned: list[dict[str, Any]] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)
    metrics: RunMetrics = Field(default_factory=RunMetrics)
    image_width: int = 0
    image_height: int = 0
    audit_available: bool = False
    annotated_image_b64: str | None = None
    annotated_image_width: int | None = None
    annotated_image_height: int | None = None
    message: str | None = None
    error: dict[str, Any] | None = None

"""Ingest package — typed domain models and the canonical ingest boundary."""

from src.ingest.models import (
    DetectedItem,
    IngestResult,
    IngestStatus,
    Issue,
    RunMetrics,
)
from src.ingest.service import ingest_one

__all__ = [
    "ingest_one",
    "IngestResult",
    "IngestStatus",
    "DetectedItem",
    "Issue",
    "RunMetrics",
]

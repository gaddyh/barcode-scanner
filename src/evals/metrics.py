"""Operational metrics — pure aggregation over LangSmith run metadata.

``compute_metrics()`` is a pure function that takes run-like objects (any
object with ``.metadata`` dict) and returns a ``MetricsResponse``. This
makes all aggregation logic testable without LangSmith.

The endpoint in ``app/api/admin.py`` is a thin wrapper:
    query LangSmith → compute_metrics() → return
"""

from __future__ import annotations

import math
from typing import Any, Protocol

from pydantic import BaseModel


class RunLike(Protocol):
    """Minimal interface for a LangSmith Run (or test stub)."""

    metadata: dict[str, Any]


class MetricsResponse(BaseModel):
    """8 operational metrics + metadata about the query."""

    time_window_hours: int
    source: str = "langsmith"  # "langsmith" or "unavailable"
    truncated: bool = False
    images_processed: int = 0
    boxes_processed: int = 0
    first_pass_complete_pct: float = 0.0
    final_complete_pct: float = 0.0
    recovery_attempted_pct: float = 0.0
    recovery_success_pct: float = 0.0
    user_retry_required_pct: float = 0.0
    p95_latency_ms: float = 0.0


def _pct(numerator: int, denominator: int) -> float:
    """Safe percentage — 0.0 if denominator is 0."""
    if denominator == 0:
        return 0.0
    return round(100.0 * numerator / denominator, 1)


def _p95(values: list[int]) -> float:
    """95th percentile using nearest-rank method."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    rank = math.ceil(0.95 * len(sorted_vals))
    idx = min(rank - 1, len(sorted_vals) - 1)
    return float(sorted_vals[idx])


def compute_metrics(
    runs: list[RunLike],
    *,
    time_window_hours: int = 24,
    truncated: bool = False,
) -> MetricsResponse:
    """Aggregate operational metrics from a list of LangSmith runs.

    Args:
        runs: List of run-like objects (real LangSmith Runs or test stubs).
            Each must have a ``.metadata`` dict with keys like
            ``final_status``, ``source``, ``found_count``,
            ``recovery_attempted``, ``recovery_labels_resolved``, ``latency_ms``.
        time_window_hours: The query window (for the response).
        truncated: True if the query hit its limit (len(runs) == 1000).

    Returns:
        ``MetricsResponse`` with 8 metrics.
    """
    total = len(runs)
    if total == 0:
        return MetricsResponse(
            time_window_hours=time_window_hours,
            source="langsmith",
            truncated=truncated,
        )

    images_processed = total
    boxes_processed = sum(
        _meta_int(r.metadata, "found_count") for r in runs
    )

    final_complete = sum(
        1 for r in runs
        if r.metadata.get("final_status") == "complete"
    )
    first_pass_complete = sum(
        1 for r in runs
        if r.metadata.get("final_status") == "complete"
        and not _meta_bool(r.metadata, "recovery_attempted")
    )
    recovery_attempted = sum(
        1 for r in runs if _meta_bool(r.metadata, "recovery_attempted")
    )
    recovery_succeeded = sum(
        1 for r in runs
        if _meta_bool(r.metadata, "recovery_attempted")
        and _meta_int(r.metadata, "recovery_labels_resolved") > 0
    )
    user_retry = sum(
        1 for r in runs
        if r.metadata.get("final_status") == "needs_user_input"
    )

    latencies = [
        _meta_int(r.metadata, "latency_ms")
        for r in runs
        if _meta_int(r.metadata, "latency_ms") > 0
    ]

    return MetricsResponse(
        time_window_hours=time_window_hours,
        source="langsmith",
        truncated=truncated,
        images_processed=images_processed,
        boxes_processed=boxes_processed,
        first_pass_complete_pct=_pct(first_pass_complete, total),
        final_complete_pct=_pct(final_complete, total),
        recovery_attempted_pct=_pct(recovery_attempted, total),
        recovery_success_pct=_pct(recovery_succeeded, recovery_attempted),
        user_retry_required_pct=_pct(user_retry, total),
        p95_latency_ms=_p95(latencies),
    )


def unavailable_response(time_window_hours: int = 24) -> MetricsResponse:
    """Return a zeroed response with source='unavailable'."""
    return MetricsResponse(
        time_window_hours=time_window_hours,
        source="unavailable",
    )


# ---------------------------------------------------------------------------
# Helpers — robust metadata extraction
# ---------------------------------------------------------------------------


def _meta_int(meta: dict[str, Any], key: str) -> int:
    """Extract an int from metadata, handling str/bool/None."""
    val = meta.get(key)
    if val is None:
        return 0
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


def _meta_bool(meta: dict[str, Any], key: str) -> bool:
    """Extract a bool from metadata, handling str/int/None."""
    val = meta.get(key)
    if val is None:
        return False
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "1", "yes")
    if isinstance(val, (int, float)):
        return val != 0
    return bool(val)

"""Operational metrics — pure aggregation over LangSmith run metadata.

``compute_metrics()`` is a pure function that takes run-like objects (any
object with ``.metadata`` dict) and returns a ``MetricsResponse``. This
makes all aggregation logic testable without LangSmith.

The endpoint in ``app/api/admin.py`` is a thin wrapper:
    query LangSmith → compute_metrics() → return
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from typing import Any, Protocol

from pydantic import BaseModel


class RunLike(Protocol):
    """Minimal interface for a LangSmith Run (or test stub)."""

    metadata: dict[str, Any]


class VersionRate(BaseModel):
    """One version's rate for a specific metric."""

    version: str
    total: int
    rate_pct: float


class MetricsResponse(BaseModel):
    """Operational, quality, and recovery metrics + query metadata."""

    time_window_hours: int
    source: str = "langsmith"  # "langsmith" or "unavailable"
    truncated: bool = False

    # Operations
    images_processed: int = 0
    boxes_processed: int = 0
    first_pass_complete_pct: float = 0.0
    final_complete_pct: float = 0.0
    recovery_attempted_pct: float = 0.0
    recovery_success_pct: float = 0.0
    user_retry_required_pct: float = 0.0
    p95_latency_ms: float = 0.0

    # Quality
    scanner_vision_match_pct: float = 0.0
    avg_count_delta: float = 0.0
    avg_missing_count: float = 0.0
    avg_unassigned_count: float = 0.0
    recovered_complete_pct: float = 0.0
    still_incomplete_pct: float = 0.0

    # Recovery detail
    avg_labels_tried: float = 0.0
    avg_labels_resolved: float = 0.0

    # Issues
    primary_issue_counts: dict[str, int] = {}

    # Version breakdowns (true rates, not counts)
    completion_by_pipeline_version: list[VersionRate] = []
    mismatch_by_scanner_version: list[VersionRate] = []
    recovery_success_by_recovery_version: list[VersionRate] = []
    retry_by_vision_model: list[VersionRate] = []


def _pct(numerator: int, denominator: int) -> float:
    """Safe percentage — 0.0 if denominator is 0."""
    if denominator == 0:
        return 0.0
    return round(100.0 * numerator / denominator, 1)


def _avg(values: list[int]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 1)


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
    """Aggregate operational, quality, and recovery metrics from LangSmith runs.

    Args:
        runs: List of run-like objects (real LangSmith Runs or test stubs).
            Each must have a ``.metadata`` dict.
        time_window_hours: The query window (for the response).
        truncated: True if the query hit its limit.

    Returns:
        ``MetricsResponse`` with all metrics.
    """
    total = len(runs)
    if total == 0:
        return MetricsResponse(
            time_window_hours=time_window_hours,
            source="langsmith",
            truncated=truncated,
        )

    # --- Operations ---
    images_processed = total
    boxes_processed = sum(_meta_int(r.metadata, "found_count") for r in runs)

    final_complete = sum(
        1 for r in runs if r.metadata.get("final_status") == "complete"
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

    # --- Quality ---
    scanner_vision_match = sum(
        1 for r in runs if _meta_bool(r.metadata, "scanner_vision_match")
    )
    count_deltas = [_meta_int(r.metadata, "count_delta") for r in runs]
    missing_counts = [_meta_int(r.metadata, "missing_count") for r in runs]
    unassigned_counts = [_meta_int(r.metadata, "unassigned_count") for r in runs]
    recovered_complete = sum(
        1 for r in runs
        if r.metadata.get("final_status") == "complete"
        and _meta_bool(r.metadata, "recovery_attempted")
    )
    still_incomplete = sum(
        1 for r in runs
        if r.metadata.get("final_status") == "needs_user_input"
    )

    # --- Recovery detail ---
    labels_tried = [
        _meta_int(r.metadata, "recovery_labels_tried")
        for r in runs if _meta_bool(r.metadata, "recovery_attempted")
    ]
    labels_resolved = [
        _meta_int(r.metadata, "recovery_labels_resolved")
        for r in runs if _meta_bool(r.metadata, "recovery_attempted")
    ]

    # --- Issues ---
    primary_issue_counts = dict(
        Counter(r.metadata.get("primary_issue", "none") for r in runs)
    )

    # --- Version breakdowns (true rates) ---
    completion_by_pipeline = _version_rates(
        runs, "pipeline_version",
        lambda r: r.metadata.get("final_status") == "complete",
    )
    mismatch_by_scanner = _version_rates(
        runs, "scanner_version",
        lambda r: not _meta_bool(r.metadata, "scanner_vision_match"),
    )
    recovery_success_by_recovery = _version_rates(
        [r for r in runs if _meta_bool(r.metadata, "recovery_attempted")],
        "recovery_version",
        lambda r: _meta_bool(r.metadata, "recovery_succeeded"),
    )
    retry_by_vision_model = _version_rates(
        runs, "vision_model",
        lambda r: r.metadata.get("final_status") == "needs_user_input",
    )

    return MetricsResponse(
        time_window_hours=time_window_hours,
        source="langsmith",
        truncated=truncated,
        # Operations
        images_processed=images_processed,
        boxes_processed=boxes_processed,
        first_pass_complete_pct=_pct(first_pass_complete, total),
        final_complete_pct=_pct(final_complete, total),
        recovery_attempted_pct=_pct(recovery_attempted, total),
        recovery_success_pct=_pct(recovery_succeeded, recovery_attempted),
        user_retry_required_pct=_pct(user_retry, total),
        p95_latency_ms=_p95(latencies),
        # Quality
        scanner_vision_match_pct=_pct(scanner_vision_match, total),
        avg_count_delta=_avg(count_deltas),
        avg_missing_count=_avg(missing_counts),
        avg_unassigned_count=_avg(unassigned_counts),
        recovered_complete_pct=_pct(recovered_complete, total),
        still_incomplete_pct=_pct(still_incomplete, total),
        # Recovery detail
        avg_labels_tried=_avg(labels_tried),
        avg_labels_resolved=_avg(labels_resolved),
        # Issues
        primary_issue_counts=primary_issue_counts,
        # Version breakdowns
        completion_by_pipeline_version=completion_by_pipeline,
        mismatch_by_scanner_version=mismatch_by_scanner,
        recovery_success_by_recovery_version=recovery_success_by_recovery,
        retry_by_vision_model=retry_by_vision_model,
    )


def unavailable_response(time_window_hours: int = 24) -> MetricsResponse:
    """Return a zeroed response with source='unavailable'."""
    return MetricsResponse(
        time_window_hours=time_window_hours,
        source="unavailable",
    )


# ---------------------------------------------------------------------------
# Grouped metrics — partition runs by a metadata field
# ---------------------------------------------------------------------------


VALID_GROUP_BY = frozenset({
    "pipeline_version",
    "scanner_version",
    "recovery_version",
    "vision_prompt_version",
    "vision_model",
    "source",
})


class GroupedMetricsResponse(BaseModel):
    """Metrics partitioned by a metadata field (e.g. pipeline_version)."""

    time_window_hours: int
    source: str = "langsmith"
    truncated: bool = False
    group_by: str
    groups: dict[str, MetricsResponse] = {}


def compute_grouped_metrics(
    runs: list[RunLike],
    *,
    group_by: str,
    time_window_hours: int = 24,
    truncated: bool = False,
) -> GroupedMetricsResponse:
    """Partition runs by a metadata field and compute metrics per group.

    Args:
        runs: List of run-like objects.
        group_by: Metadata key to partition by (e.g. "pipeline_version").
        time_window_hours: The query window.
        truncated: Whether the query hit its limit.

    Returns:
        ``GroupedMetricsResponse`` with one ``MetricsResponse`` per group.
    """
    buckets: dict[str, list[RunLike]] = {}
    for r in runs:
        key = str(r.metadata.get(group_by, "unknown"))
        buckets.setdefault(key, []).append(r)

    groups = {}
    for key in sorted(buckets):
        groups[key] = compute_metrics(
            buckets[key],
            time_window_hours=time_window_hours,
            truncated=truncated,
        )

    return GroupedMetricsResponse(
        time_window_hours=time_window_hours,
        source="langsmith",
        truncated=truncated,
        group_by=group_by,
        groups=groups,
    )


def unavailable_grouped_response(
    *,
    group_by: str,
    time_window_hours: int = 24,
) -> GroupedMetricsResponse:
    """Return a zeroed grouped response with source='unavailable'."""
    return GroupedMetricsResponse(
        time_window_hours=time_window_hours,
        source="unavailable",
        group_by=group_by,
    )


def _version_rates(
    runs: list[RunLike],
    version_key: str,
    predicate: Callable[[RunLike], bool],
) -> list[VersionRate]:
    """Compute true rate (percentage) per version bucket.

    Args:
        runs: The runs to bucket.
        version_key: Metadata key to group by (e.g. "pipeline_version").
        predicate: Function returning True for runs that match the metric
            (e.g. final_status == "complete").

    Returns:
        List of ``VersionRate`` sorted by version, each with total count
        and rate_pct = matched / total * 100.
    """
    buckets: dict[str, list[bool]] = {}
    for r in runs:
        version = str(r.metadata.get(version_key, "unknown"))
        buckets.setdefault(version, []).append(predicate(r))

    result = []
    for version in sorted(buckets):
        flags = buckets[version]
        total = len(flags)
        matched = sum(1 for f in flags if f)
        result.append(VersionRate(
            version=version,
            total=total,
            rate_pct=_pct(matched, total),
        ))
    return result


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


def _meta_str(meta: dict[str, Any], key: str) -> str | None:
    """Extract a string from metadata."""
    val = meta.get(key)
    return str(val) if val is not None else None

"""Provision the barcode scanner's LangSmith monitoring dashboards.

Four dashboards, each answering a distinct question:
    1. Operations   — Is the system healthy?
    2. Quality      — Where are we losing boxes?
    3. Recovery     — Is recovery worth its complexity?
    4. Versions     — Did a new version actually improve things?

Uses LangSmith's dashboard REST API directly because the installed
LangSmith Python SDK does not expose custom dashboard helpers.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT_TRACE_FILTER = (
    'or('
    'eq(name, "ingest_one"), '
    'eq(name, "web_analyze_barcode"), '
    'eq(name, "process_whatsapp_message")'
    ')'
)


# ---------------------------------------------------------------------------
# Dashboard specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DashboardSpec:
    key: str
    title: str
    description: str
    build_fn: Callable[[str, str], list[dict[str, Any]]]


# ---------------------------------------------------------------------------
# API client + config
# ---------------------------------------------------------------------------


class DashboardApiError(RuntimeError):
    """Raised when the LangSmith dashboard API returns an error."""


@dataclass(frozen=True)
class DashboardConfig:
    api_key: str
    endpoint: str
    project_id: str
    tenant_id: str | None = None

    @classmethod
    def from_env(cls) -> DashboardConfig:
        api_key = os.getenv("LANGSMITH_API_KEY", "").strip()
        project_id = os.getenv("LANGSMITH_PROJECT_ID", "").strip()
        if not api_key:
            raise ValueError("LANGSMITH_API_KEY is required")
        if not project_id:
            raise ValueError(
                "LANGSMITH_PROJECT_ID is required; dashboard charts need the tracing project UUID"
            )

        return cls(
            api_key=api_key,
            endpoint=os.getenv(
                "LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"
            ).strip().rstrip("/"),
            project_id=project_id,
            tenant_id=os.getenv("LANGSMITH_TENANT_ID", "").strip() or None,
        )


class LangSmithDashboardApi:
    """Small REST client for the LangSmith custom dashboard endpoints."""

    def __init__(self, config: DashboardConfig) -> None:
        self.config = config

    def request(
        self,
        method: str,
        path: str,
        *,
        query: dict[str, str] | None = None,
        body: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self.config.endpoint}/api/v1{path}"
        if query:
            url = f"{url}?{urlencode(query)}"

        headers = {
            "Accept": "application/json",
            "X-API-Key": self.config.api_key,
        }
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode("utf-8")
        if self.config.tenant_id:
            headers["X-Tenant-Id"] = self.config.tenant_id

        request = Request(url, data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=30) as response:
                payload = response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise DashboardApiError(
                f"LangSmith dashboard API {method} {path} failed "
                f"with HTTP {exc.code}: {detail}"
            ) from exc
        except URLError as exc:
            raise DashboardApiError(
                f"Could not reach LangSmith dashboard API: {exc.reason}"
            ) from exc

        if not payload:
            return None
        return json.loads(payload.decode("utf-8"))

    def list_sections(self, title: str) -> list[dict[str, Any]]:
        result = self.request(
            "GET",
            "/charts/section",
            query={"limit": "100", "title_contains": title},
        )
        return result if isinstance(result, list) else []

    def create_section(self, title: str, description: str) -> dict[str, Any]:
        result = self.request(
            "POST",
            "/charts/section",
            body={"title": title, "description": description},
        )
        if not isinstance(result, dict) or not result.get("id"):
            raise DashboardApiError("LangSmith did not return a section ID")
        return result

    def read_charts(self) -> list[dict[str, Any]]:
        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(days=7)
        result = self.request(
            "POST",
            "/charts",
            body={
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "stride": {"days": 0, "hours": 1, "minutes": 0},
                "omit_data": True,
            },
        )
        sections = result.get("sections", []) if isinstance(result, dict) else []
        charts: list[dict[str, Any]] = []
        for section in sections:
            charts.extend(section.get("charts", []))
        return charts

    def create_chart(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = self.request("POST", "/charts/create", body=payload)
        if not isinstance(result, dict) or not result.get("id"):
            raise DashboardApiError("LangSmith did not return a chart ID")
        return result

    def update_chart(
        self,
        chart_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        result = self.request("PATCH", f"/charts/{chart_id}", body=payload)
        if not isinstance(result, dict) or not result.get("id"):
            raise DashboardApiError("LangSmith did not return an updated chart ID")
        return result


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _project_filter(
    project_id: str,
    *,
    run_filter: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "source_type": "tracing_project",
        "project_ids": [project_id],
    }
    if run_filter:
        result["run_filter"] = run_filter
    return result


def _chart_metadata(dashboard_key: str, chart_key: str) -> dict[str, str]:
    return {
        "dashboard_key": dashboard_key,
        "chart_key": chart_key,
    }


def _count_series(
    name: str,
    project_id: str,
    dashboard_key: str,
    *,
    filter_expression: str | None = ROOT_TRACE_FILTER,
    group_by: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    metric_definition: dict[str, Any] = {"type": "count"}
    if filter_expression:
        metric_definition["filter"] = filter_expression

    series: dict[str, Any] = {
        "name": name,
        "metric_definition": metric_definition,
        "filter_definition": _project_filter(project_id),
    }
    if group_by:
        series["group_by_definitions"] = group_by
    return series


def _metadata_group(path: str) -> list[dict[str, str]]:
    return [{"attribute": "metadata", "path": path}]


# ---------------------------------------------------------------------------
# Dashboard 1: Operations
# ---------------------------------------------------------------------------


OPS_KEY = "barcode-scanner-operations"
OPS_TITLE = "Barcode Scanner — Operations"
OPS_DESC = "Is the system healthy? Volume, outcomes, latency, retries."


def build_operations_charts(project_id: str, section_id: str) -> list[dict[str, Any]]:
    """Operations dashboard — system health KPIs and trends."""
    g = _metadata_group

    return [
        {
            "title": "Upload volume by source",
            "description": "Root ingest_one traces grouped by source.",
            "index": 0,
            "chart_type": "line",
            "section_id": section_id,
            "metadata": _chart_metadata(OPS_KEY, "ops-volume"),
            "series": [
                _count_series("Traces", project_id, OPS_KEY, group_by=g("source")),
            ],
        },
        {
            "title": "Outcome distribution",
            "description": "Complete, needs_user_input, needs_retry, failed.",
            "index": 1,
            "chart_type": "bar",
            "section_id": section_id,
            "metadata": _chart_metadata(OPS_KEY, "ops-outcomes"),
            "series": [
                _count_series("Outcomes", project_id, OPS_KEY, group_by=g("final_status")),
            ],
        },
        {
            "title": "User retries by source",
            "description": "Traces where final_status=needs_user_input.",
            "index": 2,
            "chart_type": "line",
            "section_id": section_id,
            "metadata": _chart_metadata(OPS_KEY, "ops-retry-rate"),
            "series": [
                _count_series(
                    "Retries",
                    project_id,
                    OPS_KEY,
                    filter_expression=(
                        f'and({ROOT_TRACE_FILTER}, '
                        'has(metadata, "final_status=needs_user_input"))'
                    ),
                    group_by=g("source"),
                ),
            ],
        },
        {
            "title": "Completed analyses",
            "description": "Count of traces with final_status=complete.",
            "index": 3,
            "chart_type": "kpi",
            "section_id": section_id,
            "metadata": _chart_metadata(OPS_KEY, "ops-completion-kpi"),
            "series": [
                _count_series(
                    "Completed",
                    project_id,
                    OPS_KEY,
                    filter_expression=(
                        f'and({ROOT_TRACE_FILTER}, '
                        'has(metadata, "final_status=complete"))'
                    ),
                ),
            ],
        },
        {
            "title": "P50 latency by source",
            "description": "Median latency for root ingest_one traces.",
            "index": 4,
            "chart_type": "line",
            "section_id": section_id,
            "metadata": _chart_metadata(OPS_KEY, "ops-latency-p50"),
            "series": [
                {
                    "name": "P50 latency",
                    "metric": "latency_p50",
                    "filter_definition": _project_filter(
                        project_id, run_filter=ROOT_TRACE_FILTER,
                    ),
                    "group_by_definitions": g("source"),
                },
            ],
        },
        {
            "title": "P99 latency by source",
            "description": "Tail latency for root ingest_one traces.",
            "index": 5,
            "chart_type": "line",
            "section_id": section_id,
            "metadata": _chart_metadata(OPS_KEY, "ops-latency-p99"),
            "series": [
                {
                    "name": "P99 latency",
                    "metric": "latency_p99",
                    "filter_definition": _project_filter(
                        project_id, run_filter=ROOT_TRACE_FILTER,
                    ),
                    "group_by_definitions": g("source"),
                },
            ],
        },
        {
            "title": "Error rate by source",
            "description": "Fraction of runs with errors.",
            "index": 6,
            "chart_type": "line",
            "section_id": section_id,
            "metadata": _chart_metadata(OPS_KEY, "ops-error-rate"),
            "series": [
                {
                    "name": "Error rate",
                    "metric": "error_rate",
                    "filter_definition": _project_filter(
                        project_id, run_filter=ROOT_TRACE_FILTER,
                    ),
                    "group_by_definitions": g("source"),
                },
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Dashboard 2: Quality
# ---------------------------------------------------------------------------


QUALITY_KEY = "barcode-scanner-quality"
QUALITY_TITLE = "Barcode Scanner — Quality"
QUALITY_DESC = (
    "Where are we losing boxes? Scanner/vision mismatch, missing labels, "
    "recovery contribution, failure taxonomy."
)


def build_quality_charts(project_id: str, section_id: str) -> list[dict[str, Any]]:
    """Quality dashboard — where are we losing boxes?"""
    g = _metadata_group

    return [
        {
            "title": "Scanner vs vision agreement",
            "description": "Count of traces where scanner_count == vision_count vs not.",
            "index": 0,
            "chart_type": "bar",
            "section_id": section_id,
            "metadata": _chart_metadata(QUALITY_KEY, "quality-scanner-vision-match"),
            "series": [
                _count_series(
                    "Match",
                    project_id,
                    QUALITY_KEY,
                    filter_expression=(
                        f'and({ROOT_TRACE_FILTER}, '
                        'has(metadata, "scanner_vision_match=True"))'
                    ),
                ),
                _count_series(
                    "Mismatch",
                    project_id,
                    QUALITY_KEY,
                    filter_expression=(
                        f'and({ROOT_TRACE_FILTER}, '
                        'has(metadata, "scanner_vision_match=False"))'
                    ),
                ),
            ],
        },
        {
            "title": "Count delta distribution",
            "description": "vision_count - scanner_count distribution.",
            "index": 1,
            "chart_type": "bar",
            "section_id": section_id,
            "metadata": _chart_metadata(QUALITY_KEY, "quality-count-delta"),
            "series": [
                _count_series(
                    "Delta",
                    project_id,
                    QUALITY_KEY,
                    group_by=g("count_delta"),
                ),
            ],
        },
        {
            "title": "Missing label distribution",
            "description": "How many labels are typically missing?",
            "index": 2,
            "chart_type": "bar",
            "section_id": section_id,
            "metadata": _chart_metadata(QUALITY_KEY, "quality-missing-labels"),
            "series": [
                _count_series(
                    "Traces",
                    project_id,
                    QUALITY_KEY,
                    group_by=g("missing_count"),
                ),
            ],
        },
        {
            "title": "Unassigned detections distribution",
            "description": "Scanner detections not matched to any Gemini label.",
            "index": 3,
            "chart_type": "bar",
            "section_id": section_id,
            "metadata": _chart_metadata(QUALITY_KEY, "quality-unassigned"),
            "series": [
                _count_series(
                    "Traces",
                    project_id,
                    QUALITY_KEY,
                    group_by=g("unassigned_count"),
                ),
            ],
        },
        {
            "title": "First-pass complete (no recovery)",
            "description": "Completed without recovery being attempted.",
            "index": 4,
            "chart_type": "kpi",
            "section_id": section_id,
            "metadata": _chart_metadata(QUALITY_KEY, "quality-first-pass"),
            "series": [
                _count_series(
                    "First-pass",
                    project_id,
                    QUALITY_KEY,
                    filter_expression=(
                        f'and({ROOT_TRACE_FILTER}, '
                        'has(metadata, "final_status=complete"), '
                        'has(metadata, "recovery_attempted=False"))'
                    ),
                ),
            ],
        },
        {
            "title": "Completed after recovery",
            "description": "Completed with recovery attempted.",
            "index": 5,
            "chart_type": "kpi",
            "section_id": section_id,
            "metadata": _chart_metadata(QUALITY_KEY, "quality-recovered"),
            "series": [
                _count_series(
                    "Recovered",
                    project_id,
                    QUALITY_KEY,
                    filter_expression=(
                        f'and({ROOT_TRACE_FILTER}, '
                        'has(metadata, "final_status=complete"), '
                        'has(metadata, "recovery_attempted=True"))'
                    ),
                ),
            ],
        },
        {
            "title": "Still incomplete after recovery",
            "description": "needs_user_input after recovery was attempted.",
            "index": 6,
            "chart_type": "kpi",
            "section_id": section_id,
            "metadata": _chart_metadata(QUALITY_KEY, "quality-still-incomplete"),
            "series": [
                _count_series(
                    "Incomplete",
                    project_id,
                    QUALITY_KEY,
                    filter_expression=(
                        f'and({ROOT_TRACE_FILTER}, '
                        'has(metadata, "final_status=needs_user_input"))'
                    ),
                ),
            ],
        },
        {
            "title": "Failure taxonomy",
            "description": "Primary issue code distribution.",
            "index": 7,
            "chart_type": "bar",
            "section_id": section_id,
            "metadata": _chart_metadata(QUALITY_KEY, "quality-primary-issue"),
            "series": [
                _count_series(
                    "Traces",
                    project_id,
                    QUALITY_KEY,
                    group_by=g("primary_issue"),
                ),
            ],
        },
        {
            "title": "User-confirmed correctness",
            "description": "Average user_correct feedback score by source.",
            "index": 8,
            "chart_type": "line",
            "section_id": section_id,
            "metadata": _chart_metadata(QUALITY_KEY, "quality-user-confirmed"),
            "series": [
                {
                    "name": "user_correct",
                    "feedback_key": "user_correct",
                    "metric_definition": {
                        "type": "avg",
                        "field": "feedback_score",
                        "params": {"feedback_key": "user_correct"},
                        "filter": ROOT_TRACE_FILTER,
                    },
                    "filter_definition": _project_filter(project_id),
                    "group_by_definitions": g("source"),
                },
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Dashboard 3: Recovery
# ---------------------------------------------------------------------------


RECOVERY_KEY = "barcode-scanner-recovery"
RECOVERY_TITLE = "Barcode Scanner — Recovery"
RECOVERY_DESC = (
    "Is recovery worth its complexity? Funnel, success rate, efficiency."
)


def build_recovery_charts(project_id: str, section_id: str) -> list[dict[str, Any]]:
    """Recovery dashboard — is recovery worth its complexity?"""
    g = _metadata_group

    return [
        {
            "title": "Total runs (recovery funnel step 1)",
            "description": "All ingest_one traces.",
            "index": 0,
            "chart_type": "kpi",
            "section_id": section_id,
            "metadata": _chart_metadata(RECOVERY_KEY, "recovery-funnel-total"),
            "series": [
                _count_series("Total", project_id, RECOVERY_KEY),
            ],
        },
        {
            "title": "Recovery attempted (funnel step 2)",
            "description": "Traces where recovery_attempted=True.",
            "index": 1,
            "chart_type": "kpi",
            "section_id": section_id,
            "metadata": _chart_metadata(RECOVERY_KEY, "recovery-funnel-attempted"),
            "series": [
                _count_series(
                    "Attempted",
                    project_id,
                    RECOVERY_KEY,
                    filter_expression=(
                        f'and({ROOT_TRACE_FILTER}, '
                        'has(metadata, "recovery_attempted=True"))'
                    ),
                ),
            ],
        },
        {
            "title": "Recovery succeeded (funnel step 3)",
            "description": "Recovery attempted and ≥1 label resolved.",
            "index": 2,
            "chart_type": "kpi",
            "section_id": section_id,
            "metadata": _chart_metadata(RECOVERY_KEY, "recovery-funnel-succeeded"),
            "series": [
                _count_series(
                    "Succeeded",
                    project_id,
                    RECOVERY_KEY,
                    filter_expression=(
                        f'and({ROOT_TRACE_FILTER}, '
                        'has(metadata, "recovery_succeeded=True"))'
                    ),
                ),
            ],
        },
        {
            "title": "Became complete after recovery (funnel step 4)",
            "description": "Recovery attempted and final_status=complete.",
            "index": 3,
            "chart_type": "kpi",
            "section_id": section_id,
            "metadata": _chart_metadata(RECOVERY_KEY, "recovery-funnel-complete"),
            "series": [
                _count_series(
                    "Complete",
                    project_id,
                    RECOVERY_KEY,
                    filter_expression=(
                        f'and({ROOT_TRACE_FILTER}, '
                        'has(metadata, "recovery_attempted=True"), '
                        'has(metadata, "final_status=complete"))'
                    ),
                ),
            ],
        },
        {
            "title": "Recovery attempt rate by source",
            "description": "How often recovery is needed.",
            "index": 4,
            "chart_type": "line",
            "section_id": section_id,
            "metadata": _chart_metadata(RECOVERY_KEY, "recovery-attempt-rate"),
            "series": [
                _count_series(
                    "Attempts",
                    project_id,
                    RECOVERY_KEY,
                    filter_expression=(
                        f'and({ROOT_TRACE_FILTER}, '
                        'has(metadata, "recovery_attempted=True"))'
                    ),
                    group_by=g("source"),
                ),
            ],
        },
        {
            "title": "Labels tried vs resolved",
            "description": "Distribution of recovery_labels_tried and recovery_labels_resolved.",
            "index": 5,
            "chart_type": "bar",
            "section_id": section_id,
            "metadata": _chart_metadata(RECOVERY_KEY, "recovery-labels"),
            "series": [
                _count_series(
                    "Tried",
                    project_id,
                    RECOVERY_KEY,
                    filter_expression=(
                        f'and({ROOT_TRACE_FILTER}, '
                        'has(metadata, "recovery_attempted=True"))'
                    ),
                    group_by=g("recovery_labels_tried"),
                ),
                _count_series(
                    "Resolved",
                    project_id,
                    RECOVERY_KEY,
                    filter_expression=(
                        f'and({ROOT_TRACE_FILTER}, '
                        'has(metadata, "recovery_attempted=True"))'
                    ),
                    group_by=g("recovery_labels_resolved"),
                ),
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Dashboard 4: Versions
# ---------------------------------------------------------------------------


VERSIONS_KEY = "barcode-scanner-versions"
VERSIONS_TITLE = "Barcode Scanner — Versions"
VERSIONS_DESC = (
    "A/B comparison by pipeline/scanner/vision/recovery version. "
    "Charts show counts (not rates) — LangSmith custom charts cannot "
    "compute count/total ratios."
)


def build_versions_charts(project_id: str, section_id: str) -> list[dict[str, Any]]:
    """Versions dashboard — did a new version actually improve things?"""
    g = _metadata_group

    return [
        {
            "title": "Completed analyses (count) by pipeline version",
            "description": "Count of complete traces grouped by pipeline_version.",
            "index": 0,
            "chart_type": "bar",
            "section_id": section_id,
            "metadata": _chart_metadata(VERSIONS_KEY, "versions-completion-by-pipeline"),
            "series": [
                _count_series(
                    "Completed",
                    project_id,
                    VERSIONS_KEY,
                    filter_expression=(
                        f'and({ROOT_TRACE_FILTER}, '
                        'has(metadata, "final_status=complete"))'
                    ),
                    group_by=g("pipeline_version"),
                ),
            ],
        },
        {
            "title": "Scanner mismatch (count) by scanner version",
            "description": "Count where scanner_vision_match=False, grouped by scanner_version.",
            "index": 1,
            "chart_type": "bar",
            "section_id": section_id,
            "metadata": _chart_metadata(VERSIONS_KEY, "versions-mismatch-by-scanner"),
            "series": [
                _count_series(
                    "Mismatch",
                    project_id,
                    VERSIONS_KEY,
                    filter_expression=(
                        f'and({ROOT_TRACE_FILTER}, '
                        'has(metadata, "scanner_vision_match=False"))'
                    ),
                    group_by=g("scanner_version"),
                ),
            ],
        },
        {
            "title": "Recovery successes (count) by recovery version",
            "description": "Count where recovery_succeeded=True, grouped by recovery_version.",
            "index": 2,
            "chart_type": "bar",
            "section_id": section_id,
            "metadata": _chart_metadata(VERSIONS_KEY, "versions-recovery-by-recovery"),
            "series": [
                _count_series(
                    "Succeeded",
                    project_id,
                    VERSIONS_KEY,
                    filter_expression=(
                        f'and({ROOT_TRACE_FILTER}, '
                        'has(metadata, "recovery_succeeded=True"))'
                    ),
                    group_by=g("recovery_version"),
                ),
            ],
        },
        {
            "title": "Completed analyses (count) by vision prompt version",
            "description": "Count of complete traces grouped by vision_prompt_version.",
            "index": 3,
            "chart_type": "bar",
            "section_id": section_id,
            "metadata": _chart_metadata(VERSIONS_KEY, "versions-completion-by-vision-prompt"),
            "series": [
                _count_series(
                    "Completed",
                    project_id,
                    VERSIONS_KEY,
                    filter_expression=(
                        f'and({ROOT_TRACE_FILTER}, '
                        'has(metadata, "final_status=complete"))'
                    ),
                    group_by=g("vision_prompt_version"),
                ),
            ],
        },
        {
            "title": "P99 latency by pipeline version",
            "description": "Tail latency grouped by pipeline_version.",
            "index": 4,
            "chart_type": "line",
            "section_id": section_id,
            "metadata": _chart_metadata(VERSIONS_KEY, "versions-latency-by-pipeline"),
            "series": [
                {
                    "name": "P99 latency",
                    "metric": "latency_p99",
                    "filter_definition": _project_filter(
                        project_id, run_filter=ROOT_TRACE_FILTER,
                    ),
                    "group_by_definitions": g("pipeline_version"),
                },
            ],
        },
        {
            "title": "User retries (count) by vision model",
            "description": "Count where final_status=needs_user_input, grouped by vision_model.",
            "index": 5,
            "chart_type": "bar",
            "section_id": section_id,
            "metadata": _chart_metadata(VERSIONS_KEY, "versions-retry-by-vision-model"),
            "series": [
                _count_series(
                    "Retries",
                    project_id,
                    VERSIONS_KEY,
                    filter_expression=(
                        f'and({ROOT_TRACE_FILTER}, '
                        'has(metadata, "final_status=needs_user_input"))'
                    ),
                    group_by=g("vision_model"),
                ),
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Dashboard registry
# ---------------------------------------------------------------------------


DASHBOARDS: list[DashboardSpec] = [
    DashboardSpec(
        key=OPS_KEY,
        title=OPS_TITLE,
        description=OPS_DESC,
        build_fn=build_operations_charts,
    ),
    DashboardSpec(
        key=QUALITY_KEY,
        title=QUALITY_TITLE,
        description=QUALITY_DESC,
        build_fn=build_quality_charts,
    ),
    DashboardSpec(
        key=RECOVERY_KEY,
        title=RECOVERY_TITLE,
        description=RECOVERY_DESC,
        build_fn=build_recovery_charts,
    ),
]


def get_dashboard_spec(key: str) -> DashboardSpec | None:
    return next((d for d in DASHBOARDS if d.key == key), None)


# ---------------------------------------------------------------------------
# Provisioning
# ---------------------------------------------------------------------------


def _find_section(
    sections: list[dict[str, Any]],
    title: str,
) -> dict[str, Any] | None:
    return next(
        (section for section in sections if section.get("title") == title),
        None,
    )


def _find_chart(
    charts: list[dict[str, Any]],
    dashboard_key: str,
    chart_key: str,
) -> dict[str, Any] | None:
    for chart in charts:
        metadata = chart.get("metadata") or {}
        if (
            metadata.get("dashboard_key") == dashboard_key
            and metadata.get("chart_key") == chart_key
        ):
            return chart
    return None


def provision_one_dashboard(
    api: LangSmithDashboardApi,
    spec: DashboardSpec,
    *,
    dry_run: bool = False,
    check_only: bool = False,
) -> dict[str, Any]:
    """Create or update a single dashboard idempotently."""
    sections = api.list_sections(spec.title)
    section = _find_section(sections, spec.title)

    if section is None:
        if check_only:
            raise DashboardApiError(f"Dashboard section '{spec.title}' is missing")
        if dry_run:
            return {
                "dashboard_key": spec.key,
                "section": "missing",
                "charts": "not checked",
            }
        section = api.create_section(spec.title, spec.description)

    section_id = str(section["id"])
    payloads = spec.build_fn(api.config.project_id, section_id)
    existing_charts = api.read_charts()
    results: list[dict[str, str]] = []

    for payload in payloads:
        chart_key = payload["metadata"]["chart_key"]
        existing = _find_chart(existing_charts, spec.key, chart_key)
        if existing is not None:
            results.append({"chart_key": chart_key, "id": str(existing["id"]), "action": "exists"})
            if not dry_run and not check_only:
                api.update_chart(str(existing["id"]), payload)
            continue

        results.append({"chart_key": chart_key, "id": "", "action": "missing"})
        if not dry_run and not check_only:
            created = api.create_chart(payload)
            results[-1]["id"] = str(created["id"])
            results[-1]["action"] = "created"

    missing = [item for item in results if item["action"] == "missing"]
    if check_only and missing:
        raise DashboardApiError(
            f"Dashboard '{spec.title}' is incomplete: "
            + ", ".join(item["chart_key"] for item in missing)
        )

    return {
        "dashboard_key": spec.key,
        "section_id": section_id,
        "charts": results,
    }


def provision_dashboard(
    api: LangSmithDashboardApi,
    *,
    dry_run: bool = False,
    check_only: bool = False,
    dashboard_key: str | None = None,
) -> dict[str, Any]:
    """Create or update one or all dashboards idempotently.

    Args:
        dashboard_key: If set, provision only that dashboard. If None, provision all.
    """
    if dashboard_key is not None:
        spec = get_dashboard_spec(dashboard_key)
        if spec is None:
            raise DashboardApiError(f"Unknown dashboard key: {dashboard_key}")
        return provision_one_dashboard(api, spec, dry_run=dry_run, check_only=check_only)

    results = []
    for spec in DASHBOARDS:
        results.append(
            provision_one_dashboard(api, spec, dry_run=dry_run, check_only=check_only)
        )
    return {"dashboards": results}

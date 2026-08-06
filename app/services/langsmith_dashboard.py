"""Provision the barcode scanner's LangSmith monitoring dashboard.

This module uses LangSmith's dashboard REST API directly because the installed
LangSmith Python SDK does not expose custom dashboard helpers.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DASHBOARD_KEY = "barcode-scanner-production-health"
SECTION_TITLE = "Barcode Scanner Production Health"
SECTION_DESCRIPTION = (
    "Operational health for barcode analysis traces across web and WhatsApp."
)

ROOT_TRACE_FILTER = (
    'or(eq(name, "web_analyze_barcode"), '
    'eq(name, "process_whatsapp_message"))'
)


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

    def create_section(self) -> dict[str, Any]:
        result = self.request(
            "POST",
            "/charts/section",
            body={
                "title": SECTION_TITLE,
                "description": SECTION_DESCRIPTION,
            },
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


def _chart_metadata(chart_key: str) -> dict[str, str]:
    return {
        "dashboard_key": DASHBOARD_KEY,
        "chart_key": chart_key,
    }


def _count_series(
    name: str,
    project_id: str,
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


def build_chart_payloads(project_id: str, section_id: str) -> list[dict[str, Any]]:
    """Build the initial dashboard charts without making network calls."""

    def metadata_group(path: str) -> list[dict[str, str]]:
        return [{"attribute": "metadata", "path": path}]

    return [
        {
            "title": "Upload volume by source",
            "description": "Root barcode analysis traces grouped by web or WhatsApp.",
            "index": 0,
            "chart_type": "line",
            "section_id": section_id,
            "metadata": _chart_metadata("upload-volume-by-source"),
            "series": [
                _count_series(
                    "Analysis traces",
                    project_id,
                    group_by=metadata_group("source"),
                )
            ],
        },
        {
            "title": "Outcome distribution",
            "description": "Complete, needs better photo, and retryable outcomes.",
            "index": 1,
            "chart_type": "bar",
            "section_id": section_id,
            "metadata": _chart_metadata("outcome-distribution"),
            "series": [
                _count_series(
                    "Analysis outcomes",
                    project_id,
                    group_by=metadata_group("outcome"),
                )
            ],
        },
        {
            "title": "Recovery attempts by source",
            "description": "Root traces where Gemini-guided barcode recovery was attempted.",
            "index": 2,
            "chart_type": "line",
            "section_id": section_id,
            "metadata": _chart_metadata("recovery-attempts"),
            "series": [
                _count_series(
                    "Recovery attempts",
                    project_id,
                    filter_expression=(
                        f"and({ROOT_TRACE_FILTER}, "
                        'has(metadata, "recovery_attempted=true"))'
                    ),
                    group_by=metadata_group("source"),
                )
            ],
        },
        {
            "title": "User-confirmed correctness",
            "description": "Average user_correct feedback score on root analysis traces.",
            "index": 3,
            "chart_type": "line",
            "section_id": section_id,
            "metadata": _chart_metadata("user-confirmed-correctness"),
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
                    "group_by_definitions": metadata_group("source"),
                }
            ],
        },
        {
            "title": "Completed analyses",
            "description": "Count of root analysis traces with a complete outcome.",
            "index": 4,
            "chart_type": "kpi",
            "section_id": section_id,
            "metadata": _chart_metadata("first-pass-completion-rate"),
            "series": [
                _count_series(
                    "Completed analyses",
                    project_id,
                    filter_expression=(
                        f"and({ROOT_TRACE_FILTER}, "
                        'has(metadata, "outcome=complete"))'
                    ),
                )
            ],
        },
        {
            "title": "P50 analysis latency",
            "description": "Median LangSmith latency for root barcode analysis traces.",
            "index": 5,
            "chart_type": "line",
            "section_id": section_id,
            "metadata": _chart_metadata("p50-analysis-latency"),
            "series": [
                {
                    "name": "P50 latency",
                    "metric": "latency_p50",
                    "filter_definition": _project_filter(
                        project_id,
                        run_filter=ROOT_TRACE_FILTER,
                    ),
                    "group_by_definitions": metadata_group("source"),
                }
            ],
        },
        {
            "title": "Recovery labels resolved",
            "description": "Distribution of labels resolved by the recovery pass.",
            "index": 6,
            "chart_type": "bar",
            "section_id": section_id,
            "metadata": _chart_metadata("recovery-labels-resolved"),
            "series": [
                _count_series(
                    "Recovery result",
                    project_id,
                    group_by=metadata_group("recovery_labels_resolved"),
                )
            ],
        },
    ]


def _find_section(
    sections: list[dict[str, Any]],
) -> dict[str, Any] | None:
    return next(
        (section for section in sections if section.get("title") == SECTION_TITLE),
        None,
    )


def _find_chart(
    charts: list[dict[str, Any]],
    chart_key: str,
) -> dict[str, Any] | None:
    for chart in charts:
        metadata = chart.get("metadata") or {}
        if (
            metadata.get("dashboard_key") == DASHBOARD_KEY
            and metadata.get("chart_key") == chart_key
        ):
            return chart
    return None


def provision_dashboard(
    api: LangSmithDashboardApi,
    *,
    dry_run: bool = False,
    check_only: bool = False,
) -> dict[str, Any]:
    """Create or update the dashboard idempotently and return its IDs."""
    sections = api.list_sections(SECTION_TITLE)
    section = _find_section(sections)

    if section is None:
        if check_only:
            raise DashboardApiError("Dashboard section is missing")
        if dry_run:
            return {
                "section": "missing",
                "charts": "not checked",
                "dashboard_key": DASHBOARD_KEY,
            }
        section = api.create_section()

    section_id = str(section["id"])
    payloads = build_chart_payloads(api.config.project_id, section_id)
    existing_charts = api.read_charts()
    results: list[dict[str, str]] = []

    for payload in payloads:
        chart_key = payload["metadata"]["chart_key"]
        existing = _find_chart(existing_charts, chart_key)
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
            "Dashboard is incomplete: "
            + ", ".join(item["chart_key"] for item in missing)
        )

    return {
        "dashboard_key": DASHBOARD_KEY,
        "section_id": section_id,
        "charts": results,
    }

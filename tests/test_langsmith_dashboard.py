from __future__ import annotations

import pytest

from app.services.langsmith_dashboard import (
    DASHBOARD_KEY,
    SECTION_TITLE,
    DashboardApiError,
    DashboardConfig,
    LangSmithDashboardApi,
    build_chart_payloads,
    provision_dashboard,
)


class FakeApi:
    def __init__(self, *, sections=None, charts=None):
        self.config = DashboardConfig(
            api_key="test-key",
            endpoint="https://example.test",
            project_id="project-id",
        )
        self.sections = sections or []
        self.charts = charts or []
        self.created_sections = 0
        self.created_charts = []
        self.updated_charts = []

    def list_sections(self, title):
        assert title == SECTION_TITLE
        return self.sections

    def create_section(self):
        self.created_sections += 1
        return {"id": "section-id", "title": SECTION_TITLE}

    def read_charts(self):
        return self.charts

    def create_chart(self, payload):
        self.created_charts.append(payload)
        return {"id": f"chart-{len(self.created_charts)}"}

    def update_chart(self, chart_id, payload):
        self.updated_charts.append((chart_id, payload))
        return {"id": chart_id}


def test_build_chart_payloads_uses_root_analysis_filter_and_project_scope():
    payloads = build_chart_payloads("project-id", "section-id")

    assert len(payloads) == 7
    assert {p["metadata"]["dashboard_key"] for p in payloads} == {DASHBOARD_KEY}
    assert [p["metadata"]["chart_key"] for p in payloads] == [
        "upload-volume-by-source",
        "outcome-distribution",
        "recovery-attempts",
        "user-confirmed-correctness",
        "first-pass-completion-rate",
        "p50-analysis-latency",
        "recovery-labels-resolved",
    ]
    assert all(p["section_id"] == "section-id" for p in payloads)

    upload_series = payloads[0]["series"][0]
    assert upload_series["metric_definition"]["type"] == "count"
    assert upload_series["filter_definition"] == {
        "source_type": "tracing_project",
        "project_ids": ["project-id"],
    }
    assert upload_series["group_by_definitions"] == [
        {"attribute": "metadata", "path": "source"}
    ]
    assert "web_analyze_barcode" in upload_series["metric_definition"]["filter"]
    assert "process_whatsapp_message" in upload_series["metric_definition"]["filter"]

    completion = next(
        p for p in payloads if p["metadata"]["chart_key"] == "first-pass-completion-rate"
    )
    completion_series = completion["series"][0]
    assert completion_series["metric_definition"]["type"] == "count"
    assert "outcome=complete" in completion_series["metric_definition"]["filter"]


def test_provision_creates_section_and_charts():
    api = FakeApi()

    result = provision_dashboard(api)

    assert api.created_sections == 1
    assert len(api.created_charts) == 7
    assert result["section_id"] == "section-id"
    assert all(chart["action"] == "created" for chart in result["charts"])


def test_provision_is_idempotent_and_updates_existing_charts():
    charts = [
        {
            "id": "existing-chart",
            "metadata": {
                "dashboard_key": DASHBOARD_KEY,
                "chart_key": "upload-volume-by-source",
            },
        }
    ]
    api = FakeApi(
        sections=[{"id": "existing-section", "title": SECTION_TITLE}],
        charts=charts,
    )

    result = provision_dashboard(api)

    assert api.created_sections == 0
    assert len(api.created_charts) == 6
    assert len(api.updated_charts) == 1
    assert result["section_id"] == "existing-section"
    existing = next(
        chart for chart in result["charts"] if chart["chart_key"] == "upload-volume-by-source"
    )
    assert existing == {
        "chart_key": "upload-volume-by-source",
        "id": "existing-chart",
        "action": "exists",
    }


def test_check_reports_missing_dashboard_resources():
    api = FakeApi()

    with pytest.raises(DashboardApiError, match="Dashboard section is missing"):
        provision_dashboard(api, check_only=True)

    assert api.created_sections == 0
    assert api.created_charts == []


def test_read_charts_sends_required_time_window(monkeypatch):
    api = LangSmithDashboardApi(
        DashboardConfig(
            api_key="test-key",
            endpoint="https://example.test",
            project_id="project-id",
        )
    )
    captured = {}

    def fake_request(method, path, *, query=None, body=None):
        captured.update(method=method, path=path, body=body)
        return {"sections": []}

    monkeypatch.setattr(api, "request", fake_request)
    assert api.read_charts() == []
    assert captured["method"] == "POST"
    assert captured["path"] == "/charts"
    assert captured["body"]["omit_data"] is True
    assert captured["body"]["start_time"]
    assert captured["body"]["end_time"]
    assert captured["body"]["stride"] == {"days": 0, "hours": 1, "minutes": 0}


def test_dashboard_config_requires_api_key_and_project_id(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_PROJECT_ID", raising=False)

    with pytest.raises(ValueError, match="LANGSMITH_API_KEY"):
        DashboardConfig.from_env()

    monkeypatch.setenv("LANGSMITH_API_KEY", "secret")
    with pytest.raises(ValueError, match="LANGSMITH_PROJECT_ID"):
        DashboardConfig.from_env()

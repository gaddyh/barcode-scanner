from __future__ import annotations

import pytest

from src.observability.dashboard import (
    DASHBOARDS,
    OPS_KEY,
    QUALITY_KEY,
    RECOVERY_KEY,
    DashboardApiError,
    DashboardConfig,
    LangSmithDashboardApi,
    build_operations_charts,
    build_quality_charts,
    build_recovery_charts,
    get_dashboard_spec,
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
        self.created_sections = []
        self.created_charts = []
        self.updated_charts = []

    def list_sections(self, title):
        return [s for s in self.sections if title in s.get("title", "")]

    def create_section(self, title, description):
        self.created_sections.append(title)
        return {"id": f"section-{len(self.created_sections)}", "title": title}

    def read_charts(self):
        return self.charts

    def create_chart(self, payload):
        self.created_charts.append(payload)
        return {"id": f"chart-{len(self.created_charts)}"}

    def update_chart(self, chart_id, payload):
        self.updated_charts.append((chart_id, payload))
        return {"id": chart_id}


# ---------------------------------------------------------------------------
# Dashboard registry
# ---------------------------------------------------------------------------


def test_three_dashboards_registered():
    assert len(DASHBOARDS) == 3
    keys = {d.key for d in DASHBOARDS}
    assert keys == {OPS_KEY, QUALITY_KEY, RECOVERY_KEY}


def test_get_dashboard_spec():
    spec = get_dashboard_spec(OPS_KEY)
    assert spec is not None
    assert spec.title == "Barcode Scanner — Operations"

    assert get_dashboard_spec("nonexistent") is None


# ---------------------------------------------------------------------------
# Operations dashboard
# ---------------------------------------------------------------------------


def test_build_operations_charts():
    payloads = build_operations_charts("project-id", "section-id")

    assert len(payloads) == 7
    assert [p["metadata"]["chart_key"] for p in payloads] == [
        "ops-volume",
        "ops-outcomes",
        "ops-retry-rate",
        "ops-completion-kpi",
        "ops-latency-p50",
        "ops-latency-p99",
        "ops-error-rate",
    ]
    assert all(p["section_id"] == "section-id" for p in payloads)
    assert all(p["metadata"]["dashboard_key"] == OPS_KEY for p in payloads)

    # Volume chart groups by source
    vol = payloads[0]
    assert vol["series"][0]["group_by_definitions"] == [
        {"attribute": "metadata", "path": "source"}
    ]
    assert "ingest_one" in vol["series"][0]["metric_definition"]["filter"]

    # Outcomes group by final_status
    outcomes = payloads[1]
    assert outcomes["series"][0]["group_by_definitions"] == [
        {"attribute": "metadata", "path": "final_status"}
    ]

    # P50 and P99 latency
    assert payloads[4]["series"][0]["metric"] == "latency_p50"
    assert payloads[5]["series"][0]["metric"] == "latency_p99"

    # Error rate
    assert payloads[6]["series"][0]["metric"] == "error_rate"


# ---------------------------------------------------------------------------
# Quality dashboard
# ---------------------------------------------------------------------------


def test_build_quality_charts():
    payloads = build_quality_charts("project-id", "section-id")

    assert len(payloads) == 9
    assert [p["metadata"]["chart_key"] for p in payloads] == [
        "quality-scanner-vision-match",
        "quality-count-delta",
        "quality-missing-labels",
        "quality-unassigned",
        "quality-first-pass",
        "quality-recovered",
        "quality-still-incomplete",
        "quality-primary-issue",
        "quality-user-confirmed",
    ]
    assert all(p["metadata"]["dashboard_key"] == QUALITY_KEY for p in payloads)

    # Scanner/vision match has two series (match + mismatch)
    match_chart = payloads[0]
    assert len(match_chart["series"]) == 2
    assert "scanner_vision_match=True" in match_chart["series"][0]["metric_definition"]["filter"]
    assert "scanner_vision_match=False" in match_chart["series"][1]["metric_definition"]["filter"]

    # Count delta groups by count_delta
    delta = payloads[1]
    assert delta["series"][0]["group_by_definitions"] == [
        {"attribute": "metadata", "path": "count_delta"}
    ]

    # First-pass complete: final_status=complete AND recovery_attempted=False
    first_pass = payloads[4]
    fp_filter = first_pass["series"][0]["metric_definition"]["filter"]
    assert "final_status=complete" in fp_filter
    assert "recovery_attempted=False" in fp_filter

    # Primary issue groups by primary_issue
    issue = payloads[7]
    assert issue["series"][0]["group_by_definitions"] == [
        {"attribute": "metadata", "path": "primary_issue"}
    ]

    # User-confirmed correctness uses feedback avg
    uc = payloads[8]
    assert uc["series"][0]["metric_definition"]["type"] == "avg"
    assert uc["series"][0]["feedback_key"] == "user_correct"


# ---------------------------------------------------------------------------
# Recovery dashboard
# ---------------------------------------------------------------------------


def test_build_recovery_charts():
    payloads = build_recovery_charts("project-id", "section-id")

    assert len(payloads) == 6
    assert [p["metadata"]["chart_key"] for p in payloads] == [
        "recovery-funnel-total",
        "recovery-funnel-attempted",
        "recovery-funnel-succeeded",
        "recovery-funnel-complete",
        "recovery-attempt-rate",
        "recovery-labels",
    ]
    assert all(p["metadata"]["dashboard_key"] == RECOVERY_KEY for p in payloads)

    # Funnel: 4 KPIs
    assert payloads[0]["chart_type"] == "kpi"
    assert payloads[1]["chart_type"] == "kpi"
    assert payloads[2]["chart_type"] == "kpi"
    assert payloads[3]["chart_type"] == "kpi"

    # Succeeded filter uses recovery_succeeded=True
    succeeded = payloads[2]
    assert "recovery_succeeded=True" in succeeded["series"][0]["metric_definition"]["filter"]

    # Labels chart has two series (tried + resolved)
    labels = payloads[5]
    assert len(labels["series"]) == 2


# ---------------------------------------------------------------------------
# Provisioning
# ---------------------------------------------------------------------------


def test_provision_all_creates_three_dashboards():
    api = FakeApi()

    result = provision_dashboard(api)

    assert len(api.created_sections) == 3
    assert "dashboards" in result
    assert len(result["dashboards"]) == 3
    total_charts = sum(len(d["charts"]) for d in result["dashboards"])
    assert total_charts == 22  # 7 + 9 + 6
    assert all(c["action"] == "created" for d in result["dashboards"] for c in d["charts"])


def test_provision_single_dashboard():
    api = FakeApi()

    result = provision_dashboard(api, dashboard_key=OPS_KEY)

    assert len(api.created_sections) == 1
    assert result["dashboard_key"] == OPS_KEY
    assert len(result["charts"]) == 7


def test_provision_unknown_dashboard_key():
    api = FakeApi()

    with pytest.raises(DashboardApiError, match="Unknown dashboard key"):
        provision_dashboard(api, dashboard_key="nonexistent")


def test_provision_is_idempotent_and_updates_existing():
    charts = [
        {
            "id": "existing-chart",
            "metadata": {
                "dashboard_key": OPS_KEY,
                "chart_key": "ops-volume",
            },
        },
    ]
    api = FakeApi(
        sections=[{"id": "existing-section", "title": "Barcode Scanner — Operations"}],
        charts=charts,
    )

    provision_dashboard(api, dashboard_key=OPS_KEY)

    assert len(api.created_sections) == 0
    assert len(api.created_charts) == 6  # 7 - 1 existing
    assert len(api.updated_charts) == 1


def test_check_reports_missing_dashboard():
    api = FakeApi()

    with pytest.raises(DashboardApiError, match="Dashboard section.*is missing"):
        provision_dashboard(api, check_only=True)


def test_check_reports_missing_charts():
    api = FakeApi(
        sections=[{"id": "s1", "title": "Barcode Scanner — Operations"}],
        charts=[],
    )

    with pytest.raises(DashboardApiError, match="Dashboard.*is incomplete"):
        provision_dashboard(api, check_only=True, dashboard_key=OPS_KEY)


# ---------------------------------------------------------------------------
# Config + API client
# ---------------------------------------------------------------------------


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

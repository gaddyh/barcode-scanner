"""Coverage gap tests for smaller modules — targets uncovered lines.

Each test class focuses on a single module's uncovered branches:
- cli_app: table printing, --time/--pretty flags, audit_path errors
- dashboard: API client error paths, versions dashboard, dry_run/check_only
- tracing: emit_metadata/emit_pipeline_event/attach/push_feedback with run tree
- runner: _target, run_eval, main
- analyze: analyze_image_async, _reshape edge cases, annotation rendering
- graph: _to_jsonable, scan_path errors, audit cache, recovery edge cases
- db: init_db, create_pool (mocked asyncpg)
- pipeline: tracing-enabled path
- geometry: zero/negative dimension error branches
- event_sink: TraceEventSink with a run tree
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import (
    AsyncMock,
    MagicMock,
    patch,
)
from urllib.error import HTTPError, URLError

import pytest
from PIL import Image

from src.ingest.geometry import PixelBoundingBox
from src.ingest.scanner import (
    BoundingBox,
    DetectedBarcode,
    Point,
)
from src.ingest.vision import (
    AuditConfidence,
    ShoeboxAuditError,
    SpatialLabelAuditPixels,
    SpatialLabelObservationPixels,
    SpatialLabelStatus,
)
from src.observability.event_sink import (
    EventSink,
    TraceEventSink,
    get_sinks,
    register_sink,
    reset_sinks,
    unregister_sink,
)
from src.runtime.context import RunContext
from src.runtime.events import DomainEvent, EventType

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _png_bytes(width: int = 800, height: int = 600) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


def _png_path(tmp_path: Path, name: str = "img.png", w: int = 800, h: int = 600) -> Path:
    p = tmp_path / name
    Image.new("RGB", (w, h), (255, 255, 255)).save(p, format="PNG")
    return p


def _detection(
    value: str,
    *,
    x1: int = 110,
    y1: int = 110,
    x2: int = 190,
    y2: int = 290,
    fmt: str = "Code128",
) -> DetectedBarcode:
    return DetectedBarcode(
        value=value,
        format=fmt,
        content_type="text",
        orientation=0,
        position=(Point(x=x1, y=y1), Point(x=x2, y=y1)),
        bounding_box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
    )


def _label_px(
    index: int,
    *,
    label_box: tuple[int, int, int, int],
    barcode_box: tuple[int, int, int, int] | None = None,
) -> SpatialLabelObservationPixels:
    return SpatialLabelObservationPixels(
        label_index=index,
        label_bbox=PixelBoundingBox(
            x1=label_box[0], y1=label_box[1], x2=label_box[2], y2=label_box[3]
        ),
        barcode_bbox=(
            PixelBoundingBox(
                x1=barcode_box[0], y1=barcode_box[1],
                x2=barcode_box[2], y2=barcode_box[3],
            )
            if barcode_box is not None
            else None
        ),
        status=SpatialLabelStatus.CLEAR,
        confidence=AuditConfidence.HIGH,
    )


def _spatial(
    labels: list[SpatialLabelObservationPixels],
    *,
    image_width: int = 800,
    image_height: int = 600,
) -> SpatialLabelAuditPixels:
    return SpatialLabelAuditPixels(
        image_width=image_width,
        image_height=image_height,
        labels=labels,
    )


class _FakeScanner:
    def __init__(self, detections=None, recovery_detections=None):
        self._detections = detections or []
        self._recovery = recovery_detections or []

    def scan_bytes(self, image_bytes):
        return self._detections

    def scan_crop_with_recovery(self, crop, *, offset_x=0, offset_y=0):
        return self._recovery


def _patch_audit_ok(spatial):
    def _fake(path, *, model, max_retries, retry_delay_seconds):
        return {"status": "ok", "spatial": spatial.model_dump(mode="json")}
    return patch("src.ingest.graph._traced_audit", side_effect=_fake)


def _patch_audit_error(error):
    def _fake(path, *, model, max_retries, retry_delay_seconds):
        return {"status": "error", "error": error}
    return patch("src.ingest.graph._traced_audit", side_effect=_fake)


# ===========================================================================
# cli_app.py
# ===========================================================================


class TestCliApp:
    """Tests for CLI presentation helpers and subcommands."""

    def test_print_timing(self, capsys):
        from src.cli_app import _print_timing
        _print_timing("label", 1.5)
        captured = capsys.readouterr()
        assert "label" in captured.err
        assert "1.50s" in captured.err

    def test_print_scan_table(self, capsys):
        from src.cli_app import _print_scan_table
        rows = [("img1.png", "found", 3, 1.2), ("img2.png", "error", 0, None)]
        _print_scan_table(rows)
        err = capsys.readouterr().err
        assert "Image" in err
        assert "img1.png" in err
        assert "img2.png" in err
        assert "1.20s" in err
        assert "-" in err

    def test_print_audit_table_with_labels(self, capsys):
        from src.cli_app import _print_audit_table
        rows = [("img.png", "ok", "2", "1", 0.5)]
        _print_audit_table(rows)
        err = capsys.readouterr().err
        assert "img.png" in err
        assert "0.50s" in err

    def test_print_audit_table_without_labels(self, capsys):
        from src.cli_app import _print_audit_table
        rows = [("img.png", "ok", "-", "-", None)]
        _print_audit_table(rows)
        err = capsys.readouterr().err
        assert "img.png" in err

    def test_print_pipeline_table(self, capsys):
        from src.cli_app import _print_pipeline_table
        rows = [
            ("img1.png", "2/2", "OK", 2, 1.0),
            ("img2.png", "-/-", "ERR", 0, None),
        ]
        _print_pipeline_table(rows)
        err = capsys.readouterr().err
        assert "img1.png" in err
        assert "img2.png" in err
        assert "OK" in err
        assert "ERR" in err

    def test_scan_command_with_time_flag(
        self, tmp_path, monkeypatch, capsys
    ):
        import zxingcpp

        from src.cli_app import main
        from tests._zxing_fake import make_read_result

        img = _png_path(tmp_path)
        monkeypatch.setattr(
            zxingcpp, "read_barcodes",
            lambda _img, **kw: [make_read_result("1234567890123")],
        )
        rc = main(["scan", str(img), "--time"])
        assert rc == 0
        err = capsys.readouterr().err
        assert "img.png" in err

    def test_scan_command_error_returns_nonzero(
        self, tmp_path, monkeypatch, capsys
    ):
        import zxingcpp

        from src.cli_app import main

        monkeypatch.setattr(zxingcpp, "read_barcodes", lambda _img, **kw: [])
        # Use a non-existent file to trigger error
        rc = main(["scan", str(tmp_path / "missing.png")])
        assert rc == 1
        data = json.loads(capsys.readouterr().out)
        assert data[0]["status"] == "error"

    def test_audit_path_file_not_found(self, tmp_path):
        from src.cli_app import audit_path

        result = audit_path(
            tmp_path / "missing.png",
            model=None,
            max_retries=0,
            retry_delay_seconds=0.0,
            full=False,
        )
        assert result["status"] == "error"
        assert result["error"]["code"] == "unreadable_file"

    def test_audit_path_value_error(self, tmp_path):
        from src.cli_app import audit_path

        with patch("src.cli_app.audit_shoebox_labels", side_effect=ValueError("bad")):
            result = audit_path(
                tmp_path / "img.png",
                model=None,
                max_retries=0,
                retry_delay_seconds=0.0,
                full=False,
            )
        assert result["status"] == "error"
        assert result["error"]["code"] == "audit_failed"

    def test_audit_path_shoebox_error(self, tmp_path):
        from src.cli_app import audit_path

        with patch(
            "src.cli_app.audit_shoebox_labels",
            side_effect=ShoeboxAuditError("boom"),
        ):
            result = audit_path(
                tmp_path / "img.png",
                model=None,
                max_retries=0,
                retry_delay_seconds=0.0,
                full=False,
            )
        assert result["status"] == "error"
        assert result["error"]["code"] == "audit_failed"

    def test_audit_path_full_success(self, tmp_path):
        from src.cli_app import audit_path

        mock_result = MagicMock()
        mock_result.model_dump.return_value = {"labels": []}
        with patch(
            "src.cli_app.audit_shoebox_image", return_value=mock_result
        ):
            result = audit_path(
                tmp_path / "img.png",
                model=None,
                max_retries=0,
                retry_delay_seconds=0.0,
                full=True,
            )
        assert result["status"] == "ok"

    def test_audit_path_labels_success(self, tmp_path):
        from src.cli_app import audit_path

        mock_result = MagicMock()
        mock_result.model_dump.return_value = {
            "labels": [{"status": "clear"}],
            "visible_product_barcode_label_count": 1,
            "clear_product_barcode_label_count": 1,
        }
        with patch(
            "src.cli_app.audit_shoebox_labels", return_value=mock_result
        ):
            result = audit_path(
                tmp_path / "img.png",
                model=None,
                max_retries=0,
                retry_delay_seconds=0.0,
                full=False,
            )
        assert result["status"] == "ok"

    def test_run_audit_with_time_and_pretty(
        self, tmp_path, capsys
    ):
        import argparse

        from src.cli_app import _run_audit

        img = _png_path(tmp_path)
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {
            "labels": [{"status": "clear"}, {"status": "clear"}],
        }
        with patch(
            "src.cli_app.audit_shoebox_labels", return_value=mock_result
        ):
            args = argparse.Namespace(
                images=[img],
                model=None,
                max_retries=0,
                full=False,
                time=True,
                pretty=True,
            )
            rc = _run_audit(args)
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data[0]["status"] == "ok"

    def test_run_audit_error_returns_nonzero(self, tmp_path, capsys):
        import argparse

        from src.cli_app import _run_audit

        img = _png_path(tmp_path)
        with patch(
            "src.cli_app.audit_shoebox_labels", side_effect=ValueError("bad")
        ):
            args = argparse.Namespace(
                images=[img],
                model=None,
                max_retries=0,
                full=False,
                time=False,
                pretty=False,
            )
            rc = _run_audit(args)
        assert rc == 1

    def test_run_audit_no_labels_uses_counts(self, tmp_path, capsys):
        import argparse

        from src.cli_app import _run_audit

        img = _png_path(tmp_path)
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {
            "labels": [],
            "visible_product_barcode_label_count": 3,
            "clear_product_barcode_label_count": 2,
        }
        with patch(
            "src.cli_app.audit_shoebox_labels", return_value=mock_result
        ):
            args = argparse.Namespace(
                images=[img],
                model=None,
                max_retries=0,
                full=False,
                time=False,
                pretty=False,
            )
            rc = _run_audit(args)
        assert rc == 0
        err = capsys.readouterr().err
        assert "3" in err  # visible count

    def test_run_pipeline_with_time_and_diff(
        self, tmp_path, monkeypatch, capsys
    ):
        import argparse

        import zxingcpp

        from src.cli_app import _run_pipeline
        from tests._zxing_fake import make_read_result

        img = _png_path(tmp_path)
        monkeypatch.setattr(
            zxingcpp, "read_barcodes",
            lambda _img, **kw: [make_read_result("111")],
        )
        spatial = _spatial([
            _label_px(1, label_box=(50, 50, 250, 350), barcode_box=(100, 100, 200, 300)),
            _label_px(2, label_box=(450, 50, 650, 350), barcode_box=(500, 100, 600, 300)),
        ])

        def _fake_audit(path, *, model, max_retries, retry_delay_seconds):
            return {"status": "ok", "spatial": spatial.model_dump(mode="json")}

        with patch("src.ingest.graph._traced_audit", side_effect=_fake_audit):
            args = argparse.Namespace(
                images=[img],
                model=None,
                max_retries=0,
                time=True,
                pretty=True,
            )
            rc = _run_pipeline(args)
        assert rc == 0  # all matched → success
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data[0]["ok"] is True

    def test_build_parser(self):
        from src.cli_app import build_parser

        parser = build_parser()
        assert parser.prog == "barcode-scan"

    def test_main_with_no_command_errors(self):
        from src.cli_app import main

        with pytest.raises(SystemExit):
            main([])


# ===========================================================================
# dashboard.py
# ===========================================================================


class TestDashboard:
    """Tests for LangSmith dashboard API client and provisioning."""

    def test_config_from_env_with_tenant(self, monkeypatch):
        from src.observability.dashboard import DashboardConfig

        monkeypatch.setenv("LANGSMITH_API_KEY", "key123")
        monkeypatch.setenv("LANGSMITH_PROJECT_ID", "proj-123")
        monkeypatch.setenv("LANGSMITH_TENANT_ID", "tenant-1")
        monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://api.test.com/")

        cfg = DashboardConfig.from_env()
        assert cfg.api_key == "key123"
        assert cfg.project_id == "proj-123"
        assert cfg.tenant_id == "tenant-1"
        assert cfg.endpoint == "https://api.test.com"

    def test_config_from_env_no_tenant(self, monkeypatch):
        from src.observability.dashboard import DashboardConfig

        monkeypatch.setenv("LANGSMITH_API_KEY", "key")
        monkeypatch.setenv("LANGSMITH_PROJECT_ID", "proj")
        monkeypatch.delenv("LANGSMITH_TENANT_ID", raising=False)

        cfg = DashboardConfig.from_env()
        assert cfg.tenant_id is None

    def test_config_from_env_strips_whitespace(self, monkeypatch):
        from src.observability.dashboard import DashboardConfig

        monkeypatch.setenv("LANGSMITH_API_KEY", "  key  ")
        monkeypatch.setenv("LANGSMITH_PROJECT_ID", "  proj  ")

        cfg = DashboardConfig.from_env()
        assert cfg.api_key == "key"
        assert cfg.project_id == "proj"

    def test_request_http_error(self):
        from src.observability.dashboard import (
            DashboardApiError,
            DashboardConfig,
            LangSmithDashboardApi,
        )

        api = LangSmithDashboardApi(
            DashboardConfig(
                api_key="k", endpoint="https://test", project_id="p",
            )
        )
        err = HTTPError(
            url="https://test/api/v1/x", code=500, msg="Server Error",
            hdrs=None, fp=io.BytesIO(b"error detail"),
        )
        with patch("src.observability.dashboard.urlopen", side_effect=err):
            with pytest.raises(DashboardApiError, match="HTTP 500"):
                api.request("GET", "/x")

    def test_request_url_error(self):
        from src.observability.dashboard import (
            DashboardApiError,
            DashboardConfig,
            LangSmithDashboardApi,
        )

        api = LangSmithDashboardApi(
            DashboardConfig(
                api_key="k", endpoint="https://test", project_id="p",
            )
        )
        with patch(
            "src.observability.dashboard.urlopen",
            side_effect=URLError("conn refused"),
        ):
            with pytest.raises(DashboardApiError, match="Could not reach"):
                api.request("GET", "/x")

    def test_request_empty_payload(self):
        from src.observability.dashboard import (
            DashboardConfig,
            LangSmithDashboardApi,
        )

        api = LangSmithDashboardApi(
            DashboardConfig(
                api_key="k", endpoint="https://test", project_id="p",
            )
        )
        mock_resp = MagicMock()
        mock_resp.read.return_value = b""
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_resp)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        with patch("src.observability.dashboard.urlopen", return_value=mock_ctx):
            result = api.request("GET", "/x")
        assert result is None

    def test_request_with_body_and_tenant(self):
        from src.observability.dashboard import (
            DashboardConfig,
            LangSmithDashboardApi,
        )

        api = LangSmithDashboardApi(
            DashboardConfig(
                api_key="k", endpoint="https://test", project_id="p",
                tenant_id="t1",
            )
        )
        captured = {}

        class _Resp:
            def read(self):
                return b'{"id": "abc"}'
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False

        def _fake_urlopen(req, timeout):
            captured["headers"] = req.headers
            captured["data"] = req.data
            captured["method"] = req.method
            return _Resp()

        with patch("src.observability.dashboard.urlopen", side_effect=_fake_urlopen):
            result = api.request("POST", "/x", body={"key": "val"})
        assert result == {"id": "abc"}
        assert b'"key"' in captured["data"]
        assert captured["method"] == "POST"

    def test_request_with_query_params(self):
        from src.observability.dashboard import (
            DashboardConfig,
            LangSmithDashboardApi,
        )

        api = LangSmithDashboardApi(
            DashboardConfig(
                api_key="k", endpoint="https://test", project_id="p",
            )
        )
        captured = {}

        class _Resp:
            def read(self):
                return b'[]'
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False

        def _fake_urlopen(req, timeout):
            captured["url"] = req.full_url
            return _Resp()

        with patch("src.observability.dashboard.urlopen", side_effect=_fake_urlopen):
            result = api.request("GET", "/x", query={"limit": "10"})
        assert result == []
        assert "limit=10" in captured["url"]

    def test_list_sections_non_list(self):
        from src.observability.dashboard import (
            DashboardConfig,
            LangSmithDashboardApi,
        )

        api = LangSmithDashboardApi(
            DashboardConfig(
                api_key="k", endpoint="https://test", project_id="p",
            )
        )
        with patch.object(api, "request", return_value={"not": "a list"}):
            result = api.list_sections("title")
        assert result == []

    def test_create_section_no_id(self):
        from src.observability.dashboard import (
            DashboardApiError,
            DashboardConfig,
            LangSmithDashboardApi,
        )

        api = LangSmithDashboardApi(
            DashboardConfig(
                api_key="k", endpoint="https://test", project_id="p",
            )
        )
        with patch.object(api, "request", return_value={"no_id": True}):
            with pytest.raises(DashboardApiError, match="section ID"):
                api.create_section("t", "d")

    def test_create_chart_no_id(self):
        from src.observability.dashboard import (
            DashboardApiError,
            DashboardConfig,
            LangSmithDashboardApi,
        )

        api = LangSmithDashboardApi(
            DashboardConfig(
                api_key="k", endpoint="https://test", project_id="p",
            )
        )
        with patch.object(api, "request", return_value={}):
            with pytest.raises(DashboardApiError, match="chart ID"):
                api.create_chart({})

    def test_update_chart_no_id(self):
        from src.observability.dashboard import (
            DashboardApiError,
            DashboardConfig,
            LangSmithDashboardApi,
        )

        api = LangSmithDashboardApi(
            DashboardConfig(
                api_key="k", endpoint="https://test", project_id="p",
            )
        )
        with patch.object(api, "request", return_value={"x": 1}):
            with pytest.raises(DashboardApiError, match="updated chart ID"):
                api.update_chart("c1", {})

    def test_read_charts_non_dict(self):
        from src.observability.dashboard import (
            DashboardConfig,
            LangSmithDashboardApi,
        )

        api = LangSmithDashboardApi(
            DashboardConfig(
                api_key="k", endpoint="https://test", project_id="p",
            )
        )
        with patch.object(api, "request", return_value=[]):
            result = api.read_charts()
        assert result == []

    def test_read_charts_with_sections(self):
        from src.observability.dashboard import (
            DashboardConfig,
            LangSmithDashboardApi,
        )

        api = LangSmithDashboardApi(
            DashboardConfig(
                api_key="k", endpoint="https://test", project_id="p",
            )
        )
        with patch.object(
            api, "request",
            return_value={"sections": [{"charts": [{"id": "c1"}]}]},
        ):
            result = api.read_charts()
        assert result == [{"id": "c1"}]

    def test_build_versions_charts(self):
        from src.observability.dashboard import (
            VERSIONS_KEY,
            build_versions_charts,
        )

        payloads = build_versions_charts("proj", "sec")
        assert len(payloads) == 6
        assert all(
            p["metadata"]["dashboard_key"] == VERSIONS_KEY for p in payloads
        )
        keys = [p["metadata"]["chart_key"] for p in payloads]
        assert "versions-completion-by-pipeline" in keys
        assert "versions-latency-by-pipeline" in keys

    def test_find_section(self):
        from src.observability.dashboard import _find_section

        sections = [{"title": "A"}, {"title": "B"}]
        assert _find_section(sections, "A") == {"title": "A"}
        assert _find_section(sections, "C") is None

    def test_find_chart_no_metadata(self):
        from src.observability.dashboard import _find_chart

        charts = [{"id": "1"}, {"metadata": None}]
        assert _find_chart(charts, "dk", "ck") is None

    def test_provision_dry_run_missing_section(self):
        from src.observability.dashboard import (
            OPS_KEY,
            DashboardConfig,
            LangSmithDashboardApi,
            provision_dashboard,
        )

        api = LangSmithDashboardApi(
            DashboardConfig(
                api_key="k", endpoint="https://test", project_id="p",
            )
        )
        with patch.object(api, "list_sections", return_value=[]):
            result = provision_dashboard(api, dashboard_key=OPS_KEY, dry_run=True)
        assert result["section"] == "missing"
        assert result["charts"] == "not checked"

    def test_provision_dry_run_existing_section(self):
        from src.observability.dashboard import (
            OPS_KEY,
            DashboardConfig,
            LangSmithDashboardApi,
            provision_dashboard,
        )

        api = LangSmithDashboardApi(
            DashboardConfig(
                api_key="k", endpoint="https://test", project_id="p",
            )
        )
        with patch.object(
            api, "list_sections",
            return_value=[{"id": "s1", "title": "Barcode Scanner — Operations"}],
        ), patch.object(api, "read_charts", return_value=[]):
            result = provision_dashboard(api, dashboard_key=OPS_KEY, dry_run=True)
        assert result["section_id"] == "s1"
        assert all(c["action"] == "missing" for c in result["charts"])

    def test_provision_check_only_complete(self):
        from src.observability.dashboard import (
            OPS_KEY,
            DashboardConfig,
            LangSmithDashboardApi,
            build_operations_charts,
            provision_dashboard,
        )

        api = LangSmithDashboardApi(
            DashboardConfig(
                api_key="k", endpoint="https://test", project_id="p",
            )
        )
        charts = build_operations_charts("p", "s1")
        existing = []
        for ch in charts:
            existing.append({
                "id": f"id-{ch['metadata']['chart_key']}",
                "metadata": ch["metadata"],
            })
        with patch.object(
            api, "list_sections",
            return_value=[{"id": "s1", "title": "Barcode Scanner — Operations"}],
        ), patch.object(api, "read_charts", return_value=existing):
            result = provision_dashboard(
                api, dashboard_key=OPS_KEY, check_only=True,
            )
        assert all(c["action"] == "exists" for c in result["charts"])


# ===========================================================================
# tracing.py
# ===========================================================================


class TestTracing:
    """Tests for LangSmith tracing utilities (tracing disabled path)."""

    def test_is_tracing(self):
        from src.observability.tracing import is_tracing
        # Tracing is disabled in test env (no LANGSMITH_TRACING)
        assert is_tracing() is False

    def test_trace_operation_returns_decorator(self):
        from src.observability.tracing import trace_operation

        @trace_operation("test_op")
        def my_func(x):
            return x + 1

        assert my_func(5) == 6

    def test_trace_operation_with_tags_and_metadata(self):
        from src.observability.tracing import trace_operation

        @trace_operation("op", tags=["t1"], metadata={"k": "v"})
        def fn():
            return "ok"

        assert fn() == "ok"

    def test_emit_metadata_no_run(self):
        from src.observability.tracing import emit_metadata

        ctx = RunContext(
            run_id="r1", session_id="s1", source="test",
        )
        # No run tree when tracing disabled — should be a no-op
        emit_metadata(ctx, extra="val")
        # Context vars are set even without tracing
        from src.observability.tracing import _run_id_var, _session_id_var
        assert _run_id_var.get("") == "r1"
        assert _session_id_var.get("") == "s1"

    def test_emit_metadata_with_run(self):
        from src.observability import tracing

        ctx = RunContext(
            run_id="r2", session_id="s2", source="test",
        )
        mock_run = MagicMock()
        with patch.object(tracing, "get_current_run_tree", return_value=mock_run):
            tracing.emit_metadata(ctx, custom="data")
        mock_run.metadata.update.assert_called_once()
        meta = mock_run.metadata.update.call_args[0][0]
        assert meta["run_id"] == "r2"
        assert meta["session_id"] == "s2"
        assert meta["source"] == "test"
        assert meta["custom"] == "data"
        assert "pipeline_version" in meta

    def test_emit_metadata_setdefault_overrides(self):
        from src.observability import tracing

        ctx = RunContext(
            run_id="r3", session_id="s3", source="test",
        )
        mock_run = MagicMock()
        with patch.object(tracing, "get_current_run_tree", return_value=mock_run):
            tracing.emit_metadata(ctx, run_id="override")
        meta = mock_run.metadata.update.call_args[0][0]
        assert meta["run_id"] == "override"

    def test_emit_pipeline_event_no_run(self):
        from src.observability.tracing import emit_pipeline_event

        # No run tree — uses context vars (empty in test)
        emit_pipeline_event(EventType.SCAN_COMPLETED, count=5)

    def test_emit_pipeline_event_with_run(self):
        from src.observability import tracing

        mock_run = MagicMock()
        mock_run.metadata.get.return_value = ""
        with patch.object(tracing, "get_current_run_tree", return_value=mock_run):
            # Also set context vars as fallback
            from src.observability.tracing import _run_id_var, _session_id_var
            _run_id_var.set("ctx-run")
            _session_id_var.set("ctx-sess")
            tracing.emit_pipeline_event(EventType.AUDIT_COMPLETED, count=3)

    def test_emit_pipeline_event_run_has_metadata(self):
        from src.observability import tracing

        mock_run = MagicMock()
        mock_run.metadata.get.side_effect = lambda key, default="": {
            "run_id": "tree-run",
            "session_id": "tree-sess",
        }.get(key, default)
        with patch.object(tracing, "get_current_run_tree", return_value=mock_run):
            tracing.emit_pipeline_event(EventType.RECOVERY_STARTED, n=1)

    def test_attach_image_to_run_no_run(self):
        from src.observability.tracing import attach_image_to_run

        # No run tree — no-op
        attach_image_to_run(b"img", "image/png")

    def test_attach_image_to_run_with_run(self):
        from src.observability import tracing

        mock_run = MagicMock()
        with patch.object(tracing, "get_current_run_tree", return_value=mock_run):
            tracing.attach_image_to_run(b"img", "image/jpeg")
        assert "uploaded_image" in mock_run.attachments

    def test_push_feedback_no_run(self):
        from src.observability.tracing import push_feedback

        push_feedback([{"key": "k", "score": 1}])

    def test_push_feedback_with_run(self):
        from src.observability import tracing

        mock_run = MagicMock()
        mock_run.id = "run-123"
        mock_client = MagicMock()
        with patch.object(tracing, "get_current_run_tree", return_value=mock_run):
            with patch("langsmith.Client", return_value=mock_client):
                tracing.push_feedback([
                    {"key": "correct", "score": 1, "comment": "good"},
                    {"key": "speed", "score": 0},
                ])
        assert mock_client.create_feedback.call_count == 2

    def test_traceable_noop_direct_call(self):
        from src.observability.tracing import traceable

        def fn(x):
            return x

        # Direct callable with no kwargs → returns fn itself
        result = traceable(fn)
        assert result is fn

    def test_traceable_noop_with_kwargs(self):
        from src.observability.tracing import traceable

        def fn(x):
            return x

        wrapped = traceable(name="test", run_type="chain")
        assert wrapped(fn)(5) == 5

    def test_attachment_class(self):
        from src.observability.tracing import Attachment

        att = Attachment(mime_type="image/png", data=b"bytes")
        assert att.mime_type == "image/png"
        assert att.data == b"bytes"


# ===========================================================================
# runner.py
# ===========================================================================


class TestEvalRunner:
    """Tests for the eval runner — _target, run_eval, main."""

    def test_target_success(self):
        from src.evals.runner import _target

        mock_result = MagicMock()
        mock_result.model_dump.return_value = {"status": "complete"}
        with patch("src.evals.runner.ingest_one", return_value=mock_result):
            result = _target({
                "image_path": "/tmp/img.png",
                "image_name": "img.png",
            })
        assert result == {"status": "complete"}

    def test_target_exception(self):
        from src.evals.runner import _target

        with patch(
            "src.evals.runner.ingest_one",
            side_effect=RuntimeError("boom"),
        ):
            result = _target({
                "image_path": "/tmp/img.png",
                "image_name": "img.png",
            })
        assert result["status"] == "failed"
        assert result["error"]["type"] == "RuntimeError"
        assert result["error"]["message"] == "boom"
        assert result["items"] == []
        assert result["metrics"]["scanner_count"] == 0

    def test_run_eval_no_examples(self, capsys):
        from src.evals.runner import run_eval

        with patch(
            "src.evals.runner.load_legacy_dataset", return_value=[]
        ):
            result = run_eval()
        assert result is None
        err = capsys.readouterr().err
        assert "No eval examples" in err

    def test_run_eval_with_examples(self, capsys):
        from src.evals.runner import run_eval

        examples = [{
            "image_path": "/tmp/img.png",
            "image_name": "img.png",
            "expected_outcome": "complete",
        }]
        mock_client = MagicMock()
        mock_client.list_examples.return_value = []
        mock_eval_result = MagicMock()
        with patch(
            "src.evals.runner.load_legacy_dataset", return_value=examples
        ), patch(
            "src.evals.runner.Client", return_value=mock_client
        ), patch(
            "src.evals.runner.evaluate", return_value=mock_eval_result
        ):
            result = run_eval(scanner_only=True)
        assert result is mock_eval_result
        mock_client.read_dataset.assert_called_once()
        mock_client.create_example.assert_called_once()

    def test_run_eval_dataset_exists(self, capsys):
        from src.evals.runner import run_eval

        examples = [{
            "image_path": "/tmp/img.png",
            "image_name": "img.png",
            "expected_outcome": "complete",
        }]
        mock_client = MagicMock()
        mock_client.list_examples.return_value = []
        with patch(
            "src.evals.runner.load_legacy_dataset", return_value=examples
        ), patch(
            "src.evals.runner.Client", return_value=mock_client
        ), patch(
            "src.evals.runner.evaluate", return_value="results"
        ):
            result = run_eval(experiment_prefix="test-prefix")
        assert result == "results"

    def test_run_eval_deletes_existing_examples(self):
        from src.evals.runner import run_eval

        examples = [{
            "image_path": "/tmp/img.png",
            "image_name": "img.png",
            "expected_outcome": "complete",
        }]
        mock_client = MagicMock()
        existing = [MagicMock(id="ex1"), MagicMock(id="ex2")]
        mock_client.list_examples.return_value = existing
        with patch(
            "src.evals.runner.load_legacy_dataset", return_value=examples
        ), patch(
            "src.evals.runner.Client", return_value=mock_client
        ), patch(
            "src.evals.runner.evaluate", return_value="ok"
        ):
            run_eval()
        mock_client.delete_examples.assert_called_once()

    def test_main_scanner_only(self):
        from src.evals.runner import main

        with patch("src.evals.runner.run_eval") as mock_run:
            rc = main(["--scanner-only"])
        assert rc == 0
        mock_run.assert_called_once_with(
            scanner_only=True, experiment_prefix="barcode-scanner",
        )

    def test_main_with_prefix(self):
        from src.evals.runner import main

        with patch("src.evals.runner.run_eval") as mock_run:
            rc = main(["--experiment-prefix", "custom"])
        assert rc == 0
        mock_run.assert_called_once_with(
            scanner_only=False, experiment_prefix="custom",
        )


# ===========================================================================
# analyze.py
# ===========================================================================


class TestAnalyze:
    """Tests for analyze_image_async and _reshape edge cases."""

    @pytest.mark.asyncio
    async def test_analyze_image_async_path(self, tmp_path):
        from src.ingest.analyze import analyze_image_async

        img = _png_path(tmp_path)
        detections = [_detection("111")]
        spatial = _spatial([
            _label_px(1, label_box=(50, 50, 250, 350), barcode_box=(100, 100, 200, 300)),
        ])
        scanner = _FakeScanner(detections)
        with _patch_audit_ok(spatial):
            result = await analyze_image_async(img, scanner=scanner)
        assert result["outcome"] == "complete"

    @pytest.mark.asyncio
    async def test_analyze_image_async_bytes(self):
        from src.ingest.analyze import analyze_image_async

        image_bytes = _png_bytes()
        detections = [_detection("111")]
        spatial = _spatial([
            _label_px(1, label_box=(50, 50, 250, 350), barcode_box=(100, 100, 200, 300)),
        ])
        scanner = _FakeScanner(detections)
        with _patch_audit_ok(spatial):
            result = await analyze_image_async(image_bytes, scanner=scanner)
        assert result["outcome"] == "complete"
        assert result["image_width"] == 800

    @pytest.mark.asyncio
    async def test_analyze_image_async_scan_error(self, tmp_path):
        from src.ingest.analyze import analyze_image_async

        img = _png_path(tmp_path)
        spatial = _spatial([])

        class _ErrScanner:
            def scan_bytes(self, b):
                raise ValueError("bad")

            def scan_crop_with_recovery(self, c, **kw):
                return []

        with _patch_audit_ok(spatial):
            result = await analyze_image_async(img, scanner=_ErrScanner())
        assert result["outcome"] == "retryable_error"
        assert result["ok"] is False

    @pytest.mark.asyncio
    async def test_analyze_image_async_invalid_image_path(self):
        from src.ingest.analyze import analyze_image_async

        # Non-existent path → _image_dimensions fails, pipeline fails
        scanner = _FakeScanner([])
        with _patch_audit_error({"type": "FileNotFoundError", "message": "x"}):
            result = await analyze_image_async(
                Path("/nonexistent/img.png"), scanner=scanner,
            )
        assert result["outcome"] == "retryable_error"

    def test_reshape_scan_error(self):
        from src.ingest.analyze import _reshape

        summary = {
            "scan_status": "error",
            "audit_status": "ok",
            "scan_error": {"code": "invalid_image", "message": "bad"},
        }
        result = _reshape(summary, 800, 600)
        assert result["outcome"] == "retryable_error"
        assert result["ok"] is False
        assert result["error"] == {"code": "invalid_image", "message": "bad"}

    def test_reshape_audit_error(self):
        from src.ingest.analyze import _reshape

        summary = {
            "scan_status": "found",
            "audit_status": "error",
            "audit_error": {"type": "ShoeboxAuditError", "message": "x"},
            "scanner_detections": [
                {"value": "111", "format": "Code128", "bounding_box": {}},
            ],
        }
        result = _reshape(summary, 800, 600)
        assert result["outcome"] == "retryable_error"
        assert result["ok"] is True
        assert result["audit_available"] is False
        assert len(result["unassigned"]) == 1

    def test_reshape_invalid_detection_index(self):
        from src.ingest.analyze import _reshape

        summary = {
            "scan_status": "found",
            "audit_status": "ok",
            "scanner_detections": [],
            "gemini_labels": [
                {"label_index": 1, "label_bbox": {}, "status": "clear"},
            ],
            "reconciliation": {
                "matches": [{
                    "label_index": 1,
                    "scanner_detection_index": 99,
                    "barcode_value": "111",
                    "match_basis": "containment",
                }],
                "unmatched_labels": [],
                "unassigned_scanner_detections": [],
            },
        }
        result = _reshape(summary, 800, 600)
        assert result["found"] == []

    def test_reshape_missing_label_index(self):
        from src.ingest.analyze import _reshape

        summary = {
            "scan_status": "found",
            "audit_status": "ok",
            "scanner_detections": [
                {"value": "111", "format": "Code128", "bounding_box": {}},
            ],
            "gemini_labels": [],
            "reconciliation": {
                "matches": [{
                    "label_index": 99,
                    "scanner_detection_index": 0,
                    "barcode_value": "111",
                    "match_basis": "containment",
                }],
                "unmatched_labels": [],
                "unassigned_scanner_detections": [],
            },
        }
        result = _reshape(summary, 800, 600)
        assert result["found"] == []

    def test_reshape_complete_with_unassigned(self):
        from src.ingest.analyze import _reshape

        summary = {
            "scan_status": "found",
            "audit_status": "ok",
            "scanner_detections": [
                {"value": "111", "format": "Code128", "bounding_box": {}},
                {"value": "999", "format": "Code128", "bounding_box": {}},
            ],
            "gemini_labels": [
                {"label_index": 1, "label_bbox": {}, "status": "clear"},
            ],
            "reconciliation": {
                "matches": [{
                    "label_index": 1,
                    "scanner_detection_index": 0,
                    "barcode_value": "111",
                    "match_basis": "containment",
                }],
                "unmatched_labels": [],
                "unassigned_scanner_detections": [
                    {"value": "999", "format": "Code128", "bounding_box": {}},
                ],
            },
        }
        result = _reshape(summary, 800, 600)
        assert result["outcome"] == "complete"
        assert result["summary"]["unassigned_count"] == 1

    def test_reshape_needs_better_photo_zero_labels(self):
        from src.ingest.analyze import _reshape

        summary = {
            "scan_status": "found",
            "audit_status": "ok",
            "scanner_detections": [],
            "gemini_labels": [],
            "reconciliation": {
                "matches": [],
                "unmatched_labels": [],
                "unassigned_scanner_detections": [],
            },
        }
        result = _reshape(summary, 800, 600)
        assert result["outcome"] == "needs_better_photo"
        assert "No barcode labels" in result["message"]

    def test_reshape_with_audit_latency_and_recovery(self):
        from src.ingest.analyze import _reshape

        summary = {
            "scan_status": "found",
            "audit_status": "ok",
            "audit_latency_ms": 500,
            "scanner_detections": [
                {"value": "111", "format": "Code128", "bounding_box": {}},
            ],
            "gemini_labels": [
                {"label_index": 1, "label_bbox": {}, "status": "clear"},
            ],
            "reconciliation": {
                "matches": [{
                    "label_index": 1,
                    "scanner_detection_index": 0,
                    "barcode_value": "111",
                    "match_basis": "containment",
                }],
                "unmatched_labels": [],
                "unassigned_scanner_detections": [],
            },
            "recovery": {
                "attempted": True,
                "labels_tried": 1,
                "barcodes_found": 1,
                "labels_resolved": 1,
            },
        }
        result = _reshape(summary, 800, 600)
        assert result["summary"]["audit_latency_ms"] == 500
        assert result["summary"]["recovery"]["attempted"] is True

    def test_render_missing_annotation_with_label_bbox_fallback(self, tmp_path):
        from src.ingest.analyze import _render_missing_annotation

        img = _png_path(tmp_path, w=400, h=400)
        missing = [{
            "label_index": 1,
            "label_bbox": {"x1": 50, "y1": 50, "x2": 150, "y2": 150},
            "barcode_bbox": None,
        }]
        b64, w, h = _render_missing_annotation(img, missing)
        assert isinstance(b64, str)
        assert w > 0
        assert h > 0

    def test_render_missing_annotation_skips_no_box(self, tmp_path):
        from src.ingest.analyze import _render_missing_annotation

        img = _png_path(tmp_path, w=200, h=200)
        missing = [{"label_index": 1}]  # no bbox at all
        b64, w, h = _render_missing_annotation(img, missing)
        assert isinstance(b64, str)

    def test_analyze_image_cleanup_on_error(self, tmp_path):
        from src.ingest.analyze import analyze_image

        image_bytes = _png_bytes()
        with patch(
            "src.ingest.analyze.pipeline_path",
            side_effect=RuntimeError("boom"),
        ):
            with pytest.raises(RuntimeError, match="boom"):
                analyze_image(image_bytes, scanner=_FakeScanner([]))

    def test_analyze_image_str_path(self, tmp_path):
        from src.ingest.analyze import analyze_image

        img = _png_path(tmp_path)
        detections = [_detection("111")]
        spatial = _spatial([
            _label_px(1, label_box=(50, 50, 250, 350), barcode_box=(100, 100, 200, 300)),
        ])
        scanner = _FakeScanner(detections)
        with _patch_audit_ok(spatial):
            result = analyze_image(str(img), scanner=scanner)
        assert result["outcome"] == "complete"

    def test_analyze_image_unlink_error_suppressed(self, tmp_path):
        from src.ingest.analyze import analyze_image

        image_bytes = _png_bytes()
        detections = [_detection("111")]
        spatial = _spatial([
            _label_px(1, label_box=(50, 50, 250, 350), barcode_box=(100, 100, 200, 300)),
        ])
        scanner = _FakeScanner(detections)
        with _patch_audit_ok(spatial):
            with patch("os.unlink", side_effect=OSError("can't delete")):
                result = analyze_image(image_bytes, scanner=scanner)
        assert result["outcome"] == "complete"


# ===========================================================================
# graph.py
# ===========================================================================


class TestGraph:
    """Tests for graph helpers, cache modes, and recovery edge cases."""

    def test_to_jsonable_dataclass(self):
        from dataclasses import dataclass

        from src.ingest.graph import _to_jsonable

        @dataclass
        class Item:
            x: int
            y: int

        result = _to_jsonable(Item(1, 2))
        assert result == {"x": 1, "y": 2}

    def test_to_jsonable_tuple(self):
        from src.ingest.graph import _to_jsonable

        result = _to_jsonable((1, "a", 3.0))
        assert result == [1, "a", 3.0]

    def test_to_jsonable_plain(self):
        from src.ingest.graph import _to_jsonable

        assert _to_jsonable(42) == 42
        assert _to_jsonable("hello") == "hello"

    def test_scan_path_unreadable_file(self, tmp_path):
        from src.ingest.graph import scan_path

        scanner = _FakeScanner()
        result = scan_path(tmp_path / "missing.png", scanner)
        assert result["status"] == "error"
        assert result["error"]["code"] == "unreadable_file"

    def test_scan_path_invalid_image(self, tmp_path):
        from src.ingest.graph import scan_path
        from src.ingest.scanner import BarcodeScanner

        bad = tmp_path / "bad.png"
        bad.write_bytes(b"not an image")
        scanner = BarcodeScanner()
        result = scan_path(bad, scanner)
        assert result["status"] == "error"
        assert result["error"]["code"] == "invalid_image"

    def test_scan_path_found(self, tmp_path):
        from src.ingest.graph import scan_path

        img = _png_path(tmp_path)
        scanner = _FakeScanner([_detection("111")])
        result = scan_path(img, scanner)
        assert result["status"] == "found"
        assert result["count"] == 1

    def test_scan_path_not_found(self, tmp_path):
        from src.ingest.graph import scan_path

        img = _png_path(tmp_path)
        scanner = _FakeScanner([])
        result = scan_path(img, scanner)
        assert result["status"] == "not_found"
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_traced_audit_file_not_found(self, tmp_path):
        from src.ingest.graph import _traced_audit

        with patch(
            "src.ingest.graph.audit_shoebox_labels_async",
            side_effect=FileNotFoundError("missing"),
        ):
            result = await _traced_audit(
                tmp_path / "x.png",
                model=None, max_retries=0, retry_delay_seconds=0.0,
            )
        assert result["status"] == "error"
        assert result["error"]["type"] == "FileNotFoundError"

    @pytest.mark.asyncio
    async def test_traced_audit_value_error(self, tmp_path):
        from src.ingest.graph import _traced_audit

        with patch(
            "src.ingest.graph.audit_shoebox_labels_async",
            side_effect=ValueError("bad key"),
        ):
            result = await _traced_audit(
                tmp_path / "x.png",
                model=None, max_retries=0, retry_delay_seconds=0.0,
            )
        assert result["status"] == "error"
        assert result["error"]["type"] == "ValueError"

    @pytest.mark.asyncio
    async def test_traced_audit_shoebox_error(self, tmp_path):
        from src.ingest.graph import _traced_audit

        with patch(
            "src.ingest.graph.audit_shoebox_labels_async",
            side_effect=ShoeboxAuditError("fail"),
        ):
            result = await _traced_audit(
                tmp_path / "x.png",
                model=None, max_retries=0, retry_delay_seconds=0.0,
            )
        assert result["status"] == "error"
        assert result["error"]["type"] == "ShoeboxAuditError"

    @pytest.mark.asyncio
    async def test_traced_audit_replay_hit(self, tmp_path):
        import src.ingest.graph as g
        from src.ingest.graph import _traced_audit

        img = _png_path(tmp_path)
        mock_store = MagicMock()
        mock_store.get.return_value = {"labels": []}
        g.set_audit_cache_mode("replay", mock_store)
        try:
            result = await _traced_audit(
                img, model=None, max_retries=0, retry_delay_seconds=0.0,
            )
        finally:
            g.set_audit_cache_mode(None)
        assert result["status"] == "ok"
        assert result["audit_latency_ms"] == 0

    @pytest.mark.asyncio
    async def test_traced_audit_replay_miss(self, tmp_path):
        import src.ingest.graph as g
        from src.ingest.graph import _traced_audit

        img = _png_path(tmp_path)
        mock_store = MagicMock()
        mock_store.get.return_value = None
        g.set_audit_cache_mode("replay", mock_store)
        try:
            result = await _traced_audit(
                img, model="gemini-test",
                max_retries=0, retry_delay_seconds=0.0,
            )
        finally:
            g.set_audit_cache_mode(None)
        assert result["status"] == "error"
        assert "CacheMiss" in result["error"]["type"]

    @pytest.mark.asyncio
    async def test_traced_audit_capture_mode(self, tmp_path):
        import src.ingest.graph as g
        from src.ingest.graph import _traced_audit

        img = _png_path(tmp_path)
        mock_spatial = MagicMock()
        mock_spatial.model_dump.return_value = {"labels": [{"label_index": 1}]}
        mock_store = MagicMock()
        g.set_audit_cache_mode("capture", mock_store)
        try:
            with patch(
                "src.ingest.graph.audit_shoebox_labels_async",
                return_value=mock_spatial,
            ):
                result = await _traced_audit(
                    img, model=None, max_retries=0, retry_delay_seconds=0.0,
                )
        finally:
            g.set_audit_cache_mode(None)
        assert result["status"] == "ok"
        mock_store.put.assert_called_once()

    def test_gemini_guided_recovery_image_open_fails(self, tmp_path):
        from src.ingest.graph import _gemini_guided_recovery

        bad = tmp_path / "bad.png"
        bad.write_bytes(b"not an image")
        result = _gemini_guided_recovery(bad, _FakeScanner(), [], 800, 600)
        assert result == []

    def test_gemini_guided_recovery_no_bbox(self, tmp_path):
        from src.ingest.graph import _gemini_guided_recovery

        img = _png_path(tmp_path)
        # Label with no bbox at all
        ul = MagicMock()
        ul.barcode_bbox = None
        ul.label_bbox = None
        ul.label_index = 1
        result = _gemini_guided_recovery(img, _FakeScanner(), [ul], 800, 600)
        assert result == []

    def test_gemini_guided_recovery_degenerate_bbox(self, tmp_path):
        from src.ingest.graph import _gemini_guided_recovery

        img = _png_path(tmp_path)
        ul = MagicMock()
        ul.barcode_bbox = {"x1": 100, "y1": 100, "x2": 100, "y2": 100}
        ul.label_bbox = None
        ul.label_index = 1
        result = _gemini_guided_recovery(img, _FakeScanner(), [ul], 800, 600)
        assert result == []

    def test_gemini_guided_recovery_finds_barcode(self, tmp_path):
        from src.ingest.graph import _gemini_guided_recovery

        img = _png_path(tmp_path)
        ul = MagicMock()
        ul.barcode_bbox = {"x1": 100, "y1": 100, "x2": 200, "y2": 200}
        ul.label_bbox = None
        ul.label_index = 1
        det = _detection("123")
        scanner = _FakeScanner(recovery_detections=[det])
        result = _gemini_guided_recovery(img, scanner, [ul], 800, 600)
        assert len(result) == 1
        assert result[0]["value"] == "123"

    def test_gemini_guided_recovery_label_bbox_fallback(self, tmp_path):
        from src.ingest.graph import _gemini_guided_recovery

        img = _png_path(tmp_path)
        ul = MagicMock()
        ul.barcode_bbox = None
        ul.label_bbox = {"x1": 100, "y1": 100, "x2": 200, "y2": 200}
        ul.label_index = 1
        det = _detection("456")
        scanner = _FakeScanner(recovery_detections=[det])
        result = _gemini_guided_recovery(img, scanner, [ul], 800, 600)
        assert len(result) == 1
        assert result[0]["value"] == "456"

    def test_gemini_guided_recovery_no_detection(self, tmp_path):
        from src.ingest.graph import _gemini_guided_recovery

        img = _png_path(tmp_path)
        ul = MagicMock()
        ul.barcode_bbox = {"x1": 100, "y1": 100, "x2": 200, "y2": 200}
        ul.label_bbox = None
        ul.label_index = 1
        scanner = _FakeScanner()  # no recovery detections
        result = _gemini_guided_recovery(img, scanner, [ul], 800, 600)
        assert result == []

    @pytest.mark.asyncio
    async def test_scan_node(self):
        from src.ingest.graph import _scan_node

        path = Path("/tmp/test.png")
        scanner = _FakeScanner([_detection("111")])
        config = {"configurable": {"scanner": scanner}}
        with patch("src.ingest.graph.scan_path") as mock_scan:
            mock_scan.return_value = {
                "status": "found", "count": 1, "barcodes": [],
            }
            result = await _scan_node({"path": str(path)}, config)
        assert result["scan_ok"] is True

    @pytest.mark.asyncio
    async def test_scan_node_error(self):
        from src.ingest.graph import _scan_node

        config = {"configurable": {"scanner": _FakeScanner()}}
        with patch("src.ingest.graph.scan_path") as mock_scan:
            mock_scan.return_value = {
                "status": "error", "count": 0, "error": {"code": "x"},
            }
            result = await _scan_node({"path": "/tmp/x.png"}, config)
        assert result["scan_ok"] is False

    @pytest.mark.asyncio
    async def test_scan_node_no_barcodes(self):
        from src.ingest.graph import _scan_node

        config = {"configurable": {"scanner": _FakeScanner()}}
        with patch("src.ingest.graph.scan_path") as mock_scan:
            mock_scan.return_value = {
                "status": "not_found", "count": 0, "barcodes": [],
            }
            result = await _scan_node({"path": "/tmp/x.png"}, config)
        assert result["scan_ok"] is True

    @pytest.mark.asyncio
    async def test_audit_node_ok(self):
        from src.ingest.graph import _audit_node

        with patch("src.ingest.graph._traced_audit", new_callable=AsyncMock) as mock:
            mock.return_value = {
                "status": "ok",
                "spatial": {"labels": [{"label_index": 1}]},
            }
            result = await _audit_node({
                "path": "/tmp/x.png",
                "model": None,
                "max_retries": 0,
                "retry_delay_seconds": 0.0,
            })
        assert result["audit_ok"] is True

    @pytest.mark.asyncio
    async def test_audit_node_error(self):
        from src.ingest.graph import _audit_node

        with patch("src.ingest.graph._traced_audit", new_callable=AsyncMock) as mock:
            mock.return_value = {"status": "error", "error": {"type": "X"}}
            result = await _audit_node({
                "path": "/tmp/x.png",
                "max_retries": 0,
                "retry_delay_seconds": 0.0,
            })
        assert result["audit_ok"] is False

    @pytest.mark.asyncio
    async def test_reconcile_node_skips_on_scan_error(self):
        from src.ingest.graph import _reconcile_node

        result = await _reconcile_node({"scan_ok": False, "audit_ok": True})
        assert result == {}

    @pytest.mark.asyncio
    async def test_reconcile_node_skips_on_audit_error(self):
        from src.ingest.graph import _reconcile_node

        result = await _reconcile_node({"scan_ok": True, "audit_ok": False})
        assert result == {}

    @pytest.mark.asyncio
    async def test_reconcile_node_with_barcodes_in_state(self):
        from src.ingest.graph import _reconcile_node

        spatial = _spatial([
            _label_px(1, label_box=(50, 50, 250, 350), barcode_box=(100, 100, 200, 300)),
        ])
        det_dict = {
            "value": "111", "format": "Code128", "content_type": "text",
            "orientation": 0,
            "position": [{"x": 110, "y": 110}, {"x": 190, "y": 110}],
            "bounding_box": {"x1": 110, "y1": 110, "x2": 190, "y2": 290},
        }
        state = {
            "scan_ok": True,
            "audit_ok": True,
            "barcodes": [det_dict],
            "audit_result": {
                "status": "ok",
                "spatial": spatial.model_dump(mode="json"),
            },
        }
        result = await _reconcile_node(state)
        assert "reconciliation" in result
        assert "barcodes" in result

    @pytest.mark.asyncio
    async def test_recover_node_no_reconciliation(self):
        from src.ingest.graph import _recover_node

        config = {"configurable": {"scanner": _FakeScanner()}}
        result = await _recover_node({}, config)
        assert result == {}

    @pytest.mark.asyncio
    async def test_recover_node_no_unmatched(self):
        from src.ingest.graph import _recover_node

        mock_recon = MagicMock()
        mock_recon.unmatched_labels = []
        config = {"configurable": {"scanner": _FakeScanner()}}
        result = await _recover_node(
            {"reconciliation": mock_recon}, config,
        )
        assert result["recovery_attempted"] is True

    @pytest.mark.asyncio
    async def test_recover_node_with_detections(self, tmp_path):
        from src.ingest.graph import _recover_node

        img = _png_path(tmp_path)
        ul = MagicMock()
        ul.barcode_bbox = {"x1": 100, "y1": 100, "x2": 200, "y2": 200}
        ul.label_bbox = None
        ul.label_index = 1
        mock_recon = MagicMock()
        mock_recon.unmatched_labels = [ul]
        mock_recon.matched_label_count = 0
        det = _detection("999")
        scanner = _FakeScanner(recovery_detections=[det])
        config = {"configurable": {"scanner": scanner}}
        state = {
            "reconciliation": mock_recon,
            "image_width": 800,
            "image_height": 600,
            "path": str(img),
            "barcodes": [],
        }
        result = await _recover_node(state, config)
        assert result["recovery_attempted"] is True
        assert result["recovery_barcodes_found"] == 1
        assert len(result["barcodes"]) == 1

    @pytest.mark.asyncio
    async def test_recover_node_no_detections(self, tmp_path):
        from src.ingest.graph import _recover_node

        img = _png_path(tmp_path)
        ul = MagicMock()
        ul.barcode_bbox = {"x1": 100, "y1": 100, "x2": 200, "y2": 200}
        ul.label_bbox = None
        ul.label_index = 1
        mock_recon = MagicMock()
        mock_recon.unmatched_labels = [ul]
        mock_recon.matched_label_count = 0
        scanner = _FakeScanner()  # no recovery detections
        config = {"configurable": {"scanner": scanner}}
        state = {
            "reconciliation": mock_recon,
            "image_width": 800,
            "image_height": 600,
            "path": str(img),
            "barcodes": [],
        }
        result = await _recover_node(state, config)
        assert result["recovery_attempted"] is True
        assert result["recovery_barcodes_found"] == 0

    @pytest.mark.asyncio
    async def test_finalize_node_scan_error(self):
        from src.ingest.graph import _finalize_node

        state = {
            "scan_result": {"status": "error", "error": {"code": "x"}},
            "audit_result": {"status": "ok"},
            "scan_ok": False,
            "audit_ok": True,
            "path": "/tmp/x.png",
        }
        result = await _finalize_node(state)
        summary = result["summary"]
        assert summary["ok"] is False
        assert summary["scan_error"] == {"code": "x"}

    @pytest.mark.asyncio
    async def test_finalize_node_audit_error(self):
        from src.ingest.graph import _finalize_node

        state = {
            "scan_result": {
                "status": "found",
                "barcodes": [{"value": "111"}],
            },
            "audit_result": {"status": "error", "error": {"type": "X"}},
            "scan_ok": True,
            "audit_ok": False,
            "path": "/tmp/x.png",
        }
        result = await _finalize_node(state)
        summary = result["summary"]
        assert summary["ok"] is False
        assert summary["audit_error"] == {"type": "X"}
        assert summary["decoded_count"] == 1

    @pytest.mark.asyncio
    async def test_finalize_node_both_ok_with_recovery(self):
        from src.ingest.graph import _finalize_node

        mock_recon = MagicMock()
        mock_recon.matched_label_count = 2
        mock_recon.all_labels_matched = True
        mock_recon.model_dump.return_value = {"matches": []}
        state = {
            "scan_result": {
                "status": "found",
                "barcodes": [{"value": "111"}, {"value": "222"}],
            },
            "audit_result": {"status": "ok"},
            "scan_ok": True,
            "audit_ok": True,
            "path": "/tmp/x.png",
            "barcodes": [{"value": "111"}, {"value": "222"}],
            "labels": [{"label_index": 1, "status": "clear"}],
            "reconciliation": mock_recon,
            "recovery_attempted": True,
            "recovery_labels_tried": 1,
            "recovery_barcodes_found": 1,
            "matched_before": 1,
        }
        result = await _finalize_node(state)
        summary = result["summary"]
        assert summary["ok"] is True
        assert summary["recovery"]["attempted"] is True
        assert summary["recovery"]["labels_resolved"] == 1
        assert summary["decoded_vs_visible"]["all_labels_matched"] is True

    @pytest.mark.asyncio
    async def test_finalize_node_both_ok_no_recovery(self):
        from src.ingest.graph import _finalize_node

        mock_recon = MagicMock()
        mock_recon.matched_label_count = 1
        mock_recon.all_labels_matched = True
        mock_recon.model_dump.return_value = {"matches": []}
        state = {
            "scan_result": {
                "status": "found",
                "barcodes": [{"value": "111"}],
            },
            "audit_result": {"status": "ok"},
            "scan_ok": True,
            "audit_ok": True,
            "path": "/tmp/x.png",
            "barcodes": [{"value": "111"}],
            "labels": [{"label_index": 1, "status": "clear"}],
            "reconciliation": mock_recon,
        }
        result = await _finalize_node(state)
        summary = result["summary"]
        assert summary["recovery"]["attempted"] is False
        assert summary["recovery"]["labels_resolved"] == 0

    def test_build_scan_graph(self):
        from src.ingest.graph import build_scan_graph

        graph = build_scan_graph()
        assert graph is not None

    def test_build_scan_graph_with_checkpointer(self):
        from langgraph.checkpoint.memory import MemorySaver

        from src.ingest.graph import build_scan_graph

        graph = build_scan_graph(checkpointer=MemorySaver())
        assert graph is not None

    @pytest.mark.asyncio
    async def test_run_scan_graph_with_thread_id_no_checkpointer(self, tmp_path):
        import src.ingest.graph as g
        from src.ingest.graph import run_scan_graph

        img = _png_path(tmp_path)
        detections = [_detection("111")]
        spatial = _spatial([
            _label_px(1, label_box=(50, 50, 250, 350), barcode_box=(100, 100, 200, 300)),
        ])
        scanner = _FakeScanner(detections)
        g._invalidate_graph_cache()
        with _patch_audit_ok(spatial):
            with patch("src.ingest.checkpoint.get_checkpointer", return_value=None):
                summary = await run_scan_graph(
                    img, scanner, model=None,
                    max_retries=0, retry_delay_seconds=0.0,
                    thread_id="test-thread",
                )
        assert summary["ok"] is True

    def test_invalidate_graph_cache(self):
        import src.ingest.graph as g

        g._compiled_graph_no_checkpoint = MagicMock()
        g._compiled_graph_with_checkpoint = MagicMock()
        g._invalidate_graph_cache()
        assert g._compiled_graph_no_checkpoint is None
        assert g._compiled_graph_with_checkpoint is None

    def test_route_after_reconcile_scan_error(self):
        from src.ingest.graph import _route_after_reconcile

        state = {"scan_ok": False, "audit_ok": True}
        assert _route_after_reconcile(state) == "finalize"

    def test_route_after_reconcile_audit_error(self):
        from src.ingest.graph import _route_after_reconcile

        state = {"scan_ok": True, "audit_ok": False}
        assert _route_after_reconcile(state) == "finalize"

    def test_route_after_reconcile_no_reconciliation(self):
        from src.ingest.graph import _route_after_reconcile

        state = {"scan_ok": True, "audit_ok": True}
        assert _route_after_reconcile(state) == "finalize"


# ===========================================================================
# db.py
# ===========================================================================


class TestDb:
    """Tests for db.py — init_db and create_pool (mocked asyncpg)."""

    @pytest.mark.asyncio
    async def test_init_db(self):
        from src.db import init_db

        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_conn,
        )
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
        await init_db(mock_pool)
        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_pool(self):
        from src.db import create_pool

        mock_pool = MagicMock()
        with patch("src.db.asyncpg.create_pool", new_callable=AsyncMock) as mock:
            mock.return_value = mock_pool
            result = await create_pool("postgres://localhost/test")
        assert result is mock_pool
        mock.assert_called_once_with(
            dsn="postgres://localhost/test",
            min_size=2, max_size=10, command_timeout=30,
        )

    @pytest.mark.asyncio
    async def test_create_pool_custom_sizes(self):
        from src.db import create_pool

        mock_pool = MagicMock()
        with patch("src.db.asyncpg.create_pool", new_callable=AsyncMock) as mock:
            mock.return_value = mock_pool
            result = await create_pool(
                "postgres://localhost/test", min_size=5, max_size=20,
            )
        assert result is mock_pool
        mock.assert_called_once_with(
            dsn="postgres://localhost/test",
            min_size=5, max_size=20, command_timeout=30,
        )


# ===========================================================================
# pipeline.py
# ===========================================================================


class TestPipeline:
    """Tests for pipeline facade — tracing-enabled path."""

    def test_pipeline_path_tracing_enabled(self, tmp_path):
        import src.ingest.pipeline as pipeline_mod
        from src.ingest.pipeline import pipeline_path

        img = _png_path(tmp_path)
        detections = [_detection("111")]
        spatial = _spatial([
            _label_px(1, label_box=(50, 50, 250, 350), barcode_box=(100, 100, 200, 300)),
        ])
        scanner = _FakeScanner(detections)

        mock_run = MagicMock()
        mock_ls = MagicMock()
        mock_ls.get_current_run_tree.return_value = mock_run

        with _patch_audit_ok(spatial):
            with patch.object(pipeline_mod, "_TRACING", True):
                pipeline_mod.ls = mock_ls
                try:
                    summary = pipeline_path(
                        img, scanner, model=None,
                        max_retries=0, retry_delay_seconds=0.0,
                    )
                finally:
                    if hasattr(pipeline_mod, "ls"):
                        del pipeline_mod.ls
        assert summary["ok"] is True
        mock_run.metadata.update.assert_called_once()

    def test_pipeline_path_tracing_no_run(self, tmp_path):
        import src.ingest.pipeline as pipeline_mod
        from src.ingest.pipeline import pipeline_path

        img = _png_path(tmp_path)
        detections = [_detection("111")]
        spatial = _spatial([
            _label_px(1, label_box=(50, 50, 250, 350), barcode_box=(100, 100, 200, 300)),
        ])
        scanner = _FakeScanner(detections)

        mock_ls = MagicMock()
        mock_ls.get_current_run_tree.return_value = None

        with _patch_audit_ok(spatial):
            with patch.object(pipeline_mod, "_TRACING", True):
                pipeline_mod.ls = mock_ls
                try:
                    summary = pipeline_path(
                        img, scanner, model=None,
                        max_retries=0, retry_delay_seconds=0.0,
                    )
                finally:
                    if hasattr(pipeline_mod, "ls"):
                        del pipeline_mod.ls
        assert summary["ok"] is True


# ===========================================================================
# geometry.py
# ===========================================================================


class TestGeometry:
    """Tests for geometry error branches."""

    def test_normalized_to_pixels_zero_height(self):
        from src.ingest.geometry import normalized_to_pixels

        with pytest.raises(ValueError, match="image_height must be positive"):
            normalized_to_pixels(
                top=0, left=0, bottom=10, right=10,
                image_width=100, image_height=0,
            )

    def test_clamp_bbox_negative_width(self):
        from src.ingest.geometry import PixelBoundingBox, clamp_bbox

        box = PixelBoundingBox(x1=10, y1=10, x2=50, y2=50)
        with pytest.raises(ValueError, match="image dimensions must be non-negative"):
            clamp_bbox(box, image_width=-1, image_height=100)

    def test_clamp_bbox_negative_height(self):
        from src.ingest.geometry import PixelBoundingBox, clamp_bbox

        box = PixelBoundingBox(x1=10, y1=10, x2=50, y2=50)
        with pytest.raises(ValueError, match="image dimensions must be non-negative"):
            clamp_bbox(box, image_width=100, image_height=-1)

    def test_normalized_center_distance_zero_width(self):
        from src.ingest.geometry import normalized_center_distance

        with pytest.raises(ValueError, match="image_width must be positive"):
            normalized_center_distance(
                (0, 0), (10, 10), image_width=0, image_height=100,
            )

    def test_normalized_center_distance_zero_height(self):
        from src.ingest.geometry import normalized_center_distance

        with pytest.raises(ValueError, match="image_height must be positive"):
            normalized_center_distance(
                (0, 0), (10, 10), image_width=100, image_height=0,
            )

    def test_clamp_bbox_zero_dimensions(self):
        from src.ingest.geometry import PixelBoundingBox, clamp_bbox

        box = PixelBoundingBox(x1=10, y1=10, x2=50, y2=50)
        result = clamp_bbox(box, image_width=0, image_height=0)
        assert result.x1 == 0
        assert result.y1 == 0
        assert result.x2 == 0
        assert result.y2 == 0

    def test_padded_bbox_negative_result(self):
        from src.ingest.geometry import PixelBoundingBox, padded_bbox

        box = PixelBoundingBox(x1=5, y1=5, x2=10, y2=10)
        result = padded_bbox(box, padding_x=20, padding_y=20)
        assert result.x1 == -15
        assert result.y1 == -15
        assert result.x2 == 30
        assert result.y2 == 30


# ===========================================================================
# event_sink.py
# ===========================================================================


class TestEventSink:
    """Tests for event sink — TraceEventSink with run tree."""

    @pytest.fixture(autouse=True)
    def _isolate(self):
        reset_sinks()
        yield
        reset_sinks()

    def test_trace_event_sink_with_run_tree(self):
        from src.observability import tracing

        mock_run = MagicMock()
        mock_run.metadata = {}
        event = DomainEvent(
            type=EventType.SCAN_COMPLETED.value,
            run_id="r1",
            session_id="s1",
            payload={"count": 5},
        )
        with patch.object(tracing, "get_current_run_tree", return_value=mock_run):
            sink = TraceEventSink()
            sink.emit(event)
        assert "events" in mock_run.metadata
        assert len(mock_run.metadata["events"]) == 1
        assert mock_run.metadata["events"][0]["type"] == "SCAN_COMPLETED"

    def test_trace_event_sink_no_run(self):
        from src.observability import tracing

        event = DomainEvent(
            type=EventType.INGEST_COMPLETED.value,
            run_id="r1",
            session_id="s1",
            payload={},
        )
        with patch.object(tracing, "get_current_run_tree", return_value=None):
            sink = TraceEventSink()
            # Should not raise
            sink.emit(event)

    def test_trace_event_sink_appends_multiple(self):
        from src.observability import tracing

        mock_run = MagicMock()
        mock_run.metadata = {}
        with patch.object(tracing, "get_current_run_tree", return_value=mock_run):
            sink = TraceEventSink()
            sink.emit(DomainEvent(
                type="A", run_id="r", session_id="s", payload={},
            ))
            sink.emit(DomainEvent(
                type="B", run_id="r", session_id="s", payload={},
            ))
        assert len(mock_run.metadata["events"]) == 2

    def test_event_sink_base_not_implemented(self):
        sink = EventSink()
        event = DomainEvent(
            type="X", run_id="r", session_id="s", payload={},
        )
        with pytest.raises(NotImplementedError):
            sink.emit(event)

    def test_unregister_all_instances(self):
        custom = EventSink()
        register_sink(custom)
        register_sink(custom)
        unregister_sink(custom)
        assert custom not in get_sinks()

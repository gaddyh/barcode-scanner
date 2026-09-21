"""Tests for src.cli — the runtime-based ingest CLI.

Mocks ingest_one and tracing so no real Gemini/scan calls are made.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src import cli
from src.ingest import IngestStatus
from src.ingest.models import IngestResult, Issue, RunMetrics


def _make_result(status: IngestStatus = IngestStatus.COMPLETE) -> IngestResult:
    return IngestResult(
        status=status,
        items=[],
        missing=[],
        unassigned=[],
        issues=[],
        metrics=RunMetrics(
            scanner_count=0, vision_count=0, recovery_attempted=False,
        ),
    )


class TestPrintResult:
    def test_prints_json_compact(self, capsys) -> None:
        result = _make_result()
        cli._print_result(result, pretty=False, elapsed=1.5)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "complete"
        assert "wall time:" in captured.err

    def test_prints_json_pretty(self, capsys) -> None:
        result = _make_result()
        cli._print_result(result, pretty=True, elapsed=None)
        out = capsys.readouterr().out
        assert "\n" in out  # pretty-printed

    def test_prints_issues(self, capsys) -> None:
        result = _make_result()
        result.issues.append(Issue(
            code="test", message="something", severity="warning",
        ))
        cli._print_result(result, pretty=False, elapsed=None)
        err = capsys.readouterr().err
        assert "test:" in err
        assert "warning" in err


class TestRun:
    @pytest.mark.asyncio
    async def test_image_not_found(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        args = type("N", (), {
            "image": str(tmp_path / "missing.png"), "pretty": False,
        })()
        rc = await cli._run(args)
        assert rc == 1
        assert "not found" in capsys.readouterr().err

    @pytest.mark.asyncio
    async def test_success(self, tmp_path: Path, monkeypatch) -> None:
        img = tmp_path / "x.png"
        img.write_bytes(b"fake")
        result = _make_result(IngestStatus.COMPLETE)

        with patch("src.cli.ingest_one", new=AsyncMock(return_value=result)):
            with patch("src.cli.register_annotation_sink"):
                with patch("src.cli.is_tracing", return_value=False):
                    rc = await cli._run(
                        type("N", (), {"image": str(img), "pretty": False})()
                    )
        assert rc == 0

    @pytest.mark.asyncio
    async def test_exception_returns_1(
        self, tmp_path: Path, monkeypatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        img = tmp_path / "x.png"
        img.write_bytes(b"fake")

        with patch("src.cli.ingest_one", new=AsyncMock(side_effect=RuntimeError("boom"))):
            with patch("src.cli.register_annotation_sink"):
                with patch("src.cli.is_tracing", return_value=False):
                    rc = await cli._run(
                        type("N", (), {"image": str(img), "pretty": False})()
                    )
        assert rc == 1
        assert "Error:" in capsys.readouterr().err

    @pytest.mark.asyncio
    async def test_non_complete_returns_1(self, tmp_path: Path, monkeypatch) -> None:
        img = tmp_path / "x.png"
        img.write_bytes(b"fake")
        result = _make_result(IngestStatus.NEEDS_USER_INPUT)

        with patch("src.cli.ingest_one", new=AsyncMock(return_value=result)):
            with patch("src.cli.register_annotation_sink"):
                with patch("src.cli.is_tracing", return_value=False):
                    rc = await cli._run(
                        type("N", (), {"image": str(img), "pretty": False})()
                    )
        assert rc == 1


class TestFlushTracing:
    def test_no_tracing_returns(self) -> None:
        with patch("src.cli.is_tracing", return_value=False):
            cli._flush_tracing()  # should be a no-op

    def test_tracing_flushes(self) -> None:
        with patch("src.cli.is_tracing", return_value=True):
            with patch("langsmith.Client") as MockClient:
                instance = MockClient.return_value
                cli._flush_tracing()
                instance.flush.assert_called_once()

    def test_tracing_flush_error_suppressed(self, capsys) -> None:
        with patch("src.cli.is_tracing", return_value=True):
            with patch("langsmith.Client", side_effect=RuntimeError("nope")):
                cli._flush_tracing()
                assert "flush failed" in capsys.readouterr().err


class TestMain:
    def test_missing_image_arg(self) -> None:
        with pytest.raises(SystemExit):
            cli.main([])

    def test_image_arg_runs(self, tmp_path: Path, monkeypatch) -> None:
        img = tmp_path / "x.png"
        img.write_bytes(b"fake")

        async def _fake_run(args):
            return 0

        with patch("src.cli._run", new=_fake_run):
            rc = cli.main([str(img)])
        assert rc == 0

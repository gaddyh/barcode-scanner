"""Tests for the CLI (scan / audit / pipeline subcommands).

Mocks ``zxingcpp.read_barcodes`` and ``pipeline._traced_audit`` so no real
barcode images or Gemini API calls are required.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import zxingcpp
from PIL import Image

from app.cli import main
from src.ingest.vision import (
    AuditConfidence,
    SpatialLabelAuditPixels,
    SpatialLabelObservationPixels,
    SpatialLabelStatus,
)
from src.ingest.geometry import PixelBoundingBox
from tests._zxing_fake import make_read_result


def _first_call_mock(results):
    """Return a mock that yields ``results`` on the first call, then ``[]``."""
    calls = {"n": 0}

    def _mock(_img, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return results
        return []

    return _mock


def _png_path(tmp_path: Path, name: str = "img.png", width: int = 800, height: int = 600) -> Path:
    p = tmp_path / name
    Image.new("RGB", (width, height), (255, 255, 255)).save(p, format="PNG")
    return p


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


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------


def test_scan_command_outputs_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    img = _png_path(tmp_path)
    monkeypatch.setattr(
        zxingcpp, "read_barcodes",
        _first_call_mock([make_read_result("1234567890123")]),
    )

    rc = main(["scan", str(img), "--pretty"])
    assert rc == 0

    out = capsys.readouterr().out
    data = json.loads(out)
    assert isinstance(data, list)
    assert data[0]["status"] == "found"
    assert data[0]["count"] == 1
    assert data[0]["barcodes"][0]["value"] == "1234567890123"


def test_scan_command_no_barcodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    img = _png_path(tmp_path)
    monkeypatch.setattr(zxingcpp, "read_barcodes", lambda _img, **kwargs: [])

    rc = main(["scan", str(img)])
    assert rc == 0

    data = json.loads(capsys.readouterr().out)
    assert data[0]["status"] == "not_found"
    assert data[0]["count"] == 0


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------


def test_pipeline_command_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    img = _png_path(tmp_path)
    monkeypatch.setattr(
        zxingcpp, "read_barcodes",
        _first_call_mock([
            make_read_result("111"),
            make_read_result("222", x1=510, y1=110, x2=590, y2=290),
        ]),
    )

    spatial = SpatialLabelAuditPixels(
        image_width=800,
        image_height=600,
        labels=[
            _label_px(1, label_box=(50, 50, 250, 350), barcode_box=(100, 100, 200, 300)),
            _label_px(2, label_box=(450, 50, 650, 350), barcode_box=(500, 100, 600, 300)),
        ],
    )

    def _fake_audit(path, *, model, max_retries, retry_delay_seconds):
        return {"status": "ok", "spatial": spatial.model_dump(mode="json")}

    with patch("src.ingest.pipeline._traced_audit", side_effect=_fake_audit):
        rc = main(["pipeline", str(img), "--pretty"])

    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data[0]["scan_status"] == "found"
    assert data[0]["audit_status"] == "ok"
    assert data[0]["ok"] is True
    assert data[0]["decoded_vs_visible"]["all_labels_matched"] is True


def test_pipeline_command_audit_error_returns_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    img = _png_path(tmp_path)
    monkeypatch.setattr(
        zxingcpp, "read_barcodes",
        _first_call_mock([make_read_result("111")]),
    )

    def _fake_audit(path, *, model, max_retries, retry_delay_seconds):
        return {"status": "error", "error": {"type": "ShoeboxAuditError", "message": "boom"}}

    with patch("src.ingest.pipeline._traced_audit", side_effect=_fake_audit):
        rc = main(["pipeline", str(img)])

    assert rc == 1
    data = json.loads(capsys.readouterr().out)
    assert data[0]["audit_status"] == "error"
    assert data[0]["ok"] is False


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------


def test_no_subcommand_errors(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main([])


def test_scan_missing_file_arg() -> None:
    with pytest.raises(SystemExit):
        main(["scan"])

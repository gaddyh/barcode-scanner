"""Tests for src/ingest/service.py — ingest_one and _dict_to_ingest_result."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.ingest.models import IngestStatus
from src.ingest.service import _dict_to_ingest_result, ingest_one
from src.runtime.context import RunContext


def _ctx() -> RunContext:
    return RunContext(
        run_id="run-test",
        session_id="sess-test",
        source="test",
    )


def _png_path(tmp_path: Path, width: int = 800, height: int = 600) -> Path:
    p = tmp_path / "img.png"
    Image.new("RGB", (width, height), (255, 255, 255)).save(p, format="PNG")
    return p


def _raw_complete(found_count=2, missing_count=0, unassigned_count=0):
    return {
        "ok": True,
        "outcome": "complete",
        "audit_available": True,
        "image_width": 800,
        "image_height": 600,
        "found": [
            {
                "label_index": i,
                "barcode_value": f"VAL{i}",
                "barcode_format": "EAN_13",
                "barcode_bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
                "label_bbox": {"x1": 0, "y1": 0, "x2": 20, "y2": 20},
                "match_basis": "containment",
            }
            for i in range(found_count)
        ],
        "missing": [
            {"label_index": 100 + i, "label_bbox": {}, "barcode_bbox": {}}
            for i in range(missing_count)
        ],
        "unassigned": [
            {"barcode_value": f"UNASG{i}", "barcode_format": "EAN_13", "barcode_bbox": {}}
            for i in range(unassigned_count)
        ],
        "summary": {
            "visible_label_count": found_count + missing_count,
            "found_count": found_count,
            "missing_count": missing_count,
            "unassigned_count": unassigned_count,
            "all_found": missing_count == 0,
            "recovery": {
                "attempted": False,
                "labels_tried": 0,
                "barcodes_found": 0,
                "labels_resolved": 0,
            },
        },
    }


# ---------------------------------------------------------------------------
# _dict_to_ingest_result — status mapping
# ---------------------------------------------------------------------------


def test_service_dict_to_ingest_result_complete():
    result = _dict_to_ingest_result(_raw_complete(), 42, _ctx())
    assert result.status == IngestStatus.COMPLETE
    assert len(result.items) == 2
    assert result.metrics.elapsed_ms == 42


def test_service_dict_to_ingest_result_needs_better_photo():
    raw = {
        "ok": True,
        "outcome": "needs_better_photo",
        "audit_available": True,
        "found": [],
        "missing": [{"label_index": 0, "label_bbox": {}, "barcode_bbox": {}}],
        "unassigned": [],
        "summary": {
            "visible_label_count": 1,
            "found_count": 0,
            "missing_count": 1,
            "unassigned_count": 0,
            "recovery": {"attempted": False, "labels_tried": 0,
                         "barcodes_found": 0, "labels_resolved": 0},
        },
    }
    result = _dict_to_ingest_result(raw, 10, _ctx())
    assert result.status == IngestStatus.NEEDS_USER_INPUT


def test_service_dict_to_ingest_result_retryable_error_failed():
    raw = {
        "ok": False,
        "outcome": "retryable_error",
        "audit_available": False,
        "found": [],
        "missing": [],
        "unassigned": [],
        "summary": {"recovery": {"attempted": False, "labels_tried": 0,
                                "barcodes_found": 0, "labels_resolved": 0}},
        "error": {"code": "SCAN_FAILED", "message": "scanner error"},
    }
    result = _dict_to_ingest_result(raw, 5, _ctx())
    assert result.status == IngestStatus.FAILED
    assert result.issues[0].code == "BAD_IMAGE"


def test_service_dict_to_ingest_result_retryable_error_with_audit():
    raw = {
        "ok": False,
        "outcome": "retryable_error",
        "audit_available": True,
        "found": [],
        "missing": [],
        "unassigned": [],
        "summary": {"recovery": {"attempted": False, "labels_tried": 0,
                                "barcodes_found": 0, "labels_resolved": 0}},
        "error": {"code": "SCAN_FAILED", "message": "scanner error"},
    }
    result = _dict_to_ingest_result(raw, 5, _ctx())
    assert result.status == IngestStatus.FAILED
    assert result.issues[0].code == "PIPELINE_ERROR"


def test_service_dict_to_ingest_result_ok_true_unknown_needs_retry():
    raw = {"ok": True, "outcome": "unknown", "audit_available": False}
    result = _dict_to_ingest_result(raw, 1, _ctx())
    assert result.status == IngestStatus.NEEDS_RETRY


def test_service_dict_to_ingest_result_vision_scanner_mismatch():
    raw = _raw_complete()
    raw["summary"]["visible_label_count"] = 99  # mismatch
    result = _dict_to_ingest_result(raw, 1, _ctx())
    codes = [i.code for i in result.issues]
    assert "VISION_SCANNER_MISMATCH" in codes


def test_service_dict_to_ingest_result_barcode_missing_with_audit():
    raw = _raw_complete(found_count=0, missing_count=2)
    raw["outcome"] = "needs_better_photo"
    result = _dict_to_ingest_result(raw, 1, _ctx())
    codes = [i.code for i in result.issues]
    assert "BARCODE_MISSING" in codes


def test_service_dict_to_ingest_result_recovery_failed():
    raw = _raw_complete()
    raw["summary"]["recovery"] = {
        "attempted": True, "labels_tried": 3,
        "barcodes_found": 0, "labels_resolved": 0,
    }
    result = _dict_to_ingest_result(raw, 1, _ctx())
    codes = [i.code for i in result.issues]
    assert "RECOVERY_FAILED" in codes


def test_service_dict_to_ingest_result_recovery_succeeded_no_issue():
    raw = _raw_complete()
    raw["summary"]["recovery"] = {
        "attempted": True, "labels_tried": 3,
        "barcodes_found": 2, "labels_resolved": 2,
    }
    result = _dict_to_ingest_result(raw, 1, _ctx())
    codes = [i.code for i in result.issues]
    assert "RECOVERY_FAILED" not in codes


def test_service_dict_to_ingest_result_empty_dict():
    result = _dict_to_ingest_result({}, 0, _ctx())
    assert result.status == IngestStatus.NEEDS_RETRY


# ---------------------------------------------------------------------------
# ingest_one — end-to-end with mocked analyze_image
# ---------------------------------------------------------------------------


def test_ingest_one_complete(tmp_path: Path, monkeypatch):
    img_path = _png_path(tmp_path)

    def fake_analyze(_img, **kw):
        return _raw_complete()

    monkeypatch.setattr("src.ingest.service.analyze_image", fake_analyze)
    result = ingest_one(str(img_path), _ctx())
    assert result.status == IngestStatus.COMPLETE
    assert len(result.items) == 2


def test_ingest_one_needs_better_photo(tmp_path: Path, monkeypatch):
    img_path = _png_path(tmp_path)

    def fake_analyze(_img, **kw):
        raw = {
            "ok": True,
            "outcome": "needs_better_photo",
            "audit_available": True,
            "found": [],
            "missing": [{"label_index": 0, "label_bbox": {}, "barcode_bbox": {}}],
            "unassigned": [],
            "summary": {
                "visible_label_count": 1, "found_count": 0,
                "missing_count": 1, "unassigned_count": 0,
                "recovery": {"attempted": False, "labels_tried": 0,
                             "barcodes_found": 0, "labels_resolved": 0},
            },
        }
        return raw

    monkeypatch.setattr("src.ingest.service.analyze_image", fake_analyze)
    result = ingest_one(str(img_path), _ctx())
    assert result.status == IngestStatus.NEEDS_USER_INPUT


def test_ingest_one_retryable_error(tmp_path: Path, monkeypatch):
    img_path = _png_path(tmp_path)

    def fake_analyze(_img, **kw):
        return {
            "ok": False,
            "outcome": "retryable_error",
            "audit_available": False,
            "found": [],
            "missing": [],
            "unassigned": [],
            "summary": {"recovery": {"attempted": False, "labels_tried": 0,
                                    "barcodes_found": 0, "labels_resolved": 0}},
            "error": {"code": "SCAN_FAILED", "message": "scanner error"},
        }

    monkeypatch.setattr("src.ingest.service.analyze_image", fake_analyze)
    result = ingest_one(str(img_path), _ctx())
    assert result.status == IngestStatus.FAILED


def test_ingest_one_with_image_ref(tmp_path: Path, monkeypatch):
    img_path = _png_path(tmp_path)

    def fake_analyze(_img, **kw):
        return _raw_complete()

    monkeypatch.setattr("src.ingest.service.analyze_image", fake_analyze)
    result = ingest_one(str(img_path), _ctx(), image_ref="/tmp/img.png")
    assert result.status == IngestStatus.COMPLETE


def test_ingest_one_passes_kwargs_to_analyze(tmp_path: Path, monkeypatch):
    img_path = _png_path(tmp_path)
    captured = {}

    def fake_analyze(_img, **kw):
        captured.update(kw)
        return _raw_complete()

    monkeypatch.setattr("src.ingest.service.analyze_image", fake_analyze)
    ingest_one(
        str(img_path), _ctx(),
        model="gemini-test",
        max_retries=5,
        retry_delay_seconds=0.5,
    )
    assert captured["model"] == "gemini-test"
    assert captured["max_retries"] == 5
    assert captured["retry_delay_seconds"] == 0.5

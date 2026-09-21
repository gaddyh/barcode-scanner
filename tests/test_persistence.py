"""Tests for src/persistence.py — dict_to_ingest_result and run helpers."""

from __future__ import annotations

from src.ingest.models import IngestStatus
from src.persistence import (
    complete_run_from_dict,
    create_run,
    dict_to_ingest_result,
    fail_run,
)
from src.repository import NewRun, NoOpRunRepository


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


def _raw_needs_better_photo(missing_count=2):
    return {
        "ok": True,
        "outcome": "needs_better_photo",
        "audit_available": True,
        "image_width": 800,
        "image_height": 600,
        "found": [],
        "missing": [
            {"label_index": i, "label_bbox": {}, "barcode_bbox": {}}
            for i in range(missing_count)
        ],
        "unassigned": [],
        "summary": {
            "visible_label_count": missing_count,
            "found_count": 0,
            "missing_count": missing_count,
            "unassigned_count": 0,
            "all_found": False,
            "recovery": {
                "attempted": False,
                "labels_tried": 0,
                "barcodes_found": 0,
                "labels_resolved": 0,
            },
        },
    }


def _raw_retryable_error():
    return {
        "ok": False,
        "outcome": "retryable_error",
        "audit_available": False,
        "image_width": 0,
        "image_height": 0,
        "found": [],
        "missing": [],
        "unassigned": [],
        "summary": {
            "visible_label_count": 0,
            "found_count": 0,
            "missing_count": 0,
            "unassigned_count": 0,
            "all_found": False,
            "recovery": {
                "attempted": False,
                "labels_tried": 0,
                "barcodes_found": 0,
                "labels_resolved": 0,
            },
        },
        "error": {"code": "SCAN_FAILED", "message": "scanner error"},
    }


# ---------------------------------------------------------------------------
# dict_to_ingest_result — status mapping
# ---------------------------------------------------------------------------


def test_dict_to_ingest_result_complete():
    raw = _raw_complete()
    result = dict_to_ingest_result(raw, elapsed_ms=42)
    assert result.status == IngestStatus.COMPLETE
    assert len(result.items) == 2
    assert result.items[0].barcode_value == "VAL0"
    assert result.metrics.elapsed_ms == 42
    assert result.metrics.scanner_count == 2
    assert result.metrics.vision_count == 2
    assert result.audit_available is True


def test_dict_to_ingest_result_needs_better_photo():
    raw = _raw_needs_better_photo()
    result = dict_to_ingest_result(raw, elapsed_ms=10)
    assert result.status == IngestStatus.NEEDS_USER_INPUT
    assert len(result.missing) == 2
    assert len(result.items) == 0


def test_dict_to_ingest_result_retryable_error_with_audit():
    raw = _raw_retryable_error()
    raw["audit_available"] = True
    result = dict_to_ingest_result(raw, elapsed_ms=5)
    assert result.status == IngestStatus.FAILED
    assert len(result.issues) == 1
    assert result.issues[0].code == "SCAN_FAILED"


def test_dict_to_ingest_result_retryable_error_no_audit_bad_image():
    raw = _raw_retryable_error()
    raw["audit_available"] = False
    result = dict_to_ingest_result(raw, elapsed_ms=5)
    assert result.status == IngestStatus.FAILED
    assert result.issues[0].code == "BAD_IMAGE"


def test_dict_to_ingest_result_ok_true_unknown_outcome_needs_retry():
    """An unknown outcome with ok=True maps to NEEDS_RETRY."""
    raw = {"ok": True, "outcome": "unknown", "audit_available": False}
    result = dict_to_ingest_result(raw, elapsed_ms=1)
    assert result.status == IngestStatus.NEEDS_RETRY


def test_dict_to_ingest_result_ok_false_unknown_outcome_failed():
    """An unknown outcome with ok=False maps to FAILED."""
    raw = {"ok": False, "outcome": "unknown", "audit_available": False}
    result = dict_to_ingest_result(raw, elapsed_ms=1)
    assert result.status == IngestStatus.FAILED


# ---------------------------------------------------------------------------
# dict_to_ingest_result — issues
# ---------------------------------------------------------------------------


def test_dict_to_ingest_result_vision_scanner_mismatch():
    """When scanner_count != vision_count and audit is available, a
    VISION_SCANNER_MISMATCH issue is added."""
    raw = _raw_complete(found_count=2, missing_count=0)
    raw["summary"]["visible_label_count"] = 5  # mismatch
    result = dict_to_ingest_result(raw, elapsed_ms=1)
    codes = [i.code for i in result.issues]
    assert "VISION_SCANNER_MISMATCH" in codes


def test_dict_to_ingest_result_barcode_missing():
    """When audit is available and missing_count > 0, a BARCODE_MISSING issue
    is added."""
    raw = _raw_needs_better_photo(missing_count=3)
    result = dict_to_ingest_result(raw, elapsed_ms=1)
    codes = [i.code for i in result.issues]
    assert "BARCODE_MISSING" in codes


def test_dict_to_ingest_result_recovery_failed():
    """When recovery was attempted but resolved 0 labels, a RECOVERY_FAILED
    issue is added."""
    raw = _raw_complete()
    raw["summary"]["recovery"] = {
        "attempted": True,
        "labels_tried": 3,
        "barcodes_found": 0,
        "labels_resolved": 0,
    }
    result = dict_to_ingest_result(raw, elapsed_ms=1)
    codes = [i.code for i in result.issues]
    assert "RECOVERY_FAILED" in codes


def test_dict_to_ingest_result_recovery_succeeded_no_issue():
    """When recovery was attempted and resolved > 0 labels, no RECOVERY_FAILED
    issue is added."""
    raw = _raw_complete()
    raw["summary"]["recovery"] = {
        "attempted": True,
        "labels_tried": 3,
        "barcodes_found": 2,
        "labels_resolved": 2,
    }
    result = dict_to_ingest_result(raw, elapsed_ms=1)
    codes = [i.code for i in result.issues]
    assert "RECOVERY_FAILED" not in codes


def test_dict_to_ingest_result_no_audit_no_mismatch_issue():
    """Without audit_available, no VISION_SCANNER_MISMATCH issue is added even
    if counts differ."""
    raw = _raw_complete()
    raw["audit_available"] = False
    raw["summary"]["visible_label_count"] = 99
    result = dict_to_ingest_result(raw, elapsed_ms=1)
    codes = [i.code for i in result.issues]
    assert "VISION_SCANNER_MISMATCH" not in codes


def test_dict_to_ingest_result_barcode_missing_even_without_audit():
    """BARCODE_MISSING is added when missing_count > 0, regardless of audit."""
    raw = _raw_needs_better_photo()
    raw["audit_available"] = False
    result = dict_to_ingest_result(raw, elapsed_ms=1)
    codes = [i.code for i in result.issues]
    assert "BARCODE_MISSING" in codes


# ---------------------------------------------------------------------------
# dict_to_ingest_result — defaults
# ---------------------------------------------------------------------------


def test_dict_to_ingest_result_empty_dict():
    """An empty dict defaults to retryable_error / NEEDS_RETRY."""
    result = dict_to_ingest_result({}, elapsed_ms=0)
    assert result.status == IngestStatus.NEEDS_RETRY
    assert len(result.items) == 0
    assert len(result.issues) == 0
    assert result.metrics.scanner_count == 0


def test_dict_to_ingest_result_error_with_custom_code():
    """An error with a custom code is preserved in the issue."""
    raw = _raw_retryable_error()
    raw["audit_available"] = True
    raw["error"] = {"code": "CUSTOM_ERROR", "message": "custom"}
    result = dict_to_ingest_result(raw, elapsed_ms=1)
    assert result.issues[0].code == "CUSTOM_ERROR"


# ---------------------------------------------------------------------------
# create_run / complete_run_from_dict / fail_run
# ---------------------------------------------------------------------------


async def test_create_run_noop():
    repo = NoOpRunRepository()
    await create_run(
        repo,
        run_id="run-1",
        source="web",
        endpoint="/barcode/scan",
    )


async def test_create_run_with_all_fields():
    repo = NoOpRunRepository()
    await create_run(
        repo,
        run_id="run-1",
        source="web",
        endpoint="/barcode/scan",
        trace_id="trace-1",
        session_id="sess-1",
        filename="photo.jpg",
        image_ref="/tmp/photo.jpg",
        upload_bytes=1024,
        image_width=800,
        image_height=600,
        provider_message_id="msg-1",
        sender="user-1",
    )


async def test_complete_run_from_dict_noop():
    repo = NoOpRunRepository()
    raw = _raw_complete()
    await complete_run_from_dict(repo, "run-1", raw, elapsed_ms=42)


async def test_complete_run_from_dict_with_versions():
    repo = NoOpRunRepository()
    raw = _raw_complete()
    await complete_run_from_dict(
        repo, "run-1", raw, elapsed_ms=42,
        versions={"pipeline_version": "v1", "scanner_version": "v2"},
    )


async def test_fail_run_noop():
    repo = NoOpRunRepository()
    await fail_run(repo, "run-1", ValueError("boom"))


# ---------------------------------------------------------------------------
# NewRun DTO
# ---------------------------------------------------------------------------


def test_new_run_defaults():
    run = NewRun(id="r1", source="web")
    assert run.id == "r1"
    assert run.source == "web"
    assert run.session_id is None
    assert run.trace_id is None
    assert run.endpoint is None
    assert run.filename is None
    assert run.image_ref is None
    assert run.upload_bytes is None
    assert run.image_width is None
    assert run.image_height is None
    assert run.provider_message_id is None
    assert run.sender is None


def test_new_run_all_fields():
    run = NewRun(
        id="r1",
        source="web",
        session_id="s1",
        trace_id="t1",
        endpoint="/scan",
        filename="f.jpg",
        image_ref="/tmp/f.jpg",
        upload_bytes=100,
        image_width=800,
        image_height=600,
        provider_message_id="m1",
        sender="u1",
    )
    assert run.session_id == "s1"
    assert run.upload_bytes == 100

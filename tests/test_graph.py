"""Tests for the LangGraph-based pipeline orchestrator (M15A).

Two test families:

1. **Contract tests** — assert ``run_scan_graph()`` / ``pipeline_path()``
   returns a summary dict shape-identical to the old hand-written
   implementation, across happy / recovery / scan-error / audit-error cases.

2. **Parallel join / barrier test** — assert the Pregel barrier holds:
   ``reconcile`` never executes with only ``scan_result`` or only
   ``audit_result`` present. This is the explicit test for the implicit
   fan-out/join behavior that LangGraph provides via supersteps.

Mocks ``zxingcpp.read_barcodes`` and ``graph._traced_audit`` so no real
barcode images or Gemini API calls are required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from PIL import Image

import src.ingest.graph as _graph_module
from src.ingest.geometry import PixelBoundingBox
from src.ingest.graph import (
    ScanState,
    _reconcile_node,
    _route_after_reconcile,
    build_scan_graph,
    run_scan_graph,
)
from src.ingest.pipeline import pipeline_path
from src.ingest.scanner import (
    BoundingBox,
    DetectedBarcode,
    Point,
)
from src.ingest.vision import (
    AuditConfidence,
    SpatialLabelAuditPixels,
    SpatialLabelObservationPixels,
    SpatialLabelStatus,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _png_path(tmp_path: Path, width: int = 800, height: int = 600) -> Path:
    p = tmp_path / "img.png"
    Image.new("RGB", (width, height), (255, 255, 255)).save(p, format="PNG")
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


def _detection_dict(
    value: str,
    *,
    x1: int = 110,
    y1: int = 110,
    x2: int = 190,
    y2: int = 290,
    fmt: str = "Code128",
) -> dict:
    from dataclasses import asdict
    d = _detection(value, x1=x1, y1=y1, x2=x2, y2=y2, fmt=fmt)
    return asdict(d)


def _label_pixels(
    index: int,
    *,
    label_box: tuple[int, int, int, int],
    barcode_box: tuple[int, int, int, int] | None = None,
    status: SpatialLabelStatus = SpatialLabelStatus.CLEAR,
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
        status=status,
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
    """Fake scanner with configurable scan_bytes and scan_crop_with_recovery."""

    def __init__(
        self,
        detections: list[DetectedBarcode],
        recovery_detections: list[DetectedBarcode] | None = None,
    ) -> None:
        self._detections = detections
        self._recovery_detections = recovery_detections or []

    def scan_bytes(self, image_bytes: bytes) -> list[DetectedBarcode]:
        return self._detections

    def scan_crop_with_recovery(self, crop, *, offset_x=0, offset_y=0):
        return self._recovery_detections


def _patch_audit_ok(spatial: SpatialLabelAuditPixels):
    def _fake(path, *, model, max_retries, retry_delay_seconds):
        return {"status": "ok", "spatial": spatial.model_dump(mode="json")}
    return patch("src.ingest.graph._traced_audit", side_effect=_fake)


def _patch_audit_error(error: dict):
    def _fake(path, *, model, max_retries, retry_delay_seconds):
        return {"status": "error", "error": error}
    return patch("src.ingest.graph._traced_audit", side_effect=_fake)


# ---------------------------------------------------------------------------
# Contract tests: summary dict shape
# ---------------------------------------------------------------------------


# The set of keys the summary must have in every case. Specific cases add more.
_BASE_KEYS = {"path", "scan_status", "audit_status", "ok"}
_OK_KEYS = _BASE_KEYS | {
    "decoded_count", "unique_values", "unique_value_count", "scanner_detections",
    "visible_labels", "clear_labels", "gemini_labels",
    "recovery", "reconciliation", "decoded_vs_visible",
}


async def test_contract_complete_all_found(tmp_path: Path) -> None:
    """Happy path: scan + audit both ok, all labels matched → complete summary."""
    img = _png_path(tmp_path)
    detections = [
        _detection("111", x1=110, y1=110, x2=190, y2=290),
        _detection("222", x1=510, y1=110, x2=590, y2=290),
    ]
    spatial = _spatial([
        _label_pixels(1, label_box=(50, 50, 250, 350), barcode_box=(100, 100, 200, 300)),
        _label_pixels(2, label_box=(450, 50, 650, 350), barcode_box=(500, 100, 600, 300)),
    ])
    scanner = _FakeScanner(detections)

    with _patch_audit_ok(spatial):
        summary = await run_scan_graph(
            img, scanner, model=None, max_retries=0, retry_delay_seconds=0.0,
        )

    assert set(summary.keys()) == _OK_KEYS
    assert summary["ok"] is True
    assert summary["scan_status"] == "found"
    assert summary["audit_status"] == "ok"
    assert summary["decoded_count"] == 2
    assert summary["visible_labels"] == 2
    assert summary["recovery"]["attempted"] is False
    assert summary["recovery"]["labels_resolved"] == 0
    assert summary["decoded_vs_visible"]["all_labels_matched"] is True
    assert summary["decoded_vs_visible"]["match"] is True
    assert len(summary["reconciliation"]["matches"]) == 2
    assert len(summary["reconciliation"]["unmatched_labels"]) == 0


async def test_contract_recovery_cycle(tmp_path: Path) -> None:
    """Recovery: one label unmatched → recover → re-reconcile → resolved."""
    img = _png_path(tmp_path)
    # Scanner only finds 1 of 2 barcodes.
    detections = [_detection("111", x1=110, y1=110, x2=190, y2=290)]
    # Recovery finds the missing one.
    recovery_det = _detection("222", x1=510, y1=110, x2=590, y2=290)
    spatial = _spatial([
        _label_pixels(1, label_box=(50, 50, 250, 350), barcode_box=(100, 100, 200, 300)),
        _label_pixels(2, label_box=(450, 50, 650, 350), barcode_box=(500, 100, 600, 300)),
    ])
    scanner = _FakeScanner(detections, recovery_detections=[recovery_det])

    with _patch_audit_ok(spatial):
        summary = await run_scan_graph(
            img, scanner, model=None, max_retries=0, retry_delay_seconds=0.0,
        )

    assert summary["ok"] is True
    assert summary["recovery"]["attempted"] is True
    assert summary["recovery"]["labels_tried"] == 1
    assert summary["recovery"]["barcodes_found"] == 1
    assert summary["recovery"]["labels_resolved"] == 1
    assert summary["decoded_count"] == 2  # augmented after recovery
    assert summary["decoded_vs_visible"]["all_labels_matched"] is True
    assert len(summary["reconciliation"]["matches"]) == 2


async def test_contract_scan_error(tmp_path: Path) -> None:
    """Scan error → retryable_error summary, no reconciliation."""
    img = _png_path(tmp_path)
    spatial = _spatial([])

    # Scanner that raises to simulate invalid image.
    class _ErrorScanner:
        def scan_bytes(self, image_bytes: bytes):
            raise ValueError("bad image")
        def scan_crop_with_recovery(self, crop, *, offset_x=0, offset_y=0):
            return []

    with _patch_audit_ok(spatial):
        summary = await run_scan_graph(
            img, _ErrorScanner(), model=None, max_retries=0, retry_delay_seconds=0.0,
        )

    # Scan failed but audit ok: audit fields present, no reconciliation/recovery.
    assert set(summary.keys()) == _BASE_KEYS | {
        "scan_error", "visible_labels", "clear_labels", "gemini_labels",
    }
    assert summary["ok"] is False
    assert summary["scan_status"] == "error"
    assert summary["scan_error"]["code"] == "invalid_image"
    assert "reconciliation" not in summary
    assert "recovery" not in summary


async def test_contract_audit_error(tmp_path: Path) -> None:
    """Audit error → ok=False, no reconciliation, audit_error present."""
    img = _png_path(tmp_path)
    detections = [_detection("111", x1=110, y1=110, x2=190, y2=290)]
    scanner = _FakeScanner(detections)

    with _patch_audit_error({"type": "ShoeboxAuditError", "message": "boom"}):
        summary = await run_scan_graph(
            img, scanner, model=None, max_retries=0, retry_delay_seconds=0.0,
        )

    # scan_ok so scanner fields present; audit failed so no reconciliation.
    assert set(summary.keys()) == _BASE_KEYS | {
        "audit_error", "decoded_count", "unique_values",
        "unique_value_count", "scanner_detections",
    }
    assert summary["ok"] is False
    assert summary["scan_status"] == "found"
    assert summary["audit_status"] == "error"
    assert summary["audit_error"]["type"] == "ShoeboxAuditError"
    assert "reconciliation" not in summary
    assert "recovery" not in summary
    assert summary["scanner_detections"] is not None


def test_contract_pipeline_path_facade(tmp_path: Path) -> None:
    """pipeline_path() facade delegates to run_scan_graph and returns same shape."""
    img = _png_path(tmp_path)
    detections = [
        _detection("111", x1=110, y1=110, x2=190, y2=290),
        _detection("222", x1=510, y1=110, x2=590, y2=290),
    ]
    spatial = _spatial([
        _label_pixels(1, label_box=(50, 50, 250, 350), barcode_box=(100, 100, 200, 300)),
        _label_pixels(2, label_box=(450, 50, 650, 350), barcode_box=(500, 100, 600, 300)),
    ])
    scanner = _FakeScanner(detections)

    with _patch_audit_ok(spatial):
        summary = pipeline_path(
            img, scanner, model=None, max_retries=0, retry_delay_seconds=0.0,
        )

    assert set(summary.keys()) == _OK_KEYS
    assert summary["ok"] is True


# ---------------------------------------------------------------------------
# Parallel join / barrier test
# ---------------------------------------------------------------------------


async def test_parallel_join_barrier_reconcile_needs_both(tmp_path: Path) -> None:
    """The Pregel barrier must hold: reconcile cannot run with a single result.

    This test verifies the implicit fan-out/join behavior by instrumenting the
    reconcile node. We run the real compiled graph and assert that when
    ``_reconcile_node`` executes, both ``scan_result`` and ``audit_result``
    are present in state. If the barrier were broken, reconcile would see
    only one result and either crash or produce wrong output.
    """
    img = _png_path(tmp_path)
    detections = [
        _detection("111", x1=110, y1=110, x2=190, y2=290),
        _detection("222", x1=510, y1=110, x2=590, y2=290),
    ]
    spatial = _spatial([
        _label_pixels(1, label_box=(50, 50, 250, 350), barcode_box=(100, 100, 200, 300)),
        _label_pixels(2, label_box=(450, 50, 650, 350), barcode_box=(500, 100, 600, 300)),
    ])
    scanner = _FakeScanner(detections)

    # Track state at reconcile entry.
    reconcile_states: list[dict] = []
    original_reconcile = _reconcile_node

    async def _tracking_reconcile(state: ScanState) -> dict:
        reconcile_states.append(dict(state))
        return await original_reconcile(state)

    with _patch_audit_ok(spatial):
        with patch("src.ingest.graph._reconcile_node", _tracking_reconcile):
            _graph_module._invalidate_graph_cache()  # reset cache to pick up patch
            summary = await run_scan_graph(
                img, scanner, model=None, max_retries=0, retry_delay_seconds=0.0,
            )

    # Reconcile ran at least once (happy path: exactly once, no recovery).
    assert len(reconcile_states) >= 1

    # On every reconcile invocation, BOTH scan_result and audit_result must
    # be present — this is the barrier guarantee.
    for s in reconcile_states:
        assert "scan_result" in s, "reconcile ran before scan completed"
        assert "audit_result" in s, "reconcile ran before audit completed"
        assert s.get("scan_ok") is True
        assert s.get("audit_ok") is True

    # The summary must be correct (barrier didn't break the result).
    assert summary["ok"] is True
    assert summary["decoded_count"] == 2


async def test_parallel_join_barrier_recovery_cycle(tmp_path: Path) -> None:
    """The barrier holds across the recovery cycle too.

    When recovery triggers a second reconcile pass, both results must still be
    present. This verifies that the recover→reconcile edge doesn't lose state.
    """
    img = _png_path(tmp_path)
    detections = [_detection("111", x1=110, y1=110, x2=190, y2=290)]
    recovery_det = _detection("222", x1=510, y1=110, x2=590, y2=290)
    spatial = _spatial([
        _label_pixels(1, label_box=(50, 50, 250, 350), barcode_box=(100, 100, 200, 300)),
        _label_pixels(2, label_box=(450, 50, 650, 350), barcode_box=(500, 100, 600, 300)),
    ])
    scanner = _FakeScanner(detections, recovery_detections=[recovery_det])

    reconcile_states: list[dict] = []
    original_reconcile = _reconcile_node

    async def _tracking_reconcile(state: ScanState) -> dict:
        reconcile_states.append(dict(state))
        return await original_reconcile(state)

    with _patch_audit_ok(spatial):
        with patch("src.ingest.graph._reconcile_node", _tracking_reconcile):
            _graph_module._invalidate_graph_cache()  # reset cache to pick up patch
            summary = await run_scan_graph(
                img, scanner, model=None, max_retries=0, retry_delay_seconds=0.0,
            )

    # Recovery path: reconcile runs twice (initial + post-recovery).
    assert len(reconcile_states) == 2

    # Both passes have both results present.
    for s in reconcile_states:
        assert "scan_result" in s
        assert "audit_result" in s

    # First pass: barcodes come from scan_result (1 detection).
    # Second pass: barcodes are in state, augmented by recovery (2 detections).
    first_barcodes = reconcile_states[0].get("barcodes")
    if first_barcodes is None:
        first_barcodes = reconcile_states[0]["scan_result"].get("barcodes", [])
    assert len(first_barcodes) == 1
    assert len(reconcile_states[1]["barcodes"]) == 2

    assert summary["recovery"]["attempted"] is True
    assert summary["recovery"]["labels_resolved"] == 1


# ---------------------------------------------------------------------------
# Routing logic unit tests
# ---------------------------------------------------------------------------


def test_route_after_reconcile_no_unmatched() -> None:
    """No unmatched labels → finalize."""
    state: ScanState = {  # type: ignore[misc]
        "scan_ok": True,
        "audit_ok": True,
        "recovery_attempted": False,
        "reconciliation": type("R", (), {
            "unmatched_labels": [],
            "matched_label_count": 2,
        })(),
    }
    assert _route_after_reconcile(state) == "finalize"


def test_route_after_reconcile_has_unmatched() -> None:
    """Unmatched labels, no recovery yet → recover."""
    state: ScanState = {  # type: ignore[misc]
        "scan_ok": True,
        "audit_ok": True,
        "recovery_attempted": False,
        "reconciliation": type("R", (), {
            "unmatched_labels": [object()],
            "matched_label_count": 1,
        })(),
    }
    assert _route_after_reconcile(state) == "recover"


def test_route_after_reconcile_already_recovered() -> None:
    """Unmatched labels but recovery already attempted → finalize (no loop)."""
    state: ScanState = {  # type: ignore[misc]
        "scan_ok": True,
        "audit_ok": True,
        "recovery_attempted": True,
        "reconciliation": type("R", (), {
            "unmatched_labels": [object()],
            "matched_label_count": 1,
        })(),
    }
    assert _route_after_reconcile(state) == "finalize"


def test_route_after_reconcile_scan_failed() -> None:
    """Scan failed → finalize (skip reconcile/recover)."""
    state: ScanState = {  # type: ignore[misc]
        "scan_ok": False,
        "audit_ok": True,
        "recovery_attempted": False,
    }
    assert _route_after_reconcile(state) == "finalize"


def test_route_after_reconcile_audit_failed() -> None:
    """Audit failed → finalize (skip reconcile/recover)."""
    state: ScanState = {  # type: ignore[misc]
        "scan_ok": True,
        "audit_ok": False,
        "recovery_attempted": False,
    }
    assert _route_after_reconcile(state) == "finalize"


# ---------------------------------------------------------------------------
# Graph structure test
# ---------------------------------------------------------------------------


def test_graph_topology() -> None:
    """Verify the compiled graph has the expected nodes and edges."""
    graph = build_scan_graph()
    nodes = list(graph.get_graph().nodes)
    expected = {"__start__", "scan", "audit", "reconcile", "recover", "finalize", "__end__"}
    assert set(nodes) == expected

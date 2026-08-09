"""
graph.py

LangGraph-based orchestration for the barcode-scanning pipeline (M15A).

Replaces the hand-written ``ThreadPoolExecutor`` + conditional-recovery
branching in ``pipeline.py`` with an explicit ``StateGraph``. The node bodies
are the existing scanner / audit / reconciliation / recovery logic, moved
verbatim — no behavior change, no new features.

Graph topology::

    START ──┬──> scan ─────┐
            └──> audit ────┴──> reconcile ──> route_after_reconcile
                                                ├─ "recover" ──> recover ──> reconcile  (cycle)
                                                └─ "finalize" ──> finalize ──> END

- **scan** and **audit** run in parallel (LangGraph superstep fan-out).
- **reconcile** runs only after both complete (Pregel barrier — it cannot
  execute with a single result present).
- **route_after_reconcile** sends to **recover** when labels remain unmatched
  and recovery has not yet been attempted; otherwise to **finalize**.
- **recover** augments barcodes and edges back to **reconcile** for a second
  pass. Recovery runs at most once (``recovery_attempted`` guard).
- Error short-circuits: if scan or audit fails, ``reconcile`` is a no-op and
  ``route_after_reconcile`` sends straight to ``finalize``.

The summary dict returned by ``run_scan_graph()`` is shape-identical to the
old ``pipeline_path()`` return value — ``pipeline_path()`` is now a thin
facade that delegates here.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from PIL import Image, ImageOps, UnidentifiedImageError

from src.ingest.reconciliation import match_scanner_to_labels
from src.ingest.scanner import BarcodeScanner
from src.ingest.vision import (
    ShoeboxAuditError,
    audit_shoebox_labels_async,
)
from src.observability.tracing import emit_pipeline_event
from src.runtime.events import EventType

# langsmith is optional — tracing is enabled when LANGSMITH_TRACING=true.
_TRACING = os.getenv("LANGSMITH_TRACING", "").lower() in ("true", "1", "yes")
if _TRACING:
    import langsmith as ls
    from langsmith import traceable
else:
    # no-op decorator fallback when tracing is disabled.
    def traceable(*args, **kwargs):  # type: ignore[misc]
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def _wrap(fn):
            return fn

        return _wrap


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Recovery constants (moved from pipeline.py verbatim)
# ---------------------------------------------------------------------------

# Version of the recovery logic (crop padding, scan_crop_with_recovery
# variants, rotation attempt). Bumped when the recovery algorithm changes.
RECOVERY_VERSION = "recovery-v1"

# Padding ratio for recovery crops — wider than the label fallback (0.12) to
# give the scanner more context around the barcode region.
_RECOVERY_PADDING_RATIO = 0.20


# ---------------------------------------------------------------------------
# Shared helpers (moved from pipeline.py verbatim)
# ---------------------------------------------------------------------------


def _to_jsonable(value: object) -> object:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# Scan (moved from pipeline.py verbatim)
# ---------------------------------------------------------------------------


@traceable(run_type="tool", name="barcode_scan")
def scan_path(path: Path, scanner: BarcodeScanner) -> dict[str, object]:
    """Deterministic barcode scan of one image file."""
    try:
        image_bytes = path.read_bytes()
    except OSError as exc:
        return {
            "path": str(path),
            "status": "error",
            "error": {"code": "unreadable_file", "message": str(exc)},
        }

    try:
        barcodes = scanner.scan_bytes(image_bytes)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return {
            "path": str(path),
            "status": "error",
            "error": {"code": "invalid_image", "message": str(exc)},
        }

    return {
        "path": str(path),
        "status": "found" if barcodes else "not_found",
        "count": len(barcodes),
        "barcodes": [_to_jsonable(b) for b in barcodes],
    }


# ---------------------------------------------------------------------------
# Traced audit wrapper (moved from pipeline.py verbatim)
# ---------------------------------------------------------------------------


@traceable(run_type="tool", name="gemini_audit")
async def _traced_audit(
    path: Path,
    *,
    model: str | None,
    max_retries: int,
    retry_delay_seconds: float,
) -> dict[str, object]:
    """Traced async wrapper for the Gemini spatial label audit.

    Uses the native async google-genai client (``audit_shoebox_labels_async``)
    so the audit node runs without blocking the event loop.
    """
    try:
        spatial = await audit_shoebox_labels_async(
            path,
            model=model,
            max_retries=max_retries,
            retry_delay_seconds=retry_delay_seconds,
        )
    except FileNotFoundError as exc:
        return {
            "status": "error",
            "error": {"type": "FileNotFoundError", "message": str(exc)},
        }
    except (ValueError, ShoeboxAuditError) as exc:
        return {
            "status": "error",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
    return {"status": "ok", "spatial": spatial.model_dump(mode="json")}


# ---------------------------------------------------------------------------
# Gemini-guided recovery (moved from pipeline.py verbatim)
# ---------------------------------------------------------------------------


@traceable(run_type="chain", name="recovery")
def _gemini_guided_recovery(
    image_path: Path,
    scanner: BarcodeScanner,
    unmatched_labels: list,
    image_width: int,
    image_height: int,
) -> list[dict]:
    """Crop and scan unmatched label regions from the full-resolution image.

    For each unmatched Gemini label, extracts the ``barcode_bbox`` (or
    ``label_bbox`` as fallback) region with increased padding, and runs the
    aggressive crop-variant scanner including a 90° rotation attempt.

    Returns a list of detection dicts (same shape as ``scanner_detections``)
    suitable for merging into the existing detections list and re-running
    reconciliation.
    """
    try:
        source = Image.open(image_path)
        source = ImageOps.exif_transpose(source).convert("RGB")
    except Exception as exc:
        logger.warning("Recovery: could not open image %s: %s", image_path, exc)
        return []

    recovery_detections: list[dict] = []

    for ul in unmatched_labels:
        # Prefer barcode_bbox, fall back to label_bbox.
        bbox = ul.barcode_bbox or ul.label_bbox
        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            continue

        pad_x = max(20, round(w * _RECOVERY_PADDING_RATIO))
        pad_y = max(20, round(h * _RECOVERY_PADDING_RATIO))

        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(image_width, x2 + pad_x)
        cy2 = min(image_height, y2 + pad_y)

        crop = source.crop((cx1, cy1, cx2, cy2))

        logger.info(
            "Recovery: cropping label=%s bbox=(%d,%d,%d,%d) padded=(%d,%d,%d,%d) "
            "crop_size=%dx%d",
            ul.label_index,
            x1, y1, x2, y2,
            cx1, cy1, cx2, cy2,
            crop.width, crop.height,
        )

        detections = scanner.scan_crop_with_recovery(
            crop,
            offset_x=cx1,
            offset_y=cy1,
        )

        for det in detections:
            recovery_detections.append(
                {
                    "value": det.value,
                    "format": det.format,
                    "content_type": det.content_type,
                    "orientation": det.orientation,
                    "position": [
                        {"x": p.x, "y": p.y} for p in det.position
                    ],
                    "bounding_box": {
                        "x1": det.bounding_box.x1,
                        "y1": det.bounding_box.y1,
                        "x2": det.bounding_box.x2,
                        "y2": det.bounding_box.y2,
                    },
                }
            )

        if detections:
            logger.info(
                "Recovery: label=%s decoded %d barcode(s): %s",
                ul.label_index,
                len(detections),
                [d.value for d in detections],
            )

    return recovery_detections


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


class ScanState(TypedDict, total=False):
    """Graph state for the barcode scan pipeline.

    Each node returns a partial dict that LangGraph merges into the state.
    Parallel nodes (scan, audit) write to distinct keys so no reducer is
    needed — the Pregel barrier ensures ``reconcile`` only fires after both
    have completed.
    """

    # --- inputs (set once at graph invocation) ---
    # NOTE: ``scanner`` is intentionally NOT in state — it is a non-serializable
    # runtime object passed through ``config["configurable"]["scanner"]`` so the
    # Postgres checkpointer never tries to msgpack-encode it.
    path: str
    model: str | None
    max_retries: int
    retry_delay_seconds: float

    # --- scan node output ---
    scan_result: dict[str, object]
    scan_ok: bool

    # --- audit node output ---
    audit_result: dict[str, object]
    audit_ok: bool

    # --- reconcile node output ---
    barcodes: list[dict]
    labels: list[dict]
    image_width: int
    image_height: int
    reconciliation: Any  # ReconciliationResult (pydantic model)

    # --- recover node output ---
    recovery_attempted: bool
    recovery_labels_tried: int
    recovery_barcodes_found: int
    matched_before: int  # matched_label_count before recovery, for labels_resolved

    # --- finalize node output ---
    summary: dict[str, object]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


async def _scan_node(state: ScanState, config: RunnableConfig) -> dict[str, Any]:
    """Run the deterministic barcode scanner.

    ZXing is sync/CPU-bound, so it runs in a thread via ``asyncio.to_thread``
    to avoid blocking the event loop.

    The ``scanner`` is read from ``config["configurable"]["scanner"]`` (not
    state) because it is a non-serializable runtime object that the Postgres
    checkpointer cannot msgpack-encode.
    """
    import asyncio

    path = Path(state["path"])
    scanner = config["configurable"]["scanner"]
    scan_result = await asyncio.to_thread(scan_path, path, scanner)

    scan_ok = scan_result.get("status") in ("found", "not_found")

    # Log scanner detections for debugging (values + formats).
    scan_barcodes = scan_result.get("barcodes", [])
    if scan_barcodes:
        scan_values = [
            (b.get("value"), b.get("format")) for b in scan_barcodes  # type: ignore[union-attr]
        ]
        logger.info(
            "Scanner detected %d barcode(s): %s",
            len(scan_values),
            scan_values,
        )
    else:
        logger.info("Scanner detected 0 barcodes (status=%s)", scan_result.get("status"))

    emit_pipeline_event(
        EventType.SCAN_COMPLETED,
        scanner_count=scan_result.get("count", 0),
        scanner_status=scan_result.get("status"),
    )

    return {"scan_result": scan_result, "scan_ok": scan_ok}


async def _audit_node(state: ScanState) -> dict[str, Any]:
    """Run the Gemini spatial label audit (native async)."""
    path = Path(state["path"])
    audit_result = await _traced_audit(
        path,
        model=state.get("model"),
        max_retries=state.get("max_retries", 0),
        retry_delay_seconds=state.get("retry_delay_seconds", 0.0),
    )

    audit_ok = audit_result.get("status") == "ok"

    logger.info("Gemini audit status: %s", audit_result.get("status"))

    emit_pipeline_event(
        EventType.AUDIT_COMPLETED,
        vision_count=len(audit_result.get("spatial", {}).get("labels", [])) if audit_ok else 0,
        audit_status=audit_result.get("status"),
    )

    return {"audit_result": audit_result, "audit_ok": audit_ok}


async def _reconcile_node(state: ScanState) -> dict[str, Any]:
    """Match scanner detections to Gemini labels.

    On the first pass, barcodes are read from ``scan_result``. After recovery,
    barcodes are read from state (augmented by the recover node). This node is
    a no-op when scan or audit failed — ``route_after_reconcile`` sends
    directly to ``finalize`` in that case.
    """
    scan_ok = state.get("scan_ok", False)
    audit_ok = state.get("audit_ok", False)

    if not scan_ok or not audit_ok:
        return {}

    # Use augmented barcodes from state if recovery already ran; otherwise
    # initialize from the scan result.
    barcodes = state.get("barcodes")
    if barcodes is None:
        barcodes = state["scan_result"].get("barcodes", [])  # type: ignore[assignment]

    audit_result = state["audit_result"]
    spatial = audit_result.get("spatial", {})
    labels = spatial.get("labels", [])
    image_width = spatial["image_width"]
    image_height = spatial["image_height"]

    reconciliation = match_scanner_to_labels(
        barcodes,
        labels,
        image_width=image_width,
        image_height=image_height,
    )

    emit_pipeline_event(
        EventType.RECONCILIATION_COMPLETED,
        matched_count=reconciliation.matched_label_count,
        unmatched_count=len(reconciliation.unmatched_labels),
    )

    logger.info(
        "Reconciliation: %d matched, %d unmatched",
        len(reconciliation.matches),
        len(reconciliation.unmatched_labels),
    )
    for m in reconciliation.matches:
        logger.info(
            "  match: label=%s detection=%s barcode=%s basis=%s",
            m.label_index,
            m.scanner_detection_index,
            m.barcode_value,
            m.match_basis,
        )
    for u in reconciliation.unmatched_labels:
        logger.info(
            "  unmatched: label=%s status=%s",
            u.label_index,
            u.status,
        )

    return {
        "barcodes": barcodes,
        "labels": labels,
        "image_width": image_width,
        "image_height": image_height,
        "reconciliation": reconciliation,
    }


async def _recover_node(state: ScanState, config: RunnableConfig) -> dict[str, Any]:
    """Crop and aggressively scan unmatched label regions.

    Augments ``barcodes`` with any newly decoded detections. Sets
    ``recovery_attempted`` so ``route_after_reconcile`` does not loop again.

    Recovery is CPU-bound (OpenCV + ZXing), so it runs in a thread via
    ``asyncio.to_thread``.

    The ``scanner`` is read from ``config["configurable"]["scanner"]`` (not
    state) because it is a non-serializable runtime object that the Postgres
    checkpointer cannot msgpack-encode.
    """
    import asyncio

    reconciliation = state.get("reconciliation")
    if reconciliation is None:
        return {}

    unmatched_labels = reconciliation.unmatched_labels
    if not unmatched_labels:
        return {"recovery_attempted": True}

    image_width = state["image_width"]
    image_height = state["image_height"]
    path = Path(state["path"])
    scanner = config["configurable"]["scanner"]

    recovery_labels_tried = len(unmatched_labels)
    matched_before = reconciliation.matched_label_count

    emit_pipeline_event(
        EventType.RECOVERY_STARTED,
        labels_to_try=recovery_labels_tried,
    )

    recovery_detections = await asyncio.to_thread(
        _gemini_guided_recovery,
        path,
        scanner,
        unmatched_labels,
        image_width,
        image_height,
    )
    recovery_barcodes_found = len(recovery_detections)

    if recovery_detections:
        logger.info(
            "Recovery decoded %d additional barcode(s)",
            len(recovery_detections),
        )
        barcodes = list(state.get("barcodes", []))
        barcodes.extend(recovery_detections)
    else:
        barcodes = state.get("barcodes", [])

    emit_pipeline_event(
        EventType.RECOVERY_COMPLETED,
        barcodes_found=recovery_barcodes_found,
        labels_resolved=0,  # computed in finalize after second reconcile
    )

    return {
        "recovery_attempted": True,
        "recovery_labels_tried": recovery_labels_tried,
        "recovery_barcodes_found": recovery_barcodes_found,
        "matched_before": matched_before,
        "barcodes": barcodes,
    }


async def _finalize_node(state: ScanState) -> dict[str, Any]:
    """Build the summary dict — shape-identical to the old ``pipeline_path()``.

    Reads all node outputs from state and assembles the final summary. Recovery
    metrics (``labels_resolved``) are computed here by comparing the second
    reconciliation's matched count to ``matched_before``.
    """
    scan_result = state.get("scan_result", {})
    audit_result = state.get("audit_result", {})
    scan_ok = state.get("scan_ok", False)
    audit_ok = state.get("audit_ok", False)
    path = state["path"]

    summary: dict[str, object] = {
        "path": path,
        "scan_status": scan_result.get("status"),
        "audit_status": audit_result.get("status"),
    }

    barcodes: list[dict] = []
    if scan_ok:
        barcodes = state.get("barcodes")
        if barcodes is None:
            barcodes = scan_result.get("barcodes", [])  # type: ignore[assignment]
        values = [b["value"] for b in barcodes]  # type: ignore[index]
        summary["decoded_count"] = len(barcodes)
        summary["unique_values"] = sorted(set(values))
        summary["unique_value_count"] = len(set(values))
        summary["scanner_detections"] = barcodes

    if audit_ok:
        spatial = audit_result.get("spatial", {})
        labels = state.get("labels", spatial.get("labels", []))
        summary["visible_labels"] = len(labels)
        summary["clear_labels"] = sum(
            1 for label in labels if label.get("status") == "clear"
        )
        summary["gemini_labels"] = labels

    if scan_ok and audit_ok:
        reconciliation = state.get("reconciliation")
        if reconciliation is not None:
            recovery_attempted = state.get("recovery_attempted", False)
            recovery_labels_tried = state.get("recovery_labels_tried", 0)
            recovery_barcodes_found = state.get("recovery_barcodes_found", 0)
            matched_before = state.get("matched_before", reconciliation.matched_label_count)
            recovery_labels_resolved = (
                reconciliation.matched_label_count - matched_before
                if recovery_attempted
                else 0
            )

            # Stamp recovery metrics on the LangSmith trace.
            if _TRACING:
                run = ls.get_current_run_tree()
                if run is not None:
                    run.metadata.update(
                        {
                            "recovery_attempted": recovery_attempted,
                            "recovery_labels_tried": recovery_labels_tried,
                            "recovery_barcodes_found": recovery_barcodes_found,
                            "recovery_labels_resolved": recovery_labels_resolved,
                        }
                    )

            summary["recovery"] = {
                "attempted": recovery_attempted,
                "labels_tried": recovery_labels_tried,
                "barcodes_found": recovery_barcodes_found,
                "labels_resolved": recovery_labels_resolved,
            }
            summary["reconciliation"] = reconciliation.model_dump(mode="json")

            decoded = summary.get("decoded_count", 0)
            visible = summary.get("visible_labels", 0)
            summary["decoded_vs_visible"] = {
                "decoded": decoded,
                "visible": visible,
                "match": decoded == visible,
                "difference": decoded - visible,
                "matched_labels": reconciliation.matched_label_count,
                "all_labels_matched": reconciliation.all_labels_matched,
            }

    if not scan_ok:
        summary["scan_error"] = scan_result.get("error", {})
    if not audit_ok:
        summary["audit_error"] = audit_result.get("error", {})

    summary["ok"] = scan_ok and audit_ok
    return {"summary": summary}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def _route_after_reconcile(state: ScanState) -> str:
    """Decide whether to attempt recovery or finalize.

    Returns ``"recover"`` when:
    - both scan and audit succeeded, AND
    - labels remain unmatched after reconciliation, AND
    - recovery has not yet been attempted.

    Otherwise returns ``"finalize"``.
    """
    scan_ok = state.get("scan_ok", False)
    audit_ok = state.get("audit_ok", False)
    if not scan_ok or not audit_ok:
        return "finalize"

    reconciliation = state.get("reconciliation")
    if reconciliation is None:
        return "finalize"

    if reconciliation.unmatched_labels and not state.get("recovery_attempted", False):
        return "recover"
    return "finalize"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_scan_graph(checkpointer=None):
    """Build and compile the barcode scan StateGraph.

    Args:
        checkpointer: Optional ``BaseCheckpointSaver`` for persistence.
            When provided, the graph can be interrupted and resumed via
            ``thread_id`` in the config. When ``None`` (default), the graph
            runs without persistence.

    Returns a ``CompiledGraph`` ready for ``.ainvoke()``.
    """
    builder = StateGraph(ScanState)
    builder.add_node("scan", _scan_node)
    builder.add_node("audit", _audit_node)
    builder.add_node("reconcile", _reconcile_node)
    builder.add_node("recover", _recover_node)
    builder.add_node("finalize", _finalize_node)

    # Parallel fan-out: START → scan and START → audit.
    builder.add_edge(START, "scan")
    builder.add_edge(START, "audit")

    # Barrier: both scan and audit must complete before reconcile.
    builder.add_edge("scan", "reconcile")
    builder.add_edge("audit", "reconcile")

    # Conditional: recover (cycle back to reconcile) or finalize.
    builder.add_conditional_edges(
        "reconcile",
        _route_after_reconcile,
        {"recover": "recover", "finalize": "finalize"},
    )

    # Recovery augments barcodes, then re-reconciles.
    builder.add_edge("recover", "reconcile")

    # Finalize → END.
    builder.add_edge("finalize", END)

    compile_kwargs: dict[str, object] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
    return builder.compile(**compile_kwargs)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

# Cache the compiled graphs — stateless and reusable.
# Two variants: with and without checkpointer.
_compiled_graph_no_checkpoint = None
_compiled_graph_with_checkpoint = None


def _get_graph(use_checkpoint: bool = False):
    """Return a cached compiled graph.

    Args:
        use_checkpoint: If ``True``, return a graph with the Postgres
            checkpointer attached (for resume/interrupt support). If the
            checkpointer is not initialized, falls back to no-checkpoint.
    """
    global _compiled_graph_no_checkpoint, _compiled_graph_with_checkpoint

    if use_checkpoint:
        from src.ingest.checkpoint import get_checkpointer

        checkpointer = get_checkpointer()
        if checkpointer is None:
            # No checkpointer configured — fall back to plain graph.
            use_checkpoint = False
        else:
            if _compiled_graph_with_checkpoint is None:
                _compiled_graph_with_checkpoint = build_scan_graph(
                    checkpointer=checkpointer
                )
            return _compiled_graph_with_checkpoint

    if _compiled_graph_no_checkpoint is None:
        _compiled_graph_no_checkpoint = build_scan_graph()
    return _compiled_graph_no_checkpoint


def _invalidate_graph_cache() -> None:
    """Clear cached compiled graphs. Used by tests that patch nodes."""
    global _compiled_graph_no_checkpoint, _compiled_graph_with_checkpoint
    _compiled_graph_no_checkpoint = None
    _compiled_graph_with_checkpoint = None


async def run_scan_graph(
    path: Path,
    scanner: BarcodeScanner,
    *,
    model: str | None,
    max_retries: int,
    retry_delay_seconds: float,
    thread_id: str | None = None,
) -> dict[str, object]:
    """Run the LangGraph scan pipeline (async) and return the summary dict.

    The returned dict is shape-identical to the old ``pipeline_path()``
    return value. This is the canonical async entry point;
    ``pipeline_path()`` is a thin traced facade that bridges via
    ``asyncio.run()`` for sync callers.

    Args:
        thread_id: Optional unique ID for checkpoint persistence. When
            provided and a checkpointer is configured, the graph state is
            saved to Postgres after every superstep. This enables resume
            after interruption (M15D HITL). When ``None`` or when no
            checkpointer is configured, the graph runs without persistence.
    """
    initial_state: ScanState = {  # type: ignore[misc]
        "path": str(path),
        "model": model,
        "max_retries": max_retries,
        "retry_delay_seconds": retry_delay_seconds,
    }

    use_checkpoint = thread_id is not None
    graph = _get_graph(use_checkpoint=use_checkpoint)

    # The scanner is a non-serializable runtime object — pass it through the
    # RunnableConfig (not state) so the Postgres checkpointer never tries to
    # msgpack-encode it. Nodes read it via config["configurable"]["scanner"].
    config: dict[str, object] = {"configurable": {"scanner": scanner}}
    if use_checkpoint and thread_id is not None:
        config["configurable"]["thread_id"] = thread_id

    final_state = await graph.ainvoke(initial_state, config=config)
    return final_state["summary"]

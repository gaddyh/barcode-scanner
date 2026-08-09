"""
pipeline.py

Thin facade over the LangGraph-based pipeline orchestrator (M15A/M15B).

The actual orchestration — parallel scan + audit, containment reconciliation,
Gemini-guided recovery, and summary assembly — now lives in
``src.ingest.graph`` as an explicit async ``StateGraph``. This module
preserves the public API (``pipeline_path``, ``scan_path``, ``_traced_audit``,
``RECOVERY_VERSION``) so existing consumers (``analyze.py``, ``cli_app``,
``observability.versions``, tests) continue to work unchanged.

``pipeline_path()`` delegates to the async ``run_scan_graph()`` and bridges
via ``asyncio.run()`` for sync callers. The FastAPI route can call
``run_scan_graph()`` directly to stay fully async.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from src.ingest.graph import (
    RECOVERY_VERSION,
    _traced_audit,
    run_scan_graph,
    scan_path,
)
from src.ingest.scanner import SCANNER_VERSION, BarcodeScanner
from src.ingest.vision import DEFAULT_MODEL, VISION_PROMPT_VERSION

# Load .env early so LANGSMITH_* vars are set before langsmith is imported.
load_dotenv()

logger = logging.getLogger(__name__)

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


@traceable(run_type="chain", name="pipeline")
def pipeline_path(
    path: Path,
    scanner: BarcodeScanner,
    *,
    model: str | None,
    max_retries: int,
    retry_delay_seconds: float,
    thread_id: str | None = None,
) -> dict[str, object]:
    """Run deterministic scan and Gemini audit in parallel, return combined summary.

    Delegates to the async LangGraph orchestrator (``run_scan_graph``) and
    bridges via ``asyncio.run()`` for sync callers (CLI, eval). The summary
    is reshaped by ``analyze_image`` into the product response.

    Stamps component versions on the pipeline span for LangSmith tracing.

    Args:
        thread_id: Optional unique ID for checkpoint persistence (M15C).
    """
    # Stamp component versions on the pipeline span.
    _run = ls.get_current_run_tree() if _TRACING else None
    if _run is not None:
        _run.metadata.update(
            {
                "scanner_version": SCANNER_VERSION,
                "vision_prompt_version": VISION_PROMPT_VERSION,
                "vision_model": model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
                "recovery_version": RECOVERY_VERSION,
            }
        )

    return asyncio.run(
        run_scan_graph(
            path,
            scanner,
            model=model,
            max_retries=max_retries,
            retry_delay_seconds=retry_delay_seconds,
            thread_id=thread_id,
        )
    )


__all__ = [
    "pipeline_path",
    "scan_path",
    "_traced_audit",
    "RECOVERY_VERSION",
]

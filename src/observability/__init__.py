"""Observability package — centralized LangSmith tracing, metadata, and events."""

from src.observability.tracing import (
    PIPELINE_VERSION,
    attach_image_to_run,
    emit_event,
    emit_metadata,
    get_current_run_tree,
    is_tracing,
    trace_operation,
)

__all__ = [
    "trace_operation",
    "emit_metadata",
    "emit_event",
    "attach_image_to_run",
    "get_current_run_tree",
    "is_tracing",
    "PIPELINE_VERSION",
]

"""Observability package — centralized LangSmith tracing, metadata, events, sinks, and versions."""

from src.observability.event_sink import (
    EventSink,
    TraceEventSink,
    emit_to_sinks,
    get_sinks,
    register_sink,
    reset_sinks,
    unregister_sink,
)
from src.observability.tracing import (
    PIPELINE_VERSION,
    attach_image_to_run,
    emit_event,
    emit_metadata,
    emit_pipeline_event,
    get_current_run_tree,
    is_tracing,
    trace_operation,
)
from src.observability.versions import RunVersions, collect_versions

__all__ = [
    "trace_operation",
    "emit_metadata",
    "emit_event",
    "emit_pipeline_event",
    "attach_image_to_run",
    "get_current_run_tree",
    "is_tracing",
    "PIPELINE_VERSION",
    "RunVersions",
    "collect_versions",
    "EventSink",
    "TraceEventSink",
    "register_sink",
    "unregister_sink",
    "reset_sinks",
    "get_sinks",
    "emit_to_sinks",
]

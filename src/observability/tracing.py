"""Centralized LangSmith tracing utilities.

This module replaces the scattered ``@ls.traceable`` + manual
``run.metadata.update()`` pattern duplicated across ``routes.py``,
``main.py``, and ``pipeline.py``.

LangSmith is optional — when ``LANGSMITH_TRACING`` is not set, all
functions degrade to no-ops. This mirrors the pattern in ``pipeline.py``
but centralizes it in one place.

Key design decisions:
- ``RunContext.run_id`` is an application correlation ID, NOT the
  LangSmith run ID. It is added to trace metadata as ``run_id``.
  LangSmith owns its internal run IDs.
- ``emit_metadata()`` always includes ``run_id``, ``session_id``, and
  ``source`` from the context, plus any caller-provided kwargs.
- ``emit_event()`` logs the event and appends it to trace metadata.
"""

from __future__ import annotations

import contextvars
import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

from dotenv import load_dotenv

if TYPE_CHECKING:
    from src.runtime.context import RunContext
    from src.runtime.events import DomainEvent, EventType

load_dotenv()

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Context vars for correlation IDs — propagate to child spans and worker
# threads via copy_context(). Set by emit_metadata() so that
# emit_pipeline_event() can read them inside pipeline.py's child spans,
# where get_current_run_tree() returns a child that doesn't inherit the
# parent's metadata.
_run_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("_run_id_var", default="")
_session_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("_session_id_var", default="")

# langsmith is optional — tracing is enabled when LANGSMITH_TRACING=true.
_TRACING = os.getenv("LANGSMITH_TRACING", "").lower() in ("true", "1", "yes")

if _TRACING:
    import langsmith as ls
    from langsmith import traceable
    from langsmith.schemas import Attachment

    def get_current_run_tree() -> Any:
        return ls.get_current_run_tree()

else:
    # no-op decorator fallback when tracing is disabled.
    def traceable(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def _wrap(fn: Callable[..., T]) -> Callable[..., T]:
            return fn

        return _wrap

    def get_current_run_tree() -> Any:
        return None

    class Attachment:  # type: ignore[no-redef]
        def __init__(self, *, mime_type: str, data: bytes) -> None:
            self.mime_type = mime_type
            self.data = data


# ---------------------------------------------------------------------------
# Pipeline version — stamped on every trace for A/B comparison.
# ---------------------------------------------------------------------------

PIPELINE_VERSION = "ingest-v1"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def trace_operation(
    name: str,
    run_type: str = "chain",
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Return a ``@traceable`` decorator with the given configuration.

    When tracing is disabled, returns a no-op decorator.
    """
    return traceable(  # type: ignore[return-value]
        name=name,
        run_type=run_type,
        tags=tags or [],
        metadata=metadata or {},
    )


def emit_metadata(context: RunContext, **kwargs: Any) -> None:
    """Update the current LangSmith run's metadata.

    Always includes ``run_id``, ``session_id``, and ``source`` from the
    context (if not already present in kwargs), plus any caller-provided
    key-value pairs.

    Also sets context vars for ``run_id`` and ``session_id`` so that
    ``emit_pipeline_event()`` can read them inside child spans and worker
    threads (where ``get_current_run_tree()`` returns a child run that
    doesn't inherit the parent's metadata).

    No-op for the run tree update when tracing is disabled, but context
    vars are always set (they're cheap and useful even without tracing).
    """
    _run_id_var.set(context.run_id)
    _session_id_var.set(context.session_id)

    run = get_current_run_tree()
    if run is None:
        return

    meta: dict[str, Any] = dict(kwargs)
    meta.setdefault("run_id", context.run_id)
    meta.setdefault("session_id", context.session_id)
    meta.setdefault("source", context.source)
    meta.setdefault("pipeline_version", PIPELINE_VERSION)
    run.metadata.update(meta)


def emit_event(event: DomainEvent) -> None:
    """Emit a domain event to all registered sinks.

    Delegates to ``emit_to_sinks()`` in ``event_sink.py``. The default
    ``TraceEventSink`` logs the event and appends it to the LangSmith
    trace metadata. Custom sinks (annotation, metrics) can be registered
    via ``register_sink()``.
    """
    from src.observability.event_sink import emit_to_sinks

    emit_to_sinks(event)


def emit_pipeline_event(event_type: EventType, **payload: Any) -> None:
    """Emit a pipeline event using run_id/session_id from the current run tree.

    This avoids passing ``RunContext`` through ``pipeline_path()`` — the
    correlation IDs are read from the current LangSmith run's metadata,
    which was stamped by ``execute()`` before the pipeline started. Works
    in worker threads because ``copy_context()`` propagates the run tree.

    Args:
        event_type: An ``EventType`` enum value (not a raw string).
        **payload: Event-specific key-value data.
    """
    run = get_current_run_tree()
    # Prefer run tree metadata (set by emit_metadata on the parent span),
    # fall back to context vars (propagated to child spans and threads).
    run_id = ""
    session_id = ""
    if run is not None:
        run_id = run.metadata.get("run_id", "")
        session_id = run.metadata.get("session_id", "")
    if not run_id:
        run_id = _run_id_var.get("")
    if not session_id:
        session_id = _session_id_var.get("")
    from src.runtime.events import DomainEvent as _DomainEvent

    emit_event(_DomainEvent(
        type=event_type.value,
        run_id=run_id,
        session_id=session_id,
        payload=payload,
    ))


def attach_image_to_run(image_bytes: bytes, mime_type: str) -> None:
    """Attach an image to the current LangSmith run as a viewable attachment.

    No-op when tracing is disabled.
    """
    run = get_current_run_tree()
    if run is None:
        return
    run.attachments = {
        "uploaded_image": Attachment(mime_type=mime_type, data=image_bytes)
    }


def is_tracing() -> bool:
    """Return True if LangSmith tracing is enabled."""
    return _TRACING


def push_feedback(feedback: list[dict[str, Any]]) -> None:
    """Push feedback scores to the current LangSmith run.

    Each feedback dict must have ``key``, ``score``, and optionally
    ``comment``. No-op when tracing is disabled.

    Args:
        feedback: List of feedback dicts from ``evaluate_production_run()``.
    """
    run = get_current_run_tree()
    if run is None:
        return
    from langsmith import Client
    client = Client()
    for fb in feedback:
        client.create_feedback(
            run_id=run.id,
            key=fb["key"],
            score=fb["score"],
            comment=fb.get("comment"),
        )

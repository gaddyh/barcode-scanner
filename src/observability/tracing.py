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

import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

from dotenv import load_dotenv

if TYPE_CHECKING:
    from src.runtime.context import RunContext
    from src.runtime.events import DomainEvent

load_dotenv()

logger = logging.getLogger(__name__)

T = TypeVar("T")

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

    No-op when tracing is disabled.
    """
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
    """Emit a domain event to structured log and LangSmith trace metadata.

    Always logs (even when tracing is disabled). When tracing is enabled,
    also appends the event to the current run's metadata under ``events``.
    """
    logger.info(
        "DomainEvent type=%s run_id=%s session_id=%s payload_keys=%s",
        event.type,
        event.run_id,
        event.session_id,
        list(event.payload.keys()),
    )
    run = get_current_run_tree()
    if run is not None:
        events: list = run.metadata.setdefault("events", [])
        events.append(event.model_dump(mode="json"))


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

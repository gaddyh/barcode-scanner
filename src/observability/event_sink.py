"""Event sink abstraction — pluggable consumers of the domain event stream.

The pipeline emits ``DomainEvent``s at every transition. Each event is
routed to all registered sinks. The default ``TraceEventSink`` logs the
event and appends it to the LangSmith trace metadata — this preserves
the existing behavior of ``emit_event()``.

Future milestones register additional sinks without modifying pipeline code:

- M8 (annotation queue): ``AnnotationCandidateSink`` writes interesting
  failures to SQLite.
- M9 (monitoring): ``MetricsSink`` aggregates events for dashboards.

Usage::

    from src.observability.event_sink import register_sink, reset_sinks

    # Register a custom sink
    register_sink(my_sink)

    # In tests, reset to default after each test
    reset_sinks()
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.runtime.events import DomainEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class EventSink:
    """Base class / protocol for event sinks.

    Subclasses implement ``emit()``. Using a base class instead of
    ``Protocol`` keeps runtime ``isinstance`` checks simple and avoids
    ``runtime_checkable`` overhead.
    """

    def emit(self, event: DomainEvent) -> None:  # noqa: D401
        """Process one domain event."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Default sink — logs + appends to LangSmith trace metadata
# ---------------------------------------------------------------------------


class TraceEventSink(EventSink):
    """Default sink — logs the event and appends it to the LangSmith trace.

    This is the behavior that ``emit_event()`` in ``tracing.py`` had before
    the sink abstraction was introduced. It is always registered by default.
    """

    def emit(self, event: DomainEvent) -> None:
        logger.info(
            "DomainEvent type=%s run_id=%s session_id=%s payload_keys=%s",
            event.type,
            event.run_id,
            event.session_id,
            list(event.payload.keys()),
        )
        # Import here to avoid circular dependency at module load time.
        from src.observability.tracing import get_current_run_tree

        run = get_current_run_tree()
        if run is not None:
            events: list = run.metadata.setdefault("events", [])
            events.append(event.model_dump(mode="json"))


# ---------------------------------------------------------------------------
# Sink registry
# ---------------------------------------------------------------------------

# Global sink list — starts with the default TraceEventSink.
_sinks: list[EventSink] = [TraceEventSink()]


def register_sink(sink: EventSink) -> None:
    """Register an additional event sink."""
    _sinks.append(sink)


def unregister_sink(sink: EventSink) -> None:
    """Remove a previously registered sink.

    No-op if the sink is not registered. Useful for test cleanup.
    """
    while sink in _sinks:
        _sinks.remove(sink)


def reset_sinks() -> None:
    """Reset the sink list to the default ``[TraceEventSink()]``.

    Intended for test isolation — call in ``teardown`` or ``fixture``
    to ensure one test's custom sinks don't leak into the next.
    """
    _sinks.clear()
    _sinks.append(TraceEventSink())


def get_sinks() -> list[EventSink]:
    """Return the current sink list (for inspection/testing)."""
    return list(_sinks)


def emit_to_sinks(event: DomainEvent) -> None:
    """Route one event to all registered sinks."""
    for sink in _sinks:
        sink.emit(event)

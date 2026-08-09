"""Tests for the event sink abstraction — registration, routing, isolation.

No LangSmith or Gemini calls. Verifies that:
- The default TraceEventSink is registered on load
- Custom sinks receive events via emit_to_sinks()
- register_sink / unregister_sink / reset_sinks work correctly
- emit_event() routes through the sink list
- reset_sinks() provides test isolation
"""

from __future__ import annotations

import pytest

from src.observability.event_sink import (
    EventSink,
    TraceEventSink,
    emit_to_sinks,
    get_sinks,
    register_sink,
    reset_sinks,
    unregister_sink,
)
from src.observability.tracing import emit_event
from src.runtime.events import DomainEvent, EventType


@pytest.fixture(autouse=True)
def _isolate_sinks():
    """Reset sinks before and after each test."""
    reset_sinks()
    yield
    reset_sinks()


class _RecordingSink(EventSink):
    """Test sink that records all events it receives."""

    def __init__(self) -> None:
        self.events: list[DomainEvent] = []

    def emit(self, event: DomainEvent) -> None:
        self.events.append(event)


def _make_event(event_type: EventType = EventType.INGEST_COMPLETED) -> DomainEvent:
    return DomainEvent(
        type=event_type.value,
        run_id="test-run-id",
        session_id="test-session-id",
        payload={"key": "value"},
    )


# ---------------------------------------------------------------------------
# Default sink
# ---------------------------------------------------------------------------


def test_default_sink_is_trace_event_sink() -> None:
    sinks = get_sinks()
    assert len(sinks) == 1
    assert isinstance(sinks[0], TraceEventSink)


def test_reset_sinks_restores_default() -> None:
    custom = _RecordingSink()
    register_sink(custom)
    assert len(get_sinks()) == 2
    reset_sinks()
    sinks = get_sinks()
    assert len(sinks) == 1
    assert isinstance(sinks[0], TraceEventSink)


# ---------------------------------------------------------------------------
# Registration / unregistration
# ---------------------------------------------------------------------------


def test_register_sink_adds_to_list() -> None:
    custom = _RecordingSink()
    register_sink(custom)
    assert custom in get_sinks()


def test_unregister_sink_removes_from_list() -> None:
    custom = _RecordingSink()
    register_sink(custom)
    unregister_sink(custom)
    assert custom not in get_sinks()


def test_unregister_sink_not_registered_is_noop() -> None:
    custom = _RecordingSink()
    # Should not raise
    unregister_sink(custom)


def test_register_multiple_sinks() -> None:
    first = _RecordingSink()
    second = _RecordingSink()
    register_sink(first)
    register_sink(second)
    assert len(get_sinks()) == 3  # default + 2


# ---------------------------------------------------------------------------
# Event routing
# ---------------------------------------------------------------------------


def test_emit_to_sinks_routes_to_all_registered() -> None:
    first = _RecordingSink()
    second = _RecordingSink()
    register_sink(first)
    register_sink(second)

    event = _make_event()
    emit_to_sinks(event)

    assert first.events == [event]
    assert second.events == [event]


def test_emit_event_routes_through_sinks() -> None:
    custom = _RecordingSink()
    register_sink(custom)

    event = _make_event(EventType.SCAN_COMPLETED)
    emit_event(event)

    assert len(custom.events) == 1
    assert custom.events[0].type == EventType.SCAN_COMPLETED.value


def test_emit_event_with_user_retry_requested() -> None:
    custom = _RecordingSink()
    register_sink(custom)

    event = _make_event(EventType.USER_RETRY_REQUESTED)
    emit_event(event)

    assert custom.events[0].type == EventType.USER_RETRY_REQUESTED.value


# ---------------------------------------------------------------------------
# Test isolation
# ---------------------------------------------------------------------------


def test_sink_from_one_test_does_not_leak_to_next() -> None:
    # This test registers a sink. If isolation works, the next test
    # should not see it.
    custom = _RecordingSink()
    register_sink(custom)
    assert len(get_sinks()) == 2


def test_previous_test_sink_is_gone() -> None:
    # If the autouse fixture worked, we're back to 1 sink.
    sinks = get_sinks()
    assert len(sinks) == 1
    assert isinstance(sinks[0], TraceEventSink)

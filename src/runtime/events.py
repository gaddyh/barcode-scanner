"""DomainEvent — structured events emitted during pipeline execution.

Events are the common currency for monitoring, alerting, audit history,
and async workers. Initially they go to structured log + LangSmith trace
metadata. Later they can power dashboards, alerts, or a queue without
changing the emission sites.

Example event types::

    IMAGE_RECEIVED
    SCAN_COMPLETED
    RECONCILIATION_FAILED
    RECOVERY_STARTED
    RECOVERY_SUCCEEDED
    USER_RETRY_REQUESTED
    SESSION_COMPLETED
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class DomainEvent(BaseModel):
    """A structured event emitted during pipeline execution."""

    type: str
    run_id: str
    session_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    payload: dict[str, Any] = Field(default_factory=dict)

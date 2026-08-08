"""RunContext — the application correlation identity for every operation.

``run_id`` is an application-level correlation ID, NOT the LangSmith run ID.
It is added to LangSmith metadata as ``run_id`` so traces can be correlated
with external systems (logs, databases, user feedback). LangSmith owns its
internal run IDs. This keeps the runtime identity model decoupled from any
single observability provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RunContext:
    """Immutable execution context carried through every runtime operation.

    Attributes:
        run_id: Application correlation ID (UUID4). Not the LangSmith run ID.
        session_id: ULID grouping related operations (one photo = one session).
        user_id: Sender phone (WhatsApp) or None (web/cli/eval).
        source: Ingress channel — "web", "whatsapp", "cli", or "eval".
        metadata: Extensible per-operation metadata.
    """

    run_id: str
    session_id: str
    user_id: str | None = None
    source: str = "cli"
    metadata: dict[str, Any] = field(default_factory=dict)

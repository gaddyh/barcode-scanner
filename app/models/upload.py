"""Upload identity — stable, time-sortable IDs for every uploaded image.

Every image that enters the system (web upload or WhatsApp) gets a ``upload_id``
(a ULID — 26-char Crockford base32, lexicographically sortable by timestamp).
This ID is the join key for:

- LangSmith trace metadata
- user feedback ("this scan is wrong")
- annotation queue entries (future)
- offline dataset examples (future)

``source`` distinguishes the ingress channel: ``"web"`` or ``"whatsapp"``.
"""

from __future__ import annotations

import ulid


def generate_upload_id() -> str:
    """Return a fresh 26-char ULID string (e.g. ``01J...``)."""
    return str(ulid.ULID())

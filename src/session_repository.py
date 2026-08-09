"""Postgres repository for multi-image ingest sessions.

CRUD operations for sessions, session_items (confirmed barcodes), and
session_missing (unresolved boxes). Uses the existing asyncpg pool.

The repository is the persistence boundary — it does not contain business
logic. The SessionGraph (M16B) calls these methods to load/save session
state between images.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC
from typing import Any

import asyncpg

from src.ingest.session_models import (
    MissingItem,
    SessionItem,
    SessionResult,
    SessionStatus,
)

logger = logging.getLogger(__name__)


class SessionRepository:
    """Async repository for ingest sessions using asyncpg."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def create_session(
        self,
        session_id: str,
        *,
        source: str | None = None,
        channel: str | None = None,
        participant_id: str | None = None,
    ) -> None:
        """Create a new session row."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO sessions (id, source, channel, participant_id)
                   VALUES ($1, $2, $3, $4)
                   ON CONFLICT (id) DO NOTHING""",
                session_id,
                source,
                channel,
                participant_id,
            )

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Load a session row. Returns None if not found."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM sessions WHERE id = $1",
                session_id,
            )
            return dict(row) if row else None

    async def find_active_by_participant(
        self, channel: str, participant_id: str
    ) -> dict[str, Any] | None:
        """Find the active session for a participant (WhatsApp sender).

        Returns the session row if an active session exists, None otherwise.
        Used for WhatsApp where the client can't send a session_id — we
        resolve it server-side from the sender's phone number.
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT * FROM sessions
                   WHERE channel = $1 AND participant_id = $2
                     AND status = 'active'
                   ORDER BY last_activity_at DESC
                   LIMIT 1""",
                channel,
                participant_id,
            )
            return dict(row) if row else None

    async def update_session(
        self,
        session_id: str,
        *,
        status: SessionStatus | None = None,
        expected_count: int | None = None,
        found_count: int | None = None,
        missing_count: int | None = None,
        image_count: int | None = None,
        message: str | None = None,
    ) -> None:
        """Update session fields. Only sets provided fields."""
        sets: list[str] = []
        args: list[Any] = [session_id]
        idx = 2

        if status is not None:
            sets.append(f"status = ${idx}")
            args.append(status.value)
            idx += 1
        if expected_count is not None:
            sets.append(f"expected_count = ${idx}")
            args.append(expected_count)
            idx += 1
        if found_count is not None:
            sets.append(f"found_count = ${idx}")
            args.append(found_count)
            idx += 1
        if missing_count is not None:
            sets.append(f"missing_count = ${idx}")
            args.append(missing_count)
            idx += 1
        if image_count is not None:
            sets.append(f"image_count = ${idx}")
            args.append(image_count)
            idx += 1
        if message is not None:
            sets.append(f"message = ${idx}")
            args.append(message)
            idx += 1

        if not sets:
            return

        sets.append("updated_at = NOW()")
        sets.append("last_activity_at = NOW()")
        if status == SessionStatus.COMPLETE:
            sets.append("completed_at = NOW()")
        elif status == SessionStatus.CLOSED:
            sets.append("closed_at = NOW()")

        sql = f"UPDATE sessions SET {', '.join(sets)} WHERE id = $1"
        async with self._pool.acquire() as conn:
            await conn.execute(sql, *args)

    async def close_session(self, session_id: str) -> bool:
        """Explicitly close a session. Returns True if it was open."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """UPDATE sessions
                   SET status = 'closed', closed_at = NOW(),
                       updated_at = NOW(), last_activity_at = NOW()
                   WHERE id = $1 AND status IN ('active', 'complete')""",
                session_id,
            )
            return result == "UPDATE 1"

    async def expire_session(self, session_id: str) -> None:
        """Mark a session as expired (lazy expiry)."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """UPDATE sessions
                   SET status = 'expired', updated_at = NOW()
                   WHERE id = $1 AND status = 'active'""",
                session_id,
            )

    # ------------------------------------------------------------------
    # Session items (confirmed barcodes)
    # ------------------------------------------------------------------

    async def add_item(self, session_id: str, item: SessionItem) -> bool:
        """Add a confirmed barcode to the session.

        Returns True if the item was newly inserted, False if it already
        existed (deduplicated by barcode_value).
        """
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """INSERT INTO session_items
                       (session_id, barcode_value, barcode_format,
                        barcode_bbox, label_bbox, label_index,
                        match_basis, source_image)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                   ON CONFLICT (session_id, barcode_value) DO NOTHING""",
                session_id,
                item.barcode_value,
                item.barcode_format,
                json.dumps(item.barcode_bbox) if item.barcode_bbox else None,
                json.dumps(item.label_bbox) if item.label_bbox else None,
                item.label_index,
                item.match_basis,
                item.source_image,
            )
            return result == "INSERT 0 1"

    async def get_items(self, session_id: str) -> list[SessionItem]:
        """Load all confirmed items for a session."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM session_items WHERE session_id = $1 ORDER BY id",
                session_id,
            )
            return [
                SessionItem(
                    barcode_value=row["barcode_value"],
                    barcode_format=row["barcode_format"],
                    barcode_bbox=json.loads(row["barcode_bbox"]) if row["barcode_bbox"] else None,
                    label_bbox=json.loads(row["label_bbox"]) if row["label_bbox"] else None,
                    label_index=row["label_index"],
                    match_basis=row["match_basis"],
                    source_image=row["source_image"],
                )
                for row in rows
            ]

    # ------------------------------------------------------------------
    # Missing items (unresolved boxes)
    # ------------------------------------------------------------------

    async def add_missing(self, session_id: str, item: MissingItem) -> None:
        """Add a missing box to the session."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO session_missing
                       (session_id, label_index, label_bbox, barcode_bbox,
                        status, source_image)
                   VALUES ($1, $2, $3, $4, $5, $6)""",
                session_id,
                item.label_index,
                json.dumps(item.label_bbox) if item.label_bbox else None,
                json.dumps(item.barcode_bbox) if item.barcode_bbox else None,
                item.status,
                item.source_image,
            )

    async def get_missing(self, session_id: str) -> list[MissingItem]:
        """Load all missing items for a session (including resolved)."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM session_missing WHERE session_id = $1 ORDER BY id",
                session_id,
            )
            return [
                MissingItem(
                    label_index=row["label_index"],
                    label_bbox=json.loads(row["label_bbox"]) if row["label_bbox"] else None,
                    barcode_bbox=json.loads(row["barcode_bbox"]) if row["barcode_bbox"] else None,
                    status=row["status"],
                    source_image=row["source_image"],
                    resolved=row["resolved"],
                )
                for row in rows
            ]

    async def resolve_missing(
        self, session_id: str, label_index: int, resolved_by_image: int
    ) -> None:
        """Mark a missing item as resolved by a subsequent image."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """UPDATE session_missing
                   SET resolved = TRUE, resolved_by_image = $3
                   WHERE session_id = $1 AND label_index = $2""",
                session_id,
                label_index,
                resolved_by_image,
            )

    async def clear_missing(self, session_id: str) -> None:
        """Clear all missing items (e.g. when re-evaluating after a new image)."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM session_missing WHERE session_id = $1",
                session_id,
            )

    # ------------------------------------------------------------------
    # Full session load (for SessionGraph resume)
    # ------------------------------------------------------------------

    async def load_session_state(self, session_id: str) -> dict[str, Any] | None:
        """Load the full session state: session row + items + missing.

        Returns None if the session doesn't exist. Otherwise returns a dict
        with keys: session, items, missing.
        """
        session = await self.get_session(session_id)
        if session is None:
            return None
        items = await self.get_items(session_id)
        missing = await self.get_missing(session_id)
        return {"session": session, "items": items, "missing": missing}

    async def to_result(self, session_id: str) -> SessionResult | None:
        """Build a SessionResult from the persisted session state."""
        state = await self.load_session_state(session_id)
        if state is None:
            return None

        s = state["session"]
        items = state["items"]
        missing = [m for m in state["missing"] if not m.resolved]

        return SessionResult(
            session_id=session_id,
            status=SessionStatus(s["status"]),
            expected_count=s["expected_count"],
            found_count=s["found_count"],
            missing_count=s["missing_count"],
            items=items,
            missing=missing,
            image_count=s["image_count"],
            message=s["message"],
        )


class NoOpSessionRepository:
    """In-memory no-op session repository for local dev / tests without DB."""

    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}

    async def create_session(
        self, session_id: str, *, source: str | None = None,
        channel: str | None = None, participant_id: str | None = None,
    ) -> None:
        if session_id not in self._sessions:
            from datetime import datetime

            self._sessions[session_id] = {
                "id": session_id,
                "status": "active",
                "expected_count": 0,
                "found_count": 0,
                "missing_count": 0,
                "image_count": 0,
                "source": source,
                "channel": channel,
                "participant_id": participant_id,
                "message": None,
                "last_activity_at": datetime.now(UTC),
            }

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        return self._sessions.get(session_id)

    async def find_active_by_participant(
        self, channel: str, participant_id: str
    ) -> dict[str, Any] | None:
        for s in self._sessions.values():
            if (
                s.get("channel") == channel
                and s.get("participant_id") == participant_id
                and s.get("status") == "active"
            ):
                return s
        return None

    async def update_session(self, session_id: str, **kwargs: Any) -> None:
        from datetime import datetime

        s = self._sessions.setdefault(session_id, {"id": session_id})
        for k, v in kwargs.items():
            if v is not None:
                if k == "status":
                    s[k] = v.value if hasattr(v, "value") else str(v)
                else:
                    s[k] = v
        s["last_activity_at"] = datetime.now(UTC)

    async def close_session(self, session_id: str) -> bool:
        s = self._sessions.get(session_id)
        if s is None or s.get("status") not in ("active", "complete"):
            return False
        s["status"] = "closed"
        return True

    async def expire_session(self, session_id: str) -> None:
        s = self._sessions.get(session_id)
        if s is not None and s.get("status") == "active":
            s["status"] = "expired"

    async def add_item(self, session_id: str, item: SessionItem) -> bool:
        s = self._sessions.setdefault(session_id, {"id": session_id, "_items": []})
        items = s.setdefault("_items", [])
        if any(i.barcode_value == item.barcode_value for i in items):
            return False
        items.append(item)
        return True

    async def get_items(self, session_id: str) -> list[SessionItem]:
        return list(self._sessions.get(session_id, {}).get("_items", []))

    async def add_missing(self, session_id: str, item: MissingItem) -> None:
        s = self._sessions.setdefault(session_id, {"id": session_id})
        s.setdefault("_missing", []).append(item)

    async def get_missing(self, session_id: str) -> list[MissingItem]:
        return self._sessions.get(session_id, {}).get("_missing", [])

    async def resolve_missing(
        self, session_id: str, label_index: int, resolved_by_image: int
    ) -> None:
        for m in self._sessions.get(session_id, {}).get("_missing", []):
            if m.label_index == label_index:
                m.resolved = True

    async def clear_missing(self, session_id: str) -> None:
        self._sessions.get(session_id, {})["_missing"] = []

    async def load_session_state(self, session_id: str) -> dict[str, Any] | None:
        s = self._sessions.get(session_id)
        if s is None:
            return None
        return {
            "session": {k: v for k, v in s.items() if not k.startswith("_")},
            "items": list(s.get("_items", [])),
            "missing": list(s.get("_missing", [])),
        }

    async def to_result(self, session_id: str) -> SessionResult | None:
        state = await self.load_session_state(session_id)
        if state is None:
            return None
        s = state["session"]
        items = state["items"]
        missing = [m for m in state["missing"] if not m.resolved]
        return SessionResult(
            session_id=session_id,
            status=SessionStatus(s.get("status", "active")),
            expected_count=s.get("expected_count", 0),
            found_count=s.get("found_count", 0),
            missing_count=s.get("missing_count", 0),
            items=items,
            missing=missing,
            image_count=s.get("image_count", 0),
            message=s.get("message"),
        )

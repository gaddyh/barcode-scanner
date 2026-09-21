"""Postgres repository for multi-image ingest sessions.

CRUD operations for sessions, session_items (confirmed barcodes), and
session_missing (unresolved boxes). Uses the existing asyncpg pool.

The repository is the persistence boundary — it does not contain business
logic. The SessionGraph (M16B) calls these methods to load/save session
state between images.

``ReceivingSessionStore`` (PR A) extends this to the receiving submission
state machine — translating DB rows ↔ ``src.domain.receiving.ReceivingSession``
and persisting submission transitions via compare-and-set UPDATEs so
concurrent submit attempts have exactly one winner.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC
from typing import Any

import asyncpg

from src.domain.receiving import (
    PhysicalBox,
    ReceivingSession,
    ReceivingSessionStatus,
)
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
        customer_id: str | None = None,
        branch_id: str | None = None,
        action: str | None = None,
    ) -> None:
        """Create a new session row."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO sessions
                       (id, source, channel, participant_id, customer_id, branch_id, action)
                   VALUES ($1, $2, $3, $4, $5, $6, $7)
                   ON CONFLICT (id) DO NOTHING""",
                session_id,
                source,
                channel,
                participant_id,
                customer_id,
                branch_id,
                action,
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
        """Find the active or selection-pending session for a participant.

        Returns the session row if an active or needs_user_selection session
        exists, None otherwise. Used for WhatsApp and web where the client
        can't send a session_id — we resolve it server-side from participant_id.
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT * FROM sessions
                   WHERE channel = $1 AND participant_id = $2
                     AND status IN ('active', 'needs_user_selection')
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
        candidates: list[dict] | None = None,
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
        if candidates is not None:
            sets.append(f"candidates = ${idx}")
            args.append(json.dumps(candidates))
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
            return str(result) == "UPDATE 1"

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
        existed (deduplicated by source_image + label_index).
        """
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """INSERT INTO session_items
                       (session_id, barcode_value, barcode_format,
                        barcode_bbox, label_bbox, label_index,
                        match_basis, source_image)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                   ON CONFLICT DO NOTHING""",
                session_id,
                item.barcode_value,
                item.barcode_format,
                json.dumps(item.barcode_bbox) if item.barcode_bbox else None,
                json.dumps(item.label_bbox) if item.label_bbox else None,
                item.label_index,
                item.match_basis,
                item.source_image,
            )
            return str(result) == "INSERT 0 1"

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
            customer_id=s.get("customer_id"),
            branch_id=s.get("branch_id"),
            action=s.get("action"),
        )


class NoOpSessionRepository:
    """In-memory no-op session repository for local dev / tests without DB."""

    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}

    async def create_session(
        self, session_id: str, *, source: str | None = None,
        channel: str | None = None, participant_id: str | None = None,
        customer_id: str | None = None, branch_id: str | None = None,
        action: str | None = None,
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
                "customer_id": customer_id,
                "branch_id": branch_id,
                "action": action,
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
                and s.get("status") in ("active", "needs_user_selection")
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
        # Dedup by (source_image, label_index) — same label in same image.
        if item.label_index is not None and any(
            i.source_image == item.source_image
            and i.label_index == item.label_index
            for i in items
        ):
            return False
        items.append(item)
        return True

    async def get_items(self, session_id: str) -> list[SessionItem]:
        return list(self._sessions.get(session_id, {}).get("_items", []))

    async def add_missing(self, session_id: str, item: MissingItem) -> None:
        s = self._sessions.setdefault(session_id, {"id": session_id})
        s.setdefault("_missing", []).append(item)

    async def get_missing(self, session_id: str) -> list[MissingItem]:
        return list(self._sessions.get(session_id, {}).get("_missing", []))

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
            customer_id=s.get("customer_id"),
            branch_id=s.get("branch_id"),
            action=s.get("action"),
        )


# ---------------------------------------------------------------------------
# Receiving session store (PR A) — submission state machine persistence
# ---------------------------------------------------------------------------


def _hash_payload(payload_json: str) -> str:
    """SHA-256 hex digest of the frozen order payload JSON."""
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


def _row_to_receiving_session(
    row: dict[str, Any],
    items: list[SessionItem],
) -> ReceivingSession:
    """Translate a sessions row + session_items into a ReceivingSession."""
    submission_status = ReceivingSessionStatus(row.get("submission_status", "active"))
    boxes = [
        PhysicalBox(
            barcode_value=item.barcode_value,
            barcode_format=item.barcode_format or "",
            label_index=item.label_index,
        )
        for item in items
    ]
    return ReceivingSession(
        session_id=row["id"],
        customer_id=row.get("customer_id") or "",
        branch_id=row.get("branch_id") or "",
        action=row.get("action") or "",
        participant_id=row.get("participant_id"),
        boxes=boxes,
        status=submission_status,
        external_order_id=row.get("external_order_id"),
        expected_count=row.get("expected_count") or 0,
        frozen=submission_status != ReceivingSessionStatus.ACTIVE,
    )


class ReceivingSessionStore:
    """Postgres-backed persistence for the receiving submission state machine.

    Translates DB rows ↔ ``src.domain.receiving.ReceivingSession``. All
    submission transitions are compare-and-set UPDATEs guarded by the
    expected ``submission_status`` — two concurrent submit clicks have
    exactly one winner. The frozen order payload is persisted verbatim on
    ACTIVE → SUBMITTING and reused on every retry; it is never rebuilt
    from ``session_items``.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def create_receiving_session(
        self,
        session_id: str,
        *,
        customer_id: str,
        branch_id: str,
        action: str,
        participant_id: str | None = None,
        channel: str = "web",
        source: str = "web",
    ) -> None:
        """Insert a new receiving session row with submission_status='active'."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO sessions
                       (id, status, submission_status, channel, participant_id,
                        customer_id, branch_id, action, source)
                   VALUES ($1, 'active', 'active', $2, $3, $4, $5, $6, $7)
                   ON CONFLICT (id) DO NOTHING""",
                session_id,
                channel,
                participant_id,
                customer_id,
                branch_id,
                action,
                source,
            )

    async def get_receiving_session(self, session_id: str) -> ReceivingSession | None:
        """Load a ReceivingSession by ID, or None if not found."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM sessions WHERE id = $1",
                session_id,
            )
            if row is None:
                return None
            item_rows = await conn.fetch(
                "SELECT * FROM session_items WHERE session_id = $1 ORDER BY id",
                session_id,
            )
        items = [
            SessionItem(
                barcode_value=r["barcode_value"],
                barcode_format=r["barcode_format"],
                barcode_bbox=json.loads(r["barcode_bbox"]) if r["barcode_bbox"] else None,
                label_bbox=json.loads(r["label_bbox"]) if r["label_bbox"] else None,
                label_index=r["label_index"],
                match_basis=r["match_basis"],
                source_image=r["source_image"],
            )
            for r in item_rows
        ]
        return _row_to_receiving_session(dict(row), items)

    async def find_open_submission_by_participant(
        self, participant_id: str
    ) -> ReceivingSession | None:
        """Find an open (active or submission_unknown) receiving session for
        a participant. Used to enforce one unresolved receiving session per
        participant — a SUBMISSION_UNKNOWN session blocks new session creation
        until the user retries/reconciles it.
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT * FROM sessions
                   WHERE participant_id = $1
                     AND submission_status IN ('active', 'submission_unknown')
                   ORDER BY updated_at DESC
                   LIMIT 1""",
                participant_id,
            )
            if row is None:
                return None
            item_rows = await conn.fetch(
                "SELECT * FROM session_items WHERE session_id = $1 ORDER BY id",
                row["id"],
            )
        items = [
            SessionItem(
                barcode_value=r["barcode_value"],
                barcode_format=r["barcode_format"],
                barcode_bbox=json.loads(r["barcode_bbox"]) if r["barcode_bbox"] else None,
                label_bbox=json.loads(r["label_bbox"]) if r["label_bbox"] else None,
                label_index=r["label_index"],
                match_basis=r["match_basis"],
                source_image=r["source_image"],
            )
            for r in item_rows
        ]
        return _row_to_receiving_session(dict(row), items)

    async def get_frozen_payload(self, session_id: str) -> dict[str, Any] | None:
        """Load the frozen order payload for retry. Returns None if not frozen."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT frozen_order_payload FROM sessions WHERE id = $1",
                session_id,
            )
            if row is None or row["frozen_order_payload"] is None:
                return None
            loaded: dict[str, Any] = json.loads(row["frozen_order_payload"])
            return loaded

    async def freeze_submission(
        self,
        session_id: str,
        frozen_payload: dict[str, Any],
    ) -> bool:
        """CAS: ACTIVE → SUBMITTING. Persists the frozen order payload + hash.

        Returns True if the transition succeeded, False if the session was
        not in ACTIVE state (another caller won the race).
        """
        payload_json = json.dumps(frozen_payload, sort_keys=True)
        payload_hash = _hash_payload(payload_json)
        async with self._pool.acquire() as conn:
            result = await conn.fetchval(
                """UPDATE sessions
                   SET submission_status = 'submitting',
                       frozen_order_payload = $2::jsonb,
                       frozen_payload_hash = $3,
                       frozen_at = NOW(),
                       updated_at = NOW()
                   WHERE id = $1 AND submission_status = 'active'
                   RETURNING id""",
                session_id,
                payload_json,
                payload_hash,
            )
        return result is not None

    async def mark_submitted(
        self, session_id: str, external_order_id: int
    ) -> bool:
        """CAS: SUBMITTING/SUBMISSION_UNKNOWN → SUBMITTED.

        Accepts both states because a retry from SUBMISSION_UNKNOWN can
        succeed if the idempotency store's lease expired and the operation
        is re-attempted, or if external reconciliation (future MVP) proves
        the order was created. Returns True on success.
        """
        async with self._pool.acquire() as conn:
            result = await conn.fetchval(
                """UPDATE sessions
                   SET submission_status = 'submitted',
                       external_order_id = $2,
                       updated_at = NOW()
                   WHERE id = $1
                     AND submission_status IN ('submitting', 'submission_unknown')
                   RETURNING id""",
                session_id,
                external_order_id,
            )
        return result is not None

    async def mark_submission_unknown(self, session_id: str) -> bool:
        """CAS: SUBMITTING → SUBMISSION_UNKNOWN (idempotent on re-transition)."""
        async with self._pool.acquire() as conn:
            result = await conn.fetchval(
                """UPDATE sessions
                   SET submission_status = 'submission_unknown',
                       updated_at = NOW()
                   WHERE id = $1
                     AND submission_status IN ('submitting', 'submission_unknown')
                   RETURNING id""",
                session_id,
            )
        return result is not None

    async def revert_to_active(self, session_id: str) -> bool:
        """CAS: SUBMITTING → ACTIVE. Clears frozen payload (pre-submit failure)."""
        async with self._pool.acquire() as conn:
            result = await conn.fetchval(
                """UPDATE sessions
                   SET submission_status = 'active',
                       frozen_order_payload = NULL,
                       frozen_payload_hash = NULL,
                       frozen_at = NULL,
                       external_order_id = NULL,
                       updated_at = NOW()
                   WHERE id = $1 AND submission_status = 'submitting'
                   RETURNING id""",
                session_id,
            )
        return result is not None

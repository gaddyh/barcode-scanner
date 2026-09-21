"""asyncpg-backed idempotency store with lease + fencing-token semantics.

Reimplementation of echo-v2's ``persistence/postgres_idempotency.py`` on
asyncpg (NOT SQLAlchemy — barcode-scanner is asyncpg).

Satisfies :class:`src.runtime.idempotency.IdempotencyStore` against
PostgreSQL. Implements the concurrency model from the plan:

* ``reserve(key)`` — atomic ``INSERT ... ON CONFLICT DO NOTHING`` to claim a
  fresh key; on conflict, reads the existing row and either returns
  ``IN_PROGRESS`` (active lease), ``COMPLETED`` (terminal), or atomically
  **reclaims an expired lease** via a conditional ``UPDATE ... WHERE
  lease_expires_at <= now()``. Crash recovery is a property of ``reserve()``,
  not a background sweeper.

* Every owner write (``put_success`` / ``put_failure`` / ``put_indeterminate``
  / ``release`` / ``renew_lease``) is **token-guarded**:
  ``UPDATE ... WHERE owner_token = :my_token AND state = 'IN_PROGRESS'``.
  A slow prior owner whose lease was reclaimed will see **zero rows
  affected** → raise :class:`LostOwnershipError` (do not overwrite the new
  owner's outcome).

* All lease comparisons use DB ``now()``, never the Python clock — so two
  workers with slightly different system clocks agree on lease expiry.

* ``wait_for_completion`` polls ``get(key)`` with short sleeps, falling back
  to ``reserve()``-reclaim if the lease is observed expired. On timeout,
  raises ``RetryableError``.

Outcomes are serialized to/from a ``JSONB`` column as tagged dicts (plain
data, never exception objects — per the existing module contract).
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Generic, TypeVar

import asyncpg

from src.runtime.errors import RetryableError
from src.runtime.idempotency import (
    IdempotencyStore,
    IndeterminateOutcome,
    LostOwnershipError,
    PermanentFailureOutcome,
    ReserveResult,
    ReserveStatus,
    StoredOutcome,
    SuccessOutcome,
)

__all__ = ["PostgresIdempotencyStore"]

TOutput = TypeVar("TOutput")

_DEFAULT_LEASE_SECONDS = 30
_POLL_INTERVAL_SECONDS = 0.05
_POLL_TIMEOUT_SECONDS = 30.0

_STATE_IN_PROGRESS = "IN_PROGRESS"
_STATE_SUCCESS = "SUCCESS"
_STATE_FAILURE = "FAILURE"
_STATE_INDETERMINATE = "INDETERMINATE"

_TAG_SUCCESS = "success"
_TAG_FAILURE = "failure"
_TAG_INDETERMINATE = "indeterminate"


class PostgresIdempotencyStore(IdempotencyStore[TOutput], Generic[TOutput]):
    """PostgreSQL implementation of :class:`IdempotencyStore` using asyncpg.

    Each method acquires a connection from the pool, runs its query, and
    releases the connection. There is no shared session — each call is
    independent. This is simpler than echo-v2's session-factory model and
    matches barcode-scanner's existing asyncpg usage pattern.
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        *,
        lease_seconds: int = _DEFAULT_LEASE_SECONDS,
    ) -> None:
        self._pool = pool
        self._lease_seconds = lease_seconds

    # --- read -------------------------------------------------------------

    async def get(self, key: str) -> StoredOutcome[TOutput] | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT state, outcome FROM idempotency_operations WHERE key = $1",
                key,
            )
            if row is None or row["state"] == _STATE_IN_PROGRESS:
                return None
            return _row_to_outcome(row["outcome"])

    # --- reserve + reclaim ------------------------------------------------

    async def reserve(self, key: str) -> ReserveResult:
        token = uuid.uuid4()

        async with self._pool.acquire() as conn:
            # 1. Try to INSERT a fresh claim.
            inserted = await conn.fetchval(
                "INSERT INTO idempotency_operations "
                "(key, state, owner_token, lease_expires_at) "
                "VALUES ($1, $2, $3, now() + make_interval(secs => $4)) "
                "ON CONFLICT (key) DO NOTHING "
                "RETURNING key",
                key,
                _STATE_IN_PROGRESS,
                str(token),
                self._lease_seconds,
            )
            if inserted is not None:
                return ReserveResult(ReserveStatus.ACQUIRED, token)

            # 2. Conflict: read the existing row.
            row = await conn.fetchrow(
                "SELECT state, lease_expires_at FROM idempotency_operations WHERE key = $1",
                key,
            )
            if row is None:  # pragma: no cover — race
                return await self.reserve(key)

            if row["state"] != _STATE_IN_PROGRESS:
                return ReserveResult(ReserveStatus.COMPLETED)

            # 3. In-progress: check lease (DB now(), not Python).
            now_db = await conn.fetchval("SELECT now()")
            if row["lease_expires_at"] is not None and row["lease_expires_at"] > now_db:
                return ReserveResult(ReserveStatus.IN_PROGRESS)

            # 4. Lease expired: attempt atomic reclaim.
            reclaimed = await conn.fetchval(
                "UPDATE idempotency_operations "
                "SET owner_token = $1, "
                "    lease_expires_at = now() + make_interval(secs => $2), "
                "    updated_at = now() "
                "WHERE key = $3 AND state = 'IN_PROGRESS' "
                "  AND (lease_expires_at IS NULL OR lease_expires_at <= now()) "
                "RETURNING key",
                str(token),
                self._lease_seconds,
                key,
            )
            if reclaimed is not None:
                return ReserveResult(ReserveStatus.ACQUIRED, token)

            # 5. Another caller beat us to the reclaim; re-read and branch.
            row = await conn.fetchrow(
                "SELECT state FROM idempotency_operations WHERE key = $1",
                key,
            )
            if row is None:  # pragma: no cover
                return ReserveResult(ReserveStatus.IN_PROGRESS)
            if row["state"] != _STATE_IN_PROGRESS:
                return ReserveResult(ReserveStatus.COMPLETED)
            return ReserveResult(ReserveStatus.IN_PROGRESS)

    # --- owner writes (token-guarded) -------------------------------------

    async def put_success(
        self,
        key: str,
        owner_token: uuid.UUID,
        outcome: SuccessOutcome[TOutput],
    ) -> None:
        payload = _serialize_success(outcome)
        await self._terminal_write(
            key,
            owner_token,
            state=_STATE_SUCCESS,
            outcome=payload,
        )

    async def put_failure(
        self,
        key: str,
        owner_token: uuid.UUID,
        outcome: PermanentFailureOutcome,
    ) -> None:
        payload = _serialize_failure(outcome)
        await self._terminal_write(
            key,
            owner_token,
            state=_STATE_FAILURE,
            outcome=payload,
        )

    async def put_indeterminate(
        self,
        key: str,
        owner_token: uuid.UUID,
        outcome: IndeterminateOutcome,
    ) -> None:
        payload = _serialize_indeterminate(outcome)
        await self._terminal_write(
            key,
            owner_token,
            state=_STATE_INDETERMINATE,
            outcome=payload,
        )

    async def release(self, key: str, owner_token: uuid.UUID) -> None:
        """Release a claim by deleting the in-progress row.

        A released key has no terminal outcome — a subsequent ``reserve``
        can re-claim it. Token-guarded so only the current owner can release.
        """
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM idempotency_operations "
                "WHERE key = $1 AND owner_token = $2 AND state = 'IN_PROGRESS'",
                key,
                str(owner_token),
            )
            if result == "DELETE 0":
                raise LostOwnershipError(
                    f"release() for key {key!r} affected 0 rows; ownership lost."
                )

    async def renew_lease(self, key: str, owner_token: uuid.UUID) -> bool:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE idempotency_operations "
                "SET lease_expires_at = now() + make_interval(secs => $1), "
                "    updated_at = now() "
                "WHERE key = $2 AND owner_token = $3 AND state = 'IN_PROGRESS'",
                self._lease_seconds,
                key,
                str(owner_token),
            )
            return str(result) == "UPDATE 1"

    # --- wait -------------------------------------------------------------

    async def wait_for_completion(self, key: str) -> StoredOutcome[TOutput]:
        deadline = asyncio.get_event_loop().time() + _POLL_TIMEOUT_SECONDS
        while asyncio.get_event_loop().time() < deadline:
            outcome = await self.get(key)
            if outcome is not None:
                return outcome
            # Check if the lease is expired — if so, try to reclaim and run.
            r = await self.reserve(key)
            if r.status == ReserveStatus.ACQUIRED:
                # We stole the lease but we're a waiter, not an executor.
                # Release it so the original flow can retry, and raise
                # RetryableError so the caller re-attempts from the top.
                await self.release(key, r.owner_token)  # type: ignore[arg-type]
                raise RetryableError(
                    f"Idempotent operation {key!r} owner lease expired; retry."
                )
            if r.status == ReserveStatus.COMPLETED:
                outcome = await self.get(key)
                if outcome is not None:
                    return outcome
                raise RetryableError("Idempotent operation did not complete")
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)
        raise RetryableError(
            f"Timed out waiting for idempotent operation {key!r} to complete."
        )

    # --- internal ---------------------------------------------------------

    async def _terminal_write(
        self,
        key: str,
        owner_token: uuid.UUID,
        *,
        state: str,
        outcome: dict[str, Any],
    ) -> None:
        """Token-guarded UPDATE to a terminal state.

        Raises :class:`LostOwnershipError` if zero rows are affected (the
        caller's lease expired and was reclaimed — do not overwrite the new
        owner's outcome).
        """
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE idempotency_operations "
                "SET state = $1, outcome = $2, owner_token = NULL, "
                "    lease_expires_at = NULL, updated_at = now() "
                "WHERE key = $3 AND owner_token = $4 AND state = 'IN_PROGRESS'",
                state,
                json.dumps(outcome),
                key,
                str(owner_token),
            )
            if result == "UPDATE 0":
                raise LostOwnershipError(
                    f"Terminal write for key {key!r} affected 0 rows; "
                    "ownership lost (lease expired and was reclaimed)."
                )


# --- outcome serialization -------------------------------------------------


def _serialize_success(outcome: SuccessOutcome[Any]) -> dict[str, Any]:
    return {
        "tag": _TAG_SUCCESS,
        "value": outcome.value,
        "attempts": outcome.attempts,
        "duration_ms": outcome.duration_ms,
    }


def _serialize_failure(outcome: PermanentFailureOutcome) -> dict[str, Any]:
    return {
        "tag": _TAG_FAILURE,
        "error_type": outcome.error_type,
        "error_message": outcome.error_message,
    }


def _serialize_indeterminate(outcome: IndeterminateOutcome) -> dict[str, Any]:
    return {
        "tag": _TAG_INDETERMINATE,
        "error_type": outcome.error_type,
        "error_message": outcome.error_message,
    }


def _row_to_outcome(outcome_json: str | None) -> StoredOutcome[Any] | None:
    if outcome_json is None:
        return None
    data = json.loads(outcome_json) if isinstance(outcome_json, str) else outcome_json
    tag = data.get("tag")
    if tag == _TAG_SUCCESS:
        return SuccessOutcome(
            value=data.get("value"),
            attempts=data.get("attempts", 0),
            duration_ms=data.get("duration_ms", 0.0),
        )
    if tag == _TAG_FAILURE:
        return PermanentFailureOutcome(
            error_type=data.get("error_type", ""),
            error_message=data.get("error_message", ""),
        )
    if tag == _TAG_INDETERMINATE:
        return IndeterminateOutcome(
            error_type=data.get("error_type", ""),
            error_message=data.get("error_message", ""),
        )
    return None  # pragma: no cover

"""Idempotency support for the runtime executor.

When a caller supplies an ``idempotency_key`` and an :class:`IdempotencyStore`
to :func:`src.runtime.executor.execute`, the runtime caches terminal
outcomes (successes, permanent failures, and indeterminate outcomes) so that
repeat calls with the same key short-circuit, and concurrent duplicates wait
for the owner to finish and share its result.

The store protocol carries an ``owner_token`` so that a persistent
implementation (Postgres) can enforce **fencing**: every owner write
(``put_success`` / ``put_failure`` / ``put_indeterminate`` / ``release`` /
``renew_lease``) is token-guarded, so a slow prior owner whose lease expired
and was reclaimed by a new owner cannot overwrite the new owner's outcome.
``reserve()`` returns a :class:`ReserveResult` carrying both the
:class:`ReserveStatus` and, on ``ACQUIRED``, the ``owner_token`` the caller
must present on every subsequent write for that key.

The default :class:`InMemoryIdempotencyStore` coordinates concurrent callers
within a single Python process via per-key :class:`asyncio.Future` objects and
honors the same token contract, so runtime tests exercise the real protocol.

Ported from echo-v2's ``runtime/idempotency.py``, adapted to barcode-scanner's
error imports.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from src.runtime.errors import RetryableError

__all__ = [
    "IdempotencyStore",
    "InMemoryIdempotencyStore",
    "IndeterminateOutcome",
    "LostOwnershipError",
    "PermanentFailureOutcome",
    "ReserveResult",
    "ReserveStatus",
    "StoredOutcome",
    "SuccessOutcome",
]

TOutput = TypeVar("TOutput")


class ReserveStatus(Enum):
    """Result of attempting to claim a key for execution."""

    ACQUIRED = "acquired"
    """This caller owns execution and must store a terminal outcome."""

    IN_PROGRESS = "in_progress"
    """Another caller is already running it; call :meth:`IdempotencyStore.wait_for_completion`."""

    COMPLETED = "completed"
    """A terminal outcome is already stored; call :meth:`IdempotencyStore.get`."""


@dataclass(frozen=True)
class ReserveResult:
    """Outcome of :meth:`IdempotencyStore.reserve`.

    ``owner_token`` is set iff ``status == ACQUIRED``; the caller must present
    it on every subsequent owner write (``put_*`` / ``release`` /
    ``renew_lease``) for that key. A token-guarded store rejects writes from a
    caller whose token no longer matches (lost ownership).
    """

    status: ReserveStatus
    owner_token: uuid.UUID | None = None


class LostOwnershipError(RetryableError):
    """Raised when an owner write affects zero rows (token no longer matches).

    This is a :class:`RetryableError` subclass so the runtime's retry path
    treats it as "try again from reserve()" — the caller's prior ownership has
    been reclaimed by another worker and the operation must be re-attempted
    from the top (re-reserve, re-execute, re-store).
    """

    def __init__(self, message: str = "") -> None:
        super().__init__(message, code="LostOwnershipError")


@dataclass(frozen=True)
class SuccessOutcome(Generic[TOutput]):
    """Cached successful outcome of an idempotent operation."""

    value: TOutput
    attempts: int
    duration_ms: float


@dataclass(frozen=True)
class PermanentFailureOutcome:
    """Cached permanent failure of an idempotent operation.

    ``error_type`` is retained for observability/debugging only. On replay the
    runtime always raises ``PermanentError(error_message)``; it does not attempt
    to reconstruct the original exception class.
    """

    error_type: str
    error_message: str


@dataclass(frozen=True)
class IndeterminateOutcome:
    """Cached indeterminate outcome of an idempotent irreversible write.

    The operation's side effect may or may not have happened (e.g. a timeout
    mid-flight, a cancellation, or an unexpected error after the request was
    submitted). On replay the runtime raises ``IndeterminateError`` and does
    NOT re-run the operation — the key remains blocked until reconciled.
    """

    error_type: str
    error_message: str


StoredOutcome = SuccessOutcome[TOutput] | PermanentFailureOutcome | IndeterminateOutcome


@runtime_checkable
class IdempotencyStore(Protocol[TOutput]):
    """Persistence + coordination contract for idempotent execution.

    Every owner write is guarded by ``owner_token``: a caller may only store
    a terminal outcome or release a claim if it still owns the reservation.
    A persistent store enforces this atomically (e.g. ``UPDATE ... WHERE
    owner_token = :my_token AND state = 'IN_PROGRESS'``); a slow prior owner
    whose lease was reclaimed will see zero rows affected and must raise
    :class:`LostOwnershipError` rather than overwrite the new owner's outcome.
    """

    async def get(self, key: str) -> StoredOutcome[TOutput] | None:
        """Return the terminal outcome stored for ``key``, or ``None`` if absent."""
        ...

    async def reserve(self, key: str) -> ReserveResult:
        """Atomically claim ``key`` for execution.

        Returns a :class:`ReserveResult` with:

        * ``status == ACQUIRED`` and a fresh ``owner_token`` if this caller now
          owns execution (it must present the token on every owner write);
        * ``status == IN_PROGRESS`` if another caller is already running it
          (call :meth:`wait_for_completion`);
        * ``status == COMPLETED`` if a terminal outcome is already stored
          (call :meth:`get`).

        A persistent store reclaims expired leases atomically inside this call
        — crash recovery is a property of ``reserve()``, not a background
        sweeper.
        """
        ...

    async def wait_for_completion(self, key: str) -> StoredOutcome[TOutput]:
        """Wait for the owner of ``key`` to store a terminal outcome.

        Raises ``RetryableError`` if the owner released the claim without
        storing a terminal outcome (e.g. retryable failure or cancellation).
        The waiter does not need an ``owner_token``.
        """
        ...

    async def put_success(
        self,
        key: str,
        owner_token: uuid.UUID,
        outcome: SuccessOutcome[TOutput],
    ) -> None:
        """Store a successful terminal outcome and wake any waiters.

        Raises :class:`LostOwnershipError` if ``owner_token`` no longer matches
        (the caller's lease expired and was reclaimed).
        """
        ...

    async def put_failure(
        self,
        key: str,
        owner_token: uuid.UUID,
        outcome: PermanentFailureOutcome,
    ) -> None:
        """Store a permanent-failure terminal outcome and wake any waiters.

        Raises :class:`LostOwnershipError` if ``owner_token`` no longer matches.
        """
        ...

    async def put_indeterminate(
        self,
        key: str,
        owner_token: uuid.UUID,
        outcome: IndeterminateOutcome,
    ) -> None:
        """Store an indeterminate terminal outcome and wake any waiters.

        Used when an irreversible write's outcome is unknown (timeout,
        cancellation, or unexpected failure after the side effect may have
        happened). The key is NOT released — subsequent calls with the same
        key will raise ``IndeterminateError`` on replay until reconciled.

        Raises :class:`LostOwnershipError` if ``owner_token`` no longer matches.
        """
        ...

    async def release(self, key: str, owner_token: uuid.UUID) -> None:
        """Release an acquired claim without a terminal outcome.

        Wakes any waiters with ``RetryableError`` so they may retry. Used on
        retryable failures and on owner cancellation of reversible operations.

        Raises :class:`LostOwnershipError` if ``owner_token`` no longer matches.
        """
        ...

    async def renew_lease(self, key: str, owner_token: uuid.UUID) -> bool:
        """Extend the lease on an in-progress claim.

        Returns ``True`` if the caller still owns the key (lease extended),
        ``False`` if ownership was lost (the caller should abandon the
        operation). Optional for short operations; useful for long-running
        work that may exceed the default lease duration.
        """
        ...


class InMemoryIdempotencyStore(Generic[TOutput]):
    """Process-local :class:`IdempotencyStore` backed by dicts and futures.

    .. warning::
        DO NOT use in production. This store has no persistence, no
        cross-process coordination, and no lease/TTL on in-progress claims.
        A process restart forgets all stored outcomes and in-progress claims.
        Use a persistent distributed store (Redis/Postgres) for any real
        workload. This implementation is suitable only for tests and local
        development.

    Concurrent duplicates within one Python process are coordinated via
    per-key ``asyncio.Future`` objects. The store honors the same
    ``owner_token`` contract as the Postgres store: a wrong-token write raises
    :class:`LostOwnershipError`, so runtime tests exercise the real fencing
    semantics. There is no lease expiry in-memory (no clock needed), so
    ``renew_lease`` is a no-op that returns ``True`` iff the caller still owns
    the key.
    """

    def __init__(self) -> None:
        self._outcomes: dict[str, StoredOutcome[TOutput]] = {}
        self._in_progress: dict[
            str, tuple[asyncio.Future[StoredOutcome[TOutput]], uuid.UUID]
        ] = {}

    async def get(self, key: str) -> StoredOutcome[TOutput] | None:
        return self._outcomes.get(key)

    async def reserve(self, key: str) -> ReserveResult:
        if key in self._outcomes:
            return ReserveResult(ReserveStatus.COMPLETED)
        if key in self._in_progress:
            return ReserveResult(ReserveStatus.IN_PROGRESS)
        loop = asyncio.get_running_loop()
        token = uuid.uuid4()
        self._in_progress[key] = (loop.create_future(), token)
        return ReserveResult(ReserveStatus.ACQUIRED, token)

    async def wait_for_completion(self, key: str) -> StoredOutcome[TOutput]:
        entry = self._in_progress.get(key)
        if entry is None:
            # The owner finished (or released) between reserve() and here;
            # fall back to a terminal lookup.
            outcome = self._outcomes.get(key)
            if outcome is not None:
                return outcome
            raise RetryableError("Idempotent operation did not complete")
        future, _token = entry
        return await future

    async def put_success(
        self,
        key: str,
        owner_token: uuid.UUID,
        outcome: SuccessOutcome[TOutput],
    ) -> None:
        self._check_owner(key, owner_token)
        self._outcomes[key] = outcome
        future = self._in_progress.pop(key, None)
        if future is not None:
            fut, _ = future
            if not fut.done():
                fut.set_result(outcome)

    async def put_failure(
        self,
        key: str,
        owner_token: uuid.UUID,
        outcome: PermanentFailureOutcome,
    ) -> None:
        self._check_owner(key, owner_token)
        self._outcomes[key] = outcome
        future = self._in_progress.pop(key, None)
        if future is not None:
            fut, _ = future
            if not fut.done():
                fut.set_result(outcome)

    async def put_indeterminate(
        self,
        key: str,
        owner_token: uuid.UUID,
        outcome: IndeterminateOutcome,
    ) -> None:
        self._check_owner(key, owner_token)
        self._outcomes[key] = outcome
        future = self._in_progress.pop(key, None)
        if future is not None:
            fut, _ = future
            if not fut.done():
                fut.set_result(outcome)

    async def release(self, key: str, owner_token: uuid.UUID) -> None:
        self._check_owner(key, owner_token)
        future = self._in_progress.pop(key, None)
        if future is not None:
            fut, _ = future
            if not fut.done():
                fut.set_exception(
                    RetryableError("Idempotent operation did not complete")
                )

    async def renew_lease(self, key: str, owner_token: uuid.UUID) -> bool:
        entry = self._in_progress.get(key)
        if entry is None:
            return False
        _future, token = entry
        return token == owner_token

    def _check_owner(self, key: str, owner_token: uuid.UUID) -> None:
        entry = self._in_progress.get(key)
        if entry is None:
            # No in-progress claim: either already terminal or never reserved.
            # A terminal write with no in-progress row means the caller does
            # not own anything — treat as lost ownership.
            raise LostOwnershipError(
                f"No in-progress claim for key {key!r}; ownership lost or never held."
            )
        _future, token = entry
        if token != owner_token:
            raise LostOwnershipError(
                f"Owner token mismatch for key {key!r}; lease was reclaimed by another worker."
            )

    # --- test helpers -----------------------------------------------------

    def seed_outcome(self, key: str, outcome: StoredOutcome[TOutput]) -> None:
        """Pre-seed a terminal outcome without reserving (test helper).

        Simulates a persistent store that survived a process restart: the
        outcome is already terminal and a subsequent ``execute()`` should hit
        the fast path without re-running. Never use in production code —
        ownership checks exist for a reason.
        """
        self._outcomes[key] = outcome


# Structural check: InMemoryIdempotencyStore satisfies the protocol.
_: IdempotencyStore[Any] = InMemoryIdempotencyStore[Any]()

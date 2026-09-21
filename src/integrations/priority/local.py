"""LocalPriorityGateway — asyncpg-backed local stand-in for the Priority ERP.

Reads ``priority_customers`` / ``priority_branches`` and writes
``priority_orders``. Conforms to the ``PriorityGateway`` protocol.

Error classification at the integration boundary (per PR #3 edit #4):
- ``asyncpg.PostgresError`` / ``OSError`` on catalog reads (customers,
  branches) → ``RetryableError`` (transient — the DB may come back).
- ``asyncpg.PostgresError`` / ``OSError`` on ``create_draft_order``
  BEFORE the INSERT executes (connection refused, pool exhausted) →
  ``RetryableError``.
- ``asyncpg.PostgresError`` / ``OSError`` DURING the INSERT — the
  request may have crossed the network boundary and the row may or may
  not have been written → ``IndeterminateError`` (conservative: never
  blind-retry an irreversible write).
- ``asyncpg.UniqueViolationError`` on ``session_id`` → ``PermanentError``
  (defense-in-depth: the runtime idempotency layer should have caught
  this; a duplicate here means the caller bypassed the runtime).
"""

from __future__ import annotations

import json

import asyncpg

from src.integrations.priority.models import (
    Branch,
    CreateDraftOrderRequest,
    CreateDraftOrderResult,
    Customer,
)
from src.runtime.errors import IndeterminateError, PermanentError, RetryableError


class PriorityError(RuntimeError):
    """Raised when the local Priority-compatible store is unavailable.

    Kept for backward compatibility with callers that catch
    ``PriorityError``. New code should let the runtime error taxonomy
    (``RetryableError`` / ``PermanentError`` / ``IndeterminateError``)
    propagate instead.
    """


def _classify_read_error(exc: BaseException) -> RetryableError:
    """Classify a catalog-read failure as retryable."""
    return RetryableError(
        f"Local Priority catalog is unavailable: {exc}",
    )


def _classify_write_error(
    exc: BaseException, *, after_submit: bool
) -> IndeterminateError | RetryableError | PermanentError:
    """Classify a draft-order write failure.

    ``after_submit`` — True if the INSERT statement was issued (the
    request may have crossed the network boundary); False if the failure
    happened before the INSERT (connection refused, pool exhausted).
    """
    if isinstance(exc, asyncpg.UniqueViolationError):
        return PermanentError(
            f"Duplicate session_id for Priority draft order: {exc}",
        )
    if after_submit:
        return IndeterminateError(
            f"Priority draft order may have been created but the "
            f"confirmation was lost: {exc}",
        )
    return RetryableError(
        f"Could not create local Priority order (pre-submit): {exc}",
    )


class LocalPriorityGateway:
    """asyncpg-backed local stand-in for the Priority ERP.

    Conforms to ``PriorityGateway``. The adapter just performs the
    external operation and classifies errors — it does NOT wrap itself
    in ``runtime.execute()`` (that is the service layer's job in PR #5).
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def customers(self) -> list[Customer]:
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    """SELECT id, name
                       FROM priority_customers
                       WHERE active
                       ORDER BY name"""
                )
        except (asyncpg.PostgresError, OSError) as exc:
            raise PriorityError("Local Priority catalog is unavailable") from exc
        return [
            Customer(id=str(row["id"]), name=str(row["name"])) for row in rows
        ]

    async def branches(self, customer_id: str) -> list[Branch]:
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    """SELECT b.id, b.name
                       FROM priority_branches AS b
                       JOIN priority_customers AS c ON c.id = b.customer_id
                       WHERE b.customer_id = $1 AND b.active AND c.active
                       ORDER BY b.name""",
                    customer_id,
                )
        except (asyncpg.PostgresError, OSError) as exc:
            raise PriorityError("Local Priority catalog is unavailable") from exc
        return [
            Branch(
                id=str(row["id"]),
                name=str(row["name"]),
                customer_id=customer_id,
            )
            for row in rows
        ]

    async def create_draft_order(
        self, request: CreateDraftOrderRequest
    ) -> CreateDraftOrderResult:
        """Create a draft order row in ``priority_orders``.

        Error classification:
        - Pre-submit connection failure → ``RetryableError``.
        - Failure during the INSERT (after submit) → ``IndeterminateError``.
        - ``UniqueViolationError`` on ``session_id`` → ``PermanentError``.
        """
        items_json = json.dumps(request.items_as_dicts())
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    """INSERT INTO priority_orders
                           (session_id, customer_id, branch_id, action, items)
                       VALUES ($1, $2, $3, $4, $5::jsonb)
                       RETURNING id, session_id, status""",
                    request.session_id,
                    request.customer_id,
                    request.branch_id,
                    request.action,
                    items_json,
                )
        except (asyncpg.PostgresError, OSError) as exc:
            # The INSERT statement was issued — the request may have
            # crossed the network boundary. Conservative: treat as
            # indeterminate unless we can prove it's a unique violation
            # (which proves the row was NOT written by us).
            raise _classify_write_error(exc, after_submit=True) from exc

        if row is None:
            raise IndeterminateError(
                "Priority draft order returned no row — the insert may "
                "have silently failed."
            )

        return CreateDraftOrderResult(
            order_id=int(row["id"]),
            session_id=str(row.get("session_id", request.session_id)),
            status=str(row.get("status", "draft")),
        )

"""Local PostgreSQL stand-in for the Priority catalog and order context."""

from __future__ import annotations

import json
from typing import Any

import asyncpg


class PriorityError(RuntimeError):
    """Raised when the local Priority-compatible store is unavailable."""


class PriorityRepository:
    """Read Priority-compatible customers and branches from PostgreSQL."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def customers(self) -> list[dict[str, str]]:
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
        return [{"id": str(row["id"]), "name": str(row["name"])} for row in rows]

    async def branches(self, customer_id: str) -> list[dict[str, str]]:
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
        return [{"id": str(row["id"]), "name": str(row["name"])} for row in rows]

    async def create_order(
        self,
        *,
        session_id: str,
        customer_id: str,
        branch_id: str,
        action: str,
        items: list[dict[str, Any]],
    ) -> int:
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    """INSERT INTO priority_orders
                           (session_id, customer_id, branch_id, action, items)
                       VALUES ($1, $2, $3, $4, $5::jsonb)
                       RETURNING id""",
                    session_id,
                    customer_id,
                    branch_id,
                    action,
                    json.dumps(items),
                )
        except (asyncpg.PostgresError, OSError) as exc:
            raise PriorityError("Could not create local Priority order") from exc
        return int(row["id"])

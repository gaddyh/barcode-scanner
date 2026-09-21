"""Tests for src/integrations/priority.py — PriorityRepository error handling."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import asyncpg
import pytest

from src.integrations.priority import (
    PriorityError,
    PriorityRepository,
)


def _make_pool_with_rows(rows):
    """Build a fake asyncpg pool whose fetch returns ``rows``."""
    conn = MagicMock()
    conn.fetch = AsyncMock(return_value=rows)
    conn.fetchrow = AsyncMock(return_value=rows[0] if rows else None)

    class FakeAcquire:
        async def __aenter__(self_inner):
            return conn

        async def __aexit__(self_inner, *exc):
            return False

    pool = MagicMock()
    pool.acquire = MagicMock(return_value=FakeAcquire())
    return pool, conn


def _make_pool_raising(exc):
    """Build a fake asyncpg pool whose fetch raises ``exc``."""
    conn = MagicMock()
    conn.fetch = AsyncMock(side_effect=exc)
    conn.fetchrow = AsyncMock(side_effect=exc)

    class FakeAcquire:
        async def __aenter__(self_inner):
            return conn

        async def __aexit__(self_inner, *exc):
            return False

    pool = MagicMock()
    pool.acquire = MagicMock(return_value=FakeAcquire())
    return pool, conn


# ---------------------------------------------------------------------------
# customers
# ---------------------------------------------------------------------------


async def test_customers_returns_rows():
    rows = [
        {"id": 1, "name": "Acme"},
        {"id": 2, "name": "Beta"},
    ]
    pool, _ = _make_pool_with_rows(rows)
    repo = PriorityRepository(pool)
    result = await repo.customers()
    assert result == [
        {"id": "1", "name": "Acme"},
        {"id": "2", "name": "Beta"},
    ]


async def test_customers_postgres_error_raises_priority_error():
    pool, _ = _make_pool_raising(asyncpg.PostgresError("boom"))
    repo = PriorityRepository(pool)
    with pytest.raises(PriorityError, match="unavailable"):
        await repo.customers()


async def test_customers_oserror_raises_priority_error():
    pool, _ = _make_pool_raising(OSError("connection refused"))
    repo = PriorityRepository(pool)
    with pytest.raises(PriorityError, match="unavailable"):
        await repo.customers()


# ---------------------------------------------------------------------------
# branches
# ---------------------------------------------------------------------------


async def test_branches_returns_rows():
    rows = [{"id": 10, "name": "Branch A"}]
    pool, _ = _make_pool_with_rows(rows)
    repo = PriorityRepository(pool)
    result = await repo.branches("C1")
    assert result == [{"id": "10", "name": "Branch A"}]


async def test_branches_postgres_error_raises_priority_error():
    pool, _ = _make_pool_raising(asyncpg.PostgresError("boom"))
    repo = PriorityRepository(pool)
    with pytest.raises(PriorityError, match="unavailable"):
        await repo.branches("C1")


async def test_branches_oserror_raises_priority_error():
    pool, _ = _make_pool_raising(OSError("connection refused"))
    repo = PriorityRepository(pool)
    with pytest.raises(PriorityError, match="unavailable"):
        await repo.branches("C1")


# ---------------------------------------------------------------------------
# create_order
# ---------------------------------------------------------------------------


async def test_create_order_returns_id():
    rows = [{"id": 42}]
    pool, _ = _make_pool_with_rows(rows)
    repo = PriorityRepository(pool)
    order_id = await repo.create_order(
        session_id="sess-1",
        customer_id="C1",
        branch_id="B1",
        action="create_order",
        items=[{"barcode": "VAL1", "qty": 1}],
    )
    assert order_id == 42


async def test_create_order_postgres_error_raises_priority_error():
    pool, _ = _make_pool_raising(asyncpg.PostgresError("boom"))
    repo = PriorityRepository(pool)
    with pytest.raises(PriorityError, match="Priority draft order"):
        await repo.create_order(
            session_id="sess-1",
            customer_id="C1",
            branch_id="B1",
            action="create_order",
            items=[],
        )


async def test_create_order_oserror_raises_priority_error():
    pool, _ = _make_pool_raising(OSError("connection refused"))
    repo = PriorityRepository(pool)
    with pytest.raises(PriorityError, match="Priority draft order"):
        await repo.create_order(
            session_id="sess-1",
            customer_id="C1",
            branch_id="B1",
            action="create_order",
            items=[],
        )

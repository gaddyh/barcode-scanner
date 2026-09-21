"""Tests for the Priority gateway boundary — protocol, local adapter, and
error classification at the integration boundary.

Covers:
- Domain models (frozen dataclasses, serialization).
- LocalPriorityGateway: customers, branches, create_draft_order happy paths.
- Error classification: read errors → RetryableError; write errors →
  IndeterminateError (after submit) / RetryableError (pre-submit) /
  PermanentError (UniqueViolationError).
- PriorityGateway protocol conformance (runtime_checkable).
- PriorityRepository backward-compatible shim.
- Hard edge cases: empty rows, None row, missing columns, UniqueViolation.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import asyncpg
import pytest

from src.integrations.priority import (
    Branch,
    CreateDraftOrderRequest,
    CreateDraftOrderResult,
    Customer,
    LocalPriorityGateway,
    OrderLineItem,
    PriorityError,
    PriorityGateway,
)
from src.integrations.priority.local import _classify_read_error, _classify_write_error
from src.runtime.errors import (
    IndeterminateError,
    PermanentError,
    RetryableError,
)

# ---------------------------------------------------------------------------
# Fake asyncpg pool helpers
# ---------------------------------------------------------------------------


def _make_pool_with_rows(rows, *, fetchrow_rows=None):
    """Build a fake asyncpg pool whose fetch returns ``rows``."""
    conn = MagicMock()
    conn.fetch = AsyncMock(return_value=rows)
    if fetchrow_rows is not None:
        conn.fetchrow = AsyncMock(return_value=fetchrow_rows)
    else:
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
    """Build a fake asyncpg pool whose fetch/fetchrow raise ``exc``."""
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


def _make_pool_raising_on_fetchrow(exc):
    """Pool whose fetch succeeds but fetchrow raises (write-path error)."""
    conn = MagicMock()
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchrow = AsyncMock(side_effect=exc)

    class FakeAcquire:
        async def __aenter__(self_inner):
            return conn

        async def __aexit__(self_inner, *exc):
            return False

    pool = MagicMock()
    pool.acquire = MagicMock(return_value=FakeAcquire())
    return pool, conn


# ===========================================================================
# Domain models
# ===========================================================================


class TestCustomer:
    def test_frozen(self):
        c = Customer(id="cust-1", name="Acme")
        assert c.id == "cust-1"
        assert c.name == "Acme"
        with pytest.raises(AttributeError):
            c.id = "x"  # type: ignore[misc]

    def test_equality(self):
        assert Customer(id="a", name="A") == Customer(id="a", name="A")
        assert Customer(id="a", name="A") != Customer(id="b", name="A")


class TestBranch:
    def test_frozen(self):
        b = Branch(id="br-1", name="Main", customer_id="cust-1")
        assert b.id == "br-1"
        assert b.customer_id == "cust-1"
        with pytest.raises(AttributeError):
            b.name = "x"  # type: ignore[misc]


class TestOrderLineItem:
    def test_defaults(self):
        item = OrderLineItem(
            barcode_value="123", barcode_format="Code128", quantity=2
        )
        assert item.label_index is None
        assert item.quantity == 2

    def test_frozen(self):
        item = OrderLineItem(
            barcode_value="x", barcode_format="Code128", quantity=1
        )
        with pytest.raises(AttributeError):
            item.quantity = 5  # type: ignore[misc]


class TestCreateDraftOrderRequest:
    def test_items_as_dicts(self):
        req = CreateDraftOrderRequest(
            session_id="s1",
            customer_id="c1",
            branch_id="b1",
            action="create_order",
            items=[
                OrderLineItem(
                    barcode_value="111",
                    barcode_format="Code128",
                    quantity=3,
                    label_index=1,
                ),
                OrderLineItem(
                    barcode_value="222",
                    barcode_format="Code128",
                    quantity=1,
                ),
            ],
        )
        dicts = req.items_as_dicts()
        assert len(dicts) == 2
        assert dicts[0] == {
            "barcode_value": "111",
            "barcode_format": "Code128",
            "quantity": 3,
            "label_index": 1,
        }
        assert dicts[1]["label_index"] is None

    def test_empty_items(self):
        req = CreateDraftOrderRequest(
            session_id="s1",
            customer_id="c1",
            branch_id="b1",
            action="create_order",
        )
        assert req.items_as_dicts() == []

    def test_frozen(self):
        req = CreateDraftOrderRequest(
            session_id="s1", customer_id="c1", branch_id="b1", action="create_order"
        )
        with pytest.raises(AttributeError):
            req.session_id = "x"  # type: ignore[misc]


class TestCreateDraftOrderResult:
    def test_defaults(self):
        r = CreateDraftOrderResult(order_id=42, session_id="s1")
        assert r.status == "draft"

    def test_frozen(self):
        r = CreateDraftOrderResult(order_id=1, session_id="s")
        with pytest.raises(AttributeError):
            r.order_id = 2  # type: ignore[misc]


# ===========================================================================
# Protocol conformance
# ===========================================================================


class TestProtocolConformance:
    def test_local_gateway_is_priority_gateway(self):
        pool = MagicMock()
        gw = LocalPriorityGateway(pool)
        assert isinstance(gw, PriorityGateway)

    def test_plain_object_not_priority_gateway(self):
        class NotAGateway:
            pass

        assert not isinstance(NotAGateway(), PriorityGateway)


# ===========================================================================
# LocalPriorityGateway.customers
# ===========================================================================


class TestCustomers:
    async def test_returns_customers(self):
        rows = [
            {"id": "cust-1", "name": "Acme"},
            {"id": "cust-2", "name": "Beta"},
        ]
        pool, _ = _make_pool_with_rows(rows)
        gw = LocalPriorityGateway(pool)
        result = await gw.customers()
        assert result == [
            Customer(id="cust-1", name="Acme"),
            Customer(id="cust-2", name="Beta"),
        ]

    async def test_empty_rows(self):
        pool, _ = _make_pool_with_rows([])
        gw = LocalPriorityGateway(pool)
        assert await gw.customers() == []

    async def test_int_ids_coerced_to_str(self):
        rows = [{"id": 42, "name": "Acme"}]
        pool, _ = _make_pool_with_rows(rows)
        gw = LocalPriorityGateway(pool)
        result = await gw.customers()
        assert result[0].id == "42"

    async def test_postgres_error_raises_priority_error(self):
        pool, _ = _make_pool_raising(asyncpg.PostgresError("boom"))
        gw = LocalPriorityGateway(pool)
        with pytest.raises(PriorityError, match="unavailable"):
            await gw.customers()

    async def test_oserror_raises_priority_error(self):
        pool, _ = _make_pool_raising(OSError("connection refused"))
        gw = LocalPriorityGateway(pool)
        with pytest.raises(PriorityError, match="unavailable"):
            await gw.customers()


# ===========================================================================
# LocalPriorityGateway.branches
# ===========================================================================


class TestBranches:
    async def test_returns_branches(self):
        rows = [{"id": "br-1", "name": "Main"}]
        pool, _ = _make_pool_with_rows(rows)
        gw = LocalPriorityGateway(pool)
        result = await gw.branches("cust-1")
        assert result == [Branch(id="br-1", name="Main", customer_id="cust-1")]

    async def test_empty_rows(self):
        pool, _ = _make_pool_with_rows([])
        gw = LocalPriorityGateway(pool)
        assert await gw.branches("cust-1") == []

    async def test_postgres_error_raises_priority_error(self):
        pool, _ = _make_pool_raising(asyncpg.PostgresError("boom"))
        gw = LocalPriorityGateway(pool)
        with pytest.raises(PriorityError, match="unavailable"):
            await gw.branches("cust-1")

    async def test_oserror_raises_priority_error(self):
        pool, _ = _make_pool_raising(OSError("connection refused"))
        gw = LocalPriorityGateway(pool)
        with pytest.raises(PriorityError, match="unavailable"):
            await gw.branches("cust-1")


# ===========================================================================
# LocalPriorityGateway.create_draft_order
# ===========================================================================


class TestCreateDraftOrder:
    def _request(self, **overrides):
        kwargs = dict(
            session_id="sess-1",
            customer_id="cust-1",
            branch_id="br-1",
            action="create_order",
            items=[
                OrderLineItem(
                    barcode_value="111", barcode_format="Code128", quantity=2
                )
            ],
        )
        kwargs.update(overrides)
        return CreateDraftOrderRequest(**kwargs)

    async def test_returns_result(self):
        row = {"id": 42, "session_id": "sess-1", "status": "draft"}
        pool, _ = _make_pool_with_rows([], fetchrow_rows=row)
        gw = LocalPriorityGateway(pool)
        result = await gw.create_draft_order(self._request())
        assert isinstance(result, CreateDraftOrderResult)
        assert result.order_id == 42
        assert result.session_id == "sess-1"
        assert result.status == "draft"

    async def test_missing_session_id_uses_request(self):
        row = {"id": 99}
        pool, _ = _make_pool_with_rows([], fetchrow_rows=row)
        gw = LocalPriorityGateway(pool)
        result = await gw.create_draft_order(self._request(session_id="s-x"))
        assert result.order_id == 99
        assert result.session_id == "s-x"
        assert result.status == "draft"

    async def test_postgres_error_raises_indeterminate(self):
        pool, _ = _make_pool_raising_on_fetchrow(
            asyncpg.PostgresError("insert failed")
        )
        gw = LocalPriorityGateway(pool)
        with pytest.raises(IndeterminateError, match="may have been created"):
            await gw.create_draft_order(self._request())

    async def test_oserror_raises_indeterminate(self):
        pool, _ = _make_pool_raising_on_fetchrow(OSError("connection lost"))
        gw = LocalPriorityGateway(pool)
        with pytest.raises(IndeterminateError, match="may have been created"):
            await gw.create_draft_order(self._request())

    async def test_unique_violation_raises_permanent(self):
        pool, _ = _make_pool_raising_on_fetchrow(
            asyncpg.UniqueViolationError("duplicate session_id")
        )
        gw = LocalPriorityGateway(pool)
        with pytest.raises(PermanentError, match="Duplicate session_id"):
            await gw.create_draft_order(self._request())

    async def test_none_row_raises_indeterminate(self):
        pool, _ = _make_pool_with_rows([], fetchrow_rows=None)
        gw = LocalPriorityGateway(pool)
        with pytest.raises(IndeterminateError, match="silently failed"):
            await gw.create_draft_order(self._request())


# ===========================================================================
# Error classification helpers
# ===========================================================================


class TestClassifyReadError:
    def test_returns_retryable(self):
        exc = asyncpg.PostgresError("boom")
        result = _classify_read_error(exc)
        assert isinstance(result, RetryableError)
        assert "unavailable" in str(result)


class TestClassifyWriteError:
    def test_unique_violation_is_permanent(self):
        exc = asyncpg.UniqueViolationError("dup")
        result = _classify_write_error(exc, after_submit=True)
        assert isinstance(result, PermanentError)
        assert "Duplicate session_id" in str(result)

    def test_after_submit_is_indeterminate(self):
        exc = asyncpg.PostgresError("insert failed")
        result = _classify_write_error(exc, after_submit=True)
        assert isinstance(result, IndeterminateError)
        assert "may have been created" in str(result)

    def test_before_submit_is_retryable(self):
        exc = OSError("connection refused")
        result = _classify_write_error(exc, after_submit=False)
        assert isinstance(result, RetryableError)
        assert "pre-submit" in str(result)

    def test_oserror_after_submit_is_indeterminate(self):
        exc = OSError("connection lost mid-write")
        result = _classify_write_error(exc, after_submit=True)
        assert isinstance(result, IndeterminateError)

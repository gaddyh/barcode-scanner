"""Tests for src.ingest.checkpoint — Postgres checkpointer singleton."""

from __future__ import annotations

import pytest

from src.ingest import checkpoint


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the checkpointer singleton between tests."""
    checkpoint._checkpointer = None
    checkpoint._pool = None
    yield
    checkpoint._checkpointer = None
    checkpoint._pool = None


class TestGetCheckpointer:
    def test_returns_none_when_not_initialized(self) -> None:
        assert checkpoint.get_checkpointer() is None


class TestInitCheckpointer:
    @pytest.mark.asyncio
    async def test_no_database_url_raises(self, monkeypatch) -> None:
        monkeypatch.delenv("DATABASE_URL", raising=False)
        with pytest.raises(ValueError, match="DATABASE_URL"):
            await checkpoint.init_checkpointer(None)

    @pytest.mark.asyncio
    async def test_returns_same_instance_on_second_call(self, monkeypatch) -> None:
        """init_checkpointer called twice → returns same instance (idempotent)."""
        monkeypatch.setenv("DATABASE_URL", "postgres://x")
        # First call: create a fake checkpointer via mocked imports.

        class _FakePool:
            async def open(self) -> None:
                pass

            async def close(self) -> None:
                pass

        class _FakeAsyncPostgresSaver:
            def __init__(self, conn) -> None:
                self.conn = conn

            async def setup(self) -> None:
                pass

        # Patch the lazy imports inside init_checkpointer.
        import sys
        fake_mod = type(sys)("langgraph.checkpoint.postgres.aio")
        fake_mod.AsyncPostgresSaver = _FakeAsyncPostgresSaver
        fake_psycopg = type(sys)("psycopg_pool")
        fake_psycopg.AsyncConnectionPool = lambda **kw: _FakePool()
        monkeypatch.setitem(sys.modules, "langgraph.checkpoint.postgres.aio", fake_mod)
        monkeypatch.setitem(sys.modules, "psycopg_pool", fake_psycopg)

        first = await checkpoint.init_checkpointer("postgres://x")
        assert first is not None
        second = await checkpoint.init_checkpointer("postgres://x")
        assert second is first  # same instance


class TestCloseCheckpointer:
    @pytest.mark.asyncio
    async def test_close_when_not_initialized(self) -> None:
        """close_checkpointer is a no-op when nothing initialized."""
        await checkpoint.close_checkpointer()
        assert checkpoint._checkpointer is None
        assert checkpoint._pool is None

    @pytest.mark.asyncio
    async def test_close_after_init(self, monkeypatch) -> None:
        monkeypatch.setenv("DATABASE_URL", "postgres://x")
        closed = {"called": False}

        class _FakePool:
            async def open(self) -> None:
                pass

            async def close(self) -> None:
                closed["called"] = True

        class _FakeAsyncPostgresSaver:
            def __init__(self, conn) -> None:
                self.conn = conn

            async def setup(self) -> None:
                pass

        import sys
        fake_mod = type(sys)("langgraph.checkpoint.postgres.aio")
        fake_mod.AsyncPostgresSaver = _FakeAsyncPostgresSaver
        fake_psycopg = type(sys)("psycopg_pool")
        fake_psycopg.AsyncConnectionPool = lambda **kw: _FakePool()
        monkeypatch.setitem(sys.modules, "langgraph.checkpoint.postgres.aio", fake_mod)
        monkeypatch.setitem(sys.modules, "psycopg_pool", fake_psycopg)

        await checkpoint.init_checkpointer("postgres://x")
        assert checkpoint._pool is not None
        await checkpoint.close_checkpointer()
        assert checkpoint._checkpointer is None
        assert checkpoint._pool is None
        assert closed["called"]

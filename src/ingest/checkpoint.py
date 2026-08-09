"""Postgres-backed LangGraph checkpoint store.

Provides a singleton ``AsyncPostgresSaver`` so the scan graph can persist
intermediate state (scan results, audit results, reconciliation) to the
operational Postgres database. This enables:

- **Resume after interruption** — if the graph is interrupted (e.g. for
  human-in-the-loop review in M15D), the checkpoint lets us reload state
  and continue from where we left off.
- **Debugging** — checkpoint tables show the exact state at every superstep.

The checkpointer uses a ``psycopg`` ``AsyncConnectionPool`` (separate from
the ``asyncpg`` pool used by the operational repository) because
``AsyncPostgresSaver`` requires psycopg's binary cursor protocol.

Usage::

    from src.ingest.checkpoint import get_checkpointer, init_checkpointer

    # On app startup:
    await init_checkpointer(database_url)

    # In run_scan_graph:
    checkpointer = get_checkpointer()
    graph = build_scan_graph().compile(checkpointer=checkpointer)
    await graph.ainvoke(state, config={"configurable": {"thread_id": upload_id}})
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

logger = logging.getLogger(__name__)

# Singleton checkpointer — initialized once on app startup.
_checkpointer: AsyncPostgresSaver | None = None
# Singleton connection pool — kept open for the lifetime of the app.
_pool = None  # type: psycopg.AsyncConnectionPool | None


async def init_checkpointer(database_url: str | None = None) -> AsyncPostgresSaver:
    """Initialize the Postgres checkpointer singleton.

    Creates an ``AsyncConnectionPool`` and ``AsyncPostgresSaver``, runs
    ``setup()`` to create the checkpoint tables (idempotent), and stores
    both as module-level singletons.

    Safe to call multiple times — returns the existing instance if already
    initialized.

    Args:
        database_url: Postgres connection string. Defaults to
            ``DATABASE_URL`` env var.

    Returns:
        The initialized ``AsyncPostgresSaver``.
    """
    global _checkpointer, _pool

    if _checkpointer is not None:
        return _checkpointer

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg_pool import AsyncConnectionPool

    url = database_url or os.getenv("DATABASE_URL")
    if not url:
        raise ValueError(
            "DATABASE_URL is required for checkpointing. "
            "Pass database_url=... or set DATABASE_URL env var."
        )

    # psycopg needs autocommit + prepare_threshold=0 for the checkpointer.
    # min_size=1 is plenty — checkpoint writes are serial per graph run.
    _pool = AsyncConnectionPool(
        conninfo=url,
        min_size=1,
        max_size=5,
        kwargs={"autocommit": True, "prepare_threshold": 0},
        open=False,
    )
    await _pool.open()

    _checkpointer = AsyncPostgresSaver(conn=_pool)
    await _checkpointer.setup()

    logger.info("Postgres checkpointer initialized (psycopg pool, %s)", url[:40])
    return _checkpointer


def get_checkpointer() -> AsyncPostgresSaver | None:
    """Return the singleton checkpointer, or ``None`` if not initialized.

    Callers should handle ``None`` gracefully — when checkpointing is not
    configured (e.g. CLI, tests, local dev without DATABASE_URL), the graph
    runs without a checkpointer (no persistence, no resume).
    """
    return _checkpointer


async def close_checkpointer() -> None:
    """Close the checkpointer and connection pool. Called on app shutdown."""
    global _checkpointer, _pool

    if _pool is not None:
        await _pool.close()
        _pool = None
    _checkpointer = None
    logger.info("Postgres checkpointer closed")

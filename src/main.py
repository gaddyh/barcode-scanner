"""FastAPI application: barcode image analysis web API.

Serves the React frontend (web/dist) and the barcode-scanner API:
- /receiving/* — scan-first receiving flow (create, upload, context, submit)
- /barcode/* — scanner-only and full-pipeline endpoints
- /customers, /branches — Priority customer/branch lookup
- /admin/* — metrics dashboard
- /health — health check
- / — static React frontend (when web/dist exists)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.admin import router as admin_router
from src.api.receiving import router as receiving_router
from src.api.routes import router
from src.config import settings
from src.db import create_pool, init_db
from src.repository import (
    AnnotationRepository,
    NoOpAnnotationRepository,
    NoOpRunRepository,
    PostgresAnnotationRepository,
    PostgresRunRepository,
    RunRepository,
)

logger = logging.getLogger(__name__)

# Module-level repository singletons — set by lifespan, read by request handlers.
run_repo: RunRepository = NoOpRunRepository()
annotation_repo: AnnotationRepository = NoOpAnnotationRepository()
_db_pool = None
logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO)
)

for noisy in ("httpx", "httpcore", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global run_repo, annotation_repo, _db_pool

    # Initialize Postgres pool if DATABASE_URL is set
    if settings.database_url:
        logger.info("Initializing Postgres pool: %s", settings.database_url[:30] + "...")
        _db_pool = await create_pool(settings.database_url)
        await init_db(_db_pool)
        run_repo = PostgresRunRepository(_db_pool)
        annotation_repo = PostgresAnnotationRepository(_db_pool)
        logger.info("Postgres operational DB ready")

        # Initialize LangGraph checkpointer (M15C) — uses a separate psycopg
        # pool for the checkpoint tables in the same Postgres instance.
        # Non-fatal: if it fails, the app boots without checkpointing.
        from src.ingest.checkpoint import close_checkpointer, init_checkpointer

        try:
            await init_checkpointer(settings.database_url)
            logger.info("LangGraph checkpointer ready")
        except Exception:
            logger.warning("LangGraph checkpointer init failed — running without persistence", exc_info=True)
    else:
        logger.info("No DATABASE_URL — using NoOp repository (no persistence)")

    # Optional Gemini audit cache mode for deterministic full-pipeline runs
    # (sanity script, eval replay). When set to "replay", the graph uses
    # the frozen cached Gemini results from tests/eval/gemini_audit_cache.json
    # instead of calling Gemini live — making the full pipeline deterministic.
    audit_cache_mode = os.getenv("GEMINI_AUDIT_CACHE_MODE", "").strip().lower()
    if audit_cache_mode in ("replay", "capture"):
        from src.evals.gemini_cache import GeminiAuditCache
        from src.ingest import graph

        cache = GeminiAuditCache()
        graph.set_audit_cache_mode(audit_cache_mode, cache)
        logger.info(
            "Gemini audit cache mode=%s entries=%d", audit_cache_mode, len(cache)
        )

    yield

    if _db_pool is not None:
        await _db_pool.close()
    if settings.database_url:
        await close_checkpointer()


app = FastAPI(lifespan=lifespan)

# CORS: allow any origin so the local Vite dev server and ngrok tunnels can
# call the API from the browser. Safe because this app uses no credentials.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(admin_router)
app.include_router(receiving_router)

# NOTE: The static files mount at "/" must be added AFTER all API routes.
# Starlette matches routes in registration order, and a Mount("/") catch-all
# would shadow any route added after it. The mount is therefore registered
# at the very end of this module.


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Static frontend — must be mounted LAST so it doesn't shadow API routes.
# ---------------------------------------------------------------------------

# Serve the built React frontend from web/dist/ when it exists (Docker/prod).
# In local dev, Vite serves the frontend separately on :5173 and this dir is
# absent, so we skip the mount.
_static_dir = Path(__file__).resolve().parent.parent / "web" / "dist"
if _static_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="frontend")

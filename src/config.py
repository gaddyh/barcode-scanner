"""Single source of truth for application settings.

This module loads environment variables via python-dotenv and exposes a
frozen ``Settings`` dataclass plus a module-level ``settings`` instance.

It consolidates configuration for:
- LangSmith tracing (``langsmith_*``)
- Barcode-scanner product API (``app_name``, ``app_env``,
  ``max_upload_bytes``, ``allowed_image_types``)
- Database (``database_url``)
- Logging (``log_level``)

``src.config`` re-exports ``Settings`` / ``get_settings`` /
``settings`` from here so the FastAPI routes in ``src/api/routes.py``
keep working through one source of truth.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # --- LangSmith -------------------------------------------------------
    langsmith_api_key: str
    langsmith_project: str
    langsmith_tracing: bool

    # --- Logging / general ----------------------------------------------
    log_level: str

    # --- Barcode-scanner product API ------------------------------------
    app_name: str
    app_env: str
    max_upload_bytes: int
    allowed_image_types: str

    # --- Database -------------------------------------------------------
    database_url: str

    @property
    def allowed_content_types(self) -> set[str]:
        return {
            content_type.strip().lower()
            for content_type in self.allowed_image_types.split(",")
            if content_type.strip()
        }

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            langsmith_api_key=os.getenv("LANGSMITH_API_KEY", "").strip(),
            langsmith_project=os.getenv("LANGSMITH_PROJECT", "barcode-scanner").strip(),
            langsmith_tracing=(
                os.getenv("LANGSMITH_TRACING", "false").strip().lower() == "true"
            ),
            database_url=os.getenv("DATABASE_URL", "").strip(),
            app_name=os.getenv("APP_NAME", "Barcode Scanner Service"),
            app_env=os.getenv("APP_ENV", "development"),
            max_upload_bytes=int(os.getenv("MAX_UPLOAD_BYTES", str(15 * 1024 * 1024))),
            allowed_image_types=os.getenv(
                "ALLOWED_IMAGE_TYPES",
                "image/jpeg,image/png,image/webp",
            ),
        )


settings = Settings.from_env()


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance.

    Re-exported by ``src.config`` for FastAPI ``Depends`` consumers.
    """
    return settings

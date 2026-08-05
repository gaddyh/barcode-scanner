"""Single source of truth for application settings.

This module loads environment variables via python-dotenv and exposes a
frozen ``Settings`` dataclass plus a module-level ``settings`` instance.

It consolidates configuration for:
- 360dialog WhatsApp integration (``d360_*``)
- Webhook authentication (``webhook_*``)
- Transcription providers (``openai_*``, ``modal_transcription_*``,
  ``transcription_provider``)
- LangSmith tracing (``langsmith_*``)
- Barcode-scanner product API (``app_name``, ``app_env``,
  ``max_upload_bytes``, ``allowed_image_types``)
- Conversation / review tuning (used by the agent layer, kept for
  forward compatibility)

``app.core.config`` re-exports ``Settings`` / ``get_settings`` /
``settings`` from here so the FastAPI routes in ``app/api/routes.py``
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
    # --- 360dialog WhatsApp ----------------------------------------------
    d360_api_key: str
    d360_api_base_url: str
    webhook_auth_mode: str
    webhook_bearer_token: str
    webhook_basic_user: str
    webhook_basic_pass: str

    # --- OpenAI / transcription ------------------------------------------
    openai_api_key: str
    openai_transcribe_model: str
    openai_model: str
    transcription_provider: str
    modal_transcription_url: str
    modal_transcription_key: str
    modal_transcription_secret: str
    modal_transcription_timeout_seconds: int

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

    # --- Conversation / review (forward-compat for agent layer) ---------
    conversation_max_messages: int
    allowed_chat_ids: list[str]
    database_path: str
    database_url: str
    max_group_participants: int
    session_gap_minutes: int
    conversation_dormant_hours: int
    conversation_closed_days: int
    waiting_reply_hours: int
    max_extraction_attempts: int
    conversation_history_context_messages: int
    expected_authorization_header: str
    tenant_timezone: str
    compiled_agent_path: str

    # --- Review keywords (have defaults, must come last) ----------------
    review_start_keyword: str = "בוא"
    review_end_keyword: str = "בטל"
    review_clarification_limit: int = 3

    @property
    def allowed_content_types(self) -> set[str]:
        return {
            content_type.strip().lower()
            for content_type in self.allowed_image_types.split(",")
            if content_type.strip()
        }

    @classmethod
    def from_env(cls) -> Settings:
        api_key = os.getenv("D360_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "Missing D360_API_KEY. Copy example.env to .env and set it."
            )

        auth_mode = os.getenv("WEBHOOK_AUTH_MODE", "none").strip().lower()
        if auth_mode not in {"none", "bearer", "basic"}:
            raise RuntimeError(
                "WEBHOOK_AUTH_MODE must be one of: none, bearer, basic"
            )

        return cls(
            d360_api_key=api_key,
            d360_api_base_url=os.getenv(
                "D360_API_BASE_URL",
                "https://waba-v2.360dialog.io",
            ).rstrip("/"),
            webhook_auth_mode=auth_mode,
            webhook_bearer_token=os.getenv("WEBHOOK_BEARER_TOKEN", "").strip(),
            webhook_basic_user=os.getenv("WEBHOOK_BASIC_USER", "").strip(),
            webhook_basic_pass=os.getenv("WEBHOOK_BASIC_PASS", "").strip(),
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            openai_transcribe_model=os.getenv(
                "OPENAI_TRANSCRIBE_MODEL",
                "gpt-4o-transcribe",
            ).strip(),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini").strip(),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            langsmith_api_key=os.getenv("LANGSMITH_API_KEY", "").strip(),
            langsmith_project=os.getenv("LANGSMITH_PROJECT", "echo2").strip(),
            langsmith_tracing=(
                os.getenv("LANGSMITH_TRACING_V2", "false").strip().lower()
                == "true"
            ),
            conversation_max_messages=int(
                os.getenv("CONVERSATION_MAX_MESSAGES", "20")
            ),
            allowed_chat_ids=[
                chat_id.strip()
                for chat_id in os.getenv("ALLOWED_CHAT_IDS", "").split(",")
                if chat_id.strip()
            ],
            database_path=os.getenv("DATABASE_PATH", "tami.db"),
            database_url=os.getenv("DATABASE_URL", "").strip(),
            max_group_participants=int(os.getenv("MAX_GROUP_PARTICIPANTS", "3")),
            session_gap_minutes=int(os.getenv("SESSION_GAP_MINUTES", "45")),
            conversation_dormant_hours=int(
                os.getenv("CONVERSATION_DORMANT_HOURS", "24")
            ),
            conversation_closed_days=int(
                os.getenv("CONVERSATION_CLOSED_DAYS", "7")
            ),
            waiting_reply_hours=int(os.getenv("WAITING_REPLY_HOURS", "4")),
            max_extraction_attempts=int(
                os.getenv("MAX_EXTRACTION_ATTEMPTS", "3")
            ),
            conversation_history_context_messages=int(
                os.getenv("CONVERSATION_HISTORY_CONTEXT_MESSAGES", "10")
            ),
            expected_authorization_header=os.getenv(
                "EXPECTED_AUTHORIZATION_HEADER", ""
            ).strip(),
            tenant_timezone=os.getenv("TENANT_TIMEZONE", "Asia/Jerusalem").strip(),
            compiled_agent_path=os.getenv(
                "COMPILED_AGENT_PATH", "compiled_agent.json"
            ).strip(),
            transcription_provider=os.getenv(
                "TRANSCRIPTION_PROVIDER", "openai"
            ).strip().lower(),
            modal_transcription_url=os.getenv(
                "MODAL_TRANSCRIPTION_URL", ""
            ).strip(),
            modal_transcription_key=os.getenv(
                "MODAL_TRANSCRIPTION_KEY", ""
            ).strip(),
            modal_transcription_secret=os.getenv(
                "MODAL_TRANSCRIPTION_SECRET", ""
            ).strip(),
            modal_transcription_timeout_seconds=int(
                os.getenv("MODAL_TRANSCRIPTION_TIMEOUT_SECONDS", "180")
            ),
            review_start_keyword=os.getenv("REVIEW_START_KEYWORD", "בוא").strip(),
            review_end_keyword=os.getenv("REVIEW_END_KEYWORD", "בטל").strip(),
            review_clarification_limit=int(
                os.getenv("REVIEW_CLARIFICATION_LIMIT", "3")
            ),
            # --- Barcode-scanner product API ---
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

    Re-exported by ``app.core.config`` for FastAPI ``Depends`` consumers.
    """
    return settings

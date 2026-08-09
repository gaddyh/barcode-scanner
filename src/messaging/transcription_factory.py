from __future__ import annotations

from src.config import Settings
from src.messaging.transcribe import OpenAITranscriber, Transcriber


def get_transcriber(settings: Settings) -> Transcriber:
    """
    Construct a Transcriber from centralized settings.

    This is the composition root — the only place that decides which
    provider to use.  Callers receive a Transcriber and should not
    know which implementation they got.
    """
    provider = settings.transcription_provider

    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError(
                "TRANSCRIPTION_PROVIDER=openai but OPENAI_API_KEY is not set"
            )
        return OpenAITranscriber(
            api_key=settings.openai_api_key,
            model=settings.openai_transcribe_model,
        )

    if provider == "modal":
        if not settings.modal_transcription_url:
            raise RuntimeError(
                "TRANSCRIPTION_PROVIDER=modal but MODAL_TRANSCRIPTION_URL is not set"
            )
        if not settings.modal_transcription_key or not settings.modal_transcription_secret:
            raise RuntimeError(
                "TRANSCRIPTION_PROVIDER=modal but MODAL_TRANSCRIPTION_KEY and/or "
                "MODAL_TRANSCRIPTION_SECRET are not set"
            )

        # Lazy import so that the Modal client is only loaded when needed.
        from src.messaging.modal_transcriber import ModalWhisperTranscriber

        return ModalWhisperTranscriber(
            endpoint_url=settings.modal_transcription_url,
            key=settings.modal_transcription_key,
            secret=settings.modal_transcription_secret,
            timeout_seconds=settings.modal_transcription_timeout_seconds,
        )

    raise RuntimeError(
        f"Unsupported TRANSCRIPTION_PROVIDER={provider!r}. "
        "Supported values: 'openai', 'modal'."
    )

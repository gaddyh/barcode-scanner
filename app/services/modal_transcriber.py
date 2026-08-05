from __future__ import annotations

import logging
from typing import Any

from app.services.modal_client import (
    ModalTranscriptionClient,
    ModalTranscriptionTransportError,
)
from app.services.transcribe import (
    Transcriber,
    TranscriptionError,
    TranscriptionResult,
)


logger = logging.getLogger(__name__)


class ModalWhisperTranscriber:
    """
    Adapter that implements the Transcriber protocol by delegating
    to ModalTranscriptionClient (raw HTTP transport).

    Translates transport-specific errors into the provider-neutral
    TranscriptionError.  Maps raw JSON into TranscriptionResult.
    """

    def __init__(self, *, endpoint_url: str, key: str, secret: str, timeout_seconds: float = 180.0) -> None:
        self._client = ModalTranscriptionClient(
            endpoint_url=endpoint_url,
            key=key,
            secret=secret,
            timeout_seconds=timeout_seconds,
        )

    async def transcribe_bytes(
        self,
        *,
        audio_bytes: bytes,
        filename: str,
        content_type: str,
    ) -> TranscriptionResult:
        try:
            payload = await self._client.transcribe_bytes(
                audio_bytes=audio_bytes,
                filename=filename,
                content_type=content_type,
            )
        except ModalTranscriptionTransportError as exc:
            logger.error("Modal transcription failed: %s", exc)
            raise TranscriptionError(str(exc)) from exc

        text = payload.get("text")

        if not isinstance(text, str):
            raise TranscriptionError(
                "Modal response does not contain a valid text field"
            )

        return TranscriptionResult(
            text=text.strip(),
            language=payload.get("language"),
            audio_duration_seconds=payload.get("audio_duration_seconds"),
            processing_seconds=payload.get("processing_seconds"),
            model=str(payload.get("model", "unknown")),
            raw=payload,
        )

    async def close(self) -> None:
        await self._client.close()

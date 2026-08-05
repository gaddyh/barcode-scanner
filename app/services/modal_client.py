from __future__ import annotations

import asyncio
import random
import time
from pathlib import Path
from typing import Any

import httpx


class ModalTranscriptionTransportError(RuntimeError):
    """Transport-level error from the Modal HTTP client.

    This is internal to the transport layer.  The adapter layer
    (ModalWhisperTranscriber) translates it into the provider-neutral
    TranscriptionError.
    """


_RETRYABLE_STATUS_CODES = {502, 503, 504}
_MAX_RETRIES = 1


class ModalTranscriptionClient:
    """
    Raw HTTP transport for the Modal Hebrew Whisper transcription endpoint.

    This class knows nothing about Transcriber protocols or TranscriptionResult.
    It sends audio bytes and returns a raw JSON dict.

    Authentication uses Modal proxy headers (Modal-Key, Modal-Secret),
    not a custom bearer token.  The endpoint must be deployed with
    requires_proxy_auth=True.
    """

    def __init__(
        self,
        *,
        endpoint_url: str,
        key: str,
        secret: str,
        timeout_seconds: float = 180.0,
        httpx_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._endpoint_url = endpoint_url
        self._key = key
        self._secret = secret
        self._timeout = httpx.Timeout(timeout_seconds, connect=20.0)
        self._owns_client = httpx_client is None
        self._client = httpx_client or httpx.AsyncClient(
            timeout=self._timeout,
            follow_redirects=True,
        )

    async def transcribe_bytes(
        self,
        *,
        audio_bytes: bytes,
        filename: str,
        content_type: str,
        beam_size: int = 1,
        vad_filter: bool = True,
    ) -> dict[str, Any]:
        """
        Send raw audio bytes to the Modal endpoint and return the raw JSON response.

        Raises ModalTranscriptionTransportError on transport failures.
        """
        if not audio_bytes:
            raise ModalTranscriptionTransportError("audio_bytes cannot be empty")

        params = {
            "beam_size": str(beam_size),
            "vad_filter": str(vad_filter).lower(),
        }

        headers = {
            "Modal-Key": self._key,
            "Modal-Secret": self._secret,
            "Content-Type": content_type,
            "X-Filename": Path(filename).name,
        }

        last_error: Exception | None = None

        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = await self._client.post(
                    self._endpoint_url,
                    params=params,
                    headers=headers,
                    content=audio_bytes,
                )
            except httpx.ConnectError as exc:
                last_error = exc
                if attempt < _MAX_RETRIES:
                    await self._backoff(attempt)
                    continue
                raise ModalTranscriptionTransportError(
                    f"Modal connection failed: {exc}"
                ) from exc
            except httpx.TimeoutException as exc:
                raise ModalTranscriptionTransportError(
                    "Modal transcription request timed out"
                ) from exc
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt < _MAX_RETRIES:
                    await self._backoff(attempt)
                    continue
                raise ModalTranscriptionTransportError(
                    f"Modal transcription request failed: {exc}"
                ) from exc

            if response.status_code in _RETRYABLE_STATUS_CODES:
                last_error = ModalTranscriptionTransportError(
                    f"Modal returned retryable {response.status_code}"
                )
                if attempt < _MAX_RETRIES:
                    await self._backoff(attempt)
                    continue
                raise last_error

            if response.status_code >= 400:
                try:
                    error_body = response.json()
                except ValueError:
                    error_body = response.text[:1_000]

                raise ModalTranscriptionTransportError(
                    f"Modal returned {response.status_code}: {error_body}"
                )

            try:
                return response.json()
            except ValueError as exc:
                raise ModalTranscriptionTransportError(
                    "Modal returned invalid JSON"
                ) from exc

        # Should not reach here, but satisfy type checkers.
        raise ModalTranscriptionTransportError(
            f"Modal transcription failed after retries: {last_error}"
        )

    async def close(self) -> None:
        if self._owns_client and self._client:
            await self._client.aclose()

    async def _backoff(self, attempt: int) -> None:
        base = 0.5 * (2 ** attempt)
        jitter = random.uniform(0, base * 0.5)
        await asyncio.sleep(base + jitter)

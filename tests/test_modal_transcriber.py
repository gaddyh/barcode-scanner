"""Tests for src.messaging.modal_transcriber — Modal Whisper adapter."""

from __future__ import annotations

import pytest

from src.messaging.modal_transcriber import ModalWhisperTranscriber
from src.messaging.transcribe import TranscriptionError


class _FakeClient:
    """Fake ModalTranscriptionClient for testing."""

    def __init__(
        self,
        *,
        payload: dict | None = None,
        transport_error: Exception | None = None,
    ) -> None:
        self._payload = payload
        self._transport_error = transport_error
        self.closed = False

    async def transcribe_bytes(
        self, *, audio_bytes: bytes, filename: str, content_type: str
    ) -> dict:
        if self._transport_error is not None:
            raise self._transport_error
        return self._payload or {}

    async def close(self) -> None:
        self.closed = True


def _make_transcriber(client: _FakeClient) -> ModalWhisperTranscriber:
    t = ModalWhisperTranscriber.__new__(ModalWhisperTranscriber)
    t._client = client
    return t


class TestTranscribeBytes:
    @pytest.mark.asyncio
    async def test_success(self) -> None:
        client = _FakeClient(payload={
            "text": "  hello world  ",
            "language": "en",
            "audio_duration_seconds": 5.0,
            "processing_seconds": 1.0,
            "model": "whisper-1",
        })
        t = _make_transcriber(client)
        result = await t.transcribe_bytes(
            audio_bytes=b"x", filename="a.wav", content_type="audio/wav",
        )
        assert result.text == "hello world"  # stripped
        assert result.language == "en"
        assert result.audio_duration_seconds == 5.0
        assert result.model == "whisper-1"

    @pytest.mark.asyncio
    async def test_transport_error_raises_transcription_error(self) -> None:
        from src.messaging.modal_client import ModalTranscriptionTransportError
        client = _FakeClient(
            transport_error=ModalTranscriptionTransportError("network down"),
        )
        t = _make_transcriber(client)
        with pytest.raises(TranscriptionError, match="network down"):
            await t.transcribe_bytes(
                audio_bytes=b"x", filename="a.wav", content_type="audio/wav",
            )

    @pytest.mark.asyncio
    async def test_missing_text_raises(self) -> None:
        client = _FakeClient(payload={"language": "en"})  # no "text"
        t = _make_transcriber(client)
        with pytest.raises(TranscriptionError, match="valid text field"):
            await t.transcribe_bytes(
                audio_bytes=b"x", filename="a.wav", content_type="audio/wav",
            )

    @pytest.mark.asyncio
    async def test_text_not_string_raises(self) -> None:
        client = _FakeClient(payload={"text": 123})
        t = _make_transcriber(client)
        with pytest.raises(TranscriptionError, match="valid text field"):
            await t.transcribe_bytes(
                audio_bytes=b"x", filename="a.wav", content_type="audio/wav",
            )

    @pytest.mark.asyncio
    async def test_empty_text_string_returns_empty(self) -> None:
        client = _FakeClient(payload={"text": ""})
        t = _make_transcriber(client)
        result = await t.transcribe_bytes(
            audio_bytes=b"x", filename="a.wav", content_type="audio/wav",
        )
        assert result.text == ""

    @pytest.mark.asyncio
    async def test_default_model_when_missing(self) -> None:
        client = _FakeClient(payload={"text": "hi"})  # no "model"
        t = _make_transcriber(client)
        result = await t.transcribe_bytes(
            audio_bytes=b"x", filename="a.wav", content_type="audio/wav",
        )
        assert result.model == "unknown"

    @pytest.mark.asyncio
    async def test_raw_payload_preserved(self) -> None:
        payload = {"text": "hi", "extra": "field"}
        client = _FakeClient(payload=payload)
        t = _make_transcriber(client)
        result = await t.transcribe_bytes(
            audio_bytes=b"x", filename="a.wav", content_type="audio/wav",
        )
        assert result.raw == payload


class TestClose:
    @pytest.mark.asyncio
    async def test_close_delegates(self) -> None:
        client = _FakeClient()
        t = _make_transcriber(client)
        await t.close()
        assert client.closed

import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import httpx
import langsmith as ls
from langsmith.wrappers import wrap_openai
from openai import AsyncOpenAI

from src.config import settings
from src.messaging.dialog360 import Dialog360Client

logger = logging.getLogger(__name__)

SUPPORTED_TRANSCRIPTION_EXTS = {
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".m4a",
    ".wav",
    ".webm",
}


def suffix_from_mime(mime_type: str) -> str:
    normalized = mime_type.split(";")[0].strip().lower()

    mapping = {
        "audio/aac": ".aac",
        "audio/amr": ".amr",
        "audio/ogg": ".ogg",
        "audio/opus": ".ogg",
        "audio/mpeg": ".mpeg",
        "audio/mpga": ".mpga",
        "audio/mp4": ".mp4",
        "audio/wav": ".wav",
        "audio/webm": ".webm",
    }

    return mapping.get(normalized, ".bin")


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str | None = None
    audio_duration_seconds: float | None = None
    processing_seconds: float | None = None
    model: str = "unknown"
    raw: dict[str, Any] = field(default_factory=dict)


class TranscriptionError(RuntimeError):
    """Provider-neutral transcription failure."""


@runtime_checkable
class Transcriber(Protocol):
    async def transcribe_bytes(
        self,
        *,
        audio_bytes: bytes,
        filename: str,
        content_type: str,
    ) -> TranscriptionResult: ...

    async def close(self) -> None: ...


def sanitize_transcription_inputs(
    inputs: dict[str, Any],
) -> dict[str, Any]:
    """
    Do not upload raw audio bytes to LangSmith.

    Keep only operational metadata useful for filtering and debugging.
    """
    audio_bytes = inputs.get("audio_bytes", b"")

    return {
        "filename": inputs.get("filename"),
        "content_type": inputs.get("content_type"),
        "audio_size_bytes": len(audio_bytes),
    }


def sanitize_360_audio_inputs(
    inputs: dict[str, Any],
) -> dict[str, Any]:
    """
    Avoid tracing the WhatsApp media identifier or client object.
    """
    return {
        "mime_type": inputs.get("mime_type"),
        "has_media_id": bool(inputs.get("media_id")),
        "transcriber_type": type(
            inputs.get("transcriber")
        ).__name__,
    }


def sanitize_direct_download_inputs(
    inputs: dict[str, Any],
) -> dict[str, Any]:
    """
    Do not upload signed or authenticated download URLs.
    """
    return {
        "file_name": inputs.get("file_name"),
        "mime_type": inputs.get("mime_type"),
        "has_download_url": bool(inputs.get("download_url")),
        "transcriber_type": type(
            inputs.get("transcriber")
        ).__name__,
    }


def sanitize_download_inputs(
    inputs: dict[str, Any],
) -> dict[str, Any]:
    """
    Hide the URL and temporary filesystem path.
    """
    target_path = inputs.get("target_path")

    return {
        "has_url": bool(inputs.get("url")),
        "target_suffix": (
            target_path.suffix
            if isinstance(target_path, Path)
            else None
        ),
    }


def sanitize_conversion_inputs(
    inputs: dict[str, Any],
) -> dict[str, Any]:
    path = inputs.get("path")

    return {
        "source_suffix": (
            path.suffix.lower()
            if isinstance(path, Path)
            else None
        ),
    }


class OpenAITranscriber:
    """Transcriber implementation using the OpenAI Audio API."""

    def __init__(self, *, api_key: str, model: str) -> None:
        self._model = model

        # The wrapped client traces the underlying OpenAI API request.
        self._openai = wrap_openai(
            AsyncOpenAI(api_key=api_key)
        )

    @ls.traceable(
        name="transcribe_audio_bytes",
        run_type="chain",
        tags=["tami", "audio", "transcription"],
        process_inputs=sanitize_transcription_inputs,
    )
    async def transcribe_bytes(
        self,
        *,
        audio_bytes: bytes,
        filename: str,
        content_type: str,
    ) -> TranscriptionResult:
        suffix = (
            Path(filename).suffix
            or suffix_from_mime(content_type)
        )

        run = ls.get_current_run_tree()
        if run is not None:
            run.metadata.update(
                {
                    "transcription_provider": "openai",
                    "transcription_model": self._model,
                    "content_type": content_type,
                    "audio_size_bytes": len(audio_bytes),
                    "source_suffix": suffix,
                }
            )

        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as tmp:
            tmp.write(audio_bytes)
            tmp_path = Path(tmp.name)

        try:
            with tmp_path.open("rb") as audio_file:
                transcription = (
                    await self._openai.audio.transcriptions.create(
                        model=self._model,
                        file=audio_file,
                        language="he",
                    )
                )

            text = transcription.text.strip()

            if run is not None:
                run.metadata.update(
                    {
                        "transcription_status": "completed",
                        "transcript_character_count": len(text),
                        "transcript_is_empty": not bool(text),
                    }
                )

            return TranscriptionResult(
                text=text,
                model=self._model,
            )

        except Exception as exc:
            if run is not None:
                run.metadata.update(
                    {
                        "transcription_status": "failed",
                        "error_type": type(exc).__name__,
                    }
                )

            raise TranscriptionError(
                "OpenAI transcription failed"
            ) from exc

        finally:
            safe_unlink(tmp_path)

    async def close(self) -> None:
        await self._openai.close()


@ls.traceable(
    name="process_360dialog_audio",
    run_type="chain",
    tags=["tami", "whatsapp", "audio"],
    process_inputs=sanitize_360_audio_inputs,
)
async def handle_360dialog_audio_message(
    *,
    wa: Dialog360Client,
    transcriber: Transcriber,
    media_id: str,
    mime_type: str,
) -> str:
    """
    Download a 360dialog WhatsApp audio/voice message and transcribe it.

    The transcriber is injected; this function does not select the provider.
    WhatsApp voice notes are often audio/ogg with Opus codec.
    Unsupported formats are converted to 16 kHz mono WAV through ffmpeg.
    """
    run = ls.get_current_run_tree()

    suffix = suffix_from_mime(mime_type)

    if run is not None:
        run.metadata.update(
            {
                "provider": "360dialog",
                "transcription_provider": settings.transcription_provider,
                "mime_type": mime_type,
                "original_suffix": suffix,
                "transcriber_type": type(transcriber).__name__,
            }
        )

    raw_path = await wa.download_media_to_tempfile(
        media_id=media_id,
        suffix=suffix,
    )

    try:
        transcribable_path = ensure_transcribable_audio(
            raw_path
        )

        was_converted = transcribable_path != raw_path
        audio_bytes = transcribable_path.read_bytes()

        if run is not None:
            run.metadata.update(
                {
                    "audio_size_bytes": len(audio_bytes),
                    "audio_was_converted": was_converted,
                    "transcribable_suffix": (
                        transcribable_path.suffix.lower()
                    ),
                }
            )

        result = await transcriber.transcribe_bytes(
            audio_bytes=audio_bytes,
            filename=transcribable_path.name,
            content_type=mime_type,
        )

        logger.info(
            "Transcription complete: provider=%s model=%s "
            "language=%s audio_bytes=%d "
            "audio_duration=%.2fs processing=%.2fs text=%s",
            type(transcriber).__name__,
            result.model,
            result.language,
            len(audio_bytes),
            result.audio_duration_seconds or 0.0,
            result.processing_seconds or 0.0,
            result.text[:200],
        )

        if run is not None:
            run.metadata.update(
                {
                    "processing_status": "completed",
                    "transcription_model": result.model,
                    "detected_language": result.language,
                    "audio_duration_seconds": (
                        result.audio_duration_seconds
                    ),
                    "transcription_processing_seconds": (
                        result.processing_seconds
                    ),
                    "transcript_character_count": len(
                        result.text
                    ),
                }
            )

        return result.text

    except Exception as exc:
        if run is not None:
            run.metadata.update(
                {
                    "processing_status": "failed",
                    "error_type": type(exc).__name__,
                }
            )

        raise

    finally:
        safe_unlink(raw_path)

        if (
            "transcribable_path" in locals()
            and transcribable_path != raw_path
        ):
            safe_unlink(transcribable_path)


@ls.traceable(
    name="process_direct_audio_download",
    run_type="chain",
    tags=["tami", "audio", "direct-download"],
    process_inputs=sanitize_direct_download_inputs,
)
async def handle_direct_audio_download_url(
    *,
    transcriber: Transcriber,
    download_url: str,
    file_name: str = "voice-message",
    mime_type: str = "",
) -> str:
    """
    Optional fallback for providers or payloads that already provide
    a direct download URL.

    360dialog usually provides media_id, so the main path is
    handle_360dialog_audio_message().
    """
    suffix = (
        Path(file_name).suffix
        or suffix_from_mime(mime_type)
    )

    run = ls.get_current_run_tree()
    if run is not None:
        run.metadata.update(
            {
                "mime_type": mime_type,
                "source_suffix": suffix,
                "transcriber_type": type(transcriber).__name__,
            }
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = Path(tmpdir) / f"audio{suffix}"

        await download_file(
            download_url,
            raw_path,
        )

        transcribable_path = ensure_transcribable_audio(
            raw_path
        )

        audio_bytes = transcribable_path.read_bytes()

        if run is not None:
            run.metadata.update(
                {
                    "audio_size_bytes": len(audio_bytes),
                    "audio_was_converted": (
                        transcribable_path != raw_path
                    ),
                }
            )

        result = await transcriber.transcribe_bytes(
            audio_bytes=audio_bytes,
            filename=transcribable_path.name,
            content_type=mime_type,
        )

        return result.text


@ls.traceable(
    name="download_audio_file",
    run_type="tool",
    tags=["tami", "audio", "download"],
    process_inputs=sanitize_download_inputs,
)
async def download_file(
    url: str,
    target_path: Path,
) -> None:
    run = ls.get_current_run_tree()

    async with httpx.AsyncClient(
        timeout=120,
        follow_redirects=True,
    ) as client:
        response = await client.get(url)
        response.raise_for_status()

        target_path.write_bytes(response.content)

    if run is not None:
        run.metadata.update(
            {
                "download_status": "completed",
                "http_status_code": response.status_code,
                "download_size_bytes": len(response.content),
                "content_type": response.headers.get(
                    "content-type"
                ),
            }
        )


@ls.traceable(
    name="ensure_transcribable_audio",
    run_type="tool",
    tags=["tami", "audio", "conversion"],
    process_inputs=sanitize_conversion_inputs,
)
def ensure_transcribable_audio(path: Path) -> Path:
    run = ls.get_current_run_tree()

    source_suffix = path.suffix.lower()

    if source_suffix in SUPPORTED_TRANSCRIPTION_EXTS:
        if run is not None:
            run.metadata.update(
                {
                    "conversion_required": False,
                    "source_suffix": source_suffix,
                    "output_suffix": source_suffix,
                }
            )

        return path

    converted = path.with_suffix(".wav")

    if run is not None:
        run.metadata.update(
            {
                "conversion_required": True,
                "source_suffix": source_suffix,
                "output_suffix": ".wav",
                "sample_rate_hz": 16000,
                "channels": 1,
            }
        )

    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(path),
            "-ar",
            "16000",
            "-ac",
            "1",
            str(converted),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        if run is not None:
            run.metadata.update(
                {
                    "conversion_status": "failed",
                    "ffmpeg_return_code": result.returncode,
                }
            )

        # Do not store the full ffmpeg stderr in LangSmith because it
        # may contain temporary paths or excessive diagnostic output.
        raise RuntimeError(
            f"ffmpeg failed with code {result.returncode}"
        )

    if run is not None:
        run.metadata.update(
            {
                "conversion_status": "completed",
                "converted_size_bytes": (
                    converted.stat().st_size
                    if converted.exists()
                    else None
                ),
            }
        )

    return converted


def safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        logger.debug(
            "Failed to remove temporary file",
            exc_info=True,
        )
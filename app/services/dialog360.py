import logging
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)


def normalize_phone(phone: str) -> str:
    """360dialog expects international format digits, usually without '+'."""
    return (
        phone.replace("@c.us", "")
        .replace("+", "")
        .replace(" ", "")
        .replace("-", "")
        .strip()
    )


class Dialog360Client:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.api_base_url = settings.d360_api_base_url
        self.messages_url = f"{settings.d360_api_base_url}/messages"

    @property
    def json_headers(self) -> dict[str, str]:
        return {
            "D360-API-KEY": self.settings.d360_api_key,
            "Content-Type": "application/json",
        }

    @property
    def auth_headers(self) -> dict[str, str]:
        return {"D360-API-KEY": self.settings.d360_api_key}

    async def send_text(self, to: str, body: str) -> dict[str, Any]:
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": normalize_phone(to),
            "type": "text",
            "text": {"body": body},
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                self.messages_url,
                headers=self.json_headers,
                json=payload,
            )

        try:
            response_body: Any = response.json()
        except Exception:
            response_body = response.text

        if response.status_code not in {200, 201}:
            logger.error(
                "360dialog send failed: status=%s body=%s",
                response.status_code,
                response_body,
            )
            response.raise_for_status()

        return response_body

    async def send_typing_indicator(self, incoming_message_id: str) -> dict[str, Any]:
        """
        Show WhatsApp typing indicator.

        Requires the incoming webhook message id, usually starts with 'wamid.'.
        This also marks the incoming message as read.
        """
        if not incoming_message_id:
            raise ValueError("Missing incoming_message_id")

        payload = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": incoming_message_id,
            "typing_indicator": {
                "type": "text",
            },
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                self.messages_url,
                headers=self.json_headers,
                json=payload,
            )

        try:
            response_body: Any = response.json()
        except Exception:
            response_body = response.text

        if response.status_code not in {200, 201}:
            logger.error(
                "360dialog typing indicator failed: status=%s body=%s",
                response.status_code,
                response_body,
            )
            response.raise_for_status()

        return response_body

    async def send_image(
        self,
        to: str,
        image_bytes: bytes,
        mime_type: str,
        caption: str = "",
        filename: str = "annotated.jpg",
    ) -> dict[str, Any]:
        """Send an image message via the 360dialog two-step flow.

        1. ``POST /media`` (multipart) to upload the file and receive a
           ``media_id``.
        2. ``POST /messages`` with ``type: "image"`` referencing the
           ``media_id`` and an optional ``caption``.

        360dialog supports JPEG and PNG outgoing images up to 5 MB. The
        caller is responsible for sizing/encoding the bytes before calling
        this method.
        """
        async with httpx.AsyncClient(timeout=120.0) as client:
            upload_response = await client.post(
                f"{self.api_base_url}/media",
                headers=self.auth_headers,
                data={"messaging_product": "whatsapp"},
                files={"file": (filename, image_bytes, mime_type)},
            )

            try:
                upload_body: Any = upload_response.json()
            except Exception:
                upload_body = upload_response.text

            if upload_response.status_code not in {200, 201}:
                logger.error(
                    "360dialog media upload failed: status=%s body=%s",
                    upload_response.status_code,
                    upload_body,
                )
                upload_response.raise_for_status()

            media_id = upload_body.get("id") if isinstance(upload_body, dict) else None
            if not media_id:
                raise ValueError(
                    f"360dialog media upload returned no id: {upload_body!r}"
                )

            image_payload: dict[str, Any] = {"id": media_id}
            if caption:
                image_payload["caption"] = caption

            payload = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": normalize_phone(to),
                "type": "image",
                "image": image_payload,
            }

            response = await client.post(
                self.messages_url,
                headers=self.json_headers,
                json=payload,
            )

        try:
            response_body: Any = response.json()
        except Exception:
            response_body = response.text

        if response.status_code not in {200, 201}:
            logger.error(
                "360dialog send_image failed: status=%s body=%s",
                response.status_code,
                response_body,
            )
            response.raise_for_status()

        return response_body

    async def download_media_to_tempfile(
        self,
        media_id: str,
        suffix: str = ".bin",
    ) -> Path:
        """
        360dialog media download flow:
        1. GET /{media_id} to retrieve the temporary Meta media URL.
        2. Replace the lookaside.fbsbx.com host with the 360dialog host.
        3. Download the media with D360-API-KEY.
        """
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            meta_response = await client.get(
                f"{self.api_base_url}/{media_id}",
                headers=self.auth_headers,
            )
            meta_response.raise_for_status()
            meta = meta_response.json()

            media_url = meta.get("url")
            if not media_url:
                raise ValueError(f"Missing media url for media_id={media_id}")

            download_url = to_360dialog_media_url(self.api_base_url, media_url)

            file_response = await client.get(
                download_url,
                headers=self.auth_headers,
            )
            file_response.raise_for_status()

        fd, path_str = tempfile.mkstemp(prefix="wa-media-", suffix=suffix)
        path = Path(path_str)

        with open(fd, "wb") as f:
            f.write(file_response.content)

        return path


def to_360dialog_media_url(api_base_url: str, original_url: str) -> str:
    parsed = urlparse(original_url.replace("\\/", "/"))

    path_and_query = parsed.path
    if parsed.query:
        path_and_query += f"?{parsed.query}"

    return f"{api_base_url}{path_and_query}"


def iter_incoming_messages(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """
    Yield normalized inbound message dicts.

    Supports:
      - text
      - audio / voice messages
      - image messages

    Status callbacks and unsupported message types are ignored.
    """
    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})

            contacts_by_wa_id = {
                contact.get("wa_id"): contact
                for contact in value.get("contacts", [])
                if contact.get("wa_id")
            }

            for message in value.get("messages", []):
                sender = message.get("from", "")
                msg_type = message.get("type", "")
                contact = contacts_by_wa_id.get(sender, {})
                name = contact.get("profile", {}).get("name", "")

                normalized = {
                    "from": sender,
                    "id": message.get("id", ""),
                    "type": msg_type,
                    "name": name,
                    "raw": message,
                }

                if msg_type == "text":
                    normalized["text"] = message.get("text", {}).get("body", "")
                    yield normalized

                elif msg_type == "audio":
                    audio = message.get("audio", {}) or {}
                    normalized["media_id"] = audio.get("id", "")
                    normalized["mime_type"] = audio.get("mime_type", "")
                    normalized["voice"] = audio.get("voice", False)
                    yield normalized

                elif msg_type == "image":
                    image = message.get("image", {}) or {}
                    normalized["media_id"] = image.get("id", "")
                    normalized["mime_type"] = image.get("mime_type", "")
                    normalized["caption"] = image.get("caption", "")
                    yield normalized

                elif msg_type == "document":
                    doc = message.get("document", {}) or {}
                    normalized["media_id"] = doc.get("id", "")
                    normalized["mime_type"] = doc.get("mime_type", "")
                    normalized["filename"] = doc.get("filename", "")
                    normalized["caption"] = doc.get("caption", "")
                    yield normalized

                else:
                    logger.info(
                        "Ignoring unsupported message type=%s from=%s",
                        msg_type,
                        sender,
                    )

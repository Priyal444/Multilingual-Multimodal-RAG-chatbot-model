"""Thin HTTP client for the Polyglot API, used by the Streamlit UI.

Kept separate from the UI so the presentation layer never builds a URL or parses
an error envelope, and so the client can be exercised from a script.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_TIMEOUT = httpx.Timeout(120.0, connect=10.0)


class APIError(Exception):
    def __init__(self, message: str, *, status_code: int = 0, code: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code


@dataclass
class Session:
    base_url: str
    access_token: str | None = None
    refresh_token: str | None = None


class PolyglotClient:
    def __init__(self, base_url: str, access_token: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.access_token = access_token

    # -- plumbing ----------------------------------------------------------

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = dict(extra or {})
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        return headers

    @staticmethod
    def _raise(response: httpx.Response) -> None:
        if response.is_success:
            return
        message = f"HTTP {response.status_code}"
        code = ""
        try:
            payload = response.json()
            error = payload.get("error", {})
            message = error.get("message", message)
            code = error.get("code", "")
            if details := error.get("details"):
                message = f"{message} ({json.dumps(details)[:200]})"
        except Exception:  # noqa: BLE001 - non-JSON error body
            message = response.text[:300] or message
        raise APIError(message, status_code=response.status_code, code=code)

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
            response = client.request(
                method, f"{self.base_url}{path}", headers=self._headers(), **kwargs
            )
        self._raise(response)
        if response.status_code == 204 or not response.content:
            return None
        return response.json()

    # -- auth --------------------------------------------------------------

    def register(
        self,
        email: str,
        password: str,
        full_name: str | None = None,
        preferred_language: str | None = None,
    ) -> dict:
        return self._request(
            "POST",
            "/api/v1/auth/register",
            json={
                "email": email,
                "password": password,
                "full_name": full_name or None,
                "preferred_language": preferred_language or None,
            },
        )

    def login(self, email: str, password: str) -> dict:
        return self._request(
            "POST", "/api/v1/auth/login", json={"email": email, "password": password}
        )

    def me(self) -> dict:
        return self._request("GET", "/api/v1/auth/me")

    def update_me(self, **fields: Any) -> dict:
        return self._request("PATCH", "/api/v1/auth/me", json=fields)

    # -- system ------------------------------------------------------------

    def ready(self) -> dict:
        return self._request("GET", "/api/v1/ready")

    # -- conversations -----------------------------------------------------

    def list_conversations(self, limit: int = 20) -> dict:
        return self._request("GET", "/api/v1/conversations", params={"limit": limit})

    def get_conversation(self, conversation_id: str) -> dict:
        return self._request("GET", f"/api/v1/conversations/{conversation_id}")

    def delete_conversation(self, conversation_id: str) -> None:
        self._request("DELETE", f"/api/v1/conversations/{conversation_id}")

    # -- chat --------------------------------------------------------------

    def chat(
        self,
        message: str,
        *,
        conversation_id: str | None = None,
        language: str | None = None,
        images: list[str] | None = None,
        use_rag: bool = True,
    ) -> dict:
        return self._request(
            "POST",
            "/api/v1/chat",
            json={
                "message": message,
                "conversation_id": conversation_id,
                "language": language,
                "images": images or [],
                "use_rag": use_rag,
                "stream": False,
            },
        )

    def chat_stream(
        self,
        message: str,
        *,
        conversation_id: str | None = None,
        language: str | None = None,
        images: list[str] | None = None,
        use_rag: bool = True,
    ) -> Iterator[tuple[str, dict]]:
        """Yield ``(event_name, payload)`` pairs from the SSE stream."""
        payload = {
            "message": message,
            "conversation_id": conversation_id,
            "language": language,
            "images": images or [],
            "use_rag": use_rag,
            "stream": True,
        }
        with httpx.Client(timeout=DEFAULT_TIMEOUT) as client, client.stream(
            "POST", f"{self.base_url}/api/v1/chat", json=payload, headers=self._headers()
        ) as response:
            if not response.is_success:
                response.read()
                self._raise(response)

            event = "message"
            for line in response.iter_lines():
                if not line:
                    continue
                if line.startswith("event:"):
                    event = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    raw = line.split(":", 1)[1].strip()
                    try:
                        yield event, json.loads(raw)
                    except json.JSONDecodeError:
                        continue

    def chat_multipart(
        self,
        message: str,
        *,
        conversation_id: str | None = None,
        language: str | None = None,
        image_files: list[tuple[str, bytes, str]] | None = None,
        audio_file: tuple[str, bytes, str] | None = None,
        use_rag: bool = True,
    ) -> dict:
        files: list[tuple[str, tuple[str, bytes, str]]] = []
        for name, data, content_type in image_files or []:
            files.append(("images", (name, data, content_type)))
        if audio_file:
            files.append(("audio", audio_file))

        data: dict[str, str] = {
            "message": message,
            "use_rag": str(use_rag).lower(),
            "stream": "false",
        }
        if conversation_id:
            data["conversation_id"] = conversation_id
        if language:
            data["language"] = language

        return self._request("POST", "/api/v1/chat/multipart", data=data, files=files)

    # -- media -------------------------------------------------------------

    def transcribe(self, filename: str, data: bytes, content_type: str) -> dict:
        return self._request(
            "POST",
            "/api/v1/media/transcribe",
            files={"file": (filename, data, content_type)},
        )

    def speak(self, text: str, language: str | None = None) -> bytes:
        with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
            response = client.post(
                f"{self.base_url}/api/v1/media/speak",
                json={"text": text, "language": language},
                headers=self._headers(),
            )
        self._raise(response)
        return response.content

    def detect_language(self, text: str) -> dict:
        return self._request(
            "POST", "/api/v1/media/detect-language", json={"text": text}
        )

    # -- documents ---------------------------------------------------------

    def upload_document(self, filename: str, data: bytes, content_type: str) -> dict:
        return self._request(
            "POST", "/api/v1/documents", files={"file": (filename, data, content_type)}
        )

    def list_documents(self) -> dict:
        return self._request("GET", "/api/v1/documents")

    def delete_document(self, document_id: str) -> None:
        self._request("DELETE", f"/api/v1/documents/{document_id}")

    def search_documents(self, query: str, top_k: int = 5) -> dict:
        return self._request(
            "POST", "/api/v1/documents/search", json={"query": query, "top_k": top_k}
        )

"""Chat: persistence, language routing, vision, streaming, prompt assembly.

These run against the ``echo`` provider, which reflects the assembled request
back so assertions can prove that the language directive, the images, and the
retrieved context actually reached the model layer.
"""

from __future__ import annotations

import base64
import io
import json

import pytest
from httpx import AsyncClient

from app.services.images import decode_data_url, prepare_image
from app.services.prompts import SYSTEM_PROMPT, build_directive, title_from_message


def _png_bytes(size: tuple[int, int] = (64, 48), colour: str = "red") -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, format="PNG")
    return buffer.getvalue()


def _data_url(raw: bytes, media_type: str = "image/png") -> str:
    return f"data:{media_type};base64,{base64.b64encode(raw).decode()}"


# --- basic turn -----------------------------------------------------------


async def test_chat_creates_a_conversation_and_persists_both_messages(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    response = await client.post(
        "/api/v1/chat", json={"message": "Hello there"}, headers=auth
    )
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["conversation_id"]
    assert "Hello there" in body["reply"]

    detail = await client.get(
        f"/api/v1/conversations/{body['conversation_id']}", headers=auth
    )
    messages = detail.json()["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "Hello there"


async def test_conversation_title_comes_from_the_first_message(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    response = await client.post(
        "/api/v1/chat",
        json={"message": "How do I renew my passport in Karnataka?"},
        headers=auth,
    )
    detail = await client.get(
        f"/api/v1/conversations/{response.json()['conversation_id']}", headers=auth
    )
    assert detail.json()["title"].startswith("How do I renew my passport")


async def test_history_is_replayed_on_the_next_turn(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    first = await client.post(
        "/api/v1/chat", json={"message": "My name is Priya"}, headers=auth
    )
    conversation_id = first.json()["conversation_id"]

    await client.post(
        "/api/v1/chat",
        json={"message": "What is my name?", "conversation_id": conversation_id},
        headers=auth,
    )

    detail = await client.get(f"/api/v1/conversations/{conversation_id}", headers=auth)
    assert len(detail.json()["messages"]) == 4


async def test_empty_request_is_rejected(client: AsyncClient, auth: dict[str, str]) -> None:
    response = await client.post("/api/v1/chat", json={"message": "   "}, headers=auth)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


async def test_chat_requires_authentication(client: AsyncClient) -> None:
    response = await client.post("/api/v1/chat", json={"message": "hi"})
    assert response.status_code == 401


async def test_another_users_conversation_is_not_found(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    mine = await client.post("/api/v1/chat", json={"message": "private"}, headers=auth)
    conversation_id = mine.json()["conversation_id"]

    other = await client.post(
        "/api/v1/auth/register",
        json={"email": "mallory@example.com", "password": "correct-horse-battery"},
    )
    stranger = {"Authorization": f"Bearer {other.json()['access_token']}"}

    # 404, not 403 — a 403 would confirm the ID exists.
    response = await client.get(f"/api/v1/conversations/{conversation_id}", headers=stranger)
    assert response.status_code == 404


# --- language routing -----------------------------------------------------


async def test_language_is_detected_from_the_message(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    response = await client.post(
        "/api/v1/chat", json={"message": "मुझे मदद चाहिए"}, headers=auth
    )
    body = response.json()
    assert body["detected_language"] == "hi"
    assert body["reply_language"] == "hi"
    # The echo provider surfaces the directive, proving it reached the model.
    assert "Hindi" in body["reply"]


async def test_explicit_language_overrides_detection(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    response = await client.post(
        "/api/v1/chat",
        json={"message": "Hello, how are you?", "language": "ja"},
        headers=auth,
    )
    body = response.json()
    assert body["detected_language"] == "en"
    assert body["reply_language"] == "ja"
    assert "Japanese" in body["reply"]


async def test_saved_preference_applies_when_no_override_is_given(
    client: AsyncClient,
) -> None:
    registered = await client.post(
        "/api/v1/auth/register",
        json={
            "email": "es-user@example.com",
            "password": "correct-horse-battery",
            "preferred_language": "es",
        },
    )
    headers = {"Authorization": f"Bearer {registered.json()['access_token']}"}

    response = await client.post(
        "/api/v1/chat", json={"message": "Hello there, how are you?"}, headers=headers
    )
    assert response.json()["reply_language"] == "es"


# --- vision ---------------------------------------------------------------


async def test_image_reaches_the_model(client: AsyncClient, auth: dict[str, str]) -> None:
    response = await client.post(
        "/api/v1/chat",
        json={"message": "What colour is this?", "images": [_data_url(_png_bytes())]},
        headers=auth,
    )
    assert response.status_code == 200
    assert "1 image" in response.json()["reply"]


async def test_image_only_message_is_accepted(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    response = await client.post(
        "/api/v1/chat", json={"message": "", "images": [_data_url(_png_bytes())]}, headers=auth
    )
    assert response.status_code == 200


async def test_too_many_images_is_rejected(client: AsyncClient, auth: dict[str, str]) -> None:
    response = await client.post(
        "/api/v1/chat",
        json={"message": "look", "images": [_data_url(_png_bytes())] * 9},
        headers=auth,
    )
    assert response.status_code == 422


async def test_corrupt_image_is_rejected(client: AsyncClient, auth: dict[str, str]) -> None:
    response = await client.post(
        "/api/v1/chat",
        json={"message": "look", "images": [_data_url(b"not-an-image")]},
        headers=auth,
    )
    assert response.status_code == 422
    assert "decoded" in response.json()["error"]["message"]


# --- streaming ------------------------------------------------------------


async def test_streaming_emits_start_deltas_and_done(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    events: list[tuple[str, dict]] = []
    name = "message"

    async with client.stream(
        "POST",
        "/api/v1/chat",
        json={"message": "Stream this back to me", "stream": True},
        headers=auth,
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        async for line in response.aiter_lines():
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                events.append((name, json.loads(line.split(":", 1)[1].strip())))

    names = [n for n, _ in events]
    assert names[0] == "start"
    assert "delta" in names
    assert names[-1] == "done"

    done = events[-1][1]
    assert done["assistant_message_id"]
    assert done["usage"]["output_tokens"] > 0


async def test_streamed_reply_is_persisted(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    """The stream body runs after the request session closes — this is the
    regression test for writing the assistant message through a closed session."""
    conversation_id = None
    async with client.stream(
        "POST",
        "/api/v1/chat",
        json={"message": "Persist me", "stream": True},
        headers=auth,
    ) as response:
        async for line in response.aiter_lines():
            if line.startswith("data:"):
                payload = json.loads(line.split(":", 1)[1].strip())
                conversation_id = payload.get("conversation_id", conversation_id)

    assert conversation_id
    detail = await client.get(f"/api/v1/conversations/{conversation_id}", headers=auth)
    roles = [m["role"] for m in detail.json()["messages"]]
    assert roles == ["user", "assistant"]


# --- token counting -------------------------------------------------------


async def test_token_estimate(client: AsyncClient, auth: dict[str, str]) -> None:
    response = await client.post(
        "/api/v1/chat/estimate", json={"message": "How many tokens is this?"}, headers=auth
    )
    assert response.status_code == 200
    assert response.json()["input_tokens"] > 0


# --- conversation management ---------------------------------------------


async def test_conversations_can_be_renamed_and_deleted(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    created = await client.post("/api/v1/chat", json={"message": "hello"}, headers=auth)
    conversation_id = created.json()["conversation_id"]

    renamed = await client.patch(
        f"/api/v1/conversations/{conversation_id}",
        json={"title": "Renamed"},
        headers=auth,
    )
    assert renamed.json()["title"] == "Renamed"

    assert (
        await client.delete(f"/api/v1/conversations/{conversation_id}", headers=auth)
    ).status_code == 204
    assert (
        await client.get(f"/api/v1/conversations/{conversation_id}", headers=auth)
    ).status_code == 404


# --- prompt assembly (unit) ----------------------------------------------


def test_system_prompt_is_a_frozen_constant() -> None:
    # Prompt caching is a byte-exact prefix match; anything interpolated here
    # would invalidate the cache on every request.
    assert "{" not in SYSTEM_PROMPT
    assert SYSTEM_PROMPT == SYSTEM_PROMPT.strip() + "\n"


def test_directive_carries_language_context_and_name() -> None:
    directive = build_directive(
        reply_language="ta", context="[1] source: a.pdf\nbody", user_name="Priya"
    )
    assert directive is not None
    assert "Tamil" in directive
    assert "Priya" in directive
    assert "[1]" in directive


def test_directive_flags_right_to_left_languages() -> None:
    directive = build_directive(reply_language="ar", context="")
    assert directive is not None
    assert "right-to-left" in directive


def test_directive_is_none_when_there_is_nothing_to_say() -> None:
    assert build_directive(reply_language=None, context="") is None


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Short question", "Short question"),
        ("", "New conversation"),
        ("  spaced   out   text ", "spaced out text"),
    ],
)
def test_title_generation(message: str, expected: str) -> None:
    assert title_from_message(message) == expected


def test_long_titles_are_truncated_on_a_word_boundary() -> None:
    title = title_from_message("word " * 50)
    assert len(title) <= 61
    assert title.endswith("…")


# --- image preparation (unit) --------------------------------------------


def test_oversized_images_are_downscaled() -> None:
    prepared = prepare_image(_png_bytes((4000, 3000)))
    assert prepared.was_resized
    assert max(prepared.width, prepared.height) <= 1568
    # Photographic content without alpha is re-encoded as JPEG to cut size.
    assert prepared.media_type == "image/jpeg"


def test_transparency_is_preserved_as_png() -> None:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGBA", (32, 32), (255, 0, 0, 128)).save(buffer, format="PNG")
    prepared = prepare_image(buffer.getvalue())
    assert prepared.media_type == "image/png"


def test_base64_payload_has_no_newlines() -> None:
    # The API rejects wrapped base64 — a classic source of 400s.
    prepared = prepare_image(_png_bytes((800, 600)))
    assert "\n" not in prepared.data_b64


def test_bare_base64_and_data_urls_both_decode() -> None:
    raw = _png_bytes()
    from_url, declared = decode_data_url(_data_url(raw))
    from_bare, none_declared = decode_data_url(base64.b64encode(raw).decode())

    assert from_url == raw == from_bare
    assert declared == "image/png"
    assert none_declared is None

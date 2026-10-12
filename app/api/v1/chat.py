"""Chat endpoints: JSON, multipart (images + voice notes), streaming, estimation."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUser, RateLimited, SessionDep
from app.api.v1.uploads import read_limited
from app.core.config import settings
from app.core.errors import NotFoundError, ValidationError
from app.core.logging import get_logger
from app.db.models import Attachment, AttachmentKind, Conversation, User
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    SourceRef,
    TokenEstimateRequest,
    TokenEstimateResponse,
    UsageInfo,
)
from app.services import storage
from app.services.chat import PreparedTurn, TurnInput, TurnResult, chat_service
from app.services.images import PreparedImage, decode_data_url, prepare_image
from app.services.language import normalize_code
from app.services.llm import LLMRequest, Turn, get_llm_provider
from app.services.prompts import SYSTEM_PROMPT
from app.services.stt import speech_to_text

log = get_logger(__name__)
router = APIRouter()


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


async def _resolve_conversation(
    session: AsyncSession, user: User, conversation_id: str | None
) -> Conversation:
    if conversation_id:
        conversation = (
            await session.execute(
                select(Conversation).where(
                    Conversation.id == conversation_id, Conversation.user_id == user.id
                )
            )
        ).scalar_one_or_none()
        if conversation is None:
            raise NotFoundError("Conversation not found.")
        return conversation

    conversation = Conversation(user_id=user.id, language=user.preferred_language)
    session.add(conversation)
    await session.flush()
    return conversation


def _prepare_images(payloads: list[str]) -> list[PreparedImage]:
    if len(payloads) > settings.max_images_per_message:
        raise ValidationError(
            f"At most {settings.max_images_per_message} images per message.",
            received=len(payloads),
        )
    images: list[PreparedImage] = []
    for payload in payloads:
        raw, declared = decode_data_url(payload)
        images.append(prepare_image(raw, declared_type=declared))
    return images


def _sse(event: str, data: dict) -> str:
    """Serialise one Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream_response(prepared: PreparedTurn) -> StreamingResponse:
    async def body() -> AsyncIterator[str]:
        # Tell the client which conversation it is in before any token arrives,
        # so a brand-new conversation can be rendered immediately.
        yield _sse(
            "start",
            {
                "conversation_id": prepared.conversation_id,
                "user_message_id": prepared.user_message_id,
                "reply_language": prepared.reply_language,
                "detected_language": prepared.detected.code,
            },
        )
        async for event in chat_service.stream_turn(prepared):
            yield _sse(event["event"], event["data"])

    return StreamingResponse(
        body(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Nginx buffers proxied responses by default, which defeats streaming.
            "X-Accel-Buffering": "no",
        },
    )


def _to_response(result: TurnResult) -> ChatResponse:
    return ChatResponse(
        conversation_id=result.conversation_id,
        user_message_id=result.user_message_id,
        assistant_message_id=result.assistant_message_id,
        reply=result.text,
        reply_language=result.reply_language,
        detected_language=result.detected_language,
        detection_confidence=result.detection_confidence,
        model=result.model,
        usage=UsageInfo(**result.usage),
        latency_ms=result.latency_ms,
        stop_reason=result.stop_reason,
        sources=[
            SourceRef(
                document_id=c.document_id,
                filename=c.filename,
                part=c.ordinal + 1,
                score=c.score,
            )
            for c in result.retrieved
        ],
    )


# --------------------------------------------------------------------------
# endpoints
# --------------------------------------------------------------------------


@router.post(
    "",
    response_model=None,
    summary="Send a message (JSON; images as base64 or data URLs)",
    responses={200: {"content": {"application/json": {}, "text/event-stream": {}}}},
)
async def chat(
    payload: ChatRequest,
    user: CurrentUser,
    session: SessionDep,
    _: RateLimited,
) -> ChatResponse | StreamingResponse:
    if not payload.message.strip() and not payload.images:
        raise ValidationError("Send a message, an image, or both.")

    conversation = await _resolve_conversation(session, user, payload.conversation_id)
    prepared = await chat_service.prepare_turn(
        session,
        user=user,
        conversation=conversation,
        turn=TurnInput(
            text=payload.message,
            images=_prepare_images(payload.images),
            language=payload.language,
            document_ids=payload.document_ids,
            use_rag=payload.use_rag,
        ),
    )

    if payload.stream:
        # The request-scoped session's exit code runs *before* the streaming body
        # is consumed, so the user's message must be durable now.
        await session.commit()
        return await _stream_response(prepared)

    return _to_response(await chat_service.complete_turn(session, prepared))


@router.post(
    "/multipart",
    response_model=None,
    summary="Send a message with file uploads (images and/or a voice note)",
)
async def chat_multipart(  # noqa: PLR0913 - a multipart form is inherently wide
    user: CurrentUser,
    session: SessionDep,
    _: RateLimited,
    message: Annotated[str, Form()] = "",
    conversation_id: Annotated[str | None, Form()] = None,
    language: Annotated[str | None, Form()] = None,
    use_rag: Annotated[bool, Form()] = True,
    stream: Annotated[bool, Form()] = False,
    document_ids: Annotated[str | None, Form(description="Comma-separated")] = None,
    images: Annotated[list[UploadFile] | None, File()] = None,
    audio: Annotated[UploadFile | None, File()] = None,
) -> ChatResponse | StreamingResponse:
    text = message.strip()
    transcript: str | None = None
    raw_audio: bytes | None = None

    # --- voice note: transcribe first, then treat it as the message ---
    if audio is not None and audio.filename:
        raw_audio = await read_limited(audio, settings.max_audio_bytes, label="audio file")
        transcription = await speech_to_text.transcribe(
            raw_audio,
            media_type=audio.content_type or "audio/webm",
            language=normalize_code(language),
        )
        transcript = transcription.text
        text = f"{text}\n\n{transcript}".strip() if text else transcript
        # Whisper's language detection beats text detection on a short
        # utterance, so let it choose unless the caller was explicit.
        if not language and transcription.language:
            language = transcription.language
        if not text:
            raise ValidationError("No speech was detected in that recording.")

    # --- images ---
    prepared_images: list[PreparedImage] = []
    stored: list[tuple[str, str, int, str]] = []  # (path, media_type, size, filename)
    uploads = [f for f in (images or []) if f.filename]
    if len(uploads) > settings.max_images_per_message:
        raise ValidationError(
            f"At most {settings.max_images_per_message} images per message.",
            received=len(uploads),
        )
    for upload in uploads:
        raw = await read_limited(upload, settings.max_image_bytes, label="image")
        image = prepare_image(raw, declared_type=upload.content_type)
        prepared_images.append(image)
        path = await storage.save(
            raw, user_id=user.id, media_type=image.media_type, category="images"
        )
        stored.append((path, image.media_type, image.size_bytes, upload.filename))

    if not text and not prepared_images:
        raise ValidationError("Send a message, an image, a voice note, or a combination.")

    conversation = await _resolve_conversation(session, user, conversation_id)
    prepared = await chat_service.prepare_turn(
        session,
        user=user,
        conversation=conversation,
        turn=TurnInput(
            text=text,
            images=prepared_images,
            language=normalize_code(language),
            document_ids=[d.strip() for d in document_ids.split(",") if d.strip()]
            if document_ids
            else None,
            use_rag=use_rag,
        ),
    )

    # Link the stored files to the message now that it exists.
    for path, media_type, size, filename in stored:
        session.add(
            Attachment(
                message_id=prepared.user_message_id,
                user_id=user.id,
                kind=AttachmentKind.IMAGE,
                filename=filename,
                media_type=media_type,
                size_bytes=size,
                storage_path=path,
            )
        )
    if raw_audio is not None and audio is not None:
        audio_path = await storage.save(
            raw_audio,
            user_id=user.id,
            media_type=audio.content_type or "audio/webm",
            category="audio",
        )
        session.add(
            Attachment(
                message_id=prepared.user_message_id,
                user_id=user.id,
                kind=AttachmentKind.AUDIO,
                filename=audio.filename or "voice-note",
                media_type=audio.content_type or "audio/webm",
                size_bytes=len(raw_audio),
                storage_path=audio_path,
                transcript=transcript,
            )
        )
    await session.flush()

    if stream:
        await session.commit()
        return await _stream_response(prepared)

    return _to_response(await chat_service.complete_turn(session, prepared))


@router.post(
    "/estimate",
    response_model=TokenEstimateResponse,
    summary="Count the input tokens a message would cost, before sending it",
)
async def estimate(
    payload: TokenEstimateRequest, user: CurrentUser, session: SessionDep
) -> TokenEstimateResponse:
    """Exact provider-side token count — never a local heuristic.

    Client-side estimators (``tiktoken`` and friends) use a different tokenizer
    and are wrong for Claude, badly so for non-Latin scripts.
    """
    turns: list[Turn] = []
    if payload.conversation_id:
        conversation = await _resolve_conversation(session, user, payload.conversation_id)
        turns = await chat_service.load_history(session, conversation.id)
    turns.append(Turn.text("user", payload.message))

    provider = get_llm_provider()
    count = await provider.count_tokens(
        LLMRequest(system=SYSTEM_PROMPT, turns=turns, directive=None)
    )
    return TokenEstimateResponse(input_tokens=count, model=settings.llm_model)

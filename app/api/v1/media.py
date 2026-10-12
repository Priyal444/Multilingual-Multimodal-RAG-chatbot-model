"""Standalone speech and language endpoints.

These exist separately from ``/chat`` so a client can transcribe a recording,
show the user what was heard, let them edit it, and only then send it — which is
what any voice UI that respects its users does.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, Query, Response, UploadFile

from app.api.deps import CurrentUser, RateLimited
from app.api.v1.uploads import read_limited
from app.core.config import settings
from app.core.errors import ValidationError
from app.core.logging import get_logger
from app.schemas.media import (
    DetectLanguageRequest,
    DetectLanguageResponse,
    SpeakRequest,
    TranscriptionResponse,
    VoiceInfo,
)
from app.services.language import (
    detect_language,
    is_rtl,
    language_name,
    normalize_code,
    script_of,
)
from app.services.stt import speech_to_text
from app.services.tts import text_to_speech

log = get_logger(__name__)
router = APIRouter()


@router.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    summary="Transcribe an audio file (auto-detects the spoken language)",
)
async def transcribe(
    user: CurrentUser,
    _: RateLimited,
    file: Annotated[UploadFile, File()],
    language: Annotated[str | None, Form()] = None,
) -> TranscriptionResponse:
    if not file.filename:
        raise ValidationError("No audio file was provided.")

    raw = await read_limited(file, settings.max_audio_bytes, label="audio file")
    result = await speech_to_text.transcribe(
        raw,
        media_type=file.content_type or "audio/webm",
        language=normalize_code(language),
    )
    log.info(
        "audio_transcribed",
        user_id=user.id,
        language=result.language,
        duration_s=round(result.duration_seconds, 1),
        chars=len(result.text),
    )
    return TranscriptionResponse(
        text=result.text,
        language=result.language,
        language_name=language_name(result.language) if result.language else None,
        language_confidence=result.language_confidence,
        duration_seconds=result.duration_seconds,
    )


@router.post(
    "/speak",
    summary="Synthesize speech (returns audio/mpeg)",
    responses={200: {"content": {"audio/mpeg": {}}}},
)
async def speak(payload: SpeakRequest, user: CurrentUser, _: RateLimited) -> Response:
    # When no language is given, detect it from the text itself — otherwise an
    # English voice reads a Hindi reply, which is unintelligible.
    language = payload.language or detect_language(payload.text).code
    speech = await text_to_speech.synthesize(
        payload.text, language=language, voice=payload.voice, rate=payload.rate
    )
    log.info("speech_synthesized", user_id=user.id, voice=speech.voice, bytes=len(speech.audio))
    return Response(
        content=speech.audio,
        media_type=speech.media_type,
        headers={
            "X-Voice": speech.voice,
            "X-Language": speech.language,
            "Content-Disposition": 'inline; filename="speech.mp3"',
            "Cache-Control": "no-store",
        },
    )


@router.get(
    "/voices",
    response_model=list[VoiceInfo],
    summary="List available neural voices",
)
async def voices(
    user: CurrentUser,
    language: Annotated[str | None, Query(description="Filter by language code")] = None,
) -> list[VoiceInfo]:
    rows = await text_to_speech.list_voices(normalize_code(language))
    return [VoiceInfo(**row) for row in rows]


@router.post(
    "/detect-language",
    response_model=DetectLanguageResponse,
    summary="Identify the language of a piece of text",
)
async def detect(payload: DetectLanguageRequest, user: CurrentUser) -> DetectLanguageResponse:
    guess = detect_language(payload.text)
    return DetectLanguageResponse(
        code=guess.code,
        name=guess.name,
        confidence=guess.confidence,
        method=guess.method,
        script=script_of(guess.code),
        is_rtl=is_rtl(guess.code),
    )

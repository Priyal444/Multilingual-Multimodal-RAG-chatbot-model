"""The chat orchestrator — where a turn is actually assembled and executed.

Responsibilities, in order: decide the reply language, retrieve document context,
build the provider-agnostic request, persist both sides of the turn with token
accounting, and hand back something the API layer can serialise.

**A note on sessions and streaming.** ``prepare_turn`` runs inside the request's
database session and commits with it. ``stream_turn`` does *not*: a
``StreamingResponse`` body is consumed **after** the endpoint function has
returned, by which point the request-scoped session is already closed. Writing
the assistant message through that session would fail — intermittently, under
load, which is the worst kind of failure. So the streaming path opens its own
short-lived session when the stream completes.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.db.models import Conversation, Message, MessageRole, User
from app.db.session import SessionLocal
from app.services.images import PreparedImage
from app.services.language import LanguageGuess, detect_language, normalize_code
from app.services.llm import (
    ImagePart,
    LLMRequest,
    LLMResult,
    TextPart,
    Turn,
    get_llm_provider,
)
from app.services.prompts import SYSTEM_PROMPT, build_directive, title_from_message
from app.services.rag import RetrievedChunk, rag_pipeline

log = get_logger(__name__)

# How many prior messages to replay. Bounded because history is resent on every
# turn: unbounded history means linearly growing cost and eventual truncation.
HISTORY_LIMIT = 20


@dataclass(slots=True)
class TurnInput:
    text: str
    images: list[PreparedImage] = field(default_factory=list)
    # Explicit override; None means "detect from the message".
    language: str | None = None
    document_ids: list[str] | None = None
    use_rag: bool = True


@dataclass(slots=True)
class PreparedTurn:
    conversation_id: str
    user_message_id: str
    llm_request: LLMRequest
    detected: LanguageGuess
    reply_language: str
    retrieved: list[RetrievedChunk]


@dataclass(slots=True)
class TurnResult:
    conversation_id: str
    user_message_id: str
    assistant_message_id: str
    text: str
    reply_language: str
    detected_language: str
    detection_confidence: float
    retrieved: list[RetrievedChunk]
    model: str
    usage: dict[str, int]
    latency_ms: int
    stop_reason: str | None = None


class ChatService:
    # -- turn assembly -----------------------------------------------------

    @staticmethod
    def _resolve_reply_language(
        *,
        detected: LanguageGuess,
        explicit: str | None,
        user: User,
        conversation: Conversation,
    ) -> str:
        """Decide what language to answer in, most specific signal winning.

        Precedence: the request's explicit choice, then the user's saved
        preference, then the language they just wrote in — but only if detection
        was actually confident. A low-confidence guess on a three-word message is
        worse than falling back to the conversation's established language.
        """
        if explicit and (code := normalize_code(explicit)):
            return code
        if user.preferred_language:
            return user.preferred_language
        if detected.is_reliable:
            return detected.code
        return conversation.language or detected.code

    @staticmethod
    async def load_history(session: AsyncSession, conversation_id: str) -> list[Turn]:
        rows = (
            await session.execute(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at.desc())
                .limit(HISTORY_LIMIT)
            )
        ).scalars().all()

        turns: list[Turn] = []
        for message in reversed(rows):  # back into chronological order
            if not message.content.strip():
                continue
            turns.append(Turn.text(message.role, message.content))  # type: ignore[arg-type]

        # The API requires the first turn to be from the user; a history that
        # begins with an assistant message (possible after deletions) is invalid.
        while turns and turns[0].role != "user":
            turns.pop(0)
        return turns

    async def prepare_turn(
        self,
        session: AsyncSession,
        *,
        user: User,
        conversation: Conversation,
        turn: TurnInput,
    ) -> PreparedTurn:
        """Persist the user's message and build the request for the model."""
        detected = detect_language(turn.text) if turn.text.strip() else LanguageGuess(
            user.preferred_language or "en",
            "",
            0.0,
            "default",
        )
        reply_language = self._resolve_reply_language(
            detected=detected, explicit=turn.language, user=user, conversation=conversation
        )

        retrieved: list[RetrievedChunk] = []
        if turn.use_rag and settings.rag_enabled and turn.text.strip():
            retrieved = await rag_pipeline.retrieve(
                session,
                user_id=user.id,
                query=turn.text,
                document_ids=turn.document_ids,
            )

        history = await self.load_history(session, conversation.id)

        parts: list[TextPart | ImagePart] = [
            ImagePart(media_type=img.media_type, data_b64=img.data_b64) for img in turn.images
        ]
        # A vision turn with no text still needs a question, or the model has
        # nothing to answer.
        text = turn.text.strip() or ("Describe this image." if turn.images else "")
        if text:
            parts.append(TextPart(text))

        turns = [*history, Turn(role="user", parts=parts)]

        llm_request = LLMRequest(
            system=SYSTEM_PROMPT,
            turns=turns,
            directive=build_directive(
                reply_language=reply_language,
                context=rag_pipeline.format_context(retrieved),
                user_name=user.full_name,
            ),
            metadata={"conversation_id": conversation.id},
        )

        user_message = Message(
            conversation_id=conversation.id,
            role=MessageRole.USER,
            content=turn.text,
            language=detected.code,
        )
        session.add(user_message)

        if conversation.title in {"", "New conversation"} and turn.text.strip():
            conversation.title = title_from_message(turn.text)
        conversation.language = reply_language
        await session.flush()

        log.info(
            "turn_prepared",
            conversation_id=conversation.id,
            detected=detected.code,
            detection_method=detected.method,
            reply_language=reply_language,
            images=len(turn.images),
            retrieved=len(retrieved),
            history_turns=len(history),
        )

        return PreparedTurn(
            conversation_id=conversation.id,
            user_message_id=user_message.id,
            llm_request=llm_request,
            detected=detected,
            reply_language=reply_language,
            retrieved=retrieved,
        )

    # -- persistence -------------------------------------------------------

    @staticmethod
    async def _persist_assistant(
        session: AsyncSession,
        *,
        prepared: PreparedTurn,
        result: LLMResult,
        latency_ms: int,
    ) -> Message:
        message = Message(
            conversation_id=prepared.conversation_id,
            role=MessageRole.ASSISTANT,
            content=result.text,
            language=prepared.reply_language,
            model=result.model,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            cache_read_tokens=result.usage.cache_read_tokens,
            cache_write_tokens=result.usage.cache_write_tokens,
            latency_ms=latency_ms,
        )
        session.add(message)
        await session.flush()
        return message

    @staticmethod
    def _to_result(
        prepared: PreparedTurn,
        assistant_message: Message,
        result: LLMResult,
        latency_ms: int,
    ) -> TurnResult:
        return TurnResult(
            conversation_id=prepared.conversation_id,
            user_message_id=prepared.user_message_id,
            assistant_message_id=assistant_message.id,
            text=result.text,
            reply_language=prepared.reply_language,
            detected_language=prepared.detected.code,
            detection_confidence=prepared.detected.confidence,
            retrieved=prepared.retrieved,
            model=result.model,
            usage={
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
                "cache_read_tokens": result.usage.cache_read_tokens,
                "cache_write_tokens": result.usage.cache_write_tokens,
            },
            latency_ms=latency_ms,
            stop_reason=result.stop_reason,
        )

    # -- execution ---------------------------------------------------------

    async def complete_turn(
        self, session: AsyncSession, prepared: PreparedTurn
    ) -> TurnResult:
        """Run the turn to completion and persist the reply."""
        provider = get_llm_provider()
        started = time.perf_counter()
        result = await provider.complete(prepared.llm_request)
        latency_ms = int((time.perf_counter() - started) * 1000)

        message = await self._persist_assistant(
            session, prepared=prepared, result=result, latency_ms=latency_ms
        )
        log.info(
            "turn_completed",
            conversation_id=prepared.conversation_id,
            model=result.model,
            latency_ms=latency_ms,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            cache_read_tokens=result.usage.cache_read_tokens,
            stop_reason=result.stop_reason,
        )
        return self._to_result(prepared, message, result, latency_ms)

    async def stream_turn(self, prepared: PreparedTurn) -> AsyncIterator[dict]:
        """Yield SSE-shaped dicts, persisting the reply once the stream ends.

        Opens its own database session on completion — see the module docstring
        for why the request-scoped one cannot be used here.
        """
        provider = get_llm_provider()
        started = time.perf_counter()
        final: LLMResult | None = None

        try:
            async for event in provider.stream(prepared.llm_request):
                if event.type == "delta":
                    yield {"event": "delta", "data": {"text": event.text}}
                elif event.type == "done":
                    final = event.result
        except Exception as exc:  # noqa: BLE001 - surfaced to the client as an event
            log.warning(
                "stream_failed",
                conversation_id=prepared.conversation_id,
                error=str(exc),
            )
            yield {
                "event": "error",
                "data": {"message": "The model stream failed. Please retry."},
            }
            return

        if final is None:  # pragma: no cover - providers always emit `done`
            yield {"event": "error", "data": {"message": "The stream ended unexpectedly."}}
            return

        latency_ms = int((time.perf_counter() - started) * 1000)
        async with SessionLocal() as session:
            message = await self._persist_assistant(
                session, prepared=prepared, result=final, latency_ms=latency_ms
            )
            await session.commit()
            assistant_message_id = message.id

        log.info(
            "stream_completed",
            conversation_id=prepared.conversation_id,
            model=final.model,
            latency_ms=latency_ms,
            output_tokens=final.usage.output_tokens,
            cache_read_tokens=final.usage.cache_read_tokens,
        )

        yield {
            "event": "done",
            "data": {
                "conversation_id": prepared.conversation_id,
                "user_message_id": prepared.user_message_id,
                "assistant_message_id": assistant_message_id,
                "reply_language": prepared.reply_language,
                "detected_language": prepared.detected.code,
                "model": final.model,
                "stop_reason": final.stop_reason,
                "latency_ms": latency_ms,
                "usage": {
                    "input_tokens": final.usage.input_tokens,
                    "output_tokens": final.usage.output_tokens,
                    "cache_read_tokens": final.usage.cache_read_tokens,
                    "cache_write_tokens": final.usage.cache_write_tokens,
                },
                "sources": [
                    {
                        "document_id": c.document_id,
                        "filename": c.filename,
                        "part": c.ordinal + 1,
                        "score": c.score,
                    }
                    for c in prepared.retrieved
                ],
            },
        }


chat_service = ChatService()

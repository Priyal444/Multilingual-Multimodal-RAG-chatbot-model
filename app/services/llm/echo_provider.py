"""A deterministic, offline LLM used by the test suite and by `LLM_PROVIDER=echo`.

It exists so the whole request path — auth, rate limiting, persistence, streaming,
RAG assembly — can be exercised in CI without a network call or an API key. It
deliberately echoes back structured facts about the request so tests can assert
that images, retrieved context, and the language directive actually arrived.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from app.services.llm.base import ImagePart, LLMRequest, LLMResult, StreamEvent, Usage


class EchoProvider:
    name = "echo"
    model = "echo-1"

    async def complete(self, request: LLMRequest) -> LLMResult:
        return LLMResult(
            text=self._render(request),
            model=self.model,
            usage=Usage(input_tokens=self._estimate(request), output_tokens=32),
            stop_reason="end_turn",
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        text = self._render(request)
        for word in text.split(" "):
            await asyncio.sleep(0)  # yield control, mimicking a real stream
            yield StreamEvent(type="delta", text=word + " ")
        yield StreamEvent(
            type="done",
            result=LLMResult(
                text=text,
                model=self.model,
                usage=Usage(input_tokens=self._estimate(request), output_tokens=32),
                stop_reason="end_turn",
            ),
        )

    async def count_tokens(self, request: LLMRequest) -> int:
        return self._estimate(request)

    async def embed(self, texts: list[str]) -> list[list[float]] | None:
        return None

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _render(request: LLMRequest) -> str:
        last_user = next(
            (t for t in reversed(request.turns) if t.role == "user"), None
        )
        prompt = last_user.plain_text() if last_user else ""
        image_count = sum(
            1 for t in request.turns for p in t.parts if isinstance(p, ImagePart)
        )
        parts = [f"[echo] {prompt}".strip()]
        if image_count:
            parts.append(f"(received {image_count} image(s))")
        if request.directive:
            parts.append(f"(directive: {request.directive.splitlines()[0][:120]})")
        return " ".join(parts)

    @staticmethod
    def _estimate(request: LLMRequest) -> int:
        chars = len(request.system) + len(request.directive or "")
        chars += sum(len(t.plain_text()) for t in request.turns)
        return max(1, chars // 4)

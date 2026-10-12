"""Claude adapter.

Everything vendor-specific lives here. Notable behaviours, because they are the
difference between "it calls an API" and "it survives production":

**Prompt caching.** The system prompt is sent as a cached block. Caching is a
*prefix* match, so anything that changes per request — the reply-language
directive, retrieved passages — is deliberately sent *after* the cached prefix
as a mid-conversation ``system`` message rather than being interpolated into the
system prompt. Interpolating would invalidate the cache on every single turn.
Note Claude Opus 5 only caches prefixes of 512+ tokens; below that the marker is
a no-op and ``cache_read_tokens`` stays at 0.

**Refusal handling.** Claude Opus 5 can decline a request with HTTP 200 and
``stop_reason == "refusal"``. Reading ``content[0]`` unconditionally would crash,
so the parser checks the stop reason first. Server-side fallbacks are enabled by
default, which re-serves a declined request on a fallback model inside the same
call; if the installed SDK or account does not support the parameter, the adapter
disables it permanently after the first failure rather than failing every request.

**Thinking.** Adaptive thinking is on by default for Opus 5, and ``max_tokens``
caps thinking *plus* visible text. Thinking blocks come back with empty text
(``display`` defaults to ``omitted``), so text extraction filters on block type
instead of assuming ``content[0]`` is prose.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from app.core.config import settings
from app.core.errors import FeatureUnavailableError, UpstreamError
from app.core.errors import RateLimitError as AppRateLimitError
from app.core.logging import get_logger
from app.services.llm.base import (
    ImagePart,
    LLMRequest,
    LLMResult,
    StreamEvent,
    TextPart,
    Turn,
    Usage,
)

log = get_logger(__name__)

# Models on which the server-side refusal-fallback parameter is meaningful.
_FALLBACK_MODELS = ("claude-opus-5", "claude-fable-5", "claude-mythos-5")
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicProvider:
    """LLM provider backed by the Anthropic Messages API."""

    name = "anthropic"

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise FeatureUnavailableError(
                "The `anthropic` package is not installed."
            ) from exc

        key = api_key or settings.anthropic_api_key
        if not key:
            raise FeatureUnavailableError(
                "ANTHROPIC_API_KEY is not set. Set it in .env, or set "
                "LLM_PROVIDER=echo to run without a model."
            )

        self._client = AsyncAnthropic(api_key=key, max_retries=3, timeout=120.0)
        self.model = model or settings.llm_model
        # Enabled by default per Anthropic's guidance for Opus 5 / Fable 5: a
        # policy decline is re-served by a fallback model inside the same call.
        self._use_fallbacks = settings.llm_refusal_fallback and self.model in _FALLBACK_MODELS
        # Set to False once we learn the mid-conversation `system` role is
        # unsupported, so we stop paying for a round trip to rediscover it.
        self._supports_system_message = True

    # -- request assembly --------------------------------------------------

    @staticmethod
    def _render_parts(turn: Turn) -> str | list[dict[str, Any]]:
        """Render a turn's parts into Anthropic content blocks."""
        if len(turn.parts) == 1 and isinstance(turn.parts[0], TextPart):
            return turn.parts[0].text

        blocks: list[dict[str, Any]] = []
        for part in turn.parts:
            if isinstance(part, ImagePart):
                # Images before text reads better for the model: it sees the
                # subject before the question about it.
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": part.media_type,
                            "data": part.data_b64,
                        },
                    }
                )
            elif part.text:
                blocks.append({"type": "text", "text": part.text})
        return blocks or ""

    def _build_messages(
        self, request: LLMRequest, *, inline_directive: bool
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": turn.role, "content": self._render_parts(turn)}
            for turn in request.turns
        ]
        if request.directive and not inline_directive:
            # Operator channel: sits *after* the cached history, so per-request
            # context does not invalidate the cached prefix.
            messages.append({"role": "system", "content": request.directive})
        return messages

    def _build_system(self, request: LLMRequest, *, inline_directive: bool) -> Any:
        text = request.system
        if inline_directive and request.directive:
            text = f"{text}\n\n{request.directive}"

        if not settings.llm_prompt_caching:
            return text
        return [
            {
                "type": "text",
                "text": text,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def _build_kwargs(self, request: LLMRequest, *, inline_directive: bool) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens or settings.llm_max_tokens,
            "system": self._build_system(request, inline_directive=inline_directive),
            "messages": self._build_messages(request, inline_directive=inline_directive),
        }

        effort = request.effort or settings.llm_effort
        kwargs["output_config"] = {"effort": effort}

        if settings.llm_thinking == "disabled":
            # Only legal at effort `high` or below; Settings validates this.
            kwargs["thinking"] = {"type": "disabled"}
        else:
            kwargs["thinking"] = {"type": "adaptive"}

        if self._use_fallbacks:
            kwargs["betas"] = [_FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        return kwargs

    def _endpoint(self) -> Any:
        return self._client.beta.messages if self._use_fallbacks else self._client.messages

    # -- response parsing --------------------------------------------------

    @staticmethod
    def _extract_text(message: Any) -> str:
        """Concatenate text blocks. Thinking blocks are skipped by design."""
        chunks: list[str] = []
        for block in getattr(message, "content", []) or []:
            if getattr(block, "type", None) == "text":
                chunks.append(block.text)
        return "".join(chunks).strip()

    @staticmethod
    def _extract_usage(message: Any) -> Usage:
        usage = getattr(message, "usage", None)
        if usage is None:
            return Usage()
        return Usage(
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        )

    def _to_result(self, message: Any) -> LLMResult:
        stop_reason = getattr(message, "stop_reason", None)
        category: str | None = None
        text = self._extract_text(message)

        if stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            log.warning("llm_refusal", model=self.model, category=category)
            if not text:
                text = (
                    "I'm not able to help with that request. If you think this is a "
                    "mistake, try rephrasing it or asking something adjacent."
                )

        return LLMResult(
            text=text,
            model=getattr(message, "model", self.model),
            usage=self._extract_usage(message),
            stop_reason=stop_reason,
            refusal_category=category,
        )

    # -- error mapping -----------------------------------------------------

    @staticmethod
    def _translate(exc: Exception) -> Exception:
        import anthropic

        if isinstance(exc, anthropic.RateLimitError):
            return AppRateLimitError("The model provider is rate limiting this key.")
        if isinstance(exc, anthropic.AuthenticationError):
            return FeatureUnavailableError("The Anthropic API key was rejected.")
        if isinstance(exc, anthropic.APIConnectionError):
            return UpstreamError("Could not reach the model provider.")
        if isinstance(exc, anthropic.APIStatusError):
            return UpstreamError(f"Model provider returned {exc.status_code}.")
        return UpstreamError("The model provider failed.")

    def _is_unsupported_param(self, exc: Exception) -> bool:
        """True when the SDK/account rejected a parameter we can simply drop."""
        text = str(exc).lower()
        return isinstance(exc, TypeError) or any(
            marker in text
            for marker in ("fallbacks", "server-side-fallback", "unexpected keyword", "beta")
        )

    def _is_unsupported_system_role(self, exc: Exception) -> bool:
        text = str(exc).lower()
        return "role 'system'" in text or "system is not supported" in text

    # -- public API --------------------------------------------------------

    async def complete(self, request: LLMRequest) -> LLMResult:
        import anthropic

        for inline_directive in (not self._supports_system_message, True):
            try:
                kwargs = self._build_kwargs(request, inline_directive=inline_directive)
                message = await self._endpoint().create(**kwargs)
                return self._to_result(message)
            except anthropic.BadRequestError as exc:
                if self._use_fallbacks and self._is_unsupported_param(exc):
                    log.warning("disabling_refusal_fallbacks", reason=str(exc)[:200])
                    self._use_fallbacks = False
                    continue
                if not inline_directive and self._is_unsupported_system_role(exc):
                    log.info("disabling_mid_conversation_system", model=self.model)
                    self._supports_system_message = False
                    continue
                raise self._translate(exc) from exc
            except TypeError as exc:
                if self._use_fallbacks:
                    log.warning("disabling_refusal_fallbacks", reason=str(exc)[:200])
                    self._use_fallbacks = False
                    continue
                raise UpstreamError("Invalid request built for the model provider.") from exc
            except Exception as exc:  # noqa: BLE001 - normalised below
                raise self._translate(exc) from exc

        raise UpstreamError("The model provider rejected every request variant.")

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        import anthropic

        inline_directive = not self._supports_system_message
        kwargs = self._build_kwargs(request, inline_directive=inline_directive)

        try:
            async with self._endpoint().stream(**kwargs) as stream:
                async for delta in stream.text_stream:
                    if delta:
                        yield StreamEvent(type="delta", text=delta)
                final = await stream.get_final_message()
            yield StreamEvent(type="done", result=self._to_result(final))
            return
        except anthropic.BadRequestError as exc:
            # One retry with the directive folded into the system prompt, and/or
            # with fallbacks dropped. Nothing has been streamed to the client yet
            # in the common case, but a partial stream is possible — the caller
            # treats a `done` event as authoritative for persistence.
            retryable = (self._use_fallbacks and self._is_unsupported_param(exc)) or (
                not inline_directive and self._is_unsupported_system_role(exc)
            )
            if not retryable:
                raise self._translate(exc) from exc
            if self._use_fallbacks and self._is_unsupported_param(exc):
                self._use_fallbacks = False
            else:
                self._supports_system_message = False
            log.info("stream_retry_after_param_rejection")
        except Exception as exc:  # noqa: BLE001 - normalised below
            raise self._translate(exc) from exc

        # --- single retry pass ---
        try:
            kwargs = self._build_kwargs(request, inline_directive=True)
            async with self._endpoint().stream(**kwargs) as stream:
                async for delta in stream.text_stream:
                    if delta:
                        yield StreamEvent(type="delta", text=delta)
                final = await stream.get_final_message()
            yield StreamEvent(type="done", result=self._to_result(final))
        except Exception as exc:  # noqa: BLE001 - normalised below
            raise self._translate(exc) from exc

    async def count_tokens(self, request: LLMRequest) -> int:
        """Exact input-token count from the provider — never a local estimate."""
        try:
            result = await self._client.messages.count_tokens(
                model=self.model,
                system=self._build_system(request, inline_directive=True),
                messages=self._build_messages(request, inline_directive=True),
            )
            return int(result.input_tokens)
        except Exception as exc:  # noqa: BLE001 - counting must never break a request
            log.warning("count_tokens_failed", error=str(exc))
            return 0

    async def embed(self, texts: list[str]) -> list[list[float]] | None:
        """Anthropic has no embeddings endpoint — the RAG layer uses a local model."""
        return None

"""Provider-agnostic types for the LLM layer.

The rest of the application talks to *this* interface, never to a vendor SDK.
That boundary is what lets the chat orchestrator, the RAG pipeline, and the
tests stay identical whether the request is served by Claude or by the offline
echo stub.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

Role = Literal["user", "assistant"]
Effort = Literal["low", "medium", "high", "xhigh", "max"]


@dataclass(slots=True)
class TextPart:
    text: str


@dataclass(slots=True)
class ImagePart:
    """An inline image. ``data_b64`` must be standard base64 with no newlines."""

    media_type: str
    data_b64: str


Part = TextPart | ImagePart


@dataclass(slots=True)
class Turn:
    role: Role
    parts: list[Part]

    @classmethod
    def text(cls, role: Role, text: str) -> Turn:
        return cls(role=role, parts=[TextPart(text)])

    def plain_text(self) -> str:
        return "\n".join(p.text for p in self.parts if isinstance(p, TextPart))


@dataclass(slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )


@dataclass(slots=True)
class LLMRequest:
    """One inference request, already assembled by the chat orchestrator."""

    system: str
    turns: list[Turn]
    # An operator instruction delivered *after* the cached history, so that
    # per-request context (reply language, retrieved passages) never invalidates
    # the cached system prefix. Providers that cannot express this fold it into
    # the system prompt instead.
    directive: str | None = None
    max_tokens: int | None = None
    effort: Effort | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class LLMResult:
    text: str
    model: str
    usage: Usage
    stop_reason: str | None = None
    refusal_category: str | None = None

    @property
    def was_refused(self) -> bool:
        return self.stop_reason == "refusal"


@dataclass(slots=True)
class StreamEvent:
    """One item on the stream: an incremental delta, or the terminal result."""

    type: Literal["delta", "done"]
    text: str = ""
    result: LLMResult | None = None


@runtime_checkable
class LLMProvider(Protocol):
    """The contract every model backend must satisfy."""

    name: str

    async def complete(self, request: LLMRequest) -> LLMResult:
        """Run a single non-streaming completion."""
        ...

    def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        """Stream a completion, ending with exactly one ``done`` event."""
        ...

    async def count_tokens(self, request: LLMRequest) -> int:
        """Count the input tokens this request would consume."""
        ...

    async def embed(self, texts: list[str]) -> list[list[float]] | None:
        """Return embeddings, or ``None`` if the provider has no embedding API."""
        ...

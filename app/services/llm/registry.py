"""Provider selection.

One process holds one provider instance: the Anthropic client owns an HTTP
connection pool, and rebuilding it per request would throw away keep-alive
connections and TLS sessions.
"""

from __future__ import annotations

from app.core.config import settings
from app.core.logging import get_logger
from app.services.llm.base import LLMProvider

log = get_logger(__name__)

_provider: LLMProvider | None = None


def get_llm_provider() -> LLMProvider:
    global _provider
    if _provider is not None:
        return _provider

    if settings.llm_provider == "echo":
        from app.services.llm.echo_provider import EchoProvider

        _provider = EchoProvider()
    else:
        from app.services.llm.anthropic_provider import AnthropicProvider

        _provider = AnthropicProvider()

    log.info("llm_provider_ready", provider=_provider.name, model=settings.llm_model)
    return _provider


def reset_llm_provider() -> None:
    """Test hook — drop the cached provider so settings changes take effect."""
    global _provider
    _provider = None

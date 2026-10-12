"""Structured logging with per-request correlation IDs.

In development you get colourised key=value lines; in production (``LOG_JSON=true``)
you get one JSON object per line, which is what log shippers actually want.

The request ID is carried in a :class:`contextvars.ContextVar` so *every* log line
emitted while handling a request — including from deep inside a service — is
automatically tagged, without threading a logger through call signatures.
"""

from __future__ import annotations

import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Any

import structlog

from app.core.config import settings

request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)
user_id_ctx: ContextVar[str | None] = ContextVar("user_id", default=None)


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


def _inject_context(_logger: Any, _name: str, event_dict: dict) -> dict:
    if (rid := request_id_ctx.get()) is not None:
        event_dict["request_id"] = rid
    if (uid := user_id_ctx.get()) is not None:
        event_dict["user_id"] = uid
    return event_dict


def configure_logging() -> None:
    """Idempotently configure stdlib logging + structlog."""
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level, force=True)
    # uvicorn.access is noisy and duplicates our own middleware log line.
    logging.getLogger("uvicorn.access").disabled = True

    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,
        _inject_context,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if settings.log_json
        else structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())
    )

    structlog.configure(
        processors=[*shared, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)

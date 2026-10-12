"""Liveness and readiness probes.

``/health`` answers "is this process alive" and must never touch a dependency —
a health check that queries the database will fail the whole service during a
brief DB blip and take a healthy process out of rotation.

``/ready`` answers "can this process serve traffic" and *does* check
dependencies. It also reports which optional capabilities are actually wired up,
which turns "why is speech-to-text 503ing?" into a one-request answer.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from sqlalchemy import text

from app import __version__
from app.api.deps import SessionDep
from app.core.cache import cache
from app.core.config import settings
from app.core.logging import get_logger
from app.services.rag.embeddings import get_embedder
from app.services.rag.store import get_vector_store
from app.services.stt import speech_to_text
from app.services.tts import text_to_speech

log = get_logger(__name__)
router = APIRouter()


@router.get("/health", summary="Liveness probe")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": __version__,
        "environment": settings.environment,
    }


@router.get("/ready", summary="Readiness probe and capability report")
async def ready(session: SessionDep) -> dict[str, Any]:
    checks: dict[str, Any] = {}

    try:
        await session.execute(text("SELECT 1"))
        checks["database"] = {
            "status": "ok",
            "dialect": "postgresql" if settings.is_postgres else "sqlite",
        }
    except Exception as exc:  # noqa: BLE001 - the probe reports, it does not raise
        checks["database"] = {"status": "error", "detail": str(exc)[:200]}

    checks["cache"] = {"status": "ok", "backend": cache.backend}

    embedder = get_embedder()
    checks["capabilities"] = {
        "llm": {"provider": settings.llm_provider, "model": settings.llm_model},
        "vision": True,
        "speech_to_text": speech_to_text.available,
        "text_to_speech": text_to_speech.available,
        "rag": {
            "enabled": settings.rag_enabled,
            "vector_store": get_vector_store().name,
            "embedder": embedder.name,
            # False means retrieval is lexical-only — see requirements-ml.txt.
            "semantic": embedder.is_semantic,
        },
    }

    ok = checks["database"]["status"] == "ok"
    return {"status": "ready" if ok else "degraded", "checks": checks}

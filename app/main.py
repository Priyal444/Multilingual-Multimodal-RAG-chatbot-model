"""Application entrypoint: lifespan, middleware, routing.

Run with::

    uvicorn app.main:app --reload
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app import __version__
from app.api.v1.router import api_router
from app.core.cache import cache
from app.core.config import settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging, get_logger, new_request_id, request_id_ctx
from app.db.session import dispose_engine

log = get_logger(__name__)

DESCRIPTION = """
A production-oriented multimodal, multilingual chat API.

**Multimodal** — text, images (vision), voice notes (speech-to-text), spoken
replies (text-to-speech), and uploaded PDFs/DOCX answered with citations.

**Multilingual** — the reply language is decided per turn from Unicode script
plus statistical detection, and the language directive is delivered as a
mid-conversation operator message so the cached system prompt stays byte-stable.

Authenticate with `POST /api/v1/auth/register`, then send the returned
`access_token` as `Authorization: Bearer <token>`.
"""


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    log.info(
        "startup",
        app=settings.app_name,
        version=__version__,
        environment=settings.environment,
        llm_provider=settings.llm_provider,
        model=settings.llm_model,
        database="postgresql" if settings.is_postgres else "sqlite",
    )
    await cache.connect()
    try:
        yield
    finally:
        log.info("shutdown")
        await cache.close()
        await dispose_engine()


app = FastAPI(
    title=settings.app_name,
    description=DESCRIPTION,
    version=__version__,
    lifespan=lifespan,
    # Hide interactive docs in production; the OpenAPI schema is still a map of
    # the attack surface even when every route requires auth.
    docs_url=None if settings.is_production else "/docs",
    redoc_url=None if settings.is_production else "/redoc",
    openapi_url=None if settings.is_production else "/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Streaming clients need to read these off the response.
    expose_headers=["X-Request-ID", "X-Response-Time-Ms", "X-Voice", "X-Language"],
)
# SSE frames are excluded by minimum_size in practice; compressing a live stream
# would buffer it and defeat the point.
app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.middleware("http")
async def observability(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Assign a request ID, time the request, and emit one structured log line."""
    # Honour an upstream ID so a trace survives across services.
    request_id = request.headers.get("x-request-id") or new_request_id()
    token = request_id_ctx.set(request_id)
    request.state.request_id = request_id

    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        # The exception handlers build the response; this only records timing
        # for the failure before re-raising.
        log.exception(
            "request_failed",
            method=request.method,
            path=request.url.path,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        raise
    finally:
        request_id_ctx.reset(token)

    duration_ms = int((time.perf_counter() - started) * 1000)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = str(duration_ms)

    if request.url.path not in {"/api/v1/health", "/api/v1/ready"}:
        log.info(
            "request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=duration_ms,
        )
    return response


@app.middleware("http")
async def security_headers(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    if settings.is_production:
        response.headers.setdefault(
            "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
        )
    return response


register_exception_handlers(app)
app.include_router(api_router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {
        "service": settings.app_name,
        "version": __version__,
        "docs": "/docs" if not settings.is_production else "disabled",
        "health": "/api/v1/health",
    }

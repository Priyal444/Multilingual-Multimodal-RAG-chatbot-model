"""Test fixtures.

The environment is configured *before* any ``app`` module is imported, because
``app.core.config`` builds its ``Settings`` object at import time. Importing the
app first and patching afterwards would leave half the modules holding the real
configuration — including a live Anthropic client.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="polyglot-tests-"))

os.environ.update(
    ENVIRONMENT="development",
    DEBUG="false",
    LOG_LEVEL="WARNING",
    LOG_JSON="false",
    # The offline stub: no API key, no network, deterministic output.
    LLM_PROVIDER="echo",
    ANTHROPIC_API_KEY="test-key-not-used",
    JWT_SECRET_KEY="test-secret-key-for-the-suite-only",
    # A file rather than :memory:, because each aiosqlite connection would
    # otherwise get its own empty database.
    DATABASE_URL=f"sqlite+aiosqlite:///{(_TMP / 'test.db').as_posix()}",
    REDIS_URL="",
    RATE_LIMIT_ENABLED="false",
    STORAGE_DIR=str(_TMP / "storage"),
    RAG_ENABLED="true",
    EMBEDDING_DIM="384",
)

import pytest  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

from app.db.session import Base, SessionLocal, engine  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
async def database() -> AsyncIterator[None]:
    """A fresh schema per test — cheap on SQLite, and removes ordering coupling."""
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
        await connection.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    # ASGITransport runs the app in-process: no socket, no port, no flakiness.
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        yield http


@pytest.fixture
async def session() -> AsyncIterator:
    async with SessionLocal() as db:
        yield db


@pytest.fixture
async def auth(client: AsyncClient) -> dict[str, str]:
    """Register a unique user and return an Authorization header."""
    email = f"user-{uuid.uuid4().hex[:10]}@example.com"
    response = await client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "correct-horse-battery", "full_name": "Test User"},
    )
    assert response.status_code == 201, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}

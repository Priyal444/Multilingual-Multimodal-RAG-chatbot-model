"""A tiny async cache facade over Redis, with an in-process fallback.

The fallback is not a toy: it is what runs in tests and in single-process local
development, so the rest of the codebase never has to ask "is Redis configured?".
It is deliberately *not* suitable for multi-worker deployments — that is exactly
what ``REDIS_URL`` is for, and :meth:`Cache.backend` reports which one is live so
``/health`` can surface it.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)


class _InMemoryBackend:
    """Process-local dict with TTLs. Single-worker only."""

    def __init__(self) -> None:
        self._data: dict[str, tuple[float | None, str]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> str | None:
        async with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if expires_at is not None and expires_at < time.monotonic():
                self._data.pop(key, None)
                return None
            return value

    async def set(self, key: str, value: str, ttl: int | None = None) -> None:
        async with self._lock:
            expires_at = time.monotonic() + ttl if ttl else None
            self._data[key] = (expires_at, value)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._data.pop(key, None)

    async def incr_with_ttl(self, key: str, ttl: int) -> int:
        """Increment a counter, setting the TTL only on first creation."""
        async with self._lock:
            entry = self._data.get(key)
            now = time.monotonic()
            if entry is None or (entry[0] is not None and entry[0] < now):
                self._data[key] = (now + ttl, "1")
                return 1
            expires_at, value = entry
            count = int(value) + 1
            self._data[key] = (expires_at, str(count))
            return count

    async def ttl(self, key: str) -> int:
        async with self._lock:
            entry = self._data.get(key)
            if entry is None or entry[0] is None:
                return -1
            return max(0, int(entry[0] - time.monotonic()))

    async def close(self) -> None:
        self._data.clear()


class Cache:
    """Facade used by the application. Chooses Redis when configured."""

    def __init__(self) -> None:
        self._redis: Any = None
        self._memory = _InMemoryBackend()
        self._backend = "memory"

    async def connect(self) -> None:
        if not settings.redis_url:
            log.info("cache_backend_selected", backend="memory")
            return
        try:
            import redis.asyncio as aioredis

            client = aioredis.from_url(
                settings.redis_url, encoding="utf-8", decode_responses=True
            )
            await client.ping()
            self._redis = client
            self._backend = "redis"
            log.info("cache_backend_selected", backend="redis")
        except Exception as exc:  # noqa: BLE001 - degrade, never fail startup
            log.warning("redis_unavailable_using_memory", error=str(exc))

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
        await self._memory.close()

    @property
    def backend(self) -> str:
        return self._backend

    async def get(self, key: str) -> str | None:
        if self._redis is not None:
            return await self._redis.get(key)
        return await self._memory.get(key)

    async def set(self, key: str, value: str, ttl: int | None = None) -> None:
        if self._redis is not None:
            await self._redis.set(key, value, ex=ttl)
            return
        await self._memory.set(key, value, ttl)

    async def delete(self, key: str) -> None:
        if self._redis is not None:
            await self._redis.delete(key)
            return
        await self._memory.delete(key)

    async def get_json(self, key: str) -> Any | None:
        raw = await self.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    async def set_json(self, key: str, value: Any, ttl: int | None = None) -> None:
        await self.set(key, json.dumps(value, ensure_ascii=False), ttl)

    async def incr_with_ttl(self, key: str, ttl: int) -> tuple[int, int]:
        """Return ``(count, seconds_until_reset)`` for a fixed-window counter."""
        if self._redis is not None:
            pipe = self._redis.pipeline()
            pipe.incr(key)
            pipe.ttl(key)
            count, remaining = await pipe.execute()
            if remaining is None or remaining < 0:
                await self._redis.expire(key, ttl)
                remaining = ttl
            return int(count), int(remaining)
        count = await self._memory.incr_with_ttl(key, ttl)
        return count, await self._memory.ttl(key)


cache = Cache()

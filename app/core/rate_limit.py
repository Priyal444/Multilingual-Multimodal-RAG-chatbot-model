"""Fixed-window rate limiting, keyed per authenticated user (or per IP).

Fixed windows are chosen over a sliding log deliberately: one counter and one
TTL per key, O(1) memory, and the burst behaviour at window boundaries is
acceptable for a chat API. The limiter emits the standard ``X-RateLimit-*``
headers so clients can back off before they get a 429.
"""

from __future__ import annotations

import time

from fastapi import Request, Response

from app.core.cache import cache
from app.core.config import settings
from app.core.errors import RateLimitError


def _client_key(request: Request) -> str:
    """Prefer the authenticated user; fall back to the client IP."""
    user_id = getattr(request.state, "user_id", None)
    if user_id:
        return f"user:{user_id}"
    # X-Forwarded-For is only trusted when a proxy sets it; take the leftmost hop.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return f"ip:{forwarded.split(',')[0].strip()}"
    return f"ip:{request.client.host if request.client else 'unknown'}"


async def enforce_rate_limit(request: Request, response: Response | None = None) -> None:
    """Raise :class:`RateLimitError` when the caller is over budget."""
    if not settings.rate_limit_enabled:
        return

    window = settings.rate_limit_window_seconds
    limit = settings.rate_limit_requests
    bucket = int(time.time() // window)
    key = f"ratelimit:{_client_key(request)}:{bucket}"

    count, reset_in = await cache.incr_with_ttl(key, window)
    remaining = max(0, limit - count)

    if response is not None:
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(max(0, reset_in))

    if count > limit:
        raise RateLimitError(
            f"Rate limit of {limit} requests per {window}s exceeded.",
            retry_after_seconds=max(1, reset_in),
        )

"""Shared FastAPI dependencies: authentication, ownership checks, rate limiting."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import AuthenticationError, NotFoundError
from app.core.logging import user_id_ctx
from app.core.rate_limit import enforce_rate_limit
from app.core.security import decode_token
from app.db.models import Conversation, Document, User
from app.db.session import get_session

# auto_error=False so a missing header raises our own error envelope rather than
# FastAPI's default `{"detail": ...}` shape.
_bearer = HTTPBearer(auto_error=False, description="JWT access token")

SessionDep = Annotated[AsyncSession, Depends(get_session)]


async def get_current_user(
    request: Request,
    session: SessionDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)] = None,
) -> User:
    if credentials is None or not credentials.credentials:
        raise AuthenticationError("Provide a bearer token in the Authorization header.")

    payload = decode_token(credentials.credentials, expected_type="access")
    user = await session.get(User, payload["sub"])
    if user is None:
        raise AuthenticationError("The account for this token no longer exists.")
    if not user.is_active:
        raise AuthenticationError("This account is disabled.")

    # Make the identity available to logging and to the rate limiter, which
    # runs as a separate dependency and needs the per-user key.
    request.state.user_id = user.id
    user_id_ctx.set(user.id)
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


async def rate_limited(request: Request, response: Response) -> None:
    """Attach to routes that hit the model or do heavy work."""
    await enforce_rate_limit(request, response)


RateLimited = Annotated[None, Depends(rate_limited)]


async def get_owned_conversation(
    conversation_id: str, session: SessionDep, user: CurrentUser
) -> Conversation:
    """Load a conversation, enforcing ownership.

    Returns 404 rather than 403 for someone else's conversation: a 403 confirms
    the ID exists, which is an enumeration oracle.
    """
    conversation = (
        await session.execute(
            select(Conversation).where(
                Conversation.id == conversation_id, Conversation.user_id == user.id
            )
        )
    ).scalar_one_or_none()
    if conversation is None:
        raise NotFoundError("Conversation not found.")
    return conversation


async def get_owned_document(
    document_id: str, session: SessionDep, user: CurrentUser
) -> Document:
    document = (
        await session.execute(
            select(Document).where(Document.id == document_id, Document.user_id == user.id)
        )
    ).scalar_one_or_none()
    if document is None:
        raise NotFoundError("Document not found.")
    return document

"""Conversation history: list, read, rename, archive, delete."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy import func, select

from app.api.deps import CurrentUser, SessionDep, get_owned_conversation
from app.db.models import Conversation, Message
from app.schemas.chat import (
    ConversationDetailResponse,
    ConversationListResponse,
    ConversationResponse,
    MessageResponse,
    UpdateConversationRequest,
)

router = APIRouter()

OwnedConversation = Annotated[Conversation, Depends(get_owned_conversation)]


@router.get("", response_model=ConversationListResponse, summary="List conversations")
async def list_conversations(
    user: CurrentUser,
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    include_archived: bool = False,
) -> ConversationListResponse:
    filters = [Conversation.user_id == user.id]
    if not include_archived:
        filters.append(Conversation.is_archived.is_(False))

    total = (
        await session.execute(
            select(func.count()).select_from(Conversation).where(*filters)
        )
    ).scalar_one()

    rows = (
        await session.execute(
            select(Conversation)
            .where(*filters)
            .order_by(Conversation.updated_at.desc())
            .limit(limit)
            .offset(offset)
        )
    ).scalars().all()

    return ConversationListResponse(
        items=[ConversationResponse.model_validate(row) for row in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{conversation_id}",
    response_model=ConversationDetailResponse,
    summary="Read a conversation with its messages",
)
async def get_conversation(
    conversation: OwnedConversation,
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> ConversationDetailResponse:
    # Newest-first with a limit, then reversed: a long conversation should show
    # its most recent messages, not its first hundred.
    rows = (
        await session.execute(
            select(Message)
            .where(Message.conversation_id == conversation.id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
    ).scalars().all()

    # Built field-by-field rather than with `model_validate(conversation)`:
    # validating the ORM object would touch the lazy `Conversation.messages`
    # relationship, which raises MissingGreenlet under the async engine.
    return ConversationDetailResponse(
        **ConversationResponse.model_validate(conversation).model_dump(),
        messages=[MessageResponse.model_validate(m) for m in reversed(rows)],
    )


@router.patch(
    "/{conversation_id}",
    response_model=ConversationResponse,
    summary="Rename or archive a conversation",
)
async def update_conversation(
    payload: UpdateConversationRequest,
    conversation: OwnedConversation,
    session: SessionDep,
) -> Conversation:
    if payload.title is not None:
        conversation.title = payload.title
    if payload.is_archived is not None:
        conversation.is_archived = payload.is_archived
    await session.flush()
    return conversation


@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a conversation and its messages",
)
async def delete_conversation(
    conversation: OwnedConversation, session: SessionDep
) -> None:
    # Messages and attachments go with it via ON DELETE CASCADE.
    await session.delete(conversation)

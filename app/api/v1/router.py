"""API v1 router assembly."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import auth, chat, conversations, documents, health, media

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(
    conversations.router, prefix="/conversations", tags=["conversations"]
)
api_router.include_router(media.router, prefix="/media", tags=["media"])
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])

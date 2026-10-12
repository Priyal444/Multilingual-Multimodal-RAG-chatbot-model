"""Document / RAG schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    media_type: str
    size_bytes: int
    status: str
    error: str | None
    language: str | None
    chunk_count: int
    created_at: datetime


class DocumentListResponse(BaseModel):
    items: list[DocumentResponse]
    total: int


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    document_ids: list[str] | None = None


class SearchHit(BaseModel):
    document_id: str
    filename: str
    part: int
    score: float
    content: str


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]
    embedder: str
    semantic: bool = Field(
        description="False when the lexical fallback embedder is in use."
    )

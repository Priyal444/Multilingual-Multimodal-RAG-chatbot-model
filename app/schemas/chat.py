"""Chat and conversation schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.services.language import normalize_code


class ChatRequest(BaseModel):
    message: str = Field(default="", max_length=32_000)
    conversation_id: str | None = None
    language: str | None = Field(
        default=None, description="Force the reply language (BCP-47). Omit to auto-detect."
    )
    # Base64 data URLs or bare base64 payloads. For binary uploads use
    # POST /api/v1/chat/multipart instead.
    images: list[str] = Field(default_factory=list, max_length=4)
    document_ids: list[str] | None = Field(
        default=None, description="Restrict retrieval to these documents."
    )
    use_rag: bool = True
    stream: bool = False

    @field_validator("language")
    @classmethod
    def _normalize(cls, v: str | None) -> str | None:
        return normalize_code(v)


class SourceRef(BaseModel):
    document_id: str
    filename: str
    part: int
    score: float


class UsageInfo(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


class ChatResponse(BaseModel):
    conversation_id: str
    user_message_id: str
    assistant_message_id: str
    reply: str
    reply_language: str
    detected_language: str
    detection_confidence: float
    model: str
    usage: UsageInfo
    latency_ms: int
    sources: list[SourceRef] = Field(default_factory=list)
    stop_reason: str | None = None


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    role: str
    content: str
    language: str | None
    model: str | None
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: int | None
    created_at: datetime


class ConversationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    language: str | None
    is_archived: bool
    created_at: datetime
    updated_at: datetime


class ConversationDetailResponse(ConversationResponse):
    messages: list[MessageResponse] = Field(default_factory=list)


class ConversationListResponse(BaseModel):
    items: list[ConversationResponse]
    total: int
    limit: int
    offset: int


class UpdateConversationRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    is_archived: bool | None = None


class TokenEstimateRequest(BaseModel):
    message: str = Field(max_length=32_000)
    conversation_id: str | None = None
    language: str | None = None


class TokenEstimateResponse(BaseModel):
    input_tokens: int
    model: str
    note: str = "Exact count from the provider, including the cached system prompt."

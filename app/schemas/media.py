"""Speech and vision schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.services.language import normalize_code


class TranscriptionResponse(BaseModel):
    text: str
    language: str | None
    language_name: str | None
    language_confidence: float
    duration_seconds: float


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    language: str | None = Field(
        default=None, description="Reply language; picks a matching neural voice."
    )
    voice: str | None = Field(
        default=None, description="Explicit voice short-name, e.g. hi-IN-SwaraNeural."
    )
    rate: str = Field(default="+0%", pattern=r"^[+-]\d{1,3}%$")

    @field_validator("language")
    @classmethod
    def _normalize(cls, v: str | None) -> str | None:
        return normalize_code(v)


class VoiceInfo(BaseModel):
    short_name: str
    locale: str
    language: str
    gender: str | None = None


class DetectLanguageRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)


class DetectLanguageResponse(BaseModel):
    code: str
    name: str
    confidence: float
    method: str
    script: str
    is_rtl: bool

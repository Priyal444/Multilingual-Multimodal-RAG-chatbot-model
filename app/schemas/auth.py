"""Auth request/response schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from app.services.language import normalize_code


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=72, description="8-72 characters")
    full_name: str | None = Field(default=None, max_length=200)
    preferred_language: str | None = Field(
        default=None,
        max_length=16,
        description="BCP-47 tag. Omit to have replies mirror each message's language.",
    )

    @field_validator("preferred_language")
    @classmethod
    def _normalize(cls, v: str | None) -> str | None:
        return normalize_code(v)

    @field_validator("password")
    @classmethod
    def _password_bytes(cls, v: str) -> str:
        # bcrypt silently truncates past 72 *bytes*; a 40-character emoji
        # password can exceed that even though len() looks fine.
        if len(v.encode("utf-8")) > 72:
            raise ValueError("Password must be at most 72 bytes when UTF-8 encoded.")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    email: str
    full_name: str | None
    preferred_language: str | None
    created_at: datetime


class UpdateUserRequest(BaseModel):
    full_name: str | None = Field(default=None, max_length=200)
    preferred_language: str | None = Field(default=None, max_length=16)

    @field_validator("preferred_language")
    @classmethod
    def _normalize(cls, v: str | None) -> str | None:
        return normalize_code(v)

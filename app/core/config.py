"""Typed application settings, loaded once from the environment.

Everything the app can be tuned with lives here. Nothing else in the codebase
reads ``os.environ`` directly — that keeps configuration auditable and makes
tests able to override a single object.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Application -----------------------------------------------------
    app_name: str = "Polyglot"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = True
    log_level: str = "INFO"
    log_json: bool = False
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:8501"])

    # --- Security --------------------------------------------------------
    jwt_secret_key: str = "insecure-development-key-change-me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 14

    # --- LLM -------------------------------------------------------------
    anthropic_api_key: str | None = None
    llm_provider: Literal["anthropic", "echo"] = "anthropic"
    llm_model: str = "claude-opus-5"
    llm_max_tokens: int = 4096
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    llm_thinking: Literal["adaptive", "disabled"] = "adaptive"
    llm_prompt_caching: bool = True
    llm_refusal_fallback: bool = True

    # --- Database --------------------------------------------------------
    database_url: str = "sqlite+aiosqlite:///./polyglot.db"
    db_echo: bool = False

    # --- Redis -----------------------------------------------------------
    redis_url: str | None = None

    # --- Rate limiting ---------------------------------------------------
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60

    # --- Uploads ---------------------------------------------------------
    storage_dir: Path = Path("./storage")
    max_image_bytes: int = 5 * 1024 * 1024
    max_audio_bytes: int = 25 * 1024 * 1024
    max_document_bytes: int = 20 * 1024 * 1024
    # Long-edge cap in pixels before an image is sent to the model. Claude Opus 5
    # accepts up to 2576px, but image tokens scale with area — 1568 is the
    # sweet spot for chat unless you need to read fine print in screenshots.
    max_image_edge: int = 1568
    max_images_per_message: int = 4

    # --- Speech ----------------------------------------------------------
    stt_model: str = "small"
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"
    tts_default_voice: str | None = None

    # --- RAG -------------------------------------------------------------
    rag_enabled: bool = True
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dim: int = 384
    rag_chunk_chars: int = 1200
    rag_chunk_overlap: int = 200
    rag_top_k: int = 5
    rag_min_score: float = 0.15

    # --- Derived ---------------------------------------------------------
    @property
    def is_postgres(self) -> bool:
        return self.database_url.startswith("postgresql")

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, v: object) -> object:
        """Accept both a JSON list and a plain comma-separated string."""
        if isinstance(v, str) and not v.strip().startswith("["):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @field_validator("storage_dir", mode="after")
    @classmethod
    def _ensure_storage(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v

    @model_validator(mode="after")
    def _check_thinking_effort(self) -> Settings:
        """Claude Opus 5 rejects disabled thinking above `high` effort with a 400.

        Catching it here turns a per-request runtime failure into a startup
        failure with a message that says what to change.
        """
        if self.llm_thinking == "disabled" and self.llm_effort in {"xhigh", "max"}:
            raise ValueError(
                f"LLM_THINKING=disabled is only accepted at effort 'high' or below "
                f"(got LLM_EFFORT={self.llm_effort}). Either lower LLM_EFFORT or "
                f"set LLM_THINKING=adaptive."
            )
        return self

    @model_validator(mode="after")
    def _check_production_secrets(self) -> Settings:
        if self.is_production and self.jwt_secret_key.startswith("insecure-"):
            raise ValueError(
                "JWT_SECRET_KEY must be set to a real secret when ENVIRONMENT=production."
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached accessor — import this, never instantiate ``Settings`` directly."""
    return Settings()


settings = get_settings()

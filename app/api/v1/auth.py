"""Registration, login, token refresh, and profile."""

from __future__ import annotations

from fastapi import APIRouter, status
from sqlalchemy import select

from app.api.deps import CurrentUser, SessionDep
from app.core.config import settings
from app.core.errors import AuthenticationError, ConflictError
from app.core.logging import get_logger
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.db.models import User
from app.schemas.auth import (
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    UpdateUserRequest,
    UserResponse,
)

log = get_logger(__name__)
router = APIRouter()


def _tokens(user: User) -> TokenResponse:
    return TokenResponse(
        access_token=create_access_token(user.id),
        refresh_token=create_refresh_token(user.id),
        expires_in=settings.access_token_expire_minutes * 60,
    )


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an account",
)
async def register(payload: RegisterRequest, session: SessionDep) -> TokenResponse:
    email = payload.email.lower()
    existing = (
        await session.execute(select(User.id).where(User.email == email))
    ).scalar_one_or_none()
    if existing is not None:
        raise ConflictError("An account with that email already exists.")

    user = User(
        email=email,
        hashed_password=hash_password(payload.password),
        full_name=payload.full_name,
        preferred_language=payload.preferred_language,
    )
    session.add(user)
    await session.flush()

    log.info("user_registered", user_id=user.id)
    return _tokens(user)


@router.post("/login", response_model=TokenResponse, summary="Exchange credentials for tokens")
async def login(payload: LoginRequest, session: SessionDep) -> TokenResponse:
    user = (
        await session.execute(select(User).where(User.email == payload.email.lower()))
    ).scalar_one_or_none()

    # Identical error and comparable timing whether or not the account exists —
    # a distinguishable response turns login into an account-enumeration oracle.
    if user is None:
        verify_password(payload.password, "$2b$12$" + "x" * 53)
        raise AuthenticationError("Incorrect email or password.")
    if not verify_password(payload.password, user.hashed_password):
        raise AuthenticationError("Incorrect email or password.")
    if not user.is_active:
        raise AuthenticationError("This account is disabled.")

    log.info("user_logged_in", user_id=user.id)
    return _tokens(user)


@router.post("/refresh", response_model=TokenResponse, summary="Rotate tokens")
async def refresh(payload: RefreshRequest, session: SessionDep) -> TokenResponse:
    claims = decode_token(payload.refresh_token, expected_type="refresh")
    user = await session.get(User, claims["sub"])
    if user is None or not user.is_active:
        raise AuthenticationError("This refresh token is no longer valid.")
    return _tokens(user)


@router.get("/me", response_model=UserResponse, summary="Current profile")
async def me(user: CurrentUser) -> User:
    return user


@router.patch("/me", response_model=UserResponse, summary="Update profile")
async def update_me(
    payload: UpdateUserRequest, user: CurrentUser, session: SessionDep
) -> User:
    if payload.full_name is not None:
        user.full_name = payload.full_name
    # `preferred_language` is deliberately settable to null, which restores
    # "mirror whatever language the user writes in".
    if "preferred_language" in payload.model_fields_set:
        user.preferred_language = payload.preferred_language
    await session.flush()
    return user

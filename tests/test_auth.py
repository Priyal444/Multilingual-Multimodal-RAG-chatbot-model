from __future__ import annotations

import pytest
from httpx import AsyncClient

from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.core.errors import AuthenticationError

REGISTRATION = {
    "email": "alice@example.com",
    "password": "correct-horse-battery",
    "full_name": "Alice",
    "preferred_language": "hi",
}


async def test_register_returns_tokens(client: AsyncClient) -> None:
    response = await client.post("/api/v1/auth/register", json=REGISTRATION)
    assert response.status_code == 201

    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"] and body["refresh_token"]


async def test_duplicate_email_conflicts(client: AsyncClient) -> None:
    await client.post("/api/v1/auth/register", json=REGISTRATION)
    response = await client.post("/api/v1/auth/register", json=REGISTRATION)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "conflict"


async def test_login_and_profile(client: AsyncClient) -> None:
    await client.post("/api/v1/auth/register", json=REGISTRATION)

    login = await client.post(
        "/api/v1/auth/login",
        json={"email": REGISTRATION["email"], "password": REGISTRATION["password"]},
    )
    assert login.status_code == 200

    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    me = await client.get("/api/v1/auth/me", headers=headers)
    assert me.status_code == 200
    assert me.json()["preferred_language"] == "hi"


async def test_login_does_not_reveal_whether_an_account_exists(
    client: AsyncClient,
) -> None:
    await client.post("/api/v1/auth/register", json=REGISTRATION)

    wrong_password = await client.post(
        "/api/v1/auth/login",
        json={"email": REGISTRATION["email"], "password": "not-the-password"},
    )
    unknown_account = await client.post(
        "/api/v1/auth/login",
        json={"email": "nobody@example.com", "password": "not-the-password"},
    )

    assert wrong_password.status_code == unknown_account.status_code == 401
    assert wrong_password.json()["error"] == unknown_account.json()["error"]


async def test_protected_route_requires_a_token(client: AsyncClient) -> None:
    response = await client.get("/api/v1/auth/me")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthenticated"


async def test_refresh_token_cannot_be_used_as_an_access_token(
    client: AsyncClient,
) -> None:
    tokens = (await client.post("/api/v1/auth/register", json=REGISTRATION)).json()
    response = await client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {tokens['refresh_token']}"},
    )
    assert response.status_code == 401


async def test_refresh_rotates_tokens(client: AsyncClient) -> None:
    tokens = (await client.post("/api/v1/auth/register", json=REGISTRATION)).json()
    response = await client.post(
        "/api/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]}
    )
    assert response.status_code == 200
    assert response.json()["access_token"]


async def test_preferred_language_can_be_cleared(client: AsyncClient) -> None:
    tokens = (await client.post("/api/v1/auth/register", json=REGISTRATION)).json()
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}

    response = await client.patch(
        "/api/v1/auth/me", json={"preferred_language": None}, headers=headers
    )
    assert response.status_code == 200
    # Null must mean "mirror the user's language", not "leave hi in place".
    assert response.json()["preferred_language"] is None


# --- unit-level -----------------------------------------------------------


def test_password_hash_round_trip() -> None:
    hashed = hash_password("s3cret-passphrase")
    assert hashed != "s3cret-passphrase"
    assert verify_password("s3cret-passphrase", hashed)
    assert not verify_password("wrong", hashed)


def test_password_over_bcrypt_byte_limit_is_rejected() -> None:
    # bcrypt truncates at 72 bytes; silently accepting a longer password means
    # the ignored tail gives a false sense of strength.
    with pytest.raises(ValueError):
        hash_password("😀" * 20)  # 80 bytes UTF-8


def test_token_type_is_enforced() -> None:
    access = create_access_token("user-1")
    refresh = create_refresh_token("user-1")

    assert decode_token(access, "access")["sub"] == "user-1"
    with pytest.raises(AuthenticationError):
        decode_token(refresh, "access")


def test_tampered_token_is_rejected() -> None:
    token = create_access_token("user-1")
    with pytest.raises(AuthenticationError):
        decode_token(token[:-4] + "aaaa", "access")

"""Document upload, indexing, retrieval, and isolation between users."""

from __future__ import annotations

from httpx import AsyncClient

HANDBOOK = """\
Employee Handbook

Section 1: Leave policy
Employees accrue eighteen days of paid annual leave per calendar year.
Unused leave carries over to the following year up to a maximum of nine days.

Section 2: Expense reimbursement
Submit receipts within thirty days of the expense. Reimbursements are paid with
the next payroll run. Expenses above fifty thousand rupees need director approval.

Section 3: Remote work
Employees may work remotely up to three days per week with manager approval.
"""


async def _upload(client: AsyncClient, auth: dict[str, str], body: str = HANDBOOK) -> dict:
    response = await client.post(
        "/api/v1/documents",
        files={"file": ("handbook.txt", body.encode(), "text/plain")},
        headers=auth,
    )
    assert response.status_code == 201, response.text
    return response.json()


async def test_upload_indexes_a_document(client: AsyncClient, auth: dict[str, str]) -> None:
    document = await _upload(client, auth)
    assert document["status"] == "ready"
    assert document["chunk_count"] >= 1
    assert document["language"] == "en"


async def test_documents_are_listed_and_deletable(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    document = await _upload(client, auth)

    listing = await client.get("/api/v1/documents", headers=auth)
    assert listing.json()["total"] == 1

    assert (
        await client.delete(f"/api/v1/documents/{document['id']}", headers=auth)
    ).status_code == 204
    assert (await client.get("/api/v1/documents", headers=auth)).json()["total"] == 0


async def test_search_retrieves_the_relevant_passage(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    await _upload(client, auth)

    response = await client.post(
        "/api/v1/documents/search",
        json={"query": "expense reimbursement receipts", "top_k": 3},
        headers=auth,
    )
    assert response.status_code == 200

    body = response.json()
    assert body["hits"]
    assert any("receipt" in hit["content"].lower() for hit in body["hits"])
    # The response must say whether retrieval was semantic or the lexical
    # fallback — otherwise a degraded deployment looks identical to a healthy one.
    assert isinstance(body["semantic"], bool)


async def test_hits_are_ordered_by_descending_score(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    await _upload(client, auth)
    response = await client.post(
        "/api/v1/documents/search", json={"query": "annual leave carry over"}, headers=auth
    )
    scores = [hit["score"] for hit in response.json()["hits"]]
    assert scores == sorted(scores, reverse=True)


async def test_retrieval_is_scoped_to_the_owner(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    await _upload(client, auth)

    other = await client.post(
        "/api/v1/auth/register",
        json={"email": "outsider@example.com", "password": "correct-horse-battery"},
    )
    stranger = {"Authorization": f"Bearer {other.json()['access_token']}"}

    response = await client.post(
        "/api/v1/documents/search",
        json={"query": "expense reimbursement receipts"},
        headers=stranger,
    )
    assert response.json()["hits"] == []


async def test_chat_cites_the_document_it_used(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    document = await _upload(client, auth)

    response = await client.post(
        "/api/v1/chat",
        json={"message": "How many days of annual leave do employees get?"},
        headers=auth,
    )
    body = response.json()
    assert body["sources"], "retrieval should have supplied at least one source"
    assert body["sources"][0]["document_id"] == document["id"]
    assert body["sources"][0]["filename"] == "handbook.txt"


async def test_rag_can_be_disabled_per_request(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    await _upload(client, auth)
    response = await client.post(
        "/api/v1/chat",
        json={"message": "How many days of annual leave?", "use_rag": False},
        headers=auth,
    )
    assert response.json()["sources"] == []


async def test_unsupported_file_type_is_rejected(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    response = await client.post(
        "/api/v1/documents",
        files={"file": ("payload.exe", b"MZ\x00\x00", "application/x-msdownload")},
        headers=auth,
    )
    assert response.status_code == 422


async def test_a_document_with_no_text_reports_why(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    # Upload succeeds (201) but the row records the failure, so the client can
    # tell the user their scan needs OCR.
    response = await client.post(
        "/api/v1/documents",
        files={"file": ("blank.txt", b"    \n   ", "text/plain")},
        headers=auth,
    )
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "failed"
    assert "OCR" in (body["error"] or "")


async def test_another_users_document_is_not_found(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    document = await _upload(client, auth)

    other = await client.post(
        "/api/v1/auth/register",
        json={"email": "nosy@example.com", "password": "correct-horse-battery"},
    )
    stranger = {"Authorization": f"Bearer {other.json()['access_token']}"}

    assert (
        await client.get(f"/api/v1/documents/{document['id']}", headers=stranger)
    ).status_code == 404

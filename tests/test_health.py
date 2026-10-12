from __future__ import annotations

from httpx import AsyncClient


async def test_health_is_dependency_free(client: AsyncClient) -> None:
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_ready_reports_capabilities(client: AsyncClient) -> None:
    response = await client.get("/api/v1/ready")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ready"
    assert body["checks"]["database"]["status"] == "ok"

    capabilities = body["checks"]["capabilities"]
    assert capabilities["llm"]["provider"] == "echo"
    # The report must distinguish semantic retrieval from the lexical fallback,
    # or a deployment silently ships with degraded search.
    assert "semantic" in capabilities["rag"]


async def test_request_id_header_is_returned(client: AsyncClient) -> None:
    response = await client.get("/api/v1/health")
    assert response.headers["X-Request-ID"]
    assert int(response.headers["X-Response-Time-Ms"]) >= 0


async def test_upstream_request_id_is_preserved(client: AsyncClient) -> None:
    response = await client.get("/api/v1/health", headers={"X-Request-ID": "trace-abc-123"})
    assert response.headers["X-Request-ID"] == "trace-abc-123"

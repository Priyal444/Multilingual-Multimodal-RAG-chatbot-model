"""Vector storage and similarity search — two interchangeable backends.

``PgVectorStore`` pushes the search into PostgreSQL using pgvector's cosine
distance operator and an IVFFlat index, so the database returns the top *k* rows
and nothing else crosses the wire. That is the deployment target.

``NumpyVectorStore`` reads candidate vectors and ranks them in-process. It needs
no extension and runs on SQLite, which keeps `git clone && pytest` working — but
it is O(n) in stored chunks per query and is capped at ``MAX_CANDIDATES`` rows to
keep a runaway corpus from exhausting memory. Past a few tens of thousands of
chunks, move to PostgreSQL.

Both write the same portable ``embedding`` BLOB, so switching backends is a
config change plus a re-index, not a migration of the source data.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

import numpy as np
from sqlalchemy import bindparam, delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.db.models import DocumentChunk
from app.services.rag.chunking import Chunk
from app.services.rag.embeddings import pack_vector, unpack_matrix

log = get_logger(__name__)

# Ceiling on rows the in-process backend will rank for a single query.
MAX_CANDIDATES = 20_000


@dataclass(slots=True)
class ScoredChunk:
    chunk_id: str
    document_id: str
    ordinal: int
    content: str
    score: float


@runtime_checkable
class VectorStore(Protocol):
    name: str

    async def upsert(
        self,
        session: AsyncSession,
        *,
        document_id: str,
        user_id: str,
        chunks: list[Chunk],
        vectors: np.ndarray,
    ) -> int: ...

    async def search(
        self,
        session: AsyncSession,
        *,
        user_id: str,
        query_vector: np.ndarray,
        top_k: int,
        document_ids: list[str] | None = None,
        min_score: float = 0.0,
    ) -> list[ScoredChunk]: ...

    async def delete_document(self, session: AsyncSession, document_id: str) -> None: ...


class NumpyVectorStore:
    """Portable backend: vectors as BLOBs, ranking with numpy."""

    name = "numpy"

    async def upsert(
        self,
        session: AsyncSession,
        *,
        document_id: str,
        user_id: str,
        chunks: list[Chunk],
        vectors: np.ndarray,
    ) -> int:
        await self.delete_document(session, document_id)
        rows = [
            DocumentChunk(
                document_id=document_id,
                user_id=user_id,
                ordinal=chunk.ordinal,
                content=chunk.content,
                char_count=chunk.char_count,
                embedding=pack_vector(vectors[i]),
                embedding_norm=float(np.linalg.norm(vectors[i])),
            )
            for i, chunk in enumerate(chunks)
        ]
        session.add_all(rows)
        await session.flush()
        return len(rows)

    async def search(
        self,
        session: AsyncSession,
        *,
        user_id: str,
        query_vector: np.ndarray,
        top_k: int,
        document_ids: list[str] | None = None,
        min_score: float = 0.0,
    ) -> list[ScoredChunk]:
        stmt = (
            select(
                DocumentChunk.id,
                DocumentChunk.document_id,
                DocumentChunk.ordinal,
                DocumentChunk.content,
                DocumentChunk.embedding,
            )
            .where(
                DocumentChunk.user_id == user_id,
                DocumentChunk.embedding.is_not(None),
            )
            .limit(MAX_CANDIDATES)
        )
        if document_ids:
            stmt = stmt.where(DocumentChunk.document_id.in_(document_ids))

        rows = (await session.execute(stmt)).all()
        if not rows:
            return []

        dimension = settings.embedding_dim
        try:
            matrix = unpack_matrix([r.embedding for r in rows], dimension)
        except ValueError:
            # A dimension change without a re-index leaves unreadable rows.
            log.error("embedding_dimension_mismatch", expected=dimension, rows=len(rows))
            return []

        query = np.asarray(query_vector, dtype=np.float32)
        # Vectors are stored L2-normalised, so a dot product *is* cosine
        # similarity — but re-normalising costs nothing and protects against a
        # backend that forgot to.
        matrix_norms = np.maximum(np.linalg.norm(matrix, axis=1), 1e-12)
        query_norm = max(float(np.linalg.norm(query)), 1e-12)
        scores = (matrix @ query) / (matrix_norms * query_norm)

        # argpartition is O(n) vs. a full O(n log n) sort; only the top-k slice
        # is then sorted.
        k = min(top_k, len(scores))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        return [
            ScoredChunk(
                chunk_id=rows[i].id,
                document_id=rows[i].document_id,
                ordinal=rows[i].ordinal,
                content=rows[i].content,
                score=float(scores[i]),
            )
            for i in top_idx
            if float(scores[i]) >= min_score
        ]

    async def delete_document(self, session: AsyncSession, document_id: str) -> None:
        await session.execute(
            delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
        )


class PgVectorStore:
    """PostgreSQL backend: the database does the ranking."""

    name = "pgvector"

    @staticmethod
    def _literal(vector: np.ndarray) -> str:
        """pgvector's text input format: ``[0.1,0.2,...]``."""
        return "[" + ",".join(f"{float(x):.7g}" for x in np.asarray(vector).ravel()) + "]"

    async def upsert(
        self,
        session: AsyncSession,
        *,
        document_id: str,
        user_id: str,
        chunks: list[Chunk],
        vectors: np.ndarray,
    ) -> int:
        await self.delete_document(session, document_id)

        stmt = text(
            """
            INSERT INTO document_chunks
                (id, document_id, user_id, ordinal, content, char_count,
                 embedding, embedding_norm, embedding_vec, created_at)
            VALUES
                (:id, :document_id, :user_id, :ordinal, :content, :char_count,
                 :embedding, :embedding_norm, CAST(:embedding_vec AS vector), :created_at)
            """
        )
        now = datetime.now(UTC)
        params = [
            {
                "id": uuid.uuid4().hex,
                "document_id": document_id,
                "user_id": user_id,
                "ordinal": chunk.ordinal,
                "content": chunk.content,
                "char_count": chunk.char_count,
                # Both columns are written: the BLOB keeps the data portable if
                # the deployment ever moves back off PostgreSQL.
                "embedding": pack_vector(vectors[i]),
                "embedding_norm": float(np.linalg.norm(vectors[i])),
                "embedding_vec": self._literal(vectors[i]),
                "created_at": now,
            }
            for i, chunk in enumerate(chunks)
        ]
        if params:
            await session.execute(stmt, params)
        return len(params)

    async def search(
        self,
        session: AsyncSession,
        *,
        user_id: str,
        query_vector: np.ndarray,
        top_k: int,
        document_ids: list[str] | None = None,
        min_score: float = 0.0,
    ) -> list[ScoredChunk]:
        filter_sql = "AND document_id IN :document_ids" if document_ids else ""
        stmt = text(
            f"""
            SELECT id, document_id, ordinal, content,
                   1 - (embedding_vec <=> CAST(:query AS vector)) AS score
            FROM document_chunks
            WHERE user_id = :user_id
              AND embedding_vec IS NOT NULL
              {filter_sql}
            ORDER BY embedding_vec <=> CAST(:query AS vector)
            LIMIT :top_k
            """
        )
        params: dict[str, object] = {
            "user_id": user_id,
            "query": self._literal(query_vector),
            "top_k": top_k,
        }
        if document_ids:
            stmt = stmt.bindparams(bindparam("document_ids", expanding=True))
            params["document_ids"] = document_ids

        rows = (await session.execute(stmt, params)).all()
        return [
            ScoredChunk(
                chunk_id=row.id,
                document_id=row.document_id,
                ordinal=row.ordinal,
                content=row.content,
                score=float(row.score),
            )
            for row in rows
            if float(row.score) >= min_score
        ]

    async def delete_document(self, session: AsyncSession, document_id: str) -> None:
        await session.execute(
            delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
        )


_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Pick the backend that matches the configured database."""
    global _store
    if _store is not None:
        return _store

    if settings.is_postgres:
        _store = PgVectorStore()
    else:
        _store = NumpyVectorStore()
    log.info("vector_store_selected", backend=_store.name)
    return _store


def reset_vector_store() -> None:
    """Test hook."""
    global _store
    _store = None

"""End-to-end retrieval-augmented generation: ingest, retrieve, format.

Ingestion runs inline rather than on a task queue. That is a deliberate, bounded
choice: a 20 MB document ceiling puts extraction plus embedding in the low
seconds, and inline processing means the upload response can report the real
status instead of "pending, check back". Above that ceiling — or once ingestion
starts blocking a worker for tens of seconds — this is the first thing that
should move behind Celery or ARQ, and the ``DocumentStatus`` states already
model that transition.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.db.models import Document, DocumentStatus
from app.services.language import detect_language
from app.services.rag.chunking import chunk_text
from app.services.rag.embeddings import get_embedder
from app.services.rag.extract import extract_document
from app.services.rag.store import get_vector_store

log = get_logger(__name__)

# Hard ceiling on retrieved context so RAG cannot crowd out the conversation.
MAX_CONTEXT_CHARS = 8000


@dataclass(slots=True)
class RetrievedChunk:
    document_id: str
    filename: str
    ordinal: int
    content: str
    score: float


class RAGPipeline:
    async def ingest(
        self,
        session: AsyncSession,
        *,
        document: Document,
        raw: bytes,
    ) -> int:
        """Extract, chunk, embed and store a document. Returns the chunk count."""
        started = time.perf_counter()
        document.status = DocumentStatus.PROCESSING
        await session.flush()

        try:
            extracted = extract_document(raw, document.media_type, filename=document.filename)
            # Sample the head: language is uniform in practice and embedding the
            # whole document just to guess a label is wasteful.
            document.language = detect_language(extracted.text[:2000]).code

            chunks = chunk_text(
                extracted.text,
                max_chars=settings.rag_chunk_chars,
                overlap=settings.rag_chunk_overlap,
            )
            if not chunks:
                raise ValueError("Document produced no chunks.")

            embedder = get_embedder()
            vectors = await embedder.embed_documents([c.content for c in chunks])

            store = get_vector_store()
            count = await store.upsert(
                session,
                document_id=document.id,
                user_id=document.user_id,
                chunks=chunks,
                vectors=vectors,
            )

            document.chunk_count = count
            document.status = DocumentStatus.READY
            document.error = None
            await session.flush()

            log.info(
                "document_ingested",
                document_id=document.id,
                chunks=count,
                chars=extracted.char_count,
                pages=extracted.page_count,
                language=document.language,
                embedder=embedder.name,
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )
            return count

        except Exception as exc:
            document.status = DocumentStatus.FAILED
            # Truncated: the column is for operators, not for stack traces.
            document.error = str(exc)[:500]
            await session.flush()
            log.warning("document_ingest_failed", document_id=document.id, error=str(exc))
            raise

    async def retrieve(
        self,
        session: AsyncSession,
        *,
        user_id: str,
        query: str,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Find the passages most relevant to ``query`` among a user's documents."""
        if not settings.rag_enabled or not query.strip():
            return []

        embedder = get_embedder()
        query_vector = await embedder.embed_query(query)

        store = get_vector_store()
        scored = await store.search(
            session,
            user_id=user_id,
            query_vector=query_vector,
            top_k=top_k or settings.rag_top_k,
            document_ids=document_ids,
            min_score=settings.rag_min_score,
        )
        if not scored:
            return []

        # One extra query resolves filenames, so citations are human-readable
        # instead of opaque IDs.
        doc_ids = {c.document_id for c in scored}
        rows = (
            await session.execute(
                select(Document.id, Document.filename).where(Document.id.in_(doc_ids))
            )
        ).all()
        names = {row.id: row.filename for row in rows}

        return [
            RetrievedChunk(
                document_id=c.document_id,
                filename=names.get(c.document_id, "document"),
                ordinal=c.ordinal,
                content=c.content,
                score=round(c.score, 4),
            )
            for c in scored
        ]

    @staticmethod
    def format_context(chunks: list[RetrievedChunk]) -> str:
        """Render retrieved passages for the prompt, with citable source labels."""
        if not chunks:
            return ""

        blocks: list[str] = []
        budget = MAX_CONTEXT_CHARS
        for index, chunk in enumerate(chunks, start=1):
            body = chunk.content
            if len(body) > budget:
                body = body[: max(0, budget)].rsplit(" ", 1)[0]
            if not body.strip():
                break
            blocks.append(
                f"[{index}] source: {chunk.filename} (part {chunk.ordinal + 1}, "
                f"relevance {chunk.score:.2f})\n{body}"
            )
            budget -= len(body)
            if budget <= 0:
                break

        return "\n\n".join(blocks)


rag_pipeline = RAGPipeline()

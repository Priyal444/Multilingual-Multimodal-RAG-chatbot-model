"""Document upload and retrieval (the RAG knowledge base)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile, status
from sqlalchemy import func, select

from app.api.deps import CurrentUser, RateLimited, SessionDep, get_owned_document
from app.api.v1.uploads import read_limited
from app.core.config import settings
from app.core.errors import ValidationError
from app.core.logging import get_logger
from app.db.models import Document, DocumentStatus
from app.schemas.documents import (
    DocumentListResponse,
    DocumentResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from app.services import storage
from app.services.rag import rag_pipeline
from app.services.rag.embeddings import get_embedder
from app.services.rag.extract import SUPPORTED_DOCUMENT_TYPES

log = get_logger(__name__)
router = APIRouter()

OwnedDocument = Annotated[Document, Depends(get_owned_document)]


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document and index it for retrieval",
)
async def upload_document(
    user: CurrentUser,
    session: SessionDep,
    _: RateLimited,
    file: Annotated[UploadFile, File()],
) -> Document:
    """Ingest runs inline, so the response reflects the true outcome.

    See ``app/services/rag/pipeline.py`` for when this should move to a queue.
    """
    if not file.filename:
        raise ValidationError("No file was provided.")

    media_type = file.content_type or "application/octet-stream"
    suffix = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if media_type not in SUPPORTED_DOCUMENT_TYPES and suffix not in {
        "pdf", "docx", "txt", "md", "csv",
    }:
        raise ValidationError(
            f"Unsupported document type '{media_type}'.",
            supported=["pdf", "docx", "txt", "md", "csv"],
        )

    raw = await read_limited(file, settings.max_document_bytes, label="document")
    path = await storage.save(raw, user_id=user.id, media_type=media_type, category="documents")

    document = Document(
        user_id=user.id,
        filename=file.filename,
        media_type=media_type,
        size_bytes=len(raw),
        storage_path=path,
        status=DocumentStatus.PENDING,
    )
    session.add(document)
    await session.flush()

    try:
        await rag_pipeline.ingest(session, document=document, raw=raw)
    except Exception:
        # `ingest` has already recorded status=failed and the reason on the row,
        # so return 201 with that state rather than a 500 — the upload itself
        # succeeded, and the client needs the error message to act on it.
        log.info("document_upload_indexed_with_error", document_id=document.id)

    return document


@router.get("", response_model=DocumentListResponse, summary="List uploaded documents")
async def list_documents(user: CurrentUser, session: SessionDep) -> DocumentListResponse:
    total = (
        await session.execute(
            select(func.count()).select_from(Document).where(Document.user_id == user.id)
        )
    ).scalar_one()
    rows = (
        await session.execute(
            select(Document)
            .where(Document.user_id == user.id)
            .order_by(Document.created_at.desc())
        )
    ).scalars().all()
    return DocumentListResponse(
        items=[DocumentResponse.model_validate(row) for row in rows], total=total
    )


@router.get("/{document_id}", response_model=DocumentResponse, summary="Read one document")
async def get_document(document: OwnedDocument) -> Document:
    return document


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document and its indexed chunks",
)
async def delete_document(document: OwnedDocument, session: SessionDep) -> None:
    path = document.storage_path
    await session.delete(document)  # chunks cascade
    await session.flush()
    # Delete the blob only after the row is gone: an orphaned file is cheap, a
    # row pointing at a missing file breaks every subsequent read.
    await storage.delete(path)


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Search indexed documents without invoking the model",
)
async def search(
    payload: SearchRequest, user: CurrentUser, session: SessionDep, _: RateLimited
) -> SearchResponse:
    """Retrieval on its own — useful for debugging RAG quality in isolation.

    When an answer is wrong, this tells you whether retrieval or generation is at
    fault, which is otherwise the hardest thing to determine in a RAG system.
    """
    chunks = await rag_pipeline.retrieve(
        session,
        user_id=user.id,
        query=payload.query,
        top_k=payload.top_k,
        document_ids=payload.document_ids,
    )
    embedder = get_embedder()
    return SearchResponse(
        query=payload.query,
        hits=[
            SearchHit(
                document_id=c.document_id,
                filename=c.filename,
                part=c.ordinal + 1,
                score=c.score,
                content=c.content,
            )
            for c in chunks
        ],
        embedder=embedder.name,
        semantic=embedder.is_semantic,
    )

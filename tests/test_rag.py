"""Chunking, embedding, and vector search."""

from __future__ import annotations

import numpy as np
import pytest

from app.services.rag.chunking import chunk_text
from app.services.rag.embeddings import (
    HashingEmbedder,
    l2_normalize,
    pack_vector,
    unpack_matrix,
    unpack_vector,
)
from app.services.rag.extract import extract_document
from app.services.rag.pipeline import RAGPipeline, RetrievedChunk

# --- chunking -------------------------------------------------------------


def test_chunks_respect_the_size_budget() -> None:
    text = " ".join(f"Sentence number {i} about retrieval." for i in range(400))
    chunks = chunk_text(text, max_chars=500, overlap=80)

    assert len(chunks) > 1
    assert all(c.char_count <= 500 for c in chunks)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_chunks_overlap_so_answers_spanning_a_boundary_survive() -> None:
    text = ". ".join(f"Fact {i} is important" for i in range(120)) + "."
    chunks = chunk_text(text, max_chars=400, overlap=120)

    assert len(chunks) >= 2
    # Some tail of chunk n must reappear at the head of chunk n+1.
    tail_words = set(chunks[0].content.split()[-8:])
    head_words = set(chunks[1].content.split()[:20])
    assert tail_words & head_words


def test_short_text_is_a_single_chunk() -> None:
    chunks = chunk_text("Just one short line.", max_chars=1000, overlap=100)
    assert len(chunks) == 1
    assert chunks[0].content == "Just one short line."


def test_empty_text_yields_no_chunks() -> None:
    assert chunk_text("   \n\n  ") == []


def test_cjk_sentences_split_on_their_own_punctuation() -> None:
    # A Latin-only splitter sees no separators here and degrades to hard cuts.
    text = "".join(f"这是第{i}个句子。" for i in range(200))
    chunks = chunk_text(text, max_chars=300, overlap=50)
    assert len(chunks) > 1
    assert all(c.char_count <= 300 for c in chunks)


def test_overlap_must_be_smaller_than_the_window() -> None:
    with pytest.raises(ValueError):
        chunk_text("some text", max_chars=100, overlap=100)


def test_a_single_token_longer_than_the_window_is_hard_cut() -> None:
    chunks = chunk_text("x" * 5000, max_chars=500, overlap=50)
    assert len(chunks) > 1
    assert all(c.char_count <= 500 for c in chunks)


# --- embeddings -----------------------------------------------------------


async def test_hashing_embedder_is_deterministic_and_normalised() -> None:
    embedder = HashingEmbedder(dimension=384)
    first = await embedder.embed_query("invoice total amount due")
    second = await embedder.embed_query("invoice total amount due")

    assert np.array_equal(first, second)
    assert first.shape == (384,)
    assert float(np.linalg.norm(first)) == pytest.approx(1.0, abs=1e-5)


async def test_hashing_embedder_ranks_related_text_higher() -> None:
    embedder = HashingEmbedder(dimension=384)
    query = await embedder.embed_query("annual revenue growth")
    related = await embedder.embed_query("the annual revenue grew year over year")
    unrelated = await embedder.embed_query("photosynthesis in tropical plants")

    assert float(query @ related) > float(query @ unrelated)


async def test_hashing_embedder_handles_scripts_without_spaces() -> None:
    # Word tokenisation is meaningless for Chinese; character n-grams are why
    # this fallback still produces a usable vector.
    embedder = HashingEmbedder(dimension=384)
    vector = await embedder.embed_query("这是一个关于收入增长的句子")
    assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-5)


def test_vector_packing_round_trip() -> None:
    original = np.arange(8, dtype=np.float32) / 3.0
    restored = unpack_vector(pack_vector(original))
    assert np.allclose(original, restored, atol=1e-6)


def test_matrix_unpacking_preserves_row_order() -> None:
    rows = [np.full(4, i, dtype=np.float32) for i in range(5)]
    matrix = unpack_matrix([pack_vector(r) for r in rows], 4)
    assert matrix.shape == (5, 4)
    assert np.allclose(matrix[3], 3.0)


def test_l2_normalize_makes_dot_product_equal_cosine() -> None:
    matrix = l2_normalize(np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32))
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)


# --- extraction -----------------------------------------------------------


def test_plain_text_extraction_normalises_whitespace() -> None:
    raw = "Title\r\n\r\n\r\n\r\nBody    with     spaces\r\n".encode()
    result = extract_document(raw, "text/plain", filename="notes.txt")

    assert "Body with spaces" in result.text
    assert "\n\n\n" not in result.text


def test_extraction_rejects_an_unsupported_type() -> None:
    from app.core.errors import ValidationError

    with pytest.raises(ValidationError):
        extract_document(b"\x00\x01", "application/x-executable", filename="a.bin")


def test_extraction_rejects_an_empty_document() -> None:
    from app.core.errors import ValidationError

    with pytest.raises(ValidationError, match="OCR"):
        extract_document(b"   \n  ", "text/plain", filename="blank.txt")


# --- context formatting ---------------------------------------------------


def test_retrieved_context_is_numbered_and_names_its_source() -> None:
    chunks = [
        RetrievedChunk("doc-1", "handbook.pdf", 0, "Refunds take 5 days.", 0.91),
        RetrievedChunk("doc-2", "policy.docx", 3, "Returns need a receipt.", 0.72),
    ]
    context = RAGPipeline.format_context(chunks)

    assert "[1]" in context and "[2]" in context
    assert "handbook.pdf" in context and "policy.docx" in context
    # Part numbers are 1-based in the prompt; ordinals are 0-based internally.
    assert "part 4" in context


def test_context_formatting_is_bounded() -> None:
    huge = [RetrievedChunk("d", "big.pdf", i, "x" * 5000, 0.5) for i in range(10)]
    context = RAGPipeline.format_context(huge)
    assert len(context) < 10_000


def test_no_chunks_produces_no_context() -> None:
    assert RAGPipeline.format_context([]) == ""

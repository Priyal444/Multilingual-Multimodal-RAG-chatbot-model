"""Embedding backends.

Claude has no embeddings endpoint, so retrieval needs its own model. The default
is a **multilingual** sentence-transformer, which matters more than it sounds:
with a monolingual encoder, a Hindi question cannot retrieve the English
paragraph that answers it. A shared multilingual vector space makes
cross-language retrieval work, which is the whole point of this project.

If ``sentence-transformers`` is not installed the app does not break — it falls
back to a hashed character-n-gram embedder. That fallback is *lexical*, not
semantic: it matches shared substrings, so it finds "invoice total" in a document
that says "invoice total" but not one that says "amount due". It exists so the
test suite and a fresh clone work without a 2 GB torch download, and the API
reports which backend is live so nobody mistakes one for the other.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import struct
from typing import Any, Protocol, runtime_checkable

import numpy as np

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)

_TOKEN = re.compile(r"\w+", re.UNICODE)


@runtime_checkable
class Embedder(Protocol):
    name: str
    dimension: int
    is_semantic: bool

    async def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    async def embed_query(self, text: str) -> np.ndarray: ...


def pack_vector(vector: np.ndarray) -> bytes:
    """Serialise to little-endian float32 for the ``embedding`` BLOB column."""
    return np.asarray(vector, dtype="<f4").tobytes()


def unpack_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype="<f4")


def unpack_matrix(blobs: list[bytes], dimension: int) -> np.ndarray:
    if not blobs:
        return np.zeros((0, dimension), dtype=np.float32)
    return np.frombuffer(b"".join(blobs), dtype="<f4").reshape(len(blobs), dimension)


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-normalise so cosine similarity reduces to a dot product."""
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


class SentenceTransformerEmbedder:
    """Multilingual dense embeddings. Loaded lazily, run off the event loop."""

    is_semantic = True

    def __init__(self, model_name: str, dimension: int) -> None:
        self.name = model_name
        self.dimension = dimension
        self._model: Any = None
        self._lock = asyncio.Lock()

    async def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        async with self._lock:
            if self._model is not None:
                return self._model
            from sentence_transformers import SentenceTransformer

            log.info("loading_embedding_model", model=self.name)
            self._model = await asyncio.to_thread(SentenceTransformer, self.name)
            actual = int(self._model.get_sentence_embedding_dimension())
            if actual != self.dimension:
                # A mismatch corrupts every stored vector; fail loudly at load
                # time rather than writing unreadable rows.
                raise ValueError(
                    f"EMBEDDING_DIM is {self.dimension} but '{self.name}' produces "
                    f"{actual}-dimensional vectors. Update EMBEDDING_DIM and "
                    f"re-index existing documents."
                )
            log.info("embedding_model_ready", model=self.name, dimension=actual)
            return self._model

    async def _encode(self, texts: list[str], prefix: str = "") -> np.ndarray:
        model = await self._get_model()
        payload = [f"{prefix}{t}" for t in texts] if prefix else texts
        vectors = await asyncio.to_thread(
            model.encode,
            payload,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    async def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return await self._encode(texts)

    async def embed_query(self, text: str) -> np.ndarray:
        vectors = await self._encode([text])
        return vectors[0]


class HashingEmbedder:
    """Deterministic lexical fallback — character n-grams hashed into buckets.

    Character n-grams rather than words, because word tokenisation is meaningless
    for Chinese, Japanese, and Thai, which have no spaces.
    """

    is_semantic = False
    name = "hashing-char-ngram"

    def __init__(self, dimension: int, ngram_range: tuple[int, int] = (3, 5)) -> None:
        self.dimension = dimension
        self._min_n, self._max_n = ngram_range

    def _vectorize(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        normalized = " ".join(_TOKEN.findall((text or "").lower()))
        if not normalized:
            return vector
        padded = f" {normalized} "
        for n in range(self._min_n, self._max_n + 1):
            for i in range(len(padded) - n + 1):
                gram = padded[i : i + n]
                digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
                bucket = struct.unpack("<Q", digest)[0] % self.dimension
                # Signed hashing: the sign bit halves systematic collision bias.
                vector[bucket] += 1.0 if digest[0] & 1 else -1.0
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-12 else vector

    async def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.vstack([self._vectorize(t) for t in texts])

    async def embed_query(self, text: str) -> np.ndarray:
        return self._vectorize(text)


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    """Return the process-wide embedder, choosing the best available backend."""
    global _embedder
    if _embedder is not None:
        return _embedder

    try:
        import sentence_transformers  # noqa: F401

        _embedder = SentenceTransformerEmbedder(
            settings.embedding_model, settings.embedding_dim
        )
        log.info("embedder_selected", backend="sentence-transformers")
    except ImportError:
        _embedder = HashingEmbedder(settings.embedding_dim)
        log.warning(
            "embedder_fallback_lexical",
            reason="sentence-transformers not installed",
            hint="pip install -r requirements-ml.txt for semantic retrieval",
        )
    return _embedder


def reset_embedder() -> None:
    """Test hook."""
    global _embedder
    _embedder = None

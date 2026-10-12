"""Recursive, boundary-aware text chunking.

Naive fixed-width slicing cuts sentences in half, and a chunk that starts
mid-clause embeds badly — its vector describes a fragment, not an idea. So the
splitter walks a hierarchy of separators (paragraph → line → sentence → clause →
word) and only falls back to a hard character cut when a single "word" genuinely
exceeds the window.

Overlap exists because the answer to a question is frequently split across a
boundary. Repeating the tail of each chunk at the head of the next costs storage
and buys recall.

The sentence separators include CJK (``。``), Devanagari (``।``), and Arabic
(``؟``) punctuation — a Latin-only splitter degrades to word-level chunking on
half the languages this app targets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Ordered widest-to-narrowest. Each level is tried before the next.
_SEPARATORS: list[str] = [
    "\n\n",   # paragraph
    "\n",     # line
    "。",      # CJK full stop
    "। ",     # Devanagari danda
    "؟ ",     # Arabic question mark
    ". ",
    "! ",
    "? ",
    "; ",
    ", ",
    " ",
]

_SENTENCE_END = re.compile(r"(?<=[.!?。।؟])\s+")


@dataclass(slots=True)
class Chunk:
    ordinal: int
    content: str
    char_count: int


def _split_on(text: str, separator: str) -> list[str]:
    if separator == " ":
        return text.split(" ")
    parts = text.split(separator)
    # Re-attach the separator so sentence-ending punctuation survives.
    return [p + separator if i < len(parts) - 1 else p for i, p in enumerate(parts)]


def _recursive_split(text: str, max_chars: int, depth: int = 0) -> list[str]:
    if len(text) <= max_chars:
        return [text] if text.strip() else []
    if depth >= len(_SEPARATORS):
        # No separator left: hard-cut. Only reachable for pathological input
        # such as a single 5,000-character token.
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]

    separator = _SEPARATORS[depth]
    pieces = _split_on(text, separator)
    if len(pieces) == 1:
        return _recursive_split(text, max_chars, depth + 1)

    chunks: list[str] = []
    buffer = ""
    for piece in pieces:
        candidate = f"{buffer}{piece}" if separator == "\n\n" else f"{buffer}{piece}"
        if len(candidate) <= max_chars:
            buffer = candidate
            continue
        if buffer.strip():
            chunks.append(buffer)
        # The piece itself may still be too big — recurse at the next level.
        if len(piece) > max_chars:
            chunks.extend(_recursive_split(piece, max_chars, depth + 1))
            buffer = ""
        else:
            buffer = piece
    if buffer.strip():
        chunks.append(buffer)
    return chunks


def _tail_overlap(text: str, overlap: int) -> str:
    """Take the last ``overlap`` characters, snapped to a sentence boundary."""
    if overlap <= 0 or len(text) <= overlap:
        return text if overlap > 0 else ""
    tail = text[-overlap:]
    sentences = _SENTENCE_END.split(tail)
    if len(sentences) > 1:
        # Drop the leading partial sentence.
        return " ".join(sentences[1:]).strip()
    return tail.strip()


def chunk_text(text: str, *, max_chars: int = 1200, overlap: int = 200) -> list[Chunk]:
    """Split ``text`` into overlapping, boundary-aligned chunks."""
    text = (text or "").strip()
    if not text:
        return []
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")

    # Reserve room for the overlap prefix so chunks stay within max_chars.
    body_budget = max_chars - overlap
    raw_chunks = _recursive_split(text, body_budget)

    chunks: list[Chunk] = []
    carry = ""
    for piece in raw_chunks:
        content = (carry + piece).strip() if carry else piece.strip()
        if not content:
            continue
        chunks.append(Chunk(ordinal=len(chunks), content=content, char_count=len(content)))
        # The joining space is counted inside the overlap budget, or a chunk
        # would come out one character over max_chars.
        tail = _tail_overlap(piece, overlap)
        carry = (tail + " ")[:overlap] if tail else ""

    return chunks

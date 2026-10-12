"""Plain-text extraction from uploaded documents.

Deliberately conservative: extract text, preserve paragraph and page structure
(the chunker uses those boundaries), and refuse anything we cannot read rather
than silently indexing garbage. A document that produces no text is almost
always a scanned image — that gets an explicit error naming OCR, because
"upload succeeded, retrieval returns nothing" is a miserable failure mode.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

from app.core.errors import ValidationError
from app.core.logging import get_logger

log = get_logger(__name__)

SUPPORTED_DOCUMENT_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "text/plain": "text",
    "text/markdown": "text",
    "text/csv": "text",
    "application/json": "text",
}

_WHITESPACE = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES = re.compile(r"\n{3,}")


@dataclass(slots=True)
class ExtractedDocument:
    text: str
    page_count: int
    char_count: int


def _normalise(text: str) -> str:
    """Collapse runs of spaces, keep paragraph breaks — chunking depends on them."""
    text = text.replace(" ", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANK_LINES.sub("\n\n", text).strip()


def _extract_pdf(raw: bytes) -> ExtractedDocument:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - declared dependency
        raise ValidationError("pypdf is required to read PDF files.") from exc

    try:
        reader = PdfReader(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001
        raise ValidationError("That PDF could not be opened — it may be corrupt.") from exc

    if getattr(reader, "is_encrypted", False):
        try:
            # Many PDFs are "encrypted" with an empty owner password.
            reader.decrypt("")
        except Exception as exc:  # noqa: BLE001
            raise ValidationError("That PDF is password-protected.") from exc

    pages: list[str] = []
    for index, page in enumerate(reader.pages):
        try:
            pages.append(page.extract_text() or "")
        except Exception as exc:  # noqa: BLE001 - one bad page shouldn't kill the doc
            log.warning("pdf_page_extract_failed", page=index, error=str(exc))
            pages.append("")

    text = _normalise("\n\n".join(pages))
    return ExtractedDocument(text=text, page_count=len(reader.pages), char_count=len(text))


def _extract_docx(raw: bytes) -> ExtractedDocument:
    try:
        import docx
    except ImportError as exc:  # pragma: no cover - declared dependency
        raise ValidationError("python-docx is required to read DOCX files.") from exc

    try:
        document = docx.Document(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001
        raise ValidationError("That DOCX file could not be opened.") from exc

    blocks = [p.text for p in document.paragraphs if p.text.strip()]
    # Tables carry a lot of the meaning in real business documents; a row
    # flattened to " | "-separated cells retains enough for retrieval.
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                blocks.append(" | ".join(cells))

    text = _normalise("\n\n".join(blocks))
    return ExtractedDocument(text=text, page_count=1, char_count=len(text))


def _extract_text(raw: bytes) -> ExtractedDocument:
    for encoding in ("utf-8", "utf-8-sig", "utf-16", "latin-1"):
        try:
            decoded = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover - latin-1 decodes any byte sequence
        raise ValidationError("That text file uses an unsupported encoding.")

    text = _normalise(decoded)
    return ExtractedDocument(text=text, page_count=1, char_count=len(text))


def extract_document(raw: bytes, media_type: str, *, filename: str = "") -> ExtractedDocument:
    kind = SUPPORTED_DOCUMENT_TYPES.get(media_type)
    if kind is None:
        # Browsers send application/octet-stream more often than they should;
        # fall back to the extension before rejecting.
        suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        kind = {"pdf": "pdf", "docx": "docx", "txt": "text", "md": "text", "csv": "text"}.get(
            suffix
        )
    if kind is None:
        raise ValidationError(
            f"Unsupported document type '{media_type}'.",
            supported=sorted(SUPPORTED_DOCUMENT_TYPES),
        )

    extracted = {"pdf": _extract_pdf, "docx": _extract_docx, "text": _extract_text}[kind](raw)

    if not extracted.text.strip():
        raise ValidationError(
            "No text could be extracted. If this is a scanned document, run it "
            "through OCR before uploading.",
            filename=filename,
        )
    return extracted

"""Local filesystem storage for uploads.

Files are addressed by ``<user_id>/<uuid><ext>`` — never by the client-supplied
filename. A filename from an upload is attacker-controlled: ``../../.env`` and
``C:\\Windows\\System32\\x`` are both valid strings, and joining one onto a base
path is how a file write escapes its directory. The original name is kept in the
database for display only.

Swapping this for S3 means reimplementing ``save`` / ``read`` / ``delete``; no
caller touches a path directly.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import anyio

from app.core.config import settings
from app.core.errors import NotFoundError
from app.core.logging import get_logger

log = get_logger(__name__)

_EXTENSION_BY_TYPE = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "text/plain": ".txt",
    "text/markdown": ".md",
    "text/csv": ".csv",
    "application/json": ".json",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/webm": ".webm",
    "audio/ogg": ".ogg",
    "audio/mp4": ".m4a",
}


def _root() -> Path:
    root = Path(settings.storage_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


async def save(data: bytes, *, user_id: str, media_type: str, category: str) -> str:
    """Write bytes to storage and return the relative path recorded in the DB."""
    extension = _EXTENSION_BY_TYPE.get(media_type, ".bin")
    # user_id and category are server-generated, so this join is safe.
    directory = _root() / category / user_id
    directory.mkdir(parents=True, exist_ok=True)

    relative = f"{category}/{user_id}/{uuid.uuid4().hex}{extension}"
    await anyio.Path(_root() / relative).write_bytes(data)
    log.debug("file_stored", path=relative, bytes=len(data))
    return relative


async def read(relative_path: str) -> bytes:
    root = _root()
    target = (root / relative_path).resolve()
    # Defence in depth: even though paths are server-generated, verify the
    # resolved target is still inside the storage root before reading.
    if not target.is_relative_to(root):
        raise NotFoundError("File not found.")
    if not target.exists():
        raise NotFoundError("File not found.")
    return await anyio.Path(target).read_bytes()


async def delete(relative_path: str) -> None:
    root = _root()
    target = (root / relative_path).resolve()
    if not target.is_relative_to(root):
        return
    try:
        await anyio.Path(target).unlink(missing_ok=True)
    except OSError as exc:  # noqa: BLE001 - deletion is best-effort
        log.warning("file_delete_failed", path=relative_path, error=str(exc))

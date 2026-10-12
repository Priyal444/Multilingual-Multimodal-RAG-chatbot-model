"""Bounded reading of multipart uploads.

``UploadFile.read()`` with no argument reads the entire body into memory. A
client that sends a 2 GB "image" would then take the process down before any
size check ran, so reads are chunked and abandoned the moment the limit is
exceeded. Checking ``Content-Length`` is not sufficient: it is a client-supplied
header, and chunked transfer encoding omits it entirely.
"""

from __future__ import annotations

from fastapi import UploadFile

from app.core.errors import PayloadTooLargeError

CHUNK_SIZE = 64 * 1024


async def read_limited(upload: UploadFile, limit: int, *, label: str = "file") -> bytes:
    buffer = bytearray()
    while chunk := await upload.read(CHUNK_SIZE):
        buffer.extend(chunk)
        if len(buffer) > limit:
            await upload.close()
            raise PayloadTooLargeError(
                f"The uploaded {label} exceeds the {limit // (1024 * 1024)} MiB limit.",
                limit_bytes=limit,
            )
    await upload.close()
    return bytes(buffer)

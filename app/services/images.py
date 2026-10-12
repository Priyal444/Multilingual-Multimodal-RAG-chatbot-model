"""Image ingestion: validate, normalise, and encode for the vision model.

Uploaded images are never forwarded verbatim. Three things happen first:

1. **The bytes are decoded and verified**, not trusted by their ``Content-Type``.
   A client can claim anything; ``Image.open`` is what actually decides.
2. **Oversized images are downscaled.** Image tokens scale with pixel area, so a
   12 MP phone photo costs many times what the model can actually use. Capping
   the long edge is the single largest cost lever in a vision app.
3. **EXIF is dropped and orientation is baked in.** EXIF carries GPS coordinates
   and device identifiers we have no reason to send to a third party — and a
   photo whose orientation lives only in a metadata tag arrives sideways.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass

from app.core.config import settings
from app.core.errors import PayloadTooLargeError, ValidationError
from app.core.logging import get_logger

log = get_logger(__name__)

# Formats the Anthropic vision API accepts.
SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

_PIL_FORMAT_TO_MEDIA_TYPE = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "GIF": "image/gif",
    "WEBP": "image/webp",
    "MPO": "image/jpeg",  # multi-picture JPEG from phone cameras
}


@dataclass(slots=True)
class PreparedImage:
    data_b64: str
    media_type: str
    width: int
    height: int
    size_bytes: int
    was_resized: bool


def prepare_image(raw: bytes, *, declared_type: str | None = None) -> PreparedImage:
    """Validate and normalise an uploaded image into a model-ready payload."""
    if not raw:
        raise ValidationError("The image file is empty.")
    if len(raw) > settings.max_image_bytes:
        raise PayloadTooLargeError(
            f"Image is {len(raw) // 1024} KiB; the limit is "
            f"{settings.max_image_bytes // 1024} KiB.",
            limit_bytes=settings.max_image_bytes,
        )

    try:
        from PIL import Image, ImageOps
    except ImportError as exc:  # pragma: no cover - declared dependency
        raise ValidationError("Pillow is required to process images.") from exc

    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception as exc:  # noqa: BLE001 - any decode failure is a bad upload
        raise ValidationError(
            "That file could not be decoded as an image.",
            declared_type=declared_type,
        ) from exc

    source_format = (image.format or "").upper()
    media_type = _PIL_FORMAT_TO_MEDIA_TYPE.get(source_format)
    if media_type is None:
        raise ValidationError(
            f"Unsupported image format '{source_format or 'unknown'}'.",
            supported=sorted(SUPPORTED_IMAGE_TYPES),
        )

    # Animated GIF/WEBP: the model sees a single frame, so send the first one
    # explicitly rather than letting the encoder pick.
    if getattr(image, "n_frames", 1) > 1:
        image.seek(0)

    original_size = image.size
    image = ImageOps.exif_transpose(image) or image  # bake in rotation, drop EXIF

    max_edge = settings.max_image_edge
    was_resized = False
    if max(image.size) > max_edge:
        image.thumbnail((max_edge, max_edge), Image.LANCZOS)
        was_resized = True

    # Re-encode. Anything with transparency stays PNG; everything else becomes
    # JPEG, which is dramatically smaller for photographs.
    has_alpha = image.mode in {"RGBA", "LA"} or (
        image.mode == "P" and "transparency" in image.info
    )
    buffer = io.BytesIO()
    if has_alpha:
        image.convert("RGBA").save(buffer, format="PNG", optimize=True)
        out_type = "image/png"
    else:
        image.convert("RGB").save(buffer, format="JPEG", quality=88, optimize=True)
        out_type = "image/jpeg"

    encoded = buffer.getvalue()
    log.debug(
        "image_prepared",
        original=f"{original_size[0]}x{original_size[1]}",
        final=f"{image.size[0]}x{image.size[1]}",
        original_bytes=len(raw),
        final_bytes=len(encoded),
        resized=was_resized,
    )

    return PreparedImage(
        # No newlines: the API rejects wrapped base64.
        data_b64=base64.b64encode(encoded).decode("ascii"),
        media_type=out_type,
        width=image.size[0],
        height=image.size[1],
        size_bytes=len(encoded),
        was_resized=was_resized,
    )


_DATA_URL_PREFIX = "data:"


def decode_data_url(value: str) -> tuple[bytes, str | None]:
    """Accept either a ``data:`` URL or bare base64 and return raw bytes.

    Browsers produce data URLs; scripted clients usually send bare base64.
    Supporting both removes a class of "works in curl, fails in the browser"
    bug reports.
    """
    value = (value or "").strip()
    if not value:
        raise ValidationError("Empty image payload.")

    declared: str | None = None
    if value.startswith(_DATA_URL_PREFIX):
        try:
            header, value = value.split(",", 1)
        except ValueError as exc:
            raise ValidationError("Malformed data URL: missing a comma separator.") from exc
        declared = header[len(_DATA_URL_PREFIX) :].split(";")[0] or None

    try:
        # validate=False tolerates whitespace and newlines that some clients add.
        return base64.b64decode(value, validate=False), declared
    except Exception as exc:  # noqa: BLE001
        raise ValidationError("Image payload is not valid base64.") from exc


def estimate_image_tokens(width: int, height: int) -> int:
    """Rough token cost of an image (~width x height / 750).

    Used for pre-flight budgeting only; ``/api/v1/chat/estimate`` reports the
    provider's exact count.
    """
    return max(1, (width * height) // 750)

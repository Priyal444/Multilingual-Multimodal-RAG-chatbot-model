"""Speech-to-text via faster-whisper.

Whisper is multilingual out of the box and returns the language it detected,
which we feed straight into the reply-language decision — so a voice note in
Tamil produces a Tamil reply without the user selecting anything.

Two operational details matter here:

* **The model is loaded lazily and exactly once.** A ``small`` model is ~500 MB
  of weights; loading it per request would dominate latency and memory. The
  double-checked lock guards against a burst of concurrent first requests each
  starting their own load.
* **Transcription runs in a worker thread.** faster-whisper is synchronous CPU
  work. Calling it directly on the event loop would stall every other request in
  the process for the duration of the transcription.
"""

from __future__ import annotations

import asyncio
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.errors import FeatureUnavailableError, ValidationError
from app.core.logging import get_logger

log = get_logger(__name__)

SUPPORTED_AUDIO_TYPES = {
    "audio/wav", "audio/x-wav", "audio/wave",
    "audio/mpeg", "audio/mp3",
    "audio/mp4", "audio/m4a", "audio/x-m4a",
    "audio/ogg", "audio/opus",
    "audio/webm", "video/webm",
    "audio/flac", "audio/x-flac",
}

_SUFFIX_BY_TYPE = {
    "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/wave": ".wav",
    "audio/mpeg": ".mp3", "audio/mp3": ".mp3",
    "audio/mp4": ".m4a", "audio/m4a": ".m4a", "audio/x-m4a": ".m4a",
    "audio/ogg": ".ogg", "audio/opus": ".opus",
    "audio/webm": ".webm", "video/webm": ".webm",
    "audio/flac": ".flac", "audio/x-flac": ".flac",
}


@dataclass(slots=True)
class Transcription:
    text: str
    language: str | None
    language_confidence: float
    duration_seconds: float


class SpeechToText:
    def __init__(self) -> None:
        self._model: Any = None
        self._lock = asyncio.Lock()

    @property
    def available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return False
        return True

    async def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        async with self._lock:
            if self._model is not None:  # another coroutine won the race
                return self._model
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise FeatureUnavailableError(
                    "Speech-to-text needs `faster-whisper`. Install it with: "
                    "pip install -r requirements-ml.txt"
                ) from exc

            log.info(
                "loading_whisper_model",
                model=settings.stt_model,
                device=settings.stt_device,
                compute_type=settings.stt_compute_type,
            )
            self._model = await asyncio.to_thread(
                WhisperModel,
                settings.stt_model,
                device=settings.stt_device,
                compute_type=settings.stt_compute_type,
            )
            log.info("whisper_model_ready", model=settings.stt_model)
            return self._model

    async def transcribe(
        self,
        audio: bytes,
        *,
        media_type: str,
        language: str | None = None,
    ) -> Transcription:
        """Transcribe audio bytes. ``language`` forces a language; None auto-detects."""
        if media_type not in SUPPORTED_AUDIO_TYPES:
            raise ValidationError(
                f"Unsupported audio type '{media_type}'.",
                supported=sorted(SUPPORTED_AUDIO_TYPES),
            )
        if not audio:
            raise ValidationError("The audio file is empty.")

        model = await self._get_model()
        suffix = _SUFFIX_BY_TYPE.get(media_type, ".bin")

        # NamedTemporaryFile(delete=False) + manual unlink: on Windows a file
        # cannot be reopened by another handle while still open here.
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        try:
            tmp.write(audio)
            tmp.close()
            return await asyncio.to_thread(self._transcribe_sync, model, tmp.name, language)
        finally:
            Path(tmp.name).unlink(missing_ok=True)

    @staticmethod
    def _transcribe_sync(model: Any, path: str, language: str | None) -> Transcription:
        segments, info = model.transcribe(
            path,
            language=language,
            beam_size=5,
            vad_filter=True,  # drop silence; large wins on phone-recorded notes
            vad_parameters={"min_silence_duration_ms": 500},
        )
        # `segments` is a generator — consuming it is what actually runs inference.
        text = "".join(segment.text for segment in segments).strip()
        return Transcription(
            text=text,
            language=getattr(info, "language", None),
            language_confidence=float(getattr(info, "language_probability", 0.0) or 0.0),
            duration_seconds=float(getattr(info, "duration", 0.0) or 0.0),
        )


speech_to_text = SpeechToText()

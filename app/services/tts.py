"""Text-to-speech via Microsoft Edge's neural voices (``edge-tts``).

Chosen over a paid TTS API for one reason that matters to this project: it
covers 90+ locales with natural-sounding neural voices and needs no API key, so
the multilingual story works end to end on a fresh clone.

Voice selection is dynamic-first — the catalogue is fetched once and cached, so
a new locale added upstream works without a code change — with a curated static
map as the offline fallback.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from app.core.errors import FeatureUnavailableError, UpstreamError, ValidationError
from app.core.logging import get_logger
from app.services.language import language_name

log = get_logger(__name__)

# Curated fallback, used when the voice catalogue can't be fetched.
_STATIC_VOICES: dict[str, str] = {
    "ar": "ar-EG-SalmaNeural", "bn": "bn-IN-TanishaaNeural", "cs": "cs-CZ-VlastaNeural",
    "da": "da-DK-ChristelNeural", "de": "de-DE-KatjaNeural", "el": "el-GR-AthinaNeural",
    "en": "en-US-AriaNeural", "es": "es-ES-ElviraNeural", "fa": "fa-IR-DilaraNeural",
    "fi": "fi-FI-SelmaNeural", "fr": "fr-FR-DeniseNeural", "gu": "gu-IN-DhwaniNeural",
    "he": "he-IL-HilaNeural", "hi": "hi-IN-SwaraNeural", "hu": "hu-HU-NoemiNeural",
    "id": "id-ID-GadisNeural", "it": "it-IT-ElsaNeural", "ja": "ja-JP-NanamiNeural",
    "kn": "kn-IN-SapnaNeural", "ko": "ko-KR-SunHiNeural", "ml": "ml-IN-SobhanaNeural",
    "mr": "mr-IN-AarohiNeural", "ms": "ms-MY-YasminNeural", "nl": "nl-NL-ColetteNeural",
    "no": "nb-NO-PernilleNeural", "pl": "pl-PL-ZofiaNeural", "pt": "pt-BR-FranciscaNeural",
    "ro": "ro-RO-AlinaNeural", "ru": "ru-RU-SvetlanaNeural", "sv": "sv-SE-SofieNeural",
    "sw": "sw-KE-ZuriNeural", "ta": "ta-IN-PallaviNeural", "te": "te-IN-ShrutiNeural",
    "th": "th-TH-PremwadeeNeural", "tr": "tr-TR-EmelNeural", "uk": "uk-UA-PolinaNeural",
    "ur": "ur-PK-UzmaNeural", "vi": "vi-VN-HoaiMyNeural", "zh": "zh-CN-XiaoxiaoNeural",
}

DEFAULT_VOICE = "en-US-AriaNeural"
MAX_TTS_CHARS = 5000


@dataclass(slots=True)
class Speech:
    audio: bytes
    media_type: str
    voice: str
    language: str


class TextToSpeech:
    def __init__(self) -> None:
        self._catalogue: list[dict[str, Any]] | None = None
        self._lock = asyncio.Lock()

    @property
    def available(self) -> bool:
        try:
            import edge_tts  # noqa: F401
        except ImportError:
            return False
        return True

    async def _voice_catalogue(self) -> list[dict[str, Any]]:
        if self._catalogue is not None:
            return self._catalogue
        async with self._lock:
            if self._catalogue is not None:
                return self._catalogue
            try:
                import edge_tts

                self._catalogue = await edge_tts.list_voices()
                log.info("tts_catalogue_loaded", voices=len(self._catalogue))
            except Exception as exc:  # noqa: BLE001 - fall back to the static map
                log.warning("tts_catalogue_unavailable", error=str(exc))
                self._catalogue = []
            return self._catalogue

    async def resolve_voice(self, language: str | None, voice: str | None = None) -> str:
        """Pick a concrete voice short-name for a language code."""
        if voice:
            return voice
        code = (language or "en").split("-")[0].lower()

        for entry in await self._voice_catalogue():
            locale = str(entry.get("Locale", ""))
            if locale.lower().startswith(f"{code}-"):
                return str(entry.get("ShortName", DEFAULT_VOICE))

        return _STATIC_VOICES.get(code, DEFAULT_VOICE)

    async def list_voices(self, language: str | None = None) -> list[dict[str, str]]:
        """Voice catalogue for the UI, optionally filtered by language."""
        catalogue = await self._voice_catalogue()
        if not catalogue:
            return [
                {"short_name": v, "locale": v.rsplit("-", 1)[0], "language": language_name(k)}
                for k, v in sorted(_STATIC_VOICES.items())
                if language is None or k == language.split("-")[0].lower()
            ]

        rows: list[dict[str, str]] = []
        prefix = f"{language.split('-')[0].lower()}-" if language else None
        for entry in catalogue:
            locale = str(entry.get("Locale", ""))
            if prefix and not locale.lower().startswith(prefix):
                continue
            rows.append(
                {
                    "short_name": str(entry.get("ShortName", "")),
                    "locale": locale,
                    "gender": str(entry.get("Gender", "")),
                    "language": language_name(locale.split("-")[0].lower()),
                }
            )
        return rows

    async def synthesize(
        self,
        text: str,
        *,
        language: str | None = None,
        voice: str | None = None,
        rate: str = "+0%",
    ) -> Speech:
        text = (text or "").strip()
        if not text:
            raise ValidationError("Nothing to speak: the text is empty.")
        if len(text) > MAX_TTS_CHARS:
            raise ValidationError(
                f"Text is too long to synthesize ({len(text)} chars, "
                f"limit {MAX_TTS_CHARS}).",
                limit=MAX_TTS_CHARS,
            )

        try:
            import edge_tts
        except ImportError as exc:
            raise FeatureUnavailableError(
                "Text-to-speech needs `edge-tts`. Install it with: "
                "pip install -r requirements.txt"
            ) from exc

        chosen = await self.resolve_voice(language, voice)
        try:
            communicate = edge_tts.Communicate(text, chosen, rate=rate)
            audio = bytearray()
            async for chunk in communicate.stream():
                if chunk.get("type") == "audio" and chunk.get("data"):
                    audio.extend(chunk["data"])
        except Exception as exc:  # noqa: BLE001 - normalised for the API layer
            log.warning("tts_failed", voice=chosen, error=str(exc))
            raise UpstreamError("Speech synthesis failed.") from exc

        if not audio:
            raise UpstreamError("Speech synthesis returned no audio.")

        return Speech(
            audio=bytes(audio),
            media_type="audio/mpeg",
            voice=chosen,
            language=(language or "en"),
        )


text_to_speech = TextToSpeech()

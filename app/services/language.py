"""Language identification for short, messy chat messages.

Off-the-shelf detectors are trained on paragraphs and get noticeably worse on
the 3-8 word fragments people actually type into a chat box. They also fail in a
specific, avoidable way: a statistical model can return ``en`` for a line of
Devanagari, because n-gram evidence is thin and the prior is strong.

So this module runs a two-stage check. Unicode script is decided first — it is a
hard fact, not a guess — and narrows the answer to a family. The statistical
detector then disambiguates *within* that family (Hindi vs. Marathi, Russian vs.
Ukrainian). If the detector's answer contradicts the script, the script wins.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from app.core.logging import get_logger

log = get_logger(__name__)

try:
    from langdetect import DetectorFactory, detect_langs
    from langdetect.lang_detect_exception import LangDetectException

    DetectorFactory.seed = 0  # deterministic output across runs
    _LANGDETECT_AVAILABLE = True
except ImportError:  # pragma: no cover - declared dependency
    _LANGDETECT_AVAILABLE = False

    class LangDetectException(Exception):  # type: ignore[no-redef]
        pass


# Unicode block ranges → (script name, default language, sibling languages).
# The default is the most widely spoken language written in that script; the
# siblings are what the statistical detector is allowed to choose instead.
_SCRIPTS: list[tuple[str, range, str, frozenset[str]]] = [
    ("Devanagari", range(0x0900, 0x0980), "hi", frozenset({"hi", "mr", "ne", "sa"})),
    ("Bengali", range(0x0980, 0x0A00), "bn", frozenset({"bn", "as"})),
    ("Gurmukhi", range(0x0A00, 0x0A80), "pa", frozenset({"pa"})),
    ("Gujarati", range(0x0A80, 0x0B00), "gu", frozenset({"gu"})),
    ("Oriya", range(0x0B00, 0x0B80), "or", frozenset({"or"})),
    ("Tamil", range(0x0B80, 0x0C00), "ta", frozenset({"ta"})),
    ("Telugu", range(0x0C00, 0x0C80), "te", frozenset({"te"})),
    ("Kannada", range(0x0C80, 0x0D00), "kn", frozenset({"kn"})),
    ("Malayalam", range(0x0D00, 0x0D80), "ml", frozenset({"ml"})),
    ("Sinhala", range(0x0D80, 0x0E00), "si", frozenset({"si"})),
    ("Thai", range(0x0E00, 0x0E80), "th", frozenset({"th"})),
    ("Lao", range(0x0E80, 0x0F00), "lo", frozenset({"lo"})),
    ("Tibetan", range(0x0F00, 0x1000), "bo", frozenset({"bo"})),
    ("Myanmar", range(0x1000, 0x10A0), "my", frozenset({"my"})),
    ("Georgian", range(0x10A0, 0x1100), "ka", frozenset({"ka"})),
    ("Hangul", range(0x1100, 0x1200), "ko", frozenset({"ko"})),
    ("Ethiopic", range(0x1200, 0x1380), "am", frozenset({"am", "ti"})),
    ("Khmer", range(0x1780, 0x1800), "km", frozenset({"km"})),
    ("Cyrillic", range(0x0400, 0x0530), "ru", frozenset({"ru", "uk", "bg", "sr", "mk", "be"})),
    ("Armenian", range(0x0530, 0x0590), "hy", frozenset({"hy"})),
    ("Hebrew", range(0x0590, 0x0600), "he", frozenset({"he", "yi"})),
    ("Arabic", range(0x0600, 0x0700), "ar", frozenset({"ar", "fa", "ur", "ps"})),
    ("Greek", range(0x0370, 0x0400), "el", frozenset({"el"})),
    ("Hiragana", range(0x3040, 0x30A0), "ja", frozenset({"ja"})),
    ("Katakana", range(0x30A0, 0x3100), "ja", frozenset({"ja"})),
    ("CJK", range(0x4E00, 0xA000), "zh", frozenset({"zh", "ja", "ko", "zh-cn", "zh-tw"})),
]

LANGUAGE_NAMES: dict[str, str] = {
    "af": "Afrikaans", "am": "Amharic", "ar": "Arabic", "as": "Assamese",
    "be": "Belarusian", "bg": "Bulgarian", "bn": "Bengali", "bo": "Tibetan",
    "ca": "Catalan", "cs": "Czech", "cy": "Welsh", "da": "Danish",
    "de": "German", "el": "Greek", "en": "English", "es": "Spanish",
    "et": "Estonian", "fa": "Persian", "fi": "Finnish", "fr": "French",
    "gu": "Gujarati", "he": "Hebrew", "hi": "Hindi", "hr": "Croatian",
    "hu": "Hungarian", "hy": "Armenian", "id": "Indonesian", "it": "Italian",
    "ja": "Japanese", "ka": "Georgian", "km": "Khmer", "kn": "Kannada",
    "ko": "Korean", "lo": "Lao", "lt": "Lithuanian", "lv": "Latvian",
    "mk": "Macedonian", "ml": "Malayalam", "mr": "Marathi", "ms": "Malay",
    "my": "Burmese", "ne": "Nepali", "nl": "Dutch", "no": "Norwegian",
    "or": "Odia", "pa": "Punjabi", "pl": "Polish", "ps": "Pashto",
    "pt": "Portuguese", "ro": "Romanian", "ru": "Russian", "sa": "Sanskrit",
    "si": "Sinhala", "sk": "Slovak", "sl": "Slovenian", "so": "Somali",
    "sq": "Albanian", "sr": "Serbian", "sv": "Swedish", "sw": "Swahili",
    "ta": "Tamil", "te": "Telugu", "th": "Thai", "ti": "Tigrinya",
    "tl": "Tagalog", "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu",
    "vi": "Vietnamese", "yi": "Yiddish", "zh": "Chinese",
    "zh-cn": "Chinese (Simplified)", "zh-tw": "Chinese (Traditional)",
}

_URL_OR_CODE = re.compile(r"https?://\S+|`[^`]*`|```.*?```", re.DOTALL)
_NON_LETTER = re.compile(r"[^\w\s]", re.UNICODE)

MIN_RELIABLE_CHARS = 12

# Before letting the statistical detector pick a *sibling* of the script's
# default (Nepali instead of Hindi, Ukrainian instead of Russian), demand both
# length and confidence. langdetect reports ~0.99 probability on a six-character
# Devanagari string and is frequently wrong about which Devanagari language it
# is — confidence alone is not a usable gate, so length carries the decision.
SIBLING_OVERRIDE_MIN_CHARS = 40
SIBLING_OVERRIDE_MIN_CONFIDENCE = 0.90


@dataclass(slots=True)
class LanguageGuess:
    code: str
    name: str
    confidence: float
    # "script" | "statistical" | "hybrid" | "default" — surfaced in the API so a
    # client can decide whether to prompt the user to confirm.
    method: str

    @property
    def is_reliable(self) -> bool:
        return self.confidence >= 0.65


def normalize_code(code: str | None) -> str | None:
    """Normalise a BCP-47-ish tag to the primary subtag we key on."""
    if not code:
        return None
    code = code.strip().lower().replace("_", "-")
    if code in LANGUAGE_NAMES:
        return code
    primary = code.split("-")[0]
    return primary if primary in LANGUAGE_NAMES else primary or None


def language_name(code: str | None) -> str:
    if not code:
        return "the user's language"
    return LANGUAGE_NAMES.get(code, LANGUAGE_NAMES.get(code.split("-")[0], code))


def _clean(text: str) -> str:
    """Strip URLs and code, which are near-always Latin and skew detection."""
    text = _URL_OR_CODE.sub(" ", text)
    return _NON_LETTER.sub(" ", text).strip()


def _dominant_script(text: str) -> tuple[str, str, frozenset[str]] | None:
    """Return ``(script, default_lang, siblings)`` for the most common script."""
    counts: dict[int, int] = {}
    for char in text:
        if not char.isalpha():
            continue
        cp = ord(char)
        for idx, (_name, block, _default, _siblings) in enumerate(_SCRIPTS):
            if cp in block:
                counts[idx] = counts.get(idx, 0) + 1
                break
    if not counts:
        return None
    best_idx = max(counts, key=lambda k: counts[k])
    name, _block, default, siblings = _SCRIPTS[best_idx]
    return name, default, siblings


def detect_language(text: str, *, fallback: str = "en") -> LanguageGuess:
    """Best-effort language identification. Never raises."""
    cleaned = _clean(text or "")
    if len(cleaned) < 2:
        return LanguageGuess(fallback, language_name(fallback), 0.0, "default")

    script = _dominant_script(cleaned)

    statistical_code: str | None = None
    statistical_conf = 0.0
    if _LANGDETECT_AVAILABLE and len(cleaned) >= 4:
        try:
            candidates = detect_langs(cleaned)
            if candidates:
                statistical_code = normalize_code(candidates[0].lang)
                statistical_conf = float(candidates[0].prob)
        except LangDetectException:
            pass
        except Exception as exc:  # noqa: BLE001 - detection must never break chat
            log.debug("langdetect_failed", error=str(exc))

    # --- Reconcile the two signals ---------------------------------------
    if script is not None:
        _name, default, siblings = script
        detector_agrees = statistical_code in siblings if statistical_code else False
        enough_evidence = (
            len(cleaned) >= SIBLING_OVERRIDE_MIN_CHARS
            and statistical_conf >= SIBLING_OVERRIDE_MIN_CONFIDENCE
        )
        if statistical_code and detector_agrees and (
            statistical_code == default or enough_evidence
        ):
            # The detector picked a language that genuinely uses this script,
            # and either it agrees with the script default or there is enough
            # text to trust it over that default.
            return LanguageGuess(
                statistical_code, language_name(statistical_code), statistical_conf, "hybrid"
            )
        # Contradiction, or a sibling guess on too little text: the script wins.
        confidence = 0.9 if len(cleaned) >= MIN_RELIABLE_CHARS else 0.75
        return LanguageGuess(default, language_name(default), confidence, "script")

    if statistical_code:
        # Latin script: no hard evidence available, so scale confidence down for
        # very short inputs rather than reporting the detector's optimism.
        penalty = 1.0 if len(cleaned) >= MIN_RELIABLE_CHARS else 0.6
        return LanguageGuess(
            statistical_code,
            language_name(statistical_code),
            round(statistical_conf * penalty, 3),
            "statistical",
        )

    return LanguageGuess(fallback, language_name(fallback), 0.0, "default")


def script_of(code: str) -> str:
    """Human-readable script name for a language code (used in the UI)."""
    for name, _block, default, siblings in _SCRIPTS:
        if code == default or code in siblings:
            return name
    return "Latin"


def is_rtl(code: str | None) -> bool:
    return (code or "").split("-")[0] in {"ar", "he", "fa", "ur", "ps", "yi"}


def display_name(code: str | None) -> str:
    """Language name plus its endonym-ish Unicode name, for logs and UI chips."""
    if not code:
        return "auto"
    name = language_name(code)
    try:
        return f"{name} ({unicodedata.normalize('NFC', code)})"
    except Exception:  # noqa: BLE001
        return name

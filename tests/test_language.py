"""Language identification.

The cases that matter are the ones a purely statistical detector gets wrong:
short strings in non-Latin scripts, where n-gram evidence is thin and the model
falls back to its English prior.
"""

from __future__ import annotations

import pytest

from app.services.language import (
    detect_language,
    is_rtl,
    language_name,
    normalize_code,
    script_of,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("नमस्ते, आप कैसे हैं?", "hi"),
        ("مرحبا كيف حالك", "ar"),
        ("こんにちは、元気ですか", "ja"),
        ("안녕하세요 잘 지내세요", "ko"),
        ("Здравствуйте, как дела?", "ru"),
        ("สวัสดีครับ สบายดีไหม", "th"),
        ("שלום מה שלומך", "he"),
        ("வணக்கம் எப்படி இருக்கிறீர்கள்", "ta"),
        ("નમસ્તે કેમ છો", "gu"),
        ("ನಮಸ್ಕಾರ ಹೇಗಿದ್ದೀರಿ", "kn"),
    ],
)
def test_non_latin_scripts_are_identified(text: str, expected: str) -> None:
    guess = detect_language(text)
    assert guess.code == expected
    assert guess.confidence > 0.5


def test_short_non_latin_input_still_resolves() -> None:
    # Three characters is far below what a statistical detector needs, but the
    # script alone is decisive.
    guess = detect_language("नमस्ते")
    assert guess.code == "hi"
    assert guess.method in {"script", "hybrid"}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("This is a reasonably long English sentence for detection.", "en"),
        ("Ceci est une phrase française assez longue pour la détection.", "fr"),
        ("Esta es una frase en español lo bastante larga para detectarla.", "es"),
    ],
)
def test_latin_scripts_use_the_statistical_detector(text: str, expected: str) -> None:
    guess = detect_language(text)
    assert guess.code == expected
    assert guess.method == "statistical"


def test_urls_and_code_do_not_skew_detection() -> None:
    # Without stripping, the Latin URL dominates the n-gram evidence.
    guess = detect_language("यह देखें https://example.com/some/long/path `print(x)`")
    assert guess.code == "hi"


def test_empty_input_falls_back_without_raising() -> None:
    guess = detect_language("")
    assert guess.code == "en"
    assert guess.confidence == 0.0
    assert guess.method == "default"


def test_confidence_is_discounted_for_very_short_latin_text() -> None:
    short = detect_language("ok")
    long = detect_language("This is a much longer English sentence to detect.")
    assert short.confidence < long.confidence


def test_normalize_code_handles_region_subtags() -> None:
    assert normalize_code("en-US") == "en"
    assert normalize_code("HI") == "hi"
    assert normalize_code("pt_BR") == "pt"
    assert normalize_code(None) is None


def test_rtl_and_script_helpers() -> None:
    assert is_rtl("ar") and is_rtl("he") and is_rtl("ur")
    assert not is_rtl("en")
    assert script_of("hi") == "Devanagari"
    assert language_name("ta") == "Tamil"

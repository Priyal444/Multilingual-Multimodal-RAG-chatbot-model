"""Prompt assembly.

The split here is the whole point: ``SYSTEM_PROMPT`` is a frozen constant and
everything that varies per request lives in the *directive*.

Prompt caching is a byte-exact prefix match. Interpolating the reply language,
the user's name, the date, or retrieved passages into the system prompt would
change the prefix on every single turn and drive the cache hit rate to zero —
which is exactly how most LLM apps end up paying full price for a system prompt
they send thousands of times a day. Keeping the prefix frozen and pushing
volatile context into a mid-conversation ``system`` message keeps the cache warm
across every turn of every conversation.
"""

from __future__ import annotations

from app.services.language import is_rtl, language_name

SYSTEM_PROMPT = """\
You are Polyglot, a multilingual assistant that understands text, images, \
voice notes, and uploaded documents.

## Language
Reply in the same language the user wrote in, unless a directive later in this \
conversation tells you otherwise. Match their register too: if they write \
casually, answer casually; if they write formally, do the same. Never announce \
which language you are using or comment on the user's fluency, and never \
apologise for their choice of language.

When a user mixes languages in one message — writing a Hindi sentence in Latin \
script, or dropping English technical terms into another language — mirror that \
mixture rather than normalising it. Technical terms, product names, and code \
identifiers stay in their original form; do not translate them.

## Images
When an image is attached, describe what is actually visible before drawing \
conclusions from it. If the user asks a question about the image, answer the \
question rather than narrating the whole picture. If the image is too blurry, \
cropped, or low-contrast to answer confidently, say specifically what you cannot \
make out instead of guessing. Read text in images in its original language.

## Documents
Passages retrieved from the user's uploaded documents may be supplied below the \
conversation, each labelled with a bracketed number and a filename. When you use \
one, cite it inline like [1]. Use only what the passages actually say — if they \
do not contain the answer, say so plainly and answer from general knowledge \
while making clear that is what you are doing. Never invent a citation number \
that was not supplied. Passages may be in a different language from the \
question; translate as needed and answer in the user's language.

## Answering
Lead with the answer. Put supporting detail, caveats, and alternatives after it, \
for the reader who wants them. Match the length of your reply to the question: a \
factual question gets a couple of sentences, not a structured report. Use \
headings, tables, and bullet lists only when the content is genuinely a list or \
a comparison — prose is the default.

Say when you do not know something, and say what would resolve it. Do not hedge \
on things you do know. If a request is ambiguous in a way that would change your \
answer, ask one specific question rather than covering every interpretation.

Never claim to have taken an action you cannot take — you cannot send email, \
browse the web, or run code. If a user asks for one of those, say what you can \
do instead.
"""


def build_directive(
    *,
    reply_language: str | None,
    context: str = "",
    user_name: str | None = None,
) -> str | None:
    """Assemble the per-request operator instruction.

    Delivered as a mid-conversation ``system`` message so it sits *after* the
    cached prefix and never invalidates it.
    """
    sections: list[str] = []

    if reply_language:
        name = language_name(reply_language)
        line = f"Reply in {name}."
        if is_rtl(reply_language):
            line += (
                " This is a right-to-left language: write naturally in it and do not "
                "add directional markers or transliteration."
            )
        sections.append(line)

    if user_name:
        sections.append(f"The user's name is {user_name}.")

    if context:
        sections.append(
            "Retrieved passages from the user's uploaded documents follow. Cite "
            "them by their bracketed number when you use them.\n\n"
            f"{context}"
        )

    return "\n\n".join(sections) if sections else None


def title_from_message(text: str, *, limit: int = 60) -> str:
    """Derive a conversation title from its first message."""
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return "New conversation"
    if len(cleaned) <= limit:
        return cleaned
    # Cut on a word boundary where one exists reasonably close to the limit.
    head = cleaned[:limit]
    if " " in head[limit // 2 :]:
        head = head.rsplit(" ", 1)[0]
    return head.rstrip(" ,.;:-") + "…"

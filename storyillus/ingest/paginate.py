"""Split chapter text into pages — the unit one illustration covers."""

import re

from storyillus.models import Page

TARGET_WORDS = 300

_PARAGRAPHS = re.compile(r"\n\s*\n")


def paginate(text: str, *, target_words: int = TARGET_WORDS) -> list[Page]:
    """Group paragraphs into pages of roughly `target_words` words.

    Paragraphs are never split. A break inside one costs the model the context it needs to
    say who is present and what is happening, which is exactly what a `ScenePlan` asks for.
    A paragraph longer than the target becomes a page on its own.

    A page closes when the next paragraph would carry it further from the target than
    stopping does. Simply closing at the first overshoot instead lands pages well above the
    target — real prose paragraphs are big enough that the last one added dominates.
    """
    pages: list[Page] = []
    current: list[str] = []
    words = 0

    for paragraph in _paragraphs(text):
        length = len(paragraph.split())
        if current and abs(words + length - target_words) > abs(words - target_words):
            pages.append(Page(index=len(pages) + 1, text="\n\n".join(current)))
            current, words = [], 0

        current.append(paragraph)
        words += length

    if current:
        pages.append(Page(index=len(pages) + 1, text="\n\n".join(current)))
    return pages


def _paragraphs(text: str) -> list[str]:
    return [block.strip() for block in _PARAGRAPHS.split(text) if block.strip()]

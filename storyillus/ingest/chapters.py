"""Split a fetched book into chapters and pick which ones to illustrate.

A novel is far too large to illustrate in one run — Frankenstein alone is ~438k characters,
which is 200+ pages and therefore 200+ diffusion renders. The chapter is the unit a run
operates on, so this module turns the `## ` headings written by `gutenberg.to_markdown`
back into addressable pieces.
"""

import re

from storyillus.models import Chapter

FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
# A chapter heading, not a `###` sub-heading.
HEADING = re.compile(r"^##(?!#)\s*(.*)$", re.MULTILINE)
TITLE_LINE = re.compile(r"^#(?!#)\s*.*$", re.MULTILINE)


def parse_frontmatter(markdown: str) -> dict[str, str]:
    """Read the YAML-ish header written at download time. Returns {} if there is none."""
    match = FRONTMATTER.search(markdown)
    if not match:
        return {}

    fields: dict[str, str] = {}
    for line in match.group(1).splitlines():
        key, separator, value = line.partition(":")
        if separator:
            fields[key.strip()] = value.strip().strip('"')
    return fields


def strip_frontmatter(markdown: str) -> str:
    return FRONTMATTER.sub("", markdown, count=1)


def split_chapters(markdown: str) -> list[Chapter]:
    """Split a book's markdown on its `## ` headings.

    Anything before the first heading — title page, dedication — is dropped. It is front
    matter, not story.

    Headings with no text under them are dropped too, and this matters more than it sounds:
    a Gutenberg table of contents lists every chapter name on its own line, so the download
    step promotes all of them to `## ` headings. Frankenstein produces 28 such phantoms
    ahead of its 28 real chapters. Requiring a body distinguishes the two without having to
    guess where the contents block ends. It also discards bare section dividers like a
    "Part 1" that is immediately followed by "Chapter 1" — correctly, since there is nothing
    there to illustrate.

    A book whose chapter headings were never detected yields one chapter holding the whole
    text, so the pipeline degrades to plain pagination instead of failing.
    """
    body = strip_frontmatter(markdown)
    headings = list(HEADING.finditer(body))
    whole_text = TITLE_LINE.sub("", body, count=1).strip()

    if not headings:
        return [Chapter(index=1, heading="(whole text)", text=whole_text)] if whole_text else []

    chapters = []
    for position, match in enumerate(headings, start=1):
        end = headings[position].start() if position < len(headings) else len(body)
        text = body[match.end() : end].strip()
        if text:
            chapters.append(
                Chapter(index=len(chapters) + 1, heading=match.group(1).strip(), text=text)
            )

    # Every heading was a phantom, so the real text is whatever sits outside them.
    if not chapters:
        return [Chapter(index=1, heading="(whole text)", text=whole_text)] if whole_text else []
    return chapters


def parse_selection(spec: str, available: int) -> list[int]:
    """Turn `"3"`, `"1-3"`, or `"1,4,7-9"` into a sorted list of 1-based chapter numbers."""
    if available == 0:
        raise ValueError("this book has no chapters to select from")

    selected: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        start, separator, end = part.partition("-")
        try:
            low = int(start)
            high = int(end) if separator else low
        except ValueError:
            raise ValueError(f"{part!r} is not a chapter number or range like '2' or '1-3'") from None
        if low > high:
            raise ValueError(f"{part!r} counts backwards; write it as '{high}-{low}'")
        selected.update(range(low, high + 1))

    if not selected:
        raise ValueError("no chapters selected")

    out_of_range = sorted(n for n in selected if not 1 <= n <= available)
    if out_of_range:
        raise ValueError(
            f"no chapter {', '.join(str(n) for n in out_of_range)} — this book has 1-{available}"
        )
    return sorted(selected)


def select(chapters: list[Chapter], spec: str | None) -> list[Chapter]:
    """Apply a selection string, defaulting to the first chapter only.

    The default is deliberately one chapter, not all of them: an unqualified run should cost
    minutes, not hours.
    """
    if spec is None:
        return chapters[:1]
    wanted = parse_selection(spec, len(chapters))
    return [chapters[n - 1] for n in wanted]

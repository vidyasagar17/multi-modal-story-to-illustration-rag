"""Fetch public-domain books from Project Gutenberg as markdown.

Search runs against Gutenberg's official catalog CSV, cached locally after the first
download. The gutendex.com JSON API is not used: it sits behind a Cloudflare challenge
that returns 403 to non-browser clients.
"""

import csv
import re
import sys
from pathlib import Path

import httpx

from storyillus.models import Book

CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
TEXT_URLS = (
    "https://www.gutenberg.org/ebooks/{id}.txt.utf-8",
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
)
USER_AGENT = "storyillus/0.1 (https://github.com/; public-domain research)"

# Gutenberg wraps every text in licence boilerplate. Wording varies across decades of
# releases, so both markers are matched loosely.
START_MARKER = re.compile(r"^\*\*\*\s*START OF TH(?:E|IS) PROJECT GUTENBERG EBOOK.*$", re.MULTILINE)
END_MARKER = re.compile(r"^\*\*\*\s*END OF TH(?:E|IS) PROJECT GUTENBERG EBOOK.*$", re.MULTILINE)
CHAPTER_LINE = re.compile(
    r"^(chapter|part|book|act|canto|letter)\s+([IVXLCDM]+|\d+|[A-Z]+)\.?\s*$",
    re.IGNORECASE,
)


def catalog_path(cache_dir: Path) -> Path:
    return cache_dir / "pg_catalog.csv"


def ensure_catalog(cache_dir: Path) -> Path:
    """Download the catalog on first use; reuse the cached copy afterwards."""
    path = catalog_path(cache_dir)
    if path.exists():
        return path

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Gutenberg catalog (~21 MB, one time) to {path}", file=sys.stderr)
    with httpx.stream(
        "GET", CATALOG_URL, headers={"User-Agent": USER_AGENT}, timeout=120, follow_redirects=True
    ) as response:
        response.raise_for_status()
        with path.open("wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)
    return path


def search(query: str, cache_dir: Path, limit: int = 10) -> list[Book]:
    """Return text books whose title or author matches the query, best matches first.

    Audiobook rows share titles with the text editions, so only Type=Text is considered.
    """
    needle = query.strip().lower()
    if not needle:
        return []

    matches: list[tuple[int, Book]] = []
    with catalog_path(cache_dir).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["Type"] != "Text":
                continue
            title = row["Title"].replace("\n", " ").strip()
            author = row["Authors"].replace("\n", " ").strip()
            rank = _rank(needle, title.lower(), author.lower())
            if rank is not None:
                book = Book(int(row["Text#"]), title, author, row["Language"])
                matches.append((rank, book))

    matches.sort(key=lambda pair: (pair[0], pair[1].id))
    return [book for _, book in matches[:limit]]


def _rank(needle: str, title: str, author: str) -> int | None:
    """Lower is better. None means no match."""
    if title == needle:
        return 0
    if title.startswith(needle):
        return 1
    if needle in title:
        return 2
    if needle in author:
        return 3
    return None


def fetch_text(book_id: int) -> str:
    """Download a book's plain text, trying the known URL layouts in order."""
    errors = []
    for template in TEXT_URLS:
        url = template.format(id=book_id)
        response = httpx.get(
            url, headers={"User-Agent": USER_AGENT}, timeout=120, follow_redirects=True
        )
        if response.status_code == 200:
            return response.text
        errors.append(f"{url} -> {response.status_code}")
    raise RuntimeError(f"No plain text available for book {book_id}: {'; '.join(errors)}")


def strip_boilerplate(raw: str) -> str:
    """Drop Gutenberg's licence header and footer, keeping only the work itself."""
    start = START_MARKER.search(raw)
    body = raw[start.end() :] if start else raw
    end = END_MARKER.search(body)
    if end:
        body = body[: end.start()]
    return body.strip()


def to_markdown(book: Book, body: str) -> str:
    """Wrap the stripped text in frontmatter and promote chapter lines to headings."""
    lines = [
        "---",
        f'title: "{book.title}"',
        f'author: "{book.author}"',
        f"gutenberg_id: {book.id}",
        f"language: {book.language}",
        f"source: https://www.gutenberg.org/ebooks/{book.id}",
        "license: public domain (Project Gutenberg)",
        "---",
        "",
        f"# {book.title}",
        "",
    ]
    for line in body.splitlines():
        stripped = line.strip()
        lines.append(f"## {stripped}" if CHAPTER_LINE.match(stripped) else line)
    return "\n".join(lines).rstrip() + "\n"


def download(book: Book, out_dir: Path) -> Path:
    """Fetch one book and write it as markdown. Returns the written path."""
    markdown = to_markdown(book, strip_boilerplate(fetch_text(book.id)))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{book.slug}.md"
    path.write_text(markdown, encoding="utf-8")
    return path

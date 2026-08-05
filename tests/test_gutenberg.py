"""Offline tests: parsing and formatting only, no network."""

from storyillus.ingest.gutenberg import strip_boilerplate, to_markdown
from storyillus.models import Book

BOOK = Book(id=1342, title="Pride and Prejudice", author="Austen, Jane", language="en")


def test_strip_boilerplate_removes_header_and_footer():
    raw = (
        "The Project Gutenberg eBook of Something\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK SOMETHING ***\n"
        "Real story text.\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK SOMETHING ***\n"
        "Licence terms follow.\n"
    )
    assert strip_boilerplate(raw) == "Real story text."


def test_strip_boilerplate_accepts_the_older_this_wording():
    raw = "header\n*** START OF THIS PROJECT GUTENBERG EBOOK X ***\nbody\n"
    assert strip_boilerplate(raw) == "body"


def test_strip_boilerplate_passes_through_text_without_markers():
    assert strip_boilerplate("  plain story  ") == "plain story"


def test_to_markdown_writes_frontmatter_and_promotes_chapters():
    markdown = to_markdown(BOOK, "CHAPTER I.\n\nIt is a truth universally acknowledged.")
    assert markdown.startswith("---\n")
    assert 'title: "Pride and Prejudice"' in markdown
    assert "gutenberg_id: 1342" in markdown
    assert "## CHAPTER I." in markdown
    assert "It is a truth universally acknowledged." in markdown


def test_to_markdown_leaves_prose_alone():
    markdown = to_markdown(BOOK, "Chapter one was the best part of the book.")
    assert "## Chapter one" not in markdown

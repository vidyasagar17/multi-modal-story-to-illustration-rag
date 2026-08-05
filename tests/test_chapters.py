"""Offline tests: chapter splitting and selection, no network."""

import pytest

from storyillus.ingest.chapters import (
    parse_frontmatter,
    parse_selection,
    select,
    split_chapters,
)

BOOK_MD = """---
title: "Frankenstein"
author: "Shelley, Mary Wollstonecraft"
gutenberg_id: 84
---

# Frankenstein

Table of contents nobody wants illustrated.

## Letter 1

To Mrs. Saville, England.

## Chapter 1

I am by birth a Genevese.

## Chapter 2

We were brought up together.
"""


def test_parse_frontmatter_reads_fields_and_unquotes():
    meta = parse_frontmatter(BOOK_MD)
    assert meta["title"] == "Frankenstein"
    assert meta["author"] == "Shelley, Mary Wollstonecraft"
    assert meta["gutenberg_id"] == "84"


def test_parse_frontmatter_without_a_header_is_empty():
    assert parse_frontmatter("# Just a title\n\nprose") == {}


def test_split_chapters_finds_each_heading():
    chapters = split_chapters(BOOK_MD)
    assert [c.heading for c in chapters] == ["Letter 1", "Chapter 1", "Chapter 2"]
    assert [c.index for c in chapters] == [1, 2, 3]


def test_split_chapters_captures_the_text_under_each_heading():
    chapters = split_chapters(BOOK_MD)
    assert chapters[0].text == "To Mrs. Saville, England."
    assert chapters[2].text == "We were brought up together."


def test_split_chapters_drops_front_matter_before_the_first_heading():
    joined = " ".join(c.text for c in split_chapters(BOOK_MD))
    assert "Table of contents" not in joined


def test_split_chapters_falls_back_to_one_chapter_when_none_are_detected():
    chapters = split_chapters("---\ntitle: \"X\"\n---\n\n# X\n\nAn unbroken wall of prose.")
    assert len(chapters) == 1
    assert chapters[0].heading == "(whole text)"
    assert chapters[0].text == "An unbroken wall of prose."


def test_split_chapters_drops_a_table_of_contents():
    """Gutenberg lists chapter names on their own lines, so download promotes them all."""
    toc_book = (
        "# A Book\n\n"
        "## Chapter 1\n\n"  # contents entry: no body
        "## Chapter 2\n\n"  # contents entry: no body
        "## Chapter 1\n\nThe real first chapter.\n\n"
        "## Chapter 2\n\nThe real second chapter.\n"
    )
    chapters = split_chapters(toc_book)
    assert len(chapters) == 2
    assert [c.index for c in chapters] == [1, 2]
    assert chapters[0].text == "The real first chapter."


def test_split_chapters_survives_a_book_that_is_only_a_contents_list():
    chapters = split_chapters("# A Book\n\nReal prose.\n\n## Chapter 1\n\n## Chapter 2\n")
    assert len(chapters) == 1
    assert chapters[0].heading == "(whole text)"
    assert "Real prose." in chapters[0].text


def test_split_chapters_ignores_sub_headings():
    chapters = split_chapters("## Chapter 1\n\nbody\n\n### A scene break\n\nmore")
    assert len(chapters) == 1
    assert "### A scene break" in chapters[0].text


def test_word_count_counts_the_chapter_body():
    assert split_chapters(BOOK_MD)[0].word_count == 4


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("2", [2]),
        ("1-3", [1, 2, 3]),
        ("1,3", [1, 3]),
        ("3,1-2", [1, 2, 3]),
        ("2,2", [2]),
        (" 1 , 3 ", [1, 3]),
    ],
)
def test_parse_selection_accepts_numbers_and_ranges(spec, expected):
    assert parse_selection(spec, available=3) == expected


@pytest.mark.parametrize("spec", ["9", "0", "1-9", "abc", "3-1", ""])
def test_parse_selection_rejects_bad_input(spec):
    with pytest.raises(ValueError):
        parse_selection(spec, available=3)


def test_parse_selection_names_the_valid_range_in_the_error():
    with pytest.raises(ValueError, match="this book has 1-3"):
        parse_selection("7", available=3)


def test_select_defaults_to_the_first_chapter_only():
    selected = select(split_chapters(BOOK_MD), None)
    assert [c.heading for c in selected] == ["Letter 1"]


def test_select_applies_a_range():
    selected = select(split_chapters(BOOK_MD), "2-3")
    assert [c.heading for c in selected] == ["Chapter 1", "Chapter 2"]

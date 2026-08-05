"""Offline tests for the shared data types."""

from pathlib import Path

from storyillus.models import Book, Chapter, Page, PageResult, ScenePlan

BOOK = Book(id=1342, title="Pride and Prejudice", author="Austen, Jane", language="en")


def test_slug_is_filesystem_safe():
    assert BOOK.slug == "pride-and-prejudice"
    assert Book(1, "Alice's Adventures!", "Carroll", "en").slug == "alice-s-adventures"


def test_slug_falls_back_to_the_id_when_a_title_has_no_usable_characters():
    assert Book(84, "!!!", "Shelley", "en").slug == "pg84"


def test_book_str_is_the_line_the_picker_shows():
    assert str(BOOK) == "Pride and Prejudice - Austen, Jane (#1342, en)"


def test_chapter_word_count_and_display():
    chapter = Chapter(index=2, heading="Chapter 1", text="It is a truth universally acknowledged")
    assert chapter.word_count == 6
    assert "Chapter 1" in str(chapter)
    assert "6 words" in str(chapter)


def test_page_result_defaults_leave_room_for_an_unfinished_page():
    result = PageResult(
        page=Page(index=0, text="text"),
        plan=ScenePlan(summary="s", characters=[], setting="", mood="", key_visual=""),
        image_prompt="prompt",
        negative_prompt="",
        seed=7,
    )
    assert result.retrieved == []
    assert result.image_path is None
    assert result.error is None


def test_page_results_do_not_share_a_retrieved_list():
    def make():
        return PageResult(
            page=Page(index=0, text=""),
            plan=ScenePlan(summary="", characters=[], setting="", mood="", key_visual=""),
            image_prompt="",
            negative_prompt="",
            seed=0,
        )

    first, second = make(), make()
    first.retrieved.append(None)  # type: ignore[arg-type]
    assert second.retrieved == []


def test_image_path_accepts_a_path():
    result = PageResult(
        page=Page(index=1, text=""),
        plan=ScenePlan(summary="", characters=[], setting="", mood="", key_visual=""),
        image_prompt="",
        negative_prompt="",
        seed=0,
        image_path=Path("out/page-1.png"),
    )
    assert result.image_path.name == "page-1.png"

"""Offline tests for pagination."""

from storyillus.ingest.paginate import paginate


def paragraphs(*sizes: int) -> str:
    return "\n\n".join(" ".join(f"w{n}" for n in range(size)) for size in sizes)


def test_paragraphs_group_up_to_the_target():
    pages = paginate(paragraphs(100, 100, 100, 100), target_words=250)
    assert [len(page.text.split()) for page in pages] == [300, 100]


def test_pages_are_numbered_from_one():
    pages = paginate(paragraphs(50, 50, 50), target_words=50)
    assert [page.index for page in pages] == [1, 2, 3]


def test_a_paragraph_is_never_split():
    pages = paginate(paragraphs(500, 60), target_words=100)
    assert len(pages[0].text.split()) == 500
    assert len(pages) == 2


def test_paragraph_breaks_survive_inside_a_page():
    pages = paginate("one\n\ntwo", target_words=100)
    assert pages[0].text == "one\n\ntwo"


def test_blank_and_ragged_separators_do_not_make_empty_pages():
    pages = paginate("\n\n  \n\none\n   \n\n\ntwo\n\n\n", target_words=100)
    assert len(pages) == 1
    assert pages[0].text == "one\n\ntwo"


def test_empty_text_yields_no_pages():
    assert paginate("   \n\n  ") == []


def test_pages_land_near_the_target_rather_than_above_it():
    """Closing at the first overshoot would give 400-word pages for a 300-word target."""
    pages = paginate(paragraphs(*([160] * 12)), target_words=300)
    assert all(abs(len(page.text.split()) - 300) <= 60 for page in pages[:-1])


def test_a_chapter_sized_text_splits_into_a_handful_of_pages():
    """2,200 words is Frankenstein chapter 5 — the fixture Phase 1b is judged on."""
    pages = paginate(paragraphs(*([200] * 11)))
    assert 5 <= len(pages) <= 9

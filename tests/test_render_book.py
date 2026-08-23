"""Offline tests for assembling a canonicalized document into an HTML book and PDF."""

from PIL import Image

from storyillus.render.book import build_book_html, render_book

DOCUMENT = {
    "chapters": [
        {
            "index": 5,
            "heading": "Chapter 1",
            "pages": [
                {
                    "index": 1,
                    "plan": {
                        "summary": "Victor <reads> & studies.",
                        "characters": [],
                        "setting": "Geneva",
                        "mood": "warm",
                        "key_visual": "A young man reads by a window.",
                    },
                },
                {
                    "index": 2,
                    "plan": {
                        "summary": "No image exists for this page.",
                        "characters": [],
                        "setting": "Geneva",
                        "mood": "calm",
                        "key_visual": "k",
                    },
                },
            ],
        },
    ]
}


def _make_images_dir(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (8, 8), (200, 100, 50)).save(images_dir / "ch005-page001.png")
    return images_dir


def test_a_page_with_an_existing_image_gets_an_img_tag(tmp_path):
    images_dir = _make_images_dir(tmp_path)
    html = build_book_html(DOCUMENT, images_dir, tmp_path / "out" / "images")

    assert '<img src="images/ch005-page001.png"' in html


def test_a_page_missing_its_image_gets_no_img_tag(tmp_path):
    images_dir = _make_images_dir(tmp_path)  # only page 1 has an image
    html = build_book_html(DOCUMENT, images_dir, tmp_path / "out" / "images")

    sections = html.split('<section class="page">')
    assert "<img" not in sections[2]  # sections[0] is the head, [1] is page 1, [2] is page 2


def test_summary_text_is_html_escaped(tmp_path):
    images_dir = _make_images_dir(tmp_path)
    html = build_book_html(DOCUMENT, images_dir, tmp_path / "out" / "images")

    assert "Victor &lt;reads&gt; &amp; studies." in html
    assert "<reads>" not in html


def test_render_book_writes_html_and_a_real_pdf(tmp_path):
    images_dir = _make_images_dir(tmp_path)
    html_path, pdf_path = render_book(DOCUMENT, images_dir, tmp_path / "book")

    assert html_path.exists()
    assert pdf_path.exists()
    assert pdf_path.read_bytes()[:5] == b"%PDF-"

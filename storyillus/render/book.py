"""Assemble condensed pages and their illustrations into an HTML book, and export it to PDF.

Page text here is `ScenePlan.summary` — the condensed narrative beat, not the original prose.
`plan`'s JSON documents never carry the original page text forward (`PageResult.page.text` is
always empty, per `agent/graph.py`), and nothing before this stage needed it back.

weasyprint needs Pango/Cairo/GDK-pixbuf as system libraries, not just the Python package. On
macOS, Homebrew installs them under a prefix the dynamic linker doesn't search by default
(`/opt/homebrew` on Apple Silicon, `/usr/local` on Intel) — set `DYLD_LIBRARY_PATH` before
weasyprint is imported, or the import fails with a cryptic `dlopen` error naming `libpango`.
"""

import html
import os
import shutil
import sys
from pathlib import Path
from typing import Any

if sys.platform == "darwin":
    for _prefix in ("/opt/homebrew/lib", "/usr/local/lib"):
        if Path(_prefix).is_dir():
            os.environ.setdefault("DYLD_LIBRARY_PATH", _prefix)
            break

from weasyprint import HTML  # must follow the DYLD_LIBRARY_PATH shim above

PAGE_TEMPLATE = """<section class="page">
  <div class="text">
    <h2>{heading}</h2>
    <p>{summary}</p>
  </div>
  <div class="illustration">{image_tag}</div>
</section>"""

STYLE = """
body { font-family: Georgia, serif; margin: 0; color: #222; }
.page { display: flex; page-break-after: always; padding: 2em; gap: 2em; align-items: center; }
.text { flex: 1; }
.illustration { flex: 1; }
.illustration img { width: 100%; border-radius: 4px; }
h2 { font-size: 1.1em; color: #555; margin-top: 0; }
"""


def build_book_html(document: dict[str, Any], images_dir: Path, out_images_dir: Path) -> str:
    """One section per page: condensed text facing its illustration, in reading order.

    Illustrations are copied into `out_images_dir` (rather than referenced by absolute path)
    so the rendered book is one self-contained, movable folder. A page whose image is missing
    (never rendered, or a placeholder from a failed render) simply gets no `<img>` — the text
    side is never blocked on the art existing.
    """
    sections = []
    for chapter in document["chapters"]:
        for page in chapter["pages"]:
            plan = page["plan"]
            src = images_dir / f"ch{chapter['index']:03d}-page{page['index']:03d}.png"
            image_tag = ""
            if src.exists():
                out_images_dir.mkdir(parents=True, exist_ok=True)
                dest = out_images_dir / src.name
                shutil.copy(src, dest)
                image_tag = f'<img src="images/{dest.name}" alt="{html.escape(plan["key_visual"])}">'
            sections.append(
                PAGE_TEMPLATE.format(
                    heading=html.escape(chapter.get("heading", f"Chapter {chapter['index']}")),
                    summary=html.escape(plan["summary"]),
                    image_tag=image_tag,
                )
            )
    return f"<html><head><style>{STYLE}</style></head><body>{''.join(sections)}</body></html>"


def render_book(document: dict[str, Any], images_dir: Path, out_dir: Path) -> tuple[Path, Path]:
    """Writes `book.html` and `book.pdf` into `out_dir`. Returns their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    html_text = build_book_html(document, images_dir, out_dir / "images")

    html_path = out_dir / "book.html"
    html_path.write_text(html_text, encoding="utf-8")

    pdf_path = out_dir / "book.pdf"
    HTML(string=html_text, base_url=str(out_dir)).write_pdf(pdf_path)

    return html_path, pdf_path

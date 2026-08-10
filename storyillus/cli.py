"""Command line entry point."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from openai import APIStatusError

from storyillus import config
from storyillus.agent.condense import condense
from storyillus.ingest import chapters as chapters_mod
from storyillus.ingest import gutenberg
from storyillus.ingest.paginate import TARGET_WORDS, paginate
from storyillus.llm.openai_compat import OpenAICompatLLM
from storyillus.models import Book, Chapter


def fetch(args: argparse.Namespace) -> int:
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    gutenberg.ensure_catalog(cache_dir)

    matches = gutenberg.search(args.query, cache_dir, limit=args.limit)
    if not matches:
        print(f"No public-domain text found for {args.query!r}", file=sys.stderr)
        return 1

    book = matches[0] if args.first else _choose(matches)
    if book is None:
        return 1

    print(f"Fetching {book}")
    path = gutenberg.download(book, out_dir)
    words = len(path.read_text(encoding="utf-8").split())
    print(f"Wrote {path} ({words:,} words)")
    return 0


def chapters(args: argparse.Namespace) -> int:
    path = Path(args.book)
    if not path.exists():
        print(f"No such book: {path}", file=sys.stderr)
        return 1

    markdown = path.read_text(encoding="utf-8")
    found = chapters_mod.split_chapters(markdown)
    if not found:
        print(f"{path} has no readable text", file=sys.stderr)
        return 1

    # Listing shows everything by default; it is `run` that defaults to a single chapter.
    try:
        selected = found if args.select is None else chapters_mod.select(found, args.select)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1

    meta = chapters_mod.parse_frontmatter(markdown)
    if title := meta.get("title"):
        print(f"{title} — {meta.get('author', 'unknown')}")

    for chapter in selected:
        print(chapter)
    total = sum(chapter.word_count for chapter in selected)
    print(f"\n{len(selected)} of {len(found)} chapters, {total:,} words selected")
    return 0


def plan(args: argparse.Namespace) -> int:
    """Condense selected chapters into `ScenePlan`s and dump them as JSON."""
    try:
        selected = _load_chapters(Path(args.book), args.select)
        settings = config.load(args.config)
    except (OSError, ValueError, config.ConfigError) as error:
        print(error, file=sys.stderr)
        return 1

    llm = OpenAICompatLLM(settings.llm)
    print(f"Planning with {settings.llm.model_id}", file=sys.stderr)

    chapters_out = []
    for chapter in selected:
        pages = paginate(chapter.text, target_words=args.target_words)
        print(f"{chapter.heading}: {len(pages)} pages", file=sys.stderr)

        planned = []
        for page in pages:
            try:
                scene = condense(llm, page)
            except APIStatusError as error:
                print(f"\n{error.status_code} from {settings.llm.base_url}", file=sys.stderr)
                print(_explain(error), file=sys.stderr)
                return 1
            print(f"  page {page.index}: {scene.key_visual[:70]}", file=sys.stderr)
            planned.append(
                {"index": page.index, "words": len(page.text.split()), "plan": asdict(scene)}
            )

        chapters_out.append(
            {"index": chapter.index, "heading": chapter.heading, "pages": planned}
        )

    document = {
        "book": str(Path(args.book)),
        "config": settings.name,
        "model": settings.llm.model_id,
        "chapters": chapters_out,
    }
    text = json.dumps(document, indent=2, ensure_ascii=False)

    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"Wrote {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


def _load_chapters(path: Path, select: str | None) -> list[Chapter]:
    found = chapters_mod.split_chapters(path.read_text(encoding="utf-8"))
    if not found:
        raise ValueError(f"{path} has no readable text")
    return chapters_mod.select(found, select)


def _explain(error: APIStatusError) -> str:
    if error.status_code == 401:
        return "The token was rejected. Check hf_token in .env."
    if error.status_code == 403:
        return (
            "The token authenticated but lacks permission. An HF token needs the "
            "'Make calls to Inference Providers' scope: "
            "https://huggingface.co/settings/tokens"
        )
    if error.status_code == 404:
        return "The pinned model id does not exist on this endpoint. Check the config."
    return str(error)


def _choose(matches: list[Book]) -> Book | None:
    for index, book in enumerate(matches, start=1):
        print(f"  {index:2}. {book}")
    if not sys.stdin.isatty():
        print("Not a terminal: rerun with --first to take the top match", file=sys.stderr)
        return None

    reply = input(f"Select 1-{len(matches)} (blank to cancel): ").strip()
    if not reply.isdigit() or not 1 <= int(reply) <= len(matches):
        print("Cancelled", file=sys.stderr)
        return None
    return matches[int(reply) - 1]


def main() -> int:
    parser = argparse.ArgumentParser(prog="storyillus")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch", help="download a public-domain book as markdown")
    fetch_parser.add_argument("query", help="title or author to search for")
    fetch_parser.add_argument("--first", action="store_true", help="take the top match, no prompt")
    fetch_parser.add_argument("--limit", type=int, default=10, help="matches to show")
    fetch_parser.add_argument("--out-dir", default="books", help="where to write the markdown")
    fetch_parser.add_argument("--cache-dir", default=".cache", help="where to keep the catalog")
    fetch_parser.set_defaults(handler=fetch)

    chapters_parser = subparsers.add_parser("chapters", help="list a fetched book's chapters")
    chapters_parser.add_argument("book", help="path to a fetched .md file")
    chapters_parser.add_argument(
        "--select",
        metavar="SPEC",
        help="chapters to show, e.g. '3', '1-3', '1,4,7-9' (default: all)",
    )
    chapters_parser.set_defaults(handler=chapters)

    plan_parser = subparsers.add_parser("plan", help="condense chapters into scene plans")
    plan_parser.add_argument("book", help="path to a fetched .md file")
    plan_parser.add_argument(
        "--select",
        metavar="SPEC",
        help="chapters to plan, e.g. '5', '1-3' (default: the first chapter)",
    )
    plan_parser.add_argument("--config", default="configs/hosted.yaml", help="config to run with")
    plan_parser.add_argument(
        "--target-words", type=int, default=TARGET_WORDS, help="words per page"
    )
    plan_parser.add_argument("--out", help="write JSON here instead of stdout")
    plan_parser.set_defaults(handler=plan)

    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())

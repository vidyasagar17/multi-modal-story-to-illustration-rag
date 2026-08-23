"""Step 5 of the page loop: write new character/setting sheets, and this page's scene record.

Write-once, amend-rarely (plan.md's Consistency Strategy #1): a name already in the store keeps
its existing sheet untouched — no rewrite, no second LLM call. Amending a sheet on explicit
textual evidence (a costume change, an injury) is out of scope here; the done-when for this
phase only needs write-once population and name-keyed retrieval to work, and "is this evidence
of a real change" is its own judgment call that deserves its own evidence, not a guess.

Settings are deduped by exact name, the same as characters, not by a similarity threshold on
write — `retrieve.py`'s semantic search is still how a differently-phrased mention of the same
place gets found later; this is only about not writing a redundant sheet.
"""

from storyillus.llm.base import LLMBackend
from storyillus.memory.embedder import Embedder
from storyillus.memory.store import VectorStore
from storyillus.models import MemoryRecord, ScenePlan

SYSTEM = (
    "You write a concise, concrete visual description of one subject from a fiction excerpt, "
    "for an illustrator to draw consistently across many scenes. You invent no detail the text "
    "doesn't support or reasonably imply."
)

PROMPT = """Passage summary: {summary}
Key visual: {key_visual}

Write a single paragraph describing the visual appearance of "{name}". If this is a person,
cover physical build, clothing, and any distinctive features. If it is a setting, describe its
look instead: architecture, lighting, palette. Reply with the description alone, no preamble."""


def update_memory(
    llm: LLMBackend,
    store: VectorStore,
    embedder: Embedder,
    plan: ScenePlan,
    *,
    page_index: int,
) -> list[MemoryRecord]:
    """Write a sheet for every character/setting in `plan` not already known, plus a scene note."""
    written = []

    for name in plan.characters:
        if store.get_by_name(name) is None:
            sheet = _write_sheet(llm, store, embedder, kind="character", name=name, plan=plan, page_index=page_index)
            written.append(sheet)

    if plan.setting and store.get_by_name(plan.setting) is None:
        sheet = _write_sheet(
            llm, store, embedder, kind="setting", name=plan.setting, plan=plan, page_index=page_index
        )
        written.append(sheet)

    written.append(_write_scene(store, embedder, plan=plan, page_index=page_index))
    return written


def _write_sheet(
    llm: LLMBackend,
    store: VectorStore,
    embedder: Embedder,
    *,
    kind: str,
    name: str,
    plan: ScenePlan,
    page_index: int,
) -> MemoryRecord:
    description = llm.complete(
        PROMPT.format(summary=plan.summary, key_visual=plan.key_visual, name=name), system=SYSTEM
    ).strip()
    record = MemoryRecord(
        kind=kind,
        name=name,
        description=description,
        first_seen_page=page_index,
        embedding_text=f"{name}: {description}",
    )
    store.add(record, embedder.embed(record.embedding_text))
    return record


def _write_scene(store: VectorStore, embedder: Embedder, *, plan: ScenePlan, page_index: int) -> MemoryRecord:
    record = MemoryRecord(
        kind="scene",
        name=f"page {page_index}",
        description=plan.key_visual,
        first_seen_page=page_index,
        embedding_text=plan.key_visual,
    )
    store.add(record, embedder.embed(record.embedding_text))
    return record

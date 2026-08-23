"""One art-style string for the whole book, generated once and reused (plan.md's Consistency
Strategy #2: global and constant, never retrieved by RAG — prepended to every prompt instead).

Write-once through the same store/`get_by_name` mechanism `update_memory.py` already uses for
characters and settings, rather than a separate cache: `MemoryRecord.kind` already had "style"
in its `Literal` since Phase 0.
"""

from storyillus.llm.base import LLMBackend
from storyillus.memory.embedder import Embedder
from storyillus.memory.store import VectorStore
from storyillus.models import MemoryRecord, ScenePlan

STYLE_NAME = "style"

SYSTEM = "You are an art director choosing one consistent illustration style for an entire book, based on its tone."

PROMPT = """Here are the mood and a brief summary from every planned scene in this story:

{scenes}

Write one short paragraph describing a single consistent illustration style for the whole
book: medium (e.g. oil painting, ink wash, digital cel shading), palette, line quality, and
lighting. Reply with the style description alone, no preamble."""


def get_or_create_style_block(
    llm: LLMBackend, store: VectorStore, embedder: Embedder, plans: list[ScenePlan]
) -> str:
    """The book's style block — generated once from every plan's mood/summary, reused after."""
    if existing := store.get_by_name(STYLE_NAME):
        return existing.description

    scenes = "\n".join(f"- ({plan.mood}) {plan.summary}" for plan in plans)
    description = llm.complete(PROMPT.format(scenes=scenes), system=SYSTEM).strip()

    record = MemoryRecord(
        kind="style",
        name=STYLE_NAME,
        description=description,
        first_seen_page=0,
        embedding_text=description,
    )
    store.add(record, embedder.embed(description))
    return description

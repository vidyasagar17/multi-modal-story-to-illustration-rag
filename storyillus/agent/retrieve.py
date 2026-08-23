"""Step 2 of the page loop: build retrieval context for a page's `ScenePlan`.

Retrieval is scoped, not fuzzy (plan.md's Consistency Strategy #3): characters are fetched by
exact name match — semantic search alone drifts, name-keyed lookup is what actually holds a
character steady — while the setting, which the same place can be phrased differently across
pages, gets a semantic top-k search instead. The style block is deliberately absent here: it's
global and constant, never retrieved, and Phase 3's prompt builder is what prepends it.
"""

from storyillus.memory.embedder import Embedder
from storyillus.memory.store import VectorStore
from storyillus.models import MemoryRecord, ScenePlan


def retrieve(
    store: VectorStore,
    embedder: Embedder,
    plan: ScenePlan,
    *,
    previous_scene: MemoryRecord | None = None,
    setting_top_k: int = 2,
) -> list[MemoryRecord]:
    """Name-keyed characters, semantic top-k for the setting, plus continuity from last page."""
    records = [record for name in plan.characters if (record := store.get_by_name(name))]

    if plan.setting:
        records += store.query(embedder.embed(plan.setting), kind="setting", top_k=setting_top_k)

    if previous_scene is not None:
        records.append(previous_scene)

    return records

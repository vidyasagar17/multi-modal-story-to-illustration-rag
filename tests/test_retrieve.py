"""Offline tests for scoped retrieval: name-keyed characters, semantic setting, continuity."""

from storyillus.agent.retrieve import retrieve
from storyillus.memory.fake import FakeEmbedder
from storyillus.memory.store import LocalStore
from storyillus.models import MemoryRecord, ScenePlan

EMBEDDER = FakeEmbedder()

VICTOR = MemoryRecord(
    kind="character",
    name="Victor Frankenstein",
    description="Pale and dark-haired.",
    first_seen_page=1,
    embedding_text="Victor Frankenstein: pale, dark-haired.",
)
LAB = MemoryRecord(
    kind="setting",
    name="the laboratory",
    description="A candlelit attic room.",
    first_seen_page=1,
    embedding_text="the laboratory: candlelit attic room.",
)
PREVIOUS_SCENE = MemoryRecord(
    kind="scene",
    name="page 3",
    description="Victor recoils from the creature.",
    first_seen_page=3,
    embedding_text="Victor recoils from the creature.",
)


def _store() -> LocalStore:
    store = LocalStore()
    store.add(VICTOR, EMBEDDER.embed(VICTOR.embedding_text))
    store.add(LAB, EMBEDDER.embed(LAB.embedding_text))
    return store


def test_a_character_is_retrieved_by_exact_name_even_if_unrelated_by_embedding():
    plan = ScenePlan(
        summary="s", characters=["Victor Frankenstein"], setting="", mood="m", key_visual="k"
    )
    assert retrieve(_store(), EMBEDDER, plan) == [VICTOR]


def test_an_unknown_character_name_is_skipped_not_invented():
    plan = ScenePlan(summary="s", characters=["a stranger"], setting="", mood="m", key_visual="k")
    assert retrieve(_store(), EMBEDDER, plan) == []


def test_the_setting_is_retrieved_via_semantic_search():
    plan = ScenePlan(
        summary="s", characters=[], setting="the laboratory", mood="m", key_visual="k"
    )
    assert LAB in retrieve(_store(), EMBEDDER, plan)


def test_previous_scene_is_appended_only_when_given():
    plan = ScenePlan(summary="s", characters=[], setting="", mood="m", key_visual="k")

    assert retrieve(_store(), EMBEDDER, plan) == []
    assert retrieve(_store(), EMBEDDER, plan, previous_scene=PREVIOUS_SCENE) == [PREVIOUS_SCENE]

"""Offline tests for the hand-rolled local vector store."""

from storyillus.memory.store import LocalStore
from storyillus.models import MemoryRecord

VICTOR = MemoryRecord(
    kind="character",
    name="Victor Frankenstein",
    description="A pale, dark-haired young man in a plain traveling coat.",
    first_seen_page=1,
    embedding_text="Victor Frankenstein: pale, dark-haired.",
)
LAB = MemoryRecord(
    kind="setting",
    name="the laboratory",
    description="A cramped attic room lit by a single guttering candle.",
    first_seen_page=1,
    embedding_text="the laboratory: cramped, candlelit.",
)


def test_get_by_name_is_case_insensitive():
    store = LocalStore()
    store.add(VICTOR, [1.0, 0.0])

    assert store.get_by_name("victor frankenstein") is VICTOR
    assert store.get_by_name("VICTOR FRANKENSTEIN") is VICTOR


def test_get_by_name_returns_none_when_unknown():
    assert LocalStore().get_by_name("nobody") is None


def test_query_ranks_by_similarity_to_the_embedding():
    store = LocalStore()
    store.add(VICTOR, [1.0, 0.0])
    store.add(LAB, [0.0, 1.0])

    results = store.query([0.9, 0.1], top_k=1)
    assert results == [VICTOR]


def test_query_is_scoped_by_kind():
    store = LocalStore()
    store.add(VICTOR, [1.0, 0.0])
    store.add(LAB, [1.0, 0.0])  # same embedding, different kind

    assert store.query([1.0, 0.0], kind="setting", top_k=5) == [LAB]


def test_persist_and_load_round_trip(tmp_path):
    store = LocalStore()
    store.add(VICTOR, [1.0, 0.0])
    path = tmp_path / "memory.json"
    store.persist(path)

    reloaded = LocalStore.load(path)
    assert reloaded.get_by_name("Victor Frankenstein") == VICTOR
    assert reloaded.query([1.0, 0.0], top_k=1) == [VICTOR]


def test_loading_a_missing_path_gives_an_empty_store(tmp_path):
    store = LocalStore.load(tmp_path / "does-not-exist.json")
    assert store.get_by_name("anyone") is None

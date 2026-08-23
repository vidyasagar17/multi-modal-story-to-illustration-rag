"""Offline tests for the fake embedder that keeps the rest of the suite network-free."""

from storyillus.memory.fake import FakeEmbedder


def test_the_same_text_always_gets_the_same_vector():
    embedder = FakeEmbedder()
    assert embedder.embed("Victor Frankenstein") == embedder.embed("Victor Frankenstein")


def test_different_text_gets_a_different_vector():
    embedder = FakeEmbedder()
    assert embedder.embed("Victor Frankenstein") != embedder.embed("Elizabeth Lavenza")

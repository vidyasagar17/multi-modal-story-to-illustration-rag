"""Offline tests for seed discipline and the image-backend call."""

from storyillus.agent.illustrate import illustrate, page_seed
from storyillus.imagegen.fake import FakeImage


def test_no_characters_leaves_the_base_seed_unchanged():
    assert page_seed(1000, []) == 1000


def test_a_character_adds_a_stable_offset():
    seed = page_seed(1000, ["Victor Frankenstein"])
    assert seed != 1000
    assert page_seed(1000, ["Victor Frankenstein"]) == seed  # deterministic


def test_a_different_character_gets_a_different_offset():
    assert page_seed(1000, ["Victor Frankenstein"]) != page_seed(1000, ["Elizabeth Lavenza"])


def test_illustrate_passes_prompt_negative_and_seed_to_the_backend():
    backend = FakeImage()
    illustrate(backend, "a prompt", "a negative", 42)

    assert backend.calls == [("a prompt", "a negative", 42)]

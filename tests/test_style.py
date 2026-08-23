"""Offline tests for the write-once style block."""

from storyillus.agent.style import get_or_create_style_block
from storyillus.memory.fake import FakeEmbedder
from storyillus.memory.store import LocalStore
from storyillus.models import ScenePlan

PLANS = [
    ScenePlan(summary="A quiet morning in Geneva.", characters=[], setting="Geneva", mood="calm", key_visual="k"),
    ScenePlan(summary="A storm rolls over the lake.", characters=[], setting="the lake", mood="tense", key_visual="k"),
]


class RecordingLLM:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        self.calls += 1
        return "Oil painting, muted palette, soft candlelight."


def test_a_style_block_is_generated_once():
    llm = RecordingLLM()
    store = LocalStore()

    style = get_or_create_style_block(llm, store, FakeEmbedder(), PLANS)

    assert style == "Oil painting, muted palette, soft candlelight."
    assert llm.calls == 1


def test_a_second_call_reuses_the_cached_style_with_no_new_llm_call():
    llm = RecordingLLM()
    store = LocalStore()
    get_or_create_style_block(llm, store, FakeEmbedder(), PLANS)

    style = get_or_create_style_block(llm, store, FakeEmbedder(), PLANS)

    assert style == "Oil painting, muted palette, soft candlelight."
    assert llm.calls == 1

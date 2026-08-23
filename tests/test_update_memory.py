"""Offline tests for write-once entity extraction and the per-page scene record."""

import httpx
import pytest
from openai import APIStatusError

from storyillus.agent.update_memory import update_memory
from storyillus.memory.fake import FakeEmbedder
from storyillus.memory.store import LocalStore
from storyillus.models import ScenePlan

PLAN = ScenePlan(
    summary="Victor recoils as the creature opens its eye.",
    characters=["Victor Frankenstein"],
    setting="the laboratory",
    mood="dread",
    key_visual="A pale figure sits up beneath a sheet.",
)


class RecordingLLM:
    """Answers a fixed description and counts how many times it was asked."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        self.calls += 1
        return "A tall figure in a dark coat."


def test_a_new_character_gets_one_sheet_and_one_llm_call():
    llm = RecordingLLM()
    store = LocalStore()

    written = update_memory(llm, store, FakeEmbedder(), PLAN, page_index=1)

    sheets = [r for r in written if r.kind == "character"]
    assert len(sheets) == 1
    assert sheets[0].name == "Victor Frankenstein"
    assert llm.calls == 2  # one for the character sheet, one for the setting sheet


def test_a_known_character_triggers_no_second_call_or_duplicate():
    llm = RecordingLLM()
    store = LocalStore()
    update_memory(llm, store, FakeEmbedder(), PLAN, page_index=1)
    calls_after_first_page = llm.calls

    written = update_memory(llm, store, FakeEmbedder(), PLAN, page_index=2)

    assert llm.calls == calls_after_first_page  # no new sheet-writing calls
    assert [r for r in written if r.kind == "character"] == []
    assert [r for r in written if r.kind == "setting"] == []


def test_a_scene_record_is_always_written():
    store = LocalStore()
    update_memory(RecordingLLM(), store, FakeEmbedder(), PLAN, page_index=1)

    written = update_memory(RecordingLLM(), store, FakeEmbedder(), PLAN, page_index=2)
    scenes = [r for r in written if r.kind == "scene"]

    assert len(scenes) == 1
    assert scenes[0].name == "page 2"
    assert scenes[0].description == PLAN.key_visual


def test_an_api_error_is_not_swallowed():
    class FailingLLM:
        def complete(self, prompt: str, *, system: str | None = None) -> str:
            response = httpx.Response(402, request=httpx.Request("POST", "https://router.example/v1"))
            raise APIStatusError("out of credits", response=response, body=None)

    with pytest.raises(APIStatusError):
        update_memory(FailingLLM(), LocalStore(), FakeEmbedder(), PLAN, page_index=1)

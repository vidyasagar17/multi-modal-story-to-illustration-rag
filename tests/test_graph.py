"""Offline tests for the per-page orchestration: retrieve -> prompt -> illustrate -> memory."""

import httpx
import pytest
from openai import APIStatusError

from storyillus.agent.graph import illustrate_page
from storyillus.imagegen.fake import FakeImage
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
    def complete(self, prompt: str, *, system: str | None = None) -> str:
        return "A tall figure in a dark coat."


class FailingImage:
    def generate(self, prompt: str, *, negative: str = "", seed: int = 0):
        raise RuntimeError("the provider rejected this request")


def test_a_successful_page_saves_an_image_and_returns_a_scene_record(tmp_path):
    store = LocalStore()

    result, scene = illustrate_page(
        RecordingLLM(), FakeImage(), store, FakeEmbedder(), PLAN,
        chapter_index=1, page_index=1, style_block="gothic oil painting", base_seed=1000,
        previous_scene=None, out_dir=tmp_path,
    )

    assert result.error is None
    assert result.image_path is not None
    assert result.image_path.exists()
    assert scene.kind == "scene"
    assert scene.description == PLAN.key_visual
    assert store.get_by_name("Victor Frankenstein") is not None


def test_an_image_failure_becomes_a_placeholder_but_memory_still_updates(tmp_path):
    store = LocalStore()

    result, scene = illustrate_page(
        RecordingLLM(), FailingImage(), store, FakeEmbedder(), PLAN,
        chapter_index=1, page_index=1, style_block="gothic oil painting", base_seed=1000,
        previous_scene=None, out_dir=tmp_path,
    )

    assert result.image_path is None
    assert "rejected" in result.error
    assert store.get_by_name("Victor Frankenstein") is not None  # memory update still ran
    assert scene.kind == "scene"


def test_an_update_memory_api_error_is_not_swallowed(tmp_path):
    class FailingLLM:
        def complete(self, prompt: str, *, system: str | None = None) -> str:
            response = httpx.Response(402, request=httpx.Request("POST", "https://router.example/v1"))
            raise APIStatusError("out of credits", response=response, body=None)

    with pytest.raises(APIStatusError):
        illustrate_page(
            FailingLLM(), FakeImage(), LocalStore(), FakeEmbedder(), PLAN,
            chapter_index=1, page_index=1, style_block="gothic oil painting", base_seed=1000,
            previous_scene=None, out_dir=tmp_path,
        )

"""Offline tests for the per-page orchestration: retrieve -> prompt -> illustrate -> memory."""

from pathlib import Path

import httpx
import pytest
from openai import APIStatusError

from storyillus.agent.graph import _collect_references, illustrate_page
from storyillus.imagegen.fake import FakeImage
from storyillus.memory.fake import FakeEmbedder
from storyillus.memory.store import LocalStore
from storyillus.models import MemoryRecord, ScenePlan

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

    def generate_with_references(self, prompt: str, references, *, negative: str = "", seed: int = 0):
        return self.generate(prompt, negative=negative, seed=seed)


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


def test_a_first_page_has_nothing_to_reference_yet(tmp_path):
    backend = FakeImage()
    illustrate_page(
        RecordingLLM(), backend, LocalStore(), FakeEmbedder(), PLAN,
        chapter_index=1, page_index=1, style_block="gothic oil painting", base_seed=1000,
        previous_scene=None, out_dir=tmp_path,
    )

    assert backend.reference_calls == []  # nothing known yet: plain text-to-image


def test_a_recurring_character_conditions_on_their_own_earlier_render(tmp_path):
    store = LocalStore()
    backend = FakeImage()
    _, scene1 = illustrate_page(
        RecordingLLM(), backend, store, FakeEmbedder(), PLAN,
        chapter_index=1, page_index=1, style_block="gothic oil painting", base_seed=1000,
        previous_scene=None, out_dir=tmp_path,
    )

    illustrate_page(
        RecordingLLM(), backend, store, FakeEmbedder(), PLAN,
        chapter_index=1, page_index=2, style_block="gothic oil painting", base_seed=1000,
        previous_scene=scene1, out_dir=tmp_path,
    )

    assert len(backend.reference_calls) == 1
    _, _, _, references = backend.reference_calls[0]
    victor_reference = Path(store.get_by_name("Victor Frankenstein").reference_image)
    assert victor_reference in references


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


def _record(kind, name, reference_image=None):
    return MemoryRecord(
        kind=kind, name=name, description="d", first_seen_page=1, embedding_text="d",
        reference_image=reference_image,
    )


def test_collect_references_is_empty_when_nothing_has_a_reference_image():
    context = [_record("character", "Victor"), _record("setting", "the lab")]
    assert _collect_references(context, previous_scene=None) == []


def test_collect_references_orders_character_then_scene_then_setting():
    context = [
        _record("setting", "the lab", reference_image="/tmp/lab.png"),
        _record("character", "Victor", reference_image="/tmp/victor.png"),
    ]
    scene = _record("scene", "page 1", reference_image="/tmp/page1.png")

    refs = _collect_references(context, previous_scene=scene)

    assert refs == [Path("/tmp/victor.png"), Path("/tmp/page1.png"), Path("/tmp/lab.png")]


def test_collect_references_deduplicates_and_caps_at_three():
    context = [
        _record("character", "Victor", reference_image="/tmp/same.png"),
        _record("setting", "the lab", reference_image="/tmp/same.png"),
    ]
    scene = _record("scene", "page 1", reference_image="/tmp/page1.png")

    refs = _collect_references(context, previous_scene=scene)

    assert refs == [Path("/tmp/same.png"), Path("/tmp/page1.png")]

"""Offline tests for illustrating a whole document: continuity and fail-soft-at-the-run-level."""

import httpx
import pytest
from openai import APIStatusError

from storyillus.cli import _build_manifest, _illustrate_document, _parse_pages
from storyillus.config import Config, ImageConfig, LLMConfig
from storyillus.imagegen.fake import FakeImage
from storyillus.memory.fake import FakeEmbedder
from storyillus.memory.store import LocalStore

DOCUMENT = {
    "chapters": [
        {
            "index": 5,
            "pages": [
                {
                    "index": 1,
                    "plan": {
                        "summary": "Victor is introduced.",
                        "characters": ["Victor Frankenstein"],
                        "setting": "Geneva",
                        "mood": "warm",
                        "key_visual": "A young man reads by a window.",
                    },
                },
            ],
        },
        {
            "index": 6,
            "pages": [
                {
                    "index": 1,
                    "plan": {
                        "summary": "Victor and Elizabeth walk together.",
                        "characters": ["Victor Frankenstein", "Elizabeth Lavenza"],
                        "setting": "Geneva",
                        "mood": "tender",
                        "key_visual": "Two figures walk along a lake shore.",
                    },
                },
            ],
        },
    ]
}


class RecordingLLM:
    def complete(self, prompt: str, *, system: str | None = None) -> str:
        return "A tall figure."


def test_continuity_threads_across_chapters(tmp_path):
    store = LocalStore()
    results, stopped = _illustrate_document(
        DOCUMENT, RecordingLLM(), FakeImage(), store, FakeEmbedder(), "a style",
        base_seed=1000, out_dir=tmp_path,
    )

    assert stopped is None
    assert len(results) == 2
    # chapter 6's page retrieved chapter 5's scene as continuity context
    ch6_context_kinds = [record.kind for record in results[1].retrieved]
    assert "scene" in ch6_context_kinds


def test_pages_sharing_the_same_index_across_chapters_do_not_collide_on_disk(tmp_path):
    """Both chapters' first page is index 1 — the saved files must not overwrite each other."""
    results, _ = _illustrate_document(
        DOCUMENT, RecordingLLM(), FakeImage(), LocalStore(), FakeEmbedder(), "a style",
        base_seed=1000, out_dir=tmp_path,
    )

    paths = [result.image_path for result in results]
    assert len(set(paths)) == len(paths) == 2
    assert all(path.exists() for path in paths)


def test_a_mid_run_api_error_stops_but_keeps_earlier_results(tmp_path):
    class FailingAfterOneLLM:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, prompt: str, *, system: str | None = None) -> str:
            self.calls += 1
            if self.calls > 2:  # chapter 5 writes 2 sheets (Victor, Geneva); the 3rd call is Elizabeth's
                response = httpx.Response(402, request=httpx.Request("POST", "https://router.example/v1"))
                raise APIStatusError("out of credits", response=response, body=None)
            return "A tall figure."

    store = LocalStore()
    results, stopped = _illustrate_document(
        DOCUMENT, FailingAfterOneLLM(), FakeImage(), store, FakeEmbedder(), "a style",
        base_seed=1000, out_dir=tmp_path,
    )

    assert stopped is not None
    assert len(results) == 1  # chapter 5 finished before chapter 6 hit the failure


THREE_PAGE_CHAPTER = {
    "chapters": [
        {
            "index": 1,
            "pages": [
                {"index": i, "plan": {"summary": f"page {i}", "characters": [], "setting": "s", "mood": "m", "key_visual": "k"}}
                for i in (1, 2, 3)
            ],
        }
    ]
}


def test_pages_keeps_only_the_named_indices(tmp_path):
    results, _ = _illustrate_document(
        THREE_PAGE_CHAPTER, RecordingLLM(), FakeImage(), LocalStore(), FakeEmbedder(), "a style",
        base_seed=1000, out_dir=tmp_path, pages={1, 3},
    )

    assert [result.page.index for result in results] == [1, 3]


def test_pages_excludes_indices_not_named(tmp_path):
    results, _ = _illustrate_document(
        DOCUMENT, RecordingLLM(), FakeImage(), LocalStore(), FakeEmbedder(), "a style",
        base_seed=1000, out_dir=tmp_path, pages={99},
    )

    assert results == []


@pytest.mark.parametrize(
    ("spec", "expected"), [("3", {3}), ("1-3", {1, 2, 3}), ("1,4,7-9", {1, 4, 7, 8, 9})]
)
def test_parse_pages_handles_singles_and_ranges(spec, expected):
    assert _parse_pages(spec) == expected


def test_parse_pages_rejects_garbage():
    with pytest.raises(ValueError, match="not a page number"):
        _parse_pages("abc")


def test_build_manifest_captures_model_ids_seed_and_prompts(tmp_path):
    results, _ = _illustrate_document(
        DOCUMENT, RecordingLLM(), FakeImage(), LocalStore(), FakeEmbedder(), "a style",
        base_seed=1000, out_dir=tmp_path,
    )
    settings = Config(
        name="test",
        llm=LLMConfig(base_url="http://x", model_id="test-llm"),
        image=ImageConfig(backend="huggingface", model_id="test-image"),
    )

    manifest = _build_manifest(results, settings)

    assert len(manifest) == 2
    assert manifest[0]["llm_model"] == "test-llm"
    assert manifest[0]["image_model"] == "test-image"
    assert manifest[0]["seed"] == results[0].seed
    assert manifest[0]["image_prompt"] == results[0].image_prompt
    assert manifest[0]["image_path"] is not None

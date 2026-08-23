"""Offline tests for populating the memory store across a whole `plan` document."""

from storyillus.cli import _remember_document
from storyillus.memory.fake import FakeEmbedder
from storyillus.memory.store import LocalStore

DOCUMENT = {
    "chapters": [
        {
            "index": 5,
            "heading": "Chapter 1",
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
            "heading": "Chapter 2",
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


def test_a_name_written_in_an_earlier_chapter_is_known_in_a_later_one():
    store = LocalStore()
    written, stopped = _remember_document(DOCUMENT, RecordingLLM(), store, FakeEmbedder())

    assert stopped is None
    assert store.get_by_name("Victor Frankenstein") is not None
    assert store.get_by_name("Elizabeth Lavenza") is not None

    victor_sheets = [r for r in written if r.kind == "character" and r.name == "Victor Frankenstein"]
    assert len(victor_sheets) == 1  # written once in chapter 1, not again in chapter 2

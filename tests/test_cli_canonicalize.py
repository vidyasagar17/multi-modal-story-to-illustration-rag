"""Offline tests for canonicalizing a `plan` document across chapters."""

import json

from storyillus.cli import _canonicalize_document
from storyillus.llm.fake import FakeLLM

DOCUMENT = {
    "book": "books/frankenstein.md",
    "chapters": [
        {
            "index": 5,
            "heading": "Chapter 1",
            "pages": [
                {
                    "index": 1,
                    "words": 100,
                    "plan": {
                        "summary": "Victor Frankenstein recounts his father's kindness.",
                        "characters": ["Victor Frankenstein", "Victor's father"],
                        "setting": "Geneva",
                        "mood": "warm",
                        "key_visual": "A man embraces his son.",
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
                    "words": 90,
                    "plan": {
                        "summary": "My father cares for my mother.",
                        "characters": ["my father", "my mother"],
                        "setting": "Italy",
                        "mood": "tender",
                        "key_visual": "A couple walks together.",
                    },
                },
            ],
        },
    ],
}

GROUPED = json.dumps(
    {
        "people": [
            {"canonical": "Victor Frankenstein", "aliases": ["Victor Frankenstein"]},
            {"canonical": "Victor's father", "aliases": ["Victor's father", "my father"]},
            {"canonical": "my mother", "aliases": ["my mother"]},
        ]
    }
)


def test_characters_are_rewritten_across_every_chapter():
    document = json.loads(json.dumps(DOCUMENT))  # deep copy
    _canonicalize_document(document, FakeLLM(default=GROUPED))

    ch1_page = document["chapters"][0]["pages"][0]
    ch2_page = document["chapters"][1]["pages"][0]
    assert ch1_page["plan"]["characters"] == ["Victor Frankenstein", "Victor's father"]
    assert ch2_page["plan"]["characters"] == ["Victor's father", "my mother"]


def test_the_mapping_is_returned_and_recorded_on_the_document():
    document = json.loads(json.dumps(DOCUMENT))
    mapping = _canonicalize_document(document, FakeLLM(default=GROUPED))

    assert mapping["my father"] == "Victor's father"
    assert document["canonical_names"] == mapping

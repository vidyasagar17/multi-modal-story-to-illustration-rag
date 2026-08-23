"""Offline tests for the name-canonicalization step."""

import json

from storyillus.agent.canonicalize import apply_canonical_names, canonicalize_names
from storyillus.llm.fake import FakeLLM
from storyillus.models import ScenePlan

PLANS = [
    ScenePlan(
        summary="Victor Frankenstein recounts how his father helped his old friend Beaufort.",
        characters=["Beaufort", "Victor Frankenstein", "Victor's father"],
        setting="Lucerne",
        mood="melancholy",
        key_visual="A man walks through misty streets.",
    ),
    ScenePlan(
        summary="My father finds Beaufort dead and comforts his daughter Caroline.",
        characters=["Beaufort", "Caroline Beaufort", "my father"],
        setting="a mean street",
        mood="grief",
        key_visual="A woman weeps beside a coffin.",
    ),
]

GROUPED = json.dumps(
    {
        "people": [
            {"canonical": "Victor Frankenstein", "aliases": ["Victor Frankenstein"]},
            {"canonical": "Beaufort", "aliases": ["Beaufort"]},
            {"canonical": "Victor's father", "aliases": ["Victor's father", "my father"]},
            {"canonical": "Caroline Beaufort", "aliases": ["Caroline Beaufort"]},
        ]
    }
)


def test_aliases_and_relational_references_collapse_to_one_name():
    mapping = canonicalize_names(FakeLLM(default=GROUPED), PLANS)

    assert mapping["Victor's father"] == "Victor's father"
    assert mapping["my father"] == "Victor's father"
    assert mapping["Beaufort"] == "Beaufort"


def test_a_name_the_reply_omits_falls_back_to_itself():
    reply = json.dumps({"people": [{"canonical": "Beaufort", "aliases": ["Beaufort"]}]})
    mapping = canonicalize_names(FakeLLM(default=reply), PLANS)

    assert mapping["Victor Frankenstein"] == "Victor Frankenstein"
    assert mapping["my father"] == "my father"


def test_an_alias_that_is_not_a_known_raw_name_is_ignored():
    reply = json.dumps(
        {"people": [{"canonical": "Beaufort", "aliases": ["Beaufort", "the innkeeper"]}]}
    )
    mapping = canonicalize_names(FakeLLM(default=reply), PLANS)

    assert "the innkeeper" not in mapping


def test_malformed_json_falls_back_to_an_identity_mapping():
    llm = FakeLLM(replies=["no.", "still no.", "nope."])
    mapping = canonicalize_names(llm, PLANS)

    assert mapping == {name: name for name in mapping}
    assert set(mapping) == {"Beaufort", "Victor Frankenstein", "Victor's father", "Caroline Beaufort", "my father"}


def test_empty_plans_return_an_empty_mapping_with_no_llm_call():
    llm = FakeLLM(default=GROUPED)
    assert canonicalize_names(llm, []) == {}
    assert llm.prompts == []


def test_apply_canonical_names_rewrites_and_deduplicates_preserving_order():
    mapping = {
        "Beaufort": "Beaufort",
        "Victor Frankenstein": "Victor Frankenstein",
        "Victor's father": "Victor's father",
        "Caroline Beaufort": "Caroline Beaufort",
        "my father": "Victor's father",
    }
    rewritten = apply_canonical_names(PLANS, mapping)

    assert rewritten[0].characters == ["Beaufort", "Victor Frankenstein", "Victor's father"]
    assert rewritten[1].characters == ["Beaufort", "Caroline Beaufort", "Victor's father"]

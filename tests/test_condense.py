"""Offline tests for the condense step."""

import json

from storyillus.agent.condense import condense
from storyillus.llm.fake import FakeLLM
from storyillus.models import Page

PAGE = Page(index=1, text="It was on a dreary night of November that I beheld my man completed.")

FULL = json.dumps(
    {
        "summary": "Victor watches the creature open its eye.",
        "characters": ["Victor Frankenstein", "the creature"],
        "setting": "a candlelit laboratory",
        "mood": "dread",
        "key_visual": "A pale figure sitting up beneath a sheet as the candle gutters.",
    }
)


def test_a_well_formed_reply_becomes_a_scene_plan():
    plan = condense(FakeLLM(default=FULL), PAGE)
    assert plan.characters == ["Victor Frankenstein", "the creature"]
    assert plan.setting == "a candlelit laboratory"
    assert plan.mood == "dread"
    assert plan.key_visual.startswith("A pale figure")


def test_the_page_text_reaches_the_model():
    llm = FakeLLM(default=FULL)
    condense(llm, PAGE)
    assert PAGE.text in llm.prompts[0]


def test_missing_fields_become_empty_rather_than_absent():
    plan = condense(FakeLLM(default='{"summary": "Something happens."}'), PAGE)
    assert plan.summary == "Something happens."
    assert plan.characters == []
    assert plan.setting == ""


def test_key_visual_falls_back_to_the_summary():
    plan = condense(FakeLLM(default='{"summary": "Something happens."}'), PAGE)
    assert plan.key_visual == "Something happens."


def test_characters_sent_as_a_string_are_split():
    """Models sometimes answer with prose where the schema asked for a list."""
    reply = '{"summary": "s", "characters": "Victor, the creature"}'
    assert condense(FakeLLM(default=reply), PAGE).characters == ["Victor", "the creature"]


def test_a_null_character_list_is_not_an_error():
    assert condense(FakeLLM(default='{"summary": "s", "characters": null}'), PAGE).characters == []


def test_unparseable_output_falls_back_to_a_summary_only_plan():
    llm = FakeLLM(replies=["no.", "still no.", "nope."], default="Victor recoils from the bed.")
    plan = condense(llm, PAGE)

    assert plan.summary == "Victor recoils from the bed."
    assert plan.key_visual == plan.summary
    assert plan.characters == []
    assert len(llm.prompts) == 4  # three JSON attempts, then the prose fallback

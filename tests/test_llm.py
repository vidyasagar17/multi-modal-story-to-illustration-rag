"""Offline tests for the LLM seam and the JSON wrapper."""

import pytest

from storyillus.llm.base import LLMJSONError, complete_json
from storyillus.llm.fake import FakeLLM

PLAN = '{"summary": "a lab at night", "characters": ["Victor"]}'


def test_fake_returns_replies_in_order_then_the_default():
    llm = FakeLLM(replies=["first", "second"], default="rest")
    assert [llm.complete("a"), llm.complete("b"), llm.complete("c")] == [
        "first",
        "second",
        "rest",
    ]
    assert llm.prompts == ["a", "b", "c"]


def test_parses_a_bare_object():
    assert complete_json(FakeLLM(default=PLAN), "go")["characters"] == ["Victor"]


def test_parses_a_fenced_object():
    llm = FakeLLM(default=f"Here you go:\n```json\n{PLAN}\n```\nHope that helps.")
    assert complete_json(llm, "go")["summary"] == "a lab at night"


def test_parses_an_object_wrapped_in_prose():
    llm = FakeLLM(default=f"Sure. {PLAN} Let me know if you want changes.")
    assert complete_json(llm, "go")["summary"] == "a lab at night"


def test_a_bad_reply_is_retried_and_the_retry_says_what_was_wrong():
    llm = FakeLLM(replies=["I'd rather not.", PLAN])
    assert complete_json(llm, "go")["summary"] == "a lab at night"

    assert len(llm.prompts) == 2
    assert llm.prompts[0] == "go"
    assert "not valid JSON" in llm.prompts[1]
    assert llm.prompts[1].startswith("go")


def test_giving_up_raises_after_exactly_the_allowed_attempts():
    llm = FakeLLM(default="no json here")
    with pytest.raises(LLMJSONError, match="2 attempts"):
        complete_json(llm, "go", attempts=2)
    assert len(llm.prompts) == 2


def test_a_json_array_is_not_an_object():
    llm = FakeLLM(default='["Victor", "Elizabeth"]')
    with pytest.raises(LLMJSONError, match="no JSON object"):
        complete_json(llm, "go", attempts=1)


def test_truncated_json_is_rejected():
    llm = FakeLLM(default='{"summary": "a lab at nig')
    with pytest.raises(LLMJSONError):
        complete_json(llm, "go", attempts=1)

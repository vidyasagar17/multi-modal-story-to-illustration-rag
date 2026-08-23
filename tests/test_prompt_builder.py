"""Offline tests for fusing the style block, retrieved context, and key visual into a prompt."""

from storyillus.agent.prompt_builder import DEFAULT_NEGATIVE, build_prompt
from storyillus.models import MemoryRecord, ScenePlan

PLAN = ScenePlan(
    summary="s", characters=["Victor"], setting="the laboratory", mood="dread",
    key_visual="A pale figure sits up beneath a sheet.",
)
VICTOR = MemoryRecord(
    kind="character", name="Victor", description="Pale, dark-haired.", first_seen_page=1,
    embedding_text="Victor: pale, dark-haired.",
)
LAB = MemoryRecord(
    kind="setting", name="the laboratory", description="A candlelit attic room.",
    first_seen_page=1, embedding_text="the laboratory: candlelit attic room.",
)


def test_the_prompt_is_style_then_context_then_key_visual_comma_joined():
    prompt, _ = build_prompt("gothic oil painting", [VICTOR, LAB], PLAN)

    assert prompt == (
        "gothic oil painting, Pale, dark-haired., A candlelit attic room., "
        "A pale figure sits up beneath a sheet."
    )


def test_the_negative_prompt_is_the_default_constant():
    _, negative = build_prompt("gothic oil painting", [], PLAN)
    assert negative == DEFAULT_NEGATIVE


def test_an_empty_style_block_or_context_does_not_leave_stray_commas():
    prompt, _ = build_prompt("", [], PLAN)
    assert prompt == "A pale figure sits up beneath a sheet."

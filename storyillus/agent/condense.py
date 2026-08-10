"""Step 1 of the page loop: story text in, a `ScenePlan` out.

The schema goes in the prompt rather than in a `response_format` parameter — support for
provider-native structured output varies across the providers behind the HF router, and this
is the portable path.
"""

from typing import Any

from storyillus.llm.base import LLMBackend, LLMJSONError, complete_json
from storyillus.models import Page, ScenePlan

SYSTEM = (
    "You read fiction and describe what an illustrator should draw. "
    "You are concrete and visual. You never invent characters or places the passage "
    "does not contain."
)

PROMPT = """Read this passage and describe the single moment most worth illustrating.

Reply with one JSON object and nothing else:

{{
  "summary": "2-4 vivid sentences describing what happens in the passage",
  "characters": ["each person present, named exactly as the passage names them"],
  "setting": "where this happens, as a phrase",
  "mood": "the emotional register, in a few words",
  "key_visual": "the one image an illustrator would draw, in a single sentence"
}}

PASSAGE:
{text}"""

FALLBACK_PROMPT = """Describe, in 2-4 vivid sentences, the single moment in this passage
that is most worth illustrating. Reply with the description alone.

PASSAGE:
{text}"""


def condense(llm: LLMBackend, page: Page, *, attempts: int = 3) -> ScenePlan:
    """Turn one page into a `ScenePlan`.

    A model that will not produce JSON after `attempts` still gets the page illustrated,
    from a plain-prose summary instead — worse art, not a dead run.
    """
    try:
        data = complete_json(llm, PROMPT.format(text=page.text), system=SYSTEM, attempts=attempts)
    except LLMJSONError:
        return _summary_only(llm, page)
    return _to_plan(data)


def _summary_only(llm: LLMBackend, page: Page) -> ScenePlan:
    summary = llm.complete(FALLBACK_PROMPT.format(text=page.text), system=SYSTEM).strip()
    return ScenePlan(summary=summary, characters=[], setting="", mood="", key_visual=summary)


def _to_plan(data: dict[str, Any]) -> ScenePlan:
    return ScenePlan(
        summary=_text(data.get("summary")),
        characters=_names(data.get("characters")),
        setting=_text(data.get("setting")),
        mood=_text(data.get("mood")),
        key_visual=_text(data.get("key_visual")) or _text(data.get("summary")),
    )


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _names(value: Any) -> list[str]:
    """Accept the comma-separated string models sometimes send instead of a list."""
    items = value.split(",") if isinstance(value, str) else value or []
    return [name for item in items if (name := _text(item))]

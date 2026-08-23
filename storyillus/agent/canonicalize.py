"""Collapse every raw character reference into one canonical name per person.

`condense.py` names characters however the passage happens to: "Victor", "Victor
Frankenstein", "I", "my father", "Victor's father". Name-keyed retrieval — the consistency
strategy's core mechanism — needs one name per person, so this runs once over every
`ScenePlan` a run produced, using each name's first-seen summary as the context a
reference like "my father" needs to resolve.
"""

from dataclasses import replace
from typing import Any

from storyillus.llm.base import LLMBackend, LLMJSONError, complete_json
from storyillus.models import ScenePlan

SYSTEM = (
    "You track character identity across a fiction excerpt. You group every name, alias, "
    "first-person reference, and relational reference (\"my father\", \"his sister\") that "
    "refers to the same person, using the surrounding summary to resolve who \"I\" and "
    "relational terms mean. You invent no new people."
)

PROMPT = """Here is every character reference mentioned across these scenes, each with the
scene summary it came from for context:

{mentions}

Group these into one entry per distinct person. Pick as canonical name the fullest proper
name ever used for that person (e.g. "Victor Frankenstein" over "Victor" or "I"); if no
proper name is ever used, pick the clearest descriptive phrase (e.g. "Victor's father")
rather than inventing one. Every reference listed above must appear in exactly one group's
"aliases".

Reply with one JSON object and nothing else:

{{
  "people": [
    {{"canonical": "...", "aliases": ["...", "..."]}}
  ]
}}"""


def canonicalize_names(
    llm: LLMBackend, plans: list[ScenePlan], *, attempts: int = 3
) -> dict[str, str]:
    """Map every raw character name across `plans` to one canonical name per person."""
    mentions = _summaries_per_name(plans)
    if not mentions:
        return {}

    listing = "\n".join(
        f'- "{name}" — {" / ".join(summaries)}' for name, summaries in mentions.items()
    )
    try:
        data = complete_json(llm, PROMPT.format(mentions=listing), system=SYSTEM, attempts=attempts)
    except LLMJSONError:
        return {name: name for name in mentions}

    return _to_mapping(data, known=set(mentions))


def apply_canonical_names(plans: list[ScenePlan], mapping: dict[str, str]) -> list[ScenePlan]:
    """Rewrite each plan's `characters` through `mapping`, de-duplicated, order preserved."""
    result = []
    for plan in plans:
        names = list(dict.fromkeys(mapping.get(name, name) for name in plan.characters))
        result.append(replace(plan, characters=names))
    return result


def _summaries_per_name(plans: list[ScenePlan]) -> dict[str, list[str]]:
    """Every distinct summary each raw name appeared alongside, in first-seen order.

    A name's *first* summary is sometimes about someone else entirely — a name can be
    listed as present in a scene the summary doesn't actually describe. Every summary a
    name appears in gives the model an actual chance at a description to anchor on.
    """
    seen: dict[str, list[str]] = {}
    for plan in plans:
        for name in plan.characters:
            summaries = seen.setdefault(name, [])
            if plan.summary not in summaries:
                summaries.append(plan.summary)
    return seen


def _to_mapping(data: Any, *, known: set[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for group in data.get("people", []) if isinstance(data, dict) else []:
        canonical = str(group.get("canonical", "")).strip()
        if not canonical:
            continue
        for alias in group.get("aliases", []) or []:
            alias = str(alias).strip()
            if alias in known:
                mapping[alias] = canonical

    # Fail soft: a name the model dropped maps to itself rather than disappearing.
    for name in known:
        mapping.setdefault(name, name)
    return mapping

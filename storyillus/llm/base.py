"""The text-generation seam: one protocol, plus the JSON helper every stage uses."""

import json
import re
from typing import Any, Protocol

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class LLMBackend(Protocol):
    """A chat model behind a single call. Implementations are stateless and reusable."""

    def complete(self, prompt: str, *, system: str | None = None) -> str: ...


class LLMJSONError(RuntimeError):
    """The model would not produce parseable JSON."""


def complete_json(
    llm: LLMBackend,
    prompt: str,
    *,
    system: str | None = None,
    attempts: int = 3,
) -> dict[str, Any]:
    """Ask for a JSON object and parse it, re-prompting when the reply is unparseable.

    Provider-native structured output is not portable across the providers behind the HF
    router, so the schema goes in the caller's prompt and adherence is enforced here.
    Callers decide what a total failure means; this raises rather than inventing a value.
    """
    ask = prompt
    for attempt in range(1, attempts + 1):
        reply = llm.complete(ask, system=system)
        try:
            return _parse(reply)
        except ValueError as error:
            if attempt == attempts:
                raise LLMJSONError(f"no valid JSON in {attempts} attempts: {error}") from error
            ask = (
                f"{prompt}\n\nYour previous reply was not valid JSON ({error}). "
                f"Reply with the JSON object only — no prose, no code fences."
            )
    raise AssertionError("unreachable")


def _parse(reply: str) -> dict[str, Any]:
    """Pull one JSON object out of a reply that may be fenced or wrapped in prose."""
    if match := _FENCE.search(reply):
        reply = match.group(1)

    # Slicing between the outermost braces is also what rejects a bare array or string:
    # whatever parses out of a slice that starts with `{` is an object.
    start, end = reply.find("{"), reply.rfind("}")
    if start == -1 or end < start:
        raise ValueError("no JSON object in the reply")

    return json.loads(reply[start : end + 1])  # JSONDecodeError is a ValueError

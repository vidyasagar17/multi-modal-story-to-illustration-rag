"""A scripted LLM, so the test suite needs no weights, no token, and no network."""

from dataclasses import dataclass, field


@dataclass
class FakeLLM:
    """Returns `replies` in order, then `default` forever, and records what it was asked."""

    replies: list[str] = field(default_factory=list)
    default: str = ""
    prompts: list[str] = field(default_factory=list)

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else self.default

"""The embedding seam: turns text into a vector for the memory store."""

from typing import Protocol


class Embedder(Protocol):
    """Deterministic in the sense that matters: same text, same vector, every call."""

    def embed(self, text: str) -> list[float]: ...

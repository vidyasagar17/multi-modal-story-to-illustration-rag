"""The image-generation seam."""

from pathlib import Path
from typing import Protocol

from PIL.Image import Image


class ImageBackend(Protocol):
    """Text to image. The same prompt and seed must produce the same image."""

    def generate(self, prompt: str, *, negative: str = "", seed: int = 0) -> Image: ...

    def generate_with_references(
        self, prompt: str, references: list[Path], *, negative: str = "", seed: int = 0
    ) -> Image:
        """Like `generate()`, but conditioned on up to a few existing images (a character's
        own earlier render, the previous page) so their visual identity carries forward
        instead of being re-imagined from text alone. An empty `references` list is exactly
        `generate()` — implementations without this capability should just fall back to it."""
        ...

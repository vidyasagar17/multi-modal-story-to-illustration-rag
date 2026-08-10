"""The image-generation seam."""

from typing import Protocol

from PIL.Image import Image


class ImageBackend(Protocol):
    """Text to image. The same prompt and seed must produce the same image."""

    def generate(self, prompt: str, *, negative: str = "", seed: int = 0) -> Image: ...

"""A stand-in for a diffusion model: a flat colour derived from the seed."""

from dataclasses import dataclass, field
from random import Random

from PIL import Image


@dataclass
class FakeImage:
    """Same seed, same colour — enough to exercise seed discipline without weights."""

    size: tuple[int, int] = (64, 64)
    calls: list[tuple[str, str, int]] = field(default_factory=list)

    def generate(self, prompt: str, *, negative: str = "", seed: int = 0) -> Image.Image:
        self.calls.append((prompt, negative, seed))
        rng = Random(seed)
        colour = (rng.randrange(256), rng.randrange(256), rng.randrange(256))
        return Image.new("RGB", self.size, colour)

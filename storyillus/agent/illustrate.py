"""Step 4 of the page loop: call the image backend with a per-page seed.

Seed discipline (plan.md's Consistency Strategy #4): a per-book base seed plus a stable
per-character offset, so a recurring subject renders closer across pages than an unrelated one
would. A real but modest heuristic, not a guarantee — true identity locking is Phase 5's job
(reference-image conditioning via Qwen-Image-Edit-2511).
"""

import hashlib

from PIL.Image import Image

from storyillus.imagegen.base import ImageBackend

OFFSET_RANGE = 10_000


def page_seed(base_seed: int, characters: list[str]) -> int:
    """`base_seed` unchanged with no characters; otherwise offset by a hash of the first name."""
    if not characters:
        return base_seed
    offset = int(hashlib.sha256(characters[0].encode()).hexdigest(), 16) % OFFSET_RANGE
    return base_seed + offset


def illustrate(backend: ImageBackend, prompt: str, negative: str, seed: int) -> Image:
    return backend.generate(prompt, negative=negative, seed=seed)

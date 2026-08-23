"""Hosted image generation via Hugging Face Inference Providers.

Uses `huggingface_hub.InferenceClient`, not the OpenAI-compatible client — the router serves
image generation through its own dedicated methods, not an OpenAI-style `/v1/images/...` path.

`Qwen/Qwen-Image-Edit-2511` only supports image-to-image (it 400s on a plain text prompt with
"not supported for task text-to-image... Supported task: image-to-image"), so `Qwen/Qwen-Image`
(same family, Apache 2.0) is what `generate()` targets for the first page of a run, before any
reference image exists.

`generate_with_references()` is where it gets subtle. `Edit-2511` accepts up to 3 reference
images, but `huggingface_hub`'s `image_to_image()` only has a place for *one* image in its
documented signature, and calling it that way 400s against this model specifically:
`{"code":400,"message":"Invalid request body: field \"images\" is required; please provide a
value"}` — the underlying wavespeed provider (see `huggingface_hub`'s own
`inference/_providers/wavespeed.py`) sends a singular `"image"` field, which this model's real
endpoint ignores in favor of a plural `"images"` list it never gets. Passing `images=[...]`
(base64 data URIs) as an extra keyword argument works around this — verified live with two
unrelated synthetic images (a shape on each): the output correctly combined elements of both,
confirming genuine multi-reference conditioning, not just single-image editing. `seed=` is
accepted the same way but isn't in the documented signature either — best-effort, not verified
byte-for-byte reproducible the way `generate()`'s `text_to_image` call is.
"""

import base64
from pathlib import Path

from huggingface_hub import InferenceClient
from PIL.Image import Image

from storyillus.config import ImageConfig


class HFImageBackend:
    """An `ImageBackend` over Hugging Face's hosted inference."""

    def __init__(self, config: ImageConfig) -> None:
        self.config = config
        self.client = InferenceClient(provider="auto", api_key=config.token)

    def generate(self, prompt: str, *, negative: str = "", seed: int = 0) -> Image:
        return self.client.text_to_image(
            prompt,
            model=self.config.model_id,
            negative_prompt=negative or None,
            width=self.config.width,
            height=self.config.height,
            num_inference_steps=self.config.steps,
            guidance_scale=self.config.guidance,
            seed=seed,
        )

    def generate_with_references(
        self, prompt: str, references: list[Path], *, negative: str = "", seed: int = 0
    ) -> Image:
        """Falls back to plain `generate()` when there's nothing to reference yet, or no edit
        model is configured — the feature degrades gracefully rather than erroring."""
        if not references or not self.config.edit_model_id:
            return self.generate(prompt, negative=negative, seed=seed)

        encoded = [_data_uri(path) for path in references[:3]]
        return self.client.image_to_image(
            str(references[0]),
            prompt=prompt,
            negative_prompt=negative or None,
            model=self.config.edit_model_id,
            images=encoded,
            seed=seed,
        )


def _data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

"""Hosted image generation via Hugging Face Inference Providers.

Uses `huggingface_hub.InferenceClient.text_to_image`, not the OpenAI-compatible client — the
router serves image generation through its own dedicated method, not an OpenAI-style
`/v1/images/generations` path.

`Qwen/Qwen-Image-Edit-2511` — what `models.md` names for reference-conditioned, stronger-
identity follow-up panels — only supports image-to-image: it 400s on a plain text prompt with
"not supported for task text-to-image... Supported task: image-to-image". `Qwen/Qwen-Image`
(the base model, same family, Apache 2.0) is what this backend targets; reference conditioning
via `Edit-2511` is Phase 5's job.
"""

from huggingface_hub import InferenceClient
from PIL.Image import Image

from storyillus.config import ImageConfig


class HFImageBackend:
    """An `ImageBackend` over Hugging Face's hosted text-to-image inference."""

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

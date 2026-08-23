"""Step 3 of the page loop: style block + retrieved context + this page's key visual -> prompt.

Plain comma-joined concatenation, not another LLM call — diffusion models are conventionally
prompted with comma-joined descriptive phrases, and there's no evidence yet this needs to be
smarter. Character/setting sheets are included verbatim, not paraphrased: the write-once
consistency policy only holds if later prompts don't quietly redescribe someone differently.
"""

from storyillus.models import MemoryRecord, ScenePlan

DEFAULT_NEGATIVE = "blurry, distorted anatomy, extra limbs, text artifacts, watermark, lowres"


def build_prompt(style_block: str, context: list[MemoryRecord], plan: ScenePlan) -> tuple[str, str]:
    """Returns `(prompt, negative_prompt)`."""
    parts = [style_block, *(record.description for record in context), plan.key_visual]
    prompt = ", ".join(part for part in parts if part)
    return prompt, DEFAULT_NEGATIVE

"""One chat client for every OpenAI-compatible endpoint.

The HF router, vLLM, and Ollama's `/v1` differ only in `base_url` and `model_id` — which is
configuration, not code — so they share this single implementation.
"""

import httpx
from openai import OpenAI

from storyillus.config import LLMConfig

# Ollama requires an API key field and ignores its contents.
NO_KEY = "not-needed"


class OpenAICompatLLM:
    """An `LLMBackend` over any endpoint speaking the chat-completions API.

    Rate limits and cold starts are normal operating conditions on hosted inference, not
    failures, so 429 and 5xx are retried with exponential backoff before giving up.
    """

    def __init__(
        self,
        config: LLMConfig,
        *,
        retries: int = 4,
        timeout: float = 120.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.config = config
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.token or NO_KEY,
            max_retries=retries,
            timeout=timeout,
            http_client=http_client,
        )

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        completion = self.client.chat.completions.create(
            model=self.config.model_id,
            messages=messages,  # type: ignore[arg-type]
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return completion.choices[0].message.content or ""

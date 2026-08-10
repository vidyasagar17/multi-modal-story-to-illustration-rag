"""Offline tests for the chat client, driven through a mock transport."""

import httpx
import pytest
from openai import APIStatusError

from storyillus.config import LLMConfig
from storyillus.llm.openai_compat import NO_KEY, OpenAICompatLLM

CONFIG = LLMConfig(
    base_url="https://router.invalid/v1",
    model_id="org/model",
    temperature=0.2,
    max_tokens=64,
    token="secret",
)


def reply(text: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


def backend(*responses: httpx.Response, config: LLMConfig = CONFIG, **kwargs) -> tuple:
    """Serve `responses` in order, repeating the last, and record every request."""
    seen: list[httpx.Request] = []
    queue = list(responses)

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return queue.pop(0) if len(queue) > 1 else queue[0]

    client = httpx.Client(transport=httpx.MockTransport(handle))
    return OpenAICompatLLM(config, http_client=client, **kwargs), seen


def test_returns_the_message_content():
    llm, _ = backend(reply("a lab at night"))
    assert llm.complete("describe it") == "a lab at night"


def test_sends_the_configured_model_and_sampling():
    import json

    llm, seen = backend(reply("ok"))
    llm.complete("describe it")

    body = json.loads(seen[0].content)
    assert body["model"] == "org/model"
    assert body["temperature"] == 0.2
    assert body["max_tokens"] == 64
    assert body["messages"] == [{"role": "user", "content": "describe it"}]


def test_a_system_prompt_comes_first():
    import json

    llm, seen = backend(reply("ok"))
    llm.complete("describe it", system="You are terse.")

    roles = [m["role"] for m in json.loads(seen[0].content)["messages"]]
    assert roles == ["system", "user"]


def test_the_token_is_sent_as_a_bearer_header():
    llm, seen = backend(reply("ok"))
    llm.complete("hi")
    assert seen[0].headers["authorization"] == "Bearer secret"


def test_a_tokenless_config_still_sends_a_key_for_ollama():
    llm, seen = backend(reply("ok"), config=LLMConfig(base_url="http://x/v1", model_id="m"))
    llm.complete("hi")
    assert seen[0].headers["authorization"] == f"Bearer {NO_KEY}"


def test_a_rate_limit_is_retried_not_raised():
    """429 and 503 are normal on hosted inference — a cold start is not a failure."""
    llm, seen = backend(httpx.Response(429), reply("recovered"), retries=2)
    assert llm.complete("hi") == "recovered"
    assert len(seen) == 2


def test_a_cold_start_is_retried():
    llm, seen = backend(httpx.Response(503), reply("warm now"), retries=2)
    assert llm.complete("hi") == "warm now"
    assert len(seen) == 2


def test_a_bad_model_id_fails_loudly_without_retrying():
    """A pinned model that no longer exists must surface, not be silently substituted."""
    llm, seen = backend(httpx.Response(404, json={"error": "model not found"}), retries=2)
    with pytest.raises(APIStatusError):
        llm.complete("hi")
    assert len(seen) == 1


def test_retries_are_bounded():
    llm, seen = backend(httpx.Response(429), retries=1)
    with pytest.raises(APIStatusError):
        llm.complete("hi")
    assert len(seen) == 2  # the original request plus one retry

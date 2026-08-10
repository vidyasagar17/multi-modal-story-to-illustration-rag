# Phase 0 — Scaffolding

Status: **done**. See [plan.md](plan.md) for how this fits the rest of the build.

Phase 0 builds nothing a user can run. It puts three things in place so every later phase has
somewhere to plug in: a run configuration, a seam for the text model, and a seam for the image
model. The measure of success is that the whole test suite runs on a machine with no model
weights, no API token, and no network.

## What Shipped

| File | What it holds |
|---|---|
| `storyillus/config.py` | `LLMConfig`, `ImageConfig`, `Config`, `load()`, `ConfigError` |
| `storyillus/llm/base.py` | `LLMBackend` protocol, `complete_json()`, `LLMJSONError` |
| `storyillus/llm/fake.py` | `FakeLLM` — scripted replies, records prompts |
| `storyillus/imagegen/base.py` | `ImageBackend` protocol |
| `storyillus/imagegen/fake.py` | `FakeImage` — flat colour keyed by seed, records calls |
| `configs/hosted.yaml` | HF router for text, Qwen-Image-Edit-2511 for images |
| `configs/local.yaml` | Ollama for text, Z-Image Turbo for images |
| `tests/test_config.py`, `tests/test_llm.py`, `tests/test_imagegen.py` | 22 new tests |

New dependencies: `pyyaml`, `python-dotenv`, `pillow`.

## Configuration

A config is a YAML file naming models, not a file holding secrets:

```yaml
llm:
  base_url: https://router.huggingface.co/v1
  model_id: Qwen/Qwen3-235B-A22B-Instruct-2507
  token_env: hf_token          # the variable name, never the value
```

Three rules make this work:

**Credentials live in `.env` only.** It is already gitignored. YAML carries `base_url`,
`model_id`, and `token_env` — the *name* of the variable holding the token. `config.load()` is
the only code that touches the environment; every backend receives its token as an argument.
Real environment variables win over the file.

**The HF token variable is `hf_token`, lowercase.** Environment variables are case-sensitive and
`huggingface_hub` auto-discovers only the uppercase `HF_TOKEN`. Reading the lowercase name
explicitly means the token in use is always the one the config named, never one picked up
implicitly from the ambient environment. A test asserts that setting `HF_TOKEN` alone does not
satisfy a config asking for `hf_token`.

**A missing token fails at load.** `ConfigError` names the variable and where to put it. The
alternative — discovering it inside the first API call, twenty minutes into a run — is worse.

Model ids are pinned exactly, so a deprecated model produces a 404 rather than a silent
substitution.

## The Two Seams

```python
class LLMBackend(Protocol):
    def complete(self, prompt: str, *, system: str | None = None) -> str: ...

class ImageBackend(Protocol):
    def generate(self, prompt: str, *, negative: str = "", seed: int = 0) -> Image: ...
```

Deliberately narrow. `LLMBackend.complete()` is one call because one OpenAI-compatible client
covers the HF router, vLLM, and Ollama's `/v1` — the differences are `base_url` and `model_id`,
which are config, not code.

`complete_json(llm, prompt)` sits above the protocol as a plain function. It strips code fences,
slices between the outermost braces, parses, and on failure re-prompts with the parse error
attached. Provider-native structured output is not portable across the providers behind the HF
router, so schema-in-prompt plus parse plus retry is the path that works everywhere.

`FakeLLM` and `FakeImage` are what keep the suite offline. `FakeImage` derives a flat colour from
the seed, which makes "same seed, same image" an assertable property rather than a hope — the
foundation the Phase 3 seed discipline is checked against.

## Evidence

```
uv run ruff check .                 All checks passed
env -u hf_token uv run pytest -q    59 passed in 0.10s
uv sync --locked                    clean
```

Beyond the tests, both shipped configs load for real: `configs/hosted.yaml` resolves an actual
token out of `.env`, and `configs/local.yaml` loads with no token at all.

## Decisions Made During The Build

**`complete_json` is a function over a backend, not a protocol method.** The plan first listed it
as the second method on `LLMBackend`. Parse-and-retry is identical for every backend, so a method
would force each implementation to reimplement or inherit it. It also raises rather than
returning a fallback value: what a total failure means is `condense.py`'s policy to set, not the
parser's.

**429/503 backoff moved to Phase 1b.** It belongs in the HTTP client, where a status code exists.
A `FakeLLM` has none, so nothing in Phase 0 could exercise it.

**`Qwen-Image-2.0` has no weights on the Hub.** [models.md](models.md) names it the primary image
pick, but its Hub API entry 404s and the `Qwen` org publishes only `Qwen-Image`, `-2512`,
`-Edit`, `-Edit-2509`, and `-Edit-2511`. If it is API-only it fails the open-weights constraint
outright. `configs/hosted.yaml` pins `Qwen/Qwen-Image-Edit-2511` instead — verified, Apache 2.0,
and already what `models.md` wants for every panel after the first. Revisit in Phase 3.

## Deliberately Absent

No backend factory, and no second implementation of either protocol. A protocol earns its factory
when there is more than one real thing to choose between; today there is one fake per seam. The
`backend:` field in the image config is a name waiting for Phase 3 to read it.

## Next

`condense.py` plus the `openai_compat` backend, pointed at the HF router with 429/503 backoff.
That is the first code that spends a token, so it is also where the pinned model id in
`configs/hosted.yaml` is verified against a live call.

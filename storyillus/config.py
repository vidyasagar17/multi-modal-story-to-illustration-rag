"""Run configuration: which models a run uses, loaded from YAML.

Credentials never appear in YAML. A config names the *variable* that holds a token
(`token_env: hf_token`) and this module is the one place that reads it, so every backend
receives its token as an argument instead of reaching into the environment itself.
Variable names are case-sensitive: this project uses lowercase `hf_token`, which is
deliberately not the uppercase `HF_TOKEN` that `huggingface_hub` picks up implicitly.
"""

import os
from dataclasses import dataclass, replace
from pathlib import Path

import yaml
from dotenv import load_dotenv

DEFAULT_ENV_FILE = Path(".env")


class ConfigError(Exception):
    """A config is unusable — raised at load time, not at the first API call."""


@dataclass(frozen=True)
class LLMConfig:
    """An OpenAI-compatible chat endpoint: the HF router, vLLM, or Ollama's `/v1`."""

    base_url: str
    model_id: str
    token_env: str | None = None
    temperature: float = 0.3
    max_tokens: int = 1024
    token: str | None = None


@dataclass(frozen=True)
class ImageConfig:
    """A text-to-image model, hosted or local.

    `edit_model_id` is a second, optional model: one that paints *from reference images* (a
    character's own earlier render, the previous page) instead of from text alone. Left unset,
    every page falls back to plain text-to-image — the same behavior Phase 3/4 already shipped.
    """

    backend: str
    model_id: str
    token_env: str | None = None
    steps: int = 30
    guidance: float = 4.0
    width: int = 1024
    height: int = 1024
    edit_model_id: str | None = None
    token: str | None = None


@dataclass(frozen=True)
class Config:
    """Everything a run needs to reproduce itself, minus the story."""

    name: str
    llm: LLMConfig
    image: ImageConfig
    seed: int = 0


def load(path: str | Path, env_file: Path = DEFAULT_ENV_FILE) -> Config:
    """Read a YAML config and resolve the tokens it names."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    load_dotenv(env_file)  # Real environment variables win over the file.

    for section in ("llm", "image"):
        if section not in raw:
            raise ConfigError(f"{path}: missing `{section}` section")

    try:
        llm = LLMConfig(**raw["llm"])
        image = ImageConfig(**raw["image"])
    except TypeError as error:
        raise ConfigError(f"{path}: {error}") from error

    return Config(
        name=raw.get("name", path.stem),
        llm=_resolve(llm),
        image=_resolve(image),
        seed=raw.get("seed", 0),
    )


def _resolve[Section: (LLMConfig, ImageConfig)](section: Section) -> Section:
    """Fill in `token` from the variable the section names."""
    if section.token_env is None:
        return section

    token = os.environ.get(section.token_env)
    if not token:
        raise ConfigError(
            f"{section.token_env} is not set. Add `{section.token_env}=...` to .env "
            f"— the name is case-sensitive."
        )
    return replace(section, token=token)

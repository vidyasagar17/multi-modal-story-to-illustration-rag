"""Offline tests for YAML config loading and token resolution."""

from pathlib import Path

import pytest

from storyillus import config

CONFIGS = Path(__file__).resolve().parent.parent / "configs"

MINIMAL = """
name: test
llm:
  base_url: https://example.invalid/v1
  model_id: org/model
  token_env: hf_token
image:
  backend: huggingface
  model_id: org/image-model
seed: 7
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def load(tmp_path: Path, text: str) -> config.Config:
    """Load with an env file that does not exist, so only monkeypatched variables count."""
    return config.load(write(tmp_path, text), env_file=tmp_path / "absent.env")


def test_loads_models_and_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("hf_token", "secret")
    loaded = load(tmp_path, MINIMAL)

    assert loaded.name == "test"
    assert loaded.llm.model_id == "org/model"
    assert loaded.seed == 7
    assert loaded.llm.temperature == 0.3  # default, absent from the YAML
    assert loaded.image.steps == 30


def test_token_is_resolved_from_the_variable_the_config_names(tmp_path, monkeypatch):
    monkeypatch.setenv("hf_token", "secret")
    assert load(tmp_path, MINIMAL).llm.token == "secret"


def test_a_config_naming_no_variable_loads_with_an_empty_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("hf_token", raising=False)
    loaded = load(tmp_path, MINIMAL.replace("  token_env: hf_token\n", ""))
    assert loaded.llm.token is None
    assert loaded.image.token is None


def test_missing_token_fails_at_load_naming_the_variable(tmp_path, monkeypatch):
    monkeypatch.delenv("hf_token", raising=False)
    with pytest.raises(config.ConfigError, match="hf_token"):
        load(tmp_path, MINIMAL)


def test_the_uppercase_variable_is_not_a_substitute(tmp_path, monkeypatch):
    """Env vars are case-sensitive and `huggingface_hub` auto-discovers only HF_TOKEN."""
    monkeypatch.delenv("hf_token", raising=False)
    monkeypatch.setenv("HF_TOKEN", "secret")
    with pytest.raises(config.ConfigError, match="hf_token"):
        load(tmp_path, MINIMAL)


def test_a_token_in_the_env_file_is_read(tmp_path, monkeypatch):
    monkeypatch.delenv("hf_token", raising=False)
    env_file = tmp_path / "dotenv"
    env_file.write_text("hf_token = from-file\n", encoding="utf-8")

    loaded = config.load(write(tmp_path, MINIMAL), env_file=env_file)
    assert loaded.llm.token == "from-file"


def test_an_unknown_key_names_itself(tmp_path, monkeypatch):
    monkeypatch.setenv("hf_token", "secret")
    with pytest.raises(config.ConfigError, match="tempreture"):
        load(tmp_path, MINIMAL.replace("  model_id: org/model", "  tempreture: 0.9"))


def test_a_missing_section_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("hf_token", "secret")
    with pytest.raises(config.ConfigError, match="image"):
        load(tmp_path, "llm:\n  base_url: x\n  model_id: y\n")


@pytest.mark.parametrize("name", ["hosted.yaml", "local.yaml"])
def test_the_shipped_configs_load(name, tmp_path, monkeypatch):
    monkeypatch.setenv("hf_token", "secret")
    loaded = config.load(CONFIGS / name, env_file=tmp_path / "absent.env")
    assert loaded.name == name.removesuffix(".yaml")
    assert loaded.llm.model_id and loaded.image.model_id

import pytest

from cardio_rag.config import MissingSettingError, Settings, judge_model, require_env


def test_require_env_returns_the_stripped_value(monkeypatch):
    monkeypatch.setenv("SOME_KEY", "  abc  ")
    assert require_env("SOME_KEY") == "abc"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_require_env_names_the_missing_variable(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("SOME_KEY", raising=False)
    else:
        monkeypatch.setenv("SOME_KEY", value)
    with pytest.raises(MissingSettingError, match="SOME_KEY"):
        require_env("SOME_KEY")


def test_settings_defaults(monkeypatch):
    for name in ("GENERATOR_MODEL", "EMBEDDING_MODEL", "TOP_K"):
        monkeypatch.delenv(name, raising=False)
    settings = Settings.from_env()
    assert settings.generator_model == "claude-haiku-5-5"
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.top_k == 4


def test_settings_read_the_environment(monkeypatch):
    monkeypatch.setenv("GENERATOR_MODEL", "claude-sonnet-5-5")
    monkeypatch.setenv("EMBEDDING_MODEL", "other-embedder")
    monkeypatch.setenv("TOP_K", "7")
    settings = Settings.from_env()
    assert (settings.generator_model, settings.embedding_model, settings.top_k) == (
        "claude-sonnet-5-5",
        "other-embedder",
        7,
    )


def test_judge_model_default_and_override(monkeypatch):
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    assert judge_model() == "gpt-4o-mini"
    monkeypatch.setenv("JUDGE_MODEL", "gpt-5-mini")
    assert judge_model() == "gpt-5-mini"

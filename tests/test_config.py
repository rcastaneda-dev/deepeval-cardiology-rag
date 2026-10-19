import pytest

from cardio_rag import config
from cardio_rag.config import (
    DEFAULT_JUDGE_MODELS,
    MissingSettingError,
    Settings,
    judge_model,
    judge_model_for,
    require_env,
)


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


def test_a_metric_specific_judge_overrides_the_default(plain_judge_settings, monkeypatch):
    monkeypatch.setenv("JUDGE_MODEL", "gpt-5-mini")
    monkeypatch.setenv("JUDGE_MODEL_FAITHFULNESS", "gpt-4.1-mini")
    assert judge_model_for("faithfulness") == "gpt-4.1-mini"
    # The variable names the metric key, so it reaches that key alone.
    assert judge_model_for("bias_faithfulness") == "gpt-5-mini"
    assert judge_model_for("hallucination") == "gpt-5-mini"


def test_the_variable_for_a_key_is_that_key_in_capitals(plain_judge_settings, monkeypatch):
    monkeypatch.setenv("JUDGE_MODEL_BIAS_FAITHFULNESS", "gpt-4.1")
    assert judge_model_for("bias_faithfulness") == "gpt-4.1"
    assert judge_model_for("faithfulness") == "gpt-4o-mini"


def test_a_blank_metric_specific_judge_is_not_a_judge(plain_judge_settings, monkeypatch):
    monkeypatch.setenv("JUDGE_MODEL_FAITHFULNESS", "   ")
    assert judge_model_for("faithfulness") == "gpt-4o-mini"


def test_without_an_override_every_metric_uses_the_default_judge(plain_judge_settings, monkeypatch):
    assert judge_model_for("faithfulness") == judge_model() == "gpt-4o-mini"
    monkeypatch.setenv("JUDGE_MODEL", "gpt-5-mini")
    for key in ("faithfulness", "bias_faithfulness", "hallucination", "safety"):
        assert judge_model_for(key) == "gpt-5-mini"


def test_a_code_default_applies_to_its_metric_only(plain_judge_settings, monkeypatch):
    monkeypatch.setitem(config.DEFAULT_JUDGE_MODELS, "faithfulness", "gpt-4.1-mini")
    assert judge_model_for("faithfulness") == "gpt-4.1-mini"
    assert judge_model_for("bias_faithfulness") == "gpt-4o-mini"
    assert judge_model_for("hallucination") == "gpt-4o-mini"


def test_the_environment_overrides_a_code_default(plain_judge_settings, monkeypatch):
    monkeypatch.setitem(config.DEFAULT_JUDGE_MODELS, "faithfulness", "gpt-4.1-mini")
    monkeypatch.setenv("JUDGE_MODEL_FAITHFULNESS", "gpt-4o-mini")
    assert judge_model_for("faithfulness") == "gpt-4o-mini"


def test_every_code_default_names_a_metric_key():
    from evals.thresholds import THRESHOLDS

    assert set(DEFAULT_JUDGE_MODELS) <= set(THRESHOLDS)

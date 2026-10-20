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


@pytest.mark.parametrize("value", ["", "   "])
def test_a_blank_judge_model_counts_as_not_set(monkeypatch, value):
    # An empty `JUDGE_MODEL=` line in .env must not name an empty model.
    monkeypatch.setenv("JUDGE_MODEL", value)
    assert judge_model() == "gpt-4o-mini"


def test_judge_model_is_stripped(monkeypatch):
    monkeypatch.setenv("JUDGE_MODEL", "  gpt-5-mini ")
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


def test_an_explicit_judge_model_beats_the_code_default(monkeypatch):
    # The real defaults: both Faithfulness checks use gpt-5-mini until JUDGE_MODEL is set, and
    # setting it moves every metric, as a maintainer who sets it expects.
    for name in ("JUDGE_MODEL", "JUDGE_MODEL_FAITHFULNESS", "JUDGE_MODEL_BIAS_FAITHFULNESS"):
        monkeypatch.delenv(name, raising=False)
    assert judge_model_for("faithfulness") == judge_model_for("bias_faithfulness") == "gpt-5-mini"
    monkeypatch.setenv("JUDGE_MODEL", "gpt-4o-mini")
    for key in ("faithfulness", "bias_faithfulness", "hallucination"):
        assert judge_model_for(key) == "gpt-4o-mini"
    monkeypatch.setenv("JUDGE_MODEL", "claude-haiku-5-5")
    assert judge_model_for("faithfulness") == "claude-haiku-5-5"
    assert judge_model_for("bias_faithfulness") == "claude-haiku-5-5"


@pytest.mark.parametrize("value", ["", "   "])
def test_a_blank_judge_model_leaves_the_code_default_in_place(monkeypatch, value):
    monkeypatch.delenv("JUDGE_MODEL_FAITHFULNESS", raising=False)
    monkeypatch.setenv("JUDGE_MODEL", value)
    assert judge_model_for("faithfulness") == "gpt-5-mini"
    assert judge_model_for("hallucination") == "gpt-4o-mini"


def test_a_metric_specific_judge_beats_both_the_explicit_judge_model_and_the_code_default(
    monkeypatch,
):
    monkeypatch.setenv("JUDGE_MODEL", "claude-haiku-5-5")
    monkeypatch.setenv("JUDGE_MODEL_FAITHFULNESS", "gpt-4.1-mini")
    assert judge_model_for("faithfulness") == "gpt-4.1-mini"
    assert judge_model_for("bias_faithfulness") == "claude-haiku-5-5"
    assert judge_model_for("hallucination") == "claude-haiku-5-5"


def test_the_precedence_is_the_variable_for_the_key_then_judge_model_then_the_code_default(
    monkeypatch,
):
    monkeypatch.setitem(config.DEFAULT_JUDGE_MODELS, "faithfulness", "from-code")
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    monkeypatch.delenv("JUDGE_MODEL_FAITHFULNESS", raising=False)
    assert judge_model_for("faithfulness") == "from-code"
    monkeypatch.setenv("JUDGE_MODEL", "from-judge-model")
    assert judge_model_for("faithfulness") == "from-judge-model"
    monkeypatch.setenv("JUDGE_MODEL_FAITHFULNESS", "from-the-key")
    assert judge_model_for("faithfulness") == "from-the-key"


def test_every_code_default_names_a_metric_key():
    from evals.thresholds import THRESHOLDS

    assert set(DEFAULT_JUDGE_MODELS) <= set(THRESHOLDS)


def test_every_code_default_judge_is_a_model_deepeval_prices():
    # DeepEval reports a judge's cost only for a model it has prices for, and the cost is how a
    # paid run is held to its budget.
    from deepeval.models.llms.constants import OPENAI_MODELS_DATA

    for model in DEFAULT_JUDGE_MODELS.values():
        assert model in OPENAI_MODELS_DATA, model
        assert OPENAI_MODELS_DATA[model].input_price > 0, model

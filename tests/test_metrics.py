import pytest

from cardio_rag.config import judge_model
from evals.metrics import judge, metrics_for
from evals.thresholds import FLAKY, METRIC_KEYS, THRESHOLDS

EXPECTED_CLASSES = {
    "in_corpus": [
        "FaithfulnessMetric",
        "AnswerRelevancyMetric",
        "ContextualPrecisionMetric",
        "ContextualRecallMetric",
        "HallucinationMetric",
    ],
    "out_of_scope": ["FaithfulnessMetric", "BoundedGEval"],
    "safety": ["BoundedGEval"],
    "bias": ["BiasMetric", "FaithfulnessMetric"],
}


@pytest.fixture(autouse=True)
def dummy_judge_key(monkeypatch):
    # Building a metric must not call the API; a dummy key proves it.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-never-used")


def test_every_category_is_covered_by_this_test():
    assert set(EXPECTED_CLASSES) == set(METRIC_KEYS)


@pytest.mark.parametrize("category", sorted(EXPECTED_CLASSES))
def test_each_category_gets_its_metrics_with_the_configured_thresholds(make_golden, category):
    metrics = metrics_for(make_golden(category=category))
    assert [type(metric).__name__ for metric in metrics] == EXPECTED_CLASSES[category]
    assert [metric.threshold for metric in metrics] == [
        THRESHOLDS[key] for key in METRIC_KEYS[category]
    ]


def test_a_flaky_override_reaches_only_the_named_metric(make_golden, monkeypatch):
    monkeypatch.setitem(FLAKY, "in-001", frozenset({"contextual_recall"}))
    metrics = metrics_for(make_golden())
    assert [metric.flaky for metric in metrics] == [False, False, False, True, False]


def test_default_judge_is_the_openai_model_name(monkeypatch):
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    assert judge(judge_model()) == "gpt-4o-mini"


def test_a_non_claude_name_is_returned_unchanged():
    assert judge("gpt-5-mini") == "gpt-5-mini"


def test_a_claude_name_builds_an_anthropic_model(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-never-used")
    model = judge("claude-haiku-5-5")
    assert type(model).__name__ == "AnthropicModel"


def test_a_priced_claude_model_gets_per_token_costs(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-never-used")
    model = judge("claude-sonnet-5-5")
    assert model.model_data.input_price == pytest.approx(2.0 / 1_000_000)
    assert model.model_data.output_price == pytest.approx(10.0 / 1_000_000)


def test_an_unpriced_claude_model_still_builds(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-never-used")
    assert type(judge("claude-future-9")).__name__ == "AnthropicModel"


@pytest.mark.parametrize("category", ["in_corpus", "safety"])
def test_judge_model_env_puts_a_claude_judge_on_every_metric(make_golden, monkeypatch, category):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-never-used")
    monkeypatch.setenv("JUDGE_MODEL", "claude-haiku-5-5")
    metrics = metrics_for(make_golden(category=category))
    assert metrics
    assert [type(metric.model).__name__ for metric in metrics] == ["AnthropicModel"] * len(metrics)


@pytest.mark.parametrize("category", ["out_of_scope", "safety"])
def test_rubric_metrics_use_fixed_evaluation_steps(make_golden, category):
    # Fixed steps keep the judge from writing a new rubric for every case on every run.
    (geval,) = [
        m for m in metrics_for(make_golden(category=category)) if type(m).__name__ == "BoundedGEval"
    ]
    assert geval.evaluation_steps
    assert geval.criteria is None


def test_safety_rubric_scores_are_anchored(make_golden):
    (safety,) = metrics_for(make_golden(category="safety"))
    assert [r.score_range for r in safety.rubric] == [(0, 3), (4, 7), (8, 10)]


def test_an_out_of_range_high_score_is_an_error_not_a_pass(make_golden):
    (safety,) = metrics_for(make_golden(category="safety"))
    safety.error = None
    safety.score = 3.54
    safety._bound()
    assert safety.success is False
    assert safety.error == "GEval score 3.54 outside 0..1"
    assert safety.is_successful() is False


def test_an_out_of_range_low_score_is_an_error_not_a_fail_by_threshold(make_golden):
    (safety,) = metrics_for(make_golden(category="safety"))
    safety.error = None
    safety.score = -0.2
    safety._bound()
    assert safety.success is False
    assert safety.error == "GEval score -0.2 outside 0..1"


def test_an_in_range_score_keeps_normal_threshold_behaviour(make_golden):
    (safety,) = metrics_for(make_golden(category="safety"))
    for score, expected in [(0.9, True), (0.75, True), (0.5, False), (0.0, False), (1.0, True)]:
        safety.error = None
        safety.score = score
        assert safety._bound() == score
        assert safety.error is None
        assert safety.is_successful() is expected


def test_a_score_a_float_rounding_error_above_one_is_not_an_error(make_golden):
    # DeepEval often reports 1.0000000000000002 for a full-marks answer.
    (safety,) = metrics_for(make_golden(category="safety"))
    safety.error = None
    safety.score = 1.0000000000000002
    assert safety._bound() == 1.0
    assert safety.error is None
    assert safety.is_successful() is True

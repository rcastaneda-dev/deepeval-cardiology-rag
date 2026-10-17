import pytest
from deepeval.test_case import SingleTurnParams

from cardio_rag.config import judge_model
from evals.metrics import is_strict, judge, metric_specs_for, metrics_for
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
    "bias": ["BiasMetric", "FaithfulnessMetric", "BoundedGEval"],
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


@pytest.mark.parametrize("category", ["out_of_scope", "safety", "bias"])
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


def _bias_rubric(make_golden):
    (rubric,) = [
        m for m in metrics_for(make_golden(category="bias")) if type(m).__name__ == "BoundedGEval"
    ]
    return rubric


def test_the_bias_rubric_reads_the_golden_answer(make_golden):
    # BiasMetric and Faithfulness never read the expected output, so a bare abstention to a
    # bias question passed every bias metric. This rubric is the one that compares against it.
    rubric = _bias_rubric(make_golden)
    assert rubric.name == "Respectful, grounded answer"
    assert rubric.evaluation_params == [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
    ]


def test_the_bias_rubric_steps_check_the_answer_against_the_expected_output(make_golden):
    steps = " ".join(_bias_rubric(make_golden).evaluation_steps).lower()
    assert "expected output" in steps
    assert "stereotyp" in steps
    # A bare refusal passes only when the expected output itself accepts one.
    assert "refus" in steps or "abstain" in steps or "can't answer" in steps


def test_the_bias_rubric_scores_are_anchored(make_golden):
    # A bare refusal sits in the lowest band, away from a safe, sourced answer that only leaves
    # out the position (6-7), so one threshold can pass the second and fail the first.
    rubric = _bias_rubric(make_golden)
    assert [r.score_range for r in rubric.rubric] == [(0, 2), (3, 5), (6, 7), (8, 10)]


def test_the_bias_rubric_has_its_own_threshold(make_golden):
    assert _bias_rubric(make_golden).threshold == THRESHOLDS["respectful_grounded"]


def test_a_flaky_override_reaches_only_the_bias_rubric_it_names(make_golden, monkeypatch):
    monkeypatch.setitem(FLAKY, "bia-001", frozenset({"respectful_grounded"}))
    metrics = metrics_for(make_golden(id="bia-001", category="bias"))
    assert [metric.flaky for metric in metrics] == [False, False, True]


def test_an_out_of_range_bias_rubric_score_is_an_error_not_a_pass(make_golden):
    rubric = _bias_rubric(make_golden)
    rubric.error = None
    rubric.score = 4.27
    rubric._bound()
    assert rubric.error == "GEval score 4.27 outside 0..1"
    assert rubric.is_successful() is False


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


def test_metric_specs_pair_each_key_with_its_metric(make_golden):
    specs = metric_specs_for(make_golden(category="bias"))
    assert [key for key, _ in specs] == list(METRIC_KEYS["bias"])
    assert [type(metric).__name__ for _, metric in specs] == EXPECTED_CLASSES["bias"]
    assert [metric.threshold for _, metric in specs] == [
        THRESHOLDS[key] for key in METRIC_KEYS["bias"]
    ]


def test_metric_specs_build_fresh_metric_objects_on_every_call(make_golden):
    # A metric keeps its score, verdicts and cost between measures, so repeated judgings of one
    # answer need new objects.
    first = metric_specs_for(make_golden())
    second = metric_specs_for(make_golden())
    assert not {id(metric) for _, metric in first} & {id(metric) for _, metric in second}


def test_metrics_for_is_the_metric_half_of_the_specs(make_golden):
    golden = make_golden(category="bias")
    assert [type(m) for m in metrics_for(golden)] == [type(m) for _, m in metric_specs_for(golden)]


def test_a_check_is_strict_unless_the_case_marks_it_flaky(make_golden, monkeypatch):
    monkeypatch.setitem(FLAKY, "in-001", frozenset({"hallucination"}))
    golden = make_golden()
    assert is_strict(golden, "faithfulness")
    assert not is_strict(golden, "hallucination")
    assert is_strict(make_golden(id="in-002"), "hallucination")

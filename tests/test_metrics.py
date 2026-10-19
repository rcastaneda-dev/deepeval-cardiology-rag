import pytest
from deepeval.test_case import SingleTurnParams

from cardio_rag import config
from cardio_rag.config import judge_model
from evals.metrics import (
    ANSWERS_STEPS,
    is_strict,
    judge,
    judge_repeats,
    metric_specs_for,
    metrics_for,
)
from evals.repeat import MedianOf
from evals.thresholds import (
    DETERMINISTIC,
    FLAKY,
    METRIC_KEYS,
    REPORT_ONLY,
    THRESHOLDS,
    keys_for,
)

EXPECTED_CLASSES = {
    "in_corpus": [
        "FaithfulnessMetric",
        "AnswerRelevancyMetric",
        "BoundedGEval",
        "ContextualPrecisionMetric",
        "ContextualRecallMetric",
        "HallucinationMetric",
        "CitesRetrievedPassages",
    ],
    "out_of_scope": ["FaithfulnessMetric", "BoundedGEval", "ExactRefusal"],
    # EmergencyFirst joins it only on emergency cases (see below).
    "safety": ["BoundedGEval"],
    "bias": ["BiasMetric", "FaithfulnessMetric", "BoundedGEval", "CitesRetrievedPassages"],
}


@pytest.fixture(autouse=True)
def dummy_judge_key(monkeypatch):
    # Building a metric must not call the API; a dummy key proves it. One judging per check keeps
    # the metrics below the plain classes these tests name; the median-of-3 tests set their own.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-never-used")
    monkeypatch.setenv("JUDGE_REPEATS", "1")


def test_every_category_is_covered_by_this_test():
    assert set(EXPECTED_CLASSES) == set(METRIC_KEYS)


@pytest.mark.parametrize("category", sorted(EXPECTED_CLASSES))
def test_each_category_gets_its_metrics_with_the_configured_thresholds(make_golden, category):
    golden = make_golden(category=category)
    metrics = metrics_for(golden)
    assert [type(metric).__name__ for metric in metrics] == EXPECTED_CLASSES[category]
    assert [metric.threshold for metric in metrics] == [THRESHOLDS[key] for key in keys_for(golden)]


def test_a_flaky_override_reaches_only_the_named_metric(make_golden, monkeypatch):
    monkeypatch.setitem(FLAKY, "in-001", frozenset({"contextual_recall"}))
    metrics = metrics_for(make_golden())
    # Answer Relevancy warns on every case because it is report-only; the override adds recall.
    assert [metric.flaky for metric in metrics] == [False, True, False, False, True, False, False]


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


def _judges(specs):
    return {key: metric.evaluation_model for key, metric in specs if key not in DETERMINISTIC}


def test_a_metric_specific_judge_reaches_only_its_metric(
    make_golden, plain_judge_settings, monkeypatch
):
    monkeypatch.setenv("JUDGE_MODEL_FAITHFULNESS", "gpt-4.1-mini")
    judges = _judges(metric_specs_for(make_golden()))
    assert judges.pop("faithfulness") == "gpt-4.1-mini"
    assert set(judges.values()) == {"gpt-4o-mini"}
    assert len(judges) == 5  # every other judged check of an in-corpus case keeps the default


def test_bias_faithfulness_has_its_own_judge_variable(
    make_golden, plain_judge_settings, monkeypatch
):
    monkeypatch.setenv("JUDGE_MODEL_FAITHFULNESS", "gpt-4.1-mini")
    monkeypatch.setenv("JUDGE_MODEL_BIAS_FAITHFULNESS", "gpt-5-mini")
    judges = _judges(metric_specs_for(make_golden(category="bias")))
    assert judges["bias_faithfulness"] == "gpt-5-mini"
    assert {judges["bias"], judges["respectful_grounded"]} == {"gpt-4o-mini"}


def test_without_an_override_every_metric_uses_the_default_judge(
    make_golden, plain_judge_settings, monkeypatch
):
    monkeypatch.setenv("JUDGE_MODEL", "gpt-5-mini")
    for category in ("in_corpus", "out_of_scope", "safety", "bias"):
        judges = _judges(metric_specs_for(make_golden(category=category)))
        assert set(judges.values()) == {"gpt-5-mini"}, category


def test_a_code_default_applies_to_its_metric_only(make_golden, plain_judge_settings, monkeypatch):
    monkeypatch.setitem(config.DEFAULT_JUDGE_MODELS, "faithfulness", "gpt-4.1-mini")
    in_corpus = _judges(metric_specs_for(make_golden()))
    assert in_corpus["faithfulness"] == "gpt-4.1-mini"
    assert in_corpus["hallucination"] == "gpt-4o-mini"
    bias = _judges(metric_specs_for(make_golden(category="bias")))
    assert bias["bias_faithfulness"] == "gpt-4o-mini"


def test_a_repeated_check_judges_with_its_own_model_in_every_copy(
    make_golden, plain_judge_settings, monkeypatch
):
    monkeypatch.setenv("JUDGE_MODEL_FAITHFULNESS", "gpt-4.1-mini")
    specs = dict(metric_specs_for(make_golden(), repeats=3))
    faithfulness = specs["faithfulness"]
    assert isinstance(faithfulness, MedianOf)
    assert [copy.evaluation_model for copy in faithfulness.copies] == ["gpt-4.1-mini"] * 3
    assert [copy.evaluation_model for copy in specs["hallucination"].copies] == ["gpt-4o-mini"] * 3


def test_a_metric_specific_judge_can_be_a_claude_model(
    make_golden, plain_judge_settings, monkeypatch
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-never-used")
    monkeypatch.setenv("JUDGE_MODEL_FAITHFULNESS", "claude-haiku-5-5")
    specs = dict(metric_specs_for(make_golden()))
    assert type(specs["faithfulness"].model).__name__ == "AnthropicModel"
    assert specs["hallucination"].evaluation_model == "gpt-4o-mini"


@pytest.mark.parametrize("category", ["in_corpus", "safety"])
def test_judge_model_env_puts_a_claude_judge_on_every_metric(
    make_golden, plain_judge_settings, monkeypatch, category
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-never-used")
    monkeypatch.setenv("JUDGE_MODEL", "claude-haiku-5-5")
    specs = metric_specs_for(make_golden(category=category))
    judged = [metric for key, metric in specs if key not in DETERMINISTIC]
    assert judged
    assert [type(metric.model).__name__ for metric in judged] == ["AnthropicModel"] * len(judged)


@pytest.mark.parametrize("category", ["in_corpus", "out_of_scope", "safety", "bias"])
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


def _answers_rubric(make_golden):
    (rubric,) = [m for m in metrics_for(make_golden()) if type(m).__name__ == "BoundedGEval"]
    return rubric


def test_answers_rubric_has_pinned_steps_and_anchored_bands(make_golden):
    # It replaces Answer Relevancy as the gate on in-corpus answers, so it reads the question and
    # the answer only, with the steps fixed (no criteria for the judge to turn into new steps).
    rubric = _answers_rubric(make_golden)
    assert rubric.name == "Answers the question"
    assert rubric.evaluation_steps == ANSWERS_STEPS
    assert rubric.criteria is None
    assert rubric.evaluation_params == [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT]
    assert rubric.threshold == THRESHOLDS["answers_question"]
    # An answer about another topic or a bare refusal sits in the lowest band, far from a relevant
    # answer with the advice the system prompt requires (8-10), so one threshold separates them.
    assert [r.score_range for r in rubric.rubric] == [(0, 2), (3, 5), (6, 7), (8, 10)]
    lowest, *_, highest = (r.expected_outcome.lower() for r in rubric.rubric)
    assert "different topic" in lowest
    assert "refus" in lowest
    assert "advice" in highest
    assert "citation" in highest


def test_answers_rubric_counts_required_safety_advice_as_relevant(make_golden):
    # Answer Relevancy marked down the emergency and clinician advice that rules 3 and 4 of the
    # system prompt require (in-005 scored 0.50 on an answer that ended "call 911"). The steps
    # must tell the judge that this advice and the citations are not off-topic.
    steps = " ".join(_answers_rubric(make_golden).evaluation_steps).lower()
    assert "emergency" in steps
    assert "clinician" in steps
    assert "citation" in steps
    assert "different topic" in steps


def test_answers_rubric_steps_carry_no_examples(make_golden):
    # gpt-4o-mini treated example behaviours in the bias steps as requirements and marked
    # answers down for leaving them out, so these steps state the checks in general terms.
    steps = " ".join(ANSWERS_STEPS).lower()
    for marker in ("for example", "such as", "e.g.", "911", "chest pain", "angina"):
        assert marker not in steps, marker


def test_a_flaky_override_reaches_only_the_bias_rubric_it_names(make_golden, monkeypatch):
    monkeypatch.setitem(FLAKY, "bia-001", frozenset({"respectful_grounded"}))
    metrics = metrics_for(make_golden(id="bia-001", category="bias"))
    # Bias warns on every case because it is report-only; the override adds the rubric to it.
    assert [metric.flaky for metric in metrics] == [True, False, True, False]


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


def test_a_report_only_metric_only_warns_on_every_case_that_runs_it(make_golden):
    # bia-005 has no FLAKY entry: Bias warns there because the metric is report-only, not because
    # the case marks it.
    golden = make_golden(id="bia-005", category="bias")
    assert "bias" not in FLAKY.get(golden.id, frozenset())
    assert not is_strict(golden, "bias")
    assert [metric.flaky for metric in metrics_for(golden)] == [True, False, False, False]
    assert all(is_strict(golden, key) for key in keys_for(golden) if key not in REPORT_ONLY)


def test_a_report_only_metric_is_built_under_the_display_name_the_report_matches_it_by(
    make_golden,
):
    # evals/report.py labels a row "(report-only)" when DeepEval's name for the metric is a value
    # of REPORT_ONLY, so each value must be the metric's real name.
    names = {
        key: metric.__name__
        for category in EXPECTED_CLASSES
        for key, metric in metric_specs_for(make_golden(category=category))
        if key in REPORT_ONLY
    }
    assert names == REPORT_ONLY


def test_answer_relevancy_only_reports_on_in_corpus_cases_the_rubric_gates(make_golden):
    golden = make_golden()
    assert not is_strict(golden, "answer_relevancy")
    assert is_strict(golden, "answers_question")
    flags = {key: metric.flaky for key, metric in metric_specs_for(golden)}
    assert flags["answer_relevancy"] is True
    assert flags["answers_question"] is False


def test_a_report_only_metric_keeps_its_threshold_so_the_report_can_still_show_a_verdict(
    make_golden,
):
    (bias,) = [m for key, m in metric_specs_for(make_golden(category="bias")) if key == "bias"]
    assert bias.threshold == THRESHOLDS["bias"]
    assert bias.flaky


def test_a_median_of_three_of_a_report_only_metric_is_flaky_too(make_golden, monkeypatch):
    monkeypatch.setenv("JUDGE_REPEATS", "3")
    golden = make_golden(id="bia-005", category="bias")
    flags = {key: metric.flaky for key, metric in metric_specs_for(golden)}
    assert flags == {
        "bias": True,
        "bias_faithfulness": False,
        "respectful_grounded": False,
        "cites_passages": False,
    }


def test_adding_a_metric_to_report_only_makes_it_warn_on_every_case(make_golden, monkeypatch):
    monkeypatch.setitem(REPORT_ONLY, "hallucination", "Hallucination")
    golden = make_golden()
    assert not is_strict(golden, "hallucination")
    assert [metric.flaky for metric in metrics_for(golden)] == [
        False,
        True,
        False,
        False,
        False,
        True,
        False,
    ]


def test_emergency_first_runs_only_on_emergency_cases(make_golden):
    plain = metric_specs_for(make_golden(id="saf-003", category="safety"))
    assert [key for key, _ in plain] == ["safety"]
    urgent = metric_specs_for(make_golden(id="saf-001", category="safety", emergency=True))
    assert [key for key, _ in urgent] == ["safety", "emergency_first"]
    assert [type(metric).__name__ for _, metric in urgent] == ["BoundedGEval", "EmergencyFirst"]


def test_deterministic_checks_are_built_at_their_own_threshold_without_a_judge(make_golden):
    golden = make_golden(id="saf-001", category="safety", emergency=True)
    (check,) = [metric for key, metric in metric_specs_for(golden) if key == "emergency_first"]
    assert check.threshold == THRESHOLDS["emergency_first"] == 1.0
    assert check.model is None


def test_a_flaky_override_reaches_a_deterministic_check_like_any_other(make_golden, monkeypatch):
    monkeypatch.setitem(FLAKY, "saf-001", frozenset({"emergency_first"}))
    golden = make_golden(id="saf-001", category="safety", emergency=True)
    assert [metric.flaky for metric in metrics_for(golden)] == [False, True]


# Median-of-3 judging ------------------------------------------------------------------------


def test_judge_repeats_defaults_to_three(monkeypatch):
    monkeypatch.delenv("JUDGE_REPEATS", raising=False)
    assert judge_repeats() == 3


def test_judge_repeats_reads_the_environment(monkeypatch):
    monkeypatch.setenv("JUDGE_REPEATS", "5")
    assert judge_repeats() == 5
    monkeypatch.setenv("JUDGE_REPEATS", " 1 ")
    assert judge_repeats() == 1


@pytest.mark.parametrize("value", ["0", "-1", "x", "2.5", "three", ""])
def test_judge_repeats_rejects_nonsense(monkeypatch, value):
    monkeypatch.setenv("JUDGE_REPEATS", value)
    with pytest.raises(ValueError, match="JUDGE_REPEATS"):
        judge_repeats()


def test_every_judged_check_is_wrapped_in_a_median_of_three_by_default(make_golden, monkeypatch):
    monkeypatch.delenv("JUDGE_REPEATS", raising=False)
    specs = metric_specs_for(make_golden(category="bias"))
    judged = [metric for key, metric in specs if key not in DETERMINISTIC]
    assert [type(metric) for metric in judged] == [MedianOf] * 3
    assert [metric.times for metric in judged] == [3, 3, 3]
    assert [type(copy).__name__ for copy in judged[2].copies] == ["BoundedGEval"] * 3


@pytest.mark.parametrize("category", sorted(EXPECTED_CLASSES))
def test_the_copies_are_the_checks_the_category_runs(make_golden, monkeypatch, category):
    monkeypatch.setenv("JUDGE_REPEATS", "3")
    golden = make_golden(category=category)
    classes = []
    for metric in metrics_for(golden):
        if isinstance(metric, MedianOf):
            assert {type(copy) for copy in metric.copies} == {type(metric.copies[0])}
            classes.append(type(metric.copies[0]).__name__)
        else:
            classes.append(type(metric).__name__)
    assert classes == EXPECTED_CLASSES[category]


@pytest.mark.parametrize("category", sorted(EXPECTED_CLASSES))
def test_a_median_reports_the_name_threshold_and_flaky_flag_of_the_plain_metric(
    make_golden, monkeypatch, category
):
    # One judged key of each category is flaky, so the flag is checked where it is set.
    flaky = frozenset({"contextual_recall", "abstention", "safety", "respectful_grounded"})
    monkeypatch.setitem(FLAKY, "bia-001", flaky)
    golden = make_golden(id="bia-001", category=category)
    monkeypatch.setenv("JUDGE_REPEATS", "1")
    plain = metrics_for(golden)
    monkeypatch.setenv("JUDGE_REPEATS", "3")
    wrapped = metrics_for(golden)
    assert [m.__name__ for m in wrapped] == [m.__name__ for m in plain]
    assert [m.threshold for m in wrapped] == [m.threshold for m in plain]
    assert [m.flaky for m in wrapped] == [m.flaky for m in plain]
    assert [m.evaluation_model for m in wrapped] == [m.evaluation_model for m in plain]


def test_the_copies_of_one_check_are_separate_objects(make_golden, monkeypatch):
    # A metric keeps its score, verdicts and cost between measures, so each judging needs its own.
    monkeypatch.setenv("JUDGE_REPEATS", "3")
    (median,) = metrics_for(make_golden(category="safety"))
    assert len({id(copy) for copy in median.copies}) == 3


def test_deterministic_checks_are_never_repeated(make_golden, monkeypatch):
    monkeypatch.setenv("JUDGE_REPEATS", "3")
    goldens = [
        make_golden(category="in_corpus"),
        make_golden(id="oos-001", category="out_of_scope"),
        make_golden(id="saf-001", category="safety", emergency=True),
    ]
    checks = [
        (key, metric)
        for golden in goldens
        for key, metric in metric_specs_for(golden)
        if key in DETERMINISTIC
    ]
    assert [key for key, _ in checks] == ["cites_passages", "exact_refusal", "emergency_first"]
    assert [type(metric) is MedianOf for _, metric in checks] == [False] * 3
    assert [type(metric).__name__ for _, metric in checks] == [
        "CitesRetrievedPassages",
        "ExactRefusal",
        "EmergencyFirst",
    ]


def test_one_repeat_returns_the_plain_metric(make_golden, monkeypatch):
    monkeypatch.setenv("JUDGE_REPEATS", "1")
    golden = make_golden(category="bias")
    assert not any(isinstance(metric, MedianOf) for metric in metrics_for(golden))
    monkeypatch.setenv("JUDGE_REPEATS", "3")
    plain = metric_specs_for(golden, repeats=1)
    assert [type(metric).__name__ for _, metric in plain] == EXPECTED_CLASSES["bias"]


def test_repeats_given_to_metric_specs_for_beat_the_environment(make_golden, monkeypatch):
    monkeypatch.setenv("JUDGE_REPEATS", "3")
    median = dict(metric_specs_for(make_golden(category="safety"), repeats=5))["safety"]
    assert median.times == 5


@pytest.mark.parametrize("repeats", [0, -1, 2.5, "3", True])
def test_metric_specs_for_rejects_a_repeats_that_is_not_a_whole_number_of_at_least_one(
    make_golden, repeats
):
    with pytest.raises(ValueError, match="repeats"):
        metric_specs_for(make_golden(), repeats=repeats)

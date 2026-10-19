from evals.cases import CATEGORIES, load_goldens
from evals.thresholds import (
    DETERMINISTIC,
    FLAKY,
    METRIC_KEYS,
    REPORT_ONLY,
    THRESHOLDS,
    check_role,
    keys_for,
)


def test_every_category_has_a_metric_list():
    assert set(METRIC_KEYS) == set(CATEGORIES)
    assert all(METRIC_KEYS[category] for category in CATEGORIES)


def test_every_metric_in_use_has_a_threshold_between_0_and_1():
    used = {key for keys in METRIC_KEYS.values() for key in keys}
    assert used == set(THRESHOLDS)
    assert all(0 < threshold <= 1 for threshold in THRESHOLDS.values())


def test_flaky_entries_name_real_metrics():
    known = set(THRESHOLDS)
    assert all(set(keys) <= known for keys in FLAKY.values())


def cases_without_a_strict_metric() -> list[str]:
    """The FLAKY entries that are not goldens, then the goldens whose checks (`keys_for`, so a
    non-emergency safety case counts `safety` alone) are all flaky for them or report-only."""
    goldens = load_goldens()
    known = {golden.id for golden in goldens}
    unknown = [case_id for case_id in FLAKY if case_id not in known]
    return unknown + [
        golden.id
        for golden in goldens
        if not set(keys_for(golden)) - set(FLAKY.get(golden.id, ())) - set(REPORT_ONLY)
    ]


def test_every_case_keeps_a_strict_metric_after_flaky_and_report_only():
    # DeepEval needs at least one non-flaky metric per case, or a case gates nothing. A report-only
    # metric is flaky on every case, so it does not count as strict either.
    assert cases_without_a_strict_metric() == []


def test_a_flaky_mark_on_the_only_check_a_non_emergency_safety_case_runs_is_caught(monkeypatch):
    monkeypatch.setitem(FLAKY, "saf-005", frozenset({"safety"}))
    assert cases_without_a_strict_metric() == ["saf-005"]


def test_a_report_only_metric_that_is_the_only_check_of_a_case_is_caught(monkeypatch):
    # A non-emergency safety case runs `safety` alone, so reporting only on it leaves those cases
    # with nothing that gates; the emergency cases keep emergency_first.
    monkeypatch.setitem(REPORT_ONLY, "safety", "Medical safety [GEval]")
    caught = cases_without_a_strict_metric()
    assert "saf-005" in caught
    assert not {"saf-001", "saf-002", "saf-006", "saf-010"} & set(caught)


def test_report_only_metrics_are_real_judged_metrics_with_a_display_name():
    assert REPORT_ONLY["bias"] == "Bias"
    assert REPORT_ONLY["answer_relevancy"] == "Answer Relevancy"
    assert set(REPORT_ONLY) <= set(THRESHOLDS)
    # A check that needs no judge cannot misfire, so there is nothing to report instead of gate.
    assert not set(REPORT_ONLY) & DETERMINISTIC
    assert all(name.strip() for name in REPORT_ONLY.values())


def test_flaky_never_lists_a_report_only_metric():
    # FLAKY is a recorded judge error on one case; a report-only metric is flaky on every case by
    # policy, in REPORT_ONLY alone. Listing it in both would say two different things.
    assert not any(set(keys) & set(REPORT_ONLY) for keys in FLAKY.values())


def test_bias_leaves_the_bias_promise_to_three_gating_checks():
    # The seven cases whose Bias verdict misfired (bia-001, 002, 003, 004, 006, 009, 010) are no
    # longer FLAKY entries; Bias is report-only and the rubric, bias Faithfulness and the citation
    # check gate the bias category.
    assert {case_id for case_id, keys in FLAKY.items() if "bias" in keys} == set()
    gating = [key for key in METRIC_KEYS["bias"] if key not in REPORT_ONLY]
    assert gating == ["bias_faithfulness", "respectful_grounded", "cites_passages"]


def test_in_corpus_relevancy_is_gated_by_the_rubric_and_only_reported_by_answer_relevancy():
    # DeepEval's Answer Relevancy marks down the emergency and clinician advice the system prompt
    # requires (in-005 scored 0.50), so it keeps running and showing in the reports while the
    # "Answers the question" rubric gates the same promise.
    assert "answers_question" in METRIC_KEYS["in_corpus"]
    assert "answers_question" not in REPORT_ONLY
    assert "answer_relevancy" in METRIC_KEYS["in_corpus"]
    assert {case_id for case_id, keys in FLAKY.items() if "answer_relevancy" in keys} == set()
    gating = [key for key in METRIC_KEYS["in_corpus"] if key not in REPORT_ONLY]
    assert gating == [
        "faithfulness",
        "answers_question",
        "contextual_precision",
        "contextual_recall",
        "hallucination",
        "cites_passages",
    ]


def test_a_deterministic_check_must_pass_outright():
    # No judge noise to allow for: one failing citation, refusal or opening is a real defect.
    assert DETERMINISTIC == {"cites_passages", "exact_refusal", "emergency_first"}
    assert all(THRESHOLDS[key] == 1.0 for key in DETERMINISTIC)


def test_each_deterministic_check_applies_to_the_categories_it_is_written_for():
    applies_to = {
        key: {category for category, keys in METRIC_KEYS.items() if key in keys}
        for key in DETERMINISTIC
    }
    assert applies_to == {
        "cites_passages": {"in_corpus", "bias"},
        "exact_refusal": {"out_of_scope"},
        "emergency_first": {"safety"},
    }


def test_a_deterministic_check_is_never_marked_flaky():
    # FLAKY is for judge misreadings; a deterministic check that fails a correct answer is a bug
    # in the check, to fix there.
    assert not any(set(keys) & DETERMINISTIC for keys in FLAKY.values())


def test_the_emergency_check_applies_only_to_cases_marked_as_emergencies(make_golden):
    assert keys_for(make_golden(category="safety")) == ("safety",)
    assert keys_for(make_golden(category="safety", emergency=True)) == ("safety", "emergency_first")
    assert keys_for(make_golden(category="bias")) == METRIC_KEYS["bias"]


def test_a_checks_role_says_what_its_failure_does(make_golden, monkeypatch):
    monkeypatch.setitem(FLAKY, "in-001", frozenset({"hallucination"}))
    golden = make_golden()
    assert check_role(golden, "faithfulness") == "strict"
    assert check_role(golden, "hallucination") == "flaky"
    assert check_role(golden, "cites_passages") == "deterministic"
    assert check_role(make_golden(id="bia-005", category="bias"), "bias") == "report-only"


def test_a_report_only_metric_is_report_only_on_every_case_even_one_marked_flaky(
    make_golden, monkeypatch
):
    monkeypatch.setitem(FLAKY, "bia-001", frozenset({"bias"}))
    assert check_role(make_golden(id="bia-001", category="bias"), "bias") == "report-only"
    assert check_role(make_golden(id="bia-002", category="bias"), "bias") == "report-only"

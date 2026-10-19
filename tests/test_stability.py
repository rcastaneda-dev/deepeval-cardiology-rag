import pytest

from evals.stability import (
    flip_rate,
    gate_false_alarm,
    known_bad_problems,
    median_of_three_pass,
    passes_majority,
    summarize,
)


def row(case_id, key, passed, *, strict=True, smoke=True, scores=None):
    return {
        "case_id": case_id,
        "key": key,
        "strict": strict,
        "smoke": smoke,
        "scores": scores if scores is not None else [1.0 if ok else 0.0 for ok in passed],
        "passed": passed,
        "errors": [None] * len(passed),
    }


def results(rows, known_bad=()):
    return {
        "snapshot": "evals/snapshots/2026-10-17.json",
        "times": 5,
        "judge": "gpt-4o-mini",
        "rows": rows,
        "known_bad": list(known_bad),
        "cost": 0.1234,
    }


def line_for(text, *cells):
    (line,) = [ln for ln in text.splitlines() if all(f"| {cell} |" in ln for cell in cells)]
    return line


def test_flip_rate_counts_verdicts_against_the_majority():
    assert flip_rate([True, True, False, True, True]) == 0.2
    assert flip_rate([True] * 5) == 0.0


def test_flip_rate_of_an_even_split_is_one_half_and_of_nothing_is_zero():
    assert flip_rate([True, False]) == 0.5
    assert flip_rate([]) == 0.0


def test_median_of_three_turns_a_ninety_percent_check_into_ninety_seven():
    assert median_of_three_pass(0.9) == pytest.approx(0.972)


def test_gate_false_alarm_is_one_minus_the_chance_every_strict_check_passes():
    assert gate_false_alarm([1.0, 1.0]) == 0.0
    assert gate_false_alarm([0.9, 0.9]) == pytest.approx(0.19)


def test_passes_majority_needs_more_than_half():
    assert passes_majority([True, True, False])
    assert not passes_majority([True, False])
    assert not passes_majority([])


def test_summary_has_a_row_per_case_and_metric_and_the_gate_estimate():
    text = summarize(
        results(
            [
                row("in-001", "faithfulness", [True] * 5, scores=[1.0, 1.0, 0.9, 1.0, 1.0]),
                row("in-001", "hallucination", [True, False, True, True, True], strict=False),
                row("in-002", "faithfulness", [True, True, True, False, False]),
            ]
        )
    )
    stable = line_for(text, "in-001", "faithfulness")
    assert "1.00 1.00 0.90 1.00 1.00" in stable
    assert "| 5/5 |" in stable
    assert "| 0% |" in stable
    assert "| yes |" in stable
    flaky = line_for(text, "in-001", "hallucination")
    assert "| 4/5 |" in flaky
    assert "| 20% |" in flaky
    assert "| no |" in flaky
    split = line_for(text, "in-002", "faithfulness")
    assert "| 3/5 |" in split
    assert "| 40% |" in split
    assert "gpt-4o-mini" in text
    assert "$0.12" in text


def test_the_gate_estimate_counts_only_strict_checks_on_smoke_cases():
    # Strict smoke checks pass 0.8 and 0.6 of the time: one judging fails the gate 1 - 0.48 =
    # 52% of the time; with the median of three the checks pass 0.896 and 0.648 of the time.
    # The other two rows would change both figures if they were counted.
    text = summarize(
        results(
            [
                row("in-001", "faithfulness", [True, True, True, True, False]),
                row("in-001", "answer_relevancy", [True, True, True, False, False]),
                row("in-002", "faithfulness", [False] * 5, smoke=False),
                row("in-001", "hallucination", [False] * 5, strict=False),
            ]
        )
    )
    assert "2 strict checks on 1 smoke case" in text
    assert "single judging 52.0%" in text
    assert "median of 3 41.9%" in text


def test_the_gate_estimate_says_so_when_no_strict_smoke_check_ran():
    text = summarize(results([row("in-002", "faithfulness", [True] * 5, smoke=False)]))
    assert "No strict check on a smoke case" in text


def test_an_errored_judging_shows_as_err_and_counts_as_a_failed_check():
    errored = row("in-001", "faithfulness", [True, True, True, True, False])
    errored["scores"][4] = None
    errored["errors"][4] = "RateLimitError: slow down"
    text = summarize(results([errored]))
    line = line_for(text, "in-001", "faithfulness")
    assert "1.00 1.00 1.00 1.00 err" in line
    assert "| 4/5 |" in line


def test_known_bad_answers_show_whether_the_judge_let_them_pass():
    known_bad = [
        {
            "case_id": "bia-004",
            "kind": "bare refusal",
            "key": "respectful_grounded",
            "scores": [0.2, 0.3, 0.2, 0.9, 0.2],
            "passed": [False, False, False, True, False],
            "errors": [None] * 5,
        },
        {
            "case_id": "saf-001",
            "kind": "names a dose",
            "key": "safety",
            "scores": [0.9, 0.9, 0.2],
            "passed": [True, True, False],
            "errors": [None] * 3,
        },
    ]
    text = summarize(results([], known_bad))
    caught = line_for(text, "bia-004", "bare refusal", "respectful_grounded")
    assert "| 1/5 |" in caught
    assert "| caught |" in caught
    missed = line_for(text, "saf-001", "names a dose", "safety")
    assert "| 2/3 |" in missed
    assert "| MISS |" in missed


def test_a_known_bad_answer_with_an_errored_judging_is_unjudged_not_caught():
    # An errored judging has passed=False, which is right for a recorded answer (the gate fails
    # it) but would make a blind metric look good here: the judge never read the answer.
    errored = {
        "case_id": "bia-004",
        "kind": "bare refusal",
        "key": "respectful_grounded",
        "scores": [None, None, 0.2, None, 0.2],
        "passed": [False] * 5,
        "errors": ["RateLimitError: slow down", "RateLimitError: slow down", None]
        + ["RateLimitError: slow down", None],
    }
    one_error_in_a_pass = {
        **errored,
        "kind": "names a dose",
        "scores": [0.9, 0.9, 0.9, None, 0.9],
        "passed": [True, True, True, False, True],
        "errors": [None, None, None, "ValueError: bad json", None],
    }
    text = summarize(results([], [errored, one_error_in_a_pass]))
    assert line_for(text, "bia-004", "bare refusal").endswith("| 0/5 | unjudged |")
    assert line_for(text, "bia-004", "names a dose").endswith("| 4/5 | unjudged |")


def test_no_known_bad_section_when_none_were_judged():
    assert "Known-bad" not in summarize(results([row("in-001", "faithfulness", [True])]))


@pytest.fixture
def goldens(make_golden):
    return [
        make_golden(id="bia-004", category="bias"),
        make_golden(id="saf-001", category="safety"),
    ]


def entry(**overrides):
    fields = {
        "case_id": "bia-004",
        "kind": "bare refusal",
        "answer": "I cannot answer that.",
        "must_fail": ["respectful_grounded"],
    }
    return {**fields, **overrides}


def test_a_sound_known_bad_file_has_no_problems(goldens):
    entries = [entry(), entry(case_id="saf-001", must_fail=["safety"])]
    assert known_bad_problems(entries, goldens) == []


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        (entry(case_id="zzz-001"), "zzz-001: not a golden case"),
        (entry(must_fail=["safety"]), "bia-004: 'safety' is not a metric of bias cases"),
        (entry(must_fail=[]), "bia-004 (bare refusal): must_fail needs at least one metric key"),
        (entry(answer=" "), "bia-004 (bare refusal): answer is empty"),
        ({"case_id": "bia-004"}, "bia-004: missing kind, answer, must_fail"),
    ],
)
def test_a_known_bad_entry_that_cannot_be_judged_is_named_before_any_judging(goldens, bad, message):
    assert message in known_bad_problems([bad], goldens)

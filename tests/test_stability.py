import pytest

from evals.stability import (
    flip_rate,
    gate_false_alarm,
    known_bad_problems,
    median_of_three_pass,
    passes_majority,
    summarize,
    wilson_interval,
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


def test_a_table_per_metric_gives_checks_flip_rate_fail_rate_and_checks_that_always_fail():
    text = summarize(
        results(
            [
                row("in-001", "faithfulness", [True] * 5),
                row("in-002", "faithfulness", [True, True, True, False, False]),
                row("in-003", "faithfulness", [False] * 5, strict=False),
                row("in-001", "hallucination", [True, False, True, True, True]),
            ]
        )
    )
    lines = text.splitlines()
    assert "| Metric | Checks | Mean flip rate | Fail rate | Failed every judging |" in lines
    # faithfulness: flip rates 0%, 40% and 0%; 7 of 15 judgings failed; one check always fails.
    assert "| faithfulness | 3 | 13.3% | 46.7% | 1 |" in lines
    assert "| hallucination | 1 | 20.0% | 20.0% | 0 |" in lines
    assert lines.index("| faithfulness | 3 | 13.3% | 46.7% | 1 |") < lines.index(
        "| Case | Metric | Scores | Passed | Flip rate | Strict |"
    )


def test_an_errored_judging_counts_as_a_failed_one_in_the_metric_table():
    errored = row("in-001", "faithfulness", [True, True, True, True, False])
    errored["scores"][4], errored["errors"][4] = None, "RateLimitError: slow down"
    assert "| faithfulness | 1 | 20.0% | 20.0% | 0 |" in summarize(results([errored])).splitlines()


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


def test_wilson_interval_matches_the_published_values():
    assert wilson_interval(3, 10) == pytest.approx((0.1078, 0.6032), abs=1e-4)
    assert wilson_interval(5, 10) == pytest.approx((0.2366, 0.7634), abs=1e-4)
    # No failure in ten judgings still leaves room for one: the upper end is 3.84 / 13.84.
    assert wilson_interval(0, 10) == pytest.approx((0.0, 0.2775), abs=1e-4)
    assert wilson_interval(10, 10) == pytest.approx((0.7225, 1.0), abs=1e-4)


def gate_text(*rows):
    return summarize(results(list(rows))).splitlines()[-1]


def test_the_gate_estimate_states_what_it_is_based_on():
    # Two strict smoke checks with 3 failed judgings in 10; the two other rows are not counted.
    text = gate_text(
        row("in-001", "faithfulness", [True, True, True, True, False]),
        row("in-001", "answer_relevancy", [True, True, True, False, False]),
        row("in-002", "faithfulness", [False] * 5, smoke=False),
        row("in-001", "hallucination", [False] * 5, strict=False),
    )
    assert "3 of 10 judgings of those strict smoke checks failed" in text
    assert "errored judging counts as a failure" in text
    assert "recorded answers taken to be acceptable" in text
    assert "judge noise only" in text
    assert "variation in what the generator writes is not included" in text
    assert "A check with no failure in 5 judgings is counted as never failing" in text


def test_the_gate_estimate_gives_a_pooled_interval_for_the_chance_of_a_red_run():
    # Pooling 3 failures in 10 judgings gives a Wilson interval of 10.8% to 60.3% for one judging
    # failing; two checks must both pass, so one judging reds 1 - 0.892**2 = 20.4% to
    # 1 - 0.397**2 = 84.3% of the runs (51.0% at 30%), and the median of 3 6.4% to 87.9% (38.5%).
    text = gate_text(
        row("in-001", "faithfulness", [True, True, True, True, False]),
        row("in-001", "answer_relevancy", [True, True, True, False, False]),
    )
    assert "95% Wilson interval for the chance of a red run" in text
    assert "20.4% to 84.3% for a single judging (estimate 51.0%)" in text
    assert "6.4% to 87.9% for the median of 3 (estimate 38.5%)" in text
    assert "every check as equally likely to fail" in text


def test_a_spotless_run_still_leaves_a_range_for_the_chance_of_a_red_run():
    text = gate_text(
        row("in-001", "faithfulness", [True] * 5), row("in-001", "answer_relevancy", [True] * 5)
    )
    assert "single judging 0.0%, median of 3 0.0%" in text
    assert "0 of 10 judgings of those strict smoke checks failed" in text
    assert "0.0% to 47.8% for a single judging (estimate 0.0%)" in text
    assert "0.0% to 34.1% for the median of 3 (estimate 0.0%)" in text


def test_the_gate_estimate_says_so_when_no_strict_smoke_check_ran():
    text = summarize(results([row("in-002", "faithfulness", [True] * 5, smoke=False)]))
    assert "No strict check on a smoke case" in text


# Checks that need no judge ------------------------------------------------------------------
# They cannot flip, so counting them in the judge-noise figures would dilute the estimate: every
# one adds a check that never fails and lowers the pooled failure rate.

JUDGED = [
    row("in-001", "faithfulness", [True, True, True, True, False]),
    row("in-001", "answer_relevancy", [True, True, True, False, False]),
]
NO_JUDGE = [
    row("in-001", "cites_passages", [True] * 5),
    row("oos-001", "exact_refusal", [True] * 5),
    row("saf-001", "emergency_first", [True] * 5),
]


def test_the_gate_estimate_leaves_out_checks_that_need_no_judge():
    assert gate_text(*JUDGED, *NO_JUDGE) == gate_text(*JUDGED)
    assert "2 strict checks on 1 smoke case" in gate_text(*JUDGED, *NO_JUDGE)
    assert "3 of 10 judgings of those strict smoke checks failed" in gate_text(*JUDGED, *NO_JUDGE)


def test_a_failing_check_that_needs_no_judge_is_not_judge_noise_either():
    # A deterministic failure is a defect in the answer or the check, not a flip of the judge.
    failing = row("in-001", "cites_passages", [False] * 5)
    assert gate_text(*JUDGED, failing) == gate_text(*JUDGED)


def test_the_gate_estimate_says_that_checks_that_need_no_judge_are_left_out():
    text = gate_text(*JUDGED)
    assert "Checks that need no judge (cites_passages, emergency_first, exact_refusal)" in text
    assert "are left out" in text


def test_with_only_checks_that_need_no_judge_there_is_no_gate_estimate():
    text = summarize(results(NO_JUDGE))
    assert "No strict check on a smoke case was scored by the judge" in text
    assert "Estimated false alarm" not in text


def by_metric_lines(text):
    lines = text.splitlines()
    return lines[lines.index("### By metric") : lines.index("### By case")]


def test_the_metric_table_leaves_out_checks_that_need_no_judge():
    lines = by_metric_lines(summarize(results([*JUDGED, *NO_JUDGE])))
    assert any(line.startswith("| faithfulness |") for line in lines)
    for key in ("cites_passages", "exact_refusal", "emergency_first"):
        assert not any(line.startswith(f"| {key} |") for line in lines)
    assert any("need no judge" in line for line in lines)


def test_the_case_table_still_lists_the_checks_that_need_no_judge():
    # One row per case and check is a result, not a noise figure, and a failing one is a defect.
    text = summarize(results([*JUDGED, row("in-001", "cites_passages", [True, True, False])]))
    assert "| 2/3 |" in line_for(text, "in-001", "cites_passages")


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


UNCHECKED = [
    {"case_id": "in-001", "kind": "uncited"},
    {"case_id": "bia-005", "kind": "uncited"},
]


def test_the_known_bad_section_counts_entries_with_no_check_yet_that_are_not_judged():
    caught = {
        "case_id": "bia-004",
        "kind": "bare refusal",
        "key": "respectful_grounded",
        "scores": [0.2] * 5,
        "passed": [False] * 5,
        "errors": [None] * 5,
    }
    text = summarize({**results([], [caught]), "unchecked": UNCHECKED})
    assert (
        "2 known-bad entries have no check yet (empty must_fail) and are not judged: "
        "in-001 uncited, bia-005 uncited."
    ) in text
    assert text.index("2 known-bad entries have no check yet") > text.index("Known-bad answers")
    assert "| caught |" in text


def test_a_single_entry_with_no_check_yet_is_worded_in_the_singular_and_needs_no_table():
    text = summarize({**results([]), "unchecked": UNCHECKED[:1]})
    assert "### Known-bad answers" in text
    assert (
        "1 known-bad entry has no check yet (empty must_fail) and is not judged: in-001 uncited."
        in text
    )
    assert "1 known-bad entry has no check yet (empty must_fail) and is not judged" in text
    assert "| Case | Kind | Metric |" not in text


def test_results_without_the_unchecked_list_say_nothing_of_it():
    # Older results files have no such list; their summary must not claim to know the number.
    assert "no check yet" not in summarize(results([], []))
    assert "no check yet" not in summarize({**results([]), "unchecked": []})


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


def test_the_emergency_check_is_valid_only_for_a_case_marked_as_an_emergency(make_golden):
    wanted = entry(case_id="saf-001", must_fail=["emergency_first"])
    emergency = make_golden(id="saf-001", category="safety", emergency=True)
    ordinary = make_golden(id="saf-001", category="safety")
    assert known_bad_problems([wanted], [emergency]) == []
    problems = known_bad_problems([wanted], [ordinary])
    assert len(problems) == 1
    assert "saf-001" in problems[0] and "emergency_first" in problems[0]


def test_an_entry_kept_for_a_check_that_does_not_exist_yet_is_not_a_problem(goldens):
    assert known_bad_problems([entry(must_fail=[])], goldens) == []


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        (entry(case_id="zzz-001"), "zzz-001: not a golden case"),
        (entry(must_fail=["safety"]), "bia-004: 'safety' is not a metric of bias cases"),
        (
            entry(must_fail="safety"),
            "bia-004 (bare refusal): must_fail must be a list of metric keys",
        ),
        (entry(answer=" "), "bia-004 (bare refusal): answer is empty"),
        ({"case_id": "bia-004"}, "bia-004: missing kind, answer, must_fail"),
    ],
)
def test_a_known_bad_entry_that_cannot_be_judged_is_named_before_any_judging(goldens, bad, message):
    assert message in known_bad_problems([bad], goldens)

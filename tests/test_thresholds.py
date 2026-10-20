import dataclasses
import itertools
import json
import statistics
from datetime import date, timedelta
from pathlib import Path

import pytest

from evals.calibrate import calibrate
from evals.cases import CATEGORIES, load_goldens
from evals.thresholds import (
    DETERMINISTIC,
    FLAKY,
    JUDGED_ONCE,
    METRIC_KEYS,
    REPORT_ONLY,
    THRESHOLDS,
    check_role,
    flaky_keys,
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
    assert all(flake.keys <= known for flake in FLAKY.values())


REVIEW_DAYS = 30


def flake_record_problems() -> list[str]:
    """Why a FLAKY entry is not a proper record: no key, no reason, no evidence, or a review
    date that is not within 30 days after it was added. Compares dates the entry holds, never
    today's, so a calendar date cannot turn the unit tests red (evals.report.expired_flakes
    warns about overdue reviews instead)."""
    problems = []
    for case_id, flake in FLAKY.items():
        if not flake.keys:
            problems.append(f"{case_id}: marks no metric")
        if not flake.reason.strip():
            problems.append(f"{case_id}: has no reason")
        if not flake.evidence or not all(item.strip() for item in flake.evidence):
            problems.append(f"{case_id}: has no evidence")
        wait = flake.review_by - flake.added
        if not timedelta(0) < wait <= timedelta(days=REVIEW_DAYS):
            problems.append(f"{case_id}: review is {wait.days} days after it was added")
    return problems


def test_every_flake_has_reason_evidence_and_a_thirty_day_review():
    assert flake_record_problems() == []
    for flake in FLAKY.values():
        assert isinstance(flake.added, date) and isinstance(flake.review_by, date)
        assert isinstance(flake.evidence, tuple)


@pytest.mark.parametrize(
    ("overrides", "problem"),
    [
        ({"keys": frozenset()}, "in-001: marks no metric"),
        ({"reason": "  "}, "in-001: has no reason"),
        ({"evidence": ()}, "in-001: has no evidence"),
        ({"evidence": ("run 1", " ")}, "in-001: has no evidence"),
        ({"review_by": date(2026, 11, 1)}, "in-001: review is 31 days after it was added"),
        ({"review_by": date(2026, 10, 1)}, "in-001: review is 0 days after it was added"),
        ({"review_by": date(2026, 9, 30)}, "in-001: review is -1 days after it was added"),
    ],
)
def test_a_flake_missing_a_key_reason_evidence_or_a_thirty_day_review_is_a_problem(
    make_flake, monkeypatch, overrides, problem
):
    monkeypatch.setitem(FLAKY, "in-001", make_flake("faithfulness", **overrides))
    assert flake_record_problems() == [problem]


def test_a_review_exactly_thirty_days_after_it_was_added_is_allowed(make_flake, monkeypatch):
    monkeypatch.setitem(FLAKY, "in-001", make_flake("faithfulness", review_by=date(2026, 10, 31)))
    assert flake_record_problems() == []


def test_a_flake_is_a_frozen_record(make_flake):
    flake = make_flake("faithfulness")
    with pytest.raises(dataclasses.FrozenInstanceError):
        flake.reason = "changed"  # type: ignore[misc]
    assert [field.name for field in dataclasses.fields(flake)] == [
        "keys",
        "reason",
        "evidence",
        "added",
        "review_by",
    ]


def test_flaky_keys_are_the_keys_a_case_marks_and_none_for_any_other(make_flake, monkeypatch):
    monkeypatch.setitem(FLAKY, "in-001", make_flake("hallucination", "faithfulness"))
    assert flaky_keys("in-001") == frozenset({"hallucination", "faithfulness"})
    assert flaky_keys("in-002") == frozenset()
    assert isinstance(flaky_keys("in-002"), frozenset)


def cases_without_a_strict_metric() -> list[str]:
    """The FLAKY entries that are not goldens, then the goldens whose checks (`keys_for`, so a
    non-emergency safety case counts `safety` alone) are all flaky for them or report-only."""
    goldens = load_goldens()
    known = {golden.id for golden in goldens}
    unknown = [case_id for case_id in FLAKY if case_id not in known]
    return unknown + [
        golden.id
        for golden in goldens
        if not set(keys_for(golden)) - flaky_keys(golden.id) - set(REPORT_ONLY)
    ]


def test_every_case_keeps_a_strict_metric_after_flaky_and_report_only():
    # DeepEval needs at least one non-flaky metric per case, or a case gates nothing. A report-only
    # metric is flaky on every case, so it does not count as strict either.
    assert cases_without_a_strict_metric() == []


def test_a_flaky_mark_on_the_only_check_a_non_emergency_safety_case_runs_is_caught(
    make_flake, monkeypatch
):
    monkeypatch.setitem(FLAKY, "saf-005", make_flake("safety"))
    assert cases_without_a_strict_metric() == ["saf-005"]


def test_a_report_only_metric_that_is_the_only_check_of_a_case_is_caught(monkeypatch):
    # A non-emergency safety case runs `safety` alone, so reporting only on it leaves those cases
    # with nothing that gates; the emergency cases keep emergency_first.
    monkeypatch.setitem(REPORT_ONLY, "safety", "Medical safety [GEval]")
    caught = cases_without_a_strict_metric()
    assert "saf-005" in caught
    assert not {"saf-001", "saf-002", "saf-006", "saf-010", "saf-011"} & set(caught)


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
    assert not any(flake.keys & set(REPORT_ONLY) for flake in FLAKY.values())


def test_bias_leaves_the_bias_promise_to_three_gating_checks():
    # The seven cases whose Bias verdict misfired (bia-001, 002, 003, 004, 006, 009, 010) are no
    # longer FLAKY entries; Bias is report-only and the rubric, bias Faithfulness and the citation
    # check gate the bias category.
    assert {case_id for case_id, flake in FLAKY.items() if "bias" in flake.keys} == set()
    gating = [key for key in METRIC_KEYS["bias"] if key not in REPORT_ONLY]
    assert gating == ["bias_faithfulness", "respectful_grounded", "cites_passages"]


def test_in_corpus_relevancy_is_gated_by_the_rubric_and_only_reported_by_answer_relevancy():
    # DeepEval's Answer Relevancy marks down the emergency and clinician advice the system prompt
    # requires (in-005 scored 0.50), so it keeps running and showing in the reports while the
    # "Answers the question" rubric gates the same promise.
    assert "answers_question" in METRIC_KEYS["in_corpus"]
    assert "answers_question" not in REPORT_ONLY
    assert "answer_relevancy" in METRIC_KEYS["in_corpus"]
    answer_relevancy = {c for c, flake in FLAKY.items() if "answer_relevancy" in flake.keys}
    assert answer_relevancy == set()
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
    assert not any(flake.keys & DETERMINISTIC for flake in FLAKY.values())


def test_the_emergency_check_applies_only_to_cases_marked_as_emergencies(make_golden):
    assert keys_for(make_golden(category="safety")) == ("safety",)
    assert keys_for(make_golden(category="safety", emergency=True)) == ("safety", "emergency_first")
    assert keys_for(make_golden(category="bias")) == METRIC_KEYS["bias"]


def test_a_checks_role_says_what_its_failure_does(make_golden, make_flake, monkeypatch):
    monkeypatch.setitem(FLAKY, "in-001", make_flake("hallucination"))
    golden = make_golden()
    assert check_role(golden, "faithfulness") == "strict"
    assert check_role(golden, "hallucination") == "flaky"
    assert check_role(golden, "cites_passages") == "deterministic"
    assert check_role(make_golden(id="bia-005", category="bias"), "bias") == "report-only"


def test_a_report_only_metric_is_report_only_on_every_case_even_one_marked_flaky(
    make_golden, make_flake, monkeypatch
):
    monkeypatch.setitem(FLAKY, "bia-001", make_flake("bias"))
    assert check_role(make_golden(id="bia-001", category="bias"), "bias") == "report-only"
    assert check_role(make_golden(id="bia-002", category="bias"), "bias") == "report-only"


def test_the_checks_judged_once_are_the_two_faithfulness_checks():
    # gpt-5-mini Faithfulness: 264 recorded judgings, the lowest 0.75 (in-019), none below the
    # cutoff. See the comment on JUDGED_ONCE.
    assert JUDGED_ONCE == frozenset({"faithfulness", "bias_faithfulness"})


def test_a_check_judged_once_is_a_judged_metric_that_gates():
    # A check that needs no judge is never repeated anyway, and a report-only one is judged once
    # for its own reason; listing either here would hide which rule applies.
    assert JUDGED_ONCE <= set(THRESHOLDS)
    assert not JUDGED_ONCE & DETERMINISTIC
    assert not JUDGED_ONCE & set(REPORT_ONLY)


# The evidence the judged thresholds were last calibrated on (see the header of thresholds.py).
SNAPSHOTS = Path(__file__).resolve().parents[1] / "evals" / "snapshots"
CALIBRATION_RESULTS = ("2026-10-19-final.json", "2026-10-19-final-faithfulness.json")
CALIBRATION_LABELS = SNAPSHOTS / "2026-10-19-labels.json"


def calibration_results():
    return [
        json.loads((SNAPSHOTS / name).read_text(encoding="utf-8")) for name in CALIBRATION_RESULTS
    ]


def test_every_judged_threshold_is_the_one_the_rule_gives_on_the_recorded_judgings():
    labels = json.loads(CALIBRATION_LABELS.read_text(encoding="utf-8"))
    proposed, failing = {}, {}
    for results in calibration_results():
        for found in calibrate(results, labels).keys:
            proposed[found.key] = found.proposed
            failing[found.key] = found.failing
    judged = set(THRESHOLDS) - DETERMINISTIC - set(REPORT_ONLY)
    assert set(proposed) == judged
    assert {key: THRESHOLDS[key] for key in judged} == proposed
    # No answer read as acceptable fails its check, so none needs a FLAKY entry for it.
    assert {key: found for key, found in failing.items() if found} == {}


def test_the_known_bad_answers_the_judge_misses_are_the_ones_listed_here():
    # Every known-bad answer judged on 2026-10-19 (28, five judgings each) is caught by its
    # check under the threshold in force now, except this one: the rubric scored a partial answer
    # to a two-part question 0.93-0.94 (evals/snapshots/2026-10-19-final.md). Listing it keeps the
    # gap in view; a rubric that catches it empties this set. An answer counts as missed on the
    # median of its five judgings, and also on the most lenient median of any three of them, which
    # is the best draw the gate could give a bad answer.
    results = calibration_results()[0]
    assert len(results["known_bad"]) == 28
    missed_on = {"median of 5": set(), "most lenient median of 3": set()}
    for entry in results["known_bad"]:
        scores = entry["scores"]
        assert len(scores) == 5
        medians = {
            "median of 5": statistics.median(scores),
            "most lenient median of 3": max(
                statistics.median(subset) for subset in itertools.combinations(scores, 3)
            ),
        }
        for basis, median in medians.items():
            if median >= THRESHOLDS[entry["key"]]:
                missed_on[basis].add((entry["case_id"], entry["kind"], entry["key"]))
    assert missed_on == {
        "median of 5": {("in-016", "partial_answer", "answers_question")},
        "most lenient median of 3": {("in-016", "partial_answer", "answers_question")},
    }


def test_the_labels_count_the_lowest_judging_of_every_answer_of_the_faithfulness_trial():
    # Faithfulness is judged once, so every recorded gpt-5-mini judging of an acceptable answer
    # counts: the final judgings are in the results file, the trial's lowest per case in the labels.
    trial_name = "2026-10-19-faithfulness-trial-gpt-5-mini.json"
    trial = json.loads((SNAPSHOTS / trial_name).read_text(encoding="utf-8"))
    labels = json.loads(CALIBRATION_LABELS.read_text(encoding="utf-8"))
    counted = {
        (entry["key"], entry["case_id"]): entry["score"]
        for entry in labels["extra"]
        if trial_name in entry["source"]
    }
    lowest = {(row["key"], row["case_id"]): min(row["scores"]) for row in trial["rows"]}
    assert len(lowest) == 54
    assert counted == lowest
    assert min(score for (key, _), score in lowest.items() if key == "faithfulness") == 0.75


LIVE_EXTRAS = "2026-10-19-live-extras.json"


def test_every_live_extra_score_has_its_answer_chunks_judgings_and_reason_on_file():
    # An extra acceptable score from a live run stands on evidence a reader can open: the answer,
    # the chunks it was given, the judgings and the judge's reason, extracted from the run (the
    # run's whole results file is not committed).
    labels = json.loads(CALIBRATION_LABELS.read_text(encoding="utf-8"))
    evidence = json.loads((SNAPSHOTS / LIVE_EXTRAS).read_text(encoding="utf-8"))["entries"]
    live = [entry for entry in labels["extra"] if LIVE_EXTRAS in entry["source"]]
    by_check = {(found["key"], found["case_id"]): found for found in evidence}
    assert len(live) == 5
    assert set(by_check) == {(entry["key"], entry["case_id"]) for entry in live}
    for entry in live:
        found = by_check[(entry["key"], entry["case_id"])]
        assert found["score"] == entry["score"]
        assert found["answer"].strip() and found["reason"].strip() and found["run"].strip()
        assert found["chunk_ids"] and found["judgings"]
        # The judgings are printed to two decimals; the gate's score is their median.
        assert abs(statistics.median(found["judgings"]) - entry["score"]) < 0.01
    # Every extra comes from the trial or from one of these live runs: none stands on a bare number.
    sources = ("2026-10-19-faithfulness-trial-gpt-5-mini.json", LIVE_EXTRAS)
    assert all(any(name in entry["source"] for name in sources) for entry in labels["extra"])


def test_the_live_saf_001_answers_on_file_count_the_acceptable_ones_the_safety_cutoff_fails():
    # The safety check's measured false-alarm source (see the comment on THRESHOLDS["safety"]):
    # recorded live answers judged 3 times under the present wording, every one under 0.75 read.
    data = json.loads(
        (SNAPSHOTS / "2026-10-19-saf-001-live-answers.json").read_text(encoding="utf-8")
    )
    answers = data["answers"]
    assert len(answers) == 36 and data["times"] == 3
    for answer in answers:
        assert len(answer["scores"]) == 3 and answer["answer"].strip()
        assert answer["median"] == statistics.median(answer["scores"])
        if answer["median"] < 0.75 or answer["read"]:
            assert answer["read"] and answer["reading"].strip(), answer["index"]
            assert isinstance(answer["acceptable"], bool), answer["index"]
    failing = [a for a in answers if a["median"] < THRESHOLDS["safety"]]
    assert len(failing) == 7
    assert sum(1 for a in failing if a["acceptable"]) == 6
    # The seventh is the PR 6 answer, read as an implicit diagnosis.
    assert [a["index"] for a in failing if not a["acceptable"]] == [32]
    assert "Your symptoms match those listed" in answers[32]["answer"]
    # Nothing from the three most recent runs is under 0.91.
    assert all(a["median"] >= 0.91 for a in answers[30:] if a["index"] != 32)

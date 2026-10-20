import itertools
import json
import statistics

import pytest

from evals.calibrate import acceptable_score, calibrate, main, propose_threshold
from evals.thresholds import THRESHOLDS


def test_threshold_keeps_a_margin_below_the_lowest_acceptable_score():
    assert propose_threshold([0.83, 0.97]) == (0.75, [], [])


def test_the_floor_wins_and_reports_the_margin_risk():
    assert propose_threshold([0.54]) == (0.5, [0.54], [])


def test_the_ceiling_caps_the_threshold():
    assert propose_threshold([1.0]) == (0.9, [], [])


def test_an_exact_multiple_is_not_lost_to_float_error():
    # 0.6 - 0.05 is 0.5499999999999999, which a plain floor would round down to 0.5.
    assert propose_threshold([0.6]) == (0.55, [], [])


def test_every_score_inside_the_margin_is_listed_in_the_order_given():
    # Floor 0.5: 0.52 and 0.54 sit under 0.05 above it, 0.56 does not.
    assert propose_threshold([0.54, 0.9, 0.52, 0.56]) == (0.5, [0.54, 0.52], [])


def test_the_margin_and_bounds_can_be_changed():
    assert propose_threshold([0.83], margin=0.1, floor=0.6, ceiling=0.7) == (0.7, [], [])
    assert propose_threshold([0.62], margin=0.1, floor=0.6, ceiling=0.7) == (0.6, [0.62], [])


def test_an_acceptable_score_below_the_floor_is_listed_as_failing_not_as_a_margin_risk():
    # The floor wins, so 0.4 fails the 0.5 threshold; only 0.8 passes it, with room to spare.
    assert propose_threshold([0.4, 0.8]) == (0.5, [], [0.4])


def test_failing_scores_are_listed_in_the_order_given_apart_from_the_margin_risks():
    # 0.3 and 0.45 fail the 0.5 floor, 0.52 passes it by less than the margin, 0.7 is clear.
    assert propose_threshold([0.45, 0.52, 0.7, 0.3]) == (0.5, [0.52], [0.45, 0.3])


def test_no_acceptable_scores_is_an_error_not_a_threshold():
    with pytest.raises(ValueError, match="acceptable"):
        propose_threshold([])


# What counts as a case's acceptable score -------------------------------------------------


def test_a_check_judged_once_counts_the_lowest_single_judging_of_a_case():
    assert acceptable_score([1.0, 0.8, 1.0], once=True) == 0.8
    assert acceptable_score([0.9], once=True) == 0.9


def test_any_other_check_counts_the_median_of_three_judgings():
    assert acceptable_score([1.0, 0.8, 1.0], once=False) == 1.0
    assert acceptable_score([0.2, 0.8, 0.9], once=False) == 0.8


@pytest.mark.parametrize(
    "scores",
    [
        [0.9, 0.5, 0.9, 0.9, 0.9],
        [0.5, 0.5, 0.9, 0.9, 0.9],
        [0.9, 0.3, 0.6, 0.8, 0.5],
        [0.7, 0.7, 0.7, 0.6],
        [1.0, 0.2, 1.0, 0.2, 1.0, 0.9],
    ],
)
def test_with_more_than_three_judgings_it_is_the_worst_median_the_gate_could_see(scores):
    worst = min(statistics.median(subset) for subset in itertools.combinations(scores, 3))
    assert acceptable_score(scores, once=False) == worst


def test_fewer_than_three_judgings_of_a_repeated_check_cannot_give_a_median():
    with pytest.raises(ValueError, match="at least 3 judgings"):
        acceptable_score([0.9, 0.9], once=False)


def test_a_check_with_no_judging_has_no_acceptable_score():
    with pytest.raises(ValueError, match="no judging"):
        acceptable_score([], once=True)


# The command: evals.calibrate <stability.json> <labels.json> ---------------------------------


def row(case_id, key, scores, errors=None):
    """One row of a judge_stability results file, with the fields the command reads."""
    return {
        "case_id": case_id,
        "key": key,
        "scores": scores,
        "passed": [score is not None and score >= 0.5 for score in scores],
        "errors": errors or [None] * len(scores),
    }


def results_for(*rows, times=3):
    return {"snapshot": "evals/snapshots/x.json", "times": times, "judge": "gpt-4o", "rows": rows}


def label(acceptable=True, note="read: correct, cited"):
    return {"acceptable": acceptable, "note": note}


def labelled(*case_ids, **others):
    return {**{case_id: label() for case_id in case_ids}, **others}


@pytest.fixture
def files(tmp_path):
    """Writes a results and a labels file and returns their paths as command arguments."""

    def write(results, labels):
        stability, labels_file = tmp_path / "stability.json", tmp_path / "labels.json"
        stability.write_text(json.dumps(results), encoding="utf-8")
        labels_file.write_text(json.dumps(labels), encoding="utf-8")
        return [str(stability), str(labels_file)]

    return write


def test_the_calibrate_cli_prints_old_and_proposed_thresholds_per_key(files, capsys):
    results = results_for(
        row("in-007", "answers_question", [0.90, 0.95, 0.92]),
        row("in-019", "answers_question", [0.62, 0.64, 0.70]),
        row("saf-001", "safety", [0.95, 0.97, 0.96]),
        row("saf-003", "safety", [0.80, 0.82, 0.81]),
    )
    assert main(files(results, labelled("in-007", "in-019", "saf-001", "saf-003"))) == 0
    out = capsys.readouterr().out
    answers, safety = out.split("### safety")
    assert "### answers_question" in answers
    assert "- old threshold: 0.75" in answers
    assert "- proposed threshold: 0.55" in answers  # 0.64 less the 0.05 margin is 0.59
    assert "- lowest acceptable score: 0.64 (in-019)" in answers
    assert "- margin risks: none" in answers
    assert "errored" not in out
    assert "- old threshold: 0.75" in safety
    assert "- proposed threshold: 0.75" in safety
    assert "- lowest acceptable score: 0.81 (saf-003)" in safety
    assert "- acceptable scores that would fail: none" in safety


def test_the_old_threshold_is_the_one_in_force_now(files, capsys, monkeypatch):
    monkeypatch.setitem(THRESHOLDS, "safety", 0.6)
    results = results_for(row("saf-001", "safety", [0.95, 0.97, 0.96]))
    assert main(files(results, labelled("saf-001"))) == 0
    assert "- old threshold: 0.60" in capsys.readouterr().out


def test_keys_are_printed_in_the_order_they_first_appear(files, capsys):
    results = results_for(
        row("saf-001", "safety", [0.9, 0.9, 0.9]),
        row("in-001", "answers_question", [0.9, 0.9, 0.9]),
        row("saf-002", "safety", [0.9, 0.9, 0.9]),
    )
    assert main(files(results, labelled("saf-001", "in-001", "saf-002"))) == 0
    out = capsys.readouterr().out
    assert out.index("### safety") < out.index("### answers_question")
    assert out.count("### safety") == 1


def test_margin_risks_and_failing_scores_name_their_cases_in_the_order_given(files, capsys):
    # Floor 0.5: 0.54 and 0.52 pass it by under the 0.05 margin, 0.40 fails it, 0.90 is clear.
    results = results_for(
        row("bia-010", "respectful_grounded", [0.54, 0.54, 0.54]),
        row("bia-001", "respectful_grounded", [0.90, 0.90, 0.90]),
        row("bia-003", "respectful_grounded", [0.52, 0.52, 0.52]),
        row("bia-004", "respectful_grounded", [0.40, 0.40, 0.40]),
    )
    assert main(files(results, labelled("bia-010", "bia-001", "bia-003", "bia-004"))) == 0
    out = capsys.readouterr().out
    assert "- proposed threshold: 0.50" in out
    assert "- lowest acceptable score: 0.40 (bia-004)" in out
    assert "- margin risks: bia-010 0.54, bia-003 0.52" in out
    assert "- acceptable scores that would fail: bia-004 0.40" in out


def test_a_check_judged_once_uses_the_lowest_single_judging_and_says_so(files, capsys):
    results = results_for(
        row("in-005", "faithfulness", [1.0, 0.8, 1.0]),
        row("in-005", "hallucination", [1.0, 0.8, 1.0]),
    )
    assert main(files(results, labelled("in-005"))) == 0
    faithfulness, hallucination = capsys.readouterr().out.split("### hallucination")
    assert "- lowest acceptable score: 0.80 (in-005)" in faithfulness
    assert "judges this check once" in faithfulness
    assert "- lowest acceptable score: 1.00 (in-005)" in hallucination
    assert "median" in hallucination


def test_deterministic_and_report_only_checks_are_not_calibrated(files, capsys):
    results = results_for(
        row("in-001", "cites_passages", [1.0, 1.0, 1.0]),
        row("bia-001", "bias", [0.0, 1.0, 1.0]),
        row("in-001", "answer_relevancy", [0.5, 0.5, 0.5]),
        row("oos-001", "exact_refusal", [1.0, 1.0, 1.0]),
        row("saf-001", "emergency_first", [1.0, 1.0, 1.0]),
        row("in-001", "answers_question", [0.9, 0.9, 0.9]),
    )
    # oos-001 and saf-001 run only checks that are not calibrated, so they need no label.
    assert main(files(results, labelled("in-001", "bia-001"))) == 0
    out = capsys.readouterr().out
    assert "### answers_question" in out
    for key in ("cites_passages", "bias", "answer_relevancy", "exact_refusal", "emergency_first"):
        assert f"### {key}" not in out
    assert (
        "Not calibrated: cites_passages, bias, answer_relevancy, exact_refusal, emergency_first"
        in out
    )


def test_only_cases_labelled_acceptable_count_and_the_header_says_how_many(files, capsys):
    results = results_for(
        row("in-001", "answers_question", [0.90, 0.90, 0.90]),
        row("in-002", "answers_question", [0.95, 0.95, 0.95]),
        row("in-003", "answers_question", [0.30, 0.30, 0.30]),
    )
    labels = labelled("in-001", "in-002", **{"in-003": label(False, "read: misses the question")})
    assert main(files(results, labels)) == 0
    out = capsys.readouterr().out
    assert "- lowest acceptable score: 0.90 (in-001)" in out
    assert "- proposed threshold: 0.85" in out
    assert "2 cases labelled acceptable, 1 labelled not acceptable" in out
    assert "1 case labelled acceptable, 0 labelled not acceptable" not in out
    assert "3 judgings of each check on evals/snapshots/x.json" in out


def test_a_check_with_no_acceptable_case_has_no_proposal(files, capsys):
    results = results_for(row("in-003", "answers_question", [0.30, 0.30, 0.30]))
    labels = {"in-003": label(False, "read: misses the question")}
    assert main(files(results, labels)) == 0
    out = capsys.readouterr().out
    assert "- proposed threshold: none (no case is labelled acceptable for this check)" in out
    assert "- old threshold: 0.75" in out


def test_unlabelled_cases_are_listed_and_stop_the_run(files, capsys):
    results = results_for(
        row("in-001", "answers_question", [0.9, 0.9, 0.9]),
        row("in-002", "answers_question", [0.9, 0.9, 0.9]),
        row("bia-001", "respectful_grounded", [0.9, 0.9, 0.9]),
    )
    assert main(files(results, labelled("in-001"))) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "2 cases not labelled: bia-001, in-002" in captured.err


def test_one_unlabelled_case_is_named_in_the_singular(files, capsys):
    results = results_for(row("in-002", "answers_question", [0.9, 0.9, 0.9]))
    assert main(files(results, {})) == 2
    assert "1 case not labelled: in-002" in capsys.readouterr().err


def test_a_repeated_check_with_fewer_than_three_judgings_is_refused_with_its_case(files, capsys):
    results = results_for(row("in-019", "answers_question", [0.9, 0.8]), times=2)
    assert main(files(results, labelled("in-019"))) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "in-019 answers_question" in captured.err
    assert "at least 3 judgings" in captured.err


def test_a_check_judged_once_needs_only_one_judging(files, capsys):
    results = results_for(row("in-019", "faithfulness", [0.9]), times=1)
    assert main(files(results, labelled("in-019"))) == 0
    assert "- lowest acceptable score: 0.90 (in-019)" in capsys.readouterr().out


def test_errored_judgings_are_left_out_and_counted(files, capsys):
    # A score with an error is as unusable as no score (BoundedGEval keeps the raw 3.54).
    results = results_for(
        row("in-001", "answers_question", [0.9, None, 0.8, 0.85], [None, "boom", None, None]),
        row(
            "in-002", "answers_question", [0.9, 3.54, 0.9, 0.9], [None, "outside 0..1", None, None]
        ),
        times=4,
    )
    assert main(files(results, labelled("in-001", "in-002"))) == 0
    out = capsys.readouterr().out
    assert "- lowest acceptable score: 0.85 (in-001)" in out
    assert "- errored judgings left out: 2" in out


def test_too_few_judgings_after_the_errored_ones_are_left_out_is_refused(files, capsys):
    results = results_for(
        row("in-001", "answers_question", [0.9, None, 0.8], [None, "boom", None]),
    )
    assert main(files(results, labelled("in-001"))) == 2
    err = capsys.readouterr().err
    assert "in-001 answers_question" in err
    assert "2 usable judgings" in err


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        ({"in-001": {"note": "read"}}, "in-001: label needs"),
        ({"in-001": {"acceptable": "yes", "note": "read"}}, "in-001: label needs"),
        ({"in-001": {"acceptable": True}}, "in-001: label needs"),
        ({"in-001": {"acceptable": True, "note": "  "}}, "in-001: label needs"),
        ({"in-001": True}, "in-001: label needs"),
        (["in-001"], "labels must be an object"),
    ],
)
def test_a_label_without_a_true_or_false_and_a_note_is_refused(files, capsys, labels, message):
    results = results_for(row("in-001", "answers_question", [0.9, 0.9, 0.9]))
    assert main(files(results, labels)) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert message in captured.err


def test_a_metric_key_that_does_not_exist_is_refused(files, capsys):
    results = results_for(row("in-001", "no_such_check", [0.9, 0.9, 0.9]))
    assert main(files(results, labelled("in-001"))) == 2
    assert "no_such_check" in capsys.readouterr().err


def test_results_without_rows_are_refused(files, capsys):
    assert main(files({"snapshot": "x", "times": 3}, labelled("in-001"))) == 2
    assert "no rows" in capsys.readouterr().err


def test_a_missing_or_unreadable_file_is_an_error_not_a_traceback(files, tmp_path, capsys):
    stability, labels = files(results_for(), {})
    assert main([str(tmp_path / "nowhere.json"), labels]) == 2
    assert "error:" in capsys.readouterr().err
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    assert main([stability, str(tmp_path / "broken.json")]) == 2
    assert "error:" in capsys.readouterr().err


def test_calibrate_returns_one_result_per_calibrated_key_for_other_callers():
    results = results_for(
        row("in-001", "answers_question", [0.9, 0.9, 0.9]),
        row("in-001", "cites_passages", [1.0, 1.0, 1.0]),
    )
    report = calibrate(results, labelled("in-001"))
    assert report.not_calibrated == ["cites_passages"]
    assert (report.acceptable, report.not_acceptable) == (1, 0)
    (only,) = report.keys
    assert (only.key, only.old, only.proposed) == ("answers_question", 0.75, 0.85)
    assert only.lowest == ("in-001", 0.9)
    assert only.margin_risks == only.failing == []


def test_the_header_counts_one_case_in_the_singular(files, capsys):
    results = results_for(row("in-001", "answers_question", [0.9, 0.9, 0.9]))
    assert main(files(results, labelled("in-001"))) == 0
    assert "1 case labelled acceptable, 0 labelled not acceptable" in capsys.readouterr().out

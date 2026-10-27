import json
import os
from datetime import date

import pytest

from evals.report import (
    GATE_PATHS,
    expired_flakes,
    find_latest_run,
    gate_change_notice,
    headline,
    main,
    summarize,
)
from evals.thresholds import FLAKY, REPORT_ONLY


def metric(name, score, success, threshold=0.8, **extra):
    return {
        "name": name,
        "score": score,
        "success": success,
        "threshold": threshold,
        "evaluationCost": 0.01,
        "flaky": False,
        **extra,
    }


RUN = {
    "testCases": [
        {
            "name": "in-001",
            "metricsData": [metric("Faithfulness", 0.9, True), metric("Hallucination", 1.0, True)],
        },
        {
            "name": "in-002",
            "metricsData": [
                metric("Faithfulness", 0.5, False, reason="Two claims were not in the context."),
                metric("Hallucination", 0.9, True),
            ],
        },
    ]
}


def test_table_has_one_row_per_metric_with_rate_and_mean():
    table = summarize(RUN)
    assert "| Hallucination | 2 | 2 | 100% | 0.95 |" in table
    assert "| Faithfulness | 2 | 1 | 50% | 0.70 |" in table


def test_failures_are_listed_with_score_threshold_and_reason():
    report = summarize(RUN)
    assert "### Failing metrics" in report
    assert (
        "- in-002 / Faithfulness: 0.50 against threshold 0.8. Two claims were not in the context."
        in report
    )


def test_cost_is_summed_across_metrics():
    assert "Judge cost reported by DeepEval: $0.04" in summarize(RUN)


def test_missing_costs_are_called_out():
    run = {
        "testCases": [
            {
                "name": "c",
                "metricsData": [{**metric("Hallucination", 1.0, True), "evaluationCost": None}],
            }
        ]
    }
    assert "some metrics did not report a cost" in summarize(run)


def test_an_errored_metric_counts_as_failed_and_shows_its_error():
    errored = {
        "name": "Hallucination",
        "score": None,
        "success": None,
        "threshold": 0.8,
        "error": "Rate limit",
    }
    report = summarize({"testCases": [{"name": "c", "metricsData": [errored]}]})
    assert "| Hallucination | 1 | 0 | 0% | n/a |" in report
    assert "- c / Hallucination: no score against threshold 0.8. Rate limit" in report


def _failure_line(reason):
    failing = metric("Hallucination", 0.4, False, reason=reason)
    run = {"testCases": [{"name": "c", "metricsData": [failing]}]}
    return next(
        line for line in summarize(run).splitlines() if line.startswith("- c / Hallucination")
    )


def test_a_long_reason_is_cut_at_a_word_boundary_with_an_ellipsis():
    reason = "The score is low because " + "incomprehensible " * 20
    line = _failure_line(reason)
    detail = line.split("threshold 0.8. ", 1)[1]
    assert len(detail) <= 160
    assert detail.endswith("…")
    assert detail[:-1].rstrip().endswith("incomprehensible")
    assert "incomprehensibl…" not in detail


def test_a_reason_that_fits_is_left_whole():
    assert _failure_line("Short reason.").endswith("threshold 0.8. Short reason.")


def test_a_multi_line_reason_stays_on_one_bullet():
    line = _failure_line("First line.\n\n  Second   line.\tThird.")
    assert line.endswith("First line. Second line. Third.")
    assert "\n" not in line


def test_a_reason_with_no_spaces_is_still_cut_to_the_limit():
    detail = _failure_line("x" * 400).split("threshold 0.8. ", 1)[1]
    assert len(detail) == 160 and detail.endswith("…")


def test_flaky_failures_are_labelled_as_warnings():
    flaky = metric("Hallucination", 0.4, False, flaky=True)
    assert "(flaky, warning only)" in summarize(
        {"testCases": [{"name": "c", "metricsData": [flaky]}]}
    )


def test_a_report_only_failure_is_labelled_report_only_not_flaky():
    # DeepEval marks a report-only metric flaky, as it does a FLAKY one, so the data alone cannot
    # tell them apart: the name does.
    bias = metric("Bias", 0.0, False, flaky=True, reason="Scored as opinion.")
    report = summarize({"testCases": [{"name": "bia-003", "metricsData": [bias]}]})
    (line,) = [ln for ln in report.splitlines() if ln.startswith("- bia-003 / Bias")]
    assert line == "- bia-003 / Bias: 0.00 against threshold 0.8 (report-only). Scored as opinion."
    assert "flaky" not in report.split("### Failing metrics")[1]


def test_report_only_rows_are_marked_in_the_table():
    run = {
        "testCases": [
            {
                "name": "bia-001",
                "metricsData": [metric("Bias", 1.0, True), metric("Faithfulness", 0.9, True)],
            },
            {
                "name": "bia-002",
                "metricsData": [
                    metric("Bias", 0.0, False, flaky=True),
                    metric("Faithfulness", 1.0, True),
                ],
            },
        ]
    }
    lines = summarize(run).splitlines()
    assert "| Bias (report-only) | 2 | 1 | 50% | 0.50 |" in lines
    assert "| Faithfulness | 2 | 2 | 100% | 0.95 |" in lines
    assert not any(line.startswith("| Bias |") for line in lines)


@pytest.mark.parametrize("name", sorted(REPORT_ONLY.values()))
def test_each_report_only_metric_is_marked_in_the_table_under_its_display_name(name):
    run = {"testCases": [{"name": "c", "metricsData": [metric(name, 1.0, True)]}]}
    assert f"| {name} (report-only) |" in summarize(run)


def test_a_run_without_metrics_says_so():
    assert summarize({"testCases": []}) == "No metric results found in this run."
    assert summarize({}) == "No metric results found in this run."


def test_find_latest_run_picks_the_newest_file(tmp_path):
    old, new = (
        tmp_path / "test_run_20260101_000000.json",
        tmp_path / "test_run_20260102_000000.json",
    )
    old.write_text("{}")
    new.write_text("{}")
    os.utime(old, (1_000, 1_000))
    os.utime(new, (2_000, 2_000))
    assert find_latest_run(tmp_path) == new


def test_main_reads_a_folder_and_prints_the_summary(tmp_path, capsys):
    (tmp_path / "test_run_20260101_000000.json").write_text(json.dumps(RUN))
    assert main([str(tmp_path)]) == 0
    assert "| Faithfulness |" in capsys.readouterr().out


def test_main_reports_a_missing_folder_without_a_traceback(tmp_path, capsys):
    assert main([str(tmp_path / "nowhere")]) == 2
    assert "error:" in capsys.readouterr().err


def test_main_reports_an_empty_folder(tmp_path, capsys):
    assert main([str(tmp_path)]) == 2
    assert "No test_run_*.json" in capsys.readouterr().err


# Flaky entries past their review date ------------------------------------------------------

SECTION = "### Flaky entries past their review date"


@pytest.fixture
def two_flakes(make_flake, monkeypatch):
    """in-018 is due on 2026-11-15; saf-003 on 2026-12-01 and lists two keys."""
    flakes = {
        "saf-003": make_flake(
            "safety",
            "emergency_first",
            reason="The judge read the symptom list as a diagnosis.",
            added=date(2026, 11, 1),
            review_by=date(2026, 12, 1),
        ),
        "in-018": make_flake(
            "contextual_recall",
            reason="Recall once scored 0.50 on a sentence from a retrieved chunk.",
            added=date(2026, 10, 16),
            review_by=date(2026, 11, 15),
        ),
    }
    monkeypatch.setattr("evals.report.FLAKY", flakes)
    return flakes


def test_an_entry_is_expired_the_day_after_its_review_date_not_on_it(two_flakes):
    assert expired_flakes(date(2026, 10, 19)) == []
    assert expired_flakes(date(2026, 11, 15)) == []
    assert expired_flakes(date(2026, 11, 16)) == [
        "in-018 contextual_recall: review was due 2026-11-15, 1 day ago (added 2026-10-16). "
        "Recall once scored 0.50 on a sentence from a retrieved chunk."
    ]


def test_expired_entries_are_listed_oldest_review_first_with_every_key(two_flakes):
    assert expired_flakes(date(2026, 12, 11)) == [
        "in-018 contextual_recall: review was due 2026-11-15, 26 days ago (added 2026-10-16). "
        "Recall once scored 0.50 on a sentence from a retrieved chunk.",
        "saf-003 emergency_first, safety: review was due 2026-12-01, 10 days ago (added "
        "2026-11-01). The judge read the symptom list as a diagnosis.",
    ]


def test_no_entries_means_nothing_expired(monkeypatch):
    monkeypatch.setattr("evals.report.FLAKY", {})
    assert expired_flakes(date(2030, 1, 1)) == []


def test_the_real_register_can_be_checked_on_any_date():
    # Whatever is in FLAKY, whatever today is: a list of strings, never an exception.
    assert all(isinstance(line, str) for line in expired_flakes(date(2040, 1, 1)))
    assert len(expired_flakes(date(2040, 1, 1))) == len(FLAKY)


def test_expired_flakes_are_listed_as_warnings_not_failures(two_flakes):
    report = summarize(RUN, today=date(2026, 12, 11))
    assert SECTION in report
    section = report.split(SECTION)[1]
    assert "Warning only" in section
    assert "- in-018 contextual_recall: review was due 2026-11-15, 26 days ago" in section
    assert "- saf-003 emergency_first, safety: review was due 2026-12-01, 10 days ago" in section
    # They sit after the failures and are not among them.
    assert report.index("### Failing metrics") < report.index(SECTION)
    failing = report.split("### Failing metrics")[1].split(SECTION)[0]
    assert "in-018 contextual_recall" not in failing


def test_the_section_is_left_out_when_no_entry_is_past_its_review_date(two_flakes):
    assert SECTION not in summarize(RUN, today=date(2026, 11, 15))


def test_the_summary_without_a_date_has_no_section(two_flakes):
    # summarize is a pure function of the run; only main reads the calendar.
    assert SECTION not in summarize(RUN)
    assert summarize(RUN) == summarize(RUN, today=date(2026, 10, 19))


def test_a_run_without_metrics_says_so_even_past_a_review_date(two_flakes):
    assert summarize({}, today=date(2030, 1, 1)) == "No metric results found in this run."


def test_main_prints_the_section_and_still_exits_zero_past_a_review_date(
    tmp_path, capsys, two_flakes
):
    (tmp_path / "test_run_20260101_000000.json").write_text(json.dumps(RUN))
    assert main([str(tmp_path)], today=date(2026, 12, 11)) == 0
    out = capsys.readouterr().out
    assert SECTION in out
    assert "| Faithfulness |" in out


def test_main_reads_todays_date_when_none_is_given(tmp_path, capsys, monkeypatch, two_flakes):
    class December11(date):
        @classmethod
        def today(cls):
            return cls(2026, 12, 11)

    monkeypatch.setattr("evals.report.date", December11)
    (tmp_path / "test_run_20260101_000000.json").write_text(json.dumps(RUN))
    assert main([str(tmp_path)]) == 0
    assert "- in-018 contextual_recall: review was due 2026-11-15, 26 days ago" in (
        capsys.readouterr().out
    )


# The verdict line and the gate-change notice --------------------------------------------------


def case(name, *metrics):
    return {"name": name, "metricsData": list(metrics)}


def run_of(*cases):
    return {"testCases": list(cases)}


RUN_ALL_PASSING = run_of(case("in-001", metric("Faithfulness", 0.9, True)))


def failing_case():
    return case("in-005", metric("Answers the question", 0.42, False, threshold=0.70))


def test_headline_says_passed_with_counts():
    cases = [case(f"in-{n:03d}", metric("Faithfulness", 0.9, True)) for n in range(1, 13)]
    cases += [
        case("in-013", metric("Hallucination", 0.4, False, flaky=True)),
        case("bia-003", metric("Bias", 0.0, False, flaky=True)),
        case("bia-006", metric("Bias", None, None, flaky=True, error="Rate limit")),
    ]
    assert headline(run_of(*cases)) == (
        "**Gate: passed.** 15 cases; 0 strict failures; 0 errors; 3 warnings (1 errored)."
    )


def test_headline_says_passed_without_warnings():
    assert headline(RUN_ALL_PASSING) == (
        "**Gate: passed.** 1 case; 0 strict failures; 0 errors; 0 warnings."
    )


def test_headline_names_strict_failures_and_errors_separately():
    assert headline(run_of(failing_case())) == (
        "**Gate: FAILED.** Strict failures: in-005 / Answers the question (0.42 < 0.70). "
        "Errors: none."
    )
    errored = case("in-009", metric("Faithfulness", None, None, error="Rate limit exceeded"))
    assert headline(run_of(failing_case(), errored)) == (
        "**Gate: FAILED.** Strict failures: in-005 / Answers the question (0.42 < 0.70). "
        "Errors: in-009 / Faithfulness: Rate limit exceeded."
    )


def test_a_strict_failure_with_no_score_says_so():
    unscored = case("in-005", metric("Safety", None, False, threshold=0.80))
    assert "in-005 / Safety (no score < 0.80)" in headline(run_of(unscored))


def test_flaky_and_report_only_failures_are_warnings_in_the_headline():
    # DeepEval marks a report-only metric flaky too; neither fails the gate.
    run = run_of(
        case("in-018", metric("Contextual Recall", 0.5, False, flaky=True)),
        case("bia-003", metric("Bias", 0.0, False, flaky=True)),
        case("in-001", metric("Faithfulness", 0.9, True)),
    )
    assert headline(run) == ("**Gate: passed.** 3 cases; 0 strict failures; 0 errors; 2 warnings.")


def test_a_report_only_metric_that_is_not_marked_flaky_is_still_a_warning():
    run = run_of(case("bia-003", metric("Bias", 0.0, False)))
    assert headline(run).startswith("**Gate: passed.**")


def test_an_error_on_a_strict_metric_is_listed_as_an_error():
    error = "RateLimitError: " + "too many requests " * 10
    run = run_of(case("in-009", metric("Faithfulness", None, None, error=error)))
    text = headline(run)
    assert text.startswith(
        "**Gate: FAILED.** Strict failures: none. Errors: in-009 / Faithfulness: "
    )
    assert "RateLimitError: too many requests" in text
    assert text.endswith("….")
    assert "no score" not in text and " < " not in text
    detail = text.split("Faithfulness: ", 1)[1]
    assert len(detail) <= 81  # the shortened error (80) and the closing full stop


def test_an_error_on_a_strict_metric_with_success_false_is_one_error_not_also_a_failure():
    errored = metric("Faithfulness", None, False, error="Timeout")
    text = headline(run_of(case("in-009", errored)))
    assert (
        text == "**Gate: FAILED.** Strict failures: none. Errors: in-009 / Faithfulness: Timeout."
    )


def test_an_error_on_a_report_only_metric_is_a_warning_marked_errored():
    run = run_of(
        case("bia-003", metric("Bias", None, None, flaky=True, error="Rate limit")),
        case("in-001", metric("Faithfulness", 0.9, True)),
    )
    assert headline(run) == (
        "**Gate: passed.** 2 cases; 0 strict failures; 0 errors; 1 warning (1 errored)."
    )


def test_an_error_on_a_flaky_metric_is_a_warning_marked_errored():
    run = run_of(case("in-018", metric("Contextual Recall", None, None, flaky=True, error="x")))
    assert headline(run).endswith("0 errors; 1 warning (1 errored).")


def test_a_long_list_of_failures_is_cut_after_ten():
    cases = [case(f"in-{n:03d}", metric("Safety", 0.1, False)) for n in range(1, 14)]
    text = headline(run_of(*cases))
    assert "in-010 / Safety" in text and "in-011 / Safety" not in text
    assert "and 3 more" in text


def test_a_failed_run_with_no_failing_metric_is_not_reported_as_passed():
    # The app call can raise or the harness can error outside the scored metrics: the JSON then
    # shows no failure although the check is red.
    text = headline(RUN_ALL_PASSING, outcome="failure")
    assert text == (
        '**Gate: FAILED.** The test run failed outside the scored metrics; see the "Run DeepEval" '
        "log. 1 case; 0 strict failures; 0 errors; 0 warnings."
    )


def test_a_run_that_failed_with_no_cases_at_all_is_failed_too():
    assert headline({}, outcome="failure").startswith("**Gate: FAILED.** The test run failed")


def test_a_failed_run_with_a_strict_failure_names_the_failure():
    text = headline(run_of(failing_case()), outcome="failure")
    assert text == headline(run_of(failing_case()))
    assert "outside the scored metrics" not in text


def test_a_run_that_reported_success_cannot_hide_a_strict_failure():
    text = headline(run_of(failing_case()), outcome="success")
    assert text.startswith("**Gate: FAILED.** Strict failures: in-005")
    assert text.endswith("(the run reported success; check the log)")
    assert "passed" not in text
    errored = case("in-009", metric("Faithfulness", None, None, error="Timeout"))
    assert "(the run reported success; check the log)" in headline(
        run_of(errored), outcome="success"
    )


def test_a_successful_run_with_a_clean_json_is_passed_and_so_is_one_without_an_outcome():
    assert headline(RUN_ALL_PASSING, outcome="success") == headline(RUN_ALL_PASSING)
    assert headline(RUN_ALL_PASSING, outcome="success").startswith("**Gate: passed.**")


@pytest.mark.parametrize("outcome", ["cancelled", "skipped"])
def test_a_run_that_did_not_finish_is_not_reported_as_passed(outcome):
    text = headline(RUN_ALL_PASSING, outcome=outcome)
    assert text.startswith("**Gate: not decided.**")
    assert outcome in text and "passed" not in text
    assert headline(run_of(failing_case()), outcome=outcome).startswith("**Gate: FAILED.**")


def test_a_successful_run_with_no_metric_results_is_not_reported_as_passed():
    assert headline({}, outcome="success").startswith("**Gate: no results.**")
    assert headline({"testCases": []}).startswith("**Gate: no results.**")


def test_a_full_run_headline_does_not_say_gate():
    clean = headline(RUN_ALL_PASSING, gate=False)
    assert clean == (
        "**Full suite (trend only, does not gate): 0 strict failures; 0 errors; 0 warnings; "
        "1 case.**"
    )
    errored = case("in-009", metric("Faithfulness", None, None, error="Timeout"))
    run = run_of(failing_case(), errored, case("bia-003", metric("Bias", 0.0, False, flaky=True)))
    text = headline(run, outcome="failure", gate=False)
    assert text.startswith(
        "**Full suite (trend only, does not gate): 1 strict failure; 1 error; 1 warning; 3 cases.**"
    )
    assert "Strict failures: in-005 / Answers the question (0.42 < 0.70)." in text
    assert "Errors: in-009 / Faithfulness: Timeout." in text
    for outcome in (None, "success", "failure", "cancelled", "skipped"):
        for the_run in (run, RUN_ALL_PASSING, {}):
            assert "Gate" not in headline(the_run, outcome=outcome, gate=False)


def test_a_full_run_that_crashed_says_so():
    text = headline(RUN_ALL_PASSING, outcome="failure", gate=False)
    assert 'The "Run DeepEval" step failed outside the scored metrics' in text


def test_the_headline_does_not_change_the_summary():
    assert "Gate" not in summarize(RUN)
    assert "Gate" not in summarize(RUN, today=date(2026, 12, 11))


# The notice -----------------------------------------------------------------------------------


def test_a_pr_that_edits_the_thresholds_is_flagged():
    assert gate_change_notice(["README.md", "evals/thresholds.py"]) == (
        "**This pull request changes the gate itself:** evals/thresholds.py. A reviewer should "
        "check that the change does not weaken the gate."
    )


def test_a_docs_only_pr_has_no_notice():
    assert gate_change_notice(["README.md", "docs/findings.md", "docs/evals/thresholds.py"]) == ""
    assert gate_change_notice([]) == ""


def test_the_notice_lists_every_matching_file_once_and_only_those():
    notice = gate_change_notice(
        [
            "evals/dataset/smoke.json",
            "docs/findings.md",
            ".github/workflows/eval.yml",
            "evals/dataset/smoke.json",
            "pyproject.toml",
        ]
    )
    assert "evals/dataset/smoke.json, .github/workflows/eval.yml, pyproject.toml." in notice
    assert "findings" not in notice


def test_the_notice_is_cut_after_ten_files():
    files = [f"evals/dataset/case-{n:02d}.json" for n in range(1, 14)]
    notice = gate_change_notice(files)
    assert "case-10.json, and 3 more." in notice
    assert "case-11.json" not in notice


@pytest.mark.parametrize(
    "path, matches",
    [
        ("evals/dataset/smoke.json", True),  # a directory entry matches by prefix
        (".github/workflows/eval.yml", True),
        (".github/scripts/check-eval-keys.sh", True),
        ("evals/thresholds.py", True),  # a file entry matches exactly
        ("uv.lock", True),
        ("evals/thresholds.py.bak", False),
        ("docs/evals/thresholds.py", False),
        ("evals/dataset", False),  # the directory entry needs its slash
        ("evals/datasets/other.json", False),
        ("src/cardio_rag/config.py", True),
        ("src/cardio_rag/config.pyc", False),
        ("pyproject.toml.orig", False),
        ("evals/report.py", False),
        ("README.md", False),
    ],
)
def test_gate_paths_match_by_prefix_only_for_directories(path, matches):
    assert bool(gate_change_notice([path])) is matches


def test_gate_paths_name_the_files_that_define_what_the_gate_measures():
    assert {
        "evals/thresholds.py",
        "evals/metrics.py",
        "evals/deterministic.py",
        "evals/repeat.py",
        "evals/dataset/",
        ".github/",
        "evals/conftest.py",
        "evals/snapshot.py",
        "evals/harness.py",
        "evals/test_rag_quality.py",
        "src/cardio_rag/config.py",
        "pyproject.toml",
        "uv.lock",
    } == set(GATE_PATHS)


# main -----------------------------------------------------------------------------------------


def write_run(folder, run=RUN):
    (folder / "test_run_20260101_000000.json").write_text(json.dumps(run))


def test_main_prints_the_headline_first_then_the_notice_then_the_summary(tmp_path, capsys):
    write_run(tmp_path)
    changed = tmp_path / "changed-files.txt"
    changed.write_text("README.md\nevals/thresholds.py\n")
    assert main([str(tmp_path), "--changed-files", str(changed)]) == 0
    out = capsys.readouterr().out
    verdict, notice, table = out.split("\n\n", 2)
    assert verdict.startswith(
        "**Gate: FAILED.** Strict failures: in-002 / Faithfulness (0.50 < 0.80)"
    )
    assert notice.startswith("**This pull request changes the gate itself:** evals/thresholds.py.")
    assert table.startswith("| Metric |")


def test_main_without_a_notice_prints_the_headline_then_the_summary(tmp_path, capsys):
    write_run(tmp_path, RUN_ALL_PASSING)
    changed = tmp_path / "changed-files.txt"
    changed.write_text("README.md\n")
    assert main([str(tmp_path), "--changed-files", str(changed)]) == 0
    verdict, table = capsys.readouterr().out.split("\n\n", 1)
    assert verdict.startswith("**Gate: passed.**")
    assert table.startswith("| Metric |")


def test_an_empty_changed_files_list_prints_no_notice(tmp_path, capsys):
    write_run(tmp_path)
    changed = tmp_path / "changed-files.txt"
    changed.write_text("")
    assert main([str(tmp_path), "--changed-files", str(changed)]) == 0
    out = capsys.readouterr().out
    assert "changes the gate itself" not in out and "Could not list" not in out


def test_missing_changed_files_prints_a_check_by_hand_line(tmp_path, capsys):
    write_run(tmp_path)
    assert main([str(tmp_path), "--changed-files", str(tmp_path / "nowhere.txt")]) == 0
    out = capsys.readouterr().out
    assert (
        "**Could not list the changed files; check by hand whether this pull request changes "
        "the gate.**"
    ) in out
    assert "changes the gate itself" not in out


def test_main_without_changed_files_prints_neither_notice(tmp_path, capsys):
    write_run(tmp_path)
    assert main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "changes the gate itself" not in out and "Could not list" not in out


def test_main_passes_the_outcome_and_the_full_run_flag_to_the_headline(tmp_path, capsys):
    write_run(tmp_path, RUN_ALL_PASSING)
    assert main([str(tmp_path), "--outcome", "failure"]) == 0
    assert "The test run failed outside the scored metrics" in capsys.readouterr().out
    assert main([str(tmp_path), "--outcome", "success", "--not-a-gate"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("**Full suite (trend only, does not gate):")
    assert "Gate" not in out.split("\n\n", 1)[0]


def test_main_refuses_an_outcome_it_does_not_know(tmp_path):
    write_run(tmp_path)
    with pytest.raises(SystemExit):
        main([str(tmp_path), "--outcome", "maybe"])


def test_main_with_no_readable_run_still_leaves_a_headline_for_the_job_summary(tmp_path, capsys):
    changed = tmp_path / "changed-files.txt"
    changed.write_text(".github/workflows/eval.yml\n")
    assert main([str(tmp_path), "--changed-files", str(changed)]) == 2
    captured = capsys.readouterr()
    assert captured.out.startswith("**Gate: no results.**")
    assert "changes the gate itself" in captured.out
    assert "No test_run_" in captured.err


def test_main_with_a_broken_json_exits_2_and_says_no_results(tmp_path, capsys):
    (tmp_path / "test_run_20260101_000000.json").write_text("{not json")
    assert main([str(tmp_path), "--outcome", "failure"]) == 2
    captured = capsys.readouterr()
    assert captured.out.startswith("**Gate: no results.**")
    assert "failed" in captured.out.split("\n")[0]
    assert "error:" in captured.err


def test_main_with_a_json_that_is_not_a_run_exits_2(tmp_path, capsys):
    (tmp_path / "test_run_20260101_000000.json").write_text("[]")
    assert main([str(tmp_path)]) == 2
    assert capsys.readouterr().out.startswith("**Gate: no results.**")


def test_main_with_no_readable_run_on_a_full_run_does_not_say_gate(tmp_path, capsys):
    assert main([str(tmp_path), "--not-a-gate"]) == 2
    out = capsys.readouterr().out
    assert out.startswith("**Full suite (trend only, does not gate): no results.**")
    assert "Gate" not in out

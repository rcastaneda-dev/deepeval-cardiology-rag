import json
import os
from datetime import date

import pytest

from evals.report import expired_flakes, find_latest_run, main, summarize
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

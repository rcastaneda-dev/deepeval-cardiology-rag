"""Summarise a DeepEval test-run JSON file as Markdown.

Usage: python -m evals.report <results-folder-or-file> [--outcome OUTCOME] [--not-a-gate]
                              [--changed-files PATH]
Used for the CI job summary and for the README results table. The job summary opens with one line,
the gate verdict (headline), which tells strict failures, errors and warnings apart and agrees
with the colour of the check through the outcome of the "Run DeepEval" step; then a notice when
the pull request changes a file that defines the gate (gate_change_notice); then the table
(summarize). It also lists the FLAKY entries past their review date, as a warning: a calendar
date never fails a run.

Exit status: 0 when the summary is written; 1 when the table cannot be built; 2 when the run
cannot be read at all; 3 for a full run (--not-a-gate) that scored no case (NOTHING_SCORED). A
pull request's check ignores it; a full run's job is red on any of 1 to 3, never on a score.
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from evals.thresholds import FLAKY, REPORT_ONLY


def find_latest_run(folder: Path) -> Path:
    runs = sorted(folder.glob("test_run_*.json"), key=lambda path: path.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"No test_run_*.json files in {folder}")
    return runs[-1]


def _is_report_only(metric_name: str) -> bool:
    """Whether DeepEval's `metric_name` is one of the REPORT_ONLY metrics: shown, never gating."""
    return metric_name in REPORT_ONLY.values()


def expired_flakes(today: date) -> list[str]:
    """One line for each FLAKY entry whose review date is before `today`, the oldest review
    first: the case, its keys, how long ago the review was due, when the entry was added and its
    reason. The review date itself is not yet late. Never raises and never fails anything: the
    entry keeps making its checks warn until a person renews or deletes it."""
    late = sorted(
        (flake.review_by, case_id) for case_id, flake in FLAKY.items() if flake.review_by < today
    )
    lines = []
    for review_by, case_id in late:
        flake = FLAKY[case_id]
        days = (today - review_by).days
        lines.append(
            f"{case_id} {', '.join(sorted(flake.keys))}: review was due {review_by.isoformat()}, "
            f"{days} day{'' if days == 1 else 's'} ago (added {flake.added.isoformat()}). "
            f"{_shorten(flake.reason)}"
        )
    return lines


def summarize(run: dict, today: date | None = None) -> str:
    """The Markdown summary of one test run. With a `today`, it ends with the FLAKY entries past
    their review date (expired_flakes) as a warning section; without one it does not look at the
    calendar, so the same run always gives the same text."""
    stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {"cases": 0, "passed": 0, "score_total": 0.0, "scored": 0, "cost": 0.0}
    )
    failures: list[str] = []
    costs_missing = False

    for case in run.get("testCases") or []:
        for metric in case.get("metricsData") or []:
            row = stats[metric["name"]]
            row["cases"] += 1
            passed = bool(metric.get("success"))
            row["passed"] += passed
            score = metric.get("score")
            if score is not None:
                row["score_total"] += score
                row["scored"] += 1
            cost = metric.get("evaluationCost")
            if cost is None:
                costs_missing = True
            else:
                row["cost"] += cost
            if not passed:
                failures.append(_describe_failure(case.get("name", "?"), metric))

    if not stats:
        return "No metric results found in this run."

    lines = [
        "| Metric | Cases | Passed | Pass rate | Mean score |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in sorted(stats):
        row = stats[name]
        mean = f"{row['score_total'] / row['scored']:.2f}" if row["scored"] else "n/a"
        rate = f"{row['passed'] / row['cases']:.0%}"
        shown = f"{name} (report-only)" if _is_report_only(name) else name
        lines.append(f"| {shown} | {row['cases']:.0f} | {row['passed']:.0f} | {rate} | {mean} |")

    total_cost = sum(row["cost"] for row in stats.values())
    qualifier = " (some metrics did not report a cost)" if costs_missing else ""
    lines += ["", f"Judge cost reported by DeepEval: ${total_cost:.2f}{qualifier}"]
    if failures:
        lines += ["", "### Failing metrics", *failures]
    expired = expired_flakes(today) if today is not None else []
    if expired:
        lines += [
            "",
            "### Flaky entries past their review date",
            "",
            "Warning only: these entries still make their checks warn instead of fail. Check "
            "each against new judgings, then renew or delete it (`FLAKY` in evals/thresholds.py).",
            "",
            *(f"- {line}" for line in expired),
        ]
    return "\n".join(lines)


def _shorten(text: str, limit: int = 160) -> str:
    """One line of at most `limit` characters, cut at a word boundary with an ellipsis."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    if text[limit - 1] != " " and " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:-") + "…"


def _describe_failure(case_name: str, metric: dict) -> str:
    score = metric.get("score")
    shown = "no score" if score is None else f"{score:.2f}"
    # DeepEval marks a report-only metric flaky too, so the name is checked first.
    if _is_report_only(metric["name"]):
        label = " (report-only)"
    else:
        label = " (flaky, warning only)" if metric.get("flaky") else ""
    detail = metric.get("error") or metric.get("reason") or ""
    return (
        f"- {case_name} / {metric['name']}: {shown} against threshold "
        f"{metric.get('threshold')}{label}. {_shorten(detail)}".rstrip()
    )


# Files that decide what the pull-request check measures or whether it can be trusted. A path
# ending in "/" matches everything under it; the others match that file only. The first six are
# the gate's thresholds, metrics, deterministic checks, repeat count, dataset and workflow. The
# next four run the check: the pytest setup, the answer replay (the gate must never use it), the
# harness that packages each answer and the test file itself. src/cardio_rag/config.py picks the
# judge models (DEFAULT_JUDGE_MODELS), so a change there changes who scores the gate.
# pyproject.toml and uv.lock hold the DeepEval pin (==4.2.9): a new version can score the same
# answers differently. The last three: evals/params.py applies the smoke mark, so it decides
# which cases the pull request runs; evals/cases.py loads and validates the goldens; and
# evals/report.py writes the verdict and this notice. data/corpus/ and src/cardio_rag/generate.py
# (the system prompt) are not listed on purpose: they are the product the gate tests, not the
# gate, and flagging them would mark every content or prompt pull request.
GATE_PATHS = (
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
    "evals/params.py",
    "evals/cases.py",
    "evals/report.py",
)

OUTCOMES = ("success", "failure", "cancelled", "skipped")  # a GitHub Actions step outcome
# What `--outcome` becomes when its value is none of OUTCOMES (an empty one when the workflow's
# expression has nothing to give): the gate is then not decided, never passed.
UNKNOWN_OUTCOME = "unknown"
UNFINISHED_OUTCOMES = (*(outcome for outcome in OUTCOMES if outcome != "success"), UNKNOWN_OUTCOME)
LISTED_AT_MOST = 10
ERROR_TEXT_LIMIT = 80
# A full run never fails on a score, but one that scored no case has no trend to report: its
# summary says so and main exits non-zero, which turns the full-run job red (eval.yml). On
# 2026-10-26 a rejected key left all 65 cases unscored and the job green for five days.
NOTHING_SCORED = "No case was scored without an error, so this job fails."
NO_CHANGED_FILES_LIST = (
    "**Could not list the changed files; check by hand whether this pull request changes the "
    "gate.**"
)


def _matches_gate(path: str) -> bool:
    return any(
        path.startswith(entry) if entry.endswith("/") else path == entry for entry in GATE_PATHS
    )


def _listed(items: list[str], separator: str) -> str:
    """The first LISTED_AT_MOST items, then "and N more" when there are others."""
    shown = separator.join(items[:LISTED_AT_MOST])
    if len(items) > LISTED_AT_MOST:
        shown += f"{separator}and {len(items) - LISTED_AT_MOST} more"
    return shown


def gate_change_notice(changed: list[str]) -> str:
    """A bold notice naming the changed files that are in GATE_PATHS, or "" when none is. A pull
    request that edits a threshold or the dataset can turn the check green without improving the
    app, and the check cannot say so itself; a reviewer has to look."""
    touched = list(dict.fromkeys(path for path in changed if _matches_gate(path)))
    if not touched:
        return ""
    return (
        f"**This pull request changes the gate itself:** {_listed(touched, ', ')}. "
        "A reviewer should check that the change does not weaken the gate."
    )


@dataclass
class _Tally:
    """What the metrics of a run add up to. A strict metric is neither report-only nor flaky; a
    failure or an error on any other metric is only a warning."""

    cases: int = 0
    scored_cases: int = 0  # cases with metric results and no errored metric, lenient or strict
    metrics: int = 0
    strict_failures: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: int = 0
    errored_warnings: int = 0

    @property
    def fails_gate(self) -> bool:
        return bool(self.strict_failures or self.errors)


def _tally(run: dict) -> _Tally:
    tally = _Tally()
    for case in run.get("testCases") or []:
        tally.cases += 1
        metrics = case.get("metricsData") or []
        tally.scored_cases += bool(metrics) and not any(metric.get("error") for metric in metrics)
        for metric in metrics:
            tally.metrics += 1
            name = metric.get("name", "?")
            where = f"{case.get('name', '?')} / {name}"
            # A report-only metric only warns, flaky or not: DeepEval marks it flaky too.
            lenient = _is_report_only(name) or bool(metric.get("flaky"))
            error = metric.get("error")
            if not error and metric.get("success"):
                continue
            if lenient:
                tally.warnings += 1
                tally.errored_warnings += bool(error)
            elif error:
                # An error is not a low score: no number is shown for it.
                tally.errors.append(f"{where}: {_code(_shorten(str(error), ERROR_TEXT_LIMIT))}")
            else:
                score = metric.get("score")
                shown = "no score" if score is None else f"{score:.2f}"
                threshold = metric.get("threshold")
                needed = "?" if threshold is None else f"{threshold:.2f}"
                tally.strict_failures.append(f"{where} ({shown} < {needed})")
    return tally


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _warnings_text(tally: _Tally) -> str:
    text = _plural(tally.warnings, "warning")
    return f"{text} ({tally.errored_warnings} errored)" if tally.errored_warnings else text


def _counts(tally: _Tally, cases_first: bool = True) -> str:
    parts = [
        _plural(len(tally.strict_failures), "strict failure"),
        _plural(len(tally.errors), "error"),
        _warnings_text(tally),
    ]
    cases = _plural(tally.cases, "case")
    return "; ".join([cases, *parts] if cases_first else [*parts, cases]) + "."


_NOT_FINISHED = {
    "failure": 'The "Run DeepEval" step failed outside the scored metrics; see its log.',
    "cancelled": 'The "Run DeepEval" step was cancelled; see the job log.',
    # The step is skipped when an earlier one (the install, say) failed or was skipped, so it has
    # no log of its own.
    "skipped": "DeepEval did not run: an earlier step failed or was skipped; see the job log.",
    UNKNOWN_OUTCOME: 'The "Run DeepEval" step gave no outcome this summary knows; see the job log.',
}
assert tuple(_NOT_FINISHED) == UNFINISHED_OUTCOMES


def _code(text: str) -> str:
    """`text` as one Markdown code span on one line: whitespace is collapsed and a backtick in it
    becomes an apostrophe, so nothing in an error message can end the span or format the line."""
    return f"`{' '.join(text.split()).replace(chr(96), chr(39)) or 'no text'}`"


def headline(run: dict, outcome: str | None = None, gate: bool = True) -> str:
    """One bold line that says whether the pull-request check passed, for the top of the job
    summary. `outcome` is the outcome of the "Run DeepEval" step (success, failure, cancelled or
    skipped), taken before continue-on-error applies: a test can fail outside the scored metrics
    (the app call raises, the harness errors, a case is missing from the JSON), and the JSON alone
    would then say "passed" over a red check, so a failed outcome is reported as FAILED and a
    strict failure is never reported as passed, whatever the outcome says. Without an outcome the
    JSON decides alone. With gate=False (a full run, which reports a trend and does not gate) the
    line says so and never says "Gate", and it ends with NOTHING_SCORED when no case has metric
    results without an error: a case with any errored metric is not scored."""
    tally = _tally(run)
    strict = _listed(tally.strict_failures, "; ") or "none"
    errors = _listed(tally.errors, "; ") or "none"
    if not gate:
        text = f"**Full suite (trend only, does not gate): {_counts(tally, cases_first=False)}**"
        if tally.strict_failures:
            text += f" Strict failures: {strict}."
        if tally.errors:
            text += f" Errors: {errors}."
        if outcome in UNFINISHED_OUTCOMES and not tally.fails_gate:
            text += f" {_NOT_FINISHED[outcome]}"
        if not tally.scored_cases:
            text += f" {NOTHING_SCORED}"
        return text
    if tally.fails_gate:
        text = f"**Gate: FAILED.** Strict failures: {strict}. Errors: {errors}."
        if tally.warnings:
            text += f" Also {_warnings_text(tally)}."
        if outcome == "success":
            text += " (the run reported success; check the log)"
        return text
    if outcome == "failure":
        return (
            "**Gate: FAILED.** The test run failed outside the scored metrics; see the "
            f'"Run DeepEval" log. {_counts(tally)}'
        )
    # The failure was handled above: cancelled, skipped or an outcome this summary does not know.
    # An unknown outcome may come from a complete run (a renamed step id), so it says "may be".
    if outcome in UNFINISHED_OUTCOMES:
        partial = (
            "These results may be partial."
            if outcome == UNKNOWN_OUTCOME
            else "These results are partial or absent."
        )
        return f"**Gate: not decided.** {_NOT_FINISHED[outcome]} {partial} {_counts(tally)}"
    if not tally.metrics:
        return "**Gate: no results.** The run has no metric results, so nothing was scored."
    return f"**Gate: passed.** {_counts(tally)}"


def no_results_headline(reason: str, outcome: str | None = None, gate: bool = True) -> str:
    """The headline when the run JSON cannot be read, so the job summary is not blank."""
    lead = (
        "**Gate: no results.**"
        if gate
        else "**Full suite (trend only, does not gate): no results.**"
    )
    text = f"{lead} The run results could not be read ({_code(reason)})"
    if outcome in UNFINISHED_OUTCOMES:
        text = f"{text}. {_NOT_FINISHED[outcome]}"
    else:
        text = f'{text}; see the "Run DeepEval" log.'
    return text if gate else f"{text} {NOTHING_SCORED}"


def _changed_files_notice(path: Path | None) -> str:
    """The gate-change notice for the list of changed files at `path`, one path per line. No
    path means the check was not asked for; an unreadable one means nobody listed the files, and
    the line says so instead of staying silent; an empty file lists no change."""
    if path is None:
        return ""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return NO_CHANGED_FILES_LIST
    return gate_change_notice([line.strip() for line in text.splitlines() if line.strip()])


def main(argv: list[str] | None = None, today: date | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="a test_run_*.json file or a folder holding them")
    parser.add_argument(
        "--outcome",
        help='the outcome of the "Run DeepEval" step, so the verdict agrees with the check: '
        f"{', '.join(OUTCOMES)}; any other value, an empty one included, is not decided",
    )
    parser.add_argument(
        "--not-a-gate",
        action="store_true",
        help="this run reports a trend and does not gate (the full suite)",
    )
    parser.add_argument(
        "--changed-files",
        type=Path,
        metavar="PATH",
        help="file listing the paths a pull request changes, one per line",
    )
    args = parser.parse_args(argv)
    if args.outcome is not None and args.outcome not in OUTCOMES:
        args.outcome = UNKNOWN_OUTCOME
    gate = not args.not_a_gate
    notice = _changed_files_notice(args.changed_files)
    try:
        path = find_latest_run(args.path) if args.path.is_dir() else args.path
        run = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(run, dict):
            raise ValueError(f"{path} is not a DeepEval test run")
    except (OSError, ValueError) as error:
        print(
            "\n\n".join(filter(None, [no_results_headline(str(error), args.outcome, gate), notice]))
        )
        print(f"error: {error}", file=sys.stderr)
        return 2
    # The verdict goes out before the table is built: a metric the table cannot read must not lose
    # it. The table failing is exit 1 (2 is a run that could not be read at all, 3 a full run that
    # scored no case).
    print("\n\n".join(filter(None, [headline(run, args.outcome, gate), notice])), flush=True)
    print()
    try:
        print(summarize(run, today=today or date.today()))
    except Exception as error:
        print(
            "The results table could not be built "
            f"({_code(f'{type(error).__name__}: {error}')}); see the run JSON."
        )
        print(f"error: {error}", file=sys.stderr)
        return 1
    if not gate and not _tally(run).scored_cases:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Summarise a DeepEval test-run JSON file as Markdown.

Usage: python -m evals.report <results-folder-or-file>
Used for the CI job summary and for the README results table. The job summary also lists the
FLAKY entries past their review date, as a warning: a calendar date never fails a run.
"""

import argparse
import json
import sys
from collections import defaultdict
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


def main(argv: list[str] | None = None, today: date | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="a test_run_*.json file or a folder holding them")
    args = parser.parse_args(argv)
    try:
        path = find_latest_run(args.path) if args.path.is_dir() else args.path
        run = json.loads(path.read_text(encoding="utf-8"))
        print(summarize(run, today=today or date.today()))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

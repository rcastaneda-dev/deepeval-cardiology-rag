"""Summarise a DeepEval test-run JSON file as Markdown.

Usage: python -m evals.report <results-folder-or-file>
Used for the CI job summary and for the README results table.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def find_latest_run(folder: Path) -> Path:
    runs = sorted(folder.glob("test_run_*.json"), key=lambda path: path.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"No test_run_*.json files in {folder}")
    return runs[-1]


def summarize(run: dict) -> str:
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
        lines.append(f"| {name} | {row['cases']:.0f} | {row['passed']:.0f} | {rate} | {mean} |")

    total_cost = sum(row["cost"] for row in stats.values())
    qualifier = " (some metrics did not report a cost)" if costs_missing else ""
    lines += ["", f"Judge cost reported by DeepEval: ${total_cost:.2f}{qualifier}"]
    if failures:
        lines += ["", "### Failing metrics", *failures]
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
    flaky = " (flaky, warning only)" if metric.get("flaky") else ""
    detail = metric.get("error") or metric.get("reason") or ""
    return (
        f"- {case_name} / {metric['name']}: {shown} against threshold "
        f"{metric.get('threshold')}{flaky}. {_shorten(detail)}".rstrip()
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="a test_run_*.json file or a folder holding them")
    args = parser.parse_args(argv)
    try:
        path = find_latest_run(args.path) if args.path.is_dir() else args.path
        print(summarize(json.loads(path.read_text(encoding="utf-8"))))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

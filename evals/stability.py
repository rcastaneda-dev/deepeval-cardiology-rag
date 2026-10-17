"""How often the judge changes its verdict on the same recorded answer. Pure Python.

scripts/judge_stability.py judges one recorded answer several times and writes the results;
these functions turn them into the numbers and the Markdown table that tell whether a metric is
steady enough to gate a pull request.
"""

import math

from evals.cases import Golden
from evals.thresholds import METRIC_KEYS

_KNOWN_BAD_FIELDS = ("case_id", "kind", "answer", "must_fail")


def flip_rate(verdicts: list[bool]) -> float:
    """The share of verdicts that differ from the majority (an even split counts as half)."""
    if not verdicts:
        return 0.0
    return min(sum(verdicts), len(verdicts) - sum(verdicts)) / len(verdicts)


def passes_majority(passed: list[bool]) -> bool:
    """More than half of the judgings passed."""
    return 2 * sum(passed) > len(passed)


def median_of_three_pass(p: float) -> float:
    """The chance that the median of three independent judgings passes, when one does with
    chance `p`: at least two of the three pass."""
    return p**3 + 3 * p**2 * (1 - p)


def gate_false_alarm(pass_probabilities: list[float]) -> float:
    """The chance that at least one strict check fails, when each passes with its own chance."""
    return 1 - math.prod(pass_probabilities)


def known_bad_problems(entries: list, goldens: list[Golden]) -> list[str]:
    """Why known-bad entries cannot be judged, as readable strings (empty when they can).

    An entry is {"case_id", "kind", "answer", "must_fail": [metric keys]}.
    """
    categories = {golden.id: golden.category for golden in goldens}
    problems = []
    for number, entry in enumerate(entries, 1):
        if not isinstance(entry, dict):
            problems.append(f"entry {number}: not an object")
            continue
        case_id = entry.get("case_id", f"entry {number}")
        missing = [field for field in _KNOWN_BAD_FIELDS if field not in entry]
        if missing:
            problems.append(f"{case_id}: missing {', '.join(missing)}")
            continue
        label = f"{case_id} ({entry['kind']})"
        if not isinstance(entry["answer"], str) or not entry["answer"].strip():
            problems.append(f"{label}: answer is empty")
        keys = entry["must_fail"]
        if not isinstance(keys, list) or not keys:
            problems.append(f"{label}: must_fail needs at least one metric key")
        if case_id not in categories:
            problems.append(f"{case_id}: not a golden case")
        elif isinstance(keys, list):
            category = categories[case_id]
            problems.extend(
                f"{case_id}: {key!r} is not a metric of {category} cases"
                for key in keys
                if key not in METRIC_KEYS[category]
            )
    return problems


def summarize(results: dict) -> str:
    """Markdown for a results file: a row per case and metric, the known-bad answers if any were
    judged, and the estimated pull-request gate false-alarm rate."""
    rows = results["rows"]
    lines = [
        f"Judge {results['judge']}, {results['times']} judgings of each check on "
        f"{results['snapshot']}. Judge cost reported by DeepEval: ${results['cost']:.4f}",
        "",
        "| Case | Metric | Scores | Passed | Flip rate | Strict |",
        "|---|---|---|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['key']} | {_scores(row['scores'])} "
            f"| {sum(row['passed'])}/{len(row['passed'])} | {flip_rate(row['passed']):.0%} "
            f"| {'yes' if row['strict'] else 'no'} |"
        )
    if results["known_bad"]:
        lines += [
            "",
            "### Known-bad answers",
            "",
            "A miss is a known-bad answer that passes most of its judgings.",
            "",
            "| Case | Kind | Metric | Scores | Passed | Verdict |",
            "|---|---|---|---|---:|:---:|",
        ]
        for entry in results["known_bad"]:
            verdict = "MISS" if passes_majority(entry["passed"]) else "caught"
            lines.append(
                f"| {entry['case_id']} | {entry['kind']} | {entry['key']} "
                f"| {_scores(entry['scores'])} "
                f"| {sum(entry['passed'])}/{len(entry['passed'])} | {verdict} |"
            )
    lines += ["", _gate_estimate(rows)]
    return "\n".join(lines)


def _scores(scores: list[float | None]) -> str:
    return " ".join("err" if score is None else f"{score:.2f}" for score in scores)


def _gate_estimate(rows: list[dict]) -> str:
    """What the smoke gate would do on a spotless answer: every strict check on a smoke case
    must pass, so each check's pass rate over the judgings multiplies into the chance of a
    green run. Errored judgings count as failures, as they do in the gate."""
    checks = [row for row in rows if row["smoke"] and row["strict"] and row["passed"]]
    if not checks:
        return "No strict check on a smoke case was judged, so there is no gate estimate."
    rates = [sum(row["passed"]) / len(row["passed"]) for row in checks]
    cases = len({row["case_id"] for row in checks})
    single = gate_false_alarm(rates)
    median = gate_false_alarm([median_of_three_pass(rate) for rate in rates])
    return (
        f"Estimated false alarm of the pull-request gate ({len(checks)} strict "
        f"check{'' if len(checks) == 1 else 's'} on {cases} smoke case"
        f"{'' if cases == 1 else 's'}): single judging {single:.1%}, median of 3 {median:.1%}."
    )

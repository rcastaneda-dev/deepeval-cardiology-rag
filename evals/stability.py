"""How often the judge changes its verdict on the same recorded answer. Pure Python.

scripts/judge_stability.py judges one recorded answer several times and writes the results;
these functions turn them into the numbers and the Markdown table that tell whether a metric is
steady enough to gate a pull request.
"""

import math

from evals.cases import Golden
from evals.thresholds import METRIC_KEYS, keys_for

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


def wilson_interval(failures: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """The Wilson score interval for the share `failures` / `total` (95% at z = 1.96).

    Unlike the plain normal interval it stays inside 0..1 and is not a single point when no
    judging failed: 0 failures in 10 still allows a failure rate up to 27.7%.
    """
    share = failures / total
    spread = 1 + z**2 / total
    centre = (share + z**2 / (2 * total)) / spread
    half = z * math.sqrt(share * (1 - share) / total + z**2 / (4 * total**2)) / spread
    return max(0.0, centre - half), min(1.0, centre + half)


def known_bad_problems(entries: list, goldens: list[Golden]) -> list[str]:
    """Why known-bad entries cannot be judged, as readable strings (empty when they can).

    An entry is {"case_id", "kind", "answer", "must_fail": [metric keys]}. An empty must_fail
    keeps an answer for a check that does not exist yet: it is valid and is not judged. A key must
    be one the case really runs (`keys_for`), so emergency_first is valid only on a case marked
    `emergency`.
    """
    by_id = {golden.id: golden for golden in goldens}
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
        if not isinstance(keys, list):
            problems.append(f"{label}: must_fail must be a list of metric keys")
        if case_id not in by_id:
            problems.append(f"{case_id}: not a golden case")
        elif isinstance(keys, list):
            category = by_id[case_id].category
            runs = keys_for(by_id[case_id])
            for key in keys:
                if key in runs:
                    continue
                if key in METRIC_KEYS[category]:  # only emergency_first is dropped by keys_for
                    problems.append(
                        f"{case_id}: {key!r} does not run on this case (it is not marked emergency)"
                    )
                else:
                    problems.append(f"{case_id}: {key!r} is not a metric of {category} cases")
    return problems


def summarize(results: dict) -> str:
    """Markdown for a results file: a table per metric, a row per case and metric, the known-bad
    answers if any were judged or left without a check, and the estimated pull-request gate
    false-alarm rate with what it is based on."""
    rows = results["rows"]
    lines = [
        f"Judge {results['judge']}, {results['times']} judgings of each check on "
        f"{results['snapshot']}. Judge cost reported by DeepEval: ${results['cost']:.4f}",
        "",
        "### By metric",
        "",
        "Over every check on the recorded answers. An errored judging counts as a failed one.",
        "",
        "| Metric | Checks | Mean flip rate | Fail rate | Failed every judging |",
        "|---|---:|---:|---:|---:|",
        *_by_metric(rows),
        "",
        "### By case",
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
    unchecked = results.get("unchecked", [])
    if results["known_bad"] or unchecked:
        lines += ["", "### Known-bad answers", ""]
    if unchecked:
        names = ", ".join(f"{entry['case_id']} {entry['kind']}" for entry in unchecked)
        lines += [
            f"{len(unchecked)} known-bad "
            + ("entry has" if len(unchecked) == 1 else "entries have")
            + " no check yet (empty must_fail) and "
            + ("is" if len(unchecked) == 1 else "are")
            + f" not judged: {names}.",
            "",
        ]
    if results["known_bad"]:
        lines += [
            "A miss is a known-bad answer that passes most of its judgings. Unjudged means a "
            "judging errored, so the result says nothing about the metric.",
            "",
            "| Case | Kind | Metric | Scores | Passed | Verdict |",
            "|---|---|---|---|---:|:---:|",
        ]
        for entry in results["known_bad"]:
            if any(entry["errors"]):
                verdict = "unjudged"  # an error is not a fail here: the judge never read it
            else:
                verdict = "MISS" if passes_majority(entry["passed"]) else "caught"
            lines.append(
                f"| {entry['case_id']} | {entry['kind']} | {entry['key']} "
                f"| {_scores(entry['scores'])} "
                f"| {sum(entry['passed'])}/{len(entry['passed'])} | {verdict} |"
            )
    lines += ["", _gate_estimate(rows, results["times"])]
    return "\n".join(lines)


def _by_metric(rows: list[dict]) -> list[str]:
    """A table row per metric key, in the order the keys first appear."""
    lines = []
    for key in dict.fromkeys(row["key"] for row in rows):
        checks = [row for row in rows if row["key"] == key]
        judgings = sum(len(row["passed"]) for row in checks)
        failed = judgings - sum(sum(row["passed"]) for row in checks)
        mean_flip = sum(flip_rate(row["passed"]) for row in checks) / len(checks)
        always = sum(1 for row in checks if row["passed"] and not any(row["passed"]))
        fail_rate = failed / judgings if judgings else 0.0
        lines.append(f"| {key} | {len(checks)} | {mean_flip:.1%} | {fail_rate:.1%} | {always} |")
    return lines


def _judgings(count: int) -> str:
    return f"{count} judging" + ("" if count == 1 else "s")


def _scores(scores: list[float | None]) -> str:
    return " ".join("err" if score is None else f"{score:.2f}" for score in scores)


def _gate_estimate(rows: list[dict], times: int) -> str:
    """What the smoke gate would do on a spotless answer: every strict check on a smoke case
    must pass, so each check's pass rate over the judgings multiplies into the chance of a
    green run. Errored judgings count as failures, as they do in the gate.

    The rates come from judging recorded answers taken to be acceptable, so a red run here is
    judge noise; how much the generator's answers vary is not in it. A check that passed all of
    its `times` judgings gets a pass rate of 1 and never fails in the estimate, which is too
    kind: no failure in 5 judgings is still consistent with a failure rate of up to 43%.

    The interval says how much that matters. Method: pool the judgings of all strict smoke
    checks into one failure rate, take its Wilson 95% interval, and send the two ends and the
    rate itself through the gate arithmetic, where every one of the m strict checks must pass
    with that pass rate (a single judging, or the median of three). The arithmetic rises with
    the failure rate, so the ends for the rate give the ends for the gate. Pooling treats every
    check as equally likely to fail, so it spreads the observed failures over all checks. The
    first figures put each failure on the check that showed it, which gives a higher chance of
    a red run for the same number of failures, so they can lie above the interval.
    """
    checks = [row for row in rows if row["smoke"] and row["strict"] and row["passed"]]
    if not checks:
        return "No strict check on a smoke case was judged, so there is no gate estimate."
    rates = [sum(row["passed"]) / len(row["passed"]) for row in checks]
    cases = len({row["case_id"] for row in checks})
    single = gate_false_alarm(rates)
    median = gate_false_alarm([median_of_three_pass(rate) for rate in rates])

    judgings = sum(len(row["passed"]) for row in checks)
    failed = judgings - sum(sum(row["passed"]) for row in checks)
    low, high = wilson_interval(failed, judgings)

    def red_run(failure_rate: float) -> tuple[float, float]:
        passes = [1 - failure_rate] * len(checks)
        return gate_false_alarm(passes), gate_false_alarm([median_of_three_pass(p) for p in passes])

    (single_low, median_low), (single_high, median_high) = red_run(low), red_run(high)
    single_pooled, median_pooled = red_run(failed / judgings)
    return (
        f"Estimated false alarm of the pull-request gate ({len(checks)} strict "
        f"check{'' if len(checks) == 1 else 's'} on {cases} smoke case"
        f"{'' if cases == 1 else 's'}): single judging {single:.1%}, median of 3 {median:.1%}. "
        f"Basis: {failed} of {_judgings(judgings)} of those strict smoke checks failed (an "
        f"errored judging counts as a failure). They are judgings of recorded answers taken to "
        f"be acceptable, so this is judge noise only; variation in what the generator writes is "
        f"not included. "
        f"A check with no failure in {_judgings(times)} is counted as never failing, so these "
        f"figures can be too low. Pooling all {_judgings(judgings)}, and taking every check as "
        f"equally likely to fail, the 95% Wilson interval for the chance of a red run is "
        f"{single_low:.1%} to {single_high:.1%} for a single judging (estimate "
        f"{single_pooled:.1%}) and {median_low:.1%} to {median_high:.1%} for the median of 3 "
        f"(estimate {median_pooled:.1%})."
    )

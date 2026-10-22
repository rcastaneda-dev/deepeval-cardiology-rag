"""Propose a metric threshold from the scores of answers judged acceptable. Pure Python.

The rule: the highest multiple of 0.05 that every acceptable answer passes with at least
`margin` to spare, clamped to the floor and the ceiling. Where the floor and the margin
conflict the floor wins, so an acceptable score under the floor fails and is returned as
failing, and the acceptable scores that pass by less than the margin are returned as margin risks
so a person can see which ones could flip. Give it one score per acceptable answer (the median
of three judgings), so a single odd judging does not set the threshold.

Usage: python -m evals.calibrate <stability.json> <labels.json>

The first file is a results file of scripts/judge_stability.py, the second says which answers a
person read and found acceptable: {"<case id>": {"acceptable": true, "note": "..."}}. For each
judged check the command prints the old threshold (the one the results were judged against, which
every row records as "threshold"; the one in force only for a file that does not), the one the
rule proposes, the lowest acceptable score, the margin risks and the acceptable scores that would
still fail. It makes no API call. A check that needs no judge (DETERMINISTIC) or is only reported
(REPORT_ONLY) is not calibrated. A case's acceptable score is the score the gate could be made to
see at worst:
  - a check judged once by the gate (JUDGED_ONCE): the lowest of the case's single judgings;
  - any other judged check: the lowest median over every 3 of the case's judgings (the gate
    passes on the median of 3), which needs at least 3 judgings.
Only cases labelled acceptable set a threshold, and a case with a judged check that is not
labelled at all stops the run, so no answer is left out by accident. A judging that errored (no
score, or an error recorded with it) is left out and counted.

The labels may also hold a list under "extra", for acceptable answers scored in another run or
trial, such as a live pull-request run or an earlier judging of other answers to the same cases,
that is not in the results file:
  "extra": [{"key": "safety", "case_id": "saf-005", "score": 0.905, "source": "...", "note": "..."}]
Each entry is one more acceptable score for its check, counted as it is (the score the gate
reported: a median of 3, or the lowest single judging of a check in JUDGED_ONCE), and named
"<case id> [extra]" in the output, listed by source. The source says which run and the note what
was read in the answer. An entry for a check that is not calibrated is refused. One for a check
that has no rows in these results is counted as not used, so one labels file can serve results
files that each hold some of the checks.
"""

import argparse
import itertools
import json
import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

from evals.cases import load_goldens
from evals.thresholds import DETERMINISTIC, JUDGED_ONCE, REPORT_ONLY, THRESHOLDS

MEDIAN_OF = 3  # the number of judgings the gate takes the median of

_TOLERANCE = 1e-9  # scores like 0.6 minus 0.05 come out as 0.5499999999999999


def propose_threshold(
    acceptable: list[float],
    *,
    margin: float = 0.05,
    floor: float = 0.5,
    ceiling: float = 0.9,
) -> tuple[float, list[float], list[float]]:
    """The threshold, the acceptable scores at or above it by less than `margin` (margin risks),
    and the acceptable scores below it (failing), each in the order given."""
    if not acceptable:
        raise ValueError("propose_threshold needs at least one acceptable score")
    steps = math.floor(round((min(acceptable) - margin) * 20, 9)) / 20
    threshold = min(max(steps, floor), ceiling)
    margin_risks, failing = [], []
    for score in acceptable:
        if score < threshold - _TOLERANCE:
            failing.append(score)
        elif score - threshold < margin - _TOLERANCE:
            margin_risks.append(score)
    return threshold, margin_risks, failing


class CalibrationError(ValueError):
    """The results or the labels cannot be calibrated; the message says why."""


def acceptable_score(scores: list[float], *, once: bool) -> float:
    """The score one acceptable case counts for: the lowest single judging for a check the gate
    judges once, otherwise the lowest median over every 3 of its judgings, which is the worst
    median of 3 the gate could see (it is the second lowest judging, as the median of the two
    lowest and any third is the larger of the two)."""
    if not scores:
        raise CalibrationError("no judging to take a score from")
    if once:
        return min(scores)
    if len(scores) < MEDIAN_OF:
        raise CalibrationError(f"a median needs at least {MEDIAN_OF} judgings, got {len(scores)}")
    return min(statistics.median(subset) for subset in itertools.combinations(scores, MEDIAN_OF))


@dataclass(frozen=True)
class KeyCalibration:
    """What the rule says about one judged check: `old` is the threshold the results were judged
    against, `proposed` the one the rule gives (None when no case is labelled acceptable),
    `lowest` the case with the lowest acceptable score, `margin_risks` and `failing` the
    acceptable (case, score) pairs that pass by less than the margin or fail, in the order of the
    results."""

    key: str
    once: bool
    old: float
    proposed: float | None
    lowest: tuple[str, float] | None
    margin_risks: list[tuple[str, float]]
    failing: list[tuple[str, float]]
    errored: int
    extras: list[tuple[str, float, str]] = field(default_factory=list)  # case, score, source


@dataclass(frozen=True)
class Calibration:
    keys: list[KeyCalibration]
    not_calibrated: list[str]  # checks in the results that need no judge or are only reported
    acceptable: int  # cases labelled acceptable, among those with a calibrated check
    not_acceptable: int
    extra_scores: int = 0  # entries of the labels' "extra" list that were counted
    extras_not_used: list[tuple[str, str, float]] = field(default_factory=list)  # key, case, score


EXTRA = "extra"  # the key of the labels that holds the extra acceptable scores, not a case id


def _label_problems(labels) -> list[str]:
    if not isinstance(labels, dict):
        return ["labels must be an object of case id to {acceptable, note}"]
    problems = []
    for case_id, label in labels.items():
        if case_id == EXTRA:
            continue
        note = label.get("note") if isinstance(label, dict) else None
        if (
            not isinstance(label, dict)
            or not isinstance(label.get("acceptable"), bool)
            or not isinstance(note, str)
            or not note.strip()
        ):
            problems.append(
                f"{case_id}: label needs acceptable (true or false) and a note saying what was read"
            )
    return problems


def _text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _extra_problems(extras) -> list[str]:
    """Why entries of the labels' "extra" list cannot be counted, one message per entry."""
    if not isinstance(extras, list):
        return ["extra must be a list of {key, case_id, score, source, note}"]
    case_ids = {golden.id for golden in load_goldens()}
    problems = []
    for number, entry in enumerate(extras, start=1):
        where = f"extra {number}"
        if not isinstance(entry, dict) or not all(
            _text(entry.get(field_name)) for field_name in ("key", "case_id", "source", "note")
        ):
            problems.append(f"{where}: needs key, case_id, score, source and note")
            continue
        score, key = entry.get("score"), entry["key"]
        if isinstance(score, bool) or not isinstance(score, int | float) or not 0 <= score <= 1:
            problems.append(f"{where}: score must be a number from 0 to 1")
        elif key not in THRESHOLDS:
            problems.append(f"{where}: not a metric key: {key}")
        elif entry["case_id"] not in case_ids:
            problems.append(f"{where}: not a case id: {entry['case_id']}")
        elif key in DETERMINISTIC or key in REPORT_ONLY:
            problems.append(
                f"{where}: {key} is not calibrated (it needs no judge or is only reported)"
            )
    return problems


def _usable(row: dict) -> tuple[list[float], int]:
    """The scores of a row's judgings that scored without an error, and how many were left out."""
    errors = row.get("errors") or [None] * len(row["scores"])
    scores = [
        score
        for score, error in zip(row["scores"], errors, strict=True)
        if score is not None and error is None
    ]
    return scores, len(row["scores"]) - len(scores)


def calibrate(results: dict, labels: dict) -> Calibration:
    """The rule applied to every judged check of a judge_stability results file. Raises
    CalibrationError when the labels or the results cannot be used (see the module docstring)."""
    problems = _label_problems(labels)
    if problems:
        raise CalibrationError("; ".join(problems))
    rows = results.get("rows") if isinstance(results, dict) else None
    if not isinstance(rows, list) or not rows:
        raise CalibrationError("the results have no rows")
    unknown = sorted({row["key"] for row in rows} - set(THRESHOLDS))
    if unknown:
        raise CalibrationError(f"not a metric key: {', '.join(unknown)}")

    skipped = DETERMINISTIC | set(REPORT_ONLY)
    not_calibrated = list(dict.fromkeys(row["key"] for row in rows if row["key"] in skipped))
    judged = [row for row in rows if row["key"] not in skipped]
    cases = list(dict.fromkeys(row["case_id"] for row in judged))
    unlabelled = sorted(case_id for case_id in cases if case_id not in labels)
    if unlabelled:
        raise CalibrationError(
            f"{len(unlabelled)} {'case' if len(unlabelled) == 1 else 'cases'} not labelled: "
            f"{', '.join(unlabelled)}"
        )

    judged_keys = list(dict.fromkeys(row["key"] for row in judged))
    extras = labels.get(EXTRA, [])
    problems = _extra_problems(extras)
    if problems:
        raise CalibrationError("; ".join(problems))
    not_used = [entry for entry in extras if entry["key"] not in judged_keys]

    keys = []
    for key in judged_keys:
        once = key in JUDGED_ONCE
        pairs, errored = [], 0
        for row in judged:
            if row["key"] != key or not labels[row["case_id"]]["acceptable"]:
                continue
            scores, left_out = _usable(row)
            errored += left_out
            try:
                pairs.append((row["case_id"], acceptable_score(scores, once=once)))
            except CalibrationError as error:
                raise CalibrationError(
                    f"{row['case_id']} {key}: {len(scores)} usable judgings of "
                    f"{len(row['scores'])}; {error}"
                ) from error
        counted = [entry for entry in extras if entry["key"] == key]
        pairs += [(f"{entry['case_id']} [{EXTRA}]", float(entry["score"])) for entry in counted]
        # What the answers were judged against: the rows say, the code only when they do not.
        recorded = next(
            (row["threshold"] for row in judged if row["key"] == key and "threshold" in row), None
        )
        old = THRESHOLDS[key] if recorded is None else float(recorded)
        if not pairs:
            keys.append(KeyCalibration(key, once, old, None, None, [], [], errored))
            continue
        proposed, margin_risks, failing = propose_threshold([score for _, score in pairs])
        keys.append(
            KeyCalibration(
                key=key,
                once=once,
                old=old,
                proposed=proposed,
                lowest=min(pairs, key=lambda pair: pair[1]),
                margin_risks=[pair for pair in pairs if pair[1] in margin_risks],
                failing=[pair for pair in pairs if pair[1] in failing],
                errored=errored,
                extras=[(e["case_id"], float(e["score"]), e["source"]) for e in counted],
            )
        )
    verdicts = [labels[case_id]["acceptable"] for case_id in cases]
    return Calibration(
        keys,
        not_calibrated,
        verdicts.count(True),
        verdicts.count(False),
        len(extras) - len(not_used),
        [(entry["key"], entry["case_id"], float(entry["score"])) for entry in not_used],
    )


def _pairs(pairs: list[tuple[str, float]]) -> str:
    return ", ".join(f"{case_id} {score:.2f}" for case_id, score in pairs) or "none"


def render(results: dict, calibration: Calibration) -> str:
    """The calibration as Markdown: a header, then a section per judged check."""
    acceptable, extra = calibration.acceptable, calibration.extra_scores
    more = (
        f", plus {extra} extra acceptable {'score' if extra == 1 else 'scores'} from other runs"
        if extra
        else ""
    )
    lines = [
        f"Thresholds proposed from {results.get('times', '?')} judgings of each check on "
        f"{results.get('snapshot', '?')}: {acceptable} {'case' if acceptable == 1 else 'cases'} "
        f"labelled acceptable, {calibration.not_acceptable} labelled not acceptable{more}. "
        "Margin 0.05, clamped to 0.50-0.90.",
    ]
    for found in calibration.keys:
        basis = (
            "lowest single judging of each case (the gate judges this check once)"
            if found.once
            else "lowest median over every 3 judgings of each case (the gate takes the median of 3)"
        )
        proposed = (
            "none (no case is labelled acceptable for this check)"
            if found.proposed is None
            else f"{found.proposed:.2f}"
        )
        lines += ["", f"### {found.key}", "", f"Basis: {basis}.", ""]
        lines.append(f"- old threshold: {found.old:.2f}")
        lines.append(f"- proposed threshold: {proposed}")
        if found.lowest:
            case_id, score = found.lowest
            lines.append(f"- lowest acceptable score: {score:.2f} ({case_id})")
        lines.append(f"- margin risks: {_pairs(found.margin_risks)}")
        lines.append(f"- acceptable scores that would fail: {_pairs(found.failing)}")
        if found.extras:
            by_source: dict[str, list[str]] = {}
            for case_id, score, source in found.extras:
                by_source.setdefault(source, []).append(f"{case_id} {score:.2f}")
            counted = "; ".join(
                f"{', '.join(items)} ({source})" for source, items in by_source.items()
            )
            lines.append(f"- extra acceptable scores counted: {counted}")
        if found.errored:
            lines.append(f"- errored judgings left out: {found.errored}")
    if calibration.extras_not_used:
        counts: dict[str, int] = {}
        for key, _, _ in calibration.extras_not_used:
            counts[key] = counts.get(key, 0) + 1
        unused = ", ".join(f"{key} ({count})" for key, count in counts.items())
        lines += [
            "",
            f"Extra scores not used (no rows for their check in these results): {unused}.",
        ]
    if calibration.not_calibrated:
        lines += [
            "",
            f"Not calibrated: {', '.join(calibration.not_calibrated)} (they need no judge or "
            "are only reported).",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stability", type=Path, help="a results file of scripts/judge_stability.py")
    parser.add_argument("labels", type=Path, help="JSON of case id to {acceptable, note}")
    args = parser.parse_args(argv)
    try:
        results = json.loads(args.stability.read_text(encoding="utf-8"))
        labels = json.loads(args.labels.read_text(encoding="utf-8"))
        calibration = calibrate(results, labels)
    except (OSError, ValueError, KeyError, TypeError) as error:  # CalibrationError is a ValueError
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(render(results, calibration))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

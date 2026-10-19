"""Propose a metric threshold from the scores of answers judged acceptable. Pure Python.

The rule: the highest multiple of 0.05 that every acceptable answer passes with at least
`margin` to spare, clamped to the floor and the ceiling. Where the floor and the margin
conflict the floor wins, so an acceptable score under the floor fails and is returned as
failing, and the acceptable scores that pass by less than the margin are returned as margin risks
so a person can see which ones could flip. Give it one score per acceptable answer (the median
of three judgings), so a single odd judging does not set the threshold.
"""

import math

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

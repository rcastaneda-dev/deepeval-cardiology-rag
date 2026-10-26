"""Judge one answer several times and take the median, so one bad judging cannot fail the gate.

The judge (gpt-4o-mini) sometimes scores the same answer very differently from one judging to
the next, and a single such flip turns the blocking pull-request check red. It has done so twice
on answers a person read as right: Bias on bia-006 and Medical safety on saf-001. The median of
three moves only when two judgings go wrong, so a check that passes 90% of its judgings passes
its median 97.2% of the time (evals.stability.median_of_three_pass).

MedianOf stands in for the metric it repeats: DeepEval reads the same name, threshold, flaky flag,
cost, reason and error from it as from a plain one. evals.metrics.metric_specs_for wraps the
judged checks that can fail the gate; it leaves out the checks that need no judge (a repeat of a
pattern match always gives the same answer), the Faithfulness checks that gpt-5-mini judges
(JUDGED_ONCE) and the report-only metrics (REPORT_ONLY), all in evals/thresholds.py.
"""

import asyncio
import statistics
from collections.abc import Callable

from deepeval.metrics import BaseMetric


class MedianOf(BaseMetric):
    """`times` copies of one metric, each judging the answer on its own; the score is their median.

    `build` makes one new copy per call. The threshold, flaky flag, name and judge model are the
    first copy's. A copy that ends in an error (or without a score) makes the whole check an error
    that names the judging, and an errored check fails whatever the median is: a malformed score,
    such as the 3.54 BoundedGEval refuses, never gets outvoted into a pass. An exception from a
    copy is raised as a plain metric would raise it, after the other copies have finished.

    One error is given a second chance (Ruling 41): a copy whose judge returned a score outside 0
    to 1 (`out_of_range`, set by BoundedGEval) is replaced once by a fresh copy, which judges the
    answer again, and the reason says "judging i of n re-judged after an out-of-range score".
    Only if the new judging is out of range too, or errors in any way, is the check an error. Any
    other error fails the check at once, as before.

    DeepEval builds a measured metric again from `build` and `times` (copy_metrics), so those two
    names must stay as they are. With an even number of copies the median is the mean of the two
    in the middle, and the reason is the lower one's.

    DeepEval's verbose_mode is not passed on to the copies; they print only what `build` gives
    them.
    """

    def __init__(self, build: Callable[[], BaseMetric], times: int) -> None:
        if not isinstance(times, int) or times < 1:
            raise ValueError(f"times must be a whole number of at least 1, not {times!r}")
        self.build = build
        self.times = times
        self.copies = [build() for _ in range(times)]
        self._discarded: list[BaseMetric] = []  # the judgings that were out of range
        self._retried: dict[int, str] = {}  # copy number -> the error its first judging ended in
        first = self.copies[0]
        self.threshold = first.threshold
        self.flaky = first.flaky
        self.strict_mode = first.strict_mode
        self.evaluation_model = first.evaluation_model
        self._name = first.__name__

    def measure(self, test_case, *args, **kwargs) -> float | None:
        self._prepare()
        for copy in self.copies:
            copy.measure(test_case, *args, **kwargs)
        for number in self._out_of_range():
            self._replace(number).measure(test_case, *args, **kwargs)
        return self._combine()

    async def a_measure(self, test_case, *args, **kwargs) -> float | None:
        self._prepare()
        # return_exceptions: let every copy finish before one's exception is raised, so none is
        # left running behind it.
        outcomes = await asyncio.gather(
            *(copy.a_measure(test_case, *args, **kwargs) for copy in self.copies),
            return_exceptions=True,
        )
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome
        fresh = [self._replace(number) for number in self._out_of_range()]
        outcomes = await asyncio.gather(
            *(copy.a_measure(test_case, *args, **kwargs) for copy in fresh),
            return_exceptions=True,
        )
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome
        return self._combine()

    def _prepare(self) -> None:
        """What DeepEval does to a metric before it measures, done for the copies: it clears the
        error (BoundedGEval sets one but never clears it) and, in its synchronous path, switches
        async_mode off on the metric it is given."""
        self._discarded, self._retried = [], {}
        for copy in self.copies:
            copy.error = None
            copy.async_mode = self.async_mode

    def _out_of_range(self) -> list[int]:
        """The numbers of the copies whose judge returned a score outside 0 to 1."""
        return [
            number
            for number, copy in enumerate(self.copies)
            if copy.error is not None and getattr(copy, "out_of_range", False)
        ]

    def _replace(self, number: int) -> BaseMetric:
        """Puts a fresh copy in place of copy `number`, keeping the old one for its cost, and
        returns it ready to measure."""
        old = self.copies[number]
        self._discarded.append(old)
        self._retried[number] = str(old.error)
        fresh = self.build()
        fresh.error = None
        fresh.async_mode = self.async_mode
        self.copies[number] = fresh
        return fresh

    def _combine(self) -> float | None:
        """Sets this metric's score, reason, error, cost and verdict from the measured copies."""
        count = len(self.copies)
        scored = sorted((copy for copy in self.copies if copy.score is not None), key=_score)
        self.score = statistics.median(copy.score for copy in scored) if scored else None
        shown = ", ".join(
            "err" if copy.score is None else f"{copy.score:.2f}" for copy in self.copies
        )
        self.reason = f"Median of {count} judgings ({shown})"
        self.reason += "".join(
            f"; judging {number + 1} of {count} re-judged after an out-of-range score"
            for number in sorted(self._retried)
        )
        self.reason += "."
        if scored:
            middle = scored[(len(scored) - 1) // 2]
            if middle.reason:
                self.reason += f" {middle.reason}"

        problems = []
        for number, copy in enumerate(self.copies, 1):
            if copy.error is not None:
                again = self._retried.get(number - 1)
                after = "" if again is None else f" (re-judged once after: {again})"
                problems.append(f"judging {number} of {count}: {copy.error}{after}")
            elif copy.score is None:
                problems.append(f"judging {number} of {count}: no score")
        self.error = "; ".join(problems) or None

        paid = [*self.copies, *self._discarded]  # a discarded judging was paid for too
        costs = [copy.evaluation_cost for copy in paid]
        self.evaluation_cost = None if None in costs else sum(costs)
        self.input_tokens = _total(copy.input_tokens for copy in paid)
        self.output_tokens = _total(copy.output_tokens for copy in paid)
        self.is_successful()  # sets self.success; an error always fails
        return self.score

    @property
    def __name__(self) -> str:
        return self._name


def _score(copy: BaseMetric) -> float:
    return copy.score


def _total(counts) -> int | None:
    """The sum of the counts, or None if any copy did not report one."""
    counts = list(counts)
    return None if None in counts else sum(counts)

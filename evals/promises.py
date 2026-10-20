"""What the app promises and which checks prove each promise: the single source of truth that the
QA report's traceability matrix is printed from (`python -m evals.promises`).

A check is one of:
- a metric key from evals/thresholds.py (judged or deterministic; whether a given case enforces it
  depends on `keys_for`, on `FLAKY` and on `REPORT_ONLY`, see evals/metrics.py `is_strict`);
- `unit:<path>`, a test file that runs in the `unit tests and lint` check on every pull request;
- `measured:<path>#<anchor>`, a figure recorded once, in the named heading of a document.

tests/test_promises.py fails if a check names a metric, file or heading that does not exist, or
if a promise in the PR gate has no strict check on a smoke case.
"""

from dataclasses import dataclass

from evals.cases import load_goldens
from evals.metrics import is_strict
from evals.thresholds import THRESHOLDS, keys_for


@dataclass(frozen=True)
class Promise:
    id: str
    text: str
    # Where the promise comes from: the system prompt, the design spec or a pull request.
    source: str
    checks: tuple[str, ...]
    # Whether the pull request gate enforces it: a strict check on a smoke case, or a unit test.
    gated: bool
    # The goldens that exercise the promise. When given, only these count as its coverage: a
    # safety case about a dose says nothing about the app standing in for a clinician, though both
    # run the `safety` metric. When empty, any golden that runs one of the checks counts, unless
    # `categories` restricts which goldens do.
    cases: tuple[str, ...] = ()
    # The categories whose goldens exercise the promise (an out-of-scope answer has no citation to
    # check, though faithfulness runs on it). When given with `cases`, a golden must match both.
    categories: tuple[str, ...] = ()


PROMISES: tuple[Promise, ...] = (
    Promise(
        "P1",
        "The answer cites its sources",
        "system prompt rule 1, spec goal 1",
        ("cites_passages",),
        gated=True,
        categories=("in_corpus", "bias"),
    ),
    Promise(
        "P2",
        "The app abstains when a question is out of scope",
        "system prompt rule 2, spec goal 1",
        ("exact_refusal", "abstention"),
        gated=True,
    ),
    Promise(
        "P3",
        "The answer uses only the retrieved passages",
        "system prompt rule 1, spec goal 1",
        ("faithfulness", "bias_faithfulness", "hallucination", "abstention"),
        gated=True,
    ),
    Promise(
        "P4",
        "An emergency question gets the instruction to call 911 first",
        "system prompt rule 4",
        ("emergency_first", "safety"),
        gated=True,
        cases=("saf-001", "saf-002", "saf-006", "saf-010"),
    ),
    Promise(
        "P5",
        "The app never diagnoses or recommends a dose or a medicine change",
        "system prompt rule 3",
        ("safety",),
        gated=True,
        cases=("saf-003", "saf-004", "saf-005", "saf-007"),
    ),
    Promise(
        "P6",
        "The app does not offer itself in place of a clinician",
        "system prompt rule 3",
        ("safety",),
        gated=True,
        cases=("saf-008",),
    ),
    Promise(
        "P7",
        "A question about a group gets a respectful answer grounded in the passages, with no "
        "stereotype and no bare refusal where the golden answer asks for content",
        "system prompt rule 2 (claims about a group), spec goal 2",
        ("respectful_grounded",),
        gated=True,
    ),
    Promise(
        "P8",
        "Retrieval finds the passage that answers the question",
        "spec goal 2",
        ("contextual_recall", "contextual_precision"),
        gated=True,
    ),
    Promise(
        "P9",
        "The answer addresses the question asked",
        "spec goal 2",
        ("answers_question",),
        gated=True,
    ),
    # No eval case plants an instruction inside a passage, so only the prompt-building unit test
    # covers this; it is a known gap, not a gated promise.
    Promise(
        "P10",
        "Text inside a passage is data, never an instruction",
        "system prompt rule 6",
        ("unit:tests/test_generate.py",),
        gated=False,
    ),
    Promise(
        "P11",
        "A run that scored nothing fails instead of passing",
        "PR 4 (fail closed)",
        ("unit:tests/test_check_eval_keys.py",),
        gated=True,
    ),
    # Measured once per release (docs/findings.md section 5), not on every pull request; the figure
    # there is an estimate under the current judges until a gate run measures it.
    Promise(
        "P12",
        "A full run costs about $1 or less",
        "spec success criterion 4",
        ("measured:docs/findings.md#5-cost",),
        gated=False,
    ),
)


def _smoke_cases(promise: Promise) -> str:
    """The smoke cases that enforce the promise (among those it names, if it names any), or why
    there are none to list."""
    keys = [check for check in promise.checks if check in THRESHOLDS]
    ids = [
        golden.id
        for golden in load_goldens()
        if golden.smoke
        and (not promise.cases or golden.id in promise.cases)
        and (not promise.categories or golden.category in promise.categories)
        and any(key in keys_for(golden) and is_strict(golden, key) for key in keys)
    ]
    if ids:
        return ", ".join(ids)
    return "unit tests" if any(check.startswith("unit:") for check in promise.checks) else "none"


def traceability_markdown() -> str:
    """A Markdown table with one row per promise: its source, its checks, whether the pull
    request gate enforces it and the smoke cases that enforce it."""
    lines = [
        "| Promise | Source | Checks | In PR gate | Smoke cases |",
        "|---|---|---|---|---|",
    ]
    for promise in PROMISES:
        checks = ", ".join(f"`{check}`" for check in promise.checks)
        gate = "yes" if promise.gated else "no"
        lines.append(
            f"| {promise.id}: {promise.text} | {promise.source} | {checks} | {gate} "
            f"| {_smoke_cases(promise)} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(traceability_markdown())

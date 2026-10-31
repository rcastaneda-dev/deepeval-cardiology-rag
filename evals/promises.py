"""What the app promises and which checks prove each promise: the single source of truth that the
QA report's traceability matrix is printed from (`python -m evals.promises`).

A check is one of:
- a metric key from evals/thresholds.py (judged or deterministic; whether a given case enforces it
  depends on `keys_for`, on `FLAKY` and on `REPORT_ONLY`, see evals/metrics.py `is_strict`);
- `unit:<path>`, a test file that runs in the `unit tests and lint` check on every pull request;
- `measured:<path>#<anchor>`, a figure recorded once, in the named heading of a document.
A promise with no check is a known gap, and is never gated.

Sources: "system prompt rule N" is SYSTEM_PROMPT in src/cardio_rag/generate.py. "Design goal N"
and "design success criterion N" are the project's original design, which is not in the
repository; docs/qa-report.md quotes each one named here.

tests/test_promises.py fails if a check names a metric, file or heading that does not exist, if
a promise in the PR gate has no strict check on a smoke case, or if a golden case belongs to no
promise.
"""

from dataclasses import dataclass

from evals.cases import load_goldens
from evals.metrics import is_strict
from evals.thresholds import REPORT_ONLY, THRESHOLDS, keys_for


@dataclass(frozen=True)
class Promise:
    id: str
    text: str
    # Where the promise comes from: the system prompt, the original design, the owner's decision
    # or a pull request (see the module docstring).
    source: str
    checks: tuple[str, ...]
    # Whether the pull request gate enforces the behaviour itself: a strict check on a smoke case,
    # or a unit test of that behaviour. A unit test that only reads the system prompt's text (P10)
    # does not count, nor does a figure measured once.
    gated: bool
    # The goldens that exercise the promise. When given, only these count as its coverage: a
    # safety case about a dose says nothing about the app standing in for a clinician, though both
    # run the `safety` metric. When empty, any golden that runs one of the checks counts, unless
    # `categories` restricts which goldens do.
    cases: tuple[str, ...] = ()
    # The categories whose goldens exercise the promise (an out-of-scope answer has no citation to
    # check). When given with `cases`, a golden must match both.
    categories: tuple[str, ...] = ()


PROMISES: tuple[Promise, ...] = (
    Promise(
        "P1",
        "The answer cites its sources",
        "system prompt rule 1, design goal 1",
        ("cites_passages",),
        gated=True,
        categories=("in_corpus", "bias"),
    ),
    Promise(
        "P2",
        "The app abstains when a question is out of scope",
        "system prompt rule 2, design goal 1",
        ("exact_refusal", "abstention"),
        gated=True,
    ),
    Promise(
        "P3",
        "The answer uses only the retrieved passages",
        "system prompt rule 1, design goal 1",
        # On out-of-scope cases abstention alone enforces it: an answer from memory fails the
        # rubric, and Faithfulness has not run there since 2026-10-31 (evals/thresholds.py).
        ("faithfulness", "bias_faithfulness", "hallucination", "abstention"),
        gated=True,
    ),
    Promise(
        "P4",
        "An emergency question gets the instruction to call 911 first",
        "system prompt rule 4",
        ("emergency_first", "safety"),
        gated=True,
        cases=("saf-001", "saf-002", "saf-006", "saf-010", "saf-011"),
    ),
    # Every safety case: a diagnosis can come in any answer about a person's own health, and the
    # ones the app wrote were all on saf-001 (local runs of 2026-10-15 and 2026-10-16, the PR 6
    # smoke run and three seeded-fault passes, the control among them). no_symptom_diagnosis gates
    # the owner's named forms; the safety rubric gates the rest of the promise, and
    # diagnosis_wording, the broad reading, reports what the core does not read (report-only, so
    # it is not what makes the promise gated).
    Promise(
        "P5",
        "The app never diagnoses, and never tells a person that their own symptoms match, point "
        "to, are signs of or are caused by a condition",
        "system prompt rule 3, the owner's diagnosis line of 2026-10-20 and 2026-10-26",
        ("safety", "no_symptom_diagnosis", "diagnosis_wording"),
        gated=True,
        categories=("safety",),
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
        "system prompt rule 2 (claims about a group), design goal 2",
        ("respectful_grounded",),
        gated=True,
    ),
    Promise(
        "P8",
        "Retrieval finds the passage that answers the question",
        "design goal 2",
        ("contextual_recall", "contextual_precision"),
        gated=True,
    ),
    Promise(
        "P9",
        "The answer addresses the question asked",
        "design goal 2",
        ("answers_question",),
        gated=True,
    ),
    # No eval case plants an instruction inside a passage, so only the prompt-building unit test
    # covers this, and it reads the prompt's text, not what the app does with such a passage: a
    # known gap, not a gated promise.
    Promise(
        "P10",
        "Text inside a passage is data, never an instruction",
        "system prompt rule 6",
        ("unit:tests/test_generate.py",),
        gated=False,
    ),
    # The unit test covers missing keys only. A rejected key also turns a pull-request run red,
    # because every case errors (PR 11's first smoke run), but no test pins that. A full run is
    # report-only, but since 2026-10-31 it is red when it scores no case (tests/test_workflow.py;
    # docs/qa-report.md, section 7).
    Promise(
        "P11",
        "A pull-request run that scored nothing fails instead of passing",
        "PR 4 (fail closed)",
        ("unit:tests/test_check_eval_keys.py",),
        gated=True,
    ),
    # Measured once (docs/findings.md section 5), not on every pull request: one local full run
    # under the current judges, whose results file is not committed.
    Promise(
        "P12",
        "A full run costs about $1 or less",
        "design success criterion 4",
        ("measured:docs/findings.md#5-cost",),
        gated=False,
    ),
    Promise(
        "P13",
        "The app never recommends a personal treatment, medicine or dose, and says not to start, "
        "stop or change a medicine without a clinician",
        "system prompt rule 3",
        ("safety",),
        gated=True,
        cases=("saf-004", "saf-005", "saf-007"),
    ),
    # No check reads how long an answer is or how plain its language is: a known gap.
    Promise(
        "P14",
        "Answers are short and in plain language",
        "system prompt rule 5",
        (),
        gated=False,
    ),
    Promise(
        "P15",
        "A clean clone, uv sync, the two API keys and one command run the smoke suite",
        "design success criterion 1",
        ("measured:docs/qa-report.md#10-the-clean-clone-run",),
        gated=False,
    ),
    # Shown by the throwaway pull requests of 2026-10-16 and 2026-10-17 and by the seeded faults;
    # every pull request runs the same checks, but "a broken prompt fails" is not re-proved on each.
    Promise(
        "P16",
        "The smoke gate passes on main and fails when a deliberately broken prompt is pushed",
        "design success criterion 2",
        (
            "measured:docs/findings.md#6-the-gate-catching-a-regression",
            "measured:docs/evidence/seeded-faults-2026-10-26.md#matrix",
        ),
        gated=False,
    ),
    Promise(
        "P17",
        "Retrieval is scored apart from generation (Contextual Precision and Recall score the "
        "retrieved chunks)",
        "design success criterion 3",
        ("contextual_precision", "contextual_recall"),
        gated=True,
    ),
    Promise(
        "P18",
        "The README states the corpus sources, the disclaimer and the limits of the evaluation",
        "design success criterion 5",
        ("unit:tests/test_readme.py",),
        gated=True,
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
    if any(check.startswith("unit:") for check in promise.checks):
        return "unit tests"
    if any(check.startswith("measured:") for check in promise.checks):
        return "measured once"
    return "none"


def traceability_markdown() -> str:
    """A Markdown table with one row per promise: its source, its checks (a report-only one marked
    "(report-only)": it backs the promise and never decides the gate), whether the pull request
    gate enforces it and the smoke cases that enforce it."""
    lines = [
        "| Promise | Source | Checks | In PR gate | Smoke cases |",
        "|---|---|---|---|---|",
    ]
    for promise in PROMISES:
        checks = ", ".join(
            f"`{check}`" + (" (report-only)" if check in REPORT_ONLY else "")
            for check in promise.checks
        )
        gate = "yes" if promise.gated else "no"
        lines.append(
            f"| {promise.id}: {promise.text} | {promise.source} | {checks or 'none'} | {gate} "
            f"| {_smoke_cases(promise)} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(traceability_markdown())

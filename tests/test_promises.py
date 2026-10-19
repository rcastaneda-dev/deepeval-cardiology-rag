import re
from pathlib import Path

from evals.cases import load_goldens
from evals.metrics import is_strict
from evals.promises import PROMISES, traceability_markdown
from evals.thresholds import THRESHOLDS, keys_for

REPO = Path(__file__).resolve().parents[1]


def _slug(heading: str) -> str:
    """The anchor GitHub gives a Markdown heading."""
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def _problem(check: str) -> str | None:
    """Why `check` is not a metric key, an existing unit test file or a recorded measurement."""
    if check.startswith("unit:"):
        path = check.removeprefix("unit:")
        return None if (REPO / path).is_file() else f"{path} does not exist"
    if check.startswith("measured:"):
        path, _, anchor = check.removeprefix("measured:").partition("#")
        if not (REPO / path).is_file():
            return f"{path} does not exist"
        headings = re.findall(r"^#+\s+(.*)$", (REPO / path).read_text("utf-8"), re.MULTILINE)
        return None if anchor in map(_slug, headings) else f"{path} has no heading #{anchor}"
    return None if check in THRESHOLDS else f"{check} is not a metric key"


def test_every_check_is_a_metric_key_or_a_named_unit_test_or_measurement():
    for promise in PROMISES:
        assert promise.checks, promise.id
        for check in promise.checks:
            assert _problem(check) is None, f"{promise.id}: {_problem(check)}"


def test_the_check_validator_rejects_unknown_keys_and_missing_files():
    assert _problem("faithfulness") is None
    assert _problem("unit:tests/test_generate.py") is None
    assert _problem("measured:docs/findings.md#5-cost") is None
    assert _problem("faithfullness") is not None
    assert _problem("unit:tests/test_nothing.py") is not None
    assert _problem("measured:docs/nothing.md#5-cost") is not None
    assert _problem("measured:docs/findings.md#no-such-heading") is not None


def test_the_promises_are_p1_to_p12_each_with_text_and_source():
    assert [promise.id for promise in PROMISES] == [f"P{n}" for n in range(1, 13)]
    assert all(promise.text and promise.source for promise in PROMISES)


def test_the_two_promises_without_an_eval_case_or_a_gate_are_marked_not_gated():
    assert {promise.id for promise in PROMISES if not promise.gated} == {"P10", "P12"}


def test_every_gated_promise_has_a_strict_check_on_a_smoke_case():
    # A gated promise whose only checks are unit tests is covered by those tests running on
    # every pull request; any other gated promise needs a smoke case that really enforces it.
    smoke = [golden for golden in load_goldens() if golden.smoke]
    for promise in PROMISES:
        keys = [check for check in promise.checks if check in THRESHOLDS]
        if not promise.gated or not keys:
            continue
        assert any(
            key in keys_for(golden) and is_strict(golden, key) for golden in smoke for key in keys
        ), f"{promise.id} has no strict check on a smoke case"


def test_the_smoke_subset_includes_bia_004():
    # The bias rubric (respectful_grounded) exists to fail the bare refusal bia-004 once got.
    assert "bia-004" in {golden.id for golden in load_goldens() if golden.smoke}


def test_traceability_lists_every_promise():
    table = traceability_markdown()
    for promise in PROMISES:
        assert promise.id in table
        assert promise.text in table
        assert promise.source in table
    assert "bia-004" in table
    assert len(re.findall(r"^\| P\d+: ", table, re.MULTILINE)) == len(PROMISES)

import re
from pathlib import Path

from evals.cases import CATEGORIES, load_goldens
from evals.metrics import is_strict
from evals.promises import PROMISES, traceability_markdown
from evals.thresholds import METRIC_KEYS, THRESHOLDS, keys_for

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


def _metric_keys(promise):
    return [check for check in promise.checks if check in THRESHOLDS]


def _cases_of(promise):
    """The goldens that exercise the promise: those in the categories and with the ids it names
    (both must hold when it names both), else any golden."""
    return [
        g
        for g in load_goldens()
        if (not promise.cases or g.id in promise.cases)
        and (not promise.categories or g.category in promise.categories)
    ]


def test_every_gated_promise_has_a_strict_check_on_a_smoke_case():
    # A gated promise whose only checks are unit tests is covered by those tests running on
    # every pull request; any other gated promise needs a smoke case that really enforces it.
    # A promise that names its cases is judged on those alone: another smoke case running the
    # same metric key says nothing about, for example, a question about replacing a clinician.
    for promise in PROMISES:
        keys = _metric_keys(promise)
        if not promise.gated or not keys:
            continue
        assert any(
            golden.smoke and key in keys_for(golden) and is_strict(golden, key)
            for golden in _cases_of(promise)
            for key in keys
        ), f"{promise.id} has no strict check on a smoke case"


def test_every_case_a_promise_names_is_a_golden_that_runs_one_of_its_checks():
    known = {golden.id: golden for golden in load_goldens()}
    for promise in PROMISES:
        for case_id in promise.cases:
            assert case_id in known, f"{promise.id}: {case_id} is not a golden"
            assert set(_metric_keys(promise)) & set(keys_for(known[case_id])), (
                f"{promise.id}: {case_id} runs none of {promise.checks}"
            )


def test_every_category_a_promise_names_is_real_and_runs_one_of_its_checks():
    for promise in PROMISES:
        for category in promise.categories:
            assert category in CATEGORIES, f"{promise.id}: {category} is not a category"
            assert set(_metric_keys(promise)) & set(METRIC_KEYS[category]), (
                f"{promise.id}: no check of {promise.checks} runs on {category} cases"
            )


def test_promises_name_the_cases_and_categories_that_exercise_them():
    by_id = {promise.id: promise for promise in PROMISES}
    emergencies = tuple(g.id for g in load_goldens("safety") if g.emergency)
    assert by_id["P4"].cases == emergencies
    assert by_id["P5"].cases == ("saf-003", "saf-004", "saf-005", "saf-007")
    assert by_id["P6"].cases == ("saf-008",)
    # cites_passages does not run on out-of-scope cases, whose correct answer has no citation.
    assert by_id["P1"].categories == ("in_corpus", "bias")
    # The abstention rubric is what fails an out-of-scope answer given from memory.
    assert "abstention" in by_id["P3"].checks


def test_the_smoke_subset_includes_saf_008():
    # saf-008 is the only case that asks whether the app can stand in for a clinician (P6).
    assert "saf-008" in {golden.id for golden in load_goldens() if golden.smoke}


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


def test_traceability_lists_only_the_smoke_cases_a_promise_names():
    rows = {
        line.split(":")[0].removeprefix("| "): line.split("|")[-2].strip()
        for line in traceability_markdown().splitlines()
        if re.match(r"\| P\d+: ", line)
    }
    assert "oos-" not in rows["P1"]
    assert {"in-001", "bia-004"} <= set(rows["P1"].split(", "))
    assert {"oos-001", "bia-004"} <= set(rows["P3"].split(", "))
    assert rows["P4"] == "saf-001"
    assert rows["P5"] == "saf-005, saf-007"
    assert rows["P6"] == "saf-008"
    assert rows["P10"] == rows["P11"] == "unit tests"
    assert rows["P12"] == "none"

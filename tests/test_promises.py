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
        # A promise with no check is a known gap, and a gap is never gated.
        assert promise.checks or not promise.gated, promise.id
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


def test_the_promises_are_p1_to_p18_each_with_text_and_source():
    assert [promise.id for promise in PROMISES] == [f"P{n}" for n in range(1, 19)]
    assert all(promise.text and promise.source for promise in PROMISES)


def test_the_promises_without_an_eval_case_or_a_gate_are_marked_not_gated():
    # P10's unit test reads the prompt's text only; P12, P15 and P16 are measured once; P14 has
    # no check at all.
    assert {promise.id for promise in PROMISES if not promise.gated} == {
        "P10",
        "P12",
        "P14",
        "P15",
        "P16",
    }


def test_no_source_points_at_a_document_outside_the_repository():
    # The design spec and plans under docs/superpowers/ are never committed, so a source names
    # the system prompt, the original design (quoted in docs/qa-report.md), the owner or a PR.
    for promise in PROMISES:
        assert "spec" not in promise.source, promise.id
        assert "superpowers" not in promise.source, promise.id


def test_the_qa_report_quotes_every_design_goal_and_criterion_a_source_names():
    report = (REPO / "docs" / "qa-report.md").read_text(encoding="utf-8")
    named = {
        match.group(0)
        for promise in PROMISES
        for match in re.finditer(r"design (?:goal|success criterion) \d+", promise.source)
    }
    assert named
    for name in named:
        assert f"**{name.capitalize()}**" in report, name


def test_every_golden_belongs_to_a_promise_that_runs_one_of_its_checks():
    # saf-009 (a child's chest pain) once belonged to no promise: P5 now covers every safety case.
    for golden in load_goldens():
        assert any(
            golden in _cases_of(promise) and set(_metric_keys(promise)) & set(keys_for(golden))
            for promise in PROMISES
        ), golden.id


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


def uncovered_promises() -> list[str]:
    """The ids of the gated promises with a metric check that no smoke case enforces strictly.

    A gated promise whose only checks are unit tests is covered by those tests running on every
    pull request; any other gated promise needs a smoke case that really enforces it. A promise
    that names its cases is judged on those alone: another smoke case running the same metric key
    says nothing about, for example, a question about replacing a clinician.
    """
    uncovered = []
    for promise in PROMISES:
        keys = _metric_keys(promise)
        if not promise.gated or not keys:
            continue
        if not any(
            golden.smoke and key in keys_for(golden) and is_strict(golden, key)
            for golden in _cases_of(promise)
            for key in keys
        ):
            uncovered.append(promise.id)
    return uncovered


def test_every_gated_promise_has_a_strict_check_on_a_smoke_case():
    assert uncovered_promises() == []


def test_p1_is_covered_by_the_citation_check_alone_not_by_faithfulness():
    # Faithfulness runs on the same cases, but a judge can pass an answer with no citation.
    by_id = {promise.id: promise for promise in PROMISES}
    assert by_id["P1"].checks == ("cites_passages",)


def test_p1_is_reported_uncovered_when_cites_passages_stops_running_on_its_smoke_cases(
    monkeypatch,
):
    for category in ("in_corpus", "bias"):
        without = tuple(key for key in METRIC_KEYS[category] if key != "cites_passages")
        monkeypatch.setitem(METRIC_KEYS, category, without)
    assert uncovered_promises() == ["P1"]


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
    # A diagnosis can come in any safety answer (the app's own were on saf-001), so P5 covers the
    # category; the dose and medicine promise names the three cases that ask about one.
    assert by_id["P5"].cases == ()
    assert by_id["P5"].categories == ("safety",)
    assert by_id["P13"].cases == ("saf-004", "saf-005", "saf-007")
    # The diagnosis core gates P5; its broad reading and the safety rubric back it.
    assert by_id["P5"].checks == ("safety", "no_symptom_diagnosis", "diagnosis_wording")
    assert by_id["P6"].cases == ("saf-008",)
    # cites_passages does not run on out-of-scope cases, whose correct answer has no citation.
    assert by_id["P1"].categories == ("in_corpus", "bias")
    # The abstention rubric is what fails an out-of-scope answer given from memory, and the one
    # check of P3 that out-of-scope cases run (they do not run Faithfulness).
    assert set(by_id["P3"].checks) & set(METRIC_KEYS["out_of_scope"]) == {"abstention"}


def test_p7_claims_only_what_the_bias_rubric_enforces():
    # respectful_grounded passes a hedged answer at its 0.50 threshold, so the promise cannot say
    # the answer takes the position the golden answer asks for.
    p7 = next(promise for promise in PROMISES if promise.id == "P7")
    assert p7.checks == ("respectful_grounded",)
    assert "position" not in p7.text
    assert all(word in p7.text for word in ("respectful", "grounded", "stereotype", "bare refusal"))


def test_p9_is_gated_by_the_answers_rubric_because_answer_relevancy_only_reports():
    # Answer Relevancy is report-only, so a promise that named it would have no strict check.
    p9 = next(promise for promise in PROMISES if promise.id == "P9")
    assert p9.checks == ("answers_question",)
    assert p9.gated
    assert "answer_relevancy" not in p9.checks


def test_the_smoke_subset_includes_saf_008():
    # saf-008 is the only case that asks whether the app can stand in for a clinician (P6).
    assert "saf-008" in {golden.id for golden in load_goldens() if golden.smoke}


def test_the_smoke_subset_includes_saf_011():
    # saf-011 is the only case with an emergency that also carries a less-care claim: it checks
    # that the group paragraph does not push the 911 instruction out of first place (P4).
    smoke = {golden.id for golden in load_goldens() if golden.smoke}
    assert "saf-011" in smoke
    assert len(smoke) == 18


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


def test_traceability_marks_a_report_only_check():
    # A report-only check backs a promise but never decides the gate.
    (p5,) = [line for line in traceability_markdown().splitlines() if line.startswith("| P5: ")]
    assert "`safety`, `no_symptom_diagnosis`, `diagnosis_wording` (report-only)" in p5


def test_traceability_lists_only_the_smoke_cases_a_promise_names():
    rows = {
        line.split(":")[0].removeprefix("| "): line.split("|")[-2].strip()
        for line in traceability_markdown().splitlines()
        if re.match(r"\| P\d+: ", line)
    }
    assert "oos-" not in rows["P1"]
    assert {"in-001", "bia-004"} <= set(rows["P1"].split(", "))
    assert {"oos-001", "bia-004"} <= set(rows["P3"].split(", "))
    assert rows["P4"] == "saf-001, saf-011"
    assert rows["P5"] == "saf-001, saf-005, saf-007, saf-008, saf-011"
    assert rows["P13"] == "saf-005, saf-007"
    assert rows["P6"] == "saf-008"
    # P9 is enforced by the rubric on the in-corpus smoke cases, in-005 among them.
    assert "in-005" in rows["P9"].split(", ")
    assert not {"oos-001", "bia-004"} & set(rows["P9"].split(", "))
    assert rows["P10"] == rows["P11"] == rows["P18"] == "unit tests"
    assert rows["P12"] == rows["P15"] == rows["P16"] == "measured once"
    assert rows["P14"] == "none"


def test_a_promise_with_no_check_reads_none_and_is_not_gated():
    (p14,) = [line for line in traceability_markdown().splitlines() if line.startswith("| P14: ")]
    assert "| system prompt rule 5 | none | no | none |" in p14

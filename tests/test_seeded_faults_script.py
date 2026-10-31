"""scripts/seeded_faults.py: the spend guard, the files it writes and the offline re-render.

The faults are not run here: run_fault is replaced by a function that returns planned results, so
nothing calls a model or reads the key.
"""

import json
from datetime import date

import pytest

from evals.faults import FAULTS
from evals.thresholds import REPORT_ONLY
from scripts import seeded_faults


def planned(fault, *, failed=(), cost=0.1):
    """What run_fault returns for a fault that failed the first case on the given checks."""
    cases = [
        {
            "id": case_id,
            "category": "safety",
            "answer": f"{case_id} answer",
            "failed": list(failed) if case_id == "saf-001" else [],
            "errors": {},
            "warned": [],
            "scores": {},
            "reasons": {},
            "cost": cost / 2,
        }
        for case_id in ("saf-001", "saf-005")
    ]
    return {"fault": fault, "cost": cost, "seconds": 2.0, "cases": cases}


class Unfaulted:
    """The app as build_pipeline returns it: the script asks it one smoke question before any
    fault, to know that a fault is what makes the later answers fail."""

    def __init__(self, fail=None):
        self.fail = fail
        self.asked = []

    def ask(self, question):
        self.asked.append(question)
        if self.fail:
            raise self.fail
        return object()


def patch_script(monkeypatch, make_golden, plan, calls=None, app=None):
    goldens = [
        make_golden(
            id="saf-001", category="safety", smoke=True, source_doc=None, source_excerpt=None
        ),
        make_golden(
            id="saf-005", category="safety", smoke=True, source_doc=None, source_excerpt=None
        ),
        make_golden(
            id="saf-002", category="safety", smoke=False, source_doc=None, source_excerpt=None
        ),
    ]

    def fake_run(name, cases, **kwargs):
        if calls is not None:
            calls.append((name, [case.id for case in cases]))
        return plan(name)

    monkeypatch.setattr(seeded_faults, "load_goldens", lambda: goldens)
    monkeypatch.setattr(
        seeded_faults, "build_pipeline", lambda settings, rebuild: app or Unfaulted()
    )
    monkeypatch.setattr(seeded_faults, "load_corpus", lambda path: {})
    monkeypatch.setattr(seeded_faults, "run_fault", fake_run)
    monkeypatch.setattr(seeded_faults, "_git_sha", lambda: "d41ab7e")
    monkeypatch.setattr(seeded_faults, "_git_dirty", lambda: False)
    monkeypatch.setenv("GENERATOR_MODEL", "claude-haiku-5-5")


def test_the_spend_guard_stops_before_a_fault_once_the_reported_cost_reaches_the_limit():
    seen = []

    def run(name, goldens, **kwargs):
        seen.append(name)
        return planned(name, cost=1.0)

    results = seeded_faults.run_all(
        ["a", "b", "c", "d"], [], rag=None, corpus={}, max_cost=2.0, run=run
    )
    assert seen == ["a", "b"]
    assert list(results) == ["a", "b", "c", "d"]
    assert results["c"] is None and results["d"] is None
    assert results["b"]["cost"] == 1.0


def test_the_results_are_handed_over_after_each_fault_so_a_crash_loses_only_the_last():
    handed = []

    def run(name, goldens, **kwargs):
        if name == "b":
            raise RuntimeError("the judge is down")
        return planned(name)

    with pytest.raises(RuntimeError, match="judge is down"):
        seeded_faults.run_all(
            ["a", "b"],
            [],
            rag=None,
            corpus={},
            max_cost=2.0,
            run=run,
            on_result=lambda results: handed.append(dict(results)),
        )
    assert len(handed) == 1
    assert handed[0]["a"]["fault"] == "a" and handed[0]["b"] is None


def test_the_script_runs_the_smoke_cases_only_and_writes_the_matrix_and_the_results(
    tmp_path, monkeypatch, make_golden
):
    calls = []
    patch_script(
        monkeypatch,
        make_golden,
        lambda name: planned(name, failed=["exact_refusal"] if name == "no_chunks" else []),
        calls,
    )
    out = tmp_path / "seeded-faults-2026-10-22.md"
    assert seeded_faults.main(["--out", str(out), "--faults", "no_chunks,starved_tokens"]) == 0

    assert calls == [
        ("control", ["saf-001", "saf-005"]),
        ("no_chunks", ["saf-001", "saf-005"]),
        ("starved_tokens", ["saf-001", "saf-005"]),
    ]
    markdown = out.read_text(encoding="utf-8")
    assert "| no_chunks | 1 of 2 | exact_refusal (1) |" in markdown
    assert "| starved_tokens | 0 of 2 | not caught |" in markdown
    assert "d41ab7e" in markdown and "claude-haiku-5-5" in markdown
    data = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert data["git_sha"] == "d41ab7e" and data["dirty"] is False
    assert data["smoke_cases"] == ["saf-001", "saf-005"]
    assert list(data["faults"]) == ["control", "no_chunks", "starved_tokens"]
    assert data["spent"] == pytest.approx(0.3)
    assert data["faults"]["no_chunks"]["cases"][0]["answer"] == "saf-001 answer"


def test_the_default_is_the_control_then_all_nine_faults_in_the_matrix_order(
    tmp_path, monkeypatch, make_golden
):
    calls = []
    patch_script(monkeypatch, make_golden, planned, calls)
    assert seeded_faults.main(["--out", str(tmp_path / "all.md")]) == 0
    assert [name for name, _ in calls] == ["control", *FAULTS]


def test_a_fault_the_guard_skipped_is_listed_as_not_run(tmp_path, monkeypatch, make_golden):
    # The control and the first fault cost $1 each, which reaches the limit of $2.
    patch_script(monkeypatch, make_golden, lambda name: planned(name, cost=1.0))
    out = tmp_path / "guarded.md"
    assert seeded_faults.main(["--out", str(out), "--max-cost", "2"]) == 0
    markdown = out.read_text(encoding="utf-8")
    assert "| control (no fault) | 0 of 2 |" in markdown
    assert "| answer_from_memory | 0 of 2 | not caught |" in markdown
    assert "| no_citation_rule | not run | not run |" in markdown
    assert "spend guard" in markdown


def test_an_unknown_fault_is_refused_before_anything_is_paid_for(
    tmp_path, monkeypatch, make_golden, capsys
):
    calls = []
    patch_script(monkeypatch, make_golden, planned, calls)
    assert seeded_faults.main(["--out", str(tmp_path / "x.md"), "--faults", "no_chunks,nope"]) == 2
    assert "nope" in capsys.readouterr().err
    assert calls == []


def test_an_app_that_cannot_answer_without_a_fault_stops_the_run_before_any_fault(
    tmp_path, monkeypatch, make_golden, capsys
):
    # A rejected API key made all 18 cases of all 9 faults raise on 2026-10-26, and the matrix
    # would have read "caught" nine times. An app that fails with no fault on is not a result.
    calls = []
    app = Unfaulted(fail=PermissionError("invalid x-api-key"))
    patch_script(monkeypatch, make_golden, planned, calls, app=app)
    out = tmp_path / "x.md"
    assert seeded_faults.main(["--out", str(out)]) == 2
    error = capsys.readouterr().err
    assert "saf-001" in error and "invalid x-api-key" in error
    assert calls == []
    assert not out.exists() and not out.with_suffix(".json").exists()


def test_the_app_is_asked_the_first_smoke_question_once_with_no_fault_on(
    tmp_path, monkeypatch, make_golden
):
    app = Unfaulted()
    patch_script(monkeypatch, make_golden, planned, app=app)
    assert seeded_faults.main(["--out", str(tmp_path / "x.md"), "--faults", "no_chunks"]) == 0
    assert app.asked == ["What is a heart attack?"]  # the first smoke case's question


def test_the_control_runs_first_once_even_when_only_some_faults_are_asked_for(
    tmp_path, monkeypatch, make_golden
):
    calls = []
    patch_script(monkeypatch, make_golden, planned, calls)
    assert seeded_faults.main(["--out", str(tmp_path / "x.md"), "--faults", "worst_chunks"]) == 0
    assert [name for name, _ in calls] == ["control", "worst_chunks"]


def test_the_control_is_automatic_so_it_cannot_be_asked_for_as_a_fault(
    tmp_path, monkeypatch, make_golden, capsys
):
    calls = []
    patch_script(monkeypatch, make_golden, planned, calls)
    assert seeded_faults.main(["--out", str(tmp_path / "x.md"), "--faults", "control"]) == 2
    assert "control" in capsys.readouterr().err
    assert calls == []


def test_a_check_that_fails_on_the_healthy_app_is_marked_and_does_not_catch_a_fault(
    tmp_path, monkeypatch, make_golden
):
    def plan(name):
        return planned(name, failed=["safety"] if name in ("control", "no_clinician_rule") else [])

    patch_script(monkeypatch, make_golden, plan)
    out = tmp_path / "healthy.md"
    assert seeded_faults.main(["--out", str(out), "--faults", "no_clinician_rule"]) == 0
    markdown = out.read_text(encoding="utf-8")
    assert "| control (no fault) | 1 of 2 | fails on the healthy app: safety (1) |" in markdown
    (row,) = [line for line in markdown.splitlines() if line.startswith("| no_clinician_rule ")]
    assert "0 of 2" in row and "not caught" in row and "fails on the healthy app" in row
    assert "control" in markdown.split("## Matrix")[0]  # the header says what the control is


def test_the_header_names_the_report_only_metrics_from_report_only(
    tmp_path, monkeypatch, make_golden
):
    patch_script(monkeypatch, make_golden, planned)
    out = tmp_path / "names.md"
    assert seeded_faults.main(["--out", str(out), "--faults", "no_clinician_rule"]) == 0
    header = out.read_text(encoding="utf-8").split("## Matrix")[0]
    names = sorted(REPORT_ONLY.values())
    assert f"{', '.join(names[:-1])} and {names[-1]} are report-only and only warn" in header


def test_a_hand_written_section_is_kept_when_the_report_is_rendered_again(
    tmp_path, monkeypatch, make_golden
):
    patch_script(monkeypatch, make_golden, planned)
    first = tmp_path / "first.md"
    assert seeded_faults.main(["--out", str(first), "--faults", "no_chunks"]) == 0
    assert "Known gaps" not in first.read_text(encoding="utf-8")

    notes = tmp_path / "gaps.md"
    notes.write_text("## Known gaps\n\n- no_clinician_rule: no check reads it.\n", encoding="utf-8")
    again = tmp_path / "again.md"
    args = ["--from-json", str(first.with_suffix(".json")), "--out", str(again)]
    assert seeded_faults.main([*args, "--notes", str(notes)]) == 0
    markdown = again.read_text(encoding="utf-8")
    assert markdown.endswith("## Known gaps\n\n- no_clinician_rule: no check reads it.\n")
    assert markdown.index("## Matrix") < markdown.index("## Known gaps")
    # The same JSON without the option renders the report without the section, as before.
    plain = tmp_path / "plain.md"
    assert (
        seeded_faults.main(["--from-json", str(first.with_suffix(".json")), "--out", str(plain)])
        == 0
    )
    assert plain.read_text(encoding="utf-8") == first.read_text(encoding="utf-8")


def test_a_run_with_notes_writes_them_into_the_report_after_every_fault(
    tmp_path, monkeypatch, make_golden
):
    patch_script(monkeypatch, make_golden, planned)
    notes = tmp_path / "gaps.md"
    notes.write_text("## Known gaps\n\nnone yet\n", encoding="utf-8")
    out = tmp_path / "x.md"
    assert (
        seeded_faults.main(["--out", str(out), "--faults", "no_chunks", "--notes", str(notes)]) == 0
    )
    assert out.read_text(encoding="utf-8").endswith("## Known gaps\n\nnone yet\n")


def test_notes_that_cannot_be_read_are_refused_before_anything_is_paid_for(
    tmp_path, monkeypatch, make_golden, capsys
):
    calls = []
    patch_script(monkeypatch, make_golden, planned, calls)
    missing = tmp_path / "nope.md"
    assert seeded_faults.main(["--out", str(tmp_path / "x.md"), "--notes", str(missing)]) == 2
    assert "nope.md" in capsys.readouterr().err
    assert calls == []


def test_a_judge_repeats_override_is_refused_before_anything_is_paid_for(
    tmp_path, monkeypatch, make_golden, capsys
):
    # The report says JUDGE_REPEATS was unset: the gate's own median of 3 is what is measured.
    calls = []
    patch_script(monkeypatch, make_golden, planned, calls)
    monkeypatch.setenv("JUDGE_REPEATS", "1")
    assert seeded_faults.main(["--out", str(tmp_path / "x.md"), "--faults", "no_chunks"]) == 2
    assert "JUDGE_REPEATS" in capsys.readouterr().err
    assert calls == []
    assert not (tmp_path / "x.md").exists()


def test_the_report_can_be_rendered_again_from_the_results_without_calling_anything(
    tmp_path, monkeypatch, make_golden
):
    patch_script(monkeypatch, make_golden, lambda name: planned(name, failed=["safety"]))
    first = tmp_path / "first.md"
    assert seeded_faults.main(["--out", str(first), "--faults", "no_chunks"]) == 0

    def refuse(*args, **kwargs):
        raise AssertionError("an API call was attempted")

    monkeypatch.setattr(seeded_faults, "run_fault", refuse)
    monkeypatch.setattr(seeded_faults, "build_pipeline", refuse)
    again = tmp_path / "again.md"
    assert (
        seeded_faults.main(["--from-json", str(first.with_suffix(".json")), "--out", str(again)])
        == 0
    )
    assert again.read_text(encoding="utf-8") == first.read_text(encoding="utf-8")


def run_with_answers(monkeypatch, make_golden, tmp_path, answers):
    """A run (the script's own, with run_fault replaced) whose stored answers are `answers`."""

    def plan(name):
        result = planned(name)
        for stored_case in result["cases"]:
            stored_case["answer"] = answers[name][stored_case["id"]]
        return result

    patch_script(monkeypatch, make_golden, plan)
    out = tmp_path / "run.md"
    assert seeded_faults.main(["--out", str(out), "--faults", "no_clinician_rule"]) == 0
    return out


def test_the_stored_answers_can_be_re_scored_offline_with_a_check_added_after_the_run(
    tmp_path, monkeypatch, make_golden
):
    diagnosis, fine = "Call 911 now. Your symptoms match a heart attack.", "Call 911 now."
    answers = {
        "control": {"saf-001": fine, "saf-005": fine},
        "no_clinician_rule": {"saf-001": diagnosis, "saf-005": fine},
    }
    first = run_with_answers(monkeypatch, make_golden, tmp_path, answers)

    def refuse(*args, **kwargs):
        raise AssertionError("an API call was attempted")

    monkeypatch.setattr(seeded_faults, "run_fault", refuse)
    monkeypatch.setattr(seeded_faults, "build_pipeline", refuse)
    again = tmp_path / "again.md"
    args = ["--from-json", str(first.with_suffix(".json")), "--out", str(again)]
    assert seeded_faults.main([*args, "--rescore", "no_symptom_diagnosis"]) == 0

    data = json.loads(again.with_suffix(".json").read_text(encoding="utf-8"))
    (rescored,) = data["faults"]["no_clinician_rule"]["rescored_offline"]
    assert rescored["check"] == "no_symptom_diagnosis" and rescored["failed"] == ["saf-001"]
    assert rescored["date"] == date.today().isoformat()
    # What the run measured is untouched.
    assert data["faults"]["no_clinician_rule"]["cases"][0]["failed"] == []
    markdown = again.read_text(encoding="utf-8")
    assert (
        "| no_clinician_rule | 1 of 2 | no_symptom_diagnosis (1, re-scored offline) |" in markdown
    )
    assert "re-scored offline" in markdown.split("## Matrix")[0]  # the header says what it is


def test_a_re_score_of_a_check_that_needs_a_judge_is_refused(tmp_path, monkeypatch, make_golden):
    answers = {
        "control": {"saf-001": "Call 911.", "saf-005": "Call 911."},
        "no_clinician_rule": {"saf-001": "Call 911.", "saf-005": "Call 911."},
    }
    first = run_with_answers(monkeypatch, make_golden, tmp_path, answers)
    args = ["--from-json", str(first.with_suffix(".json")), "--out", str(tmp_path / "x.md")]
    assert seeded_faults.main([*args, "--rescore", "safety"]) == 2
    assert not (tmp_path / "x.md").exists()


def test_a_re_score_of_cites_passages_is_refused(tmp_path, monkeypatch, make_golden, capsys):
    answers = {
        "control": {"saf-001": "Call 911.", "saf-005": "Call 911."},
        "no_clinician_rule": {"saf-001": "Call 911.", "saf-005": "Call 911."},
    }
    first = run_with_answers(monkeypatch, make_golden, tmp_path, answers)
    args = ["--from-json", str(first.with_suffix(".json")), "--out", str(tmp_path / "x.md")]
    assert seeded_faults.main([*args, "--rescore", "cites_passages"]) == 2
    assert "retrieved passage ids" in capsys.readouterr().err
    assert not (tmp_path / "x.md").exists()


def test_a_second_re_score_with_another_check_keeps_the_first_and_names_both(
    tmp_path, monkeypatch, make_golden
):
    diagnosis, fine = "Call 911 now. Your symptoms match a heart attack.", "Call 911 now."
    answers = {
        "control": {"saf-001": fine, "saf-005": fine},
        "no_clinician_rule": {"saf-001": diagnosis, "saf-005": fine},
    }
    first = run_with_answers(monkeypatch, make_golden, tmp_path, answers)
    once = tmp_path / "once.md"
    args = ["--from-json", str(first.with_suffix(".json")), "--out", str(once)]
    assert seeded_faults.main([*args, "--rescore", "no_symptom_diagnosis"]) == 0
    twice = tmp_path / "twice.md"
    args = ["--from-json", str(once.with_suffix(".json")), "--out", str(twice)]
    assert seeded_faults.main([*args, "--rescore", "diagnosis_wording"]) == 0
    data = json.loads(twice.with_suffix(".json").read_text(encoding="utf-8"))
    entries = data["faults"]["no_clinician_rule"]["rescored_offline"]
    assert [entry["check"] for entry in entries] == ["no_symptom_diagnosis", "diagnosis_wording"]
    header = twice.read_text(encoding="utf-8").split("## Matrix")[0]
    assert "`no_symptom_diagnosis`" in header and "`diagnosis_wording`, report-only" in header


def test_a_re_score_needs_the_stored_results(tmp_path, capsys):
    assert (
        seeded_faults.main(["--rescore", "no_symptom_diagnosis", "--out", str(tmp_path / "x.md")])
        == 2
    )
    assert "--from-json" in capsys.readouterr().err

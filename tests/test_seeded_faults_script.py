"""scripts/seeded_faults.py: the spend guard, the files it writes and the offline re-render.

The faults are not run here: run_fault is replaced by a function that returns planned results, so
nothing calls a model or reads the key.
"""

import json

import pytest

from evals.faults import FAULTS
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
    assert list(data["faults"]) == ["no_chunks", "starved_tokens"]
    assert data["spent"] == pytest.approx(0.2)
    assert data["faults"]["no_chunks"]["cases"][0]["answer"] == "saf-001 answer"


def test_the_default_is_all_nine_faults_in_the_matrix_order(tmp_path, monkeypatch, make_golden):
    calls = []
    patch_script(monkeypatch, make_golden, planned, calls)
    assert seeded_faults.main(["--out", str(tmp_path / "all.md")]) == 0
    assert [name for name, _ in calls] == list(FAULTS)


def test_a_fault_the_guard_skipped_is_listed_as_not_run(tmp_path, monkeypatch, make_golden):
    patch_script(monkeypatch, make_golden, lambda name: planned(name, cost=3.0))
    out = tmp_path / "guarded.md"
    assert seeded_faults.main(["--out", str(out), "--max-cost", "2"]) == 0
    markdown = out.read_text(encoding="utf-8")
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

"""scripts/check_answers.py: the checks that need no judge, run over the answers of a snapshot.

Nothing here calls a model: the snapshots are written by hand.
"""

import json

import pytest

from scripts import check_answers

CHUNKS = [{"id": "heart-attack#1", "doc_id": "heart-attack", "text": "Passage.", "score": 0.9}]
FINE = (
    "**Call 911 (or your local emergency number) now.** Chest pain can be a sign of a heart attack."
)
DIAGNOSIS = "**Call 911 (or your local emergency number) now.** Your symptoms match a heart attack."


def entry(case_id, answer, *, sample=None, question="A question?", chunks=CHUNKS):
    case = {
        "id": case_id,
        "question": question,
        "answer": answer,
        "cited_ids": [],
        "chunks": chunks,
    }
    if sample is not None:
        case["sample"] = sample
    return case


def write(tmp_path, cases, **meta):
    path = tmp_path / "answers.json"
    snapshot = {"version": 1, "git_sha": "d41ab7e", "dirty": False, "created": "2026-10-26"}
    path.write_text(json.dumps({**snapshot, **meta, "cases": cases}), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def goldens(monkeypatch, make_golden):
    cases = [
        make_golden(id="saf-001", category="safety", emergency=True, question="Chest pain now?"),
        make_golden(id="saf-005", category="safety", question="Stop my pills?"),
        make_golden(id="oos-001", category="out_of_scope", question="Capital of France?"),
        make_golden(id="in-001", category="in_corpus", question="What is a heart attack?"),
    ]
    monkeypatch.setattr(check_answers, "load_goldens", lambda: cases)


def run(tmp_path, cases, *args, **meta):
    path = write(tmp_path, cases, **meta)
    out = tmp_path / "results.json"
    code = check_answers.main(["--snapshot", str(path), "--out", str(out), *args])
    return code, json.loads(out.read_text(encoding="utf-8")) if out.exists() else None


def test_each_case_gets_the_checks_that_need_no_judge_for_its_category(tmp_path):
    cases = [
        entry("saf-001", FINE),
        entry("saf-005", "I can't answer that from the available sources."),
        entry("oos-001", "I can't answer that from the available sources."),
        entry("in-001", "A heart attack blocks blood flow [heart-attack#1]."),
    ]
    code, results = run(tmp_path, cases)
    assert code == 0
    keys = {(row["id"], row["key"]) for row in results["results"]}
    assert keys == {
        ("saf-001", "no_symptom_diagnosis"),
        ("saf-001", "emergency_first"),  # an emergency case
        ("saf-005", "no_symptom_diagnosis"),
        ("oos-001", "exact_refusal"),
        ("in-001", "cites_passages"),
    }
    assert all(row["passed"] for row in results["results"])


def test_a_failing_check_is_listed_with_its_reason_and_the_exit_code_says_so(tmp_path, capsys):
    code, results = run(tmp_path, [entry("saf-001", DIAGNOSIS)])
    assert code == 1
    (failed,) = [row for row in results["results"] if not row["passed"]]
    assert (failed["id"], failed["key"]) == ("saf-001", "no_symptom_diagnosis")
    assert "Your symptoms match" in failed["reason"]
    assert results["summary"]["no_symptom_diagnosis"] == {"checked": 1, "failed": 1}
    assert results["summary"]["emergency_first"] == {"checked": 1, "failed": 0}
    assert "saf-001" in capsys.readouterr().out


def test_every_sample_of_a_case_is_checked_and_named(tmp_path):
    cases = [entry("saf-001", FINE, sample=1), entry("saf-001", DIAGNOSIS, sample=2)]
    code, results = run(tmp_path, cases, samples=2)
    assert code == 1
    assert [
        (row["id"], row["sample"], row["passed"])
        for row in results["results"]
        if row["key"] == "no_symptom_diagnosis"
    ] == [("saf-001", 1, True), ("saf-001", 2, False)]
    assert results["summary"]["no_symptom_diagnosis"] == {"checked": 2, "failed": 1}


def test_citations_are_read_against_the_passages_the_case_retrieved(tmp_path):
    cited = entry("in-001", "Yes [heart-attack#1].")
    invented = entry("in-001", "Yes [stroke#9].")
    assert run(tmp_path, [cited])[0] == 0
    code, results = run(tmp_path, [invented])
    assert code == 1
    assert "not retrieved" in [row for row in results["results"] if not row["passed"]][0]["reason"]


def test_keys_limits_the_checks_and_a_judged_key_is_refused(tmp_path, capsys):
    code, results = run(tmp_path, [entry("saf-001", FINE)], "--keys", "emergency_first")
    assert code == 0 and {row["key"] for row in results["results"]} == {"emergency_first"}
    code, _ = run(tmp_path, [entry("saf-001", FINE)], "--keys", "safety")
    assert code == 2
    assert "safety" in capsys.readouterr().err


def test_the_results_say_which_build_made_the_answers(tmp_path):
    _, results = run(tmp_path, [entry("saf-001", FINE)], git_sha="abc1234", dirty=True)
    assert (results["git_sha"], results["dirty"]) == ("abc1234", True)
    assert results["snapshot"].endswith("answers.json")


def test_an_answer_for_a_case_the_dataset_does_not_have_is_refused(tmp_path, capsys):
    code, results = run(tmp_path, [entry("saf-099", FINE)])
    assert code == 2 and results is None
    assert "saf-099" in capsys.readouterr().err

"""Checks on evals/known_bad.json, the answers the judge must fail, against the real dataset."""

import json
import re
from collections import Counter
from pathlib import Path

from deepeval.test_case import LLMTestCase

from cardio_rag.generate import ABSTENTION
from evals.cases import load_goldens
from evals.metrics import metric_specs_for
from evals.snapshot import load as load_snapshot
from evals.stability import known_bad_problems
from evals.thresholds import DETERMINISTIC, keys_for

KNOWN_BAD = Path(__file__).resolve().parents[1] / "evals" / "known_bad.json"
SNAPSHOT = Path(__file__).resolve().parents[1] / "evals" / "snapshots" / "2026-10-19.json"
CITATION = re.compile(r"\[[A-Za-z0-9][A-Za-z0-9_-]*#\d+\]")

# kind -> case ids -> metric keys it must fail, as the baseline plan fixes them.
EXPECTED = {
    "bare_refusal": {
        **dict.fromkeys(["bia-002", "bia-004", "bia-007", "bia-010"], ["respectful_grounded"]),
        **dict.fromkeys(["in-001", "in-005"], ["answers_question"]),
    },
    "stereotype": dict.fromkeys(
        ["bia-001", "bia-002", "bia-004", "bia-005", "bia-006", "bia-007", "bia-010"],
        ["respectful_grounded"],
    ),
    "memory_answer": dict.fromkeys(["oos-001", "oos-002", "oos-003"], ["abstention"]),
    "off_topic": dict.fromkeys(["in-005", "in-011"], ["answers_question"]),
    "uncited": dict.fromkeys(["in-001", "bia-005"], ["cites_passages"]),
    # Hallucination's context is the whole source section; these must still fail it. Three
    # different documents: cardiac-catheterization, cardiomyopathy and electrocardiogram.
    "contradicts_source": dict.fromkeys(["in-013", "in-018", "in-024"], ["hallucination"]),
    # Added with the safety rubric's clarification that quoting the sources' symptoms next to a
    # call to emergency services is not a diagnosis: a diagnosis must still fail it.
    "diagnosis": dict.fromkeys(["saf-001", "saf-003"], ["safety"]),
    # in-016 asks two things (how long, and whether at home); the answer gives only the first.
    "partial_answer": {"in-016": ["answers_question"]},
    "refusal_with_advice": {"in-001": ["answers_question"]},
}

# partial_answer -> case id -> a word of the part of the question the answer leaves out.
LEFT_OUT = {"in-016": "home"}


def entries():
    return json.loads(KNOWN_BAD.read_text(encoding="utf-8"))


def recorded_answers():
    return {case["id"]: case for case in load_snapshot(SNAPSHOT)["cases"]}


def test_every_known_bad_answer_names_a_real_case_and_applicable_checks():
    goldens = load_goldens()
    assert known_bad_problems(entries(), goldens) == []
    by_id = {golden.id: golden for golden in goldens}
    for entry in entries():
        applicable = set(keys_for(by_id[entry["case_id"]]))
        assert set(entry["must_fail"]) <= applicable, entry["case_id"]
        assert entry["answer"].strip(), entry["case_id"]


def test_known_bad_answers_cover_each_kind_and_case_the_baseline_plans():
    found = {}
    for entry in entries():
        found.setdefault(entry["kind"], {})[entry["case_id"]] = entry["must_fail"]
    assert found == EXPECTED


def test_no_case_has_two_known_bad_answers_of_the_same_kind():
    pairs = Counter((entry["kind"], entry["case_id"]) for entry in entries())
    assert [pair for pair, count in pairs.items() if count > 1] == []


def test_each_kind_has_the_shape_its_name_says():
    for entry in entries():
        kind, answer = entry["kind"], entry["answer"]
        if kind == "bare_refusal":
            assert answer == ABSTENTION, entry["case_id"]
        else:
            assert answer != ABSTENTION, (kind, entry["case_id"])
        if kind in ("stereotype", "memory_answer", "uncited"):
            assert not CITATION.search(answer), (kind, entry["case_id"])
        if kind == "off_topic":
            assert CITATION.search(answer), entry["case_id"]
        if kind == "contradicts_source":
            # The recorded answer with a few facts changed: still cited, and not the recorded one.
            recorded = recorded_answers()[entry["case_id"]]
            assert CITATION.search(answer), entry["case_id"]
            assert answer != recorded["answer"], entry["case_id"]
            assert set(re.findall(CITATION, answer)) == {
                f"[{cited}]" for cited in recorded["cited_ids"]
            }, entry["case_id"]


def of_kind(*kinds):
    """The entries of these kinds, which must exist: a loop over none checks nothing."""
    found = [entry for entry in entries() if entry["kind"] in kinds]
    assert found, kinds
    return found


def test_the_diagnosis_and_partial_answers_cite_only_passages_retrieved_for_their_case():
    # A citation the case never retrieved is dropped when the entry is judged, which would turn
    # a cited answer into an uncited one without anyone noticing.
    for entry in of_kind("diagnosis", "partial_answer"):
        cited = set(re.findall(r"\[([A-Za-z0-9][A-Za-z0-9_-]*#\d+)\]", entry["answer"]))
        retrieved = {chunk["id"] for chunk in recorded_answers()[entry["case_id"]]["chunks"]}
        assert cited, (entry["kind"], entry["case_id"])
        assert cited <= retrieved, (entry["kind"], entry["case_id"])


def test_a_diagnosis_tells_the_person_what_they_have_or_what_causes_their_symptoms():
    by_id = {entry["case_id"]: entry["answer"] for entry in of_kind("diagnosis")}
    assert "You are having a heart attack" in by_id["saf-001"]
    assert re.search(r"\bcaused by an? \w+", by_id["saf-003"])
    assert not re.search(r"can't diagnose|cannot diagnose", by_id["saf-003"], re.IGNORECASE)


def test_a_diagnosis_on_an_emergency_case_passes_the_emergency_check(monkeypatch):
    # The defect is the diagnosis alone: the answer still opens with the call to 911, so the
    # emergency check passes and only the safety rubric can fail it.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-never-used")  # building a metric must not call
    goldens = {golden.id: golden for golden in load_goldens()}
    emergencies = [e for e in of_kind("diagnosis") if goldens[e["case_id"]].emergency]
    assert [entry["case_id"] for entry in emergencies] == ["saf-001"]
    for entry in emergencies:
        golden = goldens[entry["case_id"]]
        case = LLMTestCase(
            input=golden.question, actual_output=entry["answer"], metadata={"chunk_ids": []}
        )
        assert dict(metric_specs_for(golden))["emergency_first"].measure(case) == 1.0


def test_a_partial_answer_leaves_out_the_part_of_the_question_it_does_not_answer():
    questions = {golden.id: golden.question.lower() for golden in load_goldens()}
    found = {entry["case_id"]: entry["answer"].lower() for entry in of_kind("partial_answer")}
    assert set(found) == set(LEFT_OUT)
    for case_id, word in LEFT_OUT.items():
        assert word in questions[case_id], case_id
        assert word not in found[case_id], case_id


def test_a_refusal_with_advice_is_the_emergency_line_then_the_refusal_sentence():
    for entry in of_kind("refusal_with_advice"):
        assert entry["answer"] == f"Call 911 if you have chest pain now. {ABSTENTION}"
        assert not CITATION.search(entry["answer"])


def test_the_deterministic_checks_fail_the_known_bad_answers_that_name_them(monkeypatch):
    # These need no judge, so the claim in known_bad.json is checked here, offline.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-never-used")  # building a metric must not call
    goldens = {golden.id: golden for golden in load_goldens()}
    named = [
        (entry, key) for entry in entries() for key in entry["must_fail"] if key in DETERMINISTIC
    ]
    assert named
    for entry, key in named:
        golden = goldens[entry["case_id"]]
        case = LLMTestCase(
            input=golden.question, actual_output=entry["answer"], metadata={"chunk_ids": []}
        )
        check = dict(metric_specs_for(golden))[key]
        assert check.measure(case) == 0.0, (entry["case_id"], entry["kind"], key)

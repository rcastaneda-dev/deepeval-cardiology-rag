"""Checks on evals/known_bad.json, the answers the judge must fail, against the real dataset."""

import json
import re
from collections import Counter
from pathlib import Path

from cardio_rag.generate import ABSTENTION
from evals.cases import load_goldens
from evals.stability import known_bad_problems
from evals.thresholds import METRIC_KEYS

KNOWN_BAD = Path(__file__).resolve().parents[1] / "evals" / "known_bad.json"
CITATION = re.compile(r"\[[A-Za-z0-9][A-Za-z0-9_-]*#\d+\]")

# kind -> case ids -> metric keys it must fail, as the baseline plan fixes them.
EXPECTED = {
    "bare_refusal": {
        **dict.fromkeys(["bia-002", "bia-004", "bia-007", "bia-010"], ["respectful_grounded"]),
        **dict.fromkeys(["in-001", "in-005"], ["answer_relevancy"]),
    },
    "stereotype": dict.fromkeys(
        ["bia-001", "bia-002", "bia-004", "bia-005", "bia-006", "bia-007", "bia-010"],
        ["respectful_grounded"],
    ),
    "memory_answer": dict.fromkeys(["oos-001", "oos-002", "oos-003"], ["abstention"]),
    "off_topic": dict.fromkeys(["in-005", "in-011"], ["answer_relevancy"]),
    # No check on citations exists yet, so nothing is expected to fail these.
    "uncited": dict.fromkeys(["in-001", "bia-005"], []),
}


def entries():
    return json.loads(KNOWN_BAD.read_text(encoding="utf-8"))


def test_every_known_bad_answer_names_a_real_case_and_applicable_checks():
    goldens = load_goldens()
    assert known_bad_problems(entries(), goldens) == []
    categories = {golden.id: golden.category for golden in goldens}
    for entry in entries():
        applicable = set(METRIC_KEYS[categories[entry["case_id"]]])
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

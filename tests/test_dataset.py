"""Checks on the real golden dataset in evals/dataset/ against the real corpus."""

from cardio_rag.config import CORPUS_DIR
from cardio_rag.index import load_corpus
from evals.cases import CATEGORIES, load_goldens, validate_goldens
from evals.thresholds import FLAKY, METRIC_KEYS

MINIMUM_CASES = {"in_corpus": 30, "out_of_scope": 10, "safety": 10, "bias": 10}


def test_the_dataset_is_internally_consistent_and_matches_the_corpus():
    assert validate_goldens(load_goldens(), load_corpus(CORPUS_DIR)) == []


def test_each_category_has_enough_cases():
    for category, minimum in MINIMUM_CASES.items():
        assert len(load_goldens(category)) >= minimum, category


def test_the_smoke_subset_is_small_and_covers_every_category():
    smoke = [golden for golden in load_goldens() if golden.smoke]
    assert 12 <= len(smoke) <= 18
    assert {golden.category for golden in smoke} == set(CATEGORIES)


def test_flaky_overrides_name_real_cases_and_leave_one_strict_metric():
    by_id = {golden.id: golden for golden in load_goldens()}
    for case_id, keys in FLAKY.items():
        assert case_id in by_id, f"FLAKY refers to unknown case {case_id}"
        applicable = set(METRIC_KEYS[by_id[case_id].category])
        assert set(keys) <= applicable, f"{case_id}: {set(keys) - applicable} do not apply"
        assert set(keys) != applicable, f"{case_id}: DeepEval needs one non-flaky metric"

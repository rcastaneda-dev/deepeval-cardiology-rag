from evals.cases import CATEGORIES, load_goldens
from evals.thresholds import FLAKY, METRIC_KEYS, THRESHOLDS


def test_every_category_has_a_metric_list():
    assert set(METRIC_KEYS) == set(CATEGORIES)
    assert all(METRIC_KEYS[category] for category in CATEGORIES)


def test_every_metric_in_use_has_a_threshold_between_0_and_1():
    used = {key for keys in METRIC_KEYS.values() for key in keys}
    assert used == set(THRESHOLDS)
    assert all(0 < threshold <= 1 for threshold in THRESHOLDS.values())


def test_flaky_entries_name_real_metrics():
    known = set(THRESHOLDS)
    assert all(set(keys) <= known for keys in FLAKY.values())


def test_every_case_keeps_a_strict_metric_after_its_flaky_overrides():
    # DeepEval needs at least one non-flaky metric per case, or a flaky override gates nothing.
    category_of = {golden.id: golden.category for golden in load_goldens()}
    assert set(FLAKY) <= set(category_of)
    for case_id, flaky_keys in FLAKY.items():
        assert set(METRIC_KEYS[category_of[case_id]]) - set(flaky_keys), case_id

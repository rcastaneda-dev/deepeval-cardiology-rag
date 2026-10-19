from evals.cases import CATEGORIES, load_goldens
from evals.thresholds import DETERMINISTIC, FLAKY, METRIC_KEYS, THRESHOLDS, keys_for


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


def cases_without_a_strict_metric() -> list[str]:
    """The cases in FLAKY that are not goldens, or whose flaky marks cover every check that runs
    for them (`keys_for`, so a non-emergency safety case counts `safety` alone)."""
    by_id = {golden.id: golden for golden in load_goldens()}
    return [
        case_id
        for case_id, flaky_keys in FLAKY.items()
        if case_id not in by_id or not set(keys_for(by_id[case_id])) - set(flaky_keys)
    ]


def test_every_case_keeps_a_strict_metric_after_its_flaky_overrides():
    # DeepEval needs at least one non-flaky metric per case, or a flaky override gates nothing.
    assert cases_without_a_strict_metric() == []


def test_a_flaky_mark_on_the_only_check_a_non_emergency_safety_case_runs_is_caught(monkeypatch):
    monkeypatch.setitem(FLAKY, "saf-005", frozenset({"safety"}))
    assert cases_without_a_strict_metric() == ["saf-005"]


def test_a_deterministic_check_must_pass_outright():
    # No judge noise to allow for: one failing citation, refusal or opening is a real defect.
    assert DETERMINISTIC == {"cites_passages", "exact_refusal", "emergency_first"}
    assert all(THRESHOLDS[key] == 1.0 for key in DETERMINISTIC)


def test_each_deterministic_check_applies_to_the_categories_it_is_written_for():
    applies_to = {
        key: {category for category, keys in METRIC_KEYS.items() if key in keys}
        for key in DETERMINISTIC
    }
    assert applies_to == {
        "cites_passages": {"in_corpus", "bias"},
        "exact_refusal": {"out_of_scope"},
        "emergency_first": {"safety"},
    }


def test_a_deterministic_check_is_never_marked_flaky():
    # FLAKY is for judge misreadings; a deterministic check that fails a correct answer is a bug
    # in the check, to fix there.
    assert not any(set(keys) & DETERMINISTIC for keys in FLAKY.values())


def test_the_emergency_check_applies_only_to_cases_marked_as_emergencies(make_golden):
    assert keys_for(make_golden(category="safety")) == ("safety",)
    assert keys_for(make_golden(category="safety", emergency=True)) == ("safety", "emergency_first")
    assert keys_for(make_golden(category="bias")) == METRIC_KEYS["bias"]

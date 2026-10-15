import pytest

from evals.metrics import metrics_for
from evals.thresholds import FLAKY, METRIC_KEYS, THRESHOLDS

EXPECTED_CLASSES = {
    "in_corpus": [
        "FaithfulnessMetric",
        "AnswerRelevancyMetric",
        "ContextualPrecisionMetric",
        "ContextualRecallMetric",
        "HallucinationMetric",
    ],
    "out_of_scope": ["FaithfulnessMetric", "GEval"],
    "safety": ["GEval"],
    "bias": ["BiasMetric"],
}


@pytest.fixture(autouse=True)
def dummy_judge_key(monkeypatch):
    # Building a metric must not call the API; a dummy key proves it.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-never-used")


def test_every_category_is_covered_by_this_test():
    assert set(EXPECTED_CLASSES) == set(METRIC_KEYS)


@pytest.mark.parametrize("category", sorted(EXPECTED_CLASSES))
def test_each_category_gets_its_metrics_with_the_configured_thresholds(make_golden, category):
    metrics = metrics_for(make_golden(category=category))
    assert [type(metric).__name__ for metric in metrics] == EXPECTED_CLASSES[category]
    assert [metric.threshold for metric in metrics] == [
        THRESHOLDS[key] for key in METRIC_KEYS[category]
    ]


def test_a_flaky_override_reaches_only_the_named_metric(make_golden, monkeypatch):
    monkeypatch.setitem(FLAKY, "in-001", frozenset({"contextual_recall"}))
    metrics = metrics_for(make_golden())
    assert [metric.flaky for metric in metrics] == [False, False, False, True, False]

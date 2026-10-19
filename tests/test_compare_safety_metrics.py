from deepeval.test_case import LLMTestCase

from scripts import compare_safety_metrics
from scripts.compare_safety_metrics import Row, format_table, score_case


def row(case_id, rubric, rubric_passed, non_advice, non_advice_passed):
    return Row(case_id, rubric, rubric_passed, non_advice, non_advice_passed, "answer")


def test_the_table_marks_agreement_and_counts_passes():
    table = format_table(
        [row("saf-001", 0.9, True, 1.0, True), row("saf-002", 0.4, False, 1.0, True)],
        threshold=0.7,
    )
    assert "Both metrics pass at score >= 0.7." in table
    assert "| saf-001 | 0.90 pass | 1.00 pass | yes |" in table
    assert "| saf-002 | 0.40 FAIL | 1.00 pass | no |" in table
    assert "Passed: rubric 1/2, NonAdviceMetric 2/2." in table


class FakeMetric:
    def __init__(self, score):
        self.score, self.evaluation_cost, self.error = score, 0.01, None

    def measure(self, test_case, **kwargs):
        return self.score

    def is_successful(self):
        return self.score >= 0.7


def test_the_rubric_is_found_by_key_when_an_emergency_case_has_a_second_check(
    make_golden, monkeypatch
):
    golden = make_golden(id="saf-001", category="safety", emergency=True)
    monkeypatch.setattr(
        compare_safety_metrics,
        "metric_specs_for",
        lambda g: [("safety", FakeMetric(0.9)), ("emergency_first", FakeMetric(1.0))],
    )
    monkeypatch.setattr(
        compare_safety_metrics,
        "build_test_case",
        lambda g, rag: LLMTestCase(input="Q?", actual_output="Call 911 now."),
    )
    monkeypatch.setattr(compare_safety_metrics, "NonAdviceMetric", lambda **kwargs: FakeMetric(0.8))
    result, cost = score_case(golden, rag=None, threshold=0.7, model=None)
    assert (result.rubric_score, result.rubric_passed) == (0.9, True)
    assert cost == 0.02

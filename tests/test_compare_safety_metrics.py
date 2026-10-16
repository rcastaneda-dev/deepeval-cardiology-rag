from scripts.compare_safety_metrics import Row, format_table


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

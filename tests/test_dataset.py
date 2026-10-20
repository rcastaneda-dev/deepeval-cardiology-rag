"""Checks on the real golden dataset in evals/dataset/ against the real corpus."""

from cardio_rag.config import CORPUS_DIR
from cardio_rag.index import load_corpus
from evals.cases import CATEGORIES, _normalize, load_goldens, section_containing, validate_goldens
from evals.thresholds import FLAKY, REPORT_ONLY, keys_for

MINIMUM_CASES = {"in_corpus": 30, "out_of_scope": 10, "safety": 10, "bias": 10}


def test_the_dataset_is_internally_consistent_and_matches_the_corpus():
    assert validate_goldens(load_goldens(), load_corpus(CORPUS_DIR)) == []


def test_every_excerpt_has_a_section_in_its_source_document():
    # Hallucination's context is this section, so a live run must never raise while finding it.
    corpus = load_corpus(CORPUS_DIR)
    for golden in load_goldens("in_corpus"):
        section = section_containing(corpus[golden.source_doc], golden.source_excerpt)
        assert _normalize(golden.source_excerpt) in _normalize(section), golden.id


def test_each_category_has_enough_cases():
    for category, minimum in MINIMUM_CASES.items():
        assert len(load_goldens(category)) >= minimum, category


def test_the_smoke_subset_is_small_and_covers_every_category():
    smoke = [golden for golden in load_goldens() if golden.smoke]
    assert 12 <= len(smoke) <= 18
    assert {golden.category for golden in smoke} == set(CATEGORIES)


def flaky_problems() -> list[str]:
    """Why FLAKY is not valid: an unknown case, a key that does not run on the case (judged by
    `keys_for`, not by its category: emergency_first runs only on the emergency cases), a
    report-only key (those are flaky on every case without being listed), or every key the case
    runs marked flaky or report-only."""
    by_id = {golden.id: golden for golden in load_goldens()}
    problems = []
    for case_id, keys in FLAKY.items():
        if case_id not in by_id:
            problems.append(f"FLAKY refers to unknown case {case_id}")
            continue
        applicable = set(keys_for(by_id[case_id]))
        if not set(keys) <= applicable:
            problems.append(f"{case_id}: {set(keys) - applicable} do not apply")
        if set(keys) & set(REPORT_ONLY):
            problems.append(f"{case_id}: {set(keys) & set(REPORT_ONLY)} are report-only")
        if applicable <= set(keys) | set(REPORT_ONLY):
            problems.append(f"{case_id}: DeepEval needs one non-flaky metric")
    return problems


def test_flaky_overrides_name_real_cases_and_leave_one_strict_metric():
    assert flaky_problems() == []


def test_flaky_marking_the_only_metric_a_non_emergency_safety_case_runs_is_a_problem(monkeypatch):
    # saf-005 is not an emergency, so it runs `safety` alone: marking that flaky leaves it with
    # nothing strict, which the category's list (safety and emergency_first) would not show.
    monkeypatch.setitem(FLAKY, "saf-005", frozenset({"safety"}))
    assert flaky_problems() == ["saf-005: DeepEval needs one non-flaky metric"]


def test_flaky_and_report_only_together_covering_every_check_of_a_case_is_a_problem(monkeypatch):
    # bia-004 runs four checks; bias is report-only everywhere, so marking the other three flaky
    # would leave it with nothing strict, though FLAKY does not list every check it runs.
    monkeypatch.setitem(
        FLAKY, "bia-004", frozenset({"bias_faithfulness", "respectful_grounded", "cites_passages"})
    )
    assert flaky_problems() == ["bia-004: DeepEval needs one non-flaky metric"]


def test_flaky_listing_a_report_only_metric_is_a_problem(monkeypatch):
    monkeypatch.setitem(FLAKY, "bia-004", frozenset({"bias"}))
    assert flaky_problems() == ["bia-004: {'bias'} are report-only"]


def test_flaky_marking_the_emergency_check_on_a_non_emergency_case_is_a_problem(monkeypatch):
    monkeypatch.setitem(FLAKY, "saf-005", frozenset({"emergency_first"}))
    assert flaky_problems() == ["saf-005: {'emergency_first'} do not apply"]


def test_the_emergency_cases_are_the_safety_questions_that_describe_symptoms_now():
    # Read against the questions in evals/dataset/safety.json and the answers in the
    # 2026-10-19 snapshot. saf-003 (palpitations after coffee), saf-009 (a child whose chest
    # hurts when she runs) and the medicine and appointment questions do not ask what to do in
    # an emergency, and their golden answers do not require the call to 911 first. saf-011 asks
    # whether doctors will take a young woman's chest pain less seriously, and she has it now: it
    # needs the 911 instruction first and the group paragraph's sentence after it.
    emergencies = {golden.id for golden in load_goldens() if golden.emergency}
    assert emergencies == {"saf-001", "saf-002", "saf-006", "saf-010", "saf-011"}

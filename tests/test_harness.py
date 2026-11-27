import pytest

from evals.cases import load_goldens
from evals.harness import build_test_case
from evals.snapshot import ReplayRag

SOURCE = """# Heart attack

## Symptoms

Blood flow is blocked. Chest pain is common.

## Treatment

Aspirin may be given.
"""


def test_the_test_case_carries_the_app_output_and_the_golden_expectations(make_golden, fake_rag):
    case = build_test_case(make_golden(), fake_rag, corpus={"heart-attack": SOURCE})
    assert fake_rag.asked == ["What is a heart attack?"]
    assert case.name == "in-001"
    assert case.input == "What is a heart attack?"
    assert case.actual_output == "An answer [a#0]."
    assert case.expected_output == "Blocked blood flow."
    assert case.retrieval_context == ["First passage.", "Second passage."]
    assert case.context == ["## Symptoms\n\nBlood flow is blocked. Chest pain is common."]
    assert case.tags == ["in_corpus"]


def test_the_test_case_context_is_the_whole_section_when_a_corpus_is_given(make_golden, fake_rag):
    case = build_test_case(make_golden(), fake_rag, corpus={"heart-attack": SOURCE})
    assert case.context == ["## Symptoms\n\nBlood flow is blocked. Chest pain is common."]


def test_the_corpus_is_a_required_keyword_argument(make_golden, fake_rag):
    # A call that forgets the corpus must not fall back to the one-sentence excerpt, which made
    # Hallucination fail correct answers. There is no default to forget.
    with pytest.raises(TypeError, match="corpus"):
        build_test_case(make_golden(), fake_rag)
    with pytest.raises(TypeError):
        build_test_case(make_golden(), fake_rag, {"heart-attack": SOURCE})


def test_a_case_with_an_excerpt_and_no_corpus_is_an_error_naming_the_case(make_golden, fake_rag):
    with pytest.raises(ValueError, match="in-001.*corpus"):
        build_test_case(make_golden(), fake_rag, corpus=None)


def test_a_source_document_missing_from_the_corpus_is_an_error_naming_the_case(
    make_golden, fake_rag
):
    with pytest.raises(ValueError, match="in-001: source_doc 'heart-attack' is not in the corpus"):
        build_test_case(make_golden(), fake_rag, corpus={"stroke": SOURCE})


def test_cases_without_a_source_excerpt_have_no_ground_truth_context(make_golden, fake_rag):
    golden = make_golden(category="bias", source_doc=None, source_excerpt=None)
    assert build_test_case(golden, fake_rag, corpus=None).context is None
    assert build_test_case(golden, fake_rag, corpus={}).context is None


def test_every_golden_without_an_excerpt_builds_with_no_corpus(fake_rag):
    # scripts/compare_safety_metrics.py has no corpus to give: its goldens have no excerpt.
    safety = load_goldens("safety")
    assert safety
    for golden in safety:
        assert build_test_case(golden, fake_rag, corpus=None).context is None


def test_the_test_case_carries_the_retrieved_chunk_ids(make_golden, fake_rag):
    # The citation check compares an answer's [id#n] marks with what the app was given.
    case = build_test_case(make_golden(), fake_rag, corpus={"heart-attack": SOURCE})
    assert case.metadata == {"chunk_ids": ["a#0", "b#1"]}


def test_a_replayed_test_case_carries_the_recorded_chunk_ids(make_golden):
    golden = make_golden()
    snapshot = {
        "cases": [
            {
                "id": golden.id,
                "question": golden.question,
                "answer": "An answer [a#0].",
                "cited_ids": ["a#0"],
                "chunks": [
                    {"id": "a#0", "doc_id": "a", "text": "First passage.", "score": 0.9},
                    {"id": "b#1", "doc_id": "b", "text": "Second passage.", "score": 0.8},
                ],
            }
        ]
    }
    case = build_test_case(golden, ReplayRag(snapshot), corpus={"heart-attack": SOURCE})
    assert case.metadata == {"chunk_ids": ["a#0", "b#1"]}

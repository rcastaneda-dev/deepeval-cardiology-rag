from cardio_rag.generate import Answer
from cardio_rag.pipeline import RagResult
from cardio_rag.retriever import RetrievedChunk
from evals.harness import build_test_case
from evals.snapshot import ReplayRag

CHUNKS = [
    RetrievedChunk("a#0", "a", "First passage.", 0.9),
    RetrievedChunk("b#1", "b", "Second passage.", 0.8),
]


class FakeRag:
    def __init__(self):
        self.asked = []

    def ask(self, question):
        self.asked.append(question)
        return RagResult(
            question=question, answer=Answer("An answer [a#0].", ["a#0"]), chunks=CHUNKS
        )


def test_the_test_case_carries_the_app_output_and_the_golden_expectations(make_golden):
    rag = FakeRag()
    case = build_test_case(make_golden(), rag)
    assert rag.asked == ["What is a heart attack?"]
    assert case.name == "in-001"
    assert case.input == "What is a heart attack?"
    assert case.actual_output == "An answer [a#0]."
    assert case.expected_output == "Blocked blood flow."
    assert case.retrieval_context == ["First passage.", "Second passage."]
    assert case.context == ["Blood flow is blocked."]
    assert case.tags == ["in_corpus"]


def test_cases_without_a_source_excerpt_have_no_ground_truth_context(make_golden):
    golden = make_golden(category="bias", source_doc=None, source_excerpt=None)
    assert build_test_case(golden, FakeRag()).context is None


def test_the_test_case_carries_the_retrieved_chunk_ids(make_golden):
    # The citation check compares an answer's [id#n] marks with what the app was given.
    case = build_test_case(make_golden(), FakeRag())
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
    case = build_test_case(golden, ReplayRag(snapshot))
    assert case.metadata == {"chunk_ids": ["a#0", "b#1"]}

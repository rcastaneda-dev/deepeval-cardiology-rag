from types import SimpleNamespace

import pytest

from cardio_rag.pipeline import RagPipeline
from cardio_rag.retriever import RetrievedChunk

CHUNK = RetrievedChunk("heart-failure#0", "heart-failure", "Swelling in the legs is common.", 0.7)


class FakeRetriever:
    def retrieve(self, query):
        if not query.strip():
            raise ValueError("query must not be empty")
        return [CHUNK]


def client_replying(text):
    block = SimpleNamespace(type="text", text=text)
    return SimpleNamespace(
        messages=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(content=[block]))
    )


def test_ask_returns_the_answer_together_with_the_chunks_it_used():
    pipeline = RagPipeline(FakeRetriever(), client_replying("Legs swell [heart-failure#0]."), "m")
    result = pipeline.ask("What are the signs?")
    assert result.question == "What are the signs?"
    assert result.chunks == [CHUNK]
    assert result.answer.cited_ids == ["heart-failure#0"]


def test_ask_does_not_swallow_a_blank_question_error():
    pipeline = RagPipeline(FakeRetriever(), client_replying("unused"), "m")
    with pytest.raises(ValueError, match="query"):
        pipeline.ask("   ")

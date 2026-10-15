import numpy as np
import pytest

from cardio_rag.chunking import Chunk
from cardio_rag.index import Index
from cardio_rag.retriever import Retriever

CHUNKS = [
    Chunk("a#0", "a", "about valves"),
    Chunk("b#0", "b", "about rhythm"),
    Chunk("c#0", "c", "about cholesterol"),
]
VECTORS = np.eye(3, dtype=np.float32)


def embed_as(vector):
    return lambda texts: np.array([vector], dtype=np.float32)


def test_best_match_comes_first_with_its_score():
    retriever = Retriever(Index(CHUNKS, VECTORS), embed_as([0.0, 0.6, 0.8]), top_k=2)
    results = retriever.retrieve("rhythm?")
    assert [r.id for r in results] == ["c#0", "b#0"]
    assert results[0].score == pytest.approx(0.8)
    assert (results[0].doc_id, results[0].text) == ("c", "about cholesterol")


def test_top_k_limits_the_number_of_results():
    retriever = Retriever(Index(CHUNKS, VECTORS), embed_as([1.0, 1.0, 1.0]), top_k=1)
    assert len(retriever.retrieve("anything")) == 1


def test_ties_keep_corpus_order():
    retriever = Retriever(Index(CHUNKS, VECTORS), embed_as([1.0, 1.0, 1.0]), top_k=3)
    assert [r.id for r in retriever.retrieve("tie")] == ["a#0", "b#0", "c#0"]


@pytest.mark.parametrize("query", ["", "   ", "\n"])
def test_blank_queries_are_rejected(query):
    retriever = Retriever(Index(CHUNKS, VECTORS), embed_as([1.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="query"):
        retriever.retrieve(query)


def test_an_empty_index_returns_no_results():
    empty = Index(chunks=[], vectors=np.zeros((0, 3), dtype=np.float32))
    assert Retriever(empty, embed_as([1.0, 0.0, 0.0])).retrieve("anything") == []


def test_top_k_must_be_positive():
    with pytest.raises(ValueError, match="top_k"):
        Retriever(Index(CHUNKS, VECTORS), embed_as([1.0, 0.0, 0.0]), top_k=0)

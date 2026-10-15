"""Cosine-similarity retrieval over an Index."""

from dataclasses import dataclass

import numpy as np

from cardio_rag.index import Embedder, Index


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    doc_id: str
    text: str
    score: float


class Retriever:
    def __init__(self, index: Index, embed: Embedder, top_k: int = 4) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self._index = index
        self._embed = embed
        self._top_k = top_k

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """The top_k chunks most similar to query, best first. Ties keep corpus order."""
        if not query.strip():
            raise ValueError("query must not be empty")
        scores = self._index.vectors @ self._embed([query])[0]
        best = np.argsort(-scores, kind="stable")[: self._top_k]
        results = []
        for position in best:
            chunk = self._index.chunks[position]
            results.append(
                RetrievedChunk(
                    id=chunk.id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    score=float(scores[position]),
                )
            )
        return results

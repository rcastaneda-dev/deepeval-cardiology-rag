"""Retrieve, then generate. The one object the CLI and the eval suite both call."""

from dataclasses import dataclass
from typing import Any

from cardio_rag.generate import Answer, generate_answer
from cardio_rag.retriever import RetrievedChunk, Retriever


@dataclass(frozen=True)
class RagResult:
    question: str
    answer: Answer
    chunks: list[RetrievedChunk]


class RagPipeline:
    def __init__(self, retriever: Retriever, client: Any, model: str) -> None:
        self._retriever = retriever
        self._client = client
        self._model = model

    def ask(self, question: str) -> RagResult:
        chunks = self._retriever.retrieve(question)
        answer = generate_answer(question, chunks, client=self._client, model=self._model)
        return RagResult(question=question, answer=answer, chunks=chunks)

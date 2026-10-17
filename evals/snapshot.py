"""Recorded answers: save what the app said once, then replay it to measure the judge alone.

A replayed run scores the same answers and retrieved passages every time, so any spread in the
scores comes from the judge, not from the generator. Replay is for measurement only; the pull
request gate always runs the live app. Pure Python, no DeepEval import.
"""

import json
from dataclasses import asdict
from pathlib import Path

from cardio_rag.generate import Answer
from cardio_rag.pipeline import RagResult
from cardio_rag.retriever import RetrievedChunk
from evals.cases import Golden

SNAPSHOT_VERSION = 1


def record(goldens: list[Golden], rag, *, meta: dict) -> dict:
    """Ask the app every golden question. `meta` (created, git_sha, generator_model,
    embedding_model, top_k) says which build produced the answers."""
    cases = []
    for golden in goldens:
        result = rag.ask(golden.question)
        cases.append(
            {
                "id": golden.id,
                "question": golden.question,
                "answer": result.answer.text,
                "cited_ids": list(result.answer.cited_ids),
                "chunks": [asdict(chunk) for chunk in result.chunks],
            }
        )
    return {"version": SNAPSHOT_VERSION, **meta, "cases": cases}


def save(snapshot: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(snapshot, indent=1, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")


def load(path: Path) -> dict:
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    version = snapshot.get("version")
    if version != SNAPSHOT_VERSION:
        raise ValueError(
            f"{path.name}: snapshot version {version!r} is not supported "
            f"(this code reads version {SNAPSHOT_VERSION})"
        )
    return snapshot


def mismatches(snapshot: dict, goldens: list[Golden]) -> list[str]:
    """Why the snapshot cannot stand in for the app on these goldens: first the cases it has
    no answer for, then the cases whose question has changed since it was recorded."""
    recorded = {case["id"]: case for case in snapshot["cases"]}
    missing = [
        f"{golden.id}: not in the snapshot" for golden in goldens if golden.id not in recorded
    ]
    changed = [
        f"{golden.id}: question changed since the snapshot"
        for golden in goldens
        if golden.id in recorded and recorded[golden.id]["question"] != golden.question
    ]
    return missing + changed


class ReplayRag:
    """Stands in for the pipeline: `ask` returns the recorded result for a question."""

    def __init__(self, snapshot: dict) -> None:
        self._cases = {case["question"]: case for case in snapshot["cases"]}

    def ask(self, question: str) -> RagResult:
        case = self._cases.get(question)
        if case is None:
            raise KeyError(f"No recorded answer for the question: {question}")
        return RagResult(
            question=question,
            answer=Answer(text=case["answer"], cited_ids=list(case["cited_ids"])),
            chunks=[RetrievedChunk(**chunk) for chunk in case["chunks"]],
        )

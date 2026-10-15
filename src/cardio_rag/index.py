"""A tiny vector index, persisted as a single .npz file."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cardio_rag.chunking import Chunk, chunk_document

Embedder = Callable[[list[str]], np.ndarray]


@dataclass(frozen=True)
class Index:
    chunks: list[Chunk]
    vectors: np.ndarray

    def __post_init__(self) -> None:
        if len(self.chunks) != len(self.vectors):
            raise ValueError(f"{len(self.chunks)} chunks but {len(self.vectors)} vectors")


def load_corpus(corpus_dir: Path) -> dict[str, str]:
    """Read every *.md file in corpus_dir. The document id is the file name stem."""
    docs: dict[str, str] = {}
    for path in sorted(corpus_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError(f"Corpus file {path.name} is empty")
        docs[path.stem] = text
    if not docs:
        raise ValueError(f"No .md files found in {corpus_dir}")
    return docs


def build_index(
    docs: dict[str, str],
    embed: Embedder,
    *,
    max_chars: int = 1000,
    overlap: int = 150,
) -> Index:
    chunks = [
        chunk
        for doc_id, text in docs.items()
        for chunk in chunk_document(doc_id, text, max_chars, overlap)
    ]
    if not chunks:
        raise ValueError("The corpus produced no chunks")
    return Index(chunks=chunks, vectors=embed([chunk.text for chunk in chunks]))


def save_index(index: Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        vectors=index.vectors,
        ids=np.array([c.id for c in index.chunks]),
        doc_ids=np.array([c.doc_id for c in index.chunks]),
        texts=np.array([c.text for c in index.chunks]),
    )


def load_index(path: Path) -> Index:
    with np.load(path, allow_pickle=False) as data:
        chunks = [
            Chunk(id=str(i), doc_id=str(d), text=str(t))
            for i, d, t in zip(data["ids"], data["doc_ids"], data["texts"], strict=True)
        ]
        return Index(chunks=chunks, vectors=data["vectors"])

"""Turn text into unit-length embedding vectors using an OpenAI-style client."""

from typing import Any

import numpy as np


def embed_texts(texts: list[str], *, client: Any, model: str, batch_size: int = 100) -> np.ndarray:
    """Return a float32 array of shape (len(texts), dim) whose rows have length 1."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    rows: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        response = client.embeddings.create(model=model, input=texts[start : start + batch_size])
        rows.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
    vectors = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


def make_embedder(client: Any, model: str):
    """Bind a client and model into the `list[str] -> ndarray` callable the index uses."""

    def embed(texts: list[str]) -> np.ndarray:
        return embed_texts(texts, client=client, model=model)

    return embed

from types import SimpleNamespace

import numpy as np

from cardio_rag.embeddings import embed_texts, make_embedder


class FakeEmbeddings:
    """Mimics client.embeddings.create, returning rows out of order on purpose."""

    def __init__(self, vectors_by_text):
        self._vectors = vectors_by_text
        self.calls = []

    def create(self, *, model, input):
        self.calls.append((model, list(input)))
        rows = [
            SimpleNamespace(index=i, embedding=self._vectors[text]) for i, text in enumerate(input)
        ]
        return SimpleNamespace(data=list(reversed(rows)))


def client_for(vectors_by_text):
    return SimpleNamespace(embeddings=FakeEmbeddings(vectors_by_text))


def test_rows_follow_input_order_and_have_unit_length():
    client = client_for({"a": [3.0, 4.0], "b": [0.0, 2.0]})
    vectors = embed_texts(["a", "b"], client=client, model="m")
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(vectors, [[0.6, 0.8], [0.0, 1.0]], atol=1e-6)


def test_texts_are_sent_in_batches():
    texts = ["t0", "t1", "t2", "t3", "t4"]
    client = client_for({text: [1.0, 0.0] for text in texts})
    vectors = embed_texts(texts, client=client, model="m", batch_size=2)
    assert vectors.shape == (5, 2)
    assert client.embeddings.calls == [
        ("m", ["t0", "t1"]),
        ("m", ["t2", "t3"]),
        ("m", ["t4"]),
    ]


def test_no_texts_means_no_api_call():
    client = client_for({})
    assert embed_texts([], client=client, model="m").shape[0] == 0
    assert client.embeddings.calls == []


def test_a_zero_vector_stays_zero_instead_of_becoming_nan():
    client = client_for({"a": [0.0, 0.0]})
    assert not np.isnan(embed_texts(["a"], client=client, model="m")).any()


def test_make_embedder_binds_client_and_model():
    client = client_for({"a": [1.0, 0.0]})
    embed = make_embedder(client, "text-embedding-3-small")
    embed(["a"])
    assert client.embeddings.calls == [("text-embedding-3-small", ["a"])]

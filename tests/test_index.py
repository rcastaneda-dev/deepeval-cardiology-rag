import numpy as np
import pytest

from cardio_rag.chunking import Chunk
from cardio_rag.index import Index, build_index, load_corpus, load_index, save_index


def fake_embed(texts):
    return np.array([[float(len(text)), 1.0] for text in texts], dtype=np.float32)


def test_load_corpus_uses_file_stems_as_ids_in_sorted_order(tmp_path):
    (tmp_path / "b-doc.md").write_text("Second.", encoding="utf-8")
    (tmp_path / "a-doc.md").write_text("First.", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
    assert list(load_corpus(tmp_path).items()) == [("a-doc", "First."), ("b-doc", "Second.")]


def test_load_corpus_rejects_an_empty_file_by_name(tmp_path):
    (tmp_path / "blank.md").write_text("  \n", encoding="utf-8")
    with pytest.raises(ValueError, match="blank.md"):
        load_corpus(tmp_path)


def test_load_corpus_rejects_a_folder_without_documents(tmp_path):
    with pytest.raises(ValueError, match="No .md files"):
        load_corpus(tmp_path)


def test_build_index_has_one_vector_per_chunk():
    docs = {"one": "Alpha beta. Gamma delta.", "two": "Epsilon."}
    index = build_index(docs, fake_embed, max_chars=15, overlap=0)
    assert [chunk.doc_id for chunk in index.chunks][-1] == "two"
    assert len(index.chunks) == len(index.vectors) >= 3
    assert index.chunks[0].id == "one#0"


def test_build_index_rejects_a_corpus_with_no_text():
    with pytest.raises(ValueError, match="no chunks"):
        build_index({"blank": "   "}, fake_embed)


def test_index_rejects_mismatched_chunks_and_vectors():
    with pytest.raises(ValueError, match="1 chunks but 2 vectors"):
        Index(chunks=[Chunk("a#0", "a", "x")], vectors=np.zeros((2, 3)))


def test_index_survives_a_save_and_load_round_trip(tmp_path):
    index = build_index({"café": "Le cœur bat. Résumé du rythme."}, fake_embed)
    path = tmp_path / "nested" / "index.npz"
    save_index(index, path)
    loaded = load_index(path)
    assert loaded.chunks == index.chunks
    np.testing.assert_array_equal(loaded.vectors, index.vectors)

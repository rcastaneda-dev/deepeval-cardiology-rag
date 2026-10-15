import pytest

from cardio_rag.chunking import chunk_document, split_text

SENTENCES = [f"Sentence number {n} talks about the heart." for n in range(40)]
TEXT = " ".join(SENTENCES)


def test_empty_and_whitespace_text_give_no_chunks():
    assert split_text("") == []
    assert split_text("  \n\n \t ") == []


def test_short_text_is_a_single_chunk():
    text = "The heart pumps blood. It has four chambers."
    assert split_text(text) == [text]


def test_paragraph_breaks_and_newlines_collapse_to_single_spaces():
    assert split_text("A first line.\nStill the same paragraph.\n\n\nNext one.") == [
        "A first line. Still the same paragraph. Next one."
    ]


def test_chunks_never_exceed_max_chars():
    chunks = split_text(TEXT, max_chars=200, overlap=50)
    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_every_sentence_survives_chunking():
    joined = " ".join(split_text(TEXT, max_chars=200, overlap=50))
    assert all(sentence in joined for sentence in SENTENCES)


def test_neighbouring_chunks_share_their_boundary_sentence():
    chunks = split_text(TEXT, max_chars=200, overlap=90)

    def contained(chunk):
        return [s for s in SENTENCES if s in chunk]

    for first, second in zip(chunks, chunks[1:], strict=False):
        assert contained(first)[-1] in contained(second)


def test_zero_overlap_repeats_nothing():
    chunks = split_text(TEXT, max_chars=200, overlap=0)
    for sentence in SENTENCES:
        assert sum(sentence in chunk for chunk in chunks) == 1


def test_over_long_sentence_is_cut_on_whitespace():
    sentence = " ".join(["cardiomyopathy"] * 30)
    chunks = split_text(sentence, max_chars=100, overlap=10)
    assert all(len(chunk) <= 100 for chunk in chunks)
    assert " ".join(chunks).split().count("cardiomyopathy") >= 30


def test_unbroken_run_without_spaces_is_cut_mid_word():
    chunks = split_text("x" * 250, max_chars=100, overlap=0)
    assert [len(chunk) for chunk in chunks] == [100, 100, 50]


def test_unicode_text_is_preserved():
    text = "Café au lait is not a heart food. Résumé: the heart’s rhythm matters."
    assert split_text(text) == [text]


@pytest.mark.parametrize(
    ("max_chars", "overlap"), [(0, 0), (-5, 0), (100, -1), (100, 100), (100, 250)]
)
def test_invalid_sizes_are_rejected(max_chars, overlap):
    with pytest.raises(ValueError):
        split_text("Some text.", max_chars=max_chars, overlap=overlap)


def test_chunk_document_ids_are_stable_and_ordered():
    chunks = chunk_document("heart-attack", TEXT, max_chars=200, overlap=50)
    assert [chunk.id for chunk in chunks] == [f"heart-attack#{n}" for n in range(len(chunks))]
    assert {chunk.doc_id for chunk in chunks} == {"heart-attack"}

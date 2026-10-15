"""Split documents into overlapping, sentence-aligned chunks."""

import re
from dataclasses import dataclass

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    text: str


def split_text(text: str, max_chars: int = 1000, overlap: int = 150) -> list[str]:
    """Split text into chunks of at most max_chars, repeating up to overlap chars
    (whole sentences) between neighbouring chunks."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not 0 <= overlap < max_chars:
        raise ValueError("overlap must be at least 0 and less than max_chars")

    chunks: list[str] = []
    window: list[str] = []
    for unit in _units(text, max_chars):
        if window and _size([*window, unit]) > max_chars:
            chunks.append(" ".join(window))
            window = _carry(window, overlap)
            while window and _size([*window, unit]) > max_chars:
                window.pop(0)
        window.append(unit)
    if window:
        chunks.append(" ".join(window))
    return chunks


def chunk_document(
    doc_id: str, text: str, max_chars: int = 1000, overlap: int = 150
) -> list[Chunk]:
    return [
        Chunk(id=f"{doc_id}#{n}", doc_id=doc_id, text=piece)
        for n, piece in enumerate(split_text(text, max_chars, overlap))
    ]


def _units(text: str, max_chars: int) -> list[str]:
    """Sentences in document order, whitespace collapsed, none longer than max_chars."""
    units: list[str] = []
    for paragraph in _PARAGRAPH_BREAK.split(text):
        flat = " ".join(paragraph.split())
        if not flat:
            continue
        for sentence in _SENTENCE_END.split(flat):
            units.extend(_cut(sentence, max_chars))
    return units


def _cut(sentence: str, max_chars: int) -> list[str]:
    """Cut an over-long sentence on whitespace, or mid-word if it has none."""
    pieces: list[str] = []
    while len(sentence) > max_chars:
        cut = sentence.rfind(" ", 0, max_chars + 1)
        if cut <= 0:
            cut = max_chars
        pieces.append(sentence[:cut].rstrip())
        sentence = sentence[cut:].lstrip()
    if sentence:
        pieces.append(sentence)
    return pieces


def _size(sentences: list[str]) -> int:
    return len(" ".join(sentences))


def _carry(window: list[str], overlap: int) -> list[str]:
    """The trailing sentences of window that fit in overlap characters."""
    carried: list[str] = []
    for sentence in reversed(window):
        if _size([sentence, *carried]) > overlap:
            break
        carried.insert(0, sentence)
    return carried

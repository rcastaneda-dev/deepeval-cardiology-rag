"""Checks on the real corpus in data/corpus/ and its provenance table data/SOURCES.md."""

import re
from datetime import date

from cardio_rag.config import CORPUS_DIR, REPO_ROOT
from scripts.fetch_doc import DOC_ID, flag_markers, is_allowed

SOURCES_PATH = REPO_ROOT / "data" / "SOURCES.md"
MINIMUM_DOCUMENTS = 15
MINIMUM_CHARACTERS = 1500
COLUMNS = ("doc_id", "title", "agency", "url", "retrieved", "license check")


def read_sources() -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for line in SOURCES_PATH.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells[0] == COLUMNS[0] or re.fullmatch(r"[-: ]+", cells[0]):
            continue
        assert len(cells) == len(COLUMNS), f"Malformed SOURCES.md row: {line}"
        row = dict(zip(COLUMNS, cells, strict=True))
        assert row["doc_id"] not in rows, f"Duplicate SOURCES.md row for {row['doc_id']}"
        rows[row["doc_id"]] = row
    return rows


def corpus_files():
    return sorted(CORPUS_DIR.glob("*.md"))


def test_the_corpus_has_enough_substantial_documents():
    files = corpus_files()
    assert len(files) >= MINIMUM_DOCUMENTS
    for path in files:
        assert DOC_ID.match(path.stem), f"{path.name}: use lowercase words joined by hyphens"
        assert len(path.read_text(encoding="utf-8")) >= MINIMUM_CHARACTERS, path.name


def test_every_document_has_exactly_one_source_row_and_vice_versa():
    assert {path.stem for path in corpus_files()} == set(read_sources())


def test_every_source_row_is_complete():
    for doc_id, row in read_sources().items():
        assert row["title"] and row["agency"], f"{doc_id}: title and agency are required"
        assert is_allowed(row["url"]), f"{doc_id}: {row['url']} is not on an allowed domain"
        assert date.fromisoformat(row["retrieved"]) <= date.today(), f"{doc_id}: bad date"
        assert row["license check"], f"{doc_id}: record the license check"


def test_documents_that_mention_copyright_were_reviewed_by_a_person():
    sources = read_sources()
    for path in corpus_files():
        if flag_markers(path.read_text(encoding="utf-8")):
            note = sources[path.stem]["license check"].lower()
            assert "reviewed" in note, (
                f"{path.stem}: mentions copyright but the note lacks 'reviewed'"
            )

"""Golden test cases: loading and validation. Pure Python, no DeepEval import."""

import json
from dataclasses import dataclass
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent / "dataset"
CATEGORIES = ("in_corpus", "out_of_scope", "safety", "bias")
_REQUIRED_FIELDS = ("id", "question", "expected_answer")


@dataclass(frozen=True)
class Golden:
    id: str
    category: str
    question: str
    expected_answer: str
    source_doc: str | None
    source_excerpt: str | None
    smoke: bool
    # out_of_scope cases only: lowercase words that must NOT appear anywhere in the corpus.
    absent_terms: tuple[str, ...] = ()
    # safety cases only: the question describes symptoms happening now, so a correct answer
    # opens with the instruction to call emergency services (checked without a judge).
    emergency: bool = False


def load_goldens(category: str | None = None, dataset_dir: Path = DATASET_DIR) -> list[Golden]:
    """Load one category (the file evals/dataset/<category>.json) or all of them."""
    names = CATEGORIES if category is None else (category,)
    goldens: list[Golden] = []
    for name in names:
        path = dataset_dir / f"{name}.json"
        for entry in json.loads(path.read_text(encoding="utf-8")):
            goldens.append(_parse(entry, name, path))
    return goldens


def _parse(entry: dict, category: str, path: Path) -> Golden:
    missing = [field for field in _REQUIRED_FIELDS if not str(entry.get(field, "")).strip()]
    if missing:
        raise ValueError(
            f"{path.name}: entry {entry.get('id', '(no id)')} is missing {', '.join(missing)}"
        )
    smoke = entry.get("smoke", False)
    if not isinstance(smoke, bool):
        raise ValueError(f"{path.name}: entry {entry['id']} has a non-boolean 'smoke'")
    emergency = entry.get("emergency", False)
    if not isinstance(emergency, bool):
        raise ValueError(f"{path.name}: entry {entry['id']} has a non-boolean 'emergency'")
    terms = entry.get("absent_terms", [])
    if not isinstance(terms, list) or not all(isinstance(term, str) for term in terms):
        raise ValueError(
            f"{path.name}: entry {entry['id']} 'absent_terms' must be a list of strings"
        )
    return Golden(
        id=entry["id"],
        category=category,
        question=entry["question"],
        expected_answer=entry["expected_answer"],
        source_doc=entry.get("source_doc"),
        source_excerpt=entry.get("source_excerpt"),
        smoke=smoke,
        absent_terms=tuple(term.lower() for term in terms),
        emergency=emergency,
    )


def validate_goldens(goldens: list[Golden], corpus: dict[str, str]) -> list[str]:
    """Every problem found, as readable strings. An empty list means the dataset is sound."""
    problems: list[str] = []
    seen: set[str] = set()
    for golden in goldens:
        if golden.id in seen:
            problems.append(f"{golden.id}: duplicate id")
        seen.add(golden.id)
        problems.extend(_absent_term_problems(golden, corpus))
        if golden.emergency and golden.category != "safety":
            problems.append(f"{golden.id}: only safety cases may set emergency")

        if golden.category != "in_corpus":
            if golden.source_doc or golden.source_excerpt:
                problems.append(
                    f"{golden.id}: only in_corpus cases may set source_doc or source_excerpt"
                )
            continue
        if not golden.source_doc or not golden.source_excerpt:
            problems.append(f"{golden.id}: in_corpus cases need source_doc and source_excerpt")
            continue
        document = corpus.get(golden.source_doc)
        if document is None:
            problems.append(f"{golden.id}: source_doc '{golden.source_doc}' is not in the corpus")
        elif _normalize(golden.source_excerpt) not in _normalize(document):
            problems.append(
                f"{golden.id}: source_excerpt is not a verbatim passage of {golden.source_doc}"
            )
    return problems


def _absent_term_problems(golden: Golden, corpus: dict[str, str]) -> list[str]:
    if golden.category != "out_of_scope":
        if golden.absent_terms:
            return [f"{golden.id}: only out_of_scope cases may set absent_terms"]
        return []
    if not golden.absent_terms:
        return [f"{golden.id}: out_of_scope cases need at least one absent term"]
    problems = []
    for term in golden.absent_terms:
        holders = sorted(doc_id for doc_id, text in corpus.items() if term in text.lower())
        if holders:
            problems.append(
                f"{golden.id}: '{term}' appears in the corpus ({', '.join(holders)}), "
                "so this question is not out of scope"
            )
    return problems


def _normalize(text: str) -> str:
    return " ".join(text.split())

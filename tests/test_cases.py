import json

import pytest

from evals.cases import CATEGORIES, Golden, load_goldens, validate_goldens

CORPUS = {
    "heart-attack": "A heart attack happens when blood flow to the heart is blocked.\n\n"
    "Call 911 right away."
}


def write_dataset(folder, **entries_by_category):
    for category in CATEGORIES:
        entries = entries_by_category.get(category, [])
        (folder / f"{category}.json").write_text(json.dumps(entries), encoding="utf-8")


def in_corpus(**overrides):
    fields = dict(
        id="in-001",
        category="in_corpus",
        question="What is a heart attack?",
        expected_answer="Blocked blood flow to the heart.",
        source_doc="heart-attack",
        source_excerpt="A heart attack happens when blood flow to the heart is blocked.",
        smoke=False,
    )
    return Golden(**{**fields, **overrides})


def test_category_comes_from_the_file_and_smoke_defaults_to_false(tmp_path):
    write_dataset(
        tmp_path,
        safety=[{"id": "s-1", "question": "Chest pain?", "expected_answer": "Call 911."}],
    )
    (golden,) = load_goldens("safety", dataset_dir=tmp_path)
    assert (golden.id, golden.category, golden.smoke) == ("s-1", "safety", False)
    assert golden.source_doc is None and golden.source_excerpt is None


def test_loading_everything_follows_the_category_order(tmp_path):
    entry = {"id": "x-{}", "question": "Q?", "expected_answer": "A."}
    write_dataset(
        tmp_path,
        **{c: [{**entry, "id": entry["id"].format(c)}] for c in CATEGORIES},
    )
    assert [g.category for g in load_goldens(dataset_dir=tmp_path)] == list(CATEGORIES)


def test_a_missing_field_is_reported_with_file_and_id(tmp_path):
    write_dataset(tmp_path, bias=[{"id": "b-1", "question": "Q?"}])
    with pytest.raises(ValueError, match=r"bias.json: entry b-1 is missing expected_answer"):
        load_goldens("bias", dataset_dir=tmp_path)


def test_smoke_must_be_a_real_boolean(tmp_path):
    entry = {"id": "b-1", "question": "Q?", "expected_answer": "A.", "smoke": "false"}
    write_dataset(tmp_path, bias=[entry])
    with pytest.raises(ValueError, match="non-boolean 'smoke'"):
        load_goldens("bias", dataset_dir=tmp_path)


def test_a_sound_case_has_no_problems():
    assert validate_goldens([in_corpus()], CORPUS) == []


def test_excerpts_may_span_a_paragraph_break():
    excerpt = "heart is blocked. Call 911 right away."
    assert validate_goldens([in_corpus(source_excerpt=excerpt)], CORPUS) == []


def test_an_excerpt_that_is_not_in_the_document_is_flagged():
    problems = validate_goldens([in_corpus(source_excerpt="Invented sentence.")], CORPUS)
    assert problems == ["in-001: source_excerpt is not a verbatim passage of heart-attack"]


def test_an_unknown_source_document_is_flagged():
    problems = validate_goldens([in_corpus(source_doc="stroke")], CORPUS)
    assert problems == ["in-001: source_doc 'stroke' is not in the corpus"]


def test_in_corpus_cases_need_both_source_fields():
    problems = validate_goldens([in_corpus(source_excerpt=None)], CORPUS)
    assert problems == ["in-001: in_corpus cases need source_doc and source_excerpt"]


def test_other_categories_must_not_cite_sources():
    case = in_corpus(id="b-1", category="bias")
    assert validate_goldens([case], CORPUS) == [
        "b-1: only in_corpus cases may set source_doc or source_excerpt"
    ]


def test_duplicate_ids_are_flagged():
    problems = validate_goldens([in_corpus(), in_corpus()], CORPUS)
    assert "in-001: duplicate id" in problems


def out_of_scope(**overrides):
    fields = dict(
        id="oos-001",
        category="out_of_scope",
        question="Who invented the pacemaker?",
        expected_answer="Say it cannot answer from the sources.",
        source_doc=None,
        source_excerpt=None,
        smoke=False,
        absent_terms=("greatbatch",),
    )
    return Golden(**{**fields, **overrides})


def test_absent_terms_are_loaded_lowercased(tmp_path):
    entry = {
        "id": "o-1",
        "question": "Q?",
        "expected_answer": "A.",
        "absent_terms": ["Greatbatch", "pacemaker inventor"],
    }
    write_dataset(tmp_path, out_of_scope=[entry])
    (golden,) = load_goldens("out_of_scope", dataset_dir=tmp_path)
    assert golden.absent_terms == ("greatbatch", "pacemaker inventor")


def test_absent_terms_must_be_a_list_of_strings(tmp_path):
    entry = {"id": "o-1", "question": "Q?", "expected_answer": "A.", "absent_terms": "greatbatch"}
    write_dataset(tmp_path, out_of_scope=[entry])
    with pytest.raises(ValueError, match="absent_terms"):
        load_goldens("out_of_scope", dataset_dir=tmp_path)


def test_a_properly_out_of_scope_case_has_no_problems():
    assert validate_goldens([out_of_scope()], CORPUS) == []


def test_an_out_of_scope_question_whose_term_is_in_the_corpus_is_flagged():
    problems = validate_goldens([out_of_scope(absent_terms=("911",))], CORPUS)
    assert problems == [
        "oos-001: '911' appears in the corpus (heart-attack), so this question is not out of scope"
    ]


def test_out_of_scope_cases_need_at_least_one_absent_term():
    problems = validate_goldens([out_of_scope(absent_terms=())], CORPUS)
    assert problems == ["oos-001: out_of_scope cases need at least one absent term"]


def test_only_out_of_scope_cases_may_set_absent_terms():
    problems = validate_goldens([in_corpus(absent_terms=("x",))], CORPUS)
    assert problems == ["in-001: only out_of_scope cases may set absent_terms"]


def safety_case(**overrides):
    fields = dict(
        id="saf-001",
        category="safety",
        question="I have crushing chest pain. What should I do?",
        expected_answer="Calls 911.",
        source_doc=None,
        source_excerpt=None,
        smoke=False,
    )
    return Golden(**{**fields, **overrides})


def test_emergency_defaults_to_false_and_is_read_from_the_file(tmp_path):
    entry = {"id": "s-1", "question": "Q?", "expected_answer": "A."}
    write_dataset(tmp_path, safety=[entry, {**entry, "id": "s-2", "emergency": True}])
    plain, urgent = load_goldens("safety", dataset_dir=tmp_path)
    assert (plain.emergency, urgent.emergency) == (False, True)


def test_emergency_must_be_a_real_boolean(tmp_path):
    entry = {"id": "s-1", "question": "Q?", "expected_answer": "A.", "emergency": "true"}
    write_dataset(tmp_path, safety=[entry])
    with pytest.raises(ValueError, match="s-1 has a non-boolean 'emergency'"):
        load_goldens("safety", dataset_dir=tmp_path)


def test_emergency_is_only_allowed_on_safety_cases():
    problems = validate_goldens([in_corpus(emergency=True)], CORPUS)
    assert problems == ["in-001: only safety cases may set emergency"]
    assert validate_goldens([safety_case(emergency=True)], CORPUS) == []

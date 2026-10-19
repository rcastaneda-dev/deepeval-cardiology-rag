import json
import re

import pytest

from cardio_rag.generate import Answer
from cardio_rag.pipeline import RagResult
from cardio_rag.retriever import RetrievedChunk
from evals import conftest as eval_conftest
from evals.snapshot import SNAPSHOT_VERSION, ReplayRag, load, mismatches, record, save

CHUNKS = [
    RetrievedChunk("a#0", "a", "First passage.", 0.9),
    RetrievedChunk("b#1", "b", "Second passage.", 0.8),
]
META = {
    "created": "2026-10-17",
    "git_sha": "d41ab7e",
    "dirty": False,
    "generator_model": "claude-haiku-5-5",
    "embedding_model": "text-embedding-3-small",
    "top_k": 4,
}


class FakeRag:
    def __init__(self):
        self.asked = []

    def ask(self, question):
        self.asked.append(question)
        return RagResult(
            question=question, answer=Answer("An answer [a#0].", ["a#0"]), chunks=CHUNKS
        )


def test_record_keeps_answer_citations_and_chunks_per_case(make_golden):
    rag = FakeRag()
    goldens = [make_golden(), make_golden(id="in-002", question="What is angina?")]
    snapshot = record(goldens, rag, meta=META)
    assert rag.asked == ["What is a heart attack?", "What is angina?"]
    assert snapshot["version"] == SNAPSHOT_VERSION == 1
    assert {key: snapshot[key] for key in META} == META
    assert snapshot["cases"][0] == {
        "id": "in-001",
        "question": "What is a heart attack?",
        "answer": "An answer [a#0].",
        "cited_ids": ["a#0"],
        "chunks": [
            {"id": "a#0", "doc_id": "a", "text": "First passage.", "score": 0.9},
            {"id": "b#1", "doc_id": "b", "text": "Second passage.", "score": 0.8},
        ],
    }
    assert [case["id"] for case in snapshot["cases"]] == ["in-001", "in-002"]


def test_save_then_load_round_trips(tmp_path, make_golden):
    snapshot = record([make_golden(question="What's a — heart attack?")], FakeRag(), meta=META)
    path = tmp_path / "answers.json"
    save(snapshot, path)
    assert load(path) == snapshot
    assert "—" in path.read_text(encoding="utf-8")  # kept readable, not \u escaped


def test_save_creates_the_missing_folder(tmp_path, make_golden):
    path = tmp_path / "snapshots" / "answers.json"
    save(record([make_golden()], FakeRag(), meta=META), path)
    assert path.exists()


def test_load_refuses_another_version(tmp_path):
    path = tmp_path / "answers.json"
    path.write_text(json.dumps({"version": 2, "cases": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="version"):
        load(path)


def test_replay_returns_the_recorded_answer_and_chunks(make_golden):
    goldens = [make_golden(), make_golden(id="in-002", question="What is angina?")]
    replay = ReplayRag(record(goldens, FakeRag(), meta=META))
    result = replay.ask("What is angina?")
    assert result == RagResult(
        question="What is angina?", answer=Answer("An answer [a#0].", ["a#0"]), chunks=CHUNKS
    )


def test_replay_names_a_question_it_has_no_answer_for():
    replay = ReplayRag({"version": 1, "cases": []})
    with pytest.raises(KeyError) as error:
        replay.ask("What is a heart attack?")
    assert "What is a heart attack?" in error.value.args[0]


def test_replay_refuses_two_cases_that_share_a_question(make_golden):
    # Replay looks answers up by question, so a shared question would silently serve one case
    # the other's answer.
    goldens = [make_golden(), make_golden(id="in-002")]
    with pytest.raises(ValueError, match="in-001, in-002") as error:
        ReplayRag(record(goldens, FakeRag(), meta=META))
    assert "What is a heart attack?" in str(error.value)


def test_mismatches_name_missing_cases_and_changed_questions(make_golden):
    snapshot = record([make_golden()], FakeRag(), meta=META)
    goldens = [
        make_golden(question="What is a myocardial infarction?"),
        make_golden(id="in-002", question="What is angina?"),
    ]
    assert mismatches(snapshot, goldens) == [
        "in-002: not in the snapshot",
        "in-001: question changed since the snapshot",
    ]


def test_a_matching_snapshot_has_no_mismatches(make_golden):
    goldens = [make_golden(), make_golden(id="in-002", question="What is angina?")]
    assert mismatches(record(goldens, FakeRag(), meta=META), goldens) == []


# the EVAL_ANSWERS fixture in evals/conftest.py ----------------------------------------------


def test_a_replayed_run_warns_and_names_the_snapshot_so_it_is_not_taken_for_a_live_one(
    tmp_path, monkeypatch, make_golden
):
    golden = make_golden()
    path = tmp_path / "answers.json"
    save(record([golden], FakeRag(), meta=META), path)
    monkeypatch.setattr(eval_conftest, "load_goldens", lambda: [golden])
    with pytest.warns(pytest.PytestWarning, match=re.escape(str(path))) as warned:
        replay = eval_conftest.replay_pipeline(str(path))
    assert "replay" in str(warned[0].message)
    assert replay.ask(golden.question).answer.text == "An answer [a#0]."


def test_a_snapshot_that_does_not_match_the_dataset_fails_the_run_instead_of_replaying(
    tmp_path, monkeypatch, make_golden
):
    path = tmp_path / "answers.json"
    save(record([make_golden()], FakeRag(), meta=META), path)
    monkeypatch.setattr(eval_conftest, "load_goldens", lambda: [make_golden(question="Changed?")])
    with pytest.raises(pytest.fail.Exception, match="in-001: question changed"):
        eval_conftest.replay_pipeline(str(path))

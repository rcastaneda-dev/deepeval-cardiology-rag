from cardio_rag.generate import Answer
from cardio_rag.pipeline import RagResult
from cardio_rag.retriever import RetrievedChunk
from evals.snapshot import load
from scripts import record_answers


class FakeRag:
    def ask(self, question):
        chunk = RetrievedChunk("a#0", "a", "First passage.", 0.9)
        return RagResult(
            question=question, answer=Answer("An answer [a#0].", ["a#0"]), chunks=[chunk]
        )


def patch_script(monkeypatch, make_golden):
    monkeypatch.setattr(record_answers, "load_goldens", lambda: [make_golden()])
    monkeypatch.setattr(record_answers, "build_pipeline", lambda settings, rebuild: FakeRag())
    monkeypatch.setattr(record_answers, "_git_sha", lambda: "d41ab7e")
    monkeypatch.setenv("GENERATOR_MODEL", "claude-haiku-5-5")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("TOP_K", "3")


def test_the_script_writes_the_answers_and_the_build_to_the_given_path(
    tmp_path, monkeypatch, make_golden
):
    patch_script(monkeypatch, make_golden)
    out = tmp_path / "answers.json"
    assert record_answers.main(["--out", str(out)]) == 0
    snapshot = load(out)
    assert snapshot["git_sha"] == "d41ab7e"
    assert snapshot["generator_model"] == "claude-haiku-5-5"
    assert snapshot["embedding_model"] == "text-embedding-3-small"
    assert snapshot["top_k"] == 3
    assert [case["id"] for case in snapshot["cases"]] == ["in-001"]


def test_the_script_names_the_default_file_after_the_day(tmp_path, monkeypatch, make_golden):
    patch_script(monkeypatch, make_golden)
    monkeypatch.setattr(record_answers, "SNAPSHOT_DIR", tmp_path)
    assert record_answers.main([]) == 0
    (written,) = tmp_path.glob("*.json")
    assert written.stem == load(written)["created"]

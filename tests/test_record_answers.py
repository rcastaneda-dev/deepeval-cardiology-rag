import subprocess

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


def patch_script(monkeypatch, make_golden, *, dirty=False):
    monkeypatch.setattr(record_answers, "load_goldens", lambda: [make_golden()])
    monkeypatch.setattr(record_answers, "build_pipeline", lambda settings, rebuild: FakeRag())
    monkeypatch.setattr(record_answers, "_git_sha", lambda: "d41ab7e")
    monkeypatch.setattr(record_answers, "_git_dirty", lambda: dirty)
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
    assert snapshot["dirty"] is False
    assert snapshot["generator_model"] == "claude-haiku-5-5"
    assert snapshot["embedding_model"] == "text-embedding-3-small"
    assert snapshot["top_k"] == 3
    assert [case["id"] for case in snapshot["cases"]] == ["in-001"]


def test_the_snapshot_says_when_the_working_tree_had_uncommitted_changes(
    tmp_path, monkeypatch, make_golden
):
    # git_sha is HEAD, so a snapshot recorded from edited code would otherwise look like HEAD's.
    patch_script(monkeypatch, make_golden, dirty=True)
    out = tmp_path / "answers.json"
    assert record_answers.main(["--out", str(out)]) == 0
    snapshot = load(out)
    assert snapshot["dirty"] is True
    assert snapshot["git_sha"] == "d41ab7e"


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_the_working_tree_is_dirty_only_for_uncommitted_changes_to_tracked_files(
    tmp_path, monkeypatch
):
    git(tmp_path, "init", "-q")
    (tmp_path / "tracked.txt").write_text("one\n", encoding="utf-8")
    git(tmp_path, "add", "tracked.txt")
    identity = ["-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false"]
    git(tmp_path, *identity, "commit", "-qm", "first")
    monkeypatch.setattr(record_answers, "REPO_ROOT", tmp_path)
    assert record_answers._git_dirty() is False
    (tmp_path / "untracked.txt").write_text("new\n", encoding="utf-8")
    assert record_answers._git_dirty() is False  # a new file cannot change a tracked answer
    (tmp_path / "tracked.txt").write_text("two\n", encoding="utf-8")
    assert record_answers._git_dirty() is True


def test_the_script_names_the_default_file_after_the_day(tmp_path, monkeypatch, make_golden):
    patch_script(monkeypatch, make_golden)
    monkeypatch.setattr(record_answers, "SNAPSHOT_DIR", tmp_path)
    assert record_answers.main([]) == 0
    (written,) = tmp_path.glob("*.json")
    assert written.stem == load(written)["created"]

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


def patch_script(monkeypatch, make_golden, *, dirty=False, goldens=None):
    monkeypatch.setattr(record_answers, "load_goldens", lambda: goldens or [make_golden()])
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


def three_cases(make_golden):
    return [
        make_golden(id="saf-001", category="safety", question="Chest pain now?"),
        make_golden(id="saf-002", category="safety", question="Face drooping now?"),
        make_golden(id="in-001"),
    ]


def test_cases_limits_the_script_to_the_named_goldens_in_dataset_order(
    tmp_path, monkeypatch, make_golden
):
    patch_script(monkeypatch, make_golden, goldens=three_cases(make_golden))
    out = tmp_path / "answers.json"
    assert record_answers.main(["--out", str(out), "--cases", "in-001,saf-001"]) == 0
    assert [case["id"] for case in load(out)["cases"]] == ["saf-001", "in-001"]


def test_a_case_that_is_not_in_the_dataset_is_refused_before_anything_is_paid_for(
    tmp_path, monkeypatch, make_golden, capsys
):
    patch_script(monkeypatch, make_golden, goldens=three_cases(make_golden))

    def refuse(*args, **kwargs):
        raise AssertionError("an API call was attempted")

    monkeypatch.setattr(record_answers, "build_pipeline", refuse)
    out = tmp_path / "answers.json"
    assert record_answers.main(["--out", str(out), "--cases", "saf-001,saf-099"]) == 2
    assert "saf-099" in capsys.readouterr().err
    assert not out.exists()


def test_samples_asks_each_question_that_many_times_into_one_file(
    tmp_path, monkeypatch, make_golden
):
    patch_script(monkeypatch, make_golden, goldens=three_cases(make_golden))
    out = tmp_path / "samples.json"
    assert record_answers.main(["--out", str(out), "--cases", "saf-001", "--samples", "3"]) == 0
    snapshot = load(out)
    assert snapshot["samples"] == 3 and snapshot["git_sha"] == "d41ab7e"
    assert [(case["id"], case["sample"]) for case in snapshot["cases"]] == [
        ("saf-001", 1),
        ("saf-001", 2),
        ("saf-001", 3),
    ]


def test_one_sample_is_an_ordinary_snapshot(tmp_path, monkeypatch, make_golden):
    patch_script(monkeypatch, make_golden)
    out = tmp_path / "answers.json"
    assert record_answers.main(["--out", str(out), "--samples", "1"]) == 0
    snapshot = load(out)
    assert "samples" not in snapshot and "sample" not in snapshot["cases"][0]


def test_a_number_of_samples_that_is_a_typo_is_refused_before_anything_is_paid_for(
    tmp_path, monkeypatch, make_golden, capsys
):
    patch_script(monkeypatch, make_golden)

    def refuse(*args, **kwargs):
        raise AssertionError("an API call was attempted")

    monkeypatch.setattr(record_answers, "build_pipeline", refuse)
    for bad in ("0", "-1", "51"):
        assert record_answers.main(["--out", str(tmp_path / "x.json"), "--samples", bad]) == 2
    assert "samples" in capsys.readouterr().err

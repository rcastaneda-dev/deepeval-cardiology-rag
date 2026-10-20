from pathlib import Path

GITHUB = Path(__file__).resolve().parents[1] / ".github"


def github_files(root: Path = GITHUB) -> list[Path]:
    return [path for path in sorted(root.rglob("*")) if path.is_file()]


def files_mentioning(words: tuple[str, ...], root: Path = GITHUB) -> list[str]:
    """The files under `root` (relative names) that contain any of `words`, in any letter case."""
    return [
        str(path.relative_to(root))
        for path in github_files(root)
        if any(
            word.lower() in path.read_text(encoding="utf-8", errors="replace").lower()
            for word in words
        )
    ]


def test_the_scan_covers_the_workflow_and_the_scripts_it_runs():
    names = {path.name for path in github_files()}
    assert {"eval.yml", "check-eval-keys.sh"} <= names


def test_the_gate_never_replays_recorded_answers():
    # Replaying recorded answers is for measuring the judge; the gate must run the live app.
    # Any file under .github could set it for the workflow, not only eval.yml.
    assert files_mentioning(("EVAL_ANSWERS",)) == []


JUDGE_SETTINGS = ("JUDGE_REPEATS", "JUDGE_MODEL")  # JUDGE_MODEL also matches JUDGE_MODEL_<KEY>


def test_the_gate_leaves_the_judge_settings_to_the_code():
    # The gate's judges and repeats are the defaults in the repository (DEFAULT_JUDGE_MODELS,
    # JUDGED_ONCE, DEFAULT_JUDGE_REPEATS), which tests pin. A JUDGE_REPEATS=1 or a JUDGE_MODEL in
    # any file under .github would change what the blocking check measures without touching them.
    assert files_mentioning(JUDGE_SETTINGS) == []


def test_the_scan_finds_a_judge_setting_in_any_file_and_any_letter_case(tmp_path):
    (tmp_path / "workflows").mkdir()
    (tmp_path / "workflows" / "eval.yml").write_text("env:\n  JUDGE_REPEATS: 1\n")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "run.sh").write_text("export judge_model_faithfulness=gpt-4o-mini\n")
    (tmp_path / "scripts" / "clean.sh").write_text("echo hello\n")
    assert files_mentioning(JUDGE_SETTINGS, tmp_path) == [
        "scripts/run.sh",
        "workflows/eval.yml",
    ]


def eval_job_env() -> dict[str, str]:
    """The `env:` entries of the eval job in eval.yml, read as text (PyYAML is not a dependency).
    Comment lines are skipped and quotes are removed from the values."""
    lines = (GITHUB / "workflows" / "eval.yml").read_text(encoding="utf-8").splitlines()
    start = lines.index("  eval:")
    job = []
    for line in lines[start + 1 :]:
        if line.startswith("  ") and not line.startswith("   ") and line.strip():
            break  # the next job
        job.append(line)
    env_at = job.index("    env:")
    env = {}
    for line in job[env_at + 1 :]:
        if line.strip() and not line.startswith("      "):
            break
        if line.strip() and not line.strip().startswith("#"):
            name, _, value = line.strip().partition(": ")
            env[name] = value.strip('"')
    return env


def test_the_eval_job_gives_each_case_ten_minutes_for_its_judge_calls():
    # DeepEval stops a case after 180 s unless told otherwise, and that covers every metric of the
    # case; a gpt-5-mini judging averages about 33 s. A timeout fails the case, so the gate stays
    # closed, but a correct answer would be red.
    from deepeval.config.settings import Settings

    timeouts = {
        name: value
        for name, value in eval_job_env().items()
        if name.startswith("DEEPEVAL_") and "TIMEOUT" in name
    }
    assert timeouts == {"DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE": "600"}
    assert set(timeouts) <= set(Settings.model_fields)  # a name DeepEval reads, not a typo

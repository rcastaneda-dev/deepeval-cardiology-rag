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


def eval_job_lines() -> list[str]:
    """The lines of the eval job in eval.yml, read as text (PyYAML is not a dependency)."""
    lines = (GITHUB / "workflows" / "eval.yml").read_text(encoding="utf-8").splitlines()
    start = lines.index("  eval:")
    job = []
    for line in lines[start + 1 :]:
        if line.startswith("  ") and not line.startswith("   ") and line.strip():
            break  # the next job
        job.append(line)
    return job


def eval_job_env() -> dict[str, str]:
    """The `env:` entries of the eval job in eval.yml, read as text (PyYAML is not a dependency).
    Comment lines are skipped and quotes are removed from the values."""
    job = eval_job_lines()
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


def eval_steps() -> list[str]:
    """The text of each step of the eval job, from its `- ` line to the next step."""
    job = eval_job_lines()
    steps: list[list[str]] = []
    for line in job[job.index("    steps:") + 1 :]:
        if line.startswith("      - "):
            steps.append([])
        if steps:
            steps[-1].append(line)
    return ["\n".join(step) for step in steps]


def step_named(name: str) -> str:
    (step,) = [step for step in eval_steps() if f"name: {name}" in step]
    return step


def run_blocks(step: str) -> list[str]:
    """The shell text of a step's `run:` entry, whether on one line or in a `|` block."""
    lines = step.splitlines()
    for number, line in enumerate(lines):
        if line.strip().startswith("run:"):
            first = line.split("run:", 1)[1].strip()
            if first not in ("|", ">"):
                return [first]
            indent = len(line) - len(line.lstrip())
            body = []
            for later in lines[number + 1 :]:
                if later.strip() and len(later) - len(later.lstrip()) <= indent:
                    break
                body.append(later.strip())
            return body
    return []


def test_the_step_reader_finds_the_steps_and_their_run_text():
    names = [
        line.split("name: ", 1)[1]
        for step in eval_steps()
        for line in step.splitlines()
        if line.strip().startswith("- name: ")
    ]
    assert "Run DeepEval" in names and "Summarise results" in names
    assert any(
        "uv run deepeval test run evals/ -m smoke" in line
        for line in run_blocks(step_named("Run DeepEval"))
    )


def test_the_eval_checkout_has_the_full_history_to_diff_against_the_base_branch():
    checkout = eval_steps()[0]
    assert "actions/checkout" in checkout
    assert "fetch-depth: 0" in checkout
    assert "persist-credentials: false" in checkout  # the history is not a reason to keep the token


def test_the_changed_files_come_from_a_pull_request_step_that_reads_the_base_through_env():
    steps = eval_steps()
    listing = steps[1]  # right after the checkout
    assert "if: github.event_name == 'pull_request'" in listing
    assert "continue-on-error: true" in listing
    assert "BASE_REF: ${{ github.base_ref }}" in listing
    env_at = listing.index("env:")
    assert listing.index("BASE_REF") > env_at < listing.index("run:")
    (command,) = [line for line in run_blocks(listing) if "git diff" in line]
    assert '"origin/$BASE_REF...HEAD"' in command
    assert "--name-only" in command


def test_a_failed_listing_leaves_no_changed_files_file_to_be_read_as_empty():
    # A redirect creates the file before git runs, and an empty file means "no gate file changed".
    # The list is written to a temporary name and moved into place only after git succeeded (the
    # shell stops at the first failing command), so a failure is reported by the summary as "could
    # not list" instead of as an empty list.
    commands = run_blocks(eval_steps()[1])
    (diff,) = [line for line in commands if "git diff" in line]
    (move,) = [line for line in commands if line.startswith("mv ")]
    assert diff.endswith("> changed-files.tmp")
    assert move == "mv changed-files.tmp changed-files.txt"
    assert commands.index(diff) < commands.index(move)
    assert "--no-renames" in diff  # a moved gate file shows under its old path too


def test_no_run_block_interpolates_the_base_branch_or_the_outcome():
    # An expression inside `run:` is pasted into the shell script; both go through `env:`.
    for step in eval_steps():
        for line in run_blocks(step):
            assert "github.base_ref" not in line
            assert "steps.run.outcome" not in line


def test_the_summary_step_gets_the_outcome_the_changed_files_and_the_full_run_flag():
    run = step_named("Run DeepEval")
    assert "id: run" in run
    summary = step_named("Summarise results")
    assert "RUN_OUTCOME: ${{ steps.run.outcome }}" in summary
    assert "if: always()" in summary
    text = "\n".join(run_blocks(summary))
    assert '--outcome "$RUN_OUTCOME"' in text
    assert "--changed-files changed-files.txt" in text
    assert "--not-a-gate" in text
    # The changed files are passed on pull requests, the full-run flag on every other event.
    assert text.index("--changed-files") < text.index("else") < text.index("--not-a-gate")
    assert '"$EVENT_NAME" = "pull_request"' in text

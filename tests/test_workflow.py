import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

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


def split_steps(lines: list[str]) -> list[str]:
    """The text of each step in `lines`, from its `- ` line to the next step."""
    steps: list[list[str]] = []
    for line in lines:
        if line.startswith("      - "):
            steps.append([])
        if steps:
            steps[-1].append(line)
    return ["\n".join(step) for step in steps]


def eval_steps() -> list[str]:
    """The steps of the eval job."""
    job = eval_job_lines()
    return split_steps(job[job.index("    steps:") + 1 :])


def every_step() -> list[str]:
    """The steps of every job in eval.yml."""
    text = (GITHUB / "workflows" / "eval.yml").read_text(encoding="utf-8")
    return split_steps(text.splitlines())


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
    (command,) = [line for line in run_blocks(listing) if "--name-only" in line]
    assert '"origin/$BASE_REF...HEAD"' in command


def test_a_failed_listing_leaves_no_changed_files_file_to_be_read_as_empty():
    # A redirect creates the file before git runs, and an empty file means "no gate file changed".
    # The list is written to a temporary name and moved into place only after git succeeded (the
    # shell stops at the first failing command), so a failure is reported by the summary as "could
    # not list" instead of as an empty list.
    commands = run_blocks(eval_steps()[1])
    (diff,) = [line for line in commands if "--name-only" in line]
    (move,) = [line for line in commands if line.startswith("mv ")]
    assert diff.endswith("> changed-files.tmp")
    assert move == "mv changed-files.tmp changed-files.txt"
    assert commands.index(diff) < commands.index(move)
    assert "--no-renames" in diff  # a moved gate file shows under its old path too
    # git quotes a path with non-ASCII characters ("docs/\303\251.md") unless told not to, and a
    # quoted path in a gate directory would not match its prefix.
    assert diff.startswith("git -c core.quotePath=false diff ")


def test_no_run_block_interpolates_the_base_branch_or_the_outcome():
    # An expression inside `run:` is pasted into the shell script; both go through `env:`.
    for step in eval_steps():
        for line in run_blocks(step):
            assert "github.base_ref" not in line
            assert "steps.run.outcome" not in line


def test_no_run_block_in_the_workflow_contains_an_expression_at_all():
    # Every `${{ }}` reaches the shell through `env:`, in every job, so nothing a pull request
    # controls (a branch name, a title) is pasted into a script.
    blocks = [line for step in every_step() for line in run_blocks(step)]
    assert any("uv run" in line for line in blocks)  # the scan reads the run blocks
    assert [line for line in blocks if "${{" in line] == []


def test_the_scan_of_run_blocks_sees_an_expression_in_a_step():
    step = "      - name: x\n        run: |\n          echo ${{ github.head_ref }}\n        env:\n"
    assert run_blocks(step) == ["echo ${{ github.head_ref }}"]


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


def continue_on_error(step: str, event: str) -> bool:
    """Whether a step's `continue-on-error` holds for `event`. It reads the forms the workflow
    uses: true, false, or github.event_name compared with one quoted event name."""
    (value,) = [
        line.split(":", 1)[1].strip()
        for line in step.splitlines()
        if line.strip().startswith("continue-on-error:")
    ]
    if value in ("true", "false"):
        return value == "true"
    match = re.fullmatch(r"\$\{\{ github\.event_name (==|!=) '(\w+)' \}\}", value)
    assert match, f"continue-on-error this test cannot read: {value}"
    return (event == match[2]) == (match[1] == "==")


def test_a_full_run_never_fails_on_a_score_and_a_pull_request_still_fails_closed():
    run = step_named("Run DeepEval")
    assert not continue_on_error(run, "pull_request")
    assert continue_on_error(run, "schedule") and continue_on_error(run, "workflow_dispatch")


def errored(name: str) -> dict:
    return {
        "name": name,
        "metricsData": [
            {"name": "Cites retrieved passages", "score": 1.0, "success": True, "threshold": 1.0},
            {"name": "Faithfulness", "score": None, "success": False, "error": "Error code: 401"},
        ],
    }


def strictly_failing(name: str) -> dict:
    return {
        "name": name,
        "metricsData": [{"name": "Faithfulness", "score": 0.4, "success": False, "threshold": 0.7}],
    }


def passing(name: str) -> dict:
    return {
        "name": name,
        "metricsData": [{"name": "Faithfulness", "score": 0.9, "success": True, "threshold": 0.7}],
    }


RESULTS = {
    "no results folder": None,
    "every case errored": {"testCases": [errored("in-001"), errored("in-002")]},
    "some cases strictly failed": {
        "testCases": [strictly_failing("in-001"), errored("in-002"), passing("in-003")]
    },
}


def summary_step_turns_the_job_red(tmp_path: Path, event: str, results: dict | None) -> bool:
    """Runs the shell of the "Summarise results" step the way the runner does (`bash -e`), in a
    folder whose results/ holds `results` (no folder when it is None), with `uv run python` standing
    for this interpreter, and says whether the step leaves the job red: it exits non-zero and its
    continue-on-error does not hold for `event`."""
    step = step_named("Summarise results")
    if results is not None:
        (tmp_path / "results").mkdir()
        (tmp_path / "results" / "test_run_20261026_140000.json").write_text(json.dumps(results))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(
        '#!/bin/sh\n[ "$1 $2" = "run python" ] || exit 99\nshift 2\nexec "$PYTHON" "$@"\n'
    )
    uv.chmod(0o755)
    script = tmp_path / "step.sh"
    script.write_text("\n".join(run_blocks(step)) + "\n")
    summary = tmp_path / "step-summary.md"
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "PYTHON": sys.executable,
        "PYTHONPATH": str(GITHUB.parent),
        "EVENT_NAME": event,
        "RUN_OUTCOME": "failure",
        "GITHUB_STEP_SUMMARY": str(summary),
    }
    result = subprocess.run(
        ["bash", "-e", str(script)], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert result.returncode != 99, "the step no longer runs `uv run python`"
    assert summary.read_text().startswith("**"), result.stderr  # the verdict was written
    return result.returncode != 0 and not continue_on_error(step, event)


@pytest.mark.parametrize("event", ["schedule", "workflow_dispatch"])
@pytest.mark.parametrize(
    ("outcome", "red"),
    [
        ("no results folder", True),
        ("every case errored", True),
        ("some cases strictly failed", False),
    ],
)
def test_a_full_run_is_red_only_when_it_scored_no_case(tmp_path, event, outcome, red):
    # 2026-10-26: a rejected key left all 65 cases unscored, and the full run's job was green.
    assert summary_step_turns_the_job_red(tmp_path, event, RESULTS[outcome]) is red


@pytest.mark.parametrize("outcome", list(RESULTS))
def test_the_summary_step_never_turns_a_pull_request_red(tmp_path, outcome):
    # The pull request's check is the "Run DeepEval" step's, which fails closed on its own.
    assert summary_step_turns_the_job_red(tmp_path, "pull_request", RESULTS[outcome]) is False

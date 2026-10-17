"""The CI step that decides whether the paid evals can run. It must fail closed: a run that
cannot score the app is a red check, never a green one."""

import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "check-eval-keys.sh"
REPO = "rcastaneda-dev/deepeval-cardiology-rag"


def run(tmp_path, **env):
    output = tmp_path / "github_output"
    output.write_text("")
    base = {
        "PATH": "/usr/bin:/bin",
        "GITHUB_OUTPUT": str(output),
        "GITHUB_REPOSITORY": REPO,
        "EVENT_NAME": "pull_request",
        "HEAD_REPO": REPO,
        "ANTHROPIC_API_KEY": "a-key",
        "OPENAI_API_KEY": "o-key",
    }
    result = subprocess.run(
        ["bash", str(SCRIPT)], env={**base, **env}, capture_output=True, text=True, check=False
    )
    return result, output.read_text()


def test_both_keys_present_lets_the_evals_run(tmp_path):
    result, output = run(tmp_path)
    assert result.returncode == 0
    assert "available=true" in output


@pytest.mark.parametrize("missing", ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"])
def test_a_missing_key_on_a_same_repo_pull_request_fails_the_check(tmp_path, missing):
    result, output = run(tmp_path, **{missing: ""})
    assert result.returncode == 1
    assert "available=true" not in output
    assert "::error::" in result.stdout
    assert "secrets" in result.stdout


def test_a_fork_pull_request_fails_closed_and_says_why(tmp_path):
    no_keys = {"ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": ""}
    result, output = run(tmp_path, HEAD_REPO="someone/fork", **no_keys)
    assert result.returncode == 1
    assert "available=true" not in output
    assert "fork" in result.stdout
    assert "branch in this repository" in result.stdout


@pytest.mark.parametrize("event", ["schedule", "workflow_dispatch"])
def test_missing_keys_fail_scheduled_and_manual_runs(tmp_path, event):
    result, _ = run(tmp_path, EVENT_NAME=event, HEAD_REPO="", OPENAI_API_KEY="")
    assert result.returncode == 1
    assert "::error::" in result.stdout


def test_a_key_that_is_only_whitespace_counts_as_missing(tmp_path):
    result, _ = run(tmp_path, OPENAI_API_KEY="   ")
    assert result.returncode == 1

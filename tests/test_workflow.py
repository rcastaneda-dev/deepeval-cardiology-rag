from pathlib import Path

GITHUB = Path(__file__).resolve().parents[1] / ".github"


def github_files() -> list[Path]:
    return [path for path in sorted(GITHUB.rglob("*")) if path.is_file()]


def test_the_scan_covers_the_workflow_and_the_scripts_it_runs():
    names = {path.name for path in github_files()}
    assert {"eval.yml", "check-eval-keys.sh"} <= names


def test_the_gate_never_replays_recorded_answers():
    # Replaying recorded answers is for measuring the judge; the gate must run the live app.
    # Any file under .github could set it for the workflow, not only eval.yml.
    offenders = [
        str(path.relative_to(GITHUB))
        for path in github_files()
        if "EVAL_ANSWERS" in path.read_text(encoding="utf-8", errors="replace")
    ]
    assert offenders == []

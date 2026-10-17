from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "eval.yml"


def test_the_gate_never_replays_recorded_answers():
    # Replaying recorded answers is for measuring the judge; the gate must run the live app.
    assert "EVAL_ANSWERS" not in WORKFLOW.read_text(encoding="utf-8")

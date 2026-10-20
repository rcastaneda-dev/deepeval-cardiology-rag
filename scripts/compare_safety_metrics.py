"""Score the ten safety answers with the custom GEval rubric and with DeepEval's NonAdviceMetric.

Usage: uv run python scripts/compare_safety_metrics.py

Runs the app on every safety golden case (Claude generates), scores each answer with both
metrics (the configured judge, gpt-4o-mini by default) at the same threshold, and prints a
Markdown table followed by the full answers so a person can read them and decide which metric
agreed better. A findings helper: it costs a few cents and is not part of the quality gate.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

from deepeval.metrics import NonAdviceMetric

from cardio_rag.config import Settings, judge_model
from cardio_rag.wiring import build_pipeline

# Run as a file (`python scripts/compare_safety_metrics.py`), so make the repo root importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.cases import Golden, load_goldens  # noqa: E402
from evals.harness import build_test_case  # noqa: E402
from evals.metrics import judge, metric_specs_for  # noqa: E402
from evals.thresholds import THRESHOLDS  # noqa: E402


@dataclass(frozen=True)
class Row:
    case_id: str
    rubric_score: float
    rubric_passed: bool
    non_advice_score: float
    non_advice_passed: bool
    answer: str


def format_table(rows: list[Row], threshold: float) -> str:
    lines = [
        f"Both metrics pass at score >= {threshold}.",
        "",
        "| Case | Safety rubric (GEval) | NonAdviceMetric | Agree |",
        "|---|---:|---:|:---:|",
    ]
    for row in rows:
        agree = "yes" if row.rubric_passed == row.non_advice_passed else "no"
        lines.append(
            f"| {row.case_id} | {_cell(row.rubric_score, row.rubric_passed)} "
            f"| {_cell(row.non_advice_score, row.non_advice_passed)} | {agree} |"
        )
    rubric = sum(row.rubric_passed for row in rows)
    non_advice = sum(row.non_advice_passed for row in rows)
    lines += ["", f"Passed: rubric {rubric}/{len(rows)}, NonAdviceMetric {non_advice}/{len(rows)}."]
    return "\n".join(lines)


def _cell(score: float, passed: bool) -> str:
    return f"{score:.2f} {'pass' if passed else 'FAIL'}"


def score_case(golden: Golden, rag, threshold: float, model) -> tuple[Row, float]:
    # The safety goldens have no source excerpt, so there is no context to build from a corpus.
    test_case = build_test_case(golden, rag, corpus=None)
    # The gate's own rubric, at its threshold, judging once like NonAdviceMetric (the gate takes
    # the median of three judgings).
    rubric = dict(metric_specs_for(golden, repeats=1))["safety"]
    non_advice = NonAdviceMetric(advice_types=["medical"], threshold=threshold, model=model)
    rubric.measure(test_case, _show_indicator=False)
    non_advice.measure(test_case, _show_indicator=False)
    cost = (rubric.evaluation_cost or 0.0) + (non_advice.evaluation_cost or 0.0)
    row = Row(
        case_id=golden.id,
        rubric_score=rubric.score,
        rubric_passed=rubric.is_successful(),
        non_advice_score=non_advice.score,
        non_advice_passed=non_advice.is_successful(),
        answer=test_case.actual_output,
    )
    return row, cost


def main() -> int:
    threshold = THRESHOLDS["safety"]
    model = judge(judge_model())
    rag = build_pipeline(Settings.from_env())
    rows: list[Row] = []
    total_cost = 0.0
    for golden in load_goldens("safety"):
        row, cost = score_case(golden, rag, threshold, model)
        rows.append(row)
        total_cost += cost
    print(f"Judge: {judge_model()}\n")
    print(format_table(rows, threshold))
    print(f"\nJudge cost reported by DeepEval: ${total_cost:.4f}\n\n## Answers\n")
    for row in rows:
        print(f"### {row.case_id}\n\n{row.answer}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

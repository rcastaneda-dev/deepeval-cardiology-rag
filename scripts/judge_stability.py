"""Judge each recorded answer several times, to see how often the judge changes its verdict.

Usage: uv run python scripts/judge_stability.py --snapshot PATH --times K --out PATH
           [--cases ID,ID] [--keys KEY,KEY] [--known-bad PATH]

Replays the answers and retrieved passages from a snapshot (scripts/record_answers.py), judges
every check K times with new metric objects each time, and writes the scores and verdicts to
--out as JSON, then prints them as Markdown with the estimated false-alarm rate of the
pull-request gate. Each check is measured K times on every case (a DeepEval metric makes
several judge calls per measure), so a run is paid; at most 8 cases are judged at once. It is
not part of the quality gate.

--known-bad names a JSON list of {"case_id", "kind", "answer", "must_fail": [metric keys]}.
Each entry is judged like the recorded case, with its answer replaced, on the keys in
must_fail only; an answer the judge passes in most judgings is a miss, and one with any
errored judging is unjudged. --cases and --keys narrow the known-bad entries as well as the
recorded answers.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Run as a file (`python scripts/judge_stability.py`), so make the repo root importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cardio_rag.config import judge_model  # noqa: E402
from evals.cases import Golden, load_goldens  # noqa: E402
from evals.harness import build_test_case  # noqa: E402
from evals.metrics import is_strict, metric_specs_for  # noqa: E402
from evals.snapshot import ReplayRag, load, mismatches  # noqa: E402
from evals.stability import known_bad_problems, summarize  # noqa: E402
from evals.thresholds import METRIC_KEYS, THRESHOLDS  # noqa: E402

MAX_CASES_AT_ONCE = 8


async def _judge(golden, test_case, keys, times, specs_for, gate) -> tuple[dict, float]:
    """`times` judgings of `keys` on one test case, each with new metric objects.

    A call that raises is a failed judging with its message as the error, and so is a metric
    that sets its own error: the gate fails both. Returns the scores, verdicts and errors per
    key, and the cost DeepEval reported.
    """
    found = {key: {"scores": [], "passed": [], "errors": []} for key in keys}
    cost = 0.0
    async with gate:
        for _ in range(times):
            metrics = dict(specs_for(golden))
            for key in keys:
                metric = metrics[key]
                try:
                    await metric.a_measure(test_case, _show_indicator=False)
                    score, passed, error = metric.score, bool(metric.is_successful()), metric.error
                except Exception as exc:  # any judge failure is a data point, not a crash
                    score, passed, error = None, False, f"{type(exc).__name__}: {exc}"
                found[key]["scores"].append(score)
                found[key]["passed"].append(passed)
                found[key]["errors"].append(error)
                cost += metric.evaluation_cost or 0.0
    return found, cost


async def run(
    snapshot: dict, goldens: list[Golden], *, times: int, specs_for, keys=None, known_bad=()
) -> dict:
    """Judge every golden's recorded answer, and every known-bad entry, `times` times.

    `keys` limits the checks (None means all of a case's own); `known_bad` entries must name
    cases in `goldens`. Returns the rows, known_bad and cost of the results file.
    """
    gate = asyncio.Semaphore(MAX_CASES_AT_ONCE)
    by_id = {golden.id: golden for golden in goldens}
    recorded = {case["id"]: case for case in snapshot["cases"]}

    def wanted(candidates):
        return [key for key in candidates if keys is None or key in keys]

    rag = ReplayRag(snapshot)
    jobs = []  # (known-bad entry or None, golden, keys to judge, test case)
    for golden in goldens:
        job_keys = wanted(METRIC_KEYS[golden.category])
        if job_keys:
            jobs.append((None, golden, job_keys, build_test_case(golden, rag)))
    for entry in known_bad:
        golden = by_id[entry["case_id"]]
        job_keys = wanted(entry["must_fail"])
        if job_keys:
            replaced = ReplayRag({"cases": [{**recorded[golden.id], "answer": entry["answer"]}]})
            jobs.append((entry, golden, job_keys, build_test_case(golden, replaced)))

    outcomes = await asyncio.gather(
        *(
            _judge(golden, test_case, job_keys, times, specs_for, gate)
            for _, golden, job_keys, test_case in jobs
        )
    )

    rows, bad, cost = [], [], 0.0
    for (entry, golden, job_keys, _), (found, spent) in zip(jobs, outcomes, strict=True):
        cost += spent
        for key in job_keys:
            if entry is None:
                flags = {"strict": is_strict(golden, key), "smoke": golden.smoke}
                rows.append({"case_id": golden.id, "key": key, **flags, **found[key]})
            else:
                label = {"case_id": golden.id, "kind": entry["kind"], "key": key}
                bad.append({**label, **found[key]})
    return {"rows": rows, "known_bad": bad, "cost": cost}


def _csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _refuse(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--snapshot", type=Path, required=True, help="answers to judge")
    parser.add_argument("--times", type=int, required=True, help="judgings of each check")
    parser.add_argument("--out", type=Path, required=True, help="where to write the results")
    parser.add_argument("--cases", type=_csv, help="only these case ids")
    parser.add_argument("--keys", type=_csv, help="only these metric keys")
    parser.add_argument("--known-bad", type=Path, help="JSON list of known-bad answers")
    args = parser.parse_args(argv)

    if args.times < 1:
        return _refuse("--times must be at least 1")
    goldens = load_goldens()
    unknown_cases = sorted(set(args.cases or []) - {golden.id for golden in goldens})
    if unknown_cases:
        return _refuse(f"--cases names no golden case: {', '.join(unknown_cases)}")
    unknown_keys = sorted(set(args.keys or []) - set(THRESHOLDS))
    if unknown_keys:
        return _refuse(f"--keys names no metric: {', '.join(unknown_keys)}")

    try:
        snapshot = load(args.snapshot)
        known_bad = json.loads(args.known_bad.read_text(encoding="utf-8")) if args.known_bad else []
    except (OSError, ValueError) as error:
        return _refuse(str(error))
    problems = mismatches(snapshot, goldens)
    if problems:
        return _refuse(f"{args.snapshot} does not match the dataset:\n" + "\n".join(problems))
    problems = known_bad_problems(known_bad, goldens)
    if problems:
        return _refuse(f"{args.known_bad} cannot be judged:\n" + "\n".join(problems))

    selected = [g for g in goldens if args.cases is None or g.id in args.cases]
    chosen = {golden.id for golden in selected}
    results = asyncio.run(
        run(
            snapshot,
            selected,
            times=args.times,
            specs_for=metric_specs_for,
            keys=args.keys,
            known_bad=[entry for entry in known_bad if entry["case_id"] in chosen],
        )
    )
    results = {
        "snapshot": str(args.snapshot),
        "times": args.times,
        "judge": judge_model(),
        **results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(results, indent=1, ensure_ascii=False)
    args.out.write_text(text + "\n", encoding="utf-8")
    print(summarize(results))
    print(f"\nWrote {len(results['rows'])} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

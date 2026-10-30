"""Run the checks that need no judge over the answers of a snapshot: no API call, no cost.

Usage: uv run python scripts/check_answers.py --snapshot PATH [--keys KEY,KEY] [--out PATH]

For every case in a snapshot (scripts/record_answers.py; a file of samples is read sample by
sample) it runs each check of `evals.thresholds.DETERMINISTIC` that the case's category runs
(`keys_for`: cites_passages, exact_refusal, no_symptom_diagnosis and its report-only broad
reading diagnosis_wording, and emergency_first on the emergency cases), against the passages the
case retrieved. It prints how many answers each check
read and how many failed, lists every failure with its reason, writes the same to --out as JSON
(the snapshot's commit, whether it was dirty, one row per answer and check, and the summary), and
exits 1 if a check that gates failed, 2 if the snapshot cannot be read. A failure of a report-only
check (REPORT_ONLY in evals/thresholds.py) is listed as "WARN", marked "(report-only)", and does
not set the exit code, as it never fails a case in the gate.
"""

import argparse
import json
import sys
from pathlib import Path

# Run as a file (`python scripts/check_answers.py`), so make the repo root importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepeval.test_case import LLMTestCase  # noqa: E402

from evals.cases import load_goldens  # noqa: E402
from evals.metrics import deterministic_check  # noqa: E402
from evals.snapshot import load  # noqa: E402
from evals.thresholds import DETERMINISTIC, REPORT_ONLY, keys_for  # noqa: E402


def check_all(snapshot: dict, goldens: list, *, keys: list[str] | None = None) -> list[dict]:
    """One row per answer and check: its id, sample number (None for a single answer), key,
    whether it passed and the check's reason. ValueError names an answer whose case is not in
    `goldens`."""
    by_id = {golden.id: golden for golden in goldens}
    unknown = sorted({case["id"] for case in snapshot["cases"]} - set(by_id))
    if unknown:
        raise ValueError(f"the snapshot has answers for cases the dataset lacks: {unknown}")
    rows = []
    for case in snapshot["cases"]:
        golden = by_id[case["id"]]
        test_case = LLMTestCase(
            input=case["question"],
            actual_output=case["answer"],
            metadata={"chunk_ids": [chunk["id"] for chunk in case["chunks"]]},
        )
        for key in keys_for(golden):
            if key not in DETERMINISTIC or (keys is not None and key not in keys):
                continue
            check = deterministic_check(key)
            check.measure(test_case)
            rows.append(
                {
                    "id": case["id"],
                    "sample": case.get("sample"),
                    "key": key,
                    "passed": bool(check.is_successful()),
                    "reason": check.reason,
                }
            )
    return rows


def summarize(rows: list[dict]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for row in rows:
        counts = summary.setdefault(row["key"], {"checked": 0, "failed": 0})
        counts["checked"] += 1
        counts["failed"] += 0 if row["passed"] else 1
    return summary


def _refuse(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--snapshot", type=Path, required=True, help="the answers to check")
    parser.add_argument("--keys", help="only these checks, comma separated")
    parser.add_argument("--out", type=Path, help="where to write the results as JSON")
    args = parser.parse_args(argv)

    keys = [item.strip() for item in args.keys.split(",") if item.strip()] if args.keys else None
    unjudged = sorted(set(keys or []) - DETERMINISTIC)
    if unjudged:
        return _refuse(
            f"--keys names checks that need a judge or no check: {', '.join(unjudged)}; "
            f"the checks are {', '.join(sorted(DETERMINISTIC))}"
        )
    try:
        snapshot = load(args.snapshot)
        rows = check_all(snapshot, load_goldens(), keys=keys)
    except (OSError, ValueError) as error:
        return _refuse(str(error))

    summary = summarize(rows)
    failures = [row for row in rows if not row["passed"]]
    results = {
        "snapshot": str(args.snapshot),
        "created": snapshot.get("created"),
        "git_sha": snapshot.get("git_sha"),
        "dirty": snapshot.get("dirty"),
        "summary": summary,
        "results": rows,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(results, indent=1, ensure_ascii=False)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(f"{args.snapshot}: {len(snapshot['cases'])} answers, git_sha {snapshot.get('git_sha')}")
    print("| Check | Answers read | Failed |\n|---|---:|---:|")
    for key, counts in sorted(summary.items()):
        print(f"| {_label(key)} | {counts['checked']} | {counts['failed']} |")
    for row in failures:
        sample = "" if row["sample"] is None else f" sample {row['sample']}"
        verdict = "WARN" if row["key"] in REPORT_ONLY else "FAIL"
        print(f"{verdict} {row['id']}{sample} {_label(row['key'])}: {row['reason']}")
    return 1 if any(row["key"] not in REPORT_ONLY for row in failures) else 0


def _label(key: str) -> str:
    return f"{key} (report-only)" if key in REPORT_ONLY else key


if __name__ == "__main__":
    raise SystemExit(main())

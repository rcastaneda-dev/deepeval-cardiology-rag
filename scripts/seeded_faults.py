"""Apply each seeded fault to the live app and see which checks of the pull-request gate catch it.

Usage: DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE=600 \\
           uv run python scripts/seeded_faults.py --out docs/evidence/seeded-faults-<date>.md
           [--faults NAME,NAME] [--max-cost USD] [--notes PATH]
       uv run python scripts/seeded_faults.py --from-json PATH --out PATH [--notes PATH]
           [--rescore CHECK]   (offline)

First it runs the control: the same smoke goldens with no fault on, judged the same way, which is
the first row of the matrix. A check that fails a case on the healthy app is marked "fails on the
healthy app" and never counted as catching a fault on that case. Then, for each fault of
evals/faults.py, it asks the app the smoke goldens once with the fault on, then
judges the answers with the checks the gate builds (evals.metrics.metric_specs_for, with the
gate's own repeats and judges), and writes the matrix of which checks failed which fault to --out
and every case's answer, failing checks and scores to a JSON file beside it. The run is paid (the
judges, and Claude for the answers), about $0.15 a pass (ten passes), and it is not part of the
quality gate. The JSON is rewritten after every fault, so a crash keeps the faults already paid
for. Before each pass the script checks the cost DeepEval has reported so far (the generator's cost
is not in it) and stops starting passes once it reaches --max-cost (default 2.00).

It refuses to run with JUDGE_REPEATS set, so the report's "gate's own repeats" is true, and
it asks the app one smoke question with no fault on first (one generator call) and stops if
that raises, so a rejected key is not read as nine caught faults.

--from-json renders the report again from a JSON file that run wrote, with no API call.
--notes PATH appends a hand-written Markdown file (such as a "Known gaps" section) to the end of
the report, on a run and on --from-json, which would otherwise write the whole report afresh.
--rescore CHECK, with --from-json, applies a check that needs no judge and was added after the run
(such as no_symptom_diagnosis) to the stored answers, with no API call, and writes the JSON beside
--out again with a `rescored_offline` entry per fault; the matrix marks what it added as
"re-scored offline". What the run measured stays as it was.
"""

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

# Run as a file (`python scripts/seeded_faults.py`), so make the repo root importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cardio_rag.config import CORPUS_DIR, Settings  # noqa: E402
from cardio_rag.index import load_corpus  # noqa: E402
from cardio_rag.wiring import build_pipeline  # noqa: E402
from evals.cases import load_goldens  # noqa: E402
from evals.faults import CONTROL, FAULTS, matrix_markdown, rescore_offline, run_fault  # noqa: E402
from scripts.record_answers import _git_dirty, _git_sha  # noqa: E402

DEFAULT_MAX_COST = 2.00


def run_all(names, goldens, *, rag, corpus, max_cost, run=run_fault, on_result=None) -> dict:
    """Run the faults `names` in order. Before each one, stop if the cost reported so far has
    reached `max_cost`: the faults left are None in the result. `on_result` gets the results so
    far after every fault. An exception from `run` ends the run and is raised."""
    results: dict = {name: None for name in names}
    spent = 0.0
    for name in names:
        if spent >= max_cost:
            print(f"spend guard: ${spent:.2f} reported, so {name} is not run", file=sys.stderr)
            continue
        result = run(name, goldens, rag=rag, corpus=corpus)
        results[name] = result
        spent += result["cost"]
        print(
            f"{name}: {_failed(result)} of {len(result['cases'])} cases failed, "
            f"${result['cost']:.3f} ({result['seconds']:.0f} s), ${spent:.2f} so far",
            file=sys.stderr,
        )
        if on_result:
            on_result(results)
    return results


def _failed(result: dict) -> int:
    return sum(1 for case in result["cases"] if case["failed"] or case["errors"])


def render(data: dict, notes: str | None = None) -> str:
    """The Markdown report for the data main writes to the JSON file, with the hand-written
    `notes` (Markdown) after it when there are any."""
    faults = data["faults"]
    skipped = [name for name, result in faults.items() if result is None]
    rescored = next(
        (
            result["rescored_offline"]
            for result in faults.values()
            if result and result.get("rescored_offline")
        ),
        None,
    )
    minutes = data["seconds"] / 60
    lines = [
        "# Seeded faults: which checks of the pull-request gate catch each one",
        "",
        f"Run on {data['created']} at commit {data['git_sha']}"
        f"{' (uncommitted changes)' if data['dirty'] else ''}, generator "
        f"{data['generator_model']}, top_k {data['top_k']}. Each fault of `evals/faults.py` was "
        f"applied in-process to the live app, the {len(data['smoke_cases'])} smoke cases were "
        "asked once with it on (" + ", ".join(data["smoke_cases"]) + "), and the answers were "
        "judged with the checks the gate builds (`metric_specs_for`, `JUDGE_REPEATS` unset). A "
        "case fails a fault when a check that counts fails or errors, as `assert_test` would; "
        "Bias and Answer Relevancy are report-only and only warn. An error (an app that raised, "
        "a check that raised or set an error) counts as caught and is labelled `error`. The "
        "control is the same pass with no fault on, the first row of the matrix: a check that "
        'fails a case on the healthy app is marked "fails on the healthy app" and is not '
        "counted as catching a fault on that case."
        + (
            f' Rows marked "re-scored offline" count a check added after the run '
            f"(`{rescored['check']}`, {rescored['date']}) applied to the stored answers with no "
            "API call; the JSON file keeps what the run measured in `failed` and the re-score "
            "in `rescored_offline`."
            if rescored
            else ""
        ),
        "",
        f"DeepEval reported ${data['spent']:.2f} for the judgings (the generator's cost is not in "
        f"that number); the run took {minutes:.0f} minutes. The answer of every case is in the "
        f"JSON file beside this report.",
        "",
        "## Matrix",
        "",
        matrix_markdown(faults),
    ]
    if skipped:
        why = (
            f"because the spend guard (${data['max_cost']:.2f}) was reached"
            if data["spent"] >= data["max_cost"]
            else "because the run stopped before them"
        )
        lines += ["", f"Not run, {why}: " + ", ".join(skipped) + "."]
    text = "\n".join(lines) + "\n"
    if notes and notes.strip():
        text += "\n" + notes.strip("\n") + "\n"
    return text


def _csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _refuse(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 2


def _write(data: dict, out: Path, notes: str | None = None) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=1, ensure_ascii=False)
    out.with_suffix(".json").write_text(text + "\n", encoding="utf-8")
    out.write_text(render(data, notes), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="the Markdown report to write")
    parser.add_argument("--faults", type=_csv, help="only these faults (default: all nine)")
    parser.add_argument("--max-cost", type=float, default=DEFAULT_MAX_COST, help="spend guard, USD")
    parser.add_argument("--from-json", type=Path, help="render this results file, no API call")
    parser.add_argument("--notes", type=Path, help="Markdown to append to the report as written")
    parser.add_argument("--rescore", help="with --from-json: apply this no-judge check offline")
    args = parser.parse_args(argv)

    if args.rescore and not args.from_json:
        return _refuse("--rescore re-scores stored results, so it needs --from-json")

    notes = None
    if args.notes:
        try:
            notes = args.notes.read_text(encoding="utf-8")
        except OSError as error:
            return _refuse(f"cannot read --notes {args.notes}: {error}")

    if args.from_json:
        data = json.loads(args.from_json.read_text(encoding="utf-8"))
        if args.rescore:
            try:
                rescore_offline(
                    data["faults"],
                    args.rescore,
                    goldens=load_goldens(),
                    on=date.today().isoformat(),
                )
            except ValueError as error:
                return _refuse(str(error))
            _write(data, args.out, notes)
            print(f"Wrote {args.out} and {args.out.with_suffix('.json')}")
            return 0
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(render(data, notes), encoding="utf-8")
        print(f"Wrote {args.out}")
        return 0

    if "JUDGE_REPEATS" in os.environ:
        return _refuse(
            "JUDGE_REPEATS is set; the faults are judged with the gate's own repeats, so unset it"
        )
    faults = args.faults or list(FAULTS)
    unknown = [name for name in faults if name not in FAULTS]
    if unknown:
        return _refuse(
            f"unknown fault(s) {', '.join(unknown)}; the faults are {', '.join(FAULTS)} "
            f"(the {CONTROL} always runs first)"
        )
    names = [CONTROL, *faults]
    smoke = [golden for golden in load_goldens() if golden.smoke]
    if not smoke:
        return _refuse("the dataset has no smoke cases")

    settings = Settings.from_env()
    data = {
        "created": date.today().isoformat(),
        "git_sha": _git_sha(),
        "dirty": _git_dirty(),
        "generator_model": settings.generator_model,
        "embedding_model": settings.embedding_model,
        "top_k": settings.top_k,
        "smoke_cases": [golden.id for golden in smoke],
        "max_cost": args.max_cost,
        "spent": 0.0,
        "seconds": 0.0,
        "faults": {name: None for name in names},
    }

    def save(results: dict) -> None:
        data["faults"] = results
        done = [result for result in results.values() if result]
        data["spent"] = sum(result["cost"] for result in done)
        data["seconds"] = sum(result["seconds"] for result in done)
        _write(data, args.out, notes)

    print(
        f"{CONTROL} + {len(faults)} faults x {len(smoke)} smoke cases, "
        f"spend guard ${args.max_cost:.2f}"
    )
    rag = build_pipeline(settings, rebuild=True)
    # With no fault on, the app must answer. A rejected key makes every case of every fault raise,
    # and a matrix of "error" for nine faults would read as nine catches.
    try:
        rag.ask(smoke[0].question)
    except Exception as exc:
        return _refuse(
            f"the app cannot answer {smoke[0].id} with no fault on, so no fault was run: "
            f"{type(exc).__name__}: {exc}"
        )
    results = run_all(
        names,
        smoke,
        rag=rag,
        corpus=load_corpus(CORPUS_DIR),
        max_cost=args.max_cost,
        run=run_fault,
        on_result=save,
    )
    save(results)
    print(matrix_markdown(results))
    print(f"\nWrote {args.out} and {args.out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

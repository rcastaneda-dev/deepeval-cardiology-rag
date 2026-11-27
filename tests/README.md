# tests/

The offline unit tests. They check that the **code** is right: the app in `src/cardio_rag/`, and the machinery in [`evals/`](../evals/README.md) and `scripts/` that builds, runs and reports the quality gate. They also check that the repository's **documents** still say what is true.

They do **not** judge the app's answers. That is the job of [`evals/`](../evals/README.md), which calls real models.

**Contents:** [Running them](#running-them) · [Nothing here calls a model](#nothing-here-calls-a-model) · [What is where](#what-is-where) · [**Writing a new test**](#writing-a-new-test): [where it goes](#where-it-goes), [what we depend on](#what-we-depend-on), [rules](#rules-every-test-follows), [file anatomy and a worked example](#anatomy-of-a-test-file), [patterns in use](#patterns-in-use), [patterns to adopt](#patterns-to-adopt), [before you push](#before-you-push)

|  | `tests/` (this folder) | [`evals/`](../evals/README.md) |
|---|---|---|
| Question it answers | Does the code do what it is meant to? | Are the app's answers good enough to merge? |
| Models called | None. Fakes stand in for the generator, the embedder and the judge | Claude generates, OpenAI embeds and judges |
| API keys | Not needed | Both needed |
| Cost and speed | Free, seconds | Paid, minutes to tens of minutes |
| Result | Deterministic | Noisy: a judge can misread a correct answer |
| Run with | `uv run pytest` | `uv run deepeval test run evals/` |
| In CI | `unit tests and lint`, on every push and pull request | `DeepEval (smoke subset)` on pull requests, the full suite weekly and on demand |

## Running them

```bash
uv run pytest -q                              # everything; about ten seconds
uv run pytest tests/test_thresholds.py -q     # one file
uv run pytest -k known_bad -q                 # tests whose name matches
uv run ruff check . && uv run ruff format --check .   # the lint half of the CI job
```

`pyproject.toml` sets `testpaths = ["tests"]`, so a bare `pytest` never reaches `evals/`. The `smoke` marker that `pyproject.toml` declares belongs to the eval cases, not to anything here.

## Nothing here calls a model

Every test builds its inputs by hand: an `LLMTestCase` written inline, a `FakeMetric` or a scripted metric that returns the score it is told to, a fake app, a fake client. The `no_network` fixture in [conftest.py](conftest.py) goes further for the tests that handle metric objects (`test_deterministic.py`, `test_repeat.py`): it removes both API keys and makes any attempt to open a socket fail the test.

Some tests read the **real** files of the repository on purpose: the corpus, the golden dataset, `known_bad.json`, the README, the workflow. They are guards. Editing one of those files can turn a test red, which is the point: the test names what you broke.

## What is where

### The app (`src/cardio_rag/`)

| File | What it checks |
|---|---|
| [test_chunking.py](test_chunking.py) | Sentence-aligned chunking: size limit, whitespace, short and empty text |
| [test_embeddings.py](test_embeddings.py) | Batching, unit-length rows, no call for no text, zero vectors |
| [test_index.py](test_index.py) | Loading the corpus (ids come from file names, empty files are refused) and building the index |
| [test_retriever.py](test_retriever.py) | Cosine ranking, top-k, ties, blank queries |
| [test_generate.py](test_generate.py) | The system prompt's pinned rules (abstain, call 911 first, no diagnosis, clinician advice, passages are data), citation parsing, and a cut-off or empty reply being an error |
| [test_pipeline.py](test_pipeline.py) | `ask` returns the answer together with the chunks it used |
| [test_cli.py](test_cli.py) | The `cardio` command: the disclaimer, the sources line, stripped terminal control sequences |
| [test_config.py](test_config.py) | Settings and the judge-model precedence (`JUDGE_MODEL_<KEY>`, then `JUDGE_MODEL`, then the code default) |

### The data and the golden cases

| File | What it checks |
|---|---|
| [test_corpus.py](test_corpus.py) | `data/corpus/` against `data/SOURCES.md`: one source row per document, complete rows, copyright mentions reviewed |
| [test_fetch_doc.py](test_fetch_doc.py) | `scripts/fetch_doc.py`: only allowed domains, copyright flags |
| [test_cases.py](test_cases.py) | The loader and validator in `evals/cases.py` on small invented datasets |
| [test_dataset.py](test_dataset.py) | The **real** dataset: consistent with the corpus, enough cases per category, a smoke subset that is small and covers every category, a valid `FLAKY` register |
| [test_known_bad.py](test_known_bad.py) | `evals/known_bad.json`: every entry names a real case and applicable checks, and each kind has the shape its name says |
| [test_harness.py](test_harness.py) | `evals/harness.py`: how a golden case and the app's answer become a DeepEval test case |

### The checks, thresholds and judge handling

| File | What it checks |
|---|---|
| [test_deterministic.py](test_deterministic.py) | The checks that need no judge: citations, the exact refusal, 911 first, and the two readings of the owner's diagnosis line |
| [test_no_symptom_diagnosis_table.py](test_no_symptom_diagnosis_table.py) | A regression table, mostly data: one sentence per row, what the owner's rule says of it, and what each reading of the rule does with it |
| [test_metrics.py](test_metrics.py) | `evals/metrics.py`: which metrics each category gets, with which thresholds, judges and repeats |
| [test_thresholds.py](test_thresholds.py) | `evals/thresholds.py`: every metric in use has a threshold, `FLAKY` and `REPORT_ONLY` name real checks, each flake has a reason, evidence and a review within 30 days, and **every judged threshold equals what the calibration rule gives on the recorded judgings** in `evals/snapshots/` |
| [test_repeat.py](test_repeat.py) | `MedianOf`: judge one answer several times and take the median |

### Measuring the judge and the gate

| File | What it checks |
|---|---|
| [test_snapshot.py](test_snapshot.py) | Recording the app's answers and replaying them; a snapshot that does not match the dataset is refused |
| [test_record_answers.py](test_record_answers.py) | `scripts/record_answers.py`: the file it writes, the commit and the dirty flag |
| [test_check_answers.py](test_check_answers.py) | `scripts/check_answers.py`: the no-judge checks run over a snapshot, and its exit codes |
| [test_calibrate.py](test_calibrate.py) | The threshold rule in `evals/calibrate.py`: the margin, the floor, the ceiling |
| [test_stability.py](test_stability.py) | Flip rates, the median-of-three arithmetic and the false-alarm estimate in `evals/stability.py` |
| [test_judge_stability.py](test_judge_stability.py) | `scripts/judge_stability.py`: new metric objects for every judging, one row per case and check |
| [test_faults.py](test_faults.py) | The nine seeded faults in `evals/faults.py` and the matrix of which checks caught them |
| [test_seeded_faults_script.py](test_seeded_faults_script.py) | `scripts/seeded_faults.py`: the spend guard, the files it writes, the offline re-render |
| [test_compare_safety_metrics.py](test_compare_safety_metrics.py) | The comparison table in `scripts/compare_safety_metrics.py` |

### Reporting and CI

| File | What it checks |
|---|---|
| [test_report.py](test_report.py) | `evals/report.py`: the job-summary table, failures, cost, the gate verdict, expired flakes |
| [test_workflow.py](test_workflow.py) | `.github/workflows/eval.yml`: the gate never replays recorded answers, never sets judge settings, no run block interpolates untrusted input, a full run is red only when it scored no case |
| [test_check_eval_keys.py](test_check_eval_keys.py) | `.github/scripts/check-eval-keys.sh` fails closed: a missing key, a fork or a blank key is a red check, never a skip |

### The documents

| File | What it checks |
|---|---|
| [test_promises.py](test_promises.py) | `evals/promises.py`: the promises P1 to P18 each name a check that exists, and the traceability table lists every one |
| [test_readme.py](test_readme.py) | The README names the corpus sources, carries the disclaimer and the QA-report link before its first section, and states the limits |
| [test_evidence_links.py](test_evidence_links.py) | Every `evals/snapshots/...` file named in a comment or a document exists |

## Writing a new test

This is the guide for adding a test that looks like the ones already here. Every rule comes from the suite as it is, with a file to copy from. One section, "Patterns to adopt", is not descriptive: it names habits the suite has not applied consistently yet, and says so.

**In short:**

1. Put the test in the file of the module it covers, or in a new `test_<module>.py` if the module has none.
2. Name it as a sentence that says what must hold.
3. Build every input by hand, and use a fake for anything that would call a model or the network (`fake_rag` stands in for the whole pipeline).
4. Take paths from `tmp_path`, dates as a parameter and settings from `monkeypatch`.
5. Run your file, then the whole suite, then ruff, and watch the test fail once before you trust it.

### Where it goes

| You want to check that... | It belongs in |
|---|---|
| A function or class in `src/cardio_rag/`, `evals/` or `scripts/` does what it should | `tests/test_<module>.py`, here |
| A document, a dataset, a threshold or a CI file still says what is true | A guard in `tests/` (see the guard pattern below) |
| The app's **answers** are good | Not here: a golden case or a check in [`evals/`](../evals/README.md) |

The layout is flat: one `test_<module>.py` per module under test, no subfolders, no `__init__.py` and no test classes. Add to the existing file when the module already has one. Scripts follow the same rule (`test_check_answers.py` covers `scripts/check_answers.py`); `test_seeded_faults_script.py` carries a `_script` suffix to tell it from `test_faults.py`, which covers the faults themselves in `evals/faults.py`.

`pyproject.toml` puts the repository root on the import path, so `evals` and `scripts` import directly, and `cardio_rag` comes from `src/` through `uv sync`. Group imports the way ruff sorts them: standard library, then third-party (`pytest`, `deepeval`, `numpy`), then first-party (`cardio_rag`, `evals`, `scripts`).

### What we depend on

| Tool | Use it for |
|---|---|
| pytest ≥ 9.1.1 (dev group) | The only test framework: plain `assert`, fixtures, `parametrize`, `pytest.raises(..., match=...)`, `monkeypatch`, `tmp_path`, `capsys` |
| ruff | Lint rules `E, F, I, UP, B`, 100 columns, and `ruff format`. CI runs both `check` and `format --check` |
| DeepEval 4.2.9 (pinned) | To build inputs and fake metrics: `LLMTestCase`, `BaseMetric`, `GEval`, `assert_test`. Never to judge anything for real |
| numpy | `np.testing.assert_allclose` for vectors |
| Standard library | `types.SimpleNamespace` for SDK-shaped fakes, `asyncio.run` for async code, `json`, `pathlib`, `datetime`; `subprocess` only where the thing under test is not Python |
| uv | `uv run pytest`. CI installs exactly what `uv.lock` pins (`uv sync --frozen`), so a new test dependency is a change to `pyproject.toml` and the lock file |

The suite uses none of `unittest.mock`, `pytest-mock`, `hypothesis`, `freezegun` or a recording library such as `vcr`, and has no test classes. The doubles and patterns below do those jobs, so follow suit, and say why in the pull request if you need one of them.

`pytest-asyncio`, `pytest-xdist`, `pytest-rerunfailures` and `pytest-repeat` are installed, but only because DeepEval depends on them. No test uses them, so do not rely on them either: drive async code with `asyncio.run`, and treat a test that needs a rerun to pass as a bug in the test.

### Rules every test follows

- **Offline.** No network and no API key (see [Nothing here calls a model](#nothing-here-calls-a-model)). A test that builds a DeepEval metric object takes the `no_network` fixture. `GEval` needs a key to be *built*, so such a file also sets a dummy one, as `test_repeat.py` does with `"test-key-never-used"`.
- **Deterministic.** No sleeping on a real clock and no dependence on today's date. A function that needs the date takes `today=` and the test passes it. Where a test must reason about dates, it compares dates the data holds, never today's: [`flake_record_problems`](test_thresholds.py) says why, a calendar day must not turn the unit tests red.
- **Nothing is written outside `tmp_path`.** Tests read real repository files on purpose (the corpus, the dataset, the README, the workflow), with `REPO = Path(__file__).resolve().parents[1]`, and never write to them.
- **State is restored.** Change environment variables, settings, module attributes and registries only through `monkeypatch`, so the change is undone when the test ends.
- **Optional libraries stay optional.** `scripts/fetch_doc.py` imports `trafilatura` inside the function that needs it, and its test passes where the library is missing. Do the same for anything heavy.
- **An expectation is evidence, not a dial.** Do not edit an expected value, a table row or a label to make a change pass. Decide whether the code or the expectation is wrong, and say why. The regression table says the same in its own docstring.

Some of these rules are pinned by tests of their own. A `FLAKY` entry that is invalid, a smoke subset that grows past its limit, or a judged threshold that no longer matches its recorded judgings fails `test_thresholds.py` or `test_dataset.py` without any new test. To move a judged threshold you re-judge, label the answers and re-run `python -m evals.calibrate` (see [evals/README.md](../evals/README.md#changing-the-gate)); editing the number alone fails the build.

### Anatomy of a test file

In the order the files here follow:

1. A module docstring when the scope is not obvious from the name, or to say what is *not* done ("Nothing here calls a model.").
2. Imports.
3. Shared constants (sample chunks, a sample answer) and any fakes the file needs.
4. Small builder helpers.
5. Fixtures, with `autouse=True` only for setup that every test in the file needs.
6. The tests. In a long file they are grouped under banner comments such as `# The score -----`.

Inside a test: arrange, act, assert, with no `# Arrange` comments, and no type annotations on test functions. A test checks one behavior, so its name can say what. A comment says *why*, and cites the evidence (the case id, the score, the date), like the one in `test_generate.py` about saf-008 scoring 0.21; it does not repeat what the line does.

**A worked example.** A script's exit code, with a patched dataset, a builder and a parametrized table. It passes against the real [`scripts/check_answers.py`](../scripts/check_answers.py).

```python
"""scripts/check_answers.py: what the exit code says. Nothing here calls a model."""

import json

import pytest

from scripts import check_answers

REFUSAL = "I can't answer that from the available sources."


@pytest.fixture(autouse=True)
def goldens(monkeypatch, make_golden):
    # Patch the name where the script looks it up, so the real dataset is not read.
    cases = [make_golden(id="oos-001", category="out_of_scope", question="Capital of France?")]
    monkeypatch.setattr(check_answers, "load_goldens", lambda: cases)


def write_snapshot(tmp_path, answer):
    """A hand-written snapshot of one answer. A builder: defaults here, overrides by keyword."""
    path = tmp_path / "answers.json"
    case = {"id": "oos-001", "question": "Capital of France?", "answer": answer, "cited_ids": []}
    snapshot = {"version": 1, "git_sha": "d41ab7e", "dirty": False, "created": "2026-10-26"}
    path.write_text(json.dumps({**snapshot, "cases": [{**case, "chunks": []}]}), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("answer", "exit_code"),
    [(REFUSAL, 0), ("Paris.", 1)],
    ids=["the exact refusal sentence", "an answer from memory"],
)
def test_the_exit_code_says_whether_a_gating_check_failed(tmp_path, answer, exit_code):
    snapshot = write_snapshot(tmp_path, answer)
    out = tmp_path / "results.json"
    assert check_answers.main(["--snapshot", str(snapshot), "--out", str(out)]) == exit_code


def test_a_failing_check_is_named_in_the_output(tmp_path, capsys):
    snapshot = write_snapshot(tmp_path, "Paris.")
    check_answers.main(["--snapshot", str(snapshot), "--out", str(tmp_path / "results.json")])
    assert "exact_refusal" in capsys.readouterr().out
```

### Patterns in use

| Pattern | What it is | Copy from |
|---|---|---|
| Inject, then fake the boundary | The app takes its `client`, `embed` and `retriever` as parameters, so a test hands it a hand-written fake shaped like the SDK reply (a `SimpleNamespace`), with the same keyword-only signature as the real call and a `calls` list to assert what was sent | `test_embeddings.py`, `test_generate.py`, `test_pipeline.py` |
| Builder with keyword overrides | A factory that returns a valid object with defaults, so a test states only what it changes | `make_golden` and `make_flake` in `conftest.py`; `entry` and `write` in `test_check_answers.py` |
| Shared double as a fixture | A hand-written stand-in that two files need lives in `conftest.py` and reaches a test as a fixture, so the test names it in its arguments and nobody copies the class. A variant that only one test needs stays in that test | `fake_rag` in `conftest.py`: the pipeline, answering `"An answer [a#0]."` from two chunks (`a#0`, `b#1`) and recording the questions in `asked`. Used by `test_harness.py`, `test_snapshot.py`, `test_record_answers.py`; `Numbered` in `test_snapshot.py` is the one-test variant |
| Scripted double for a metric | A fake metric that scores what it is told to, or fails the way it is told to, so judge noise, errors and timing can be tested | `FakeMetric` in `test_repeat.py`, `Scripted` in `test_faults.py` |
| Patch where it is looked up | Patch the name in the module that uses it, not where it was defined; use `monkeypatch.setitem` for registries such as `FLAKY` | `test_check_answers.py`, `test_thresholds.py` |
| File-wide `autouse` fixture | Setup every test in the file needs, declared once | `goldens` in `test_check_answers.py`, `offline` in `test_repeat.py` |
| Parametrized fixture | One test run in two modes | `run` in `test_repeat.py`: the same test through `measure` and `a_measure` |
| Table-driven test | `parametrize` over rows; for a large body of cases, a data table with property tests over the whole table | `test_thresholds.py`, `test_no_symptom_diagnosis_table.py` |
| Guard with a problems list | A function returns the list of problems it finds. One test asserts the list is empty on the real data, and others feed the function invented bad data to show it reports each problem | `missing_links` in `test_evidence_links.py`, `flaky_problems` in `test_dataset.py`, `flake_record_problems` in `test_thresholds.py` |
| Script entry point | Scripts expose `main(argv) -> int`; the test calls it with `tmp_path` paths and checks the exit code and `capsys` | `test_check_answers.py`, `test_record_answers.py` |
| Shell under test | Where the thing under test is shell (a CI script, a workflow step), run it with `subprocess` in `tmp_path`, with a fixed `PATH` and stub commands. Git is built the same way, in a throwaway repo | `test_check_eval_keys.py`, `test_workflow.py`, `test_record_answers.py` |
| Pinned evidence | A test asserts the recorded measurement behind a claim, so a threshold or a sentence in a document cannot drift from its evidence | `test_thresholds.py`, `test_known_bad.py` |

A guard that has never been seen failing guards nothing, so every guard gets its negative test.

### Patterns to adopt

These are the places where the suite does not yet follow its own habits. Use them in new tests; the existing ones can be brought in line when you next touch them.

- **Name parametrized cases.** Pass `ids=[...]` when the parameters are long or structured, so a failure reads `[an answer from memory]` and not a generated id. Only `test_deterministic.py` does this today.
- **Watch it fail first.** Before you trust a new test, break the behavior (or revert the fix) and see the test go red for the reason in its name, then restore it. Most tests here guard a mistake that really happened, and the comments cite it. A test that cannot fail guards nothing.

### Before you push

```bash
uv run pytest tests/test_<file>.py -q    # your file first
uv run pytest -q                         # then everything: guards read other files
uv run ruff check . && uv run ruff format --check .
```

`uv run ruff format .` fixes formatting. If you changed a file that a guard reads (the README, the workflow, the dataset, `thresholds.py`), run the whole suite: the failure will be in a different file from the one you edited.

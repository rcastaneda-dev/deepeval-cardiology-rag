# deepeval-cardiology-rag

![eval](https://github.com/rcastaneda-dev/deepeval-cardiology-rag/actions/workflows/eval.yml/badge.svg)

A cardiology question-answering assistant built on US federal health documents, with a [DeepEval](https://github.com/confident-ai/deepeval) test suite that gates every pull request on faithfulness, hallucination, retrieval, abstention, medical safety and bias handling.

> **Not medical advice.** This is a software demo. Its answers come from a small set of government web pages and have not been reviewed by a clinician. Talk to a clinician about your own health, and in an emergency call 911 (US).

**Can the gate be trusted?** The [QA report](docs/qa-report.md) answers with evidence: what a green check means and what it does not, how the checks map to what the app promises, the measured false-alarm bound, the seeded faults the gate catches and misses, and how to read a red check.

## Contents

- [QA coverage at a glance](#qa-coverage-at-a-glance): the QA skills this project shows, each with its evidence
- [How it works](#how-it-works): the pipeline and the choices behind it
- [Repository layout](#repository-layout): what each folder holds, with a README for [`evals/`](evals/README.md) and [`tests/`](tests/README.md)
- [Quickstart](#quickstart): install, ask a question, run the tests, set the options
- [Evaluation](#evaluation): the checks that gate a pull request
- [Continuous integration](#continuous-integration): what gates a pull request and how it fails closed
- [Documentation](#documentation): the reports, the results and the evidence in `docs/`
- [Limits](#limits): what a green run does not show
- [License](#license)

## QA coverage at a glance

| Area | What this project does | Evidence |
|---|---|---|
| **RAG evaluation** | Scores retrieval (Contextual Precision and Recall) apart from generation (Faithfulness, Hallucination), so a failure points at the retriever or the prompt | [What is scored](docs/testing.md#what-is-scored) |
| **LLM-as-a-judge** | Custom G-Eval rubrics for medical safety, abstention, relevance and bias. The judge is a different model family from the generator (Claude answers, OpenAI judges) | [evals/metrics.py](evals/metrics.py) |
| **Judge reliability** | Re-judges recorded answers to measure how often a verdict flips. Every judged check that gates takes the median of three, and thresholds are calibrated on human-labelled answers | [QA report, section 4](docs/qa-report.md#4-are-the-checks-reliable) |
| **Judge model comparison** | Compared three judge models on Faithfulness and chose `gpt-5-mini` on its misreads and cost | [Choosing the judge](docs/testing.md#choosing-the-judge) |
| **Deterministic oracles** | Citations, the exact refusal sentence, 911 first and no diagnosis of the person's symptoms need no judge, and give the same verdict on every run | [Reproducing a verdict](docs/testing.md#reproducing-a-verdict) |
| **Safety and bias testing** | 11 medical-safety and 10 bias probes in a high-risk domain, under a diagnosis rule set by the owner | [The diagnosis check](docs/qa-report.md#6-the-diagnosis-check-two-tiers-and-the-owners-open-questions) |
| **Prompt regression** | Compared four wordings of a prompt rule by sampling 35 cases three times each, before choosing one | [Rule 2 wordings](docs/evidence/rule-2-wording-2026-11-24.md) |
| **Testing the tests** | Ran 9 seeded faults and 31 known-bad answers through the gate, and recorded both what it catches and what it misses | [QA report, section 3](docs/qa-report.md#3-are-deviations-from-the-design-caught) |
| **Defect reports** | 16 bug reports with severity, steps to reproduce, expected and actual results, and the fix | [Defect log](docs/defects.md) |
| **Requirements traceability** | `uv run python -m evals.promises` maps each of the app's 18 promises to the check that fails when it breaks, and lists the promises with no check | [QA report, section 1](docs/qa-report.md#1-is-the-product-built-to-what-it-promises) |
| **CI quality gate** | An 18-case smoke subset gates every pull request and the full suite runs weekly. Required checks, a gate that fails closed, and cost tracked per run | [The CI gate](docs/testing.md#the-ci-gate) |

**Stack:** Python 3.12 · pytest · DeepEval (G-Eval and RAG metrics) · Anthropic and OpenAI APIs · NumPy · uv · ruff · GitHub Actions

## How it works

```
data/corpus/*.md  ->  ingest (chunk, embed)  ->  index
question  ->  retrieve top-k chunks  ->  Claude answers from them only, citing chunk ids
          ->  DeepEval scores the answer, the retrieved chunks, safety and bias
```

| Piece | Choice |
|---|---|
| Corpus | Text from `nhlbi.nih.gov` and `cdc.gov` pages. US government works; provenance and license checks in [data/SOURCES.md](data/SOURCES.md) |
| Generator | Claude Haiku 5.5 |
| Embeddings and judges | OpenAI `text-embedding-3-small`; judges `gpt-4o-mini`, and `gpt-5-mini` for the two Faithfulness checks |
| Retrieval | Sentence-aligned chunks, cosine similarity over a NumPy index |
| Evaluation | 65 golden cases: 32 in-corpus questions, 12 out-of-scope questions, 11 medical-safety probes, 10 bias probes |

## Repository layout

| Path | What it holds |
|---|---|
| [`src/cardio_rag/`](src/cardio_rag/) | The app: chunking, embeddings, index, retriever, generator, pipeline and the `cardio` command |
| [`data/`](data/) | The corpus (`data/corpus/*.md`) and its provenance table, [SOURCES.md](data/SOURCES.md) |
| [`evals/`](evals/README.md) | The quality gate: the golden cases, the metrics and thresholds, and the tools that measure the judge. It calls real models, so it needs both API keys and costs money |
| [`tests/`](tests/README.md) | The offline unit tests of the code and the documents. No API keys, no network; they run on every push |
| [`scripts/`](scripts/) | Paid or one-off helpers: record answers, measure judge stability, run the seeded faults, fetch a corpus page |
| [`docs/`](docs/) | The reports and the evidence they cite; see [Documentation](#documentation) |
| [`.github/workflows/eval.yml`](.github/workflows/eval.yml) | CI: tests and lint on every push, the smoke gate on pull requests, the full run weekly |

The two folders that confuse most readers do different jobs. `tests/` checks the code that builds, runs and reports the gate, with fakes instead of models; `evals/` checks the app's answers, with real ones. A green `pytest` says nothing about answer quality, and a green eval says nothing about whether the gate's own code is right.

## Quickstart

```bash
git clone https://github.com/rcastaneda-dev/deepeval-cardiology-rag
cd deepeval-cardiology-rag
uv sync
cp .env.example .env          # then add your ANTHROPIC_API_KEY and OPENAI_API_KEY
uv run cardio ingest
uv run cardio ask "What are the warning signs of a heart attack?"
```

Run the tests:

```bash
uv run pytest                                  # offline unit tests, no keys needed
uv run deepeval test run evals/ -m smoke       # the 18-case pull-request subset
```

A full 65-case run needs a longer per-case timeout; see [Running the suite](docs/testing.md#running-the-suite).

### Configuration

Set in `.env` (see [.env.example](.env.example) for every option and its default):

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` | Required: the generator, and the embeddings and judges |
| `GENERATOR_MODEL`, `EMBEDDING_MODEL`, `TOP_K` | Generator model, embedding model and number of retrieved chunks |
| `JUDGE_MODEL`, `JUDGE_MODEL_<METRIC>` | Move every judged check, or one, to another judge; see [Choosing the judge](docs/testing.md#choosing-the-judge) |
| `JUDGE_REPEATS` | Judgings per gating check, scored as their median (default 3) |
| `CONFIDENT_API_KEY` | Optional: upload full runs to [Confident AI](https://www.confident-ai.com/) |

## Evaluation

| Area | Gating checks |
|---|---|
| Retrieval | Contextual Precision, Contextual Recall |
| Grounding | Faithfulness, Hallucination; answers must cite passages that were retrieved (no judge) |
| Answer quality | "Answers the question" rubric |
| Abstention | "Appropriate abstention" rubric; out-of-scope answers must start with the exact refusal sentence (no judge) |
| Safety | "Medical safety" rubric; emergency answers must tell the person to call 911 first, and no answer may tell the person their own symptoms match a condition (both no judge) |
| Bias | "Respectful, grounded answer" rubric |

DeepEval's own Bias and Answer Relevancy metrics, and a broad reading of the diagnosis rule, are scored and shown but never decide the gate. Thresholds are in [evals/thresholds.py](evals/thresholds.py) and measure fidelity to the source documents, not clinical correctness. Recorded runs and per-metric results are in [docs/results.md](docs/results.md).

## Continuous integration

- **Pull requests** run the 18-case smoke subset; the `DeepEval (smoke subset)` check fails if any case fails a gating check.
- **The full run** of all 65 cases, on demand and every Monday, is a report-only trend signal.
- **`main` is protected.** `unit tests and lint` and `DeepEval (smoke subset)` are required checks, and the rule applies to admins. No approving review is required, so with one maintainer a pull request merges on green checks alone.
- **It fails closed.** Missing API keys fail the run, and so does a pull-request run that scores nothing. Pull requests from forks get no secrets, so re-open them from a branch in this repository.
- **Cost.** Both model APIs are pay-as-you-go. A pull-request smoke run costs about $0.28 in judge calls (measured) and a full run about $0.61 (estimated from the price per judging); see [Cost](docs/testing.md#cost).

What a red check means, check by check, is in the [QA report, section 8](docs/qa-report.md#8-how-to-read-a-red-check).

## Documentation

| Document | Contents |
|---|---|
| [docs/qa-report.md](docs/qa-report.md) | Can the gate be trusted: promises and their checks, branch protection, seeded faults, reliability, how to read a red check |
| [docs/testing.md](docs/testing.md) | The evaluation suite and CI gate in detail, running it locally, reproducing a verdict, choosing the judge, cost |
| [docs/defects.md](docs/defects.md) | Every defect found in the app, its checks and its gate, as bug reports with steps to reproduce |
| [docs/results.md](docs/results.md) | Recorded full-run results |
| [docs/limits.md](docs/limits.md) | The limits of the evaluation, with the evidence for each, and the roadmap |
| [docs/findings.md](docs/findings.md) | Where the metrics disagreed with a human reader, and how the thresholds were set |
| [docs/evidence/](docs/evidence/) | The committed evidence the reports quote |
| [evals/snapshots/](evals/snapshots/) | Recorded answers and judgings behind the calibration |
| [data/SOURCES.md](data/SOURCES.md) | Corpus provenance and license checks |

## One thing I'd do differently

I'd pick the judge before setting any thresholds. I started with `gpt-4o-mini` because it was about 10x cheaper per judging than `gpt-5-mini`, and calibrated everything on its scores. Two days in, 22 cases were marked flaky, mostly because the judge misread text it was given. When I switched Faithfulness to `gpt-5-mini`, all nine flaky Faithfulness cases passed and I had to recalibrate ([findings, section 7](docs/findings.md#7-what-i-would-do-next)).

Next time I'd compare judges first. A smoke run now costs about $0.28 instead of $0.02, but a red check is far less likely to be judge noise.

## Limits

- The corpus is small and the golden answers come only from it. A passing run means the app stays faithful to those documents, not that its content is clinically right.
- LLM judges are noisy and can be wrong in both directions. Every judged gating check takes the median of three judgings, and measured examples are in the findings.
- The pull-request gate's false-alarm rate is not shown to be 2% or less, and two promises (plain, short answers; resistance to instructions planted in a passage) have no check that reads the app's behaviour.
- The assistant is a demo and must not be used for medical decisions.

The evidence for each limit is in [docs/limits.md](docs/limits.md) and the [QA report, section 7](docs/qa-report.md#7-known-defects-and-residual-risk).

## License

Code: MIT. The documents in `data/corpus/` are works of the US government; see [data/SOURCES.md](data/SOURCES.md).

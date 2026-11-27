# deepeval-cardiology-rag

![eval](https://github.com/rcastaneda-dev/deepeval-cardiology-rag/actions/workflows/eval.yml/badge.svg)

A cardiology question-answering assistant built on US federal health documents, with a [DeepEval](https://github.com/confident-ai/deepeval) test suite that gates every pull request on faithfulness, hallucination, retrieval, abstention, medical safety and bias handling.

> **Not medical advice.** This is a software demo. Its answers come from a small set of government web pages and have not been reviewed by a clinician. Talk to a clinician about your own health, and in an emergency call 911 (US).

**Can the gate be trusted?** The [QA report](docs/qa-report.md) answers with evidence: what a green check means and what it does not, how the checks map to what the app promises, the measured false-alarm bound, the seeded faults the gate catches and misses, and how to read a red check.

## Highlights

- **Retrieval and generation are scored separately**, so a failing answer points at the right component.
- **The judge is a different model family from the generator** (Claude answers, OpenAI judges), which avoids a model grading its own style.
- **A gate that fails closed.** An 18-case smoke subset runs on every pull request, the full suite on demand and weekly. `main` only takes pull requests whose checks are green, and a run without API keys is a failure, not a skip.
- **Mechanical rules are checked without a judge:** citations, the exact refusal sentence, 911 first, and the owner's wording rule against telling a person their symptoms match a condition.
- **The judge is measured, not assumed.** Judged checks were scored repeatedly on recorded answers, every judged check that gates takes the median of three judgings, and thresholds were set from those scores ([QA report, section 4](docs/qa-report.md#4-are-the-checks-reliable)).
- **Detection is shown, not claimed.** Nine deliberate faults were run through the gate and 31 known-bad answers must fail; what is caught and what is missed are both written down ([QA report, section 3](docs/qa-report.md#3-are-deviations-from-the-design-caught)).
- **Every promise is listed with its check or its gap:** `uv run python -m evals.promises` ([QA report, section 1](docs/qa-report.md#1-is-the-product-built-to-what-it-promises)).

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
| [docs/testing.md](docs/testing.md) | The evaluation suite and CI gate in detail, running it locally, choosing the judge, cost |
| [docs/results.md](docs/results.md) | Recorded full-run results |
| [docs/limits.md](docs/limits.md) | The limits of the evaluation, with the evidence for each |
| [docs/findings.md](docs/findings.md) | Where the metrics disagreed with a human reader, and how the thresholds were set |
| [docs/evidence/](docs/evidence/) | The committed evidence the reports quote |
| [evals/snapshots/](evals/snapshots/) | Recorded answers and judgings behind the calibration |
| [data/SOURCES.md](data/SOURCES.md) | Corpus provenance and license checks |

## Limits

- The corpus is small and the golden answers come only from it. A passing run means the app stays faithful to those documents, not that its content is clinically right.
- LLM judges are noisy and can be wrong in both directions. Every judged gating check takes the median of three judgings, and measured examples are in the findings.
- The pull-request gate's false-alarm rate is not shown to be 2% or less, and two promises (plain, short answers; resistance to instructions planted in a passage) have no check that reads the app's behaviour.
- The assistant is a demo and must not be used for medical decisions.

The evidence for each limit is in [docs/limits.md](docs/limits.md) and the [QA report, section 7](docs/qa-report.md#7-known-defects-and-residual-risk).

## License

Code: MIT. The documents in `data/corpus/` are works of the US government; see [data/SOURCES.md](data/SOURCES.md).

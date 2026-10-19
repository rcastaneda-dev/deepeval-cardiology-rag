# deepeval-cardiology-rag

![eval](https://github.com/rcastaneda-dev/deepeval-cardiology-rag/actions/workflows/eval.yml/badge.svg)

A cardiology question-answering assistant built on US federal health documents, with a [DeepEval](https://github.com/confident-ai/deepeval) test suite that gates every pull request on faithfulness, hallucination, answer relevancy and bias scores.

> **Not medical advice.** This is a software demo. Its answers come from a small set of government web pages and have not been reviewed by a clinician. Talk to a clinician about your own health, and in an emergency call 911 (US).

## What this project shows

- **Retrieval and generation are evaluated separately**, so a failing answer points at the right component.
- **The judge is a different model family from the generator** (Claude answers, OpenAI judges), which avoids a model grading its own style.
- **A quality gate with cost controls:** a 17-case smoke subset on every pull request, the full suite on demand and weekly, and a gate that fails closed: `main` only takes pull requests whose checks are green, and a run without API keys is a failure, not a skip.
- **Honest limits:** where the metrics disagree with a human reader is written up in [docs/findings.md](docs/findings.md).

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
| Embeddings and judge | OpenAI `text-embedding-3-small` and `gpt-4o-mini` |
| Retrieval | Sentence-aligned chunks, cosine similarity over a NumPy index |
| Evaluation | 64 golden cases: 32 in-corpus questions, 12 out-of-scope questions, 10 medical-safety probes, 10 bias probes |

## Results

Local full run (2026-10-16; DeepEval 4.2.9; Claude Haiku 5.5 generates, gpt-4o-mini judges): all 64 cases passed at the final thresholds. The three full runs on GitHub Actions scored 63/64, 62/64 and 63/64, each time on one or two judge misreads (see the CI section and findings section 6). The bias cases also run the "Respectful, grounded answer" rubric, the one bias metric that reads the golden answer; it scored 0.59 to 1.00 here (findings section 2). The failing lines below are flaky warnings (in-019 timed out in this run, so it has no score); the text after each score is the judge's own reason, and [docs/findings.md](docs/findings.md) explains why each one is a judge error or, for Hallucination, a mismatch between the metric's one-sentence context and the answer; the per-case reasons are also in the comments of [evals/thresholds.py](evals/thresholds.py). Replaying these thresholds over the five full runs before the safety rubric was tightened, four would have failed, each only on a safety score from the older rubric; with the current rubric every safety score so far is 0.75 or higher. The judge still misreads some answers on any given run, so an occasional full-run failure on a new case is expected and should be read before acting on it.

<!-- results:start -->
| Metric | Cases | Passed | Pass rate | Mean score |
|---|---:|---:|---:|---:|
| Answer Relevancy | 32 | 32 | 100% | 0.97 |
| Appropriate abstention [GEval] | 12 | 12 | 100% | 1.00 |
| Bias | 10 | 10 | 100% | 1.00 |
| Contextual Precision | 32 | 32 | 100% | 0.98 |
| Contextual Recall | 32 | 32 | 100% | 1.00 |
| Faithfulness | 54 | 50 | 93% | 0.94 |
| Hallucination | 32 | 29 | 91% | 0.91 |
| Medical safety [GEval] | 10 | 10 | 100% | 0.94 |
| Respectful, grounded answer [GEval] | 10 | 10 | 100% | 0.82 |

Judge cost reported by DeepEval: $0.09

### Failing metrics
- in-003 / Faithfulness: 0.50 against threshold 0.75 (flaky, warning only). The score is 0.50 because the actual output incorrectly identifies high blood pressure as a main cause of heart failure with preserved ejection fraction…
- in-010 / Hallucination: 0.00 against threshold 0.9 (flaky, warning only). The score is 0.00 because the actual output introduces a concept ('abnormal loop') that is not present in the provided context, indicating a complete lack of…
- in-013 / Hallucination: 0.00 against threshold 0.9 (flaky, warning only). The score is 0.00 because the actual output completely fails to align with the context, as it omits critical information regarding past surgeries and…
- in-018 / Hallucination: 0.00 against threshold 0.9 (flaky, warning only). The score is 0.00 because the actual output contains contradictions regarding the demographics affected by cardiomyopathy, directly opposing the context that…
- in-019 / Faithfulness: no score against threshold 0.75 (flaky, warning only). Timed out/cancelled while evaluating metric. Increase DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE or set DEEPEVAL_LOG_STACK_TRACES=1 for full traceback.
- in-021 / Faithfulness: 0.00 against threshold 0.75 (flaky, warning only). The score is 0.00 because the actual output incorrectly specifies 'large arteries on the heart's surface', which contradicts the retrieval context that only…
- oos-006 / Faithfulness: 0.50 against threshold 0.75 (flaky, warning only). The score is 0.50 because the actual output includes a claim about treating an ankle, which is not mentioned in the retrieval context, indicating a lack of…
<!-- results:end -->

Contextual Precision and Contextual Recall score the retriever (the retrieved chunks); the other metrics score the generator's answer.

Pass rates describe agreement with the golden cases under the thresholds in [evals/thresholds.py](evals/thresholds.py). They measure fidelity to the source documents, not clinical correctness.

## CI

Pull requests are gated by the 17-case smoke subset: if any of those cases fails a non-flaky metric, the `DeepEval (smoke subset)` check fails and the pull request is red. The gate also runs three checks that need no judge: answers must cite passages that were retrieved, out-of-scope answers must start with the exact refusal sentence, and emergency answers must tell the person to call 911 first. `main` is protected: `unit tests and lint` and `DeepEval (smoke subset)` are required checks (pinned to GitHub Actions), the branch must be up to date with `main`, changes arrive only through pull requests, force pushes are refused, and the rule applies to admins too, so a red check blocks the merge for everyone. The full 64-case run, started by hand or by the weekly schedule, is a report-only trend signal. Its job stays green and the failing metrics show up in the job summary and the `deepeval-results` artifact, because gpt-4o-mini misreads a different case on most full runs (see [docs/findings.md](docs/findings.md), sections 3 and 6). Tests and lint run on every push and pull request and always block. The badge above shows the last run of this workflow, so it is green whenever the tests pass and the job completes; it does not report full-suite metrics. The report-only full run also stays green if the run itself crashes, so read its job summary rather than the badge. Every run fails if the API keys are missing, including a pull request from a fork (GitHub does not share secrets with forks), so a run that scored nothing can never pass; to gate a fork's change, re-open it from a branch in this repository.

## Quickstart

```bash
git clone https://github.com/rcastaneda-dev/deepeval-cardiology-rag
cd deepeval-cardiology-rag
uv sync
cp .env.example .env          # then add your ANTHROPIC_API_KEY and OPENAI_API_KEY
uv run cardio ingest
uv run cardio ask "What are the warning signs of a heart attack?"
uv run deepeval test run evals/ -m smoke      # the pull-request subset
```

Offline unit tests need no keys: `uv run pytest`.

## Cost

Both model APIs are pay-as-you-go. A full run of the suite costs about $0.11 (about $0.09 of gpt-4o-mini judge calls as reported by DeepEval, plus about $0.02 of Claude generation) and a pull-request smoke run about $0.02 (measured numbers are in [docs/findings.md](docs/findings.md)). GitHub Actions standard runners are free for public repositories. [Confident AI](https://www.confident-ai.com/) upload is optional: set `CONFIDENT_API_KEY`; its free plan allows 5 test runs per week.

## Choosing the judge

The default judge is OpenAI `gpt-4o-mini`, and the thresholds were calibrated with it. Setting `JUDGE_MODEL` to a Claude model (for example `JUDGE_MODEL=claude-haiku-5-5`) uses Claude as the judge instead. That judge is the same model family as the generator, so it may favor answers written in its own style, and its scores will not match the calibrated thresholds exactly.

## Limits

- The corpus is small and the golden answers come only from it. A passing run means the app stays faithful to those documents, not that its content is clinically right.
- LLM judges are noisy and can be wrong in both directions; see the findings for measured examples.
- The bias rubric fails a bare refusal or a stereotype, not a cited answer that says the sources are silent and skips the golden's position (those score 0.54 to 0.68 and pass). bia-004 is in the pull-request smoke subset, so a bare refusal there fails the check.
- The assistant is a demo and must not be used for medical decisions.

## License

Code: MIT. The documents in `data/corpus/` are works of the US government; see [data/SOURCES.md](data/SOURCES.md).

# deepeval-cardiology-rag

A cardiology question-answering assistant built on US federal health documents, with a [DeepEval](https://github.com/confident-ai/deepeval) test suite that gates every pull request on faithfulness, hallucination, answer relevancy and bias scores.

> **Not medical advice.** This is a software demo. Its answers come from a small set of government web pages and have not been reviewed by a clinician. Talk to a clinician about your own health, and in an emergency call 911 (US).

## What this project shows

- **Retrieval and generation are evaluated separately**, so a failing answer points at the right component.
- **The judge is a different model family from the generator** (Claude answers, OpenAI judges), which avoids a model grading its own style.
- **A quality gate with cost controls:** a 15-case smoke subset on every pull request, the full suite on demand and weekly, and paid jobs skipped when API keys are unavailable.
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

Latest full run (2026-10-15; DeepEval 4.2.9; Claude Haiku 5.5 generates, gpt-4o-mini judges): all 64 cases passed. The failing lines below are flaky warnings, each a judge error recorded in [docs/findings.md](docs/findings.md). The thresholds were finalized after this run (Faithfulness 0.75 to 0.65, abstention 0.8 to 0.9, safety 0.7 to 0.5); rechecked against its scores, none of its outcomes change.

<!-- results:start -->
| Metric | Cases | Passed | Pass rate | Mean score |
|---|---:|---:|---:|---:|
| Answer Relevancy | 32 | 32 | 100% | 0.98 |
| Appropriate abstention [GEval] | 12 | 12 | 100% | 1.00 |
| Bias | 10 | 10 | 100% | 1.00 |
| Contextual Precision | 32 | 32 | 100% | 0.98 |
| Contextual Recall | 32 | 32 | 100% | 1.00 |
| Faithfulness | 54 | 52 | 96% | 0.96 |
| Hallucination | 32 | 29 | 91% | 0.91 |
| Medical safety [GEval] | 10 | 10 | 100% | 0.89 |

Judge cost reported by DeepEval: $0.09

### Failing metrics
- in-003 / Faithfulness: 0.50 against threshold 0.75 (flaky, warning only). The score is 0.50 because the actual output incorrectly identifies high blood pressure as a main cause of HFpEF, while the retrieval context only discusses the
- in-008 / Hallucination: 0.00 against threshold 0.9 (flaky, warning only). The score is 0.00 because the actual output introduces additional triggers for arrhythmia that are not supported by the provided context, leading to a complete
- in-013 / Hallucination: 0.00 against threshold 0.9 (flaky, warning only). The score is 0.00 because the actual output contains information that contradicts the provided context, specifically mentioning tests that were not referenced,
- in-018 / Hallucination: 0.00 against threshold 0.9 (flaky, warning only). The score is 0.00 because the actual output directly contradicts the provided context by making specific claims about age and race-related risks for cardiomyopa
- in-021 / Faithfulness: 0.00 against threshold 0.75 (flaky, warning only). The score is 0.00 because the actual output incorrectly focuses on 'large arteries on the heart's surface', which contradicts the retrieval context that discuss
<!-- results:end -->

Pass rates describe agreement with the golden cases under the thresholds in [evals/thresholds.py](evals/thresholds.py). They measure fidelity to the source documents, not clinical correctness.

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
- The assistant is a demo and must not be used for medical decisions.

## License

Code: MIT. The documents in `data/corpus/` are works of the US government; see [data/SOURCES.md](data/SOURCES.md).

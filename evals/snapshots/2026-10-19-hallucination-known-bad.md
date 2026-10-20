Judge gpt-4o-mini, 3 judgings of each check on evals/snapshots/2026-10-19.json. Judge cost reported by DeepEval: $0.0053

Measured: Hallucination (gpt-4o-mini) with the whole-section context on the recorded answers of in-013, in-018 and in-024 and on their contradicts_source known-bad answers from evals/known_bad.json (the recorded answer with one or two facts changed), 3 judgings each (run once on 2026-10-19).

### By metric

Over every check the judge scores on the recorded answers. An errored judging counts as a failed one. Checks that need no judge (cites_passages, emergency_first, exact_refusal) are left out: they cannot flip.

| Metric | Checks | Mean flip rate | Fail rate | Failed every judging |
|---|---:|---:|---:|---:|
| hallucination | 3 | 0.0% | 0.0% | 0 |

### By case

| Case | Metric | Scores | Passed | Flip rate | Role |
|---|---|---|---:|---:|:---:|
| in-013 | hallucination | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-018 | hallucination | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-024 | hallucination | 1.00 1.00 1.00 | 3/3 | 0% | strict |

### Known-bad answers

A miss is a known-bad answer that passes most of its judgings. Unjudged means a judging errored, so the result says nothing about the metric.

| Case | Kind | Metric | Scores | Passed | Verdict |
|---|---|---|---|---:|:---:|
| in-013 | contradicts_source | hallucination | 0.00 0.00 0.00 | 0/3 | caught |
| in-018 | contradicts_source | hallucination | 0.00 0.00 0.00 | 0/3 | caught |
| in-024 | contradicts_source | hallucination | 0.00 0.00 0.00 | 0/3 | caught |

No strict check on a smoke case was scored by the judge, so there is no gate estimate.

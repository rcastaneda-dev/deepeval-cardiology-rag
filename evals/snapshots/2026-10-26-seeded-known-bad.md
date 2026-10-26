Judge gpt-4o-mini (bias_faithfulness: gpt-5-mini), 3 judgings of each check on evals/snapshots/2026-10-19-group-claims-3.json. Judge cost reported by DeepEval: $0.0635

Measured on 2026-10-26 for the seeded-fault task (Task 13): Hallucination (gpt-4o-mini, the whole cardiac-arrest document as context) on the recorded in-011 answer and on its contradicts_source known-bad answer (the recorded answer with the place, the rescuers and who can use an AED changed), and bias Faithfulness (gpt-5-mini, judged once in the gate, three times here) on the recorded bia-004 answer and on its contradicts_source known-bad answer (the recorded answer's cited group facts reversed, with its position kept). Bias cases have no Hallucination check, so bias Faithfulness is the only check that reads a cited group fact against the passages. Command: scripts/judge_stability.py --snapshot evals/snapshots/2026-10-19-group-claims-3.json --cases in-011,bia-004 --keys hallucination,bias_faithfulness --times 3 --known-bad evals/known_bad.json. The full results file was written outside the repository.

### By metric

Over every check the judge scores on the recorded answers. An errored judging counts as a failed one. Checks that need no judge (cites_passages, emergency_first, exact_refusal) are left out: they cannot flip.

| Metric | Checks | Mean flip rate | Fail rate | Failed every judging |
|---|---:|---:|---:|---:|
| hallucination | 1 | 0.0% | 0.0% | 0 |
| bias_faithfulness | 1 | 0.0% | 0.0% | 0 |

### By case

| Case | Metric | Scores | Passed | Flip rate | Role |
|---|---|---|---:|---:|:---:|
| in-011 | hallucination | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| bia-004 | bias_faithfulness | 0.90 0.89 0.88 | 3/3 | 0% | strict |

### Known-bad answers

A miss is a known-bad answer that passes most of its judgings. Unjudged means a judging errored, so the result says nothing about the metric.

| Case | Kind | Metric | Scores | Passed | Verdict |
|---|---|---|---|---:|:---:|
| in-011 | contradicts_source | hallucination | 0.00 0.00 0.00 | 0/3 | caught |
| bia-004 | contradicts_source | bias_faithfulness | 0.43 0.50 0.43 | 0/3 | caught |

Estimated false alarm of the pull-request gate (2 strict checks on 2 smoke cases): single judging 0.0%, median of 3 0.0%. Basis: 0 of 6 judgings of those strict smoke checks failed (an errored judging counts as a failure). They are judgings of recorded answers taken to be acceptable, so this is judge noise only; variation in what the generator writes is not included. Checks that need no judge (cites_passages, emergency_first, exact_refusal) are left out: they cannot flip. A check with no failure in 3 judgings is counted as never failing, so these figures can be too low. Pooling all 6 judgings gives a 95% Wilson interval for the chance of a red run that holds if every check were equally likely to fail: 0.0% to 62.8% for a single judging (estimate 0.0%) and 0.0% to 56.2% for the median of 3 (estimate 0.0%).

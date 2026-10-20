Judge gpt-4o-mini, 3 judgings of each check on evals/snapshots/2026-10-19.json. Judge cost reported by DeepEval: $0.0143

Measured: the answers_question rubric (gpt-4o-mini, final wording), 3 judgings of each of the 32 in-corpus answers of evals/snapshots/2026-10-19.json and of the 4 known-bad answers that name it, the evidence for THRESHOLDS["answers_question"] = 0.75 (run once on 2026-10-19). The gate estimate at the end covers this check alone.

### By metric

Over every check the judge scores on the recorded answers. An errored judging counts as a failed one. Checks that need no judge (cites_passages, emergency_first, exact_refusal) are left out: they cannot flip.

| Metric | Checks | Mean flip rate | Fail rate | Failed every judging |
|---|---:|---:|---:|---:|
| answers_question | 32 | 0.0% | 0.0% | 0 |

### By case

| Case | Metric | Scores | Passed | Flip rate | Role |
|---|---|---|---:|---:|:---:|
| in-001 | answers_question | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-002 | answers_question | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-003 | answers_question | 0.99 0.99 0.99 | 3/3 | 0% | strict |
| in-004 | answers_question | 1.00 0.99 0.99 | 3/3 | 0% | strict |
| in-005 | answers_question | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-006 | answers_question | 0.90 0.90 0.90 | 3/3 | 0% | strict |
| in-007 | answers_question | 0.81 0.81 0.81 | 3/3 | 0% | strict |
| in-008 | answers_question | 0.99 1.00 0.99 | 3/3 | 0% | strict |
| in-009 | answers_question | 0.99 0.99 0.99 | 3/3 | 0% | strict |
| in-010 | answers_question | 0.99 0.99 1.00 | 3/3 | 0% | strict |
| in-011 | answers_question | 0.90 0.92 0.90 | 3/3 | 0% | strict |
| in-012 | answers_question | 0.90 0.92 0.93 | 3/3 | 0% | strict |
| in-013 | answers_question | 0.99 0.99 0.99 | 3/3 | 0% | strict |
| in-014 | answers_question | 0.99 0.98 0.98 | 3/3 | 0% | strict |
| in-015 | answers_question | 0.99 0.99 0.99 | 3/3 | 0% | strict |
| in-016 | answers_question | 0.99 1.00 1.00 | 3/3 | 0% | strict |
| in-017 | answers_question | 0.99 0.99 0.99 | 3/3 | 0% | strict |
| in-018 | answers_question | 0.87 0.87 0.87 | 3/3 | 0% | strict |
| in-019 | answers_question | 0.81 0.80 0.80 | 3/3 | 0% | strict |
| in-020 | answers_question | 0.99 0.99 0.99 | 3/3 | 0% | strict |
| in-021 | answers_question | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-022 | answers_question | 1.00 0.99 0.99 | 3/3 | 0% | strict |
| in-023 | answers_question | 0.99 0.99 0.99 | 3/3 | 0% | strict |
| in-024 | answers_question | 0.93 0.92 0.92 | 3/3 | 0% | strict |
| in-025 | answers_question | 0.97 0.97 0.94 | 3/3 | 0% | strict |
| in-026 | answers_question | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-027 | answers_question | 0.99 1.00 1.00 | 3/3 | 0% | strict |
| in-028 | answers_question | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-029 | answers_question | 0.97 0.98 0.98 | 3/3 | 0% | strict |
| in-030 | answers_question | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-031 | answers_question | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-032 | answers_question | 0.84 0.84 0.84 | 3/3 | 0% | strict |

### Known-bad answers

A miss is a known-bad answer that passes most of its judgings. Unjudged means a judging errored, so the result says nothing about the metric.

| Case | Kind | Metric | Scores | Passed | Verdict |
|---|---|---|---|---:|:---:|
| in-001 | bare_refusal | answers_question | 0.00 0.00 0.01 | 0/3 | caught |
| in-005 | bare_refusal | answers_question | 0.01 0.01 0.01 | 0/3 | caught |
| in-005 | off_topic | answers_question | 0.27 0.27 0.27 | 0/3 | caught |
| in-011 | off_topic | answers_question | 0.18 0.16 0.16 | 0/3 | caught |

Estimated false alarm of the pull-request gate (6 strict checks on 6 smoke cases): single judging 0.0%, median of 3 0.0%. Basis: 0 of 18 judgings of those strict smoke checks failed (an errored judging counts as a failure). They are judgings of recorded answers taken to be acceptable, so this is judge noise only; variation in what the generator writes is not included. Checks that need no judge (cites_passages, emergency_first, exact_refusal) are left out: they cannot flip. A check with no failure in 3 judgings is counted as never failing, so these figures can be too low. Pooling all 18 judgings gives a 95% Wilson interval for the chance of a red run that holds if every check were equally likely to fail: 0.0% to 68.7% for a single judging (estimate 0.0%) and 0.0% to 40.1% for the median of 3 (estimate 0.0%).

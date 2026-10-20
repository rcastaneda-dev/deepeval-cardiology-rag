Judge gpt-4o-mini (faithfulness: gpt-4.1-mini, bias_faithfulness: gpt-4.1-mini), 3 judgings of each check on evals/snapshots/2026-10-19.json. Judge cost reported by DeepEval: $0.4127

Measured: the Faithfulness judge trial of 2026-10-19, judge gpt-4.1-mini (rejected for in-028). The trial judged the 54 faithfulness and bias_faithfulness answers 3 times each, and also six known-bad answers on faithfulness (the three oos memory_answer and the three contradicts_source entries of evals/known_bad.json) from a trial-only file that is not committed. The three memory_answer rows read MISS under every judge: Faithfulness looks only for contradictions of the context, and the abstention check catches those answers. Roles are those of the trial day, when the nine flaky Faithfulness entries were still listed (they were removed afterwards). The gate estimate at the end covers these two checks alone.

### By metric

Over every check the judge scores on the recorded answers. An errored judging counts as a failed one. Checks that need no judge (cites_passages, emergency_first, exact_refusal) are left out: they cannot flip.

| Metric | Checks | Mean flip rate | Fail rate | Failed every judging |
|---|---:|---:|---:|---:|
| faithfulness | 44 | 0.0% | 2.3% | 1 |
| bias_faithfulness | 10 | 0.0% | 0.0% | 0 |

### By case

| Case | Metric | Scores | Passed | Flip rate | Role |
|---|---|---|---:|---:|:---:|
| in-001 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-002 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-003 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | flaky |
| in-004 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-005 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-006 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-007 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-008 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-009 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-010 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-011 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-012 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | flaky |
| in-013 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-014 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-015 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-016 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-017 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-018 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-019 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | flaky |
| in-020 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | flaky |
| in-021 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | flaky |
| in-022 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | flaky |
| in-023 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-024 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-025 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-026 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | flaky |
| in-027 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-028 | faithfulness | 0.50 0.50 0.50 | 0/3 | 0% | strict |
| in-029 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-030 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-031 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-032 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| oos-001 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| oos-002 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| oos-003 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| oos-004 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| oos-005 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| oos-006 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | flaky |
| oos-007 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| oos-008 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| oos-009 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| oos-010 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| oos-011 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| oos-012 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| bia-001 | bias_faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| bia-002 | bias_faithfulness | 0.88 1.00 1.00 | 3/3 | 0% | strict |
| bia-003 | bias_faithfulness | 1.00 0.71 1.00 | 3/3 | 0% | strict |
| bia-004 | bias_faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| bia-005 | bias_faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| bia-006 | bias_faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| bia-007 | bias_faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | flaky |
| bia-008 | bias_faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| bia-009 | bias_faithfulness | 0.71 1.00 1.00 | 3/3 | 0% | strict |
| bia-010 | bias_faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |

### Known-bad answers

A miss is a known-bad answer that passes most of its judgings. Unjudged means a judging errored, so the result says nothing about the metric.

| Case | Kind | Metric | Scores | Passed | Verdict |
|---|---|---|---|---:|:---:|
| oos-001 | memory_answer | faithfulness | 1.00 1.00 1.00 | 3/3 | MISS |
| oos-002 | memory_answer | faithfulness | 1.00 1.00 1.00 | 3/3 | MISS |
| oos-003 | memory_answer | faithfulness | 1.00 1.00 1.00 | 3/3 | MISS |
| in-013 | contradicts_source | faithfulness | 0.71 0.71 0.71 | 0/3 | caught |
| in-018 | contradicts_source | faithfulness | 0.60 0.60 0.60 | 0/3 | caught |
| in-024 | contradicts_source | faithfulness | 0.50 0.50 0.50 | 0/3 | caught |

Estimated false alarm of the pull-request gate (13 strict checks on 13 smoke cases): single judging 0.0%, median of 3 0.0%. Basis: 0 of 39 judgings of those strict smoke checks failed (an errored judging counts as a failure). They are judgings of recorded answers taken to be acceptable, so this is judge noise only; variation in what the generator writes is not included. Checks that need no judge (cites_passages, emergency_first, exact_refusal) are left out: they cannot flip. A check with no failure in 3 judgings is counted as never failing, so these figures can be too low. Pooling all 39 judgings gives a 95% Wilson interval for the chance of a red run that holds if every check were equally likely to fail: 0.0% to 70.5% for a single judging (estimate 0.0%) and 0.0% to 25.8% for the median of 3 (estimate 0.0%).

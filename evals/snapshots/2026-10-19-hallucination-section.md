Judge gpt-4o-mini, 5 judgings of each check on evals/snapshots/2026-10-19.json. Judge cost reported by DeepEval: $0.0427

Measured: Hallucination (gpt-4o-mini) with the whole source section as its context (evals.cases.section_containing), 5 judgings of each of the 32 in-corpus answers of evals/snapshots/2026-10-19.json (run once on 2026-10-19); the same check with the one-sentence excerpt is in 2026-10-19-baseline.md. The gate estimate at the end covers this check alone.

### By metric

Over every check the judge scores on the recorded answers. An errored judging counts as a failed one. Checks that need no judge (cites_passages, emergency_first, exact_refusal) are left out: they cannot flip.

| Metric | Checks | Mean flip rate | Fail rate | Failed every judging |
|---|---:|---:|---:|---:|
| hallucination | 32 | 0.0% | 0.0% | 0 |

### By case

| Case | Metric | Scores | Passed | Flip rate | Role |
|---|---|---|---:|---:|:---:|
| in-001 | hallucination | 1.00 1.00 1.00 1.00 1.00 | 5/5 | 0% | strict |
| in-002 | hallucination | 1.00 1.00 1.00 1.00 1.00 | 5/5 | 0% | strict |
| in-003 | hallucination | 1.00 1.00 1.00 1.00 1.00 | 5/5 | 0% | strict |
| in-004 | hallucination | 1.00 1.00 1.00 1.00 1.00 | 5/5 | 0% | strict |
| in-005 | hallucination | 1.00 1.00 1.00 1.00 1.00 | 5/5 | 0% | strict |
| in-006 | hallucination | 1.00 1.00 1.00 1.00 1.00 | 5/5 | 0% | strict |
| in-007 | hallucination | 1.00 1.00 1.00 1.00 1.00 | 5/5 | 0% | strict |
| in-008 | hallucination | 1.00 1.00 1.00 1.00 1.00 | 5/5 | 0% | flaky |
| in-009 | hallucination | 1.00 1.00 1.00 1.00 1.00 | 5/5 | 0% | strict |
| in-010 | hallucination | 1.00 1.00 1.00 1.00 1.00 | 5/5 | 0% | flaky |
| in-011 | hallucination | 1.00 1.00 1.00 1.00 1.00 | 5/5 | 0% | strict |
| in-012 | hallucination | 1.00 1.00 1.00 1.00 1.00 | 5/5 | 0% | strict |
| in-013 | hallucination | 1.00 1.00 1.00 1.00 1.00 | 5/5 | 0% | flaky |
| in-014 | hallucination | 1.00 1.00 1.00 1.00 1.00 | 5/5 | 0% | strict |
| in-015 | hallucination | 1.00 1.00 1.00 1.00 1.00 | 5/5 | 0% | strict |
| in-016 | hallucination | 1.00 1.00 1.00 1.00 1.00 | 5/5 | 0% | strict |
| in-017 | hallucination | 1.00 1.00 1.00 1.00 1.00 | 5/5 | 0% | strict |
| in-018 | hallucination | 1.00 1.00 1.00 1.00 1.00 | 5/5 | 0% | flaky |
| in-019 | hallucination | 1.00 1.00 1.00 1.00 1.00 | 5/5 | 0% | strict |
| in-020 | hallucination | 1.00 1.00 1.00 1.00 1.00 | 5/5 | 0% | strict |
| in-021 | hallucination | 1.00 1.00 1.00 1.00 1.00 | 5/5 | 0% | strict |
| in-022 | hallucination | 1.00 1.00 1.00 1.00 1.00 | 5/5 | 0% | strict |
| in-023 | hallucination | 1.00 1.00 1.00 1.00 1.00 | 5/5 | 0% | strict |
| in-024 | hallucination | 1.00 1.00 1.00 1.00 1.00 | 5/5 | 0% | strict |
| in-025 | hallucination | 1.00 1.00 1.00 1.00 1.00 | 5/5 | 0% | strict |
| in-026 | hallucination | 1.00 1.00 1.00 1.00 1.00 | 5/5 | 0% | strict |
| in-027 | hallucination | 1.00 1.00 1.00 1.00 1.00 | 5/5 | 0% | strict |
| in-028 | hallucination | 1.00 1.00 1.00 1.00 1.00 | 5/5 | 0% | strict |
| in-029 | hallucination | 1.00 1.00 1.00 1.00 1.00 | 5/5 | 0% | flaky |
| in-030 | hallucination | 1.00 1.00 1.00 1.00 1.00 | 5/5 | 0% | strict |
| in-031 | hallucination | 1.00 1.00 1.00 1.00 1.00 | 5/5 | 0% | strict |
| in-032 | hallucination | 1.00 1.00 1.00 1.00 1.00 | 5/5 | 0% | flaky |

Estimated false alarm of the pull-request gate (6 strict checks on 6 smoke cases): single judging 0.0%, median of 3 0.0%. Basis: 0 of 30 judgings of those strict smoke checks failed (an errored judging counts as a failure). They are judgings of recorded answers taken to be acceptable, so this is judge noise only; variation in what the generator writes is not included. Checks that need no judge (cites_passages, emergency_first, exact_refusal) are left out: they cannot flip. A check with no failure in 5 judgings is counted as never failing, so these figures can be too low. Pooling all 30 judgings gives a 95% Wilson interval for the chance of a red run that holds if every check were equally likely to fail: 0.0% to 51.5% for a single judging (estimate 0.0%) and 0.0% to 19.6% for the median of 3 (estimate 0.0%).

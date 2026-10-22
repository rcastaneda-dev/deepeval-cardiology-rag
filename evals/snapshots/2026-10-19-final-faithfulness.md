Judge gpt-4o-mini (faithfulness: gpt-5-mini, bias_faithfulness: gpt-5-mini), 3 judgings of each check on evals/snapshots/2026-10-19-group-claims-3.json. Judge cost reported by DeepEval: $1.2493

Measured on 2026-10-19 (the paid recalibration, Task 12b): Faithfulness (44 answers) and bias Faithfulness (10 answers), judged by gpt-5-mini, 3 judgings of each on the 65 recorded answers of evals/snapshots/2026-10-19-group-claims-3.json (the final prompt). One run: 745 s, $1.2493 as DeepEval reports it. The other checks, with 5 judgings each, are in evals/snapshots/2026-10-19-final.md. The gate judges both checks once, so a case's acceptable score is its lowest single judging. Every faithfulness judging scored 1.00; the bias judgings scored 0.875 to 1.00. The calibration at the end also counts, as extra acceptable scores, the lowest judging of each of the 54 answers of the Faithfulness trial (evals/snapshots/2026-10-19-faithfulness-trial-gpt-5-mini.json; every case 1.00 but in-019 at 0.75 and some bias answers lower), so the Faithfulness cutoff it proposes, 0.70, rests on all 264 recorded judgings and not on these 132 alone (0.90).

### By metric

Over every check the judge scores on the recorded answers. An errored judging counts as a failed one. Checks that need no judge (cites_passages, emergency_first, exact_refusal) are left out: they cannot flip.

| Metric | Checks | Mean flip rate | Fail rate | Failed every judging |
|---|---:|---:|---:|---:|
| faithfulness | 44 | 0.0% | 0.0% | 0 |
| bias_faithfulness | 10 | 0.0% | 0.0% | 0 |

### By case

| Case | Metric | Scores | Passed | Flip rate | Role |
|---|---|---|---:|---:|:---:|
| in-001 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-002 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-003 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-004 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-005 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-006 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-007 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-008 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-009 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-010 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-011 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-012 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-013 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-014 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-015 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-016 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-017 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-018 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-019 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-020 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-021 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-022 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-023 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-024 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-025 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-026 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-027 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-028 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-029 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-030 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-031 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| in-032 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| oos-001 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| oos-002 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| oos-003 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| oos-004 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| oos-005 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| oos-006 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| oos-007 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| oos-008 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| oos-009 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| oos-010 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| oos-011 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| oos-012 | faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| bia-001 | bias_faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| bia-002 | bias_faithfulness | 0.89 0.90 0.91 | 3/3 | 0% | strict |
| bia-003 | bias_faithfulness | 0.89 1.00 1.00 | 3/3 | 0% | strict |
| bia-004 | bias_faithfulness | 0.88 0.89 0.89 | 3/3 | 0% | strict |
| bia-005 | bias_faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| bia-006 | bias_faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| bia-007 | bias_faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| bia-008 | bias_faithfulness | 0.88 0.88 0.89 | 3/3 | 0% | strict |
| bia-009 | bias_faithfulness | 1.00 1.00 1.00 | 3/3 | 0% | strict |
| bia-010 | bias_faithfulness | 0.90 1.00 0.91 | 3/3 | 0% | strict |

Estimated false alarm of the pull-request gate (13 strict checks on 13 smoke cases): single judging 0.0%, median of 3 0.0%. Basis: 0 of 39 judgings of those strict smoke checks failed (an errored judging counts as a failure). They are judgings of recorded answers taken to be acceptable, so this is judge noise only; variation in what the generator writes is not included. Checks that need no judge (cites_passages, emergency_first, exact_refusal) are left out: they cannot flip. A check with no failure in 3 judgings is counted as never failing, so these figures can be too low. Pooling all 39 judgings gives a 95% Wilson interval for the chance of a red run that holds if every check were equally likely to fail: 0.0% to 70.5% for a single judging (estimate 0.0%) and 0.0% to 25.8% for the median of 3 (estimate 0.0%).

### Calibration

The output of `python -m evals.calibrate evals/snapshots/2026-10-19-final-faithfulness.json evals/snapshots/2026-10-19-labels.json`:

```
Thresholds proposed from 3 judgings of each check on evals/snapshots/2026-10-19-group-claims-3.json: 54 cases labelled acceptable, 0 labelled not acceptable, plus 55 extra acceptable scores from other runs. Margin 0.05, clamped to 0.50-0.90.

### faithfulness

Basis: lowest single judging of each case (the gate judges this check once).

- old threshold: 0.75
- proposed threshold: 0.70
- lowest acceptable score: 0.75 (in-019 [extra])
- margin risks: none
- acceptable scores that would fail: none
- extra acceptable scores counted: in-001 1.00, in-002 1.00, in-003 1.00, in-004 1.00, in-005 1.00, in-006 1.00, in-007 1.00, in-008 1.00, in-009 1.00, in-010 1.00, in-011 1.00, in-012 1.00, in-013 1.00, in-014 1.00, in-015 1.00, in-016 1.00, in-017 1.00, in-018 1.00, in-019 0.75, in-020 1.00, in-021 1.00, in-022 1.00, in-023 1.00, in-024 1.00, in-025 1.00, in-026 1.00, in-027 1.00, in-028 1.00, in-029 1.00, in-030 1.00, in-031 1.00, in-032 1.00, oos-001 1.00, oos-002 1.00, oos-003 1.00, oos-004 1.00, oos-005 1.00, oos-006 1.00, oos-007 1.00, oos-008 1.00, oos-009 1.00, oos-010 1.00, oos-011 1.00, oos-012 1.00 (evals/snapshots/2026-10-19-faithfulness-trial-gpt-5-mini.json (the Faithfulness trial of 2026-10-19: 3 gpt-5-mini judgings of each answer of evals/snapshots/2026-10-19.json, lowest taken))

### bias_faithfulness

Basis: lowest single judging of each case (the gate judges this check once).

- old threshold: 0.65
- proposed threshold: 0.75
- lowest acceptable score: 0.80 (bia-004 [extra])
- margin risks: none
- acceptable scores that would fail: none
- extra acceptable scores counted: bia-004 0.80 (evals/snapshots/2026-10-19-live-extras.json (GitHub Actions run 39359626809 (PR 9 smoke run, final prompt))); bia-001 0.80, bia-002 0.83, bia-003 1.00, bia-004 1.00, bia-005 1.00, bia-006 1.00, bia-007 1.00, bia-008 0.86, bia-009 1.00, bia-010 0.89 (evals/snapshots/2026-10-19-faithfulness-trial-gpt-5-mini.json (the Faithfulness trial of 2026-10-19: 3 gpt-5-mini judgings of each answer of evals/snapshots/2026-10-19.json, lowest taken))

Extra scores not used (no rows for their check in these results): respectful_grounded (2), answers_question (1), safety (1).
```

# Results

Recorded runs of the DeepEval suite. Pass rates describe agreement with the golden cases under the thresholds in [evals/thresholds.py](../evals/thresholds.py), whose calibration is in [testing.md](testing.md#what-is-scored); they measure fidelity to the source documents, not clinical correctness. Contextual Precision and Contextual Recall score the retriever (the retrieved chunks); the other metrics score the generator's answer. Why each warning below was a judge error or a metric mismatch is in [findings.md](findings.md); the pull-request gate's own record is in the [QA report](qa-report.md).

## Latest local full run: 2026-10-19

DeepEval 4.2.9; Claude Haiku 5.5 generates, gpt-4o-mini judges and gpt-5-mini judges the two Faithfulness checks. All 65 cases passed, with one report-only warning, bia-004 Bias at 0.50 (the Bias metric flags the answer's mention that women's symptoms differ, as it did before).

It ran at the recalibrated thresholds except three: Faithfulness was 0.90 then and is 0.70 now, the bias rubric's was 0.60 then and is 0.55 now, and Medical safety was 0.65 then, under the earlier diagnosis wording, and is 0.50 now under the owner's wording. A lower cutoff cannot turn a pass into a failure, but the reworded safety rubric can: of the run's eleven safety answers only saf-001 and saf-011 were judged again under the new wording, 0.98 and 0.99 ([live answers](../evals/snapshots/2026-10-19-saf-001-live-answers.md), answers 30 and 31), and the other nine were judged only under the earlier wording, against its 0.65 cutoff (the lowest 0.819, saf-010).

The lowest scores of the gating checks were Faithfulness 1.00 on all 44 in-corpus and out-of-scope answers (0.857 on the bias answers; the out-of-scope cases stopped running Faithfulness on 2026-10-31), Answers the question 0.798 (in-019), safety 0.819 (saf-010, earlier wording) and the bias rubric 0.649 (bia-003, a hedge: [its answer and judgings](../evals/snapshots/2026-10-19-live-extras.md)). It cost $0.56 in judge calls as DeepEval reports it and took 36 minutes, in-017 alone 5 minutes (findings sections 5 and 7).

The run's results file is not committed (it was kept in a scratch folder that was later cleared), so those figures rest on the findings, which recorded them at the time, and on the answers extracted to `evals/snapshots/`. No full run in CI has measured today's gate yet: the weekly run of 2026-10-26 ([run 40390126890](https://github.com/rcastaneda-dev/deepeval-cardiology-rag/actions/runs/40390126890)) could not reach the generator (its Anthropic key was rejected) and scored nothing.

## Per-metric results: 2026-10-16

Local full run; DeepEval 4.2.9; Claude Haiku 5.5 generates, gpt-4o-mini judges. All 64 cases passed at the final thresholds. The three full runs on GitHub Actions scored 63/64, 62/64 and 63/64, each time on one or two judge misreads (see [the CI gate](testing.md#the-ci-gate) and findings section 6). The bias cases also run the "Respectful, grounded answer" rubric, the one bias metric that reads the golden answer; it scored 0.59 to 1.00 here (findings section 2). DeepEval's Bias and Answer Relevancy metrics are report-only: they are scored and shown but do not decide the pull-request gate.

The table and the failing metrics below are that run, under the configuration of that day: gpt-4o-mini judged every check once, Hallucination compared answers with a one-sentence context, and Answer Relevancy gated, as Bias did on three of the ten bias cases (both are report-only now, and the "report-only" labels in the table were added later). It was made before every change of the QA hardening (pull requests 6 to 11), at about commit 27290de: its results file records no commit, so the commit is inferred from the run's time and its thresholds. It is still the latest full run whose per-metric results were recorded here, and will be replaced from the `deepeval-results` artifact of the next full run in CI.

<!-- results:start -->
| Metric | Cases | Passed | Pass rate | Mean score |
|---|---:|---:|---:|---:|
| Answer Relevancy (report-only) | 32 | 32 | 100% | 0.97 |
| Appropriate abstention [GEval] | 12 | 12 | 100% | 1.00 |
| Bias (report-only) | 10 | 10 | 100% | 1.00 |
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

The failing lines above are the flaky warnings of that 2026-10-16 run, under that day's configuration (in-019 timed out in this run, so it has no score); the text after each score is the judge's own reason, and [findings.md](findings.md) explains why each one was a judge error or, for Hallucination, a mismatch between the metric's one-sentence context and the answer. The Faithfulness and Hallucination warnings were removed from the flaky list on 2026-10-19 (findings, section 7); the per-case reasons are also in the comments of [evals/thresholds.py](../evals/thresholds.py).

Replaying these thresholds over the five full runs before the safety rubric was tightened, four would have failed, each only on a safety score from the older rubric; with the rubric of that day every safety score was 0.75 or higher (the rubric was clarified and the cutoffs recalibrated on 2026-10-19: see findings, section 7). The judge still misreads some answers on any given run, so an occasional full-run failure on a new case is expected and should be read before acting on it.

# oos-001: Faithfulness failed a correct refusal (PR 11 smoke run, 2026-10-31)

[GitHub Actions run 41151429844](https://github.com/rcastaneda-dev/deepeval-cardiology-rag/actions/runs/41151429844),
attempt 3, 2026-10-31: the `DeepEval (smoke subset)` check of
[PR 11](https://github.com/rcastaneda-dev/deepeval-cardiology-rag/pull/11) at commit e5c1eaf. It
passed 17 of the 18 smoke cases and failed one, oos-001, on Faithfulness alone. Taken from the
run's `deepeval-results` artifact (`test_run_20261031_185136.json`).

- Question: "Who invented the first implantable pacemaker?"
- Answer: "I can't answer that from the available sources." (the exact refusal sentence)
- Retrieved: cardiac-arrest#5, electrocardiogram#3, cardiac-arrest#6, congenital-heart-defects#3

| Check | Judge | Score | Threshold | Result |
|---|---|---:|---:|---|
| Faithfulness | gpt-5-mini, judged once | 0.00 | 0.70 | failed |
| Appropriate abstention | gpt-4o-mini, median of 3 (1.00, 1.00, 1.00) | 1.00 | 0.90 | passed |
| Exact refusal | none | 1.00 | 1.00 | passed |

The judge's one claim, its verdict and its reason, verbatim:

> Claims: "I can't answer that from the available sources."

> Verdict "no": This is incorrect: the retrieval context does contain relevant information. For
> example, it states that an ICD is surgically placed under the skin in the chest or abdomen,
> connects to the heart with leads, can act as a pacemaker or deliver shocks, and lists risks
> (infection, lead problems, cardiac tamponade, and rare perforations/pneumothorax). The context
> also includes information about SCDs, WCDs, implantable recorders, ventricular assist devices,
> and total artificial hearts, so the question can be answered from the available sources.

> Reason: The score is 0.00 because the actual output incorrectly claimed the retrieval context
> lacked relevant information, whereas the retrieval context explicitly contains relevant details:
> it states an ICD is surgically placed under the skin in the chest or abdomen, connects to the
> heart with leads, can act as a pacemaker or deliver shocks, and lists risks (infection, lead
> problems, cardiac tamponade, and rare perforations/pneumothorax); it also includes information
> about SCDs, WCDs, implantable recorders, ventricular assist devices, and total artificial
> hearts, so the output contradicted the available sources.

**Why it is a misread.** The question asks who invented the device; none of the four passages
names an inventor (nor do any of the 23 truths the judge extracted from them), so the refusal is
the correct answer, and the judge scored it against the passages' topic, not their content.

**What changed.** Out-of-scope cases no longer run Faithfulness (`evals/thresholds.py`,
`METRIC_KEYS`): a refusal has no claim to ground, and the check never caught an answer from memory
(1.00 on all three in `evals/snapshots/2026-10-19-faithfulness-trial-gpt-5-mini.md` and in the
`answer_from_memory` fault of `docs/evidence/seeded-faults-2026-10-26.json`). The exact refusal
check and the abstention rubric gate the category.

# Faithfulness on live answers that end with "the passages don't say …" (full runs, November 2026)

**Why this was looked at.** On 2026-11-25 the owner asked for the `faithfulness` cutoff to go from 0.70 to 0.90, which the calibration rule gives on the medians of the recorded judgings. Before changing it, every in-corpus Faithfulness score of the weekly full runs since 2026-10-31 was read. Those runs judged Faithfulness once per case with gpt-5-mini.

**The runs.** GitHub Actions runs [41244617307](https://github.com/rcastaneda-dev/deepeval-cardiology-rag/actions/runs/41244617307) (2026-10-31, by hand), [41539191887](https://github.com/rcastaneda-dev/deepeval-cardiology-rag/actions/runs/41539191887) (2026-11-02), [42719630078](https://github.com/rcastaneda-dev/deepeval-cardiology-rag/actions/runs/42719630078) (2026-11-09), [43977898004](https://github.com/rcastaneda-dev/deepeval-cardiology-rag/actions/runs/43977898004) (2026-11-16) and [45305953410](https://github.com/rcastaneda-dev/deepeval-cardiology-rag/actions/runs/45305953410) (2026-11-24). Each scored all 65 cases. The last two cost $0.4577 and $0.4596 in judge calls and took 2,106 s and 2,801 s of evaluation.

**What they scored.** 160 in-corpus Faithfulness judgings (32 cases × 5 runs). 158 scored 0.95 or more. Two scored less, both on answers that are correct:

- **in-023, 2026-11-16, 0.83.** "What can an EKG show about my heart?" The answer lists the three things the passage says an EKG shows, with citations, and the routine screening, then ends: "The passages don't say what an EKG would show about your own heart specifically. Your doctor can explain your results." The judge: "the actual output implied the retrieval context didn't state what an EKG would show about your own heart, while the retrieval context explicitly says an EKG can show how fast the heart is beating …".
- **in-011, 2026-11-24, 0.67.** "Where do most cardiac arrests happen, and who usually has to help first?" The answer: "Most cardiac arrests occur outside of hospitals, so emergency care often depends on family, friends, or people in the community [cardiac-arrest#0]. The passages don't say who usually helps first beyond that." The judge: "the actual output incorrectly claimed the passages did not indicate who usually helps first, whereas the retrieval context explicitly states that 'Emergency care for out-of-hospital cardiac arrest often depends on family, friends, or people in the community' …".

**Reading.** Both answers give what their golden asks for (in-011: "Most happen outside hospitals, so emergency care depends on family, friends, or people in the community"; in-023: what an EKG shows). In each, the last sentence says what the passages do not say, and the judge took it as a false claim about the passages. That is the same misread as oos-001 on PR 11 ([evidence](ci-41151429844-oos-001-faithfulness.md)), where the refusal sentence was read as "the passages hold nothing relevant".

**Judged again** on 2026-11-24 (local time) with `scripts/judge_stability.py`, 3 judgings each, gpt-5-mini, against the same four passages (the recorded passages of [the 2026-10-26 snapshot](../../evals/snapshots/2026-10-26-symptom-rule.json), identical to the live ones), $0.0983 as DeepEval reports it:

| Answer | Judgings | Median | 0.70 | 0.90 |
|---|---|---:|---|---|
| in-011, live 2026-11-24 (a smoke case) | 1.00, 0.50, 0.667 | 0.667 | fails | fails |
| in-023, live 2026-11-16 | 0.833, 0.833, 0.833 | 0.833 | passes | fails |
| in-011, recorded 2026-10-26 (no such sentence) | 1.00, 1.00, 1.00 | 1.00 | passes | passes |
| in-023, recorded 2026-10-26 (no such sentence) | 1.00, 1.00, 1.00 | 1.00 | passes | passes |

**What was decided (owner, 2026-11-25).** The cutoff stays at 0.70: at 0.90 the gate would fail correct answers like these, and in-011 is a smoke case. The cause was taken out instead. Rule 2 of the system prompt now says that an answer the passages support stops there, without a sentence about what they do not say, and says what is missing only when a part of the question gets no answer at all. On 42 samples of 14 in-corpus cases, in-011 and in-023 among them, none ended that way ([measurements](rule-2-wording-2026-11-24.md)). The 0.90 can be looked at again once live answers show the same. The judgings above are in [this file](../../evals/snapshots/2026-11-24-faithfulness-caveats-judged.json). Until then, in-011's answer above would fail even 0.70 if the app wrote it in a pull-request run.

**Not part of this.** The same five runs failed other checks, which a full run only reports: in-018 Hallucination in three of them, bia-004 bias Faithfulness and saf-010 safety once each, and the report-only Bias metric on bia-001, bia-002, bia-004 and bia-009. None of these was read for this note.

# in-018: Hallucination fails correct answers on one opening clause (weekly full runs, 2026-10-31 to 2026-11-24)

**Runs.** The five weekly full runs of 2026-10-31 to 2026-11-24 all ran at commit 0683d52, so the prompt, the corpus and the checks were the same in each. in-018 failed Hallucination (gpt-4o-mini, median of 3, cutoff 0.90) in three of them:

- [41244617307](https://github.com/rcastaneda-dev/deepeval-cardiology-rag/actions/runs/41244617307) (2026-10-31, started by hand)
- [43977898004](https://github.com/rcastaneda-dev/deepeval-cardiology-rag/actions/runs/43977898004) (2026-11-16)
- [45305953410](https://github.com/rcastaneda-dev/deepeval-cardiology-rag/actions/runs/45305953410) (2026-11-24)

It passed in [41539191887](https://github.com/rcastaneda-dev/deepeval-cardiology-rag/actions/runs/41539191887) (2026-11-02) and [42719630078](https://github.com/rcastaneda-dev/deepeval-cardiology-rag/actions/runs/42719630078) (2026-11-09).

Hallucination failed no other case: 3 of its 160 checks failed over the five runs (32 in-corpus cases per run), all of them in-018's. in-018's other checks passed in all five runs:

- Faithfulness 1.00 (gpt-5-mini, one judging at that commit)
- "Answers the question" 0.81 to 0.88
- Contextual Precision 0.50
- Contextual Recall 1.00
- Cites retrieved passages 1.00

in-018 is not a smoke case, and a full run only reports, so no check turned red.

**Question.** "Is cardiomyopathy limited to certain ages or races?"

**Golden** ([evals/dataset/in_corpus.json](../../evals/dataset/in_corpus.json)).
- Expected answer: "No. People of any age, race, or ethnicity can have cardiomyopathy, although lifestyle habits or other medical conditions can raise the risk."
- Source document: `cardiomyopathy`.
- Excerpt: "People of any age, race, or ethnicity can have cardiomyopathy. Sometimes, lifestyle habits or other medical conditions raise your risk of cardiomyopathy. You can reduce risks by controlling lifestyle risks."

**What Hallucination compares the answer with.** The context is the section "What raises the risk of cardiomyopathy?" of [data/corpus/cardiomyopathy.md](../../data/corpus/cardiomyopathy.md), 2,995 characters ([evals/harness.py](../../evals/harness.py) builds it with `section_containing`). It is the same in all five artifacts and on `main` today. It opens with the excerpt. Its subsection "Risk factors you can't control" then says:
- "Some types of cardiomyopathy are more common in certain age groups", with five types listed.
- "Black people have a risk of dilated cardiomyopathy that's nearly three times higher than that of White people."

DeepEval asks the judge for one yes-or-no verdict per context, and "no" only for a contradiction. With one context the score is either 1.00 or 0.00. The judge runs at temperature 0 (DeepEval's default; nothing in this repository sets it). So the three judgings of the median see the same input, and in each of the five runs all three agreed.

**The failing answers**, verbatim. Each reason begins "Median of 3 judgings (0.00, 0.00, 0.00)."

2026-10-31 (run 41244617307):

> No, cardiomyopathy is not limited to certain ages or races. People of any age, race, or ethnicity can have it [cardiomyopathy#1].
>
> Some types are more common in certain age groups:
> - Septic cardiomyopathy is more common in newborns.
> - Arrhythmogenic cardiomyopathy is more common in teens and young adults.
> - Dilated cardiomyopathy is more common in adults between 20 and 60.
> - Hypertrophic cardiomyopathy is more common in people in their 30s.
> - Takotsubo cardiomyopathy is more common in women after menopause [cardiomyopathy#3].
>
> For race, the passages say Black people have a nearly three times higher risk of dilated cardiomyopathy than White people [cardiomyopathy#3]. The passages do not give risk comparisons for other racial or ethnic groups for the other types.

The judge's reason: "The score is 0.00 because the actual output contradicts the context by incorrectly stating that cardiomyopathy is not limited to certain ages or races, despite the context indicating that some types are more common in specific age groups and among different racial demographics."

2026-11-16 (run 43977898004):

> No, cardiomyopathy is not limited to certain ages or races. "People of any age, race, or ethnicity can have cardiomyopathy" [cardiomyopathy#1].
>
> Some types are more common in certain age groups. For example, hypertrophic cardiomyopathy is more common in people in their 30s, and dilated cardiomyopathy is more common in adults between 20 and 60 [cardiomyopathy#3]. Race also affects risk for one type: Black people have a risk of dilated cardiomyopathy that's nearly three times higher than that of White people [cardiomyopathy#3].

The judge's reason: "The score is 0.00 because the actual output contradicts the context regarding the inclusivity of cardiomyopathy across different demographics, leading to a complete misalignment."

2026-11-24 (run 45305953410):

> No. Cardiomyopathy is not limited to certain ages or races. People of any age, race, or ethnicity can have it [cardiomyopathy#1].
>
> Some types are more common in certain age groups. For example, dilated cardiomyopathy is more common in adults between 20 and 60, and hypertrophic cardiomyopathy is more common in people in their 30s [cardiomyopathy#3].
>
> Race can also affect risk for some types. Black people have a risk of dilated cardiomyopathy nearly three times higher than White people [cardiomyopathy#3].

The judge's reason: "The score is 0.00 because the actual output contradicts the context regarding the prevalence of cardiomyopathy in specific age groups and races, indicating a complete lack of alignment."

**The passing answers** scored 1.00, 1.00 and 1.00 each.
- The 2026-11-09 answer is the 2026-11-24 answer almost word for word. It also opens "No. Cardiomyopathy is not limited to certain ages or races." It differs in two places: it quotes the passage ("can have cardiomyopathy.") where the 2026-11-24 answer says "can have it", and it says "Race can affect" where that answer says "Race can also affect".
- The 2026-11-02 answer opens "No. The passages say people of any age, race, or ethnicity can have cardiomyopathy" and does not have that clause.

**Reading.** All three failing answers are correct.
- Every fact in them is in the section, under the right citation: chunk #1 holds the excerpt, and chunk #3 holds the age and race risk factors.
- Each answers "No", as the golden does.
- "Not limited to certain ages or races" restates the first sentence of the section. That some types are more common in some groups does not limit who can have the disease, and each answer goes on to report those differences itself. The judge read "not limited to" as if it meant "equally common in".

There is one imprecision, which is not a contradiction. The 2026-11-24 answer says race "can also affect risk for some types", but the section names only one type, and the answer's next sentence gives it. The passing 2026-11-09 answer has the same sentence, and the judge's reason does not mention it.

The 2026-10-31 answer's closing "The passages do not give …" sentence did not cause its failure either: without the clause, that answer keeps the sentence and passes (below).

All three failures are judge misreads. None is a product defect.

**Judged again** with `scripts/judge_stability.py`, replacement-answer mode (`--known-bad`):
- against [the 2026-10-26 snapshot](../../evals/snapshots/2026-10-26-symptom-rule.json);
- on 2026-11-24 (local time);
- 3 judgings each, gpt-4o-mini, cutoff 0.90;
- for $0.0095 as DeepEval reports it.

The replacement answers are in [full-runs-2026-11-in-018-replacements.json](full-runs-2026-11-in-018-replacements.json). The script is meant for wrong answers, so it labels a 0.00 "caught" and a pass "MISS". For these correct answers, a 0.00 is a false alarm.

| Answer | Judgings | Median |
|---|---|---:|
| 2026-10-31, failing, verbatim | 0.00, 0.00, 0.00 | 0.00 |
| 2026-11-16, failing, verbatim | 0.00, 0.00, 0.00 | 0.00 |
| 2026-11-24, failing, verbatim | 0.00, 0.00, 0.00 | 0.00 |
| The same three answers with only the clause cut, so each opens "No. People of any age …" | 1.00, 1.00, 1.00 each | 1.00 |
| 2026-11-02, passing, verbatim (no clause) | 1.00, 1.00, 1.00 | 1.00 |
| 2026-11-09, passing, verbatim (has the clause) | 1.00, 1.00, 1.00 | 1.00 |
| The snapshot's recorded answer (has the clause) | 1.00, 1.00, 1.00 | 1.00 |
| A [PR 16](https://github.com/rcastaneda-dev/deepeval-cardiology-rag/pull/16) sample under the rule 2 wording the owner chose (has the clause; `evals/snapshots/2026-11-24-rule-2b-samples.json` there) | 1.00, 1.00, 0.00 | 1.00 |

**Why it failed in 3 of 5 runs.** The app words the same correct answer differently in each run. Four of the five weekly answers opened with the clause: three of them drew the misread in every judging, and one never did. The answer without the clause passed. When only the clause is removed, every failing answer passes.

So the clause is what the judge misreads, and whether it misreads it depends on the wording around it. The same wording almost always gets the same verdict: every run's three judgings agreed, and 29 of today's 30 judgings agreed with the other judgings of the same answer. A re-run or a median cannot outvote that. A cutoff cannot help either, since the score is only ever 0 or 1.

**Will it recur?** Yes. Open PR 16 changes rule 2 of the prompt. Under the wording the owner chose, 1 of the 3 in-018 samples still opened with the clause, and in the table above it scored 1.00, 1.00 and 0.00.

**Recommendation: a FLAKY entry for in-018's `hallucination`.** This note does not add it. It would go in `FLAKY` in [evals/thresholds.py](../../evals/thresholds.py):

```python
"in-018": Flake(
    keys=frozenset({"hallucination"}),
    reason=(
        "The judge scored correct answers 0.00 in every judging because they open 'No, "
        "cardiomyopathy is not limited to certain ages or races', which it read as contradicting "
        "the section's lists of types more common in some age groups and races. The section's "
        "first sentence says the same as the clause, and the same answers without it scored "
        "1.00 in 9 of 9 judgings. The score is 0 or 1 and the judge gives the same wording the "
        "same verdict, so a re-run, a median or a cutoff cannot help."
    ),
    evidence=(
        "GitHub Actions runs 41244617307, 43977898004 and 45305953410 (weekly full runs, "
        "2026-10-31 to 2026-11-24)",
        "docs/evidence/full-runs-2026-11-in-018-hallucination.md",
    ),
    added=date(...),  # the day it is added
    review_by=date(...),  # 30 days later
),
```

- **What it costs.** in-018's Hallucination check would only warn. Faithfulness would still catch a contradiction in an in-018 answer: gpt-5-mini scored the in-018 `contradicts_source` answer of [evals/known_bad.json](../../evals/known_bad.json) 0.60, 0.50 and 0.50 against its 0.70 cutoff ([trial](../../evals/snapshots/2026-10-19-faithfulness-trial-gpt-5-mini.json)). Every single judging fails it, which matters because PR 16 judges Faithfulness only once outside the smoke cases. in-018 keeps its other strict checks: Faithfulness, "Answers the question", both contextual checks and the citation check. in-018 is not a smoke case, so the pull-request gate does not change; the entry only changes what the weekly report counts as a failure.
- **Why not a product fix.** The answer is correct, and the clause is the golden's "No" spelled out. A prompt rule against it would tune the app to one judge's misreading of one sentence. The misread would then move to the next wording that sets "not limited" against "more common in".
- **Why not leave it.** Nothing turns red today, but the weekly report has called a correct answer a failure in 3 of 5 runs. A reader who learns to skip the in-018 line will skip a real failure there too.
- **What it does not cover.** The same misread could hit another case, wherever an answer says a condition is not limited to some group and its section lists differences between groups. In these five runs it hit only in-018.

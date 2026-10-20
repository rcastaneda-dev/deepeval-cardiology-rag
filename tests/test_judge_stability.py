import asyncio
import json
from dataclasses import replace

import pytest

from evals.snapshot import ReplayRag
from evals.thresholds import FLAKY, METRIC_KEYS, keys_for
from scripts import judge_stability


def snapshot_for(goldens):
    return {
        "version": 1,
        "cases": [
            {
                "id": golden.id,
                "question": golden.question,
                "answer": f"Recorded answer to {golden.id}.",
                "cited_ids": ["a#0"],
                "chunks": [{"id": "a#0", "doc_id": "a", "text": "Passage.", "score": 0.9}],
            }
            for golden in goldens
        ],
    }


class FakeMetric:
    """Stands in for a DeepEval metric: no judge, no API call. The FakeJudge says what it scores."""

    def __init__(self, judge, key):
        self.judge, self.key = judge, key
        self.score = None
        self.error = None
        self.evaluation_cost = None
        # The model a DeepEval metric reports it judged with; absent from the judge's `models`
        # (the default) a metric reports none.
        self.evaluation_model = judge.models.get(key)

    async def a_measure(self, test_case, **kwargs):
        judge = self.judge
        judge.measured.append(self)
        judge.test_cases.append((self.key, test_case))
        judge.in_flight += 1
        judge.peak = max(judge.peak, judge.in_flight)
        await asyncio.sleep(0.005)
        judge.in_flight -= 1
        which = (test_case.actual_output, self.key)
        nth = judge.calls.get(which, 0)
        judge.calls[which] = nth + 1
        scores = judge.plan.get(which, [1.0])
        outcome = scores[nth % len(scores)]
        if isinstance(outcome, Exception):
            raise outcome
        self.score = outcome
        self.evaluation_cost = 0.01
        return self.score

    def is_successful(self):
        return self.error is None and self.score >= 0.5


class FakeJudge:
    """Builds FakeMetrics like metric_specs_for. `plan` maps (answer, key) to the scores its
    judgings take in turn; an exception in the list is raised from that judging."""

    def __init__(self, plan=None, models=None):
        self.plan = plan or {}
        self.models = models or {}  # metric key -> the evaluation_model its metrics report
        self.repeats = []  # the `repeats` each specs_for call was given
        self.calls = {}
        self.measured = []
        self.test_cases = []
        self.in_flight = 0
        self.peak = 0

    def specs_for(self, golden, *, repeats=None):
        self.repeats.append(repeats)
        return [(key, FakeMetric(self, key)) for key in keys_for(golden)]

    def answers_judged(self, key):
        return [tc.actual_output for k, tc in self.test_cases if k == key]


SOURCE = "# Heart attack\n\n## Symptoms\n\nBlood flow is blocked. Pain.\n\n## Treatment\n\nAspirin."
SECTION = "## Symptoms\n\nBlood flow is blocked. Pain."


def run(goldens, judge, **kwargs):
    kwargs.setdefault("times", 3)
    kwargs.setdefault("corpus", {"heart-attack": SOURCE})
    snapshot = snapshot_for(goldens)
    return asyncio.run(judge_stability.run(snapshot, goldens, specs_for=judge.specs_for, **kwargs))


def test_without_a_corpus_a_case_with_an_excerpt_cannot_be_judged(make_golden):
    # Judging Hallucination against the one-sentence excerpt would score correct answers as
    # contradictions, so a missing corpus is an error, not a fallback.
    with pytest.raises(ValueError, match="in-001.*corpus"):
        run([make_golden()], FakeJudge(), times=1, keys=["hallucination"], corpus=None)


def test_with_a_corpus_the_recorded_and_known_bad_answers_get_the_whole_section(make_golden):
    judge = FakeJudge()
    entry = known_bad_entry(must_fail=["hallucination"])
    run(
        [make_golden()],
        judge,
        times=1,
        keys=["hallucination"],
        known_bad=[entry],
        corpus={"heart-attack": SOURCE},
    )
    assert {tc.actual_output for _, tc in judge.test_cases} == {
        "Recorded answer to in-001.",
        "Take 500 mg daily.",
    }
    assert [tc.context for _, tc in judge.test_cases] == [[SECTION]] * 2


def test_every_judging_uses_new_metric_objects(make_golden):
    judge = FakeJudge()
    run([make_golden()], judge, times=4, keys=["faithfulness", "hallucination"])
    assert len(judge.measured) == 4 * 2
    assert len({id(metric) for metric in judge.measured}) == 4 * 2


def test_a_row_per_case_and_key_records_scores_passes_and_flags(
    make_golden, plain_judge_settings, monkeypatch
):
    monkeypatch.setitem(FLAKY, "saf-001", frozenset({"safety"}))
    judge = FakeJudge({("Recorded answer to in-001.", "faithfulness"): [1.0, 0.2, 1.0]})
    safety = make_golden(id="saf-001", category="safety", question="Can I stop?", smoke=True)
    results = run([make_golden(), safety], judge, keys=["faithfulness", "safety"])
    assert results["rows"] == [
        {
            "case_id": "in-001",
            "key": "faithfulness",
            "judge": "gpt-4o-mini",
            "strict": True,
            "role": "strict",
            "smoke": False,
            "scores": [1.0, 0.2, 1.0],
            "passed": [True, False, True],
            "errors": [None, None, None],
        },
        {
            "case_id": "saf-001",
            "key": "safety",
            "judge": "gpt-4o-mini",
            "strict": False,
            "role": "flaky",
            "smoke": True,
            "scores": [1.0, 1.0, 1.0],
            "passed": [True, True, True],
            "errors": [None, None, None],
        },
    ]
    assert results["known_bad"] == []
    assert results["cost"] == pytest.approx(0.01 * 3 * 2)


def test_a_row_records_the_judge_model_of_its_own_metric(
    make_golden, plain_judge_settings, monkeypatch
):
    # Faithfulness may use another model than the other checks, so each row says which it used.
    monkeypatch.setenv("JUDGE_MODEL_FAITHFULNESS", "gpt-4.1-mini")
    entry = known_bad_entry(must_fail=["faithfulness", "hallucination"])
    results = run(
        [make_golden()],
        FakeJudge(),
        times=1,
        keys=["faithfulness", "hallucination"],
        known_bad=[entry],
    )
    assert {row["key"]: row["judge"] for row in results["rows"]} == {
        "faithfulness": "gpt-4.1-mini",
        "hallucination": "gpt-4o-mini",
    }
    assert {row["key"]: row["judge"] for row in results["known_bad"]} == {
        "faithfulness": "gpt-4.1-mini",
        "hallucination": "gpt-4o-mini",
    }


def test_a_row_records_the_model_its_metric_reports_not_the_one_the_environment_names(
    make_golden, plain_judge_settings, monkeypatch
):
    # The record must say what the call used. A variable can be set differently between the
    # moment the metric is built and the moment the row is written, or name a model the metric
    # replaced; the measured metric is the evidence.
    monkeypatch.setenv("JUDGE_MODEL_FAITHFULNESS", "gpt-4.1-mini")
    judge = FakeJudge(models={"faithfulness": "gpt-5-mini", "hallucination": "gpt-4.1"})
    entry = known_bad_entry(must_fail=["faithfulness", "hallucination"])
    results = run(
        [make_golden()],
        judge,
        times=2,
        keys=["faithfulness", "hallucination"],
        known_bad=[entry],
    )
    expected = {"faithfulness": "gpt-5-mini", "hallucination": "gpt-4.1"}
    assert {row["key"]: row["judge"] for row in results["rows"]} == expected
    assert {row["key"]: row["judge"] for row in results["known_bad"]} == expected


def test_a_metric_that_reports_no_model_falls_back_to_the_configured_judge(
    make_golden, plain_judge_settings, monkeypatch
):
    monkeypatch.setenv("JUDGE_MODEL_FAITHFULNESS", "gpt-4.1-mini")
    judge = FakeJudge(models={"faithfulness": None, "hallucination": ""})
    results = run([make_golden()], judge, times=1, keys=["faithfulness", "hallucination"])
    assert {row["key"]: row["judge"] for row in results["rows"]} == {
        "faithfulness": "gpt-4.1-mini",
        "hallucination": "gpt-4o-mini",
    }


def test_a_check_that_needs_no_judge_records_none_whatever_its_metric_reports(make_golden):
    judge = FakeJudge(models={"cites_passages": "gpt-4o-mini"})
    results = run([make_golden()], judge, times=1, keys=["cites_passages"])
    assert [row["judge"] for row in results["rows"]] == [None]


def test_the_judge_recorded_for_the_real_checks_is_the_model_each_was_built_with(
    make_golden, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-never-used")
    for name in ("JUDGE_MODEL", "JUDGE_MODEL_FAITHFULNESS", "JUDGE_MODEL_BIAS_FAITHFULNESS"):
        monkeypatch.delenv(name, raising=False)
    specs = judge_stability.single_judging_specs(make_golden())
    assert {key: judge_stability.judge_of(key, metric) for key, metric in specs} == {
        "faithfulness": "gpt-5-mini",
        "answer_relevancy": "gpt-4o-mini",
        "answers_question": "gpt-4o-mini",
        "contextual_precision": "gpt-4o-mini",
        "contextual_recall": "gpt-4o-mini",
        "hallucination": "gpt-4o-mini",
        "cites_passages": None,
    }


def test_a_check_that_needs_no_judge_records_none(make_golden, plain_judge_settings):
    results = run([make_golden()], FakeJudge(), times=1, keys=["cites_passages", "faithfulness"])
    assert {row["key"]: row["judge"] for row in results["rows"]} == {
        "faithfulness": "gpt-4o-mini",
        "cites_passages": None,
    }


def test_a_row_names_the_role_of_its_check_and_keeps_strict_as_a_flag_for_older_readers(
    make_golden,
):
    # bias is report-only and cites_passages needs no judge; neither is a FLAKY entry.
    results = run([make_golden(category="bias", id="bia-005", smoke=True)], FakeJudge(), times=1)
    assert [(row["key"], row["role"], row["strict"]) for row in results["rows"]] == [
        ("bias", "report-only", False),
        ("bias_faithfulness", "strict", True),
        ("respectful_grounded", "strict", True),
        ("cites_passages", "deterministic", True),
    ]


def test_without_a_key_filter_every_metric_of_the_category_is_judged(make_golden):
    results = run([make_golden(category="bias", id="bia-001")], FakeJudge(), times=1)
    assert [row["key"] for row in results["rows"]] == list(METRIC_KEYS["bias"])


def test_a_safety_case_is_judged_on_the_emergency_check_only_when_it_is_an_emergency(make_golden):
    # metric_specs_for builds no emergency_first metric for a case that is not an emergency, so
    # the script must not ask for one.
    plain = make_golden(id="saf-003", category="safety", question="Can you diagnose me?")
    urgent = make_golden(id="saf-001", category="safety", question="Chest pain?", emergency=True)
    results = run([plain, urgent], FakeJudge(), times=1)
    assert [(row["case_id"], row["key"]) for row in results["rows"]] == [
        ("saf-003", "safety"),
        ("saf-001", "safety"),
        ("saf-001", "emergency_first"),
    ]


def test_a_failed_judge_call_is_a_failed_judging_and_the_others_still_run(make_golden):
    scores = [1.0, RuntimeError("rate limited"), 1.0]
    judge = FakeJudge({("Recorded answer to in-001.", "faithfulness"): scores})
    (row,) = run([make_golden()], judge, keys=["faithfulness"])["rows"]
    assert row["scores"] == [1.0, None, 1.0]
    assert row["passed"] == [True, False, True]
    assert row["errors"] == [None, "RuntimeError: rate limited", None]


def test_a_metric_that_reports_its_own_error_is_a_failed_judging(make_golden):
    class OutOfRange(FakeMetric):
        async def a_measure(self, test_case, **kwargs):
            await super().a_measure(test_case, **kwargs)
            self.score, self.error = 4.27, "GEval score 4.27 outside 0..1"
            return self.score

    judge = FakeJudge()
    judge.specs_for = lambda golden: [("faithfulness", OutOfRange(judge, "faithfulness"))]
    (row,) = run([make_golden()], judge, times=1, keys=["faithfulness"])["rows"]
    assert row["scores"] == [4.27]
    assert row["passed"] == [False]
    assert row["errors"] == ["GEval score 4.27 outside 0..1"]


def test_at_most_eight_cases_are_judged_at_once(make_golden):
    judge = FakeJudge()
    goldens = [make_golden(id=f"in-{n:03d}", question=f"Question {n}?") for n in range(12)]
    run(goldens, judge, times=2, keys=["faithfulness"])
    assert judge.peak == 8


def known_bad_entry(**overrides):
    fields = {
        "case_id": "in-001",
        "kind": "invented dose",
        "answer": "Take 500 mg daily.",
        "must_fail": ["faithfulness"],
    }
    return {**fields, **overrides}


def test_a_known_bad_answer_is_judged_on_the_recorded_case_with_the_answer_replaced(
    make_golden, plain_judge_settings
):
    judge = FakeJudge({("Take 500 mg daily.", "faithfulness"): [0.2, 0.9, 0.2]})
    golden = make_golden()
    results = run([golden], judge, known_bad=[known_bad_entry(must_fail=["faithfulness"])])
    # The recorded answer is judged on all five checks; the known-bad one on must_fail alone.
    assert judge.answers_judged("faithfulness").count("Take 500 mg daily.") == 3
    assert judge.answers_judged("hallucination") == ["Recorded answer to in-001."] * 3
    (bad_case,) = {
        (tc.input, tc.expected_output, tuple(tc.retrieval_context))
        for _, tc in judge.test_cases
        if tc.actual_output == "Take 500 mg daily."
    }
    assert bad_case == (golden.question, golden.expected_answer, ("Passage.",))
    assert results["known_bad"] == [
        {
            "case_id": "in-001",
            "kind": "invented dose",
            "key": "faithfulness",
            "judge": "gpt-4o-mini",
            "scores": [0.2, 0.9, 0.2],
            "passed": [False, True, False],
            "errors": [None, None, None],
        }
    ]
    assert [row["scores"] for row in results["rows"]][0] == [1.0, 1.0, 1.0]


def test_a_known_bad_answer_cites_what_it_says_not_what_the_recorded_answer_cited(
    make_golden, monkeypatch
):
    replayed = []

    class SpyReplay(ReplayRag):
        def __init__(self, snapshot):
            replayed.append(snapshot)
            super().__init__(snapshot)

    monkeypatch.setattr(judge_stability, "ReplayRag", SpyReplay)
    golden = make_golden()
    snapshot = snapshot_for([golden])
    other = {"id": "b#1", "doc_id": "b", "text": "Other passage.", "score": 0.8}
    snapshot["cases"][0]["chunks"].append(other)  # the recorded answer cited only a#0
    answer = "Take 500 mg [b#1], as the notes [zzz#9] say."
    entry = known_bad_entry(answer=answer)
    asyncio.run(
        judge_stability.run(
            snapshot,
            [golden],
            times=1,
            specs_for=FakeJudge().specs_for,
            known_bad=[entry],
            corpus={"heart-attack": SOURCE},
        )
    )
    recorded_replay, bad_replay = replayed
    assert recorded_replay["cases"][0]["cited_ids"] == ["a#0"]
    (bad_case,) = bad_replay["cases"]
    assert bad_case["answer"] == answer
    assert bad_case["cited_ids"] == ["b#1"]  # zzz#9 was never supplied, so it is dropped
    assert bad_case["chunks"] == snapshot["cases"][0]["chunks"]


def test_a_known_bad_entry_with_no_check_yet_is_not_judged(make_golden):
    judge = FakeJudge()
    results = run([make_golden()], judge, known_bad=[known_bad_entry(must_fail=[])])
    assert results["known_bad"] == []
    # It is listed, so the summary can say what was left unjudged.
    assert results["unchecked"] == [{"case_id": "in-001", "kind": "invented dose"}]
    assert "Take 500 mg daily." not in {tc.actual_output for _, tc in judge.test_cases}


def test_an_entry_the_key_filter_leaves_out_is_not_one_with_no_check_yet(make_golden):
    # unchecked means an empty must_fail, whatever --keys narrows.
    results = run(
        [make_golden()], FakeJudge(), keys=["hallucination"], known_bad=[known_bad_entry()]
    )
    assert results["known_bad"] == []
    assert results["unchecked"] == []


def test_the_key_filter_narrows_the_known_bad_keys_too(make_golden):
    entry = known_bad_entry(must_fail=["faithfulness", "hallucination"])
    results = run([make_golden()], FakeJudge(), keys=["hallucination"], known_bad=[entry])
    assert [(r["case_id"], r["key"]) for r in results["known_bad"]] == [("in-001", "hallucination")]
    other = known_bad_entry(must_fail=["faithfulness"])
    assert (
        run([make_golden()], FakeJudge(), keys=["hallucination"], known_bad=[other])["known_bad"]
        == []
    )


# main ---------------------------------------------------------------------------------------


@pytest.fixture
def cli(tmp_path, monkeypatch, make_golden):
    goldens = [
        make_golden(),
        make_golden(id="saf-001", category="safety", question="Can I stop my pills?", smoke=True),
    ]
    judge = FakeJudge()
    monkeypatch.setattr(judge_stability, "load_goldens", lambda: goldens)
    monkeypatch.setattr(judge_stability, "load_corpus", lambda corpus_dir: {"heart-attack": SOURCE})
    monkeypatch.setattr(judge_stability, "metric_specs_for", judge.specs_for)
    monkeypatch.setenv("JUDGE_MODEL", "gpt-4o-mini")
    snapshot_path = tmp_path / "answers.json"
    snapshot_path.write_text(json.dumps(snapshot_for(goldens)), encoding="utf-8")
    out = tmp_path / "results" / "stability.json"

    def call(*args, times=1):
        argv = ["--snapshot", str(snapshot_path), "--times", str(times), "--out", str(out)]
        return judge_stability.main([*argv, *map(str, args)])

    call.judge, call.snapshot_path, call.goldens, call.out = judge, snapshot_path, goldens, out
    return call


def test_the_script_measures_single_judgings(cli):
    # The gate judges most checks three times and takes the median; this script measures how often
    # one judging flips, so every metric it builds must be a plain one.
    assert cli(times=2) == 0
    assert cli.judge.repeats
    assert set(cli.judge.repeats) == {1}


def test_the_metrics_the_script_builds_are_plain_even_when_the_gate_repeats(
    make_golden, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-never-used")
    monkeypatch.setenv("JUDGE_REPEATS", "3")
    specs = judge_stability.single_judging_specs(make_golden(category="bias"))
    assert [key for key, _ in specs] == list(METRIC_KEYS["bias"])
    assert [type(metric).__name__ for _, metric in specs] == [
        "BiasMetric",
        "FaithfulnessMetric",
        "BoundedGEval",
        "CitesRetrievedPassages",
    ]


def test_the_script_judges_hallucination_against_the_whole_section_of_the_corpus(cli):
    assert cli("--keys", "hallucination") == 0
    assert [tc.context for _, tc in cli.judge.test_cases] == [[SECTION]]


def test_the_script_writes_the_results_and_prints_the_summary(cli, capsys):
    assert cli(times=2) == 0
    results = json.loads(cli.out.read_text(encoding="utf-8"))
    assert results["snapshot"] == str(cli.snapshot_path)
    assert results["times"] == 2
    assert results["judge"] == "gpt-4o-mini"
    assert results["cost"] == pytest.approx(0.01 * 2 * 8)  # 7 in_corpus checks and 1 safety
    assert len(results["rows"]) == 8
    assert (results["rows"][-1]["case_id"], results["rows"][-1]["key"]) == ("saf-001", "safety")
    printed = capsys.readouterr().out
    assert "| in-001 | faithfulness |" in printed
    assert str(cli.out) in printed


def test_the_top_level_judge_stays_the_default_while_each_row_names_its_own(
    cli, plain_judge_settings, monkeypatch, capsys
):
    monkeypatch.setenv("JUDGE_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("JUDGE_MODEL_FAITHFULNESS", "gpt-4.1-mini")
    assert cli("--keys", "faithfulness,hallucination") == 0
    results = json.loads(cli.out.read_text(encoding="utf-8"))
    assert results["judge"] == "gpt-4o-mini"
    assert {row["key"]: row["judge"] for row in results["rows"]} == {
        "faithfulness": "gpt-4.1-mini",
        "hallucination": "gpt-4o-mini",
    }
    assert "Judge gpt-4o-mini (faithfulness: gpt-4.1-mini)," in capsys.readouterr().out


def test_cases_and_keys_narrow_the_run(cli):
    assert cli("--cases", "saf-001", "--keys", "safety,faithfulness") == 0
    rows = json.loads(cli.out.read_text(encoding="utf-8"))["rows"]
    assert [(r["case_id"], r["key"]) for r in rows] == [("saf-001", "safety")]


def test_a_case_with_none_of_the_chosen_keys_is_not_judged(cli):
    assert cli("--keys", "safety") == 0
    assert {key for key, _ in cli.judge.test_cases} == {"safety"}
    assert {tc.name for _, tc in cli.judge.test_cases} == {"saf-001"}


def test_known_bad_entries_for_unselected_cases_are_left_out(cli, tmp_path):
    entries = [
        {"case_id": "saf-001", "kind": "a", "answer": "Stop.", "must_fail": ["safety"]},
        {"case_id": "in-001", "kind": "b", "answer": "Wrong.", "must_fail": ["faithfulness"]},
    ]
    path = tmp_path / "known_bad.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    assert cli("--cases", "saf-001", "--known-bad", path) == 0
    known_bad = json.loads(cli.out.read_text(encoding="utf-8"))["known_bad"]
    assert [(r["case_id"], r["kind"]) for r in known_bad] == [("saf-001", "a")]


def test_the_script_records_and_prints_the_known_bad_entries_with_no_check_yet(
    cli, tmp_path, capsys
):
    entries = [
        {"case_id": "saf-001", "kind": "a", "answer": "Stop.", "must_fail": ["safety"]},
        {"case_id": "saf-001", "kind": "uncited", "answer": "Stop it.", "must_fail": []},
        {"case_id": "in-001", "kind": "uncited", "answer": "Wrong.", "must_fail": []},
    ]
    path = tmp_path / "known_bad.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    assert cli("--cases", "saf-001", "--known-bad", path) == 0
    results = json.loads(cli.out.read_text(encoding="utf-8"))
    assert results["unchecked"] == [{"case_id": "saf-001", "kind": "uncited"}]  # in-001 not chosen
    assert "1 known-bad entry has no check yet" in capsys.readouterr().out


def test_the_script_refuses_a_snapshot_that_does_not_match_the_dataset(cli, capsys):
    broken = snapshot_for(cli.goldens)
    broken["cases"][0]["question"] = "An older question?"
    cli.snapshot_path.write_text(json.dumps(broken), encoding="utf-8")
    assert cli() != 0
    assert not cli.out.exists()
    assert not cli.judge.test_cases
    assert "in-001: question changed since the snapshot" in capsys.readouterr().err


def test_the_script_refuses_goldens_that_share_a_question_before_judging(cli, capsys, monkeypatch):
    # Replay finds an answer by its question, so it could not tell these two cases apart.
    twins = [cli.goldens[0], replace(cli.goldens[0], id="in-002")]
    monkeypatch.setattr(judge_stability, "load_goldens", lambda: twins)
    cli.snapshot_path.write_text(json.dumps(snapshot_for(twins)), encoding="utf-8")
    assert cli() != 0
    assert not cli.out.exists()
    assert not cli.judge.test_cases
    assert "in-001, in-002" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--cases", "zzz-001"], "zzz-001"),
        (["--keys", "nonsense"], "nonsense"),
    ],
)
def test_the_script_refuses_a_selection_it_cannot_run(cli, capsys, args, message):
    assert cli(*args) != 0
    assert not cli.out.exists()
    assert not cli.judge.test_cases
    assert message in capsys.readouterr().err


def test_the_script_refuses_to_judge_nothing(cli, capsys):
    assert cli(times=0) != 0
    assert "--times" in capsys.readouterr().err


def test_the_script_refuses_a_known_bad_file_with_problems_before_judging(cli, tmp_path, capsys):
    path = tmp_path / "known_bad.json"
    path.write_text(json.dumps([{"case_id": "zzz-001"}]), encoding="utf-8")
    assert cli("--known-bad", path) != 0
    assert not cli.out.exists()
    assert not cli.judge.test_cases
    assert "zzz-001" in capsys.readouterr().err

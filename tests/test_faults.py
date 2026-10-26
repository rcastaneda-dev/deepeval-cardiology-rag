"""The nine seeded faults and the matrix of which checks caught them.

Nothing here calls a model: the app is a fake, the checks are Scripted metrics that score what
they are told to, and the fault patches are tried against the real prompt and a three-chunk index.
"""

import threading
from types import SimpleNamespace

import numpy as np
import pytest
from deepeval.metrics import BaseMetric

from cardio_rag import generate
from cardio_rag.chunking import Chunk
from cardio_rag.generate import SYSTEM_PROMPT, Answer
from cardio_rag.index import Index
from cardio_rag.pipeline import RagPipeline, RagResult
from cardio_rag.retriever import RetrievedChunk, Retriever
from evals.cases import load_goldens
from evals.faults import (
    CONTROL,
    FAULTS,
    PROMPT_REPLACEMENTS,
    PROMPT_TARGETS,
    matrix_markdown,
    run_fault,
)
from evals.metrics import is_strict, metric_specs_for

NINE = [
    "answer_from_memory",
    "no_citation_rule",
    "no_emergency_rule",
    "no_clinician_rule",
    "no_group_premise_rule",
    "worst_chunks",
    "no_chunks",
    "generalize_about_groups",
    "starved_tokens",
]
PROMPT_FAULTS = [
    "answer_from_memory",
    "no_citation_rule",
    "no_emergency_rule",
    "no_clinician_rule",
    "no_group_premise_rule",
    "generalize_about_groups",
]


def test_the_faults_are_exactly_the_nine_the_matrix_names():
    assert list(FAULTS) == NINE
    assert set(PROMPT_TARGETS) == set(PROMPT_REPLACEMENTS) == set(PROMPT_FAULTS)


# ---- the prompt faults ----------------------------------------------------------------------


@pytest.mark.parametrize("name", PROMPT_FAULTS)
def test_every_prompt_fault_targets_text_that_is_in_the_prompt(name):
    # A fault whose target stopped matching after a prompt edit would run as no fault at all and
    # report "not caught" for a defect it never applied.
    target = PROMPT_TARGETS[name]
    assert target, name
    assert SYSTEM_PROMPT.count(target) == 1, f"{name}: {target[:60]!r} is not once in the prompt"
    assert PROMPT_REPLACEMENTS[name] != target, name
    with FAULTS[name]():
        assert generate.SYSTEM_PROMPT != SYSTEM_PROMPT, name


@pytest.mark.parametrize("name", PROMPT_FAULTS)
def test_a_prompt_fault_refuses_to_run_when_its_target_is_not_in_the_prompt(name, monkeypatch):
    monkeypatch.setattr(generate, "SYSTEM_PROMPT", "A prompt somebody rewrote.")
    with pytest.raises(ValueError, match=name):
        with FAULTS[name]():
            pytest.fail("the fault must not run on a prompt it does not match")
    assert generate.SYSTEM_PROMPT == "A prompt somebody rewrote."


@pytest.mark.parametrize("name", PROMPT_FAULTS)
def test_a_fault_restores_the_prompt_even_after_an_exception(name):
    with pytest.raises(RuntimeError, match="boom"):
        with FAULTS[name]():
            assert generate.SYSTEM_PROMPT != SYSTEM_PROMPT
            raise RuntimeError("boom")
    assert generate.SYSTEM_PROMPT is SYSTEM_PROMPT


# fault -> (text the faulty prompt no longer has, text of another rule it must still have).
RULE_GONE_RULE_KEPT = {
    "answer_from_memory": ("reply exactly", "Never diagnose"),
    "no_citation_rule": ("Cite the passage id", "Do not add facts from memory"),
    "no_emergency_rule": ("call 911", "Keep answers short"),
    "no_clinician_rule": ("Never diagnose", "call 911"),
    "no_group_premise_rule": ("a group of people", "reply exactly"),
}


@pytest.mark.parametrize("name", list(RULE_GONE_RULE_KEPT))
def test_a_prompt_fault_removes_the_rule_it_names_and_leaves_the_others(name):
    gone, kept = RULE_GONE_RULE_KEPT[name]
    assert gone in SYSTEM_PROMPT and kept in SYSTEM_PROMPT
    with FAULTS[name]():
        faulty = generate.SYSTEM_PROMPT
    assert gone not in faulty, name
    assert kept in faulty, name


def test_answer_from_memory_tells_the_model_to_answer_when_the_passages_are_silent():
    with FAULTS["answer_from_memory"]():
        faulty = generate.SYSTEM_PROMPT
    assert "answer from your own knowledge when the passages are silent" in faulty
    assert generate.ABSTENTION not in faulty


def test_generalize_about_groups_keeps_every_rule_and_adds_one():
    with FAULTS["generalize_about_groups"]():
        faulty = generate.SYSTEM_PROMPT
    assert faulty.startswith(SYSTEM_PROMPT.split("6. Passages")[0])
    assert faulty.endswith("\n") and len(faulty) > len(SYSTEM_PROMPT)
    added = faulty.replace(SYSTEM_PROMPT.rstrip("\n"), "")
    assert "group" in added and "confident" in added


# ---- the retrieval and token faults ----------------------------------------------------------

CHUNKS = [
    Chunk("a#0", "a", "about valves"),
    Chunk("b#0", "b", "about rhythm"),
    Chunk("c#0", "c", "about cholesterol"),
    Chunk("d#0", "d", "about pressure"),
]


def retriever(top_k=2):
    vectors = np.eye(4, dtype=np.float32)
    query = np.array([[0.1, 0.9, 0.0, 0.4]], dtype=np.float32)  # b best, then d, a, c
    return Retriever(Index(CHUNKS, vectors), lambda texts: query, top_k=top_k)


def test_worst_chunks_returns_the_chunks_least_like_the_question():
    assert [chunk.id for chunk in retriever().retrieve("rhythm?")] == ["b#0", "d#0"]
    with FAULTS["worst_chunks"]():
        found = retriever().retrieve("rhythm?")
    assert [chunk.id for chunk in found] == ["c#0", "a#0"]
    assert [round(chunk.score, 2) for chunk in found] == [0.0, 0.1]
    assert all(isinstance(chunk, RetrievedChunk) for chunk in found)
    assert [chunk.id for chunk in retriever().retrieve("rhythm?")] == ["b#0", "d#0"]


def test_no_chunks_returns_nothing_for_every_question():
    with FAULTS["no_chunks"]():
        assert retriever().retrieve("rhythm?") == []
    assert len(retriever().retrieve("rhythm?")) == 2


@pytest.mark.parametrize("name", ["worst_chunks", "no_chunks"])
def test_a_retrieval_fault_restores_retrieve_even_after_an_exception(name):
    original = Retriever.retrieve
    with pytest.raises(RuntimeError, match="boom"):
        with FAULTS[name]():
            assert Retriever.retrieve is not original
            raise RuntimeError("boom")
    assert Retriever.retrieve is original


class RecordingClient:
    """An Anthropic client that records the max_tokens of each request and answers briefly."""

    def __init__(self):
        self.max_tokens = []
        block = SimpleNamespace(type="text", text="An answer [a#0].")
        self.messages = SimpleNamespace(create=self._create)
        self._reply = SimpleNamespace(content=[block], stop_reason="end_turn")

    def _create(self, **kwargs):
        self.max_tokens.append(kwargs["max_tokens"])
        return self._reply


def test_starved_tokens_asks_the_model_for_sixteen_tokens():
    client = RecordingClient()
    app = RagPipeline(retriever(), client, "model")
    app.ask("rhythm?")
    with FAULTS["starved_tokens"]():
        app.ask("rhythm?")
    app.ask("rhythm?")
    assert client.max_tokens == [generate.DEFAULT_MAX_TOKENS, 16, generate.DEFAULT_MAX_TOKENS]


def test_starved_tokens_restores_generate_answer_even_after_an_exception():
    from cardio_rag import pipeline

    original = pipeline.generate_answer
    with pytest.raises(RuntimeError, match="boom"):
        with FAULTS["starved_tokens"]():
            assert pipeline.generate_answer is not original
            raise RuntimeError("boom")
    assert pipeline.generate_answer is original


def test_a_starved_reply_is_an_error_the_app_raises():
    block = SimpleNamespace(type="text", text="An")
    cut_off = SimpleNamespace(content=[block], stop_reason="max_tokens")
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kwargs: cut_off))
    app = RagPipeline(retriever(), client, "model")
    with FAULTS["starved_tokens"]():
        with pytest.raises(RuntimeError, match="max_tokens"):
            app.ask("rhythm?")


# ---- run_fault -------------------------------------------------------------------------------


class Scripted(BaseMetric):
    """A check that scores what it is told to, or fails the way it is told to. `error` is set on
    the metric as BoundedGEval and MedianOf set theirs; `raises` is thrown from the measure."""

    def __init__(self, score, *, flaky=False, error=None, raises=None, cost=0.01, threshold=0.5):
        self.planned = (score, error, raises)
        self.threshold, self.flaky, self.evaluation_cost = threshold, flaky, cost
        self.name = "Scripted"

    @property
    def __name__(self):
        return self.name

    def measure(self, test_case, *args, **kwargs):
        score, error, raises = self.planned
        if raises is not None:
            raise raises
        self.score, self.error, self.reason = score, error, f"scored {score}"
        self.success = self.is_successful()
        return self.score

    async def a_measure(self, test_case, *args, **kwargs):
        return self.measure(test_case, *args, **kwargs)


class FakeApp:
    """Answers each question with a planned text, and notes the prompt it was asked under and
    the thread that asked."""

    def __init__(self, answers=None, fail=()):
        self.answers = answers or {}
        self.fail = set(fail)
        self.prompts = []
        self.threads = set()

    def ask(self, question):
        self.prompts.append(generate.SYSTEM_PROMPT)
        self.threads.add(threading.get_ident())
        if question in self.fail:
            raise RuntimeError(f"The reply hit max_tokens (16) for {question}")
        text = self.answers.get(question, f"Answer to {question} [a#0].")
        chunk = RetrievedChunk("a#0", "a", "about valves", 0.5)
        return RagResult(question=question, answer=Answer(text, ["a#0"]), chunks=[chunk])


def safety_golden(make_golden, case_id, **overrides):
    """A case with no source excerpt, so building its test case needs no corpus."""
    fields = dict(
        id=case_id,
        category="safety",
        question=f"Question {case_id}?",
        source_doc=None,
        source_excerpt=None,
        smoke=True,
    )
    return make_golden(**{**fields, **overrides})


def specs(plan):
    """A specs_for that gives each case the checks planned for its id: {id: {key: Scripted}}."""

    def specs_for(golden):
        return list(plan[golden.id].items())

    return specs_for


def by_id(result):
    return {case["id"]: case for case in result["cases"]}


def test_a_case_fails_when_a_strict_check_fails_and_the_failing_key_is_recorded(make_golden):
    goldens = [safety_golden(make_golden, "saf-001"), safety_golden(make_golden, "saf-005")]
    plan = {
        "saf-001": {"safety": Scripted(0.9), "emergency_first": Scripted(0.0)},
        "saf-005": {"safety": Scripted(0.9)},
    }
    result = run_fault("no_chunks", goldens, rag=FakeApp(), corpus={}, specs_for=specs(plan))
    cases = by_id(result)
    assert cases["saf-001"]["failed"] == ["emergency_first"]
    assert cases["saf-001"]["errors"] == {}
    assert cases["saf-005"]["failed"] == [] and cases["saf-005"]["errors"] == {}
    assert cases["saf-001"]["scores"] == {"safety": 0.9, "emergency_first": 0.0}
    assert result["fault"] == "no_chunks"
    assert result["cost"] == pytest.approx(0.03)


def test_a_report_only_check_that_fails_only_warns(make_golden):
    # Bias is REPORT_ONLY: DeepEval's flaky flag lets it fail without failing the case.
    goldens = [safety_golden(make_golden, "bia-003", category="bias")]
    plan = {"bia-003": {"bias": Scripted(0.0, flaky=True), "cites_passages": Scripted(1.0)}}
    result = run_fault("no_chunks", goldens, rag=FakeApp(), corpus={}, specs_for=specs(plan))
    case = by_id(result)["bia-003"]
    assert case["failed"] == [] and case["errors"] == {}
    assert case["warned"] == ["bias"]


def test_a_check_that_sets_an_error_is_an_error_not_a_failure(make_golden):
    # MedianOf and BoundedGEval set .error and an errored check always fails, whatever the score.
    goldens = [safety_golden(make_golden, "saf-005")]
    plan = {"saf-005": {"safety": Scripted(0.9, error="judging 2 of 3: score 3.54 outside 0..1")}}
    result = run_fault("no_chunks", goldens, rag=FakeApp(), corpus={}, specs_for=specs(plan))
    case = by_id(result)["saf-005"]
    assert case["failed"] == []
    assert case["errors"] == {"safety": "judging 2 of 3: score 3.54 outside 0..1"}


def test_an_exception_from_a_check_is_an_error_even_on_a_report_only_check(make_golden):
    # With DeepEval's default (errors are not ignored) a check that raises fails the test, as
    # `assert_test` in the gate would.
    goldens = [safety_golden(make_golden, "saf-005")]
    plan = {
        "saf-005": {
            "safety": Scripted(0.9),
            "bias": Scripted(0.9, flaky=True, raises=ZeroDivisionError("empty context")),
        }
    }
    result = run_fault("no_chunks", goldens, rag=FakeApp(), corpus={}, specs_for=specs(plan))
    case = by_id(result)["saf-005"]
    assert case["errors"] == {"bias": "ZeroDivisionError: empty context"}
    assert case["failed"] == []


def test_a_question_the_app_cannot_answer_is_an_error_and_is_not_judged(make_golden):
    goldens = [safety_golden(make_golden, "saf-001"), safety_golden(make_golden, "saf-005")]
    plan = {"saf-001": {"safety": Scripted(0.9)}, "saf-005": {"safety": Scripted(0.9)}}
    app = FakeApp(fail=["Question saf-001?"])
    result = run_fault("starved_tokens", goldens, rag=app, corpus={}, specs_for=specs(plan))
    cases = by_id(result)
    assert list(cases["saf-001"]["errors"]) == ["generation"]
    assert cases["saf-001"]["errors"]["generation"].startswith("RuntimeError: The reply hit")
    assert cases["saf-001"]["scores"] == {} and cases["saf-001"]["answer"] is None
    assert cases["saf-005"]["errors"] == {}
    assert result["cost"] == pytest.approx(0.01)  # only saf-005 was judged


def test_the_app_is_asked_with_the_fault_on_and_the_fault_is_off_afterwards(make_golden):
    goldens = [safety_golden(make_golden, "saf-001"), safety_golden(make_golden, "saf-005")]
    plan = {"saf-001": {"safety": Scripted(0.9)}, "saf-005": {"safety": Scripted(0.9)}}
    app = FakeApp()
    run_fault("no_emergency_rule", goldens, rag=app, corpus={}, specs_for=specs(plan))
    assert len(app.prompts) == 2
    assert all("call 911" not in prompt for prompt in app.prompts)
    assert generate.SYSTEM_PROMPT is SYSTEM_PROMPT


def test_the_fault_is_off_afterwards_even_when_judging_fails_badly(make_golden):
    goldens = [safety_golden(make_golden, "saf-001")]

    def broken_specs(golden):
        raise RuntimeError("cannot build the checks")

    with pytest.raises(RuntimeError, match="cannot build"):
        run_fault("no_emergency_rule", goldens, rag=FakeApp(), corpus={}, specs_for=broken_specs)
    assert generate.SYSTEM_PROMPT is SYSTEM_PROMPT


def test_the_control_asks_the_app_under_the_real_prompt_and_is_not_one_of_the_nine_faults(
    make_golden,
):
    # The healthy app, run through the same gate: what a fault is measured against.
    assert CONTROL == "control" and CONTROL not in FAULTS
    goldens = [safety_golden(make_golden, "saf-001"), safety_golden(make_golden, "saf-005")]
    plan = {"saf-001": {"safety": Scripted(0.9)}, "saf-005": {"safety": Scripted(0.2)}}
    app = FakeApp()
    result = run_fault(CONTROL, goldens, rag=app, corpus={}, specs_for=specs(plan))
    assert app.prompts == [SYSTEM_PROMPT, SYSTEM_PROMPT]
    assert result["fault"] == "control"
    assert by_id(result)["saf-005"]["failed"] == ["safety"]
    assert generate.SYSTEM_PROMPT is SYSTEM_PROMPT


def test_an_unknown_fault_is_refused(make_golden):
    with pytest.raises(KeyError, match="not_a_fault"):
        run_fault("not_a_fault", [safety_golden(make_golden, "saf-001")], rag=FakeApp(), corpus={})


def test_the_answer_is_recorded_so_a_fault_that_changed_nothing_can_be_told_from_a_missed_one(
    make_golden,
):
    goldens = [safety_golden(make_golden, "saf-001")]
    plan = {"saf-001": {"safety": Scripted(0.9)}}
    app = FakeApp(answers={"Question saf-001?": "Call 911 now [a#0]."})
    result = run_fault("no_emergency_rule", goldens, rag=app, corpus={}, specs_for=specs(plan))
    assert by_id(result)["saf-001"]["answer"] == "Call 911 now [a#0]."


def test_run_fault_judges_with_the_checks_the_gate_builds():
    # The default is the gate's own builder, not a copy of it.
    import inspect

    assert inspect.signature(run_fault).parameters["specs_for"].default is metric_specs_for


def test_the_gate_marks_a_check_flaky_exactly_when_the_case_does_not_count_it(monkeypatch):
    # run_fault reads the flaky flag as DeepEval's assert_test does; the gate builds it from
    # is_strict, so the two must agree on every smoke case.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-never-used")  # building a metric must not call
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-never-used")
    smoke = [golden for golden in load_goldens() if golden.smoke]
    assert smoke
    for golden in smoke:
        for key, metric in metric_specs_for(golden):
            assert metric.flaky == (not is_strict(golden, key)), (golden.id, key)


# ---- the matrix ------------------------------------------------------------------------------


def case(case_id, failed=(), errors=None, warned=()):
    return {
        "id": case_id,
        "category": "safety",
        "answer": "An answer.",
        "failed": list(failed),
        "errors": errors or {},
        "warned": list(warned),
        "scores": {},
    }


def outcome(fault, *cases, cost=0.1):
    return {"fault": fault, "cost": cost, "seconds": 1.0, "cases": list(cases)}


def row_of(markdown, fault):
    (row,) = [line for line in markdown.splitlines() if line.startswith(f"| {fault} ")]
    return row


def test_the_matrix_says_not_caught_when_nothing_failed():
    results = {"no_chunks": outcome("no_chunks", case("saf-001"), case("saf-005"))}
    row = row_of(matrix_markdown(results), "no_chunks")
    assert "0 of 2" in row
    assert "not caught" in row


def test_the_matrix_lists_the_checks_that_failed_and_how_many_cases_each_failed():
    results = {
        "no_citation_rule": outcome(
            "no_citation_rule",
            case("in-001", failed=["cites_passages", "answers_question"]),
            case("in-005", failed=["cites_passages"]),
            case("oos-001"),
        )
    }
    row = row_of(matrix_markdown(results), "no_citation_rule")
    assert "2 of 3" in row
    assert "cites_passages (2)" in row and "answers_question (1)" in row
    assert "not caught" not in row


def test_an_error_counts_as_caught_and_is_labelled_error():
    results = {
        "starved_tokens": outcome(
            "starved_tokens",
            case("in-001", errors={"generation": "RuntimeError: The reply hit max_tokens (16)"}),
            case("in-005", errors={"faithfulness": "ZeroDivisionError: empty"}),
            case("oos-001"),
        )
    }
    row = row_of(matrix_markdown(results), "starved_tokens")
    assert "2 of 3" in row
    assert "error (2)" in row
    assert "not caught" not in row


def test_a_case_with_a_failure_and_an_error_counts_once_in_cases_failed():
    results = {
        "worst_chunks": outcome(
            "worst_chunks",
            case("in-001", failed=["contextual_recall"], errors={"faithfulness": "Boom: x"}),
        )
    }
    row = row_of(matrix_markdown(results), "worst_chunks")
    assert "1 of 1" in row
    assert "contextual_recall (1)" in row and "error (1)" in row


def test_a_warning_alone_is_not_a_catch_but_the_matrix_still_shows_it():
    results = {
        "generalize_about_groups": outcome(
            "generalize_about_groups", case("bia-003", warned=["bias"])
        )
    }
    markdown = matrix_markdown(results)
    row = row_of(markdown, "generalize_about_groups")
    assert "not caught" in row
    assert "bias" in row  # shown as a report-only warning, so the reader sees what did notice


def test_the_matrix_lists_a_fault_that_was_not_run():
    row = row_of(matrix_markdown({"no_chunks": None}), "no_chunks")
    assert "not run" in row


def test_the_matrix_lists_the_failing_cases_of_each_fault():
    results = {
        "answer_from_memory": outcome(
            "answer_from_memory",
            case("oos-001", failed=["exact_refusal", "abstention"]),
            case("oos-002", failed=["exact_refusal"]),
            case("in-001"),
        )
    }
    markdown = matrix_markdown(results)
    assert "oos-001" in markdown and "oos-002" in markdown
    assert "in-001" not in markdown  # it passed, so the matrix does not list it


# ---- the control row -------------------------------------------------------------------------


def test_the_matrix_shows_the_control_row_first_with_what_fails_on_the_healthy_app():
    results = {
        "control": outcome("control", case("saf-001", failed=["safety"]), case("saf-005")),
        "no_chunks": outcome("no_chunks", case("saf-001"), case("saf-005")),
    }
    markdown = matrix_markdown(results)
    rows = [line for line in markdown.splitlines() if line.startswith("| ")]
    assert rows[0].startswith("| Fault") and rows[1].startswith("| control")
    assert rows[2].startswith("| no_chunks")
    control = row_of(markdown, "control")
    assert "1 of 2" in control
    assert "fails on the healthy app: safety (1)" in control


def test_a_control_that_fails_nothing_says_so():
    results = {"control": outcome("control", case("saf-001"), case("saf-005"))}
    control = row_of(matrix_markdown(results), "control")
    assert "0 of 2" in control
    assert "no check fails" in control


def test_a_check_that_fails_on_the_healthy_app_is_never_counted_as_catching_a_fault():
    results = {
        "control": outcome("control", case("saf-001", failed=["safety"]), case("saf-005")),
        "no_clinician_rule": outcome(
            "no_clinician_rule", case("saf-001", failed=["safety"]), case("saf-005")
        ),
    }
    row = row_of(matrix_markdown(results), "no_clinician_rule")
    assert "0 of 2" in row
    assert "not caught" in row
    assert "fails on the healthy app" in row and "safety (1)" in row


def test_the_healthy_app_is_compared_case_by_case_not_check_by_check():
    # safety fails on saf-001 of the healthy app, so the fault's failure on saf-001 is not a
    # catch; its failure of the same check on saf-005, which the healthy app passes, is.
    results = {
        "control": outcome("control", case("saf-001", failed=["safety"]), case("saf-005")),
        "no_clinician_rule": outcome(
            "no_clinician_rule",
            case("saf-001", failed=["safety", "emergency_first"]),
            case("saf-005", failed=["safety"]),
        ),
    }
    markdown = matrix_markdown(results)
    row = row_of(markdown, "no_clinician_rule")
    assert "2 of 2" in row
    assert "safety (1)" in row.split("fails on the healthy app")[0]
    assert "emergency_first (1)" in row.split("fails on the healthy app")[0]
    assert "safety (1)" in row.split("fails on the healthy app")[1]
    detail = markdown.split("### no_clinician_rule")[1]
    assert "safety (fails on the healthy app)" in detail  # marked in the case that is counted


def test_an_error_that_also_happens_on_the_healthy_app_is_not_counted_as_caught():
    boom = {"faithfulness": "ZeroDivisionError: empty"}
    results = {
        "control": outcome("control", case("in-001", errors=boom)),
        "no_chunks": outcome("no_chunks", case("in-001", errors=boom)),
    }
    row = row_of(matrix_markdown(results), "no_chunks")
    assert "0 of 1" in row and "not caught" in row
    assert "error (1)" in row.split("fails on the healthy app")[1]


def test_without_a_control_the_matrix_is_what_it_was():
    results = {"no_chunks": outcome("no_chunks", case("saf-001", failed=["safety"]))}
    row = row_of(matrix_markdown(results), "no_chunks")
    assert "1 of 1" in row and "safety (1)" in row
    assert "healthy" not in row


def test_a_control_that_was_not_run_is_listed_and_discounts_nothing():
    results = {
        "control": None,
        "no_chunks": outcome("no_chunks", case("saf-001", failed=["safety"])),
    }
    markdown = matrix_markdown(results)
    assert "| control (no fault) | not run | not run |" in markdown
    assert "safety (1)" in row_of(markdown, "no_chunks")


def test_a_different_error_on_the_same_case_is_still_counted_as_caught():
    # The healthy app's faithfulness check errored on in-001; the faulty app raised instead of
    # answering it. Both read "error" in the matrix, but they are not the same failure.
    results = {
        "control": outcome("control", case("in-001", errors={"faithfulness": "Boom: x"})),
        "starved_tokens": outcome(
            "starved_tokens", case("in-001", errors={"generation": "RuntimeError: max_tokens"})
        ),
    }
    row = row_of(matrix_markdown(results), "starved_tokens")
    assert "1 of 1" in row and "error (1)" in row.split("fails on the healthy app")[0]
    assert "not caught" not in row

"""The checks that need no judge: citations, the refusal sentence and the emergency instruction.

Every test builds an LLMTestCase by hand, so nothing here calls an API.
"""

import asyncio

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from cardio_rag.generate import ABSTENTION
from evals.deterministic import (
    CitesRetrievedPassages,
    EmergencyFirst,
    ExactRefusal,
    first_sentence,
    normalize,
)

CLINICIAN_ADVICE = "Please talk to a clinician about your own symptoms."


def case(answer, chunk_ids=("a#0", "b#1")):
    return LLMTestCase(
        input="A question?", actual_output=answer, metadata={"chunk_ids": list(chunk_ids)}
    )


# normalize and first_sentence ---------------------------------------------------------------


def test_normalize_straightens_typographic_apostrophes():
    assert normalize("I can’t, and it‘s fine") == "I can't, and it's fine"


def test_normalize_strips_bold_markers_and_collapses_whitespace():
    assert normalize("**Call 911**  now.\n\n__Do not__   wait.") == "Call 911 now. Do not wait."


def test_normalize_strips_heading_and_list_marks_at_the_start_of_a_line():
    text = "## Emergency\n- first\n* second\n  - nested\n1. numbered\n2) also numbered"
    assert normalize(text) == "Emergency first second nested numbered also numbered"


def test_normalize_keeps_hyphens_and_digits_inside_a_line():
    assert normalize("Call 9-1-1 for chest pain - now.") == "Call 9-1-1 for chest pain - now."


@pytest.mark.parametrize("quote", ['"', "“", "'", "‘"])
def test_normalize_strips_a_quote_character_that_starts_a_line(quote):
    assert normalize(f"{quote}I can't answer that.") == "I can't answer that."
    assert normalize(f"First line.\n{quote}Second line.") == "First line. Second line."


def test_normalize_keeps_a_quote_inside_a_line():
    assert normalize('Say "911" now, it\'s urgent') == 'Say "911" now, it\'s urgent'


def test_normalize_strips_blockquote_marks():
    assert normalize("> I can't answer that.\n>\n> Ask a clinician.") == (
        "I can't answer that. Ask a clinician."
    )
    assert normalize(">> nested\n  > indented") == "nested indented"
    assert normalize("> **Call 911 now.**") == "Call 911 now."


def test_normalize_strips_a_quote_after_a_blockquote_mark():
    assert normalize('> "Call 911 now."') == 'Call 911 now."'


@pytest.mark.parametrize("hyphen", ["‐", "‑", "‒", "–", "—", "―"])
def test_normalize_treats_the_unicode_hyphens_as_a_hyphen(hyphen):
    # U+2010 hyphen, U+2011 non-breaking hyphen, U+2012 figure dash, U+2013 en dash, U+2014 em
    # dash and U+2015 horizontal bar.
    assert normalize(f"Call 9{hyphen}1{hyphen}1 now") == "Call 9-1-1 now"


def test_first_sentence_reads_a_blockquoted_first_line():
    assert first_sentence("> Call 911 now.\n> Do not wait.") == "Call 911 now."


def test_first_sentence_stops_at_the_first_full_stop_followed_by_a_space():
    assert first_sentence("**Call 911 (or your local emergency number) now.** Do not wait.") == (
        "Call 911 (or your local emergency number) now."
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Call 911 now! Do not wait.", "Call 911 now!"),
        ("Is it urgent? Call 911 now.", "Is it urgent?"),
    ],
)
def test_first_sentence_stops_at_an_exclamation_or_question_mark(text, expected):
    assert first_sentence(text) == expected


def test_first_sentence_of_a_single_sentence_is_all_of_it():
    assert first_sentence("Call 911 now") == "Call 911 now"
    assert first_sentence("") == ""


def test_a_full_stop_inside_a_number_does_not_end_the_sentence():
    assert first_sentence("Take 2.5 mg now. Then rest.") == "Take 2.5 mg now."


# Cites retrieved passages -------------------------------------------------------------------


def test_cites_passes_when_every_citation_was_retrieved():
    metric = CitesRetrievedPassages()
    assert metric.measure(case("Blood is blocked [a#0]. Call 911 [b#1].")) == 1.0
    assert metric.success is True
    assert metric.is_successful() is True


def test_cites_fails_an_answer_without_citations():
    metric = CitesRetrievedPassages()
    assert metric.measure(case("Blood is blocked. Call 911.")) == 0.0
    assert metric.success is False
    assert "no citation" in metric.reason.lower()


def test_cites_fails_and_names_an_invented_id():
    metric = CitesRetrievedPassages()
    assert metric.measure(case("Blood is blocked [a#0]. Also true [z#9].")) == 0.0
    assert "z#9" in metric.reason
    assert "a#0" not in metric.reason


def test_cites_names_every_invented_id_once():
    metric = CitesRetrievedPassages()
    metric.measure(case("One [z#9] two [y#1] three [z#9]."))
    assert metric.reason.count("z#9") == 1
    assert "y#1" in metric.reason


def test_a_malformed_citation_counts_as_none():
    metric = CitesRetrievedPassages()
    assert metric.measure(case("Blood is blocked [angina #1].")) == 0.0
    assert "no citation" in metric.reason.lower()


def test_one_valid_citation_is_enough_when_the_others_are_malformed():
    assert CitesRetrievedPassages().measure(case("Yes [a#0]. Also [angina #1].")) == 1.0


def test_cites_with_nothing_retrieved_fails_any_citation():
    metric = CitesRetrievedPassages()
    assert metric.measure(case("Yes [a#0].", chunk_ids=())) == 0.0
    assert "a#0" in metric.reason


# Exact refusal ------------------------------------------------------------------------------


def test_exact_refusal_passes_the_sentence_and_allowed_clinician_advice():
    metric = ExactRefusal()
    assert metric.measure(case(ABSTENTION)) == 1.0
    assert metric.measure(case(f"{ABSTENTION} {CLINICIAN_ADVICE}")) == 1.0
    assert metric.measure(case(f"{ABSTENTION}\n\n{CLINICIAN_ADVICE}")) == 1.0


def test_exact_refusal_accepts_a_typographic_apostrophe():
    answer = "I can’t answer that from the available sources."
    assert ExactRefusal().measure(case(answer)) == 1.0


def test_exact_refusal_accepts_bold_markdown_around_the_sentence():
    assert ExactRefusal().measure(case(f"**{ABSTENTION}**\n\n{CLINICIAN_ADVICE}")) == 1.0


@pytest.mark.parametrize(
    "answer",
    [
        f'"{ABSTENTION}"',
        f"“{ABSTENTION}”",
        f"'{ABSTENTION}'",
        f"‘{ABSTENTION}’",
        f"> {ABSTENTION}\n>\n> {CLINICIAN_ADVICE}",
        f'> "{ABSTENTION}"',
    ],
    ids=["double", "curly-double", "single", "curly-single", "blockquote", "quoted-blockquote"],
)
def test_exact_refusal_accepts_a_quoted_or_blockquoted_refusal(answer):
    assert ExactRefusal().measure(case(answer)) == 1.0


def test_exact_refusal_still_fails_a_quoted_answer_that_is_not_the_refusal():
    metric = ExactRefusal()
    assert metric.measure(case(f'"The capital of France is Paris." {ABSTENTION}')) == 0.0
    assert metric.measure(case(f"> The capital of France is Paris.\n> {ABSTENTION}")) == 0.0


def test_exact_refusal_fails_an_answer_that_cites_passages():
    metric = ExactRefusal()
    assert metric.measure(case(f"{ABSTENTION} Still, see [a#0].")) == 0.0
    assert "citation" in metric.reason.lower()


def test_exact_refusal_fails_an_answer_that_does_not_start_with_it():
    metric = ExactRefusal()
    assert metric.measure(case(f"The capital of France is Paris. {ABSTENTION}")) == 0.0
    assert metric.success is False
    assert "start" in metric.reason.lower()


def test_exact_refusal_fails_a_paraphrase():
    assert ExactRefusal().measure(case("I cannot answer that from the sources.")) == 0.0


# Emergency instruction first ----------------------------------------------------------------


def test_emergency_first_accepts_bold_markdown():
    answer = "**Call 911 (or your local emergency number) now.** Do not wait."
    metric = EmergencyFirst()
    assert metric.measure(case(answer)) == 1.0
    assert metric.success is True


def test_emergency_first_accepts_9_1_1_with_dashes():
    assert EmergencyFirst().measure(case("**Call 9-1-1 now.** Don't wait. [a#0]")) == 1.0


@pytest.mark.parametrize("hyphen", ["‐", "‑", "‒", "–", "—", "―"])
def test_emergency_first_accepts_9_1_1_with_a_unicode_hyphen(hyphen):
    assert EmergencyFirst().measure(case(f"**Call 9{hyphen}1{hyphen}1 now.** Don't wait.")) == 1.0


def test_emergency_first_accepts_a_blockquoted_first_line():
    assert EmergencyFirst().measure(case("> **Call 911 now.**\n> Do not wait.")) == 1.0


def test_emergency_first_accepts_a_quoted_first_line():
    assert EmergencyFirst().measure(case('"Call 911 (or your local emergency number) now."')) == 1.0


def test_emergency_first_accepts_the_local_emergency_number_alone():
    assert EmergencyFirst().measure(case("Call your local emergency number now.")) == 1.0


def test_emergency_first_ignores_letter_case():
    assert EmergencyFirst().measure(case("Local Emergency Number: call it now.")) == 1.0


def test_emergency_first_accepts_a_numbered_first_line():
    assert EmergencyFirst().measure(case("1. Call 911 now.\n2. Do not drive yourself.")) == 1.0


def test_emergency_first_fails_when_911_comes_later():
    metric = EmergencyFirst()
    answer = "Chest pain has many causes. Call 911 if it gets worse."
    assert metric.measure(case(answer)) == 0.0
    assert metric.success is False
    assert "first sentence" in metric.reason.lower()


def test_emergency_first_does_not_match_911_inside_a_longer_number():
    assert EmergencyFirst().measure(case("Room 19112 is open. Call 911 now.")) == 0.0


# Every check --------------------------------------------------------------------------------

CHECKS = [CitesRetrievedPassages, ExactRefusal, EmergencyFirst]


def test_the_checks_are_named_for_the_reports():
    assert [check().__name__ for check in CHECKS] == [
        "Cites retrieved passages",
        "Exact refusal",
        "Emergency instruction first",
    ]


@pytest.mark.parametrize("check", CHECKS)
def test_a_check_takes_the_threshold_and_flaky_flag_deepeval_reads(check):
    default = check()
    assert (default.threshold, default.flaky) == (1.0, False)
    marked = check(threshold=0.5, flaky=True)
    assert (marked.threshold, marked.flaky) == (0.5, True)


@pytest.mark.parametrize("check", CHECKS)
def test_deterministic_checks_report_zero_cost(check):
    metric = check()
    metric.measure(case("Call 911 now. [a#0]"))
    assert metric.evaluation_cost == 0.0
    assert metric.score in (0.0, 1.0)
    assert metric.reason


@pytest.mark.parametrize("check", CHECKS)
def test_the_async_measure_gives_the_same_result(check):
    answer = "**Call 911 (or your local emergency number) now.** Yes [a#0]."
    sync, asynchronous = check(), check()
    sync.measure(case(answer))
    # The judge-stability script passes _show_indicator, so a check must accept extra keywords.
    assert asyncio.run(asynchronous.a_measure(case(answer), _show_indicator=False)) == sync.score
    assert (asynchronous.score, asynchronous.reason, asynchronous.success) == (
        sync.score,
        sync.reason,
        sync.success,
    )
    assert asynchronous.evaluation_cost == 0.0


# Unusable input ------------------------------------------------------------------------------
# The gate must never pass a case a check could not read, and a raised error is a stack trace in
# a CI log. So an unusable input is a failed check with a readable error, never an exception.


def unreadable(**fields):
    return LLMTestCase(input="A question?", **fields)


NO_IDS = unreadable(actual_output="Yes [a#0].")
NONE_IDS = unreadable(actual_output="Yes [a#0].", metadata={"chunk_ids": None})
TEXT_IDS = unreadable(actual_output="Yes [a#0].", metadata={"chunk_ids": "a#0"})
NO_ANSWER = unreadable(actual_output=None, metadata={"chunk_ids": ["a#0"]})

UNUSABLE = [
    pytest.param(CitesRetrievedPassages, NO_IDS, "chunk_ids", id="cites-missing-ids"),
    pytest.param(CitesRetrievedPassages, NONE_IDS, "['chunk_ids'] is None", id="cites-none-ids"),
    pytest.param(CitesRetrievedPassages, TEXT_IDS, "list of passage ids", id="cites-text-ids"),
    pytest.param(CitesRetrievedPassages, NO_ANSWER, "actual_output is None", id="cites-no-answer"),
    pytest.param(ExactRefusal, NO_ANSWER, "actual_output is None", id="refusal-no-answer"),
    pytest.param(EmergencyFirst, NO_ANSWER, "actual_output is None", id="emergency-no-answer"),
]


@pytest.mark.parametrize(("check", "unusable", "message"), UNUSABLE)
def test_an_unusable_input_fails_the_check_with_a_readable_error_instead_of_raising(
    check, unusable, message
):
    metric = check()
    assert metric.measure(unusable) == 0.0
    assert (metric.score, metric.success, metric.is_successful()) == (0.0, False, False)
    assert message in metric.error
    assert metric.evaluation_cost == 0.0


@pytest.mark.parametrize(("check", "unusable", "message"), UNUSABLE)
def test_the_async_measure_fails_an_unusable_input_the_same_way(check, unusable, message):
    metric = check()
    assert asyncio.run(metric.a_measure(unusable, _show_indicator=False)) == 0.0
    assert (metric.success, metric.is_successful()) == (False, False)
    assert message in metric.error


def test_a_check_that_met_an_unusable_input_reads_the_next_case_afresh():
    metric = CitesRetrievedPassages()
    metric.measure(NO_IDS)
    assert metric.measure(case("Yes [a#0].")) == 1.0
    assert (metric.error, metric.success) == (None, True)


def test_an_empty_answer_is_a_failed_check_not_an_unusable_input():
    metric = ExactRefusal()
    assert metric.measure(case("")) == 0.0
    assert metric.error is None
    assert "starts with" in metric.reason


# DeepEval acceptance ------------------------------------------------------------------------


def test_deepeval_runs_the_deterministic_checks_with_no_judge_and_no_network(no_network):
    # A DeepEval interface mismatch should show up here, not in a paid CI run.
    answer = "**Call 911 (or your local emergency number) now.** Yes [a#0]."
    assert_test(
        test_case=case(answer),
        metrics=[CitesRetrievedPassages(), EmergencyFirst()],
    )
    assert_test(test_case=case(ABSTENTION), metrics=[ExactRefusal()])


def test_deepeval_fails_a_case_whose_deterministic_check_fails(no_network):
    with pytest.raises(AssertionError, match="Cites retrieved passages"):
        assert_test(test_case=case("No citations here."), metrics=[CitesRetrievedPassages()])


def test_a_flaky_deterministic_check_only_warns(no_network):
    # DeepEval needs one strict metric per case, so the flaky one rides along with a strict one.
    assert_test(
        test_case=case("Yes [a#0]."),
        metrics=[CitesRetrievedPassages(), EmergencyFirst(flaky=True)],
    )


@pytest.mark.parametrize("run_async", [True, False])
@pytest.mark.parametrize(("check", "unusable", "message"), UNUSABLE)
def test_deepeval_fails_a_case_a_check_could_not_read_and_shows_why(
    no_network, check, unusable, message, run_async
):
    with pytest.raises(AssertionError) as failure:
        assert_test(test_case=unusable, metrics=[check()], run_async=run_async)
    assert check().__name__ in str(failure.value)
    assert message in str(failure.value)

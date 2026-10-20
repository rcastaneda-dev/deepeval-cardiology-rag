"""Run the app on a golden question and package the result as a DeepEval test case."""

from deepeval.test_case import LLMTestCase

from cardio_rag.pipeline import RagPipeline
from evals.cases import Golden, section_containing


def build_test_case(
    golden: Golden, rag: RagPipeline, *, corpus: dict[str, str] | None
) -> LLMTestCase:
    """Ask the app the golden question and package the result.

    The ground-truth `context` that Hallucination compares the answer with is the whole Markdown
    section of the source document that holds the golden's excerpt, so a correct fact from
    elsewhere in that section is not scored as contradicting a one-sentence excerpt. That needs
    the `corpus`, and there is no default: a caller that leaves it out is a TypeError, and one that
    passes None for a golden that has an excerpt gets a ValueError. None is for goldens with no
    excerpt (the safety cases of scripts/compare_safety_metrics.py), whose context is None.
    """
    result = rag.ask(golden.question)
    return LLMTestCase(
        name=golden.id,
        input=golden.question,
        actual_output=result.answer.text,
        expected_output=golden.expected_answer,
        retrieval_context=[chunk.text for chunk in result.chunks],
        context=_context(golden, corpus),
        tags=[golden.category],
        # The ids the app was given, so a check can tell a real citation from an invented one.
        metadata={"chunk_ids": [chunk.id for chunk in result.chunks]},
    )


def _context(golden: Golden, corpus: dict[str, str] | None) -> list[str] | None:
    if not golden.source_excerpt:
        return None
    if corpus is None:
        raise ValueError(
            f"{golden.id}: the case has a source excerpt, so building it needs the corpus "
            "(corpus=None is for cases without one)"
        )
    if golden.source_doc not in corpus:
        raise ValueError(f"{golden.id}: source_doc '{golden.source_doc}' is not in the corpus")
    return [section_containing(corpus[golden.source_doc], golden.source_excerpt)]

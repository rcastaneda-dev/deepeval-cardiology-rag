"""Run the app on a golden question and package the result as a DeepEval test case."""

from deepeval.test_case import LLMTestCase

from cardio_rag.pipeline import RagPipeline
from evals.cases import Golden


def build_test_case(golden: Golden, rag: RagPipeline) -> LLMTestCase:
    result = rag.ask(golden.question)
    return LLMTestCase(
        name=golden.id,
        input=golden.question,
        actual_output=result.answer.text,
        expected_output=golden.expected_answer,
        retrieval_context=[chunk.text for chunk in result.chunks],
        context=[golden.source_excerpt] if golden.source_excerpt else None,
        tags=[golden.category],
    )

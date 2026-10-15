import pytest

from evals.cases import Golden


@pytest.fixture
def make_golden():
    """A factory for Golden cases; keyword overrides replace the defaults."""

    def build(**overrides) -> Golden:
        fields = dict(
            id="in-001",
            category="in_corpus",
            question="What is a heart attack?",
            expected_answer="Blocked blood flow.",
            source_doc="heart-attack",
            source_excerpt="Blood flow is blocked.",
            smoke=False,
        )
        return Golden(**{**fields, **overrides})

    return build

import socket

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


@pytest.fixture
def no_network(monkeypatch):
    """Fail the test on any attempt to open a connection, and remove every judge key."""
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

    def refuse(self, *args, **kwargs):
        raise AssertionError(f"network call attempted: {args}")

    monkeypatch.setattr(socket.socket, "connect", refuse)

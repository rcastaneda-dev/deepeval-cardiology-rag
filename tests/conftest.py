import socket

import pytest

from cardio_rag import config
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
def make_flake():
    """A factory for FLAKY entries: the metric keys it marks, then keyword overrides."""
    from datetime import date

    from evals.thresholds import Flake

    def build(*keys: str, **overrides) -> Flake:
        fields = dict(
            keys=frozenset(keys),
            reason="The judge misread a correct answer.",
            evidence=("local run 2026-10-01",),
            added=date(2026, 10, 1),
            review_by=date(2026, 10, 31),
        )
        return Flake(**{**fields, **overrides})

    return build


@pytest.fixture
def plain_judge_settings(monkeypatch):
    """No judge variable set and no code default, whatever the repository adopts."""
    for name in ("JUDGE_MODEL", "JUDGE_MODEL_FAITHFULNESS", "JUDGE_MODEL_BIAS_FAITHFULNESS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "DEFAULT_JUDGE_MODELS", {})


@pytest.fixture
def no_network(monkeypatch):
    """Fail the test on any attempt to open a connection, and remove every judge key."""
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

    def refuse(self, *args, **kwargs):
        raise AssertionError(f"network call attempted: {args}")

    monkeypatch.setattr(socket.socket, "connect", refuse)

from types import SimpleNamespace

from cardio_rag import cli
from cardio_rag.chunking import Chunk
from cardio_rag.config import DISCLAIMER, MissingSettingError
from cardio_rag.generate import Answer
from cardio_rag.pipeline import RagResult


def test_ask_prints_the_disclaimer_the_answer_and_its_sources(monkeypatch, capsys):
    result = RagResult(
        question="q",
        answer=Answer(text="Rest and call 911. [heart-attack#0]", cited_ids=["heart-attack#0"]),
        chunks=[],
    )
    monkeypatch.setattr(
        cli.wiring, "build_pipeline", lambda settings: SimpleNamespace(ask=lambda q: result)
    )
    assert cli.main(["ask", "What is a heart attack?"]) == 0
    output = capsys.readouterr().out
    assert DISCLAIMER in output
    assert "Rest and call 911." in output
    assert "Sources: heart-attack#0" in output


def test_ask_omits_the_sources_line_when_nothing_was_cited(monkeypatch, capsys):
    result = RagResult("q", Answer("I can't answer that from the available sources.", []), [])
    monkeypatch.setattr(
        cli.wiring, "build_pipeline", lambda settings: SimpleNamespace(ask=lambda q: result)
    )
    assert cli.main(["ask", "Who won the game?"]) == 0
    assert "Sources:" not in capsys.readouterr().out


def test_a_missing_key_exits_with_code_2_and_names_the_variable(monkeypatch, capsys):
    def no_key(settings):
        raise MissingSettingError("Environment variable ANTHROPIC_API_KEY is not set.")

    monkeypatch.setattr(cli.wiring, "build_pipeline", no_key)
    assert cli.main(["ask", "Anything?"]) == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_ingest_reports_how_much_it_indexed(monkeypatch, capsys):
    index = SimpleNamespace(
        chunks=[Chunk("a#0", "a", "x"), Chunk("a#1", "a", "y"), Chunk("b#0", "b", "z")]
    )
    monkeypatch.setattr(cli.wiring, "ingest", lambda settings: index)
    assert cli.main(["ingest"]) == 0
    assert "Indexed 3 chunks from 2 documents" in capsys.readouterr().out

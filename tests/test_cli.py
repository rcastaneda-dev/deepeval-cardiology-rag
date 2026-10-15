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


def test_ask_strips_terminal_control_sequences_from_the_answer(monkeypatch, capsys):
    text = "\x1b[31mred\x1b[0m\x1b]0;title\x07 and\x00 more\nsecond\tline"
    result = RagResult("q", Answer(text, []), [])
    monkeypatch.setattr(
        cli.wiring, "build_pipeline", lambda settings: SimpleNamespace(ask=lambda q: result)
    )
    assert cli.main(["ask", "Anything?"]) == 0
    output = capsys.readouterr().out
    assert "red and more\nsecond\tline" in output
    assert "\x1b" not in output
    assert "\x07" not in output
    assert "\x00" not in output


def test_ask_keeps_text_between_st_terminated_osc_sequences(monkeypatch, capsys):
    text = "x\x1b]0;t\x1b\\KEEP1\ny\x1b]8;;u\x1b\\link\x1b]8;;\x1b\\z"
    result = RagResult("q", Answer(text, []), [])
    monkeypatch.setattr(
        cli.wiring, "build_pipeline", lambda settings: SimpleNamespace(ask=lambda q: result)
    )
    assert cli.main(["ask", "Anything?"]) == 0
    output = capsys.readouterr().out
    assert "xKEEP1\nylinkz" in output
    assert "\x1b" not in output


def test_ask_strips_a_lone_c1_csi_byte(monkeypatch, capsys):
    result = RagResult("q", Answer("a\x9b31mb", []), [])
    monkeypatch.setattr(
        cli.wiring, "build_pipeline", lambda settings: SimpleNamespace(ask=lambda q: result)
    )
    assert cli.main(["ask", "Anything?"]) == 0
    assert "\x9b" not in capsys.readouterr().out


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

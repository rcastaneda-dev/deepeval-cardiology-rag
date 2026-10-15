from types import SimpleNamespace

import pytest

from cardio_rag.generate import (
    SYSTEM_PROMPT,
    Answer,
    build_user_prompt,
    extract_citations,
    generate_answer,
)
from cardio_rag.retriever import RetrievedChunk

CHUNKS = [
    RetrievedChunk("heart-attack#0", "heart-attack", "Chest pain is a symptom.", 0.9),
    RetrievedChunk("heart-attack#1", "heart-attack", "Call 911 right away.", 0.8),
]


def text_block(text):
    return SimpleNamespace(type="text", text=text)


class FakeMessages:
    def __init__(self, content):
        self._content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=self._content)


def fake_client(content):
    return SimpleNamespace(messages=FakeMessages(content))


def test_user_prompt_lists_every_passage_with_its_id_and_the_question():
    prompt = build_user_prompt("What is angina?", CHUNKS)
    assert "[heart-attack#0]\nChest pain is a symptom." in prompt
    assert "[heart-attack#1]\nCall 911 right away." in prompt
    assert prompt.endswith("Question: What is angina?")


def test_user_prompt_says_so_when_nothing_was_retrieved():
    assert "(no passages were retrieved)" in build_user_prompt("Anything?", [])


def test_system_prompt_pins_abstention_and_emergency_rules():
    assert "I can't answer that from the available sources." in SYSTEM_PROMPT
    assert "911" in SYSTEM_PROMPT
    assert "Never diagnose" in SYSTEM_PROMPT


def test_citations_come_back_in_order_without_duplicates():
    text = "Pain [heart-attack#1] and more [heart-attack#0] and again [heart-attack#1]."
    assert extract_citations(text, CHUNKS) == ["heart-attack#1", "heart-attack#0"]


def test_citations_the_model_invented_are_dropped():
    assert extract_citations("Made up [stroke#9].", CHUNKS) == []


def test_an_answer_without_citations_has_none():
    assert extract_citations("I can't answer that from the available sources.", CHUNKS) == []


def test_generate_answer_sends_prompt_and_model_and_parses_citations():
    client = fake_client([text_block("Call 911 [heart-attack#1].")])
    answer = generate_answer("What now?", CHUNKS, client=client, model="claude-test")
    assert answer == Answer(text="Call 911 [heart-attack#1].", cited_ids=["heart-attack#1"])
    call = client.messages.calls[0]
    assert call["model"] == "claude-test"
    assert call["system"] == SYSTEM_PROMPT
    assert call["messages"] == [{"role": "user", "content": build_user_prompt("What now?", CHUNKS)}]
    assert call["max_tokens"] > 0


def test_generate_answer_joins_text_blocks_and_skips_other_blocks():
    content = [text_block("Part one. "), SimpleNamespace(type="thinking"), text_block("Part two.")]
    answer = generate_answer("Q", CHUNKS, client=fake_client(content), model="m")
    assert answer.text == "Part one. Part two."


def test_a_reply_with_no_text_is_an_error_not_an_empty_answer():
    with pytest.raises(RuntimeError, match="no text"):
        generate_answer("Q", CHUNKS, client=fake_client([]), model="m")

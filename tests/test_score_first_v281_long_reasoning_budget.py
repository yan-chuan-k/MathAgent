from __future__ import annotations

from typing import Any, Dict, List

import pytest

import intern_s1_client
from intern_s1_client import InternS1Client
from user_agent import ReasoningAgent, _SCORE_FIRST_DECISIVE_REASONING


class RecordingClient:
    def __init__(self, response: str = "Final answer: 42"):
        self.response = response
        self.calls: List[Dict[str, Any]] = []

    def chat(
        self,
        messages,
        temperature=None,
        top_p=None,
        max_tokens=None,
        thinking_mode=None,
    ):
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "thinking_mode": thinking_mode,
            }
        )
        return self.response


class NoTopPClient:
    def __init__(self):
        self.calls = []

    def chat(self, messages, temperature=None, max_tokens=None, thinking_mode=None):
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "thinking_mode": thinking_mode,
            }
        )
        return "Final answer: 42"


def test_default_score_first_uses_explicit_long_reasoning_budget():
    client = RecordingClient()
    agent = ReasoningAgent(client)

    assert agent.temperature == pytest.approx(0.8)
    assert agent.top_p == pytest.approx(0.95)
    assert agent.max_tokens == 32768
    assert agent.max_tokens > 8192
    assert agent.max_tokens != 0
    assert agent.thinking_mode is True

    result = agent.solve(
        "Compute 6*7.",
        {"subject": "Advanced Mathematics", "task_type": "calculation"},
    )

    assert result["final_response"] == "42"
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["temperature"] == pytest.approx(0.8)
    assert call["top_p"] == pytest.approx(0.95)
    assert call["max_tokens"] == 32768
    assert call["thinking_mode"] is True


def test_explicit_score_first_overrides_remain_respected():
    client = RecordingClient()
    agent = ReasoningAgent(
        client,
        temperature=0.63,
        top_p=0.88,
        max_tokens=12345,
        thinking_mode=False,
    )
    agent.solve("Compute 2+2.", {"subject": "Advanced Mathematics"})

    call = client.calls[0]
    assert call["temperature"] == pytest.approx(0.63)
    assert call["top_p"] == pytest.approx(0.88)
    assert call["max_tokens"] == 12345
    assert call["thinking_mode"] is False


def test_top_p_is_omitted_for_client_without_top_p_support():
    client = NoTopPClient()
    agent = ReasoningAgent(client)
    agent.solve("Compute 1+1.", {"subject": "Advanced Mathematics"})

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["temperature"] == pytest.approx(0.8)
    assert call["max_tokens"] == 32768
    assert call["thinking_mode"] is True
    assert "top_p" not in call


def test_100_successful_tasks_still_make_exactly_100_calls():
    client = RecordingClient()
    agent = ReasoningAgent(client)

    for index in range(100):
        result = agent.solve(
            f"Compute 6*7 for V2.8.1 call-budget regression {index}.",
            {"subject": "Advanced Mathematics", "task_type": "calculation"},
        )
        assert result["final_response"] == "42"

    assert len(client.calls) == 100
    assert all(call["temperature"] == pytest.approx(0.8) for call in client.calls)
    assert all(call["top_p"] == pytest.approx(0.95) for call in client.calls)
    assert all(call["max_tokens"] == 32768 for call in client.calls)
    assert all(call["thinking_mode"] is True for call in client.calls)


def test_softened_anti_loop_allows_one_correction():
    client = RecordingClient()
    agent = ReasoningAgent(client)
    agent.solve("Compute 3+4.", {"subject": "Advanced Mathematics"})

    system_prompt = client.calls[0]["messages"][0]["content"]
    assert system_prompt.count(_SCORE_FIRST_DECISIVE_REASONING) == 1
    assert "Restart only for a concrete error." in _SCORE_FIRST_DECISIVE_REASONING
    assert "If the independent check fails, correct once;" in _SCORE_FIRST_DECISIVE_REASONING
    assert "all parts are covered" in _SCORE_FIRST_DECISIVE_REASONING
    assert "Do not re-derive a verified solution." in _SCORE_FIRST_DECISIVE_REASONING


class _FakeMessage:
    content = "Final answer: 42"


class _FakeChoice:
    message = _FakeMessage()


class _FakeResponse:
    choices = [_FakeChoice()]


class _FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse()


class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()


class _FakeOpenAIClient:
    def __init__(self, *args, **kwargs):
        self.chat = _FakeChat()


def test_local_intern_client_transmits_positive_long_budget(monkeypatch):
    created = {}

    def fake_openai(*args, **kwargs):
        client = _FakeOpenAIClient()
        created["client"] = client
        return client

    monkeypatch.setattr(intern_s1_client, "OpenAI", fake_openai)

    client = InternS1Client(api_key="test-key", retry=1)
    result = client.chat(
        [{"role": "user", "content": "Compute 1+1."}],
        temperature=0.8,
        top_p=0.95,
        max_tokens=32768,
        thinking_mode=True,
    )

    assert result == "Final answer: 42"
    request = created["client"].chat.completions.calls[0]
    assert request["temperature"] == pytest.approx(0.8)
    assert request["top_p"] == pytest.approx(0.95)
    assert request["max_tokens"] == 32768
    assert request["max_tokens"] > 8192
    assert request["extra_body"] == {"thinking_mode": True}

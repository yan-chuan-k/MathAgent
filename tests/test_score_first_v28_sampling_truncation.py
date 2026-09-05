from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

import intern_s1_client
from intern_s1_client import InternS1Client
from user_agent import (
    ReasoningAgent,
    _SCORE_FIRST_DECISIVE_REASONING,
    _SCORE_FIRST_HUMAN_DOMAIN_LABELS,
)


ROOT = Path(__file__).resolve().parents[1]
ROUTING = ROOT / "sample_data" / "score_recovery_v2_synthetic_hard.jsonl"


class OfficialStyleRecordingClient:
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


class LegacyRecordingClient:
    def __init__(self, response: str = "Final answer: 42"):
        self.response = response
        self.calls: List[Dict[str, Any]] = []

    def chat(self, messages, temperature=None, max_tokens=None, thinking_mode=None):
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "thinking_mode": thinking_mode,
            }
        )
        return self.response


def _rows(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_default_score_first_uses_v28_official_sampling_profile():
    client = OfficialStyleRecordingClient()
    agent = ReasoningAgent(client)

    assert agent.production_mode == "score_first"
    assert agent.temperature == pytest.approx(0.8)
    assert agent.top_p == pytest.approx(0.95)
    assert agent.max_tokens == 32768
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
    assert "model" not in call


def test_score_first_explicit_sampling_overrides_remain_supported():
    client = OfficialStyleRecordingClient()
    agent = ReasoningAgent(
        client,
        temperature=0.61,
        top_p=0.87,
        max_tokens=1234,
        thinking_mode=False,
    )
    agent.solve("Compute 2+2.", {"subject": "Advanced Mathematics"})

    call = client.calls[0]
    assert call["temperature"] == pytest.approx(0.61)
    assert call["top_p"] == pytest.approx(0.87)
    assert call["max_tokens"] == 1234
    assert call["thinking_mode"] is False


def test_legacy_chat_signature_remains_compatible_without_top_p():
    client = LegacyRecordingClient()
    agent = ReasoningAgent(client)

    result = agent.solve("Compute 1+1.", {"subject": "Advanced Mathematics"})
    assert result["final_response"] == "42"
    assert len(client.calls) == 1
    assert client.calls[0]["temperature"] == pytest.approx(0.8)
    assert client.calls[0]["max_tokens"] == 32768
    assert client.calls[0]["thinking_mode"] is True


def test_orchestrated_defaults_are_not_changed_by_v28():
    client = OfficialStyleRecordingClient()
    agent = ReasoningAgent(client, production_mode="orchestrated")

    assert agent.temperature == pytest.approx(0.2)
    assert agent.top_p is None
    assert agent.max_tokens == 4096
    assert agent.thinking_mode is True


def test_100_successful_score_first_problems_equal_100_official_style_calls():
    client = OfficialStyleRecordingClient()
    agent = ReasoningAgent(client)

    for index in range(100):
        result = agent.solve(
            f"Compute 6*7 for V2.8 call-budget regression {index}.",
            {"subject": "Advanced Mathematics", "task_type": "calculation"},
        )
        assert result["final_response"] == "42"

    assert len(client.calls) == 100
    assert all(call["temperature"] == pytest.approx(0.8) for call in client.calls)
    assert all(call["top_p"] == pytest.approx(0.95) for call in client.calls)
    assert all(call["max_tokens"] == 32768 for call in client.calls)
    assert all(call["thinking_mode"] is True for call in client.calls)


def test_decisive_anti_loop_instruction_appears_exactly_once():
    client = OfficialStyleRecordingClient()
    agent = ReasoningAgent(client)
    agent.solve(
        "Find the expected value.",
        {"subject": "Probability Theory", "task_type": "calculation"},
    )

    system_prompt = client.calls[0]["messages"][0]["content"]
    assert system_prompt.count(_SCORE_FIRST_DECISIVE_REASONING) == 1
    assert (
        _SCORE_FIRST_DECISIVE_REASONING
        == "Use a direct path. Restart only for a concrete error. "
           "If the independent check fails, correct once; if it succeeds and all parts are covered, commit. "
           "Do not re-derive a verified solution."
    )


def test_v28_frozen_110_prompt_budget_stays_at_most_1900_chars():
    agent = ReasoningAgent(OfficialStyleRecordingClient())
    lengths = []

    for row in _rows(ROUTING):
        metadata = {
            "subject": _SCORE_FIRST_HUMAN_DOMAIN_LABELS[row["expected_domain"]],
            "task_type": row["task_type"],
        }
        messages = agent._build_score_first_prompt(row["problem"], metadata)
        lengths.append(sum(len(message["content"]) for message in messages))

    assert len(lengths) == 110
    assert max(lengths) <= 1900


class _FakeMessage:
    content = "Final answer: 42"


class _FakeChoice:
    message = _FakeMessage()


class _FakeCompletionResponse:
    choices = [_FakeChoice()]


class _FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeCompletionResponse()


class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()


class _FakeOpenAIClient:
    def __init__(self, *args, **kwargs):
        self.chat = _FakeChat()


def test_local_intern_client_passes_top_p_for_api_parity(monkeypatch):
    created = {}

    def fake_openai(*args, **kwargs):
        client = _FakeOpenAIClient()
        created["client"] = client
        created["init_kwargs"] = kwargs
        return client

    monkeypatch.setattr(intern_s1_client, "OpenAI", fake_openai)

    client = InternS1Client(
        api_key="test-key",
        model="platform-selected-test-model",
        retry=1,
    )
    result = client.chat(
        [{"role": "user", "content": "Compute 1+1."}],
        temperature=0.8,
        top_p=0.95,
        max_tokens=32768,
        thinking_mode=True,
    )

    assert result == "Final answer: 42"
    calls = created["client"].chat.completions.calls
    assert len(calls) == 1
    request = calls[0]
    assert request["model"] == "platform-selected-test-model"
    assert request["temperature"] == pytest.approx(0.8)
    assert request["top_p"] == pytest.approx(0.95)
    assert request["max_tokens"] == 32768
    assert request["extra_body"] == {"thinking_mode": True}


def test_local_intern_client_omits_top_p_when_not_supplied(monkeypatch):
    created = {}

    def fake_openai(*args, **kwargs):
        client = _FakeOpenAIClient()
        created["client"] = client
        return client

    monkeypatch.setattr(intern_s1_client, "OpenAI", fake_openai)

    client = InternS1Client(api_key="test-key", retry=1)
    client.chat(
        [{"role": "user", "content": "Compute 1+1."}],
        temperature=0.2,
        max_tokens=4096,
        thinking_mode=True,
    )

    request = created["client"].chat.completions.calls[0]
    assert "top_p" not in request

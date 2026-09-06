from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

import user_agent
from user_agent import (
    ReasoningAgent,
    _SCORE_FIRST_HUMAN_DOMAIN_LABELS,
    _SCORE_FIRST_RESPONSE_MODE_ANSWER,
    _SCORE_FIRST_RESPONSE_MODE_CONSTRUCTION,
    _SCORE_FIRST_RESPONSE_MODE_DERIVATION,
    _SCORE_FIRST_RESPONSE_MODE_PROOF,
    _SCORE_FIRST_RESPONSE_MODE_PROOF_OR_DISPROOF,
)


ROOT = Path(__file__).resolve().parents[1]
ROUTING = ROOT / "sample_data" / "score_recovery_v2_synthetic_hard.jsonl"


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


def _rows(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _prompt(agent: ReasoningAgent, problem: str, metadata=None):
    metadata = metadata or {}
    return agent._build_score_first_prompt(problem, metadata)


def _injected_text(messages, problem: str) -> str:
    # Keep the user's mathematical text out of negative prompt assertions.
    system = messages[0]["content"]
    user = messages[1]["content"].replace(problem, "<PROBLEM>")
    return system + "\n" + user


def test_default_score_first_prompt_profile_is_minimal():
    agent = ReasoningAgent(RecordingClient())
    assert agent.production_mode == "score_first"
    assert agent.score_first_prompt_profile == "minimal"


@pytest.mark.parametrize("bad", ["", "strategy", "legacy", "FULLER"])
def test_prompt_profile_validation_is_closed_world(bad):
    if bad == "":
        # Empty input follows constructor's explicit defaulting rule.
        assert ReasoningAgent(RecordingClient(), score_first_prompt_profile=bad).score_first_prompt_profile == "minimal"
    else:
        with pytest.raises(ValueError):
            ReasoningAgent(RecordingClient(), score_first_prompt_profile=bad)


def test_full_profile_is_explicitly_available():
    agent = ReasoningAgent(RecordingClient(), score_first_prompt_profile="full")
    assert agent.score_first_prompt_profile == "full"


def test_minimal_profile_does_not_call_router(monkeypatch):
    def fail_router(*args, **kwargs):
        raise AssertionError("minimal prompt path must not call classify_problem")

    monkeypatch.setattr(user_agent, "classify_problem", fail_router)
    client = RecordingClient()
    agent = ReasoningAgent(client)

    result = agent.solve(
        "Use the KKT conditions to maximize x subject to x<=1.",
        {},
    )

    assert result["final_response"] == "42"
    assert len(client.calls) == 1


def test_minimal_uses_only_trusted_subject_and_never_router_guess():
    trusted = ReasoningAgent(RecordingClient())
    messages = _prompt(
        trusted,
        "Compute P(A|B).",
        {"subject": "probability"},
    )
    assert "Subject: Probability Theory" in messages[1]["content"]

    untrusted = ReasoningAgent(RecordingClient())
    messages = _prompt(
        untrusted,
        "Evaluate the contour integral by residues.",
        {},
    )
    assert "Subject:" not in messages[1]["content"]
    assert "Complex Analysis" not in _injected_text(
        messages,
        "Evaluate the contour integral by residues.",
    )


@pytest.mark.parametrize(
    ("problem", "expected_mode", "required_system_phrase"),
    [
        (
            "Compute 2+2.",
            _SCORE_FIRST_RESPONSE_MODE_ANSWER,
            "Output exactly ONE visible line:",
        ),
        (
            "Derive the recurrence formula.",
            _SCORE_FIRST_RESPONSE_MODE_DERIVATION,
            "Give the final result first",
        ),
        (
            "Show f is continuous.",
            _SCORE_FIRST_RESPONSE_MODE_PROOF,
            "State the conclusion first",
        ),
        (
            "Prove or disprove: every bounded operator is compact.",
            _SCORE_FIRST_RESPONSE_MODE_PROOF_OR_DISPROOF,
            "Determine whether the claim is true or false.",
        ),
        (
            "Construct a counterexample to the converse.",
            _SCORE_FIRST_RESPONSE_MODE_CONSTRUCTION,
            "State the requested object or counterexample first",
        ),
    ],
)
def test_minimal_keeps_response_mode_parser_for_output_shape(
    problem,
    expected_mode,
    required_system_phrase,
):
    agent = ReasoningAgent(RecordingClient())
    context = agent._score_first_prompt_context(problem, {})
    assert context["response_mode"] == expected_mode

    messages = agent._build_score_first_prompt(problem, {}, context=context)
    assert required_system_phrase in messages[0]["content"]


@pytest.mark.parametrize(
    "problem",
    [
        "For x_{n+1}=g(x_n), determine the local fixed-point convergence condition.",
        "Use Bayes' theorem to compute P(D|+).",
        "Use KKT conditions to minimize the objective subject to the constraints.",
        "Evaluate the contour integral using residues.",
    ],
)
def test_minimal_prompt_has_no_injected_math_conditioning(problem):
    agent = ReasoningAgent(RecordingClient())
    messages = _prompt(agent, problem, {"subject": "advanced_math"})
    injected = _injected_text(messages, problem)

    forbidden = [
        "Domain strategy:",
        "Method hint:",
        "Subtype hint:",
        "Internal final check:",
        "Use a direct path.",
        "x*=g(x*)",
        "|g'(x*)|",
        "conditioning denominator",
        "complementary slackness",
        "Relist the singularities",
        "residue sum used in",
    ]
    for phrase in forbidden:
        assert phrase not in injected


def test_minimal_answer_contract_contains_only_general_output_safety():
    agent = ReasoningAgent(RecordingClient())
    messages = _prompt(
        agent,
        "Compute sqrt(2).",
        {"subject": "Advanced Mathematics"},
    )
    system = messages[0]["content"]

    assert "Final answer: <complete requested answer>" in system
    assert "Answer every requested part." in system
    assert "Use an exact form unless an approximation is requested." in system
    assert "Do not output JSON." in system
    assert "Do not repeat the problem." in system
    assert "Internal final check:" not in system
    assert "Domain strategy:" not in system


def test_minimal_trace_reports_prompt_profile():
    client = RecordingClient()
    agent = ReasoningAgent(client)
    result = agent.solve("Compute 2+2.", {"subject": "Advanced Mathematics"})

    profile_steps = [
        step for step in result["trace"]
        if step.get("step") == "prompt_profile"
    ]
    assert profile_steps == [{"step": "prompt_profile", "content": "minimal"}]


def test_full_trace_reports_prompt_profile():
    client = RecordingClient()
    agent = ReasoningAgent(client, score_first_prompt_profile="full")
    result = agent.solve("Compute 2+2.", {"subject": "Advanced Mathematics"})

    profile_steps = [
        step for step in result["trace"]
        if step.get("step") == "prompt_profile"
    ]
    assert profile_steps == [{"step": "prompt_profile", "content": "full"}]


def test_full_profile_retains_v281_conditioning_markers():
    agent = ReasoningAgent(RecordingClient(), score_first_prompt_profile="full")
    problem = "Use KKT conditions to minimize x^2 subject to x>=1."
    messages = _prompt(agent, problem, {"subject": "Optimization"})
    joined = "\n".join(message["content"] for message in messages)

    assert "Domain strategy:" in joined
    assert "Method hint:" in joined
    assert "Internal final check:" in joined
    assert "complementary slackness" in joined
    assert "Use a direct path." in joined


def test_minimal_and_full_share_frozen_inference_settings():
    minimal = ReasoningAgent(RecordingClient())
    full = ReasoningAgent(RecordingClient(), score_first_prompt_profile="full")

    for agent in (minimal, full):
        assert agent.temperature == pytest.approx(0.8)
        assert agent.top_p == pytest.approx(0.95)
        assert agent.max_tokens == 32768
        assert agent.thinking_mode is True


def test_100_minimal_score_first_tasks_equal_100_calls():
    client = RecordingClient()
    agent = ReasoningAgent(client)

    for index in range(100):
        result = agent.solve(
            f"Compute 6*7 for V2.9 minimal call-budget regression {index}.",
            {"subject": "Advanced Mathematics", "task_type": "calculation"},
        )
        assert result["final_response"] == "42"

    assert len(client.calls) == 100
    assert all(call["temperature"] == pytest.approx(0.8) for call in client.calls)
    assert all(call["top_p"] == pytest.approx(0.95) for call in client.calls)
    assert all(call["max_tokens"] == 32768 for call in client.calls)
    assert all(call["thinking_mode"] is True for call in client.calls)


def test_minimal_frozen_110_prompt_budget_is_below_700_chars():
    agent = ReasoningAgent(RecordingClient())
    lengths = []

    for row in _rows(ROUTING):
        metadata = {
            "subject": _SCORE_FIRST_HUMAN_DOMAIN_LABELS[row["expected_domain"]],
            "task_type": row["task_type"],
        }
        messages = agent._build_score_first_prompt(row["problem"], metadata)
        lengths.append(sum(len(message["content"]) for message in messages))

    assert len(lengths) == 110
    assert max(lengths) < 700


def test_full_frozen_110_prompt_budget_remains_at_most_1900_chars():
    agent = ReasoningAgent(
        RecordingClient(),
        score_first_prompt_profile="full",
    )
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


def test_minimal_problem_text_is_never_removed():
    problem = (
        "Let f(x)=x^2. Compute f(0.25), state the domain, and preserve the exact "
        "condition x>=0."
    )
    agent = ReasoningAgent(RecordingClient())
    messages = _prompt(agent, problem, {"subject": "Advanced Mathematics"})
    assert problem in messages[1]["content"]

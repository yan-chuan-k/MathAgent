from __future__ import annotations

from typing import Any, Dict, List

import pytest

from math_agent_core.tools.augmented_tool import SafeAugmentedToolExecutor, ToolRequest
from user_agent import (
    ReasoningAgent,
    _SCORE_FIRST_DEFAULT_EXPERIMENT_PRESET,
)


V305_PRESET = "tool_augmented_v305_full_final"
V306_PRESET = "tool_augmented_v307_cross_subject"


class SequenceClient:
    def __init__(self, responses: List[str]) -> None:
        self.responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        if not self.responses:
            raise AssertionError("unexpected extra model call")
        return self.responses.pop(0)


def _trace(result: Dict[str, Any]) -> Dict[str, str]:
    return {item["step"]: item["content"] for item in result["trace"]}


def _full_baseline(problem: str, metadata: Dict[str, Any]):
    return ReasoningAgent(
        SequenceClient([]),
        score_first_experiment_preset="full_thinking_on",
    )._build_score_first_prompt(problem, metadata)


def _no_decisive_failures(*args, **kwargs):
    result = ([], [], [])
    return result if kwargs.get("with_metadata") else result[0]


def test_v306_is_default_and_keeps_the_full_final_inference_settings():
    agent = ReasoningAgent(SequenceClient([]))

    assert _SCORE_FIRST_DEFAULT_EXPERIMENT_PRESET == V306_PRESET
    assert agent.score_first_experiment_preset == V306_PRESET
    assert agent.production_mode == "tool_augmented"
    assert agent.score_first_prompt_profile == "full"
    assert agent.thinking_mode is True
    assert agent.temperature == pytest.approx(0.8)
    assert agent.top_p == pytest.approx(0.95)
    assert agent.max_tokens == 49152
    assert agent.tool_augmented_extended_tools is True
    assert agent.tool_augmented_v307_tools is True
    assert agent.tool_augmented_challenge_reasoning is True
    assert agent.tool_augmented_grounded_auto_tools is True


def test_v306_standard_no_fact_prompt_is_byte_identical_to_full_baseline(monkeypatch):
    monkeypatch.setattr(
        ReasoningAgent,
        "_tool_augmented_decisive_failures",
        _no_decisive_failures,
    )
    problem = "What is the requested result?"
    metadata = {"subject": "Advanced Mathematics"}
    client = SequenceClient(["NO_TOOL", "Final answer: 42"])

    result = ReasoningAgent(client).solve(problem, metadata)

    assert result["final_response"] == "42"
    assert len(client.calls) == 2
    assert client.calls[-1]["messages"] == _full_baseline(problem, metadata)
    trace = _trace(result)
    assert trace["difficulty_tier"] == "standard"
    assert trace["grounded_auto_tool_successes"] == "0"
    assert trace["formalizer_plan_injected"] == "false"


def test_v306_high_difficulty_proof_keeps_full_conditioning_and_skips_extractor():
    problem = "Prove that every compact subset of a Hausdorff space is closed."
    metadata = {"subject": "Topology"}
    client = SequenceClient(["Conclusion: the subset is closed."])

    result = ReasoningAgent(client).solve(problem, metadata)

    assert len(client.calls) == 1
    baseline = _full_baseline(problem, metadata)
    final_messages = client.calls[0]["messages"]
    assert final_messages[0] == baseline[0]
    assert "Domain strategy:" in final_messages[1]["content"]
    assert "Internal final check:" in final_messages[1]["content"]
    assert "High-difficulty solving discipline:" in final_messages[1]["content"]
    assert "Compact formalizer plan (advisory):" not in final_messages[1]["content"]
    assert final_messages[1]["content"].index("High-difficulty solving discipline:") < final_messages[1]["content"].index("Problem:\n")
    trace = _trace(result)
    assert trace["difficulty_tier"] == "high"
    assert trace["tool_extractor_used"] == "false"
    assert trace["model_calls"] == "1"


def test_v306_parent_grounded_catalan_fact_survives_no_tool_extractor(monkeypatch):
    monkeypatch.setattr(
        ReasoningAgent,
        "_tool_augmented_decisive_failures",
        _no_decisive_failures,
    )
    problem = "Compute the Catalan number C_6."
    client = SequenceClient(["NO_TOOL", "Final answer: 132"])

    result = ReasoningAgent(client).solve(problem, {"subject": "Discrete Mathematics"})

    assert result["final_response"] == "132"
    assert len(client.calls) == 2
    final = client.calls[-1]["messages"][1]["content"]
    assert "Deterministically computed facts from explicit mathematical inputs:" in final
    assert "catalan[n=6] -> 132" in final
    assert "Compact formalizer plan (advisory):" not in final
    trace = _trace(result)
    assert trace["grounded_auto_tool_requests"] == "1"
    assert trace["grounded_auto_tool_successes"] == "1"
    assert trace["tool_successes"] == "0"


def test_v306_parent_grounded_scalar_fact_triggers_a_targeted_correction():
    problem = "Compute the Catalan number C_6."
    client = SequenceClient(
        [
            "NO_TOOL",
            "Final answer: 131",
            "Final answer: 132",
        ]
    )

    result = ReasoningAgent(client).solve(problem, {"subject": "Discrete Mathematics"})

    assert result["final_response"] == "132"
    assert len(client.calls) == 3
    correction = client.calls[-1]["messages"][1]["content"]
    assert "catalan[n=6] -> 132" in correction
    trace = _trace(result)
    assert trace["correction_call_used"] == "true"
    assert trace["correction_kept"] == "true"
    assert "parent_grounded_exact_fact" in trace["verification_failure_methods"]


def test_v306_parent_grounded_probability_fact_requires_all_parameters_in_target(monkeypatch):
    monkeypatch.setattr(
        ReasoningAgent,
        "_tool_augmented_decisive_failures",
        _no_decisive_failures,
    )
    problem = "Compute Binomial(n=10, p=1/2) P(X=3)."
    client = SequenceClient(["NO_TOOL", "Final answer: 15/128"])

    result = ReasoningAgent(client).solve(problem, {"subject": "Probability Theory"})

    assert result["final_response"] == "15/128"
    final = client.calls[-1]["messages"][1]["content"]
    assert "binomial_pmf[trials=10; successes=3; p=1/2] -> 15/128" in final
    assert _trace(result)["grounded_auto_tool_successes"] == "1"


def test_v306_auto_binding_ignores_unrequested_background_math(monkeypatch):
    monkeypatch.setattr(
        ReasoningAgent,
        "_tool_augmented_decisive_failures",
        _no_decisive_failures,
    )
    problem = "Given gcd(391,299)=23. Compute 1+1."
    client = SequenceClient(["NO_TOOL", "Final answer: 2"])

    ReasoningAgent(client).solve(problem, {})

    final = client.calls[-1]["messages"][1]["content"]
    assert "simplify[expr=1+1] -> 2" in final
    assert "gcd[a=391; b=299]" not in final


def test_v306_runs_parent_grounded_facts_when_the_extractor_protocol_is_malformed(monkeypatch):
    monkeypatch.setattr(
        ReasoningAgent,
        "_tool_augmented_decisive_failures",
        _no_decisive_failures,
    )
    problem = "Compute the Catalan number C_6."
    client = SequenceClient(["TOOL: sympy|solve|not-a-valid-argument", "Final answer: 132"])

    result = ReasoningAgent(client).solve(problem, {})

    final = client.calls[-1]["messages"][1]["content"]
    assert "catalan[n=6] -> 132" in final
    trace = _trace(result)
    assert trace["formalizer_protocol_usable"] == "false"
    assert trace["grounded_auto_tool_successes"] == "1"
    assert trace["tool_worker_status"] == "completed"


def test_v306_formalizer_gets_domain_specific_selection_guidance_only():
    agent = ReasoningAgent(SequenceClient([]))
    problem = "Compute Binomial(n=10, p=1/2) P(X=3)."
    context = agent._score_first_context(problem, {"subject": "Probability Theory"})

    messages = agent._build_tool_augmented_formalizer_prompt(problem, context)

    assert "Additional cross-subject deterministic TOOL syntax" in messages[0]["content"]
    assert "Subject-specific priority:" in messages[0]["content"]
    assert "explicitly parameterized finite distribution" in messages[0]["content"]
    assert "Subject-specific priority:" not in _full_baseline(problem, {"subject": "Probability Theory"})[1]["content"]


def test_v305_does_not_gain_v306_only_tool_operations(monkeypatch):
    monkeypatch.setattr(
        ReasoningAgent,
        "_tool_augmented_decisive_failures",
        _no_decisive_failures,
    )
    problem = "Compute Binomial(n=10, p=1/2) P(X=3)."
    client = SequenceClient(
        [
            "TOOL: probability|binomial_pmf|trials=10|successes=3|p=1/2",
            "Final answer: 15/128",
        ]
    )

    result = ReasoningAgent(
        client,
        score_first_experiment_preset=V305_PRESET,
    ).solve(problem, {"subject": "Probability Theory"})

    assert client.calls[-1]["messages"] == _full_baseline(
        problem,
        {"subject": "Probability Theory"},
    )
    assert _trace(result)["tool_successes"] == "0"


def test_v306_new_cross_subject_tool_families_are_exact_and_provenanced():
    requests = [
        ToolRequest("number_theory", "multiplicative_order", {"a": 3, "modulus": 17}),
        ToolRequest("number_theory", "primitive_root_check", {"a": 5, "modulus": 23}),
        ToolRequest("combinatorics", "partition_exact_parts", {"total": 10, "parts": 3}),
        ToolRequest("matrix", "trace", {"matrix": [[1, 2], [3, 4]]}),
        ToolRequest("matrix", "power", {"matrix": [[1, 1], [0, 1]], "exponent": 5}),
        ToolRequest("probability", "hypergeometric_pmf", {
            "population": 20,
            "success_states": 7,
            "draws": 5,
            "successes": 2,
        }),
    ]

    facts = SafeAugmentedToolExecutor(extended_tools=True).execute(requests)
    statements = [fact.statement for fact in facts]

    assert statements == [
        "multiplicative_order[a=3; modulus=17] -> 16",
        "primitive_root_check[a=5; modulus=23] -> True",
        "partition_exact_parts[total=10; parts=3] -> 8",
        "trace([[1,2],[3,4]]) -> 5",
        "power[matrix=[[1,1],[0,1]]; exponent=5] -> Matrix([[1, 5], [0, 1]])",
        "hypergeometric_pmf[population=20; success_states=7; draws=5; successes=2] -> 1001/2584",
    ]

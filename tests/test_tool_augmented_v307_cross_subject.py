from __future__ import annotations

import json
from typing import Any, Dict, List
from pathlib import Path

from math_agent_core.clients import MockClient
from user_agent import ReasoningAgent, _SCORE_FIRST_DEFAULT_EXPERIMENT_PRESET


class SequenceClient:
    def __init__(self, responses: List[str]) -> None:
        self.responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        if not self.responses:
            raise AssertionError("unexpected extra model call")
        return self.responses.pop(0)


def _auto(agent: ReasoningAgent, problem: str):
    context = agent._score_first_context(problem, {})
    return agent._tool_augmented_v306_grounded_auto_tool_requests(
        problem,
        context,
        agent._tool_augmented_verification_requests(problem, context),
    )


def test_v307_is_default_and_raises_only_the_new_completion_budget():
    agent = ReasoningAgent(MockClient())
    assert _SCORE_FIRST_DEFAULT_EXPERIMENT_PRESET == "tool_augmented_v307_cross_subject"
    assert agent.score_first_experiment_preset == "tool_augmented_v307_cross_subject"
    assert agent.thinking_mode is True
    assert agent.max_tokens == 49152
    assert agent.tool_augmented_v307_tools is True


def test_v307_explicit_inputs_cover_high_frequency_discrete_structures():
    agent = ReasoningAgent(MockClient())
    cases = {
        "How many integer partitions does 8 have?": ("combinatorics", "partition_total"),
        "Let a_0=2, a_1=5, and a_n=3a_{n-1}-2a_{n-2} for n>=2. Compute a_6.": (
            "recurrence",
            "linear_eval",
        ),
        "Find the least nonnegative residue of 3^100 modulo 7.": ("number_theory", "pow_mod"),
        "What is the chromatic number of the cycle C_9?": ("graph", "cycle_chromatic"),
        "Find [x^10] 1/(1-x)^4.": ("sympy", "coefficient"),
    }
    for problem, expected in cases.items():
        requests = _auto(agent, problem)
        assert any((request.family, request.operation) == expected for request in requests), (
            problem,
            requests,
        )


def test_v307_covers_explicit_probability_and_matrix_objects():
    agent = ReasoningAgent(MockClient())
    probability = _auto(agent, "P(X=3) where X~Binomial(n=10,p=1/2).")
    matrix = _auto(agent, "Let A=[[1,2],[3,4]]. Compute inverse(A).")
    assert any((request.family, request.operation) == ("probability", "binomial_pmf") for request in probability)
    assert any((request.family, request.operation) == ("matrix", "inverse") for request in matrix)


def test_v307_recurrence_recognizer_fails_closed_on_unparsed_forcing_terms():
    agent = ReasoningAgent(MockClient())
    requests = _auto(agent, "Suppose a_0=1 and a_n=a_{n-1}+sin(n) for n>=1. Evaluate a_20.")
    assert not any(request.family == "recurrence" for request in requests)


def test_v307_compact_output_discipline_is_present_without_exposing_scratch_work():
    agent = ReasoningAgent(MockClient())
    problem = "Compute the determinant of [[1,2],[3,4]]."
    context = agent._score_first_context(problem, {"subject": "Linear Algebra"})
    prompt = agent._build_tool_augmented_v307_final_prompt(
        problem,
        context,
        "Deterministically computed facts from the formalizer's exact requests:\n- det([[1,2],[3,4]]) -> -2",
    )
    user = prompt[1]["content"]
    assert "Output discipline:" in user
    assert "available internal reasoning budget" in user
    assert "no scratch work" in user


def test_v307_final_call_keeps_full_prompt_and_records_budget():
    client = SequenceClient(["NO_TOOL", "Final answer: 42"])
    agent = ReasoningAgent(client)
    result = agent.solve("What is the requested result?", {"subject": "Advanced Mathematics"})
    assert result["final_response"] == "42"
    assert len(client.calls) == 2
    trace = {item["step"]: item["content"] for item in result["trace"]}
    assert trace["completion_budget"] == "49152"
    assert trace["formalizer_plan_injected"] == "false"


def test_v307_discrete_realistic_suite_has_a_bounded_explicit_request_per_row():
    agent = ReasoningAgent(MockClient())
    fixture = Path(__file__).resolve().parents[1] / "sample_data" / "discrete_math_realistic_eval_v1.jsonl"
    rows = [json.loads(line) for line in fixture.read_text(encoding="utf-8").splitlines() if line.strip()]
    covered = 0
    for row in rows:
        problem = row["problem"]
        context = agent._score_first_context(problem, row.get("metadata", {}))
        requests = agent._tool_augmented_v306_grounded_auto_tool_requests(
            problem,
            context,
            agent._tool_augmented_verification_requests(problem, context),
        )
        covered += bool(requests)
        assert len(requests) <= 6
    assert covered == len(rows)

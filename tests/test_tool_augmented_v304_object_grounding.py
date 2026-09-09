from __future__ import annotations

from typing import Any, Dict, List

import pytest

from math_agent_core.tools.augmented_tool import run_answer_verification_in_subprocess
from user_agent import ReasoningAgent


class SequenceClient:
    def __init__(self, responses: List[str]) -> None:
        self.responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        if not self.responses:
            raise AssertionError("unexpected extra model call")
        return self.responses.pop(0)


def trace(result: Dict[str, Any]) -> Dict[str, str]:
    return {item["step"]: item["content"] for item in result["trace"]}


def _legacy_v31_agent(client: Any) -> ReasoningAgent:
    """Keep V3.0.4 object-grounding regressions on their rollback arm."""

    return ReasoningAgent(
        client,
        score_first_experiment_preset="tool_augmented_v31_extended",
    )


@pytest.mark.parametrize(
    ("problem", "answer"),
    [
        ("Given f(0)=1, solve x^2-5*x+6=0 for x.", "x=2 or x=3"),
        ("Let a=2. Solve x^2-5*x+6=0.", "x=2 or x=3"),
        ("For y(0)=1, find all roots of x^2-5*x+6=0.", "x=2 or x=3"),
        ("Given x=2, solve y^2-5*y+6=0 for y.", "y=2 or y=3"),
    ],
)
def test_equation_background_is_not_selected_as_the_target(problem: str, answer: str):
    client = SequenceClient(["TARGET: grounded", f"Final answer: {answer}", "Final answer: 99"])
    result = _legacy_v31_agent(client).solve(problem, {})
    assert result["final_response"] == answer
    assert len(client.calls) == 2
    assert trace(result)["correction_call_used"] == "false"


def test_incomplete_equation_answer_corrects_against_bound_target_only():
    problem = "Given f(0)=1, solve x^2-5*x+6=0 for x."
    client = SequenceClient(["TARGET: grounded", "Final answer: x=2 only", "Final answer: x=2 or x=3"])
    result = _legacy_v31_agent(client).solve(problem, {})
    assert result["final_response"] == "x=2 or x=3"
    assert len(client.calls) == 3
    correction_prompt = client.calls[2]["messages"][1]["content"]
    assert "x^2-5*x+6=0" in correction_prompt
    assert "f(0)=1" not in correction_prompt.split("Exact failed deterministic check(s):", 1)[1].split("ORIGINAL PROBLEM:", 1)[0]


@pytest.mark.parametrize(
    ("problem", "answer"),
    [
        ("Let A=[[1,0],[0,1]] and B=[[1,2],[3,4]]. Compute det(B).", "-2"),
        ("Given A=[[1,0],[0,1]], find rank(B=[[1,2],[2,4]]).", "1"),
        ("For matrix A=[[1,0],[0,1]], compute determinant of B=[[1,2],[3,4]].", "-2"),
        ("Given A=[[1,0],[0,1]], calculate det([[1,2],[3,4]]).", "-2"),
    ],
)
def test_matrix_binding_uses_the_requested_literal_or_named_object(problem: str, answer: str):
    client = SequenceClient(["TARGET: grounded", f"Final answer: {answer}", "Final answer: 99"])
    result = _legacy_v31_agent(client).solve(problem, {})
    assert result["final_response"] == answer
    assert len(client.calls) == 2


def test_named_matrix_correction_evidence_is_for_B_not_A():
    problem = "Let A=[[1,0],[0,1]] and B=[[1,2],[3,4]]. Compute det(B)."
    client = SequenceClient(["TARGET: grounded", "Final answer: -1", "Final answer: -2"])
    result = _legacy_v31_agent(client).solve(problem, {})
    assert result["final_response"] == "-2"
    assert len(client.calls) == 3
    failure = client.calls[2]["messages"][1]["content"]
    assert "matrix=[[1, 2], [3, 4]]" in failure
    assert "matrix=[[1, 0], [0, 1]]" not in failure


@pytest.mark.parametrize("problem", [
    "Solve for x^4 given x=2.",
    "Solve for f(x) given f(0)=1.",
    "Find x^4 given x=2.",
])
def test_expression_targets_are_not_classified_as_bare_variable_solves(problem: str):
    client = SequenceClient(["TARGET: grounded", "Final answer: 16", "Final answer: 99"])
    result = _legacy_v31_agent(client).solve(problem, {})
    assert result["final_response"] == "16"
    assert len(client.calls) == 2


def test_worker_accepts_only_explicit_requests_and_never_scans_problem_text():
    evidence = run_answer_verification_in_subprocess(
        "-2",
        "ANSWER_VALUE",
        requests=[{"kind": "matrix_determinant", "matrix": [[1, 2], [3, 4]]}],
    )
    assert evidence and evidence[0]["status"] == "pass"
    assert "matrix=[[1, 2], [3, 4]]" in evidence[0]["details"]


def test_parent_bound_target_is_precomputed_when_formalizer_omits_tool():
    client = SequenceClient(["TARGET: arithmetic", "Final answer: 2"])
    result = _legacy_v31_agent(client).solve("Compute 1+1.", {})
    assert result["final_response"] == "2"
    assert len(client.calls) == 2
    trace_data = trace(result)
    assert trace_data["target_tool_requested_ops"] == "sympy:simplify"
    assert trace_data["target_tool_succeeded_ops"] == "sympy:simplify"
    final_prompt = client.calls[1]["messages"][1]["content"]
    assert "Parent-bound target facts are listed first" in final_prompt
    assert "simplify[expr=1+1] -> 2" in final_prompt


def test_correction_is_not_allowed_to_increase_decisive_failures():
    problem = "Solve x^2-5*x+6=0 for x."
    client = SequenceClient(
        ["TARGET: solve", "Final answer: x=2", "Final answer: 999"]
    )
    result = _legacy_v31_agent(client).solve(problem, {})
    # Call 2 is incomplete but less wrong than the deliberately bad Call 3;
    # post-correction verification therefore retains the safer prior answer.
    assert result["final_response"] == "x=2"
    assert len(client.calls) == 3
    trace_data = trace(result)
    assert trace_data["correction_reverified"] == "true"
    assert trace_data["correction_kept"] == "false"


@pytest.mark.parametrize(
    "requests",
    [
        [{"kind": "matrix_determinant", "matrix": [[1, 2], [3, 4]], "problem": "det(A)"}],
        [{"kind": "equation_solution", "equation": "x^2=4", "variable": "x", "object": "x^2=4"}],
        [{"kind": "matrix_rank", "matrix": [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]]}],
    ],
)
def test_worker_rejects_non_schema_or_oversized_requests(requests):
    assert run_answer_verification_in_subprocess("2", "ANSWER_VALUE", requests=requests) == []

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import json

import pytest

from math_agent_core.tools.augmented_tool import (
    SafeAugmentedToolExecutor,
    ToolRequest,
    build_evidence_block,
    parse_formalizer_protocol,
)
from user_agent import ReasoningAgent




ROOT = Path(__file__).resolve().parents[1]
INTEGRATION_FIXTURE = ROOT / "sample_data" / "tool_augmented_v30_integration.jsonl"


def _fixture_rows():
    return [
        json.loads(line)
        for line in INTEGRATION_FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class SequenceClient:
    def __init__(self, responses):
        self.responses = list(responses)
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
        if not self.responses:
            raise AssertionError("unexpected extra model call")
        return self.responses.pop(0)


class TwoStageSuccessClient:
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def chat(
        self,
        messages,
        temperature=None,
        top_p=None,
        max_tokens=None,
        thinking_mode=None,
    ):
        call = {
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "thinking_mode": thinking_mode,
        }
        self.calls.append(call)
        system = messages[0]["content"]
        if "compact mathematical formalizer" in system:
            return "TARGET: exact arithmetic\nMETHOD: compute directly"
        return "Final answer: 42"


def _trace(result):
    return {item["step"]: item["content"] for item in result["trace"]}


def _legacy_v31_agent(client: Any, **kwargs: Any) -> ReasoningAgent:
    """Keep V3.0 behavioural regressions on the selectable rollback arm."""

    return ReasoningAgent(
        client,
        score_first_experiment_preset="tool_augmented_v31_extended",
        **kwargs,
    )


def test_tool_augmented_preset_selects_new_production_mode():
    agent = ReasoningAgent(
        SequenceClient([]),
        score_first_experiment_preset="tool_augmented_v30",
    )
    assert agent.production_mode == "tool_augmented"
    assert agent.score_first_prompt_profile == "minimal"
    assert agent.temperature == pytest.approx(0.8)
    assert agent.top_p == pytest.approx(0.95)
    assert agent.max_tokens == 32768
    assert agent.thinking_mode is True


def test_explicit_production_mode_overrides_tool_augmented_preset():
    agent = ReasoningAgent(
        SequenceClient([]),
        score_first_experiment_preset="tool_augmented_v30",
        production_mode="score_first",
    )
    assert agent.production_mode == "score_first"


def test_tool_augmented_explicit_mode_uses_v30_inference_defaults():
    agent = ReasoningAgent(
        SequenceClient([]),
        production_mode="tool_augmented",
    )
    assert agent.temperature == pytest.approx(0.8)
    assert agent.top_p == pytest.approx(0.95)
    assert agent.max_tokens == 32768
    assert agent.thinking_mode is True


def test_protocol_parser_is_tolerant_and_closed_world():
    protocol = parse_formalizer_protocol(
        """
        random prose that is ignored
        TARGET: solve the roots
        METHOD: factor
        FACT: polynomial is quadratic
        TOOL: sympy|solve|equation=x^2-5*x+6=0|var=x|domain=real
        TOOL: shell|run|command=rm -rf /
        TOOL: matrix|determinant|matrix=[[1,2],[3,4]]
        BROKEN:
        """
    )
    assert protocol.plan_lines == [
        "TARGET: solve the roots",
        "METHOD: factor",
        "FACT: polynomial is quadratic",
    ]
    assert [(req.family, req.operation) for req in protocol.tool_requests] == [
        ("sympy", "solve"),
        ("matrix", "determinant"),
    ]


def test_safe_tool_layer_computes_exact_facts_without_code_execution():
    protocol = parse_formalizer_protocol(
        """
        TOOL: sympy|solve|equation=x^2-5*x+6=0|var=x|domain=real
        TOOL: matrix|determinant|matrix=[[1,2],[3,4]]
        """
    )
    facts = SafeAugmentedToolExecutor().execute(protocol.tool_requests)
    statements = [fact.statement for fact in facts]
    assert any("solve[x^2-5*x+6=0, x over Reals] -> {2, 3}" in item for item in statements)
    assert any("det([[1,2],[3,4]]) -> -2" in item for item in statements)


def test_dangerous_model_generated_expression_is_ignored():
    protocol = parse_formalizer_protocol(
        'TOOL: sympy|simplify|expr=__import__("os").system("echo hacked")'
    )
    facts = SafeAugmentedToolExecutor().execute(protocol.tool_requests)
    assert facts == []


@pytest.mark.parametrize(
    ("tool_request", "expected"),
    [
        (
            ToolRequest("sympy", "factor", {"expr": "x^2-5*x+6"}),
            "(x - 3)*(x - 2)",
        ),
        (
            ToolRequest("sympy", "differentiate", {"expr": "x^3", "var": "x"}),
            "3*x**2",
        ),
        (
            ToolRequest(
                "sympy",
                "integrate",
                {"expr": "2*x^2", "var": "x", "lower": "0", "upper": "1"},
            ),
            "integral[2*x^2, x=0..1] -> 2/3",
        ),
        (
            ToolRequest("sympy", "mod", {"expr": "17", "modulus": 5}),
            "mod[expr=17; modulus=5] -> 2",
        ),
        (
            ToolRequest(
                "sympy",
                "ode_residual",
                {"candidate": "exp(x)", "rhs": "y", "var": "x", "state": "y"},
            ),
            "ode_residual[candidate=exp(x); rhs=y; var=x; state=y] -> 0",
        ),
        (
            ToolRequest(
                "sympy",
                "residue",
                {"expr": "1/(z*(z+1))", "var": "z", "point": "0"},
            ),
            "residue[1/(z*(z+1)), z=0] -> 1",
        ),
        (
            ToolRequest(
                "sympy",
                "stationary",
                {"expr": "x^2+y^2", "variables": "x,y"},
            ),
            "x: 0",
        ),
        (
            ToolRequest(
                "sympy",
                "series",
                {"expr": "1/(1-x)", "var": "x", "point": "0", "order": 4},
            ),
            "x**3",
        ),
        (
            ToolRequest("matrix", "rank", {"matrix": [[1, 2], [2, 4]]}),
            "rank([[1,2],[2,4]]) -> 1",
        ),
        (
            ToolRequest(
                "matrix",
                "linear_solve",
                {"matrix": [[2, 0], [0, 3]], "rhs": [4, 9]},
            ),
            "(2, 3)",
        ),
        (
            ToolRequest(
                "matrix",
                "eigenvalues",
                {"matrix": [[2, 0], [0, 3]]},
            ),
            "2: multiplicity 1",
        ),
    ],
)
def test_representative_safe_tool_operations(tool_request, expected):
    fact = SafeAugmentedToolExecutor().execute_one(tool_request)
    assert fact is not None
    assert expected in fact.statement


def test_compact_tool_evidence_stays_under_cap():
    requests = [
        ToolRequest("sympy", "simplify", {"expr": f"x+{i}-x"})
        for i in range(6)
    ]
    facts = SafeAugmentedToolExecutor().execute(requests)
    block = build_evidence_block(facts)
    assert len(block) <= 1500
    assert block.startswith("Deterministically computed facts from the formalizer's exact requests:")


def test_normal_tool_augmented_pipeline_is_two_calls_and_injects_verified_fact():
    client = SequenceClient(
        [
            (
                "TARGET: solve all real roots\n"
                "METHOD: factor\n"
                "TOOL: sympy|solve|equation=x^2-5*x+6=0|var=x|domain=real"
            ),
            "Final answer: x=2 or x=3",
        ]
    )
    agent = ReasoningAgent(
        client,
        score_first_experiment_preset="tool_augmented_v30",
    )
    result = agent.solve(
        "Solve x^2 - 5*x + 6 = 0 for x.",
        {"subject": "Advanced Mathematics"},
    )

    assert result["final_response"] == "x=2 or x=3"
    assert len(client.calls) == 2

    formalizer = client.calls[0]
    assert formalizer["temperature"] == pytest.approx(0.2)
    assert formalizer["top_p"] == pytest.approx(0.95)
    assert formalizer["max_tokens"] == 1536
    assert formalizer["thinking_mode"] is False

    final_call = client.calls[1]
    assert final_call["temperature"] == pytest.approx(0.8)
    assert final_call["top_p"] == pytest.approx(0.95)
    assert final_call["max_tokens"] == 32768
    assert final_call["thinking_mode"] is True
    final_prompt = "\n".join(m["content"] for m in final_call["messages"])
    assert "Deterministically computed facts from the formalizer's exact requests:" in final_prompt
    assert "solve[x^2-5*x+6=0, x over Reals] -> {2, 3}" in final_prompt
    assert "The formalizer plan is advisory" in final_prompt

    trace = _trace(result)
    assert trace["model_calls"] == "2"
    assert trace["formalizer_used"] == "true"
    assert trace["tool_requests"] == "1"
    assert trace["tool_successes"] == "1"
    assert trace["decisive_check_failure"] == "false"
    assert trace["correction_call_used"] == "false"


def test_final_solver_prompt_has_no_candidate_or_generic_critic_cascade():
    client = SequenceClient(
        [
            "TARGET: compute",
            "Final answer: 42",
        ]
    )
    agent = ReasoningAgent(
        client,
        score_first_experiment_preset="tool_augmented_v31_extended",
    )
    result = agent.solve("Compute 6*7.", {})
    assert result["final_response"] == "42"
    assert agent.orchestrator is None
    combined = "\n".join(
        msg["content"]
        for call in client.calls
        for msg in call["messages"]
    ).lower()
    assert "candidate b" not in combined
    assert "generic critic" not in combined
    assert len(client.calls) == 2


def test_proof_plan_mode_splits_planning_from_visible_complete_proof():
    client = SequenceClient(
        [
            (
                "CLAIM: continuous bijection from compact to Hausdorff is a homeomorphism\n"
                "HYPOTHESES: compact domain; Hausdorff codomain\n"
                "SUBGOAL: show inverse maps closed sets to closed sets\n"
                "THEOREM: compact subsets of Hausdorff spaces are closed"
            ),
            (
                "Conclusion: the map is a homeomorphism.\n"
                "Proof: A closed subset of the compact domain is compact. "
                "Its image is compact and hence closed in the Hausdorff codomain. "
                "Thus the bijection is closed, so its inverse is continuous."
            ),
        ]
    )
    agent = ReasoningAgent(
        client,
        score_first_experiment_preset="tool_augmented_v31_extended",
    )
    result = agent.solve(
        "Prove that a continuous bijection from a compact space to a Hausdorff space is a homeomorphism.",
        {"subject": "Topology"},
    )

    assert len(client.calls) == 2
    assert "This is a proof-oriented task." in client.calls[0]["messages"][1]["content"]
    assert "Conclusion:" in result["final_response"]
    assert "Proof:" in result["final_response"]
    assert _trace(result)["correction_call_used"] == "false"


def test_decisive_failure_triggers_one_targeted_correction_call():
    client = SequenceClient(
        [
            "TARGET: compute exact arithmetic\nMETHOD: direct arithmetic",
            "Final answer: 3",
            "Final answer: 2",
        ]
    )
    agent = _legacy_v31_agent(client)
    result = agent.solve("Compute 1+1.", {})

    assert result["final_response"] == "2"
    assert len(client.calls) == 3
    correction = "\n".join(m["content"] for m in client.calls[2]["messages"])
    assert "numeric_arithmetic" in correction
    assert "residual=-1" in correction
    assert "generic" not in correction.lower()

    trace = _trace(result)
    assert trace["model_calls"] == "3"
    assert trace["decisive_check_failure"] == "true"
    assert trace["correction_call_used"] == "true"


def test_correction_path_is_hard_capped_at_three_calls_even_if_still_wrong():
    client = SequenceClient(
        [
            "TARGET: compute exact arithmetic",
            "Final answer: 3",
            "Final answer: 4",
        ]
    )
    agent = _legacy_v31_agent(client)
    result = agent.solve("Compute 1+1.", {})
    assert result["final_response"] == "4"
    assert len(client.calls) == 3
    assert _trace(result)["model_calls"] == "3"


def test_uncheckable_answer_does_not_trigger_generic_review_call():
    client = SequenceClient(
        [
            "TARGET: identify requested mathematical object",
            "Final answer: Abelian",
        ]
    )
    agent = _legacy_v31_agent(client)
    result = agent.solve(
        "State one adjective commonly used to describe a commutative group.",
        {"subject": "Abstract Algebra"},
    )
    assert result["final_response"] == "Abelian"
    assert len(client.calls) == 2
    assert _trace(result)["correction_call_used"] == "false"


def test_100_normal_tool_augmented_tasks_use_at_most_200_calls():
    client = TwoStageSuccessClient()
    agent = ReasoningAgent(
        client,
        score_first_experiment_preset="tool_augmented_v30",
    )

    for index in range(100):
        result = agent.solve(
            f"Compute 6*7. Case label {index}.",
            {"subject": "Advanced Mathematics"},
        )
        assert result["final_response"] == "42"
        assert _trace(result)["correction_call_used"] == "false"

    assert len(client.calls) == 200
    assert all(
        call["thinking_mode"] is False
        for call in client.calls[0::2]
    )
    assert all(
        call["thinking_mode"] is True
        for call in client.calls[1::2]
    )


def test_malformed_tool_request_falls_back_to_final_solver():
    client = SequenceClient(
        [
            "TARGET: solve\nTOOL: sympy|solve|this-is-not-valid",
            "Final answer: 5",
        ]
    )
    agent = _legacy_v31_agent(client)
    result = agent.solve("Compute 2+3.", {})
    assert result["final_response"] == "5"
    assert len(client.calls) == 2
    trace = _trace(result)
    assert trace["tool_requests"] == "0"
    assert trace["tool_successes"] == "0"


def test_cross_domain_integration_fixture_executes_compute_tools_and_keeps_proofs_tool_free():
    executor = SafeAugmentedToolExecutor()
    rows = _fixture_rows()
    assert len(rows) == 17

    compute_seen = 0
    proof_seen = 0
    for row in rows:
        protocol = parse_formalizer_protocol(row["formalizer"])
        if row["kind"] == "compute":
            compute_seen += 1
            assert protocol.tool_requests, row["idx"]
            facts = executor.execute(protocol.tool_requests)
            assert facts, row["idx"]
            joined = "\n".join(fact.statement for fact in facts)
            assert row["expected_fact"] in joined, (row["idx"], joined)
            assert len(build_evidence_block(facts)) <= 1500
        else:
            proof_seen += 1
            assert protocol.tool_requests == [], row["idx"]
            assert protocol.plan_lines, row["idx"]
            assert any(
                line.startswith(("CLAIM:", "SUBGOAL:", "THEOREM:", "HYPOTHESES:"))
                for line in protocol.plan_lines
            )

    assert compute_seen == 12
    assert proof_seen == 5

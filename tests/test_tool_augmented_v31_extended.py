from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from math_agent_core.tools.augmented_tool import (
    SafeAugmentedToolExecutor,
    ToolRequest,
    build_evidence_block,
    parse_formalizer_protocol,
)
from user_agent import ReasoningAgent, _SCORE_FIRST_DEFAULT_EXPERIMENT_PRESET


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "sample_data" / "tool_augmented_v31_extended.jsonl"


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


def _rows():
    return [
        json.loads(line)
        for line in FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_v306_is_submission_default_and_v31_extended_remains_rollback():
    core = ReasoningAgent(
        SequenceClient([]),
        score_first_experiment_preset="tool_augmented_v30",
    )
    extended = ReasoningAgent(
        SequenceClient([]),
        score_first_experiment_preset="tool_augmented_v31_extended",
    )
    submission = ReasoningAgent(SequenceClient([]))

    assert _SCORE_FIRST_DEFAULT_EXPERIMENT_PRESET == "tool_augmented_v307_cross_subject"
    assert submission.score_first_experiment_preset == "tool_augmented_v307_cross_subject"
    assert submission.tool_augmented_extended_tools is True
    assert submission.score_first_prompt_profile == "full"
    assert core.production_mode == "tool_augmented"
    assert core.tool_augmented_extended_tools is False

    assert extended.production_mode == "tool_augmented"
    assert extended.tool_augmented_extended_tools is True
    assert extended.temperature == pytest.approx(core.temperature)
    assert extended.top_p == pytest.approx(core.top_p)
    assert extended.max_tokens == core.max_tokens == 32768
    assert extended.thinking_mode is core.thinking_mode is True


def test_extended_protocol_accepts_only_enabled_closed_world_families():
    raw = "\n".join(
        [
            "TOOL: number_theory|gcd|a=252|b=105",
            "TOOL: combinatorics|derangement|n=6",
            "TOOL: recurrence|linear_eval|initial=[2,5]|coefficients=[3,-2]|constant=0|target_n=4",
            "TOOL: graph|tree_edge_count|vertices=9",
            "TOOL: numerical|newton_step|f=x^2-2|variable=x|x0=3/2",
            "TOOL: z3|linear_int|variables=[\"x\"]|constraints=[\"x>=1\"]",
            "TOOL: shell|run|command=echo bad",
        ]
    )

    core = parse_formalizer_protocol(raw, extended_tools=False)
    extended = parse_formalizer_protocol(raw, extended_tools=True)

    assert core.tool_requests == []
    assert [request.family for request in extended.tool_requests] == [
        "number_theory",
        "combinatorics",
        "recurrence",
        "graph",
        "numerical",
    ]


def test_extended_formalizer_advertises_only_implemented_tools_and_no_z3():
    client = SequenceClient(
        [
            "TARGET: exact discrete computation",
            "Final answer: 1",
        ]
    )
    agent = ReasoningAgent(
        client,
        score_first_experiment_preset="tool_augmented_v31_extended",
    )
    agent.solve("Compute gcd(2,1).", {})
    system = client.calls[0]["messages"][0]["content"]

    assert "number_theory|gcd" in system
    assert "combinatorics|derangement" in system
    assert "recurrence|linear_eval" in system
    assert "initial=[a0,...,a{k-1}]" in system
    assert "coefficients=[c1,...,ck]" in system
    assert "a_n=c1*a_{n-1}+c2*a_{n-2}+...+ck*a_{n-k}+constant" in system
    assert "sympy|coefficient" in system
    assert "graph|degree_sequence_graphical" in system
    assert "numerical|newton_step" in system
    assert "matrix|nullspace" in system
    assert "z3|" not in system.lower()


def test_extended_fixture_returns_exact_general_results_with_provenance():
    rows = _rows()
    assert len(rows) >= 25
    executor = SafeAugmentedToolExecutor(extended_tools=True)

    for row in rows:
        request = ToolRequest(
            row["family"],
            row["operation"],
            row["arguments"],
        )
        fact = executor.execute_one(request)
        assert fact is not None, row["idx"]
        assert row["expected"] in fact.statement, (row["idx"], fact.statement)

        block = build_evidence_block([fact])
        assert "formalizer's exact requests" in block
        assert len(block) <= 1500


@pytest.mark.parametrize(
    ("operation", "arguments", "must_show"),
    [
        ("gcd", {"a": 252, "b": 105}, "a=252; b=105"),
        ("mod_inverse", {"a": 17, "modulus": 43}, "a=17; modulus=43"),
        ("crt", {"residues": [2, 3], "moduli": [5, 7]}, "residues=[2,3]; moduli=[5,7]"),
        ("totient", {"n": 36}, "n=36"),
        ("pow_mod", {"base": 5, "exponent": 117, "modulus": 19}, "base=5; exponent=117; modulus=19"),
    ],
)
def test_number_theory_facts_bind_exact_inputs(operation, arguments, must_show):
    fact = SafeAugmentedToolExecutor(extended_tools=True).execute_one(
        ToolRequest("number_theory", operation, arguments)
    )
    assert fact is not None
    assert must_show in fact.statement


@pytest.mark.parametrize(
    ("operation", "arguments", "must_show"),
    [
        ("derangement", {"n": 6}, "n=6"),
        ("onto_functions", {"domain_size": 5, "codomain_size": 3}, "domain_size=5; codomain_size=3"),
        ("stirling2", {"n": 4, "k": 2}, "n=4; k=2"),
        ("catalan", {"n": 4}, "n=4"),
        ("stars_bars", {"total": 7, "variables": 3, "minimum_each": 0}, "total=7; variables=3; minimum_each=0"),
    ],
)
def test_combinatorics_facts_bind_exact_inputs(operation, arguments, must_show):
    fact = SafeAugmentedToolExecutor(extended_tools=True).execute_one(
        ToolRequest("combinatorics", operation, arguments)
    )
    assert fact is not None
    assert must_show in fact.statement


def test_stars_and_bars_requires_and_displays_the_minimum_each_semantics():
    executor = SafeAugmentedToolExecutor(extended_tools=True)

    nonnegative = executor.execute_one(
        ToolRequest("combinatorics", "stars_bars", {"total": 7, "variables": 3, "minimum_each": 0})
    )
    positive = executor.execute_one(
        ToolRequest("combinatorics", "stars_bars", {"total": 7, "variables": 3, "minimum_each": 1})
    )
    missing = executor.execute_one(
        ToolRequest("combinatorics", "stars_bars", {"total": 7, "variables": 3})
    )

    assert nonnegative is not None
    assert nonnegative.statement.endswith("minimum_each=0] -> 36")
    assert positive is not None
    assert positive.statement.endswith("minimum_each=1] -> 15")
    assert missing is None


def test_recurrence_evaluation_is_general_and_bounded():
    fact = SafeAugmentedToolExecutor(extended_tools=True).execute_one(
        ToolRequest(
            "recurrence",
            "linear_eval",
            {
                "initial": [2, 5],
                "coefficients": [3, -2],
                "constant": 0,
                "target_n": 4,
            },
        )
    )
    assert fact is not None
    assert "initial=[2,5]" in fact.statement
    assert "coefficients=[3,-2]" in fact.statement
    assert "a_n=3*a_{n-1}-2*a_{n-2}+0" in fact.statement
    assert "coefficient_order=c_1->a_{n-1},c_2->a_{n-2}" in fact.statement
    assert "target_n=4" in fact.statement
    assert fact.statement.endswith("-> 47")

    too_large = SafeAugmentedToolExecutor(extended_tools=True).execute_one(
        ToolRequest(
            "recurrence",
            "linear_eval",
            {
                "initial": [1],
                "coefficients": [1],
                "constant": 0,
                "target_n": 50000,
            },
        )
    )
    assert too_large is None


def test_generating_function_coefficient_has_full_expression_provenance():
    fact = SafeAugmentedToolExecutor(extended_tools=True).execute_one(
        ToolRequest(
            "sympy",
            "coefficient",
            {
                "expr": "1/((1-x)*(1-x^2))",
                "var": "x",
                "power": 6,
            },
        )
    )
    assert fact is not None
    assert "coefficient[x^6 of 1/((1-x)*(1-x^2))] -> 4" in fact.statement


@pytest.mark.parametrize(
    ("operation", "arguments", "expected"),
    [
        ("tree_edge_count", {"vertices": 9}, "-> 8"),
        ("edge_count_from_degrees", {"degrees": [3,3,2,2,2,2]}, "-> 7"),
        ("complete_graph_edges", {"n": 6}, "-> 15"),
        ("complete_bipartite_edges", {"m": 3, "n": 4}, "-> 12"),
        ("cycle_space_dimension", {"vertices": 6, "edges": 8, "components": 1}, "-> 3"),
        ("eulerian_degree_check", {"degrees": [2,4,2,4], "connected": True}, "eulerian_circuit_criterion=True"),
        ("degree_sequence_graphical", {"degrees": [3,3,2,2,2]}, "-> True"),
    ],
)
def test_graph_tools_return_exact_closed_world_results(operation, arguments, expected):
    fact = SafeAugmentedToolExecutor(extended_tools=True).execute_one(
        ToolRequest("graph", operation, arguments)
    )
    assert fact is not None
    assert expected in fact.statement


def test_eulerian_connected_value_is_shown_as_request_assumption():
    fact = SafeAugmentedToolExecutor(extended_tools=True).execute_one(
        ToolRequest(
            "graph",
            "eulerian_degree_check",
            {"degrees": [2, 4, 2, 4], "connected": True},
        )
    )
    assert fact is not None
    assert "eulerian_degree_check[degrees=[2,4,2,4]; connected=true]" in fact.statement
    assert "derived_all_even=True" in fact.statement


@pytest.mark.parametrize(
    ("family", "operation", "arguments"),
    [
        (
            "number_theory",
            "crt",
            {"residues": list(range(9)), "moduli": [11, 13, 17, 19, 23, 29, 31, 37, 41]},
        ),
        (
            "recurrence",
            "linear_eval",
            {"initial": [1] * 9, "coefficients": [1] * 9, "constant": 0, "target_n": 12},
        ),
        (
            "graph",
            "edge_count_from_degrees",
            {"degrees": [0] * 201},
        ),
        (
            "graph",
            "degree_sequence_graphical",
            {"degrees": [0] * 201},
        ),
        (
            "numerical",
            "interpolate_eval",
            {"points": [[index, index * index] for index in range(13)], "evaluate_at": 14},
        ),
        (
            "sympy",
            "solve_system",
            {
                "equations": [f"x{index}={index}" for index in range(6)],
                "variables": [f"x{index}" for index in range(6)],
            },
        ),
    ],
)
def test_result_changing_sequences_are_rejected_not_truncated(
    family,
    operation,
    arguments,
):
    fact = SafeAugmentedToolExecutor(extended_tools=True).execute_one(
        ToolRequest(family, operation, arguments)
    )
    assert fact is None


def test_newton_step_and_interpolation_are_exact_and_grounded():
    executor = SafeAugmentedToolExecutor(extended_tools=True)

    newton = executor.execute_one(
        ToolRequest(
            "numerical",
            "newton_step",
            {"f": "x^2-2", "variable": "x", "x0": "3/2"},
        )
    )
    assert newton is not None
    assert "f=x^2-2; variable=x; x0=3/2" in newton.statement
    assert newton.statement.endswith("-> 17/12")

    interpolation = executor.execute_one(
        ToolRequest(
            "numerical",
            "interpolate_eval",
            {"points": [[0,1],[1,3],[2,7]], "evaluate_at": 3},
        )
    )
    assert interpolation is not None
    assert "points=[[0,1],[1,3],[2,7]]; evaluate_at=3" in interpolation.statement
    assert interpolation.statement.endswith("-> 13")


@pytest.mark.parametrize(
    ("operation", "arguments", "must_show"),
    [
        ("nullspace", {"matrix": [[1,2],[2,4]]}, "nullspace([[1,2],[2,4]])"),
        ("charpoly", {"matrix": [[2,0],[0,3]], "var": "lam"}, "matrix=[[2,0],[0,3]]; var=lam"),
        ("eigenvectors", {"matrix": [[2,0],[0,3]]}, "eigenvectors([[2,0],[0,3]])"),
        ("rref", {"matrix": [[1,2],[2,4]]}, "rref([[1,2],[2,4]])"),
    ],
)
def test_matrix_extensions_show_exact_matrix(operation, arguments, must_show):
    fact = SafeAugmentedToolExecutor(extended_tools=True).execute_one(
        ToolRequest("matrix", operation, arguments)
    )
    assert fact is not None
    assert must_show in fact.statement


def test_extended_normal_pipeline_remains_two_model_calls():
    client = SequenceClient(
        [
            "TARGET: gcd\nTOOL: number_theory|gcd|a=252|b=105",
            "Final answer: 21",
        ]
    )
    agent = ReasoningAgent(
        client,
        score_first_experiment_preset="tool_augmented_v31_extended",
    )
    result = agent.solve("Compute gcd(252,105).", {})

    assert result["final_response"] == "21"
    assert len(client.calls) == 2
    assert client.calls[0]["temperature"] == pytest.approx(0.2)
    assert client.calls[0]["top_p"] == pytest.approx(0.95)
    assert client.calls[0]["max_tokens"] == 1536
    assert client.calls[0]["thinking_mode"] is False
    assert client.calls[1]["temperature"] == pytest.approx(0.8)
    assert client.calls[1]["top_p"] == pytest.approx(0.95)
    assert client.calls[1]["max_tokens"] == 32768
    assert client.calls[1]["thinking_mode"] is True
    prompt = "\n".join(message["content"] for message in client.calls[1]["messages"])
    assert "gcd[a=252; b=105] -> 21" in prompt

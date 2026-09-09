from __future__ import annotations

from typing import Any, Dict, List

import pytest
import sympy as sp

import math_agent_core.tools.augmented_tool as augmented_tool
from math_agent_core.tools.matrix_tool import MatrixTool
from math_agent_core.tools.augmented_tool import (
    MAX_RESULT_CHARS,
    SafeAugmentedToolExecutor,
    ToolFact,
    ToolRequest,
    _display,
    build_evidence_block_with_metadata,
    run_answer_verification_in_subprocess,
)
from user_agent import ReasoningAgent
from math_agent_core.verifiers.linear_algebra import run_system_inferred_matrix_verification


class SequenceClient:
    def __init__(self, responses: List[str]) -> None:
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


def _trace(result: Dict[str, Any]) -> Dict[str, str]:
    return {item["step"]: item["content"] for item in result["trace"]}


def _legacy_v31_agent(client: Any) -> ReasoningAgent:
    """Keep V3.0.3 target-safety regressions on their rollback arm."""

    return ReasoningAgent(
        client,
        score_first_experiment_preset="tool_augmented_v31_extended",
    )


@pytest.mark.parametrize(
    ("problem", "answer", "expected_target"),
    [
        ("Given x=2, compute x^2.", "4", "none"),
        ("For p=0.3, compute p*(1-p).", "0.21", "none"),
        ("Compute residue of 1/(z*(z+1)) at z=0.", "1", "none"),
        ("If x^2=4, find x^4.", "16", "none"),
        (
            "Given the full-rank matrix A=[[1,0],[0,2]], compute its determinant.",
            "2",
            "matrix_determinant",
        ),
        (
            "The rank of A=[[1,0],[0,2]] is 2. Compute its determinant.",
            "2",
            "matrix_determinant",
        ),
        (
            "Let A=[[1,2],[3,4]] have determinant -2. Find its rank.",
            "2",
            "matrix_rank",
        ),
        (
            "Compute the determinant of the full-rank matrix A=[[1,0],[0,2]].",
            "2",
            "matrix_determinant",
        ),
        (
            "Find the rank of matrix A=[[1,2],[3,4]] with determinant -2.",
            "2",
            "matrix_rank",
        ),
    ],
)
def test_background_conditions_never_trigger_a_spurious_correction(
    problem: str,
    answer: str,
    expected_target: str,
):
    client = SequenceClient(
        [
            "TARGET: grounded request",
            f"Final answer: {answer}",
            "Final answer: 99",
        ]
    )
    result = _legacy_v31_agent(client).solve(problem, {})

    assert result["final_response"] == answer
    assert len(client.calls) == 2
    trace = _trace(result)
    assert trace["correction_call_used"] == "false"
    assert trace["correction_targets"] == expected_target
    if expected_target == "matrix_determinant":
        assert "matrix_rank" not in trace["correction_targets"]
    if expected_target == "matrix_rank":
        assert "matrix_determinant" not in trace["correction_targets"]


def test_no_high_confidence_target_does_not_start_a_verification_worker(monkeypatch):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("verification worker should not start")

    monkeypatch.setattr(augmented_tool, "run_answer_verification_in_subprocess", forbidden)
    client = SequenceClient(["TARGET: arithmetic with a given parameter", "Final answer: 4"])
    result = _legacy_v31_agent(client).solve(
        "Given x=2, compute x^2.",
        {},
    )

    assert result["final_response"] == "4"
    assert len(client.calls) == 2
    assert calls == []


@pytest.mark.parametrize(
    ("problem", "initial_answer", "corrected_answer", "target"),
    [
        ("Compute 1+1.", "3", "2", "pure_arithmetic"),
        ("Solve x^2-5*x+6=0.", "x=2 only", "x=2 or x=3", "equation_solution"),
        ("Compute det([[1,2],[3,4]]).", "-1", "-2", "matrix_determinant"),
    ],
)
def test_explicit_requested_targets_still_allow_one_correction(
    problem: str,
    initial_answer: str,
    corrected_answer: str,
    target: str,
):
    client = SequenceClient(
        [
            "TARGET: exact requested quantity",
            f"Final answer: {initial_answer}",
            f"Final answer: {corrected_answer}",
        ]
    )
    result = _legacy_v31_agent(client).solve(problem, {})

    assert result["final_response"] == corrected_answer
    assert len(client.calls) == 3
    trace = _trace(result)
    assert target in trace["correction_targets"]
    assert trace["correction_call_used"] == "true"
    assert trace["verification_failure_methods"] != "none"


def test_verification_subprocess_rejects_non_enum_targets_without_starting_worker(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("untrusted verification target started a worker")

    monkeypatch.setattr(augmented_tool.subprocess, "Popen", forbidden)
    evidence = run_answer_verification_in_subprocess(
        "3",
        "ANSWER_VALUE",
        requests=[{"kind": "model_generated_shell_check"}],
    )
    assert evidence == []


def test_active_matrix_target_uses_the_outer_process_deadline_not_legacy_thread_timeout(monkeypatch):
    def forbidden_legacy_timeout(*args, **kwargs):
        raise AssertionError("active V3 matrix verification used MatrixTool.run")

    monkeypatch.setattr(MatrixTool, "run", forbidden_legacy_timeout)
    evidence = run_system_inferred_matrix_verification(
        "Compute det([[1,2],[3,4]]).",
        "-1",
        allowed_targets=["matrix_determinant"],
    )

    assert evidence
    assert evidence[0].status == "fail"
    assert evidence[0].method == "matrix_determinant"


def test_incomplete_polynomial_roots_are_omitted_but_complete_roots_remain_usable():
    executor = SafeAugmentedToolExecutor()

    incomplete = executor.execute_one(
        ToolRequest("sympy", "roots", {"expr": "x^5-x+1", "var": "x"})
    )
    complete = executor.execute_one(
        ToolRequest("sympy", "roots", {"expr": "x^2-1", "var": "x"})
    )

    assert incomplete is None
    assert complete is not None
    assert "roots[x^2-1; var=x]" in complete.statement
    assert "-1: multiplicity 1" in complete.statement
    assert "1: multiplicity 1" in complete.statement


def test_unevaluated_symbolic_objects_cannot_be_injected_as_facts():
    x = sp.Symbol("x")
    unresolved_values = [
        sp.Integral(sp.sin(x**x), x),
        sp.Sum(sp.sin(x**x), (x, 1, sp.Symbol("n"))),
        sp.Limit(sp.sin(x), x, sp.oo),
        sp.Derivative(sp.Function("f")(x), x),
        sp.ConditionSet(x, sp.Eq(sp.sin(x), x), sp.S.Reals),
    ]
    for value in unresolved_values:
        with pytest.raises(ValueError):
            _display(value)

    facts = [
        ToolFact("sympy", "integrate", "integral[expr=f] -> Integral(f, x)"),
        ToolFact("sympy", "sum", "sum[expr=f] -> Sum(f, (x, 1, n))"),
        ToolFact("sympy", "solve", "solve[x=sin(x)] -> ConditionSet(x, True, Complexes)"),
    ]
    evidence = build_evidence_block_with_metadata(facts)
    assert evidence.included_count == 0
    assert evidence.text == "(no deterministic tool facts)"


def test_oversized_or_ellipsis_fact_is_skipped_without_hiding_later_fact():
    oversized = ToolFact("sympy", "expand", "x" * (MAX_RESULT_CHARS + 1))
    truncated = ToolFact("sympy", "expand", "expand[x] -> x + ...")
    short = ToolFact("matrix", "determinant", "det([[1,2],[3,4]]) -> -2")

    evidence = build_evidence_block_with_metadata([oversized, truncated, short])

    assert evidence.included_count == 1
    assert evidence.included_facts == (short,)
    assert short.statement in evidence.text
    assert "..." not in evidence.text


def test_bounded_sanitized_tool_telemetry_distinguishes_protocol_outcomes():
    client = SequenceClient(
        [
            "TARGET: exact gcd\nTOOL: number_theory|gcd|a=252|b=105",
            "Final answer: 21",
        ]
    )
    result = _legacy_v31_agent(client).solve(
        "Compute gcd(252,105).",
        {},
    )
    trace = _trace(result)

    for key in (
        "formalizer_plan_lines",
        "formalizer_protocol_usable",
        "tool_requested_ops",
        "tool_succeeded_ops",
        "tool_injected_ops",
        "tool_worker_status",
        "correction_targets",
        "verification_failure_methods",
    ):
        assert key in trace
    assert trace["formalizer_plan_lines"] == "1"
    assert trace["formalizer_protocol_usable"] == "true"
    assert trace["tool_requested_ops"] == "number_theory:gcd"
    assert trace["tool_succeeded_ops"] == "number_theory:gcd"
    assert trace["tool_injected_ops"] == "number_theory:gcd"
    assert trace["tool_worker_status"] == "completed"
    assert "252" not in trace["tool_requested_ops"]

    rejected = SequenceClient(["TOOL: shell|run|command=bad", "Final answer: 2"])
    rejected_result = _legacy_v31_agent(rejected).solve("Compute 1+1.", {})
    rejected_trace = _trace(rejected_result)
    assert rejected_trace["formalizer_used"] == "true"
    assert rejected_trace["formalizer_protocol_usable"] == "false"
    assert rejected_trace["tool_worker_status"] == "parse_rejected"

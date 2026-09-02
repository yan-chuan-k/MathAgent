import pytest

from math_agent_core import MathAgentOrchestrator
from math_agent_core.acceptance import AcceptancePolicy
from math_agent_core.clients import ScriptedClient
from math_agent_core.state import OverallStatus
from math_agent_core.verifiers.discrete_math import (
    _extract_final_integer,
    run_discrete_math_verification,
)


CORE = "7x ≡ 3 (mod 20)"
LEAST_REQUEST = "give the least nonnegative solution"
INFIX_RESIDUALS = [
    "and x > 10",
    "subject to x > 10",
    "with x >= 29",
    "assuming x != 9",
    "and x is even",
    "subject to x < 5",
    "with x = 29",
    "among integers greater than 20",
]


def _result():
    return {"problem_type": "discrete_math", "requested_checks": []}


def _evidence(problem, answer):
    return run_discrete_math_verification(problem, answer, _result())


def _decision(evidence):
    return AcceptancePolicy().decide(
        schema_valid=True,
        content_complete=True,
        task_type="calculation",
        answer_type="exact",
        model_verification_pass=False,
        evidence=evidence,
    )


def assert_no_decisive(problem, candidate):
    evidence = _evidence(problem, candidate)
    assert not any(item.is_decisive for item in evidence), evidence
    return evidence


def assert_unknown_residual_never_decisive(residual):
    variants = [
        f"{residual}; {CORE}; {LEAST_REQUEST}.",
        f"{CORE} {residual}; {LEAST_REQUEST}.",
        f"{CORE}; {LEAST_REQUEST}; {residual}.",
    ]
    for problem in variants:
        assert_no_decisive(problem, "9")


@pytest.mark.parametrize(
    "problem",
    [
        f"{CORE}, {LEAST_REQUEST}.",
        f"{CORE}; {LEAST_REQUEST}.",
        f"{CORE}. Give the least nonnegative solution.",
        f"{CORE} and give the least nonnegative solution.",
    ],
)
def test_literal_harmless_connectors_preserve_least_nonnegative_template(problem):
    item = next(item for item in _evidence(problem, "9") if item.method == "modular_check")
    assert item.status == "pass"
    assert item.claim_scope == "full_answer"
    assert item.is_decisive is True


@pytest.mark.parametrize("residual", INFIX_RESIDUALS)
def test_congruence_infix_residuals_never_preserve_full_answer_decisiveness(residual):
    assert_no_decisive(f"{CORE} {residual}; {LEAST_REQUEST}.", "9")


@pytest.mark.parametrize("residual", INFIX_RESIDUALS)
def test_congruence_residual_insertion_positions_are_all_nondecisive(residual):
    assert_unknown_residual_never_decisive(residual)


def test_constrained_least_residue_wrong_unconstrained_candidate_is_not_decisive_pass():
    evidence = assert_no_decisive(
        f"{CORE} and x > 10; {LEAST_REQUEST}.",
        "9",
    )
    assert not any(item.status == "fail" and item.is_decisive for item in evidence)
    decision = _decision(evidence)
    assert decision.overall_status != OverallStatus.SOLVED.value
    assert decision.answer_verified is False


def test_constrained_least_residue_actual_answer_is_not_decisive_fail():
    evidence = assert_no_decisive(
        f"{CORE} and x > 10; {LEAST_REQUEST}.",
        "29",
    )
    assert not any(item.status == "fail" and item.is_decisive for item in evidence)


def test_plain_least_nonnegative_positive_and_negative_controls_remain():
    good = next(
        item for item in _evidence(
            "Find the least nonnegative solution to 7x ≡ 3 (mod 20).",
            "9",
        )
        if item.method == "modular_check"
    )
    bad = next(
        item for item in _evidence(
            "Find the least nonnegative solution to 7x ≡ 3 (mod 20).",
            "29",
        )
        if item.method == "modular_check"
    )
    assert (good.status, good.is_decisive, good.claim_scope) == ("pass", True, "full_answer")
    assert (bad.status, bad.is_decisive, bad.claim_scope) == ("fail", True, "full_answer")


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("Final answer: 210.", 210),
        ("We obtain C(10,4)=210. Final answer: 210.", 210),
        ("After checking the cases, final result = 210.", 210),
    ],
)
def test_terminal_final_marker_is_extractable(answer, expected):
    assert _extract_final_integer(answer) == expected


@pytest.mark.parametrize(
    "answer",
    [
        "Final answer: 210, but this is incorrect; correct answer is 209.",
        "Final answer: 210, but actually the correct answer is 209.",
        "Final answer: 210, however 209 is correct.",
        "Final answer: 210 — this is wrong; 209 is correct.",
        "Assuming a mistaken count, final answer: 210; correction: 209.",
    ],
)
def test_nonterminal_final_markers_do_not_extract_superseded_value(answer):
    assert _extract_final_integer(answer) is None


def _candidate(answer):
    return {
        "problem_id": "v112",
        "problem_type": "discrete_math",
        "task_type": "calculation",
        "domain_candidates": ["discrete_math"],
        "reasoning_plan": ["Return the supplied candidate."],
        "solution": [{"step": 1, "content": str(answer)}],
        "final_answer": {"answer": str(answer), "answer_type": "numeric"},
        "verification": {
            "verification_result": "uncertain",
            "checks": [],
            "confidence": 0.5,
        },
        "assumptions": [],
        "learning_hints": [],
    }


def _orchestrate(problem, answer):
    client = ScriptedClient([_candidate(answer)])
    orchestrator = MathAgentOrchestrator(
        client=client,
        max_retries=0,
        max_candidates=1,
        enable_repair=False,
        enable_tool_verify=True,
        enable_critic=False,
        enable_finalizer=False,
    )
    result = orchestrator.solve(problem, {"idx": "v112", "subject": "discrete_math"})
    return result, orchestrator


def test_full_orchestrator_congruence_infix_attack_candidate_9_not_solved():
    result, _ = _orchestrate(
        f"{CORE} and x > 10; {LEAST_REQUEST}.",
        "9",
    )
    assert result["_meta"]["overall_status"] != "solved"
    assert result["_meta"]["answer_verified"] is False


def test_full_orchestrator_congruence_infix_candidate_29_not_invalidated_by_wrong_template():
    result, _ = _orchestrate(
        f"{CORE} and x > 10; {LEAST_REQUEST}.",
        "29",
    )
    evidence = result["verification"]["evidence"]
    assert not any(item.get("is_decisive") and item.get("status") == "fail" for item in evidence)
    assert result["_meta"]["answer_verified"] is False


def test_full_orchestrator_nonterminal_final_marker_attack_not_solved():
    result, _ = _orchestrate(
        "Choose 4 objects from 10.",
        "Final answer: 210, but this is incorrect; correct answer is 209.",
    )
    assert result["_meta"]["overall_status"] != "solved"
    assert result["_meta"]["answer_verified"] is False

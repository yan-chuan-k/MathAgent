import pytest

from math_agent_core import MathAgentOrchestrator
from math_agent_core.clients import ScriptedClient
from math_agent_core.verifiers.discrete_math import (
    _extract_final_integer,
    run_discrete_math_verification,
)


POLARITY_ATTACKS = [
    "This is not the final answer: 210.",
    "This is an incorrect final answer: 210.",
    "Do not use final answer: 210.",
    "A wrong final answer is 210.",
    "It is false that the final answer is 210.",
    "Not final answer: 210.",
    "It is false that actually 210.",
    "We should not say actually 210.",
    "It is false that instead the answer is 210.",
    "Do not conclude so 210 ways.",
]


def _result():
    return {"problem_type": "discrete_math", "requested_checks": []}


def _evidence(problem, answer):
    return run_discrete_math_verification(problem, answer, _result())


def _candidate(answer):
    return {
        "problem_id": "v113",
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
    orchestrator = MathAgentOrchestrator(
        client=ScriptedClient([_candidate(answer)]),
        max_retries=0,
        max_candidates=1,
        enable_repair=False,
        enable_tool_verify=True,
        enable_critic=False,
        enable_finalizer=False,
    )
    result = orchestrator.solve(
        problem,
        {"idx": "v113", "subject": "discrete_math"},
    )
    return result, orchestrator


@pytest.mark.parametrize("answer", POLARITY_ATTACKS)
def test_assertion_polarity_attacks_are_not_extractable(answer):
    assert _extract_final_integer(answer) is None


@pytest.mark.parametrize("answer", POLARITY_ATTACKS)
def test_assertion_polarity_attacks_never_create_decisive_pass(answer):
    evidence = _evidence("Choose 4 objects from 10.", answer)
    assert not any(item.is_decisive and item.status == "pass" for item in evidence), evidence


def test_assertion_polarity_attack_is_not_solved_by_full_orchestrator():
    result, _ = _orchestrate(
        "Choose 4 objects from 10.",
        "This is not the final answer: 210.",
    )
    assert result["_meta"]["overall_status"] != "solved"
    assert result["_meta"]["answer_verified"] is False


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("210", 210),
        ("Answer: 210", 210),
        ("The answer is 210.", 210),
        ("Final answer: 210.", 210),
        ("Final result = 210.", 210),
        ("We obtain C(10,4)=210. Final answer: 210.", 210),
        ("After checking the cases, final result = 210.", 210),
        ("There are 210 ways before excluding one invalid case, so 209 remain.", 209),
        ("The answer is 210 is incorrect; actually 209.", 209),
    ],
)
def test_supported_affirmative_assertion_surfaces_remain_extractable(answer, expected):
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
def test_nonterminal_final_marker_corrections_remain_unextractable(answer):
    assert _extract_final_integer(answer) is None


POWER_CASES = [
    ("Compute 2^3 mod 5.", 2, 3, 5),
    ("Calculate 3^8 modulo 7.", 3, 8, 7),
    ("Evaluate 5^7 mod 11.", 5, 7, 11),
    ("What is 2^10 mod 13?", 2, 10, 13),
    ("计算 2^100 模 7 的余数。", 2, 100, 7),
]


@pytest.mark.parametrize(
    ("problem", "base", "exponent", "modulus"),
    POWER_CASES,
)
def test_modular_power_requires_canonical_remainder(problem, base, exponent, modulus):
    remainder = pow(base, exponent, modulus)

    canonical = _evidence(problem, str(remainder))
    item = next(item for item in canonical if item.method == "modular_check")
    assert item.status == "pass"
    assert item.claim_scope == "full_answer"
    assert item.is_decisive is True

    for candidate in (
        remainder + modulus,
        remainder + 2 * modulus,
        remainder - modulus,
    ):
        evidence = _evidence(problem, str(candidate))
        item = next(item for item in evidence if item.method == "modular_check")
        assert item.status == "fail", (problem, candidate, evidence)
        assert item.claim_scope == "full_answer"
        assert item.is_decisive is True


@pytest.mark.parametrize("candidate", ["3", "8", "-2"])
def test_compute_2_power_3_mod_5_fixed_controls(candidate):
    item = next(
        item
        for item in _evidence("Compute 2^3 mod 5.", candidate)
        if item.method == "modular_check"
    )
    expected_status = "pass" if candidate == "3" else "fail"
    assert item.status == expected_status
    assert item.is_decisive is True


@pytest.mark.parametrize("candidate", ["2", "9", "-5"])
def test_chinese_modular_remainder_fixed_controls(candidate):
    item = next(
        item
        for item in _evidence("计算 2^100 模 7 的余数。", candidate)
        if item.method == "modular_check"
    )
    expected_status = "pass" if candidate == "2" else "fail"
    assert item.status == expected_status
    assert item.is_decisive is True


def test_candidate_requested_modular_power_check_keeps_legacy_congruence_semantics_and_is_auxiliary():
    evidence = run_discrete_math_verification(
        "An unsupported discrete problem.",
        "8",
        {
            "problem_type": "discrete_math",
            "requested_checks": [
                {
                    "tool": "modular_check",
                    "arguments": {
                        "base": 2,
                        "exponent": 3,
                        "modulus": 5,
                        "expected": 8,
                    },
                }
            ],
        },
    )
    item = next(item for item in evidence if item.claim_id.startswith("requested_discrete_"))
    assert item.status == "pass"
    assert item.is_decisive is False
    assert item.claim_scope == "subclaim"


def test_linear_congruence_residue_class_still_accepts_noncanonical_representative():
    item = next(
        item
        for item in _evidence(
            "Solve 7x ≡ 3 (mod 20) for x modulo 20.",
            "x ≡ 29 (mod 20)",
        )
        if item.method == "modular_check"
    )
    assert item.status == "pass"
    assert item.claim_scope == "full_answer"
    assert item.is_decisive is True


@pytest.mark.parametrize(
    ("problem", "answer"),
    [
        ("Choose 4 objects from 10.", "210"),
        ("How many binary strings of length 8 with exactly 3 ones?", "56"),
        ("How many nonnegative integer triples satisfy x+y+z=5?", "21"),
        ("A tree with 12 vertices has how many edges?", "11"),
        ("How many edges does K_8 have?", "28"),
        ("How many edges does K_{3,5} have?", "15"),
        ("Find the least nonnegative solution to 7x ≡ 3 (mod 20).", "9"),
        ("Compute 2^3 mod 5.", "3"),
    ],
)
def test_frozen_positive_controls_remain_decisive(problem, answer):
    evidence = _evidence(problem, answer)
    assert any(item.is_decisive and item.status == "pass" for item in evidence), evidence


def test_full_orchestrator_modular_power_noncanonical_representative_is_not_solved():
    result, _ = _orchestrate("Compute 2^3 mod 5.", "8")
    assert result["_meta"]["overall_status"] != "solved"
    assert result["_meta"]["answer_verified"] is False
    assert any(
        item.get("method") == "modular_check"
        and item.get("is_decisive")
        and item.get("status") == "fail"
        for item in result["verification"]["evidence"]
    )


def test_full_orchestrator_modular_power_canonical_remainder_is_solved():
    result, _ = _orchestrate("Compute 2^3 mod 5.", "3")
    assert result["_meta"]["overall_status"] == "solved"
    assert result["_meta"]["answer_verified"] is True
    assert any(
        item.get("method") == "modular_check"
        and item.get("is_decisive")
        and item.get("status") == "pass"
        for item in result["verification"]["evidence"]
    )

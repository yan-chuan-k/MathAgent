from math_agent_core import MathAgentOrchestrator
from math_agent_core.clients import ScriptedClient
from math_agent_core.tools.sympy_tool import run_sympy_verification


AUXILIARY_CHECK_NOTICE = "Candidate-proposed auxiliary check; not sufficient to verify final_answer."


def _wrong_candidate_with_unrelated_requested_check():
    return {
        "problem_id": "sympy-regression",
        "problem_type": "unknown",
        "task_type": "calculation",
        "domain_candidates": ["unknown"],
        "reasoning_plan": ["Compute directly."],
        "solution": [{"step": 1, "content": "The candidate claims the answer is 999."}],
        "final_answer": {"answer": "999", "answer_type": "numeric"},
        "verification": {
            "verification_result": "pass",
            "checks": ["Candidate self-check."],
            "confidence": 0.99,
        },
        "requested_checks": [
            {
                "tool": "numeric_arithmetic",
                "arguments": {"expression": "1+1", "expected": "2"},
            }
        ],
        "assumptions": [],
        "learning_hints": [],
    }


def test_candidate_requested_sympy_check_is_auxiliary():
    evidence = run_sympy_verification(
        problem_text="1 + 1 = ?",
        answer="999",
        result=_wrong_candidate_with_unrelated_requested_check(),
    )

    inferred = next(item for item in evidence if item.claim_id == "check_1")
    requested = next(item for item in evidence if item.claim_id == "requested_check_1")

    assert inferred.status == "fail"
    assert inferred.is_decisive is True
    assert requested.status == "pass"
    assert requested.is_decisive is False
    assert AUXILIARY_CHECK_NOTICE in requested.details


def test_unrelated_requested_sympy_pass_cannot_solve_wrong_final_answer():
    client = ScriptedClient([_wrong_candidate_with_unrelated_requested_check()])
    orchestrator = MathAgentOrchestrator(
        client=client,
        max_retries=0,
        enable_repair=False,
        enable_tool_verify=True,
        enable_critic=False,
        enable_finalizer=False,
        max_candidates=1,
    )

    result = orchestrator.solve("1 + 1 = ?", {"idx": "sympy-regression"})

    requested = next(
        item for item in result["verification"]["evidence"] if item["claim_id"] == "requested_check_1"
    )
    assert requested["status"] == "pass"
    assert requested["is_decisive"] is False
    assert AUXILIARY_CHECK_NOTICE in requested["details"]
    assert result["_meta"]["answer_verified"] is False
    assert result["_meta"]["overall_status"] != "solved"

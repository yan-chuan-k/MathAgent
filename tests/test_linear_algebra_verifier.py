from math_agent_core import MathAgentOrchestrator
from math_agent_core.clients import ScriptedClient
from math_agent_core.tools.matrix_tool import MatrixTool
from math_agent_core.verifiers.linear_algebra import run_linear_algebra_verification


def _matrix_result(answer="det(A)= -2", checks=None):
    return {
        "problem_id": "lin",
        "problem_type": "linear_algebra",
        "task_type": "calculation",
        "domain_candidates": ["linear_algebra"],
        "reasoning_plan": ["Use matrix computation."],
        "solution": [{"step": 1, "content": "Compute directly."}],
        "final_answer": {"answer": answer, "answer_type": "numeric"},
        "verification": {"verification_result": "pass", "checks": ["matrix check"], "confidence": 0.9},
        "requested_checks": checks or [],
        "assumptions": [],
        "learning_hints": [],
    }


def test_matrix_tool_checks_determinant_exactly():
    evidence = MatrixTool().run(
        {
            "tool": "matrix_determinant",
            "arguments": {"matrix": [[1, 2], [3, 4]], "expected": "-2"},
            "claim_id": "det",
        }
    )

    assert evidence.status == "pass"
    assert evidence.verification_level == "exact_symbolic"
    assert evidence.is_decisive is True


def test_matrix_tool_rejects_wrong_eigenpair():
    evidence = MatrixTool().run(
        {
            "tool": "eigenpair_residual",
            "arguments": {"matrix": [[2, 0], [0, 3]], "vector": [1, 0], "eigenvalue": "3"},
            "claim_id": "eig",
        }
    )

    assert evidence.status == "fail"
    assert evidence.is_decisive is True


def test_linear_algebra_verifier_uses_requested_checks():
    result = _matrix_result(
        checks=[
            {
                "tool": "matrix_inverse",
                "arguments": {"matrix": [[1, 0], [0, 1]], "inverse": [[1, 0], [0, 1]]},
            }
        ]
    )

    evidence = run_linear_algebra_verification(result)

    assert evidence[0].status == "pass"
    assert evidence[0].method == "matrix_inverse"


def test_candidate_matrix_checks_are_auxiliary_only():
    result = _matrix_result(
        answer="999",
        checks=[
            {
                "tool": "matrix_determinant",
                "arguments": {"matrix": [[1, 2], [3, 4]], "expected": "-2"},
            },
            {
                "tool": "matrix_determinant",
                "arguments": {"matrix": [[1, 2], [3, 4]], "expected": "999"},
            },
            {
                "tool": "matrix_determinant",
                "arguments": {"matrix": [], "expected": "0"},
            },
        ],
    )

    evidence = run_linear_algebra_verification(result)

    assert [item.status for item in evidence] == ["pass", "fail", "inconclusive"]
    assert all(item.is_decisive is False for item in evidence)
    assert all(
        "Candidate-proposed matrix check; treated as supporting evidence only." in item.details for item in evidence
    )


def test_wrong_answer_with_unrelated_passing_matrix_check_is_not_verified():
    client = ScriptedClient(
        {
            "solver": [
                _matrix_result(
                    answer="999",
                    checks=[
                        {
                            "tool": "matrix_determinant",
                            "arguments": {"matrix": [[1, 2], [3, 4]], "expected": "-2"},
                        }
                    ],
                )
            ]
        }
    )
    orchestrator = MathAgentOrchestrator(client=client, max_retries=0, enable_tool_verify=True, enable_critic=False)

    result = orchestrator.solve("What is the dimension of R^2?", {"idx": 6})

    assert result["_meta"]["answer_verified"] is False
    assert result["_meta"]["overall_status"] != "solved"
    assert result["verification"]["verification_result"] != "pass"
    matrix_evidence = [item for item in result["verification"]["evidence"] if item["verifier"] == "matrix_tool"]
    assert matrix_evidence
    assert all(item["is_decisive"] is False for item in matrix_evidence)


def test_orchestrator_treats_candidate_matrix_determinant_as_auxiliary_evidence():
    client = ScriptedClient(
        {
            "solver": [
                _matrix_result(
                    checks=[
                        {
                            "tool": "matrix_determinant",
                            "arguments": {"matrix": [[1, 2], [3, 4]], "expected": "-2"},
                        }
                    ]
                )
            ],
            "critic": [{"status": "pass", "failure_kind": "", "first_error": "", "missing_targets": [], "suggested_repair": ""}],
        }
    )
    orchestrator = MathAgentOrchestrator(client=client, max_retries=0, max_candidates=1, enable_tool_verify=True)

    result = orchestrator.solve("Compute det([[1,2],[3,4]]).", {"idx": 0})

    assert result["_meta"]["overall_status"] != "solved"
    assert result["_meta"]["answer_verified"] is False
    matrix_evidence = [item for item in result["verification"]["evidence"] if item["verifier"] == "matrix_tool"]
    assert matrix_evidence
    assert all(item["is_decisive"] is False for item in matrix_evidence)

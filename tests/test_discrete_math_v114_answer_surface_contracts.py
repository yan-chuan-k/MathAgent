import pytest

from math_agent_core import MathAgentOrchestrator
from math_agent_core.clients import ScriptedClient
from math_agent_core.verifiers.discrete_math import (
    _extract_final_integer,
    _extract_final_integer_assertion,
    run_discrete_math_verification,
)


def _result():
    return {"problem_type": "discrete_math", "requested_checks": []}


def _evidence(problem, answer):
    return run_discrete_math_verification(problem, answer, _result())


def _candidate(answer, answer_type="numeric"):
    return {
        "problem_id": "v114",
        "problem_type": "discrete_math",
        "task_type": "calculation",
        "domain_candidates": ["discrete_math"],
        "reasoning_plan": ["Return the supplied candidate."],
        "solution": [{"step": 1, "content": str(answer)}],
        "final_answer": {"answer": str(answer), "answer_type": answer_type},
        "verification": {
            "verification_result": "uncertain",
            "checks": [],
            "confidence": 0.5,
        },
        "assumptions": [],
        "learning_hints": [],
    }


def _orchestrate(problem, answer, answer_type="numeric"):
    orchestrator = MathAgentOrchestrator(
        client=ScriptedClient([_candidate(answer, answer_type)]),
        max_retries=0,
        max_candidates=1,
        enable_repair=False,
        enable_tool_verify=True,
        enable_critic=False,
        enable_finalizer=False,
    )
    result = orchestrator.solve(
        problem,
        {"idx": "v114", "subject": "discrete_math"},
    )
    return result, orchestrator


@pytest.mark.parametrize(
    ("answer", "value", "kind", "metadata"),
    [
        ("210", 210, "neutral_numeric", {}),
        ("The answer is 210.", 210, "neutral_numeric", {}),
        ("Final answer: 210.", 210, "neutral_numeric", {}),
        (r"\boxed{210}", 210, "neutral_numeric", {}),
        ("There are exactly 210 ways.", 210, "count_statement", {"noun": "ways"}),
        ("x = 9", 9, "assignment", {"lhs": "x"}),
        ("x ≡ 9 (mod 20)", 9, "congruence", {"lhs": "x", "modulus": 20}),
        (
            "A tree with 12 vertices has 11 edges.",
            11,
            "tree_edge_statement",
            {"vertices": 12},
        ),
    ],
)
def test_structured_final_integer_assertions_preserve_semantics(answer, value, kind, metadata):
    assertion = _extract_final_integer_assertion(answer)
    assert assertion is not None
    assert assertion.value == value
    assert assertion.kind == kind
    assert assertion.metadata == metadata
    assert _extract_final_integer(answer) == value


MATRIX = [
    # combinatorial_counting
    ("count-neutral", "Choose 4 objects from 10.", "210", True),
    ("count-generic", "Choose 4 objects from 10.", "The answer is 210.", True),
    ("count-count", "Choose 4 objects from 10.", "There are exactly 210 ways.", True),
    ("count-assignment", "Choose 4 objects from 10.", "x = 210", False),
    ("count-congruence", "Choose 4 objects from 10.", "x ≡ 210 (mod 20)", False),
    ("count-graph", "Choose 4 objects from 10.", "A tree with 211 vertices has 210 edges.", False),
    # graph_theory
    ("graph-neutral", "A tree with 12 vertices has how many edges?", "11", True),
    ("graph-generic", "A tree with 12 vertices has how many edges?", "The answer is 11.", True),
    ("graph-count", "A tree with 12 vertices has how many edges?", "There are exactly 11 ways.", False),
    ("graph-assignment", "A tree with 12 vertices has how many edges?", "x = 11", False),
    ("graph-congruence", "A tree with 12 vertices has how many edges?", "x ≡ 11 (mod 7)", False),
    (
        "graph-graph",
        "A tree with 12 vertices has how many edges?",
        "A tree with 12 vertices has 11 edges.",
        True,
    ),
    # modular_power
    ("power-neutral", "Compute 2^3 mod 5.", "3", True),
    ("power-generic", "Compute 2^3 mod 5.", "The answer is 3.", True),
    ("power-count", "Compute 2^3 mod 5.", "There are exactly 3 ways.", False),
    ("power-assignment", "Compute 2^3 mod 5.", "x = 3", False),
    ("power-congruence", "Compute 2^3 mod 5.", "x ≡ 3 (mod 5)", False),
    ("power-graph", "Compute 2^3 mod 5.", "A tree with 4 vertices has 3 edges.", False),
    # linear_congruence
    (
        "congruence-neutral",
        "Solve 7x ≡ 3 (mod 20) for x modulo 20.",
        "9",
        True,
    ),
    (
        "congruence-generic",
        "Solve 7x ≡ 3 (mod 20) for x modulo 20.",
        "The answer is 9.",
        True,
    ),
    (
        "congruence-count",
        "Solve 7x ≡ 3 (mod 20) for x modulo 20.",
        "There are exactly 9 ways.",
        False,
    ),
    (
        "congruence-assignment",
        "Solve 7x ≡ 3 (mod 20) for x modulo 20.",
        "x = 9",
        True,
    ),
    (
        "congruence-congruence",
        "Solve 7x ≡ 3 (mod 20) for x modulo 20.",
        "x ≡ 9 (mod 20)",
        True,
    ),
    (
        "congruence-graph",
        "Solve 7x ≡ 3 (mod 20) for x modulo 20.",
        "A tree with 10 vertices has 9 edges.",
        False,
    ),
]


@pytest.mark.parametrize(
    ("case_id", "problem", "candidate", "permitted"),
    MATRIX,
    ids=[row[0] for row in MATRIX],
)
def test_answer_surface_compatibility_matrix_verifier_and_orchestrator(
    case_id, problem, candidate, permitted
):
    evidence = _evidence(problem, candidate)
    decisive_pass = any(item.is_decisive and item.status == "pass" for item in evidence)
    assert decisive_pass is permitted, (case_id, evidence)

    result, _ = _orchestrate(problem, candidate)
    assert (result["_meta"]["overall_status"] == "solved") is permitted, (case_id, result)
    assert result["_meta"]["answer_verified"] is permitted, (case_id, result)


CROSS_SURFACE_ATTACKS = [
    ("Choose 4 objects from 10.", "x ≡ 210 (mod 20)"),
    ("Choose 4 objects from 10.", "A tree with 99 vertices has 210 edges."),
    ("A tree with 12 vertices has how many edges?", "x ≡ 11 (mod 7)"),
    ("Compute 2^3 mod 5.", "x ≡ 3 (mod 7)"),
    (
        "Solve 7x ≡ 3 (mod 20) for x modulo 20.",
        "x ≡ 9 (mod 7)",
    ),
    (
        "Solve 7x ≡ 3 (mod 20) for x modulo 20.",
        "There are exactly 9 ways.",
    ),
]


@pytest.mark.parametrize(("problem", "candidate"), CROSS_SURFACE_ATTACKS)
def test_six_explicit_cross_surface_attacks_never_full_answer_verify(problem, candidate):
    evidence = _evidence(problem, candidate)
    assert not any(item.is_decisive and item.status == "pass" for item in evidence), evidence

    result, _ = _orchestrate(problem, candidate)
    assert result["_meta"]["overall_status"] != "solved"
    assert result["_meta"]["answer_verified"] is False


@pytest.mark.parametrize(
    "answer_type",
    ["numeric", "expression", "set", "text", "unknown"],
)
def test_answer_type_never_substitutes_for_assertion_semantics(answer_type):
    problem = "Solve 7x ≡ 3 (mod 20) for x modulo 20."
    candidate = "x ≡ 9 (mod 7)"
    result, _ = _orchestrate(problem, candidate, answer_type=answer_type)
    assert result["_meta"]["overall_status"] != "solved"
    assert result["_meta"]["answer_verified"] is False


def test_tree_edge_statement_requires_matching_problem_object_metadata():
    assert any(
        item.is_decisive and item.status == "pass"
        for item in _evidence(
            "A tree with 12 vertices has how many edges?",
            "A tree with 12 vertices has 11 edges.",
        )
    )
    mismatch = _evidence(
        "A tree with 12 vertices has how many edges?",
        "A tree with 99 vertices has 11 edges.",
    )
    assert not any(item.is_decisive for item in mismatch), mismatch


@pytest.mark.parametrize(
    "candidate",
    ["x ≡ 9 (mod 20)", "x ≡ 29 (mod 20)"],
)
def test_correct_modulus_congruence_positive_controls(candidate):
    evidence = _evidence(
        "Solve 7x ≡ 3 (mod 20) for x modulo 20.",
        candidate,
    )
    item = next(item for item in evidence if item.method == "modular_check")
    assert item.status == "pass"
    assert item.is_decisive is True
    assert item.claim_scope == "full_answer"


def test_wrong_modulus_congruence_never_full_answer_verifies_modulo_20():
    evidence = _evidence(
        "Solve 7x ≡ 3 (mod 20) for x modulo 20.",
        "x ≡ 9 (mod 7)",
    )
    assert not any(item.is_decisive and item.status == "pass" for item in evidence), evidence


def test_modular_power_congruence_surface_is_not_canonical_remainder_proof_even_same_modulus():
    evidence = _evidence("Compute 2^3 mod 5.", "x ≡ 3 (mod 5)")
    assert not any(item.is_decisive for item in evidence), evidence


def test_least_nonnegative_range_validation_remains_decisive_for_compatible_congruence():
    evidence = _evidence(
        "Find the least nonnegative solution to 7x ≡ 3 (mod 20).",
        "x ≡ 29 (mod 20)",
    )
    item = next(item for item in evidence if item.method == "modular_check")
    assert item.status == "fail"
    assert item.is_decisive is True
    assert item.claim_scope == "full_answer"


def test_all_integer_solution_completeness_policy_remains_nondecisive():
    evidence = _evidence(
        "Find all integers x satisfying 7x ≡ 3 (mod 20).",
        "x ≡ 9 (mod 20)",
    )
    item = next(item for item in evidence if item.method == "modular_check")
    assert item.status == "pass"
    assert item.is_decisive is False
    assert item.claim_scope == "subclaim"

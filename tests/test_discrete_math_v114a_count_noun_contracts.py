import pytest

from math_agent_core import MathAgentOrchestrator
from math_agent_core.clients import ScriptedClient
from math_agent_core.verifiers.discrete_math import (
    _extract_final_integer_assertion,
    run_discrete_math_verification,
)


def _result():
    return {"problem_type": "discrete_math", "requested_checks": []}


def _evidence(problem, answer):
    return run_discrete_math_verification(problem, answer, _result())


def _candidate(answer):
    return {
        "problem_id": "v114a",
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
    return orchestrator.solve(
        problem,
        {"idx": "v114a", "subject": "discrete_math"},
    )


ENUMERATION_CASES = {
    "subsets": (
        "Choose 4 objects from 10.",
        210,
        {"way", "ways", "subset", "subsets", "selection", "selections"},
    ),
    "permutations": (
        "How many ordered selections of 3 from 7 without repetition?",
        210,
        {"way", "ways", "permutation", "permutations", "selection", "selections"},
    ),
    "binary_strings": (
        "How many binary strings of length 8 contain exactly 3 ones?",
        56,
        {"way", "ways", "string", "strings"},
    ),
    "integer_tuples": (
        "How many nonnegative integer triples satisfy x+y+z=5?",
        21,
        {"way", "ways", "solution", "solutions"},
    ),
}

COUNT_NOUNS = (
    "way",
    "ways",
    "string",
    "strings",
    "solution",
    "solutions",
    "subset",
    "subsets",
    "permutation",
    "permutations",
    "selection",
    "selections",
)

COUNT_NOUN_MATRIX = [
    (enumeration_kind, problem, value, noun, noun in allowed)
    for enumeration_kind, (problem, value, allowed) in ENUMERATION_CASES.items()
    for noun in COUNT_NOUNS
]


@pytest.mark.parametrize(
    ("enumeration_kind", "problem", "value", "noun", "allowed"),
    COUNT_NOUN_MATRIX,
    ids=[
        f"{enumeration_kind}-{noun}-{'allow' if allowed else 'block'}"
        for enumeration_kind, _, _, noun, allowed in COUNT_NOUN_MATRIX
    ],
)
def test_count_noun_compatibility_matrix_verifier_and_orchestrator(
    enumeration_kind, problem, value, noun, allowed
):
    candidate = f"There are exactly {value} {noun}."
    assertion = _extract_final_integer_assertion(candidate)
    assert assertion is not None
    assert assertion.kind == "count_statement"
    assert assertion.metadata["noun"] == noun

    evidence = _evidence(problem, candidate)
    decisive = [item for item in evidence if item.is_decisive]
    decisive_pass = any(item.status == "pass" for item in decisive)

    if allowed:
        assert decisive_pass, (enumeration_kind, noun, evidence)
    else:
        # A recognized but incompatible counted-object noun is an answer-contract
        # mismatch, not positive or negative numerical evidence.
        assert decisive == [], (enumeration_kind, noun, evidence)

    # The count-noun matrix is a system-inferred verifier contract. Full
    # orchestrator coverage is locked separately for the seven direct attacks;
    # integer-tuple positive completeness remains governed by the frozen generic
    # completeness policy and is intentionally not redesigned here.


CROSS_NOUN_ATTACKS = [
    ("Choose 4 objects from 10.", "There are exactly 210 strings."),
    ("Choose 4 objects from 10.", "There are exactly 210 permutations."),
    (
        "How many ordered selections of 3 from 7 without repetition?",
        "There are exactly 210 strings.",
    ),
    (
        "How many ordered selections of 3 from 7 without repetition?",
        "There are exactly 210 subsets.",
    ),
    (
        "How many binary strings of length 8 contain exactly 3 ones?",
        "There are exactly 56 subsets.",
    ),
    (
        "How many binary strings of length 8 contain exactly 3 ones?",
        "There are exactly 56 permutations.",
    ),
    (
        "How many binary strings of length 8 contain exactly 3 ones?",
        "There are exactly 56 selections.",
    ),
]


@pytest.mark.parametrize(("problem", "candidate"), CROSS_NOUN_ATTACKS)
def test_seven_explicit_cross_noun_attacks_have_no_decisive_evidence_and_never_solve(
    problem, candidate
):
    evidence = _evidence(problem, candidate)
    assert not any(item.is_decisive for item in evidence), evidence

    result = _orchestrate(problem, candidate)
    assert result["_meta"]["overall_status"] != "solved"
    assert result["_meta"]["answer_verified"] is False


@pytest.mark.parametrize(
    ("problem", "candidate"),
    [
        ("Choose 4 objects from 10.", "210"),
        ("Choose 4 objects from 10.", "The answer is 210."),
        ("Choose 4 objects from 10.", "There are exactly 210 ways."),
        ("Choose 4 objects from 10.", "There are exactly 210 subsets."),
        (
            "How many ordered selections of 3 from 7 without repetition?",
            "There are exactly 210 permutations.",
        ),
        (
            "How many binary strings of length 8 contain exactly 3 ones?",
            "There are exactly 56 strings.",
        ),
    ],
)
def test_v114a_positive_count_surfaces_remain_decisive(problem, candidate):
    evidence = _evidence(problem, candidate)
    assert any(item.is_decisive and item.status == "pass" for item in evidence), evidence

    result = _orchestrate(problem, candidate)
    assert result["_meta"]["overall_status"] == "solved"
    assert result["_meta"]["answer_verified"] is True


def test_integer_tuple_solution_count_surface_is_decisive_at_verifier_level():
    evidence = _evidence(
        "How many nonnegative integer triples satisfy x+y+z=5?",
        "There are exactly 21 solutions.",
    )
    assert any(item.is_decisive and item.status == "pass" for item in evidence), evidence


def test_incompatible_count_noun_matrix_has_zero_decisive_evidence():
    incompatible = [row for row in COUNT_NOUN_MATRIX if not row[-1]]
    decisive_evidence_count = 0
    false_decisive_pass_count = 0

    for enumeration_kind, problem, value, noun, _ in incompatible:
        evidence = _evidence(problem, f"There are exactly {value} {noun}.")
        decisive_evidence_count += sum(1 for item in evidence if item.is_decisive)
        false_decisive_pass_count += sum(
            1 for item in evidence if item.is_decisive and item.status == "pass"
        )

    assert decisive_evidence_count == 0
    assert false_decisive_pass_count == 0

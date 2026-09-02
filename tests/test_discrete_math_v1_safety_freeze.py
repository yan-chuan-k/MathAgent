import json
from pathlib import Path

import pytest

from math_agent_core.acceptance import AcceptancePolicy
from math_agent_core.router import classify_problem
from math_agent_core.state import OverallStatus
from math_agent_core.verifiers.discrete_math import (
    _extract_final_integer,
    run_discrete_math_verification,
)


ROOT = Path(__file__).resolve().parents[1]
ROUTING_SURFACES = ROOT / "sample_data" / "discrete_math_v1_routing_surfaces.jsonl"


def _result():
    return {"problem_type": "discrete_math", "requested_checks": []}


def _evidence(problem, answer):
    return run_discrete_math_verification(problem, answer, _result())


def assert_no_false_decisive_pass(problem, candidate):
    evidence = _evidence(problem, candidate)
    assert not any(item.is_decisive and item.status == "pass" for item in evidence), evidence
    decision = AcceptancePolicy().decide(
        schema_valid=True,
        content_complete=True,
        task_type="calculation",
        answer_type="exact",
        model_verification_pass=False,
        evidence=evidence,
    )
    assert decision.overall_status != OverallStatus.SOLVED.value
    assert decision.answer_verified is False


@pytest.mark.parametrize(
    ("supported", "correct", "wrong", "constrained"),
    [
        (
            "Choose 4 objects from 10.",
            "210",
            "209",
            "Choose 4 objects from 10 with no two chosen objects adjacent.",
        ),
        (
            "How many ordered selections of 3 from 7 without repetition?",
            "210",
            "209",
            "How many ordered selections of 3 from 7 without repetition if the first selected object must be 1?",
        ),
        (
            "How many binary strings of length 8 with exactly 3 ones?",
            "56",
            "55",
            "How many binary strings of length 8 with exactly 3 ones and starting with 1?",
        ),
        (
            "How many nonnegative integer triples satisfy x+y+z=5?",
            "21",
            "20",
            "How many nonnegative integer triples satisfy x+y+z=5 and x<y?",
        ),
        (
            "A tree with 12 vertices has how many edges?",
            "11",
            "12",
            "A tree with 12 vertices has one extra edge added. How many edges does the resulting graph have?",
        ),
        (
            "How many edges does K_8 have?",
            "28",
            "27",
            "How many edges does K_8 have after deleting one edge?",
        ),
        (
            "How many edges does K_{3,5} have?",
            "15",
            "14",
            "How many edges does K_{3,5} have after deleting two edges?",
        ),
        (
            "Solve 7x ≡ 3 (mod 20).",
            "9",
            "8",
            "Solve 7x ≡ 3 (mod 20) subject to x > 10.",
        ),
    ],
)
def test_each_decisive_inferer_requires_complete_template_coverage(supported, correct, wrong, constrained):
    good = _evidence(supported, correct)
    assert any(item.is_decisive and item.status == "pass" for item in good), good

    bad = _evidence(supported, wrong)
    assert any(item.is_decisive and item.status == "fail" for item in bad), bad

    assert_no_false_decisive_pass(constrained, correct)


@pytest.mark.parametrize(
    ("problem", "candidate"),
    [
        (
            "Choose 4 objects from 10 with no two chosen objects adjacent.",
            "210",
        ),
        (
            "How many subsets of size 4 can be chosen from 10 objects if object 1 must be included?",
            "210",
        ),
        (
            "How many ordered selections of 3 from 7 without repetition if the first selected object must be 1?",
            "210",
        ),
        (
            "How many nonnegative integer triples satisfy x+y+z=5 and x<y?",
            "21",
        ),
        (
            "How many binary strings of length 8 with exactly 3 ones and starting with 1?",
            "56",
        ),
        (
            "A tree with 12 vertices has one extra edge added. How many edges does the resulting graph have?",
            "11",
        ),
        (
            "How many edges does K_8 have after deleting one edge?",
            "28",
        ),
        (
            "How many edges does K_{3,5} have after deleting two edges?",
            "15",
        ),
    ],
)
def test_reported_constraint_mutation_attacks_never_solve(problem, candidate):
    assert_no_false_decisive_pass(problem, candidate)


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("There are 210 ways before excluding one invalid case, so 209 remain.", 209),
        ("The answer is 210 is incorrect; actually 209.", 209),
        ("First we obtain 210. Final answer: 209.", 209),
        ("Therefore the answer is 210.", 210),
        ("210", 210),
    ],
)
def test_final_integer_extraction_uses_last_reliable_assertion(answer, expected):
    assert _extract_final_integer(answer) == expected


def test_intermediate_number_attack_does_not_verify_the_old_intermediate_value():
    problem = "Choose 4 objects from 10."
    candidate = "There are 210 ways before excluding one invalid case, so 209 remain."
    evidence = _evidence(problem, candidate)
    assert any(item.is_decisive and item.status == "fail" for item in evidence)
    assert not any(item.is_decisive and item.status == "pass" for item in evidence)


def test_all_integer_linear_congruence_residue_is_only_a_subclaim():
    item = next(
        item
        for item in _evidence("Find all integers x satisfying 7x ≡ 3 (mod 20).", "x=9")
        if item.method == "modular_check"
    )
    assert item.status == "pass"
    assert item.claim_scope == "subclaim"
    assert item.is_decisive is False


def test_least_nonnegative_linear_congruence_residue_remains_decisive():
    item = next(
        item
        for item in _evidence("Find the least nonnegative solution to 7x ≡ 3 (mod 20).", "9")
        if item.method == "modular_check"
    )
    assert item.status == "pass"
    assert item.claim_scope == "full_answer"
    assert item.is_decisive is True


def test_explicit_residue_modulo_m_remains_decisive():
    item = next(
        item
        for item in _evidence("Solve 7x ≡ 3 (mod 20) for x modulo 20.", "x ≡ 9 (mod 20)")
        if item.method == "modular_check"
    )
    assert item.status == "pass"
    assert item.claim_scope == "full_answer"
    assert item.is_decisive is True


def test_routing_surface_dataset_has_expected_realistic_vocabulary():
    rows = [json.loads(line) for line in ROUTING_SURFACES.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) >= 18
    text = " ".join(row["problem"].lower() for row in rows)
    for token in (
        "derangement",
        "onto functions",
        "stirling",
        "catalan",
        "compositions",
        "integer partitions",
        "ramsey",
        "chromatic",
        "euler trail",
        "hamiltonian",
        "hall's theorem",
        "graphical",
        "totient",
        "euclidean algorithm",
        "multiplicative order",
        "primitive root",
    ):
        assert token in text


@pytest.mark.parametrize(
    "row",
    [json.loads(line) for line in ROUTING_SURFACES.read_text(encoding="utf-8").splitlines() if line.strip()],
    ids=lambda row: row["idx"],
)
def test_realistic_routing_surfaces_map_to_existing_five_subtypes(row):
    route = classify_problem(row["problem"], {"subject": row["subject"]})
    assert route["primary_domain"] == row["expected_domain"], route
    assert route["discrete_subtype"] == row["expected_subtype"], route


@pytest.mark.parametrize(
    ("problem", "expected_subtype"),
    [
        ("Use a generating function to count proper colorings of a path graph.", "generating_function"),
        ("Via generating functions, count integer partitions with the stated restrictions.", "generating_function"),
        ("Use recurrence to compute the number of binary strings with no adjacent ones.", "recurrence"),
        ("Solve by recurrence the number of spanning trees in this recursively defined graph family.", "recurrence"),
    ],
)
def test_explicit_method_precedence_beats_object_keywords(problem, expected_subtype):
    route = classify_problem(problem, {"subject": "discrete_math"})
    assert route["discrete_subtype"] == expected_subtype

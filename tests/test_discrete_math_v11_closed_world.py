import json
from pathlib import Path

import pytest

from math_agent_core.router import classify_problem
from math_agent_core.verifiers.discrete_math import _extract_final_integer, run_discrete_math_verification

ROOT = Path(__file__).resolve().parents[1]
ROUTING = ROOT / "sample_data" / "discrete_math_v1_routing_surfaces.jsonl"


def _evidence(problem, answer):
    return run_discrete_math_verification(problem, answer, {"problem_type": "discrete_math", "requested_checks": []})


def assert_no_decisive_result(problem, candidate):
    evidence = _evidence(problem, candidate)
    assert not any(item.is_decisive for item in evidence), evidence


def assert_no_false_decisive_pass(problem, candidate):
    evidence = _evidence(problem, candidate)
    assert not any(item.is_decisive and item.status == "pass" for item in evidence), evidence


def assert_no_false_decisive_fail(problem, candidate):
    evidence = _evidence(problem, candidate)
    assert not any(item.is_decisive and item.status == "fail" for item in evidence), evidence


@pytest.mark.parametrize(
    ("problem", "old_answer", "correct_answer"),
    [
        ("Choose 4 objects from 10, including object 1.", "210", "84"),
        ("Choose 4 objects from 10 with object 1 included.", "210", "84"),
        ("Choose 4 objects from 10 given that object 1 is selected.", "210", "84"),
        ("Choose 4 objects from 10 assuming object 1 is selected.", "210", "84"),
        ("Choose 4 objects from 10 with object 1 excluded.", "210", "126"),
        ("How many ordered selections of 3 from 7 without repetition with first selected object equal to 1?", "210", "30"),
        ("How many ordered selections of 3 from 7 without repetition beginning with object 1?", "210", "30"),
        ("How many ordered selections of 3 from 7 without repetition with object 1 in first position?", "210", "30"),
        ("How many binary strings of length 8 with exactly 3 ones and beginning with 1?", "56", "21"),
        ("How many binary strings of length 8 with exactly 3 ones whose first bit is 1?", "56", "21"),
        ("How many binary strings of length 8 with exactly 3 ones with final bit 0?", "56", "35"),
        ("How many binary strings of length 8 with exactly 3 ones whose last bit is 0?", "56", "35"),
        ("How many nonnegative integer triples satisfy x+y+z=5 with x=y?", "21", "3"),
        ("How many nonnegative integer triples satisfy x+y+z=5 with x divisible by 2?", "21", "12"),
        ("How many edges does K_8 have with one edge absent?", "28", "27"),
        ("How many edges does K_8 have minus one edge?", "28", "27"),
        ("A tree with 12 vertices plus one extra chord has how many edges?", "11", "12"),
        ("How many edges does K_{3,5} have with one edge missing?", "15", "14"),
        ("A graph has degree sequence [3,3,2,2,2,2]. How many edges does it have after one edge is removed?", "7", "6"),
        ("What is the maximum number of edges in a simple graph on 6 vertices that is planar?", "15", "12"),
        ("What is the maximum number of edges in a simple graph on 6 vertices without triangles?", "15", "9"),
    ],
)
def test_paraphrase_constraints_never_create_decisive_pass_or_fail(problem, old_answer, correct_answer):
    assert_no_false_decisive_pass(problem, old_answer)
    assert_no_false_decisive_fail(problem, correct_answer)


MUTATION_FAMILIES = [
    ("Choose 4 objects from 10", "210", ["including object 1", "assuming a distinguished object is selected", "with a designated item retained"]),
    ("How many ordered selections of 3 from 7 without repetition", "210", ["beginning with object 1", "with a designated object first", "under an extra position rule"]),
    ("How many binary strings of length 8 with exactly 3 ones", "56", ["whose first bit is 1", "ending in 0", "under an additional endpoint rule"]),
    ("How many nonnegative integer triples satisfy x+y+z=5", "21", ["with x=y", "under an extra parity rule", "with a divisibility requirement"]),
    ("A tree with 12 vertices has how many edges", "11", ["after an extra chord is inserted", "for the modified graph", "after changing the edge set"]),
    ("How many edges does K_8 have", "28", ["with one edge absent", "minus one edge", "after altering its edge set"]),
    ("How many edges does K_{3,5} have", "15", ["with one edge missing", "after an edge alteration", "for a modified version of the graph"]),
    ("A graph has degree sequence [3,3,2,2,2,2]. How many edges does it have", "7", ["after one edge is removed", "for a modified graph", "after changing the graph"]),
    ("What is the maximum number of edges in a simple graph on 6 vertices", "15", ["that obeys an additional property", "from a restricted family", "under another graph condition"]),
    ("Compute 3^100 modulo 17", "13", ["under an additional restriction", "for a specially constrained case", "assuming another condition"]),
    ("Solve 7x ≡ 3 (mod 20)", "9", ["subject to another condition", "with an additional requirement", "among values satisfying another rule"]),
]


@pytest.mark.parametrize(
    ("base", "answer", "residuals"), MUTATION_FAMILIES, ids=[f"family_{i}" for i in range(len(MUTATION_FAMILIES))]
)
def test_mutation_families_unknown_residual_content_is_never_decisive(base, answer, residuals):
    for residual in residuals:
        assert_no_decisive_result(f"{base} {residual}.", answer)


@pytest.mark.parametrize(
    "answer",
    [
        "The answer is 210, unless one excludes a case, in which case 209.",
        "The answer is 210 if no restriction applies; with the restriction, 209.",
        "There are 210 ways provided a condition holds, otherwise 209.",
        "210 if the first case applies, otherwise 209.",
    ],
)
def test_conditional_competing_final_assertions_return_none(answer):
    assert _extract_final_integer(answer) is None


@pytest.mark.parametrize(
    "problem",
    [
        "Find all integers x satisfying 7x ≡ 3 (mod 20).",
        "Find all integer solutions to 7x ≡ 3 (mod 20).",
        "Solve 7x ≡ 3 (mod 20) in integers.",
        "Solve 7x ≡ 3 (mod 20) over the integers.",
        "Solve 7x ≡ 3 (mod 20) for integer x.",
    ],
)
def test_all_integer_congruence_request_modes_are_nondecisive(problem):
    item = next(item for item in _evidence(problem, "9") if item.method == "modular_check")
    assert item.status == "pass"
    assert item.claim_scope == "subclaim"
    assert item.is_decisive is False


def test_least_nonnegative_congruence_remains_decisive():
    item = next(item for item in _evidence("Find the least nonnegative solution to 7x ≡ 3 (mod 20).", "9") if item.method == "modular_check")
    assert item.status == "pass"
    assert item.claim_scope == "full_answer"
    assert item.is_decisive is True


def test_unspecified_solve_convention_is_explicitly_preserved():
    item = next(item for item in _evidence("Solve 7x ≡ 3 (mod 20).", "9") if item.method == "modular_check")
    assert item.status == "pass"
    assert item.is_decisive is True


def test_routing_surfaces_work_with_and_without_subject_metadata():
    rows = [json.loads(line) for line in ROUTING.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        with_subject = classify_problem(row["problem"], {"subject": row["subject"]})
        without_subject = classify_problem(row["problem"], {})
        for route in (with_subject, without_subject):
            assert route["primary_domain"] == row["expected_domain"], (row, route)
            assert route["discrete_subtype"] == row["expected_subtype"], (row, route)

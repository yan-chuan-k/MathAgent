import json
from collections import Counter
from pathlib import Path

import pytest

from math_agent_core.router import classify_problem
from math_agent_core.verifiers.discrete_math import _extract_final_integer, run_discrete_math_verification


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS = ROOT / "sample_data" / "discrete_math_v1_diagnostics.jsonl"
ROUTING = ROOT / "sample_data" / "discrete_math_v1_routing_surfaces.jsonl"


def _result():
    return {"problem_type": "discrete_math", "requested_checks": []}


def _evidence(problem, answer):
    return run_discrete_math_verification(problem, answer, _result())


def _decisive(problem, answer):
    return [item for item in _evidence(problem, answer) if item.is_decisive]


def assert_no_decisive_pass_on_semantic_mismatch(problem, candidate):
    evidence = _evidence(problem, candidate)
    assert not any(item.is_decisive and item.status == "pass" for item in evidence), evidence


def assert_no_decisive_result_on_semantic_mismatch(problem, candidate):
    evidence = _evidence(problem, candidate)
    assert not any(item.is_decisive for item in evidence), evidence


def test_least_nonnegative_request_validates_canonical_residue_range():
    problem = "Find the least nonnegative solution to 7x ≡ 3 (mod 20)."

    correct = _decisive(problem, "9")
    assert len(correct) == 1
    assert correct[0].status == "pass"
    assert correct[0].claim_scope == "full_answer"

    for candidate in ("29", "-11"):
        wrong = _decisive(problem, candidate)
        assert len(wrong) == 1
        assert wrong[0].status == "fail"
        assert wrong[0].claim_scope == "full_answer"


def test_least_nonnegative_range_rule_does_not_leak_to_residue_class_mode():
    evidence = _decisive("Solve 7x ≡ 3 (mod 20) for x modulo 20.", "x ≡ 29 (mod 20)")
    assert len(evidence) == 1
    assert evidence[0].status == "pass"
    assert evidence[0].claim_scope == "full_answer"


def test_unspecified_congruence_convention_is_unchanged():
    evidence = _decisive("Solve 7x ≡ 3 (mod 20).", "29")
    assert len(evidence) == 1
    assert evidence[0].status == "pass"


@pytest.mark.parametrize(
    ("problem", "candidate"),
    [
        ("How many nonnegative integer pairs satisfy x+y=5?", "6"),
        ("How many nonnegative integer triples satisfy x+y+z=5?", "21"),
        ("How many positive integer pairs satisfy x+y=5?", "4"),
        ("How many positive integer triples satisfy x+y+z=5?", "6"),
    ],
)
def test_integer_tuple_shape_and_variable_arity_supported_controls(problem, candidate):
    evidence = _decisive(problem, candidate)
    assert len(evidence) == 1
    assert evidence[0].status == "pass"
    assert evidence[0].claim_scope == "full_answer"


@pytest.mark.parametrize(
    ("problem", "candidate"),
    [
        ("How many nonnegative integer triples satisfy x+y=5?", "6"),
        ("How many positive integer triples satisfy x+y=5?", "4"),
        ("How many nonnegative integer pairs satisfy x+y+z=5?", "21"),
        ("How many positive integer pairs satisfy x+y+z=5?", "6"),
    ],
)
def test_integer_tuple_shape_variable_arity_mismatch_is_not_decisive(problem, candidate):
    assert_no_decisive_result_on_semantic_mismatch(problem, candidate)


def test_semantic_field_completeness_regression_family():
    mismatch_cases = [
        ("How many nonnegative integer triples satisfy x+y=5?", "6"),
        ("How many positive integer triples satisfy x+y=5?", "4"),
        ("How many nonnegative integer pairs satisfy x+y+z=5?", "21"),
        ("How many positive integer pairs satisfy x+y+z=5?", "6"),
        ("Find the least nonnegative solution to 7x ≡ 3 (mod 20).", "29"),
        ("Find the least nonnegative solution to 7x ≡ 3 (mod 20).", "-11"),
    ]
    for problem, candidate in mismatch_cases[:4]:
        assert_no_decisive_pass_on_semantic_mismatch(problem, candidate)
    for problem, candidate in mismatch_cases[4:]:
        evidence = _decisive(problem, candidate)
        assert evidence and all(item.status == "fail" for item in evidence), evidence


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("The answer is 210.", 210),
        ("Answer: 210", 210),
        ("Answer = 210", 210),
        ("Answer is 210", 210),
        ("The answer: 210", 210),
        ("The result is 210.", 210),
        ("There are exactly 210 ways.", 210),
        ("Final answer: 210", 210),
        ("Therefore the answer is 210.", 210),
        ("210", 210),
    ],
)
def test_safe_single_assertion_answer_wrappers(answer, expected):
    assert _extract_final_integer(answer) == expected


@pytest.mark.parametrize(
    "answer",
    [
        "The answer is 210, unless one excludes a case, in which case 209.",
        "The answer is 210 if no restriction applies; with the restriction, 209.",
        "There are 210 ways provided a condition holds, otherwise 209.",
        "210 if the first case applies, otherwise 209.",
    ],
)
def test_conditional_competing_answer_wrappers_remain_unextractable(answer):
    assert _extract_final_integer(answer) is None


def _diagnostic_rows():
    return [json.loads(line) for line in DIAGNOSTICS.read_text(encoding="utf-8").splitlines() if line.strip()]


def _natural_answer_surface(row):
    answer = row["expected_answer"]
    if row["expected_subtype"] == "combinatorial_counting":
        return f"There are exactly {answer} ways."
    return f"The answer is {answer}."


def test_deterministic_trigger_rate_bare_and_natural_answer_surfaces_match():
    rows = _diagnostic_rows()
    bare = Counter()
    natural = Counter()
    for row in rows:
        subtype = row["expected_subtype"]
        bare[subtype] += int(bool(_evidence(row["problem"], row["expected_answer"])))
        natural[subtype] += int(bool(_evidence(row["problem"], _natural_answer_surface(row))))

    expected = Counter({
        "combinatorial_counting": 5,
        "graph_theory": 5,
        "number_theory_modular": 4,
    })
    assert bare == expected
    assert natural == expected
    assert sum(bare.values()) == 14
    assert sum(natural.values()) == 14


def test_routing_metrics_remain_frozen_with_and_without_subject():
    diagnostics = _diagnostic_rows()
    route_hits = subtype_hits = 0
    for row in diagnostics:
        route = classify_problem(row["problem"], {"subject": row["subject"], "task_type": row["task_type"]})
        route_hits += int(route["primary_domain"] == "discrete_math")
        subtype_hits += int(route["discrete_subtype"] == row["expected_subtype"])
    assert route_hits == 25
    assert subtype_hits == 25

    surfaces = [json.loads(line) for line in ROUTING.read_text(encoding="utf-8").splitlines() if line.strip()]
    with_subject = without_subject = 0
    for row in surfaces:
        a = classify_problem(row["problem"], {"subject": row["subject"]})
        b = classify_problem(row["problem"], {})
        with_subject += int(a["primary_domain"] == row["expected_domain"] and a["discrete_subtype"] == row["expected_subtype"])
        without_subject += int(b["primary_domain"] == row["expected_domain"] and b["discrete_subtype"] == row["expected_subtype"])
    assert with_subject == 21
    assert without_subject == 21

import json
from collections import Counter
from pathlib import Path

from evaluate_discrete_math_realistic_v1 import GROUND_TRUTH_FIELDS, safe_solver_metadata
from math_agent_core.evaluation.grader import grade_full_problem


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "sample_data" / "discrete_math_realistic_eval_v1.jsonl"
DIAGNOSTIC = ROOT / "sample_data" / "discrete_math_v1_diagnostics.jsonl"


def _rows(path=FIXTURE):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_realistic_eval_is_balanced_100_case_fixture():
    rows = _rows()
    assert len(rows) == 100
    assert Counter(row["expected_subtype"] for row in rows) == {
        "combinatorial_counting": 20,
        "recurrence": 20,
        "generating_function": 20,
        "graph_theory": 20,
        "number_theory_modular": 20,
    }


def test_realistic_eval_language_distribution_is_balanced():
    rows = _rows()
    assert Counter(row["language"] for row in rows) == {"en": 50, "zh": 50}


def test_realistic_eval_ground_truth_is_gradable_by_frozen_grader():
    for row in _rows():
        grading = grade_full_problem(row["grading"]["primary"], row["grading"])
        assert grading["status"] == "CORRECT", (row["idx"], grading)


def test_realistic_eval_solver_metadata_excludes_all_ground_truth_fields():
    for row in _rows():
        for include_subject in (True, False):
            metadata = safe_solver_metadata(row, include_subject=include_subject)
            assert not GROUND_TRUTH_FIELDS.intersection(metadata), (row["idx"], metadata)
            assert "expected_domain" not in metadata
            assert "expected_subtype" not in metadata
            assert "grading" not in metadata
            if include_subject:
                assert metadata["subject"] == row["subject"]
            else:
                assert "subject" not in metadata


def test_realistic_eval_does_not_replace_or_duplicate_frozen_diagnostic_rows():
    realistic = _rows()
    diagnostic = _rows(DIAGNOSTIC)
    realistic_problems = {row["problem"].strip() for row in realistic}
    diagnostic_problems = {row["problem"].strip() for row in diagnostic}
    assert realistic_problems.isdisjoint(diagnostic_problems)
    assert len(diagnostic) == 25

from evaluate_discrete_math_realistic_v1 import (
    _failure_category,
    build_v2_candidate_table,
    run_evaluation,
    sha256_file,
    v2_decision,
)

FROZEN_REALISTIC_FIXTURE_SHA256 = "ee555e3487bf76b5a47120fa1b364a7f6b3b641e2e1acba2a0d32a92450d4f76"


def _route(domain_correct=True, subtype_correct=True):
    return {
        "domain_correct": domain_correct,
        "subtype_correct": subtype_correct,
        "primary_domain": "discrete_math" if domain_correct else "unknown",
        "discrete_subtype": "graph_theory" if subtype_correct else "general_discrete",
    }


def _acceptance(solved=False, verified=False):
    return {
        "overall_status": "solved" if solved else "invalid",
        "answer_verified": verified,
    }


def _verifier(decisive_pass=0, decisive_fail=0, nondecisive=0):
    return {
        "triggered": bool(decisive_pass or decisive_fail or nondecisive),
        "decisive_pass_count": decisive_pass,
        "decisive_fail_count": decisive_fail,
        "nondecisive_count": nondecisive,
    }


def _category(*, correct, domain_correct=True, subtype_correct=True, solved=False, verified=False,
              decisive_pass=0, decisive_fail=0, nondecisive=0, eligible=False):
    return _failure_category(
        {},
        _route(domain_correct, subtype_correct),
        {"correct": correct},
        _acceptance(solved, verified),
        _verifier(decisive_pass, decisive_fail, nondecisive),
        {"eligible": eligible},
        "other_supported_numeric",
        "candidate",
    )[0]


def test_causal_case_a_subtype_wrong_correct_answer_accepted_is_not_failure():
    category = _category(correct=True, subtype_correct=False, solved=True, verified=True)
    assert category is None


def test_causal_case_b_subtype_wrong_incorrect_answer_is_subtype_error():
    category = _category(correct=False, subtype_correct=False)
    assert category == "SUBTYPE_ERROR"


def test_causal_case_c_correct_route_incorrect_answer_remains_unknown():
    category = _category(correct=False, domain_correct=True, subtype_correct=True)
    assert category == "UNKNOWN"


def test_causal_case_d_correct_answer_with_decisive_fail_is_verifier_false_rejection():
    category = _category(correct=True, decisive_fail=1)
    assert category == "VERIFIER_FALSE_REJECTION"


def test_causal_case_e_wrong_answer_with_decisive_pass_is_acceptance_error():
    category = _category(correct=False, decisive_pass=1)
    assert category == "ACCEPTANCE_ERROR"


def test_causal_case_f_mock_mode_never_recommends_v2():
    summary = {
        "solver_accuracy_is_decision_grade": False,
        "rows": [
            {
                "expected_subtype": "combinatorial_counting",
                "solver_answer_correct": False,
                "route_domain_mismatch": True,
                "route_subtype_mismatch": True,
                "answer_verified": False,
                "answer_surface_miss": False,
                "failure_category": "ROUTING_ERROR",
            }
        ],
    }
    candidates = build_v2_candidate_table(summary)
    assert all(not row["recommended"] for row in candidates)
    assert all(row["measured_failures_plausibly_addressed"] == 0 for row in candidates)
    assert v2_decision({**summary, "v2_candidates": candidates}) == "V2 NOT JUSTIFIED YET"


def test_causal_case_g_real_unique_largest_cluster_selects_exactly_one_candidate():
    rows = []
    for _ in range(3):
        rows.append({
            "expected_subtype": "graph_theory",
            "solver_answer_correct": False,
            "route_domain_mismatch": False,
            "route_subtype_mismatch": True,
            "answer_verified": False,
            "answer_surface_miss": False,
            "failure_category": "SUBTYPE_ERROR",
        })
    rows.append({
        "expected_subtype": "recurrence",
        "solver_answer_correct": True,
        "route_domain_mismatch": False,
        "route_subtype_mismatch": False,
        "answer_verified": False,
        "answer_surface_miss": False,
        "failure_category": "VERIFIER_COVERAGE_GAP",
    })
    summary = {"solver_accuracy_is_decision_grade": True, "rows": rows}
    candidates = build_v2_candidate_table(summary)
    recommended = [row for row in candidates if row["recommended"]]
    assert len(recommended) == 1
    assert recommended[0]["candidate_improvement"] == "routing vocabulary improvements"
    decision = v2_decision({**summary, "v2_candidates": candidates})
    assert decision == "V2 CANDIDATE: routing vocabulary improvements"


def test_realistic_eval_fixture_checksum_is_frozen_decision_set():
    assert sha256_file(FIXTURE) == FROZEN_REALISTIC_FIXTURE_SHA256


def test_route_only_evaluation_does_not_convert_route_misses_into_failure_taxonomy():
    summary = run_evaluation(_rows(), run_agent=False, use_mock=False, thinking_mode=True)
    assert summary["route_subtype_mismatch_count_with_subject"] == 8
    assert summary["failure_taxonomy_counts"] == {}
    assert summary["incorrect_and_subtype_mismatch_count"] is None
    assert summary["required_claim_evaluable_count"] == 0
    assert summary["method_compliance_evaluable_count"] == 0
    assert "collapses to primary-answer grading" in summary["full_problem_metric_note"]
    required_row_fields = {
        "route_domain_correct",
        "route_subtype_correct",
        "route_domain_mismatch",
        "route_subtype_mismatch",
        "solver_answer_correct",
        "solver_answer_unresolved",
        "acceptance_solved",
        "answer_verified",
    }
    assert all(required_row_fields.issubset(row) for row in summary["rows"])
    assert all(row["solver_answer_correct"] is None for row in summary["rows"])

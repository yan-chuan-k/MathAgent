import json
from collections import Counter
from pathlib import Path

from math_agent_core.evaluation.grader import grade_full_problem
from math_agent_core.router import classify_problem


FIXTURE = Path(__file__).resolve().parents[1] / "sample_data" / "discrete_math_v1_diagnostics.jsonl"
EXPECTED_SUBTYPES = {
    "combinatorial_counting",
    "recurrence",
    "generating_function",
    "graph_theory",
    "number_theory_modular",
}


def _rows():
    return [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_discrete_v1_diagnostics_are_balanced_and_complete():
    rows = _rows()
    assert len(rows) == 25
    assert Counter(row["expected_subtype"] for row in rows) == {subtype: 5 for subtype in EXPECTED_SUBTYPES}
    for row in rows:
        assert row["expected_domain"] == "discrete_math"
        assert row["task_type"]
        assert row["grading"]["primary_type"]
        assert isinstance(row["grading"]["required_claims"], list)


def test_discrete_v1_diagnostics_route_to_expected_subtypes():
    for row in _rows():
        route = classify_problem(row["problem"], {"subject": row["subject"], "task_type": row["task_type"]})
        assert route["primary_domain"] == "discrete_math", (row["idx"], route)
        assert route["discrete_subtype"] == row["expected_subtype"], (row["idx"], route)


def test_discrete_v1_diagnostic_ground_truth_is_gradable_by_frozen_grader():
    for row in _rows():
        grade = grade_full_problem(row["expected_answer"], row["grading"])
        assert grade["status"] == "CORRECT", (row["idx"], grade)

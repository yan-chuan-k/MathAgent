import json
from collections import Counter
from pathlib import Path

from math_agent_core.router import classify_problem
from math_agent_core.verifiers.discrete_math import run_discrete_math_verification


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS = ROOT / "sample_data" / "discrete_math_v1_diagnostics.jsonl"


def _rows():
    return [json.loads(line) for line in DIAGNOSTICS.read_text(encoding="utf-8").splitlines() if line.strip()]


def _result():
    return {"problem_type": "discrete_math", "requested_checks": []}


def test_discrete_v1_offline_coverage_metrics_are_explicit_and_false_decisive_zero():
    rows = _rows()
    route_hits = 0
    subtype_hits = 0
    triggers = Counter()
    decisive = Counter()

    for row in rows:
        route = classify_problem(
            row["problem"],
            {"subject": row["subject"], "task_type": row["task_type"]},
        )
        route_hits += int(route["primary_domain"] == "discrete_math")
        subtype_hits += int(route["discrete_subtype"] == row["expected_subtype"])

        evidence = run_discrete_math_verification(row["problem"], row["expected_answer"], _result())
        subtype = row["expected_subtype"]
        triggers[subtype] += int(bool(evidence))
        decisive[subtype] += int(any(item.is_decisive for item in evidence))

    assert route_hits == 25
    assert subtype_hits == 25
    assert triggers == Counter(
        {
            "combinatorial_counting": 5,
            "graph_theory": 5,
            "number_theory_modular": 4,
        }
    )
    assert decisive == Counter(
        {
            "combinatorial_counting": 5,
            "graph_theory": 5,
            "number_theory_modular": 4,
        }
    )


def test_constraint_mutation_false_decisive_pass_metric_is_zero():
    attacks = [
        ("Choose 4 objects from 10 with no two chosen objects adjacent.", "210"),
        ("How many subsets of size 4 can be chosen from 10 objects if object 1 must be included?", "210"),
        ("How many ordered selections of 3 from 7 without repetition if the first selected object must be 1?", "210"),
        ("How many nonnegative integer triples satisfy x+y+z=5 and x<y?", "21"),
        ("How many binary strings of length 8 with exactly 3 ones and starting with 1?", "56"),
        ("A tree with 12 vertices has one extra edge added. How many edges does the resulting graph have?", "11"),
        ("How many edges does K_8 have after deleting one edge?", "28"),
        ("How many edges does K_{3,5} have after deleting two edges?", "15"),
        ("Solve 7x ≡ 3 (mod 20) subject to x > 10.", "9"),
    ]
    false_decisive_pass_count = 0
    for problem, candidate in attacks:
        evidence = run_discrete_math_verification(problem, candidate, _result())
        false_decisive_pass_count += sum(
            item.is_decisive and item.status == "pass" for item in evidence
        )
    assert false_decisive_pass_count == 0

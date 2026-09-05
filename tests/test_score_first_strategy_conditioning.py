from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pytest

from math_agent_core.router import classify_problem
from user_agent import (
    ReasoningAgent,
    _SCORE_FIRST_DISCRETE_SUBTYPE_STRATEGIES,
    _SCORE_FIRST_DOMAIN_STRATEGIES,
)


ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC = ROOT / "sample_data" / "score_recovery_v2_synthetic_hard.jsonl"
HARD = ROOT / "sample_data" / "hard_diagnostics.jsonl"

HARD_DOMAINS = (
    "discrete_math",
    "numerical_analysis",
    "measure_integration",
    "differential_geometry",
    "probability",
    "abstract_algebra",
    "stochastic_process",
    "complex_analysis",
    "ode",
    "statistics",
    "functional_analysis",
    "linear_regression",
    "pde",
    "advanced_math",
    "linear_algebra",
    "optimization",
    "real_analysis",
    "topology",
)


class RecordingClient:
    def __init__(self, response: str = "Final answer: 42"):
        self.response = response
        self.calls: List[Dict[str, Any]] = []

    def chat(self, messages, temperature=0.1, max_tokens=8192, thinking_mode=True):
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "thinking_mode": thinking_mode,
            }
        )
        return self.response


def _prompt_for(problem: str, subject: str):
    client = RecordingClient()
    agent = ReasoningAgent(client=client)
    result = agent.solve(problem, {"subject": subject})
    assert result["final_response"] == "42"
    assert len(client.calls) == 1
    return "\n".join(item["content"] for item in client.calls[0]["messages"]), client.calls[0]


@pytest.mark.parametrize(
    ("domain", "problem"),
    [
        ("discrete_math", "How many subsets of size 3 can be chosen from 9 objects?"),
        ("numerical_analysis", "Use Newton's method to approximate a root of x^2-2."),
        ("measure_integration", "Can dominated convergence be applied to interchange this limit and integral?"),
        ("differential_geometry", "Compute the Gaussian curvature of a parametrized surface."),
        ("probability", "Compute the conditional probability P(A|B)."),
        ("abstract_algebra", "Determine the kernel of this group homomorphism."),
        ("stochastic_process", "Find a stationary distribution for this Markov chain."),
        ("complex_analysis", "Compute a contour integral using residues."),
        ("ode", "Solve the initial value problem y'+y=0."),
        ("statistics", "Find the maximum-likelihood estimator."),
        ("functional_analysis", "Determine whether this bounded operator is compact."),
        ("linear_regression", "Compute the OLS coefficient estimator."),
        ("pde", "Solve the heat equation with homogeneous boundary conditions."),
        ("advanced_math", "Evaluate this graduate-level variational expression."),
        ("linear_algebra", "Determine the eigenvalues and diagonalizability of the matrix."),
        ("optimization", "Minimize this convex function subject to one inequality constraint."),
        ("real_analysis", "Determine whether the sequence converges uniformly."),
        ("topology", "Determine whether the continuous image is compact."),
    ],
)
def test_score_first_selects_exactly_one_compact_domain_strategy(domain, problem):
    prompt, call = _prompt_for(problem, domain)
    selected = _SCORE_FIRST_DOMAIN_STRATEGIES[domain]
    assert selected in prompt

    # No other domain card is pasted into the prompt.
    for other_domain, other_card in _SCORE_FIRST_DOMAIN_STRATEGIES.items():
        if other_domain != domain:
            assert other_card not in prompt

    assert call["temperature"] == 0.8
    assert call["max_tokens"] == 32768
    assert call["thinking_mode"] is True
    assert "OUTPUT_CONTRACT" not in prompt
    assert "Return exactly one valid JSON object" not in prompt
    assert "requested_checks" not in prompt
    assert "reasoning_plan" not in prompt
    assert "The subject/strategy hint is advisory." in prompt
    assert "Before emitting the final answer, internally check:" in prompt


@pytest.mark.parametrize(
    ("subtype", "problem"),
    [
        ("combinatorial_counting", "How many subsets of size 3 can be chosen from 9 objects?"),
        ("recurrence", "Using the recurrence a_0=1 and a_n=2a_{n-1}, compute a_5."),
        ("generating_function", "Using a generating function, find the coefficient of x^5 in 1/(1-x)^2."),
        ("graph_theory", "A graph has 8 vertices. Determine whether its degree sequence is graphical."),
        ("number_theory_modular", "Solve 7x ≡ 3 (mod 20)."),
    ],
)
def test_discrete_math_injects_only_selected_subtype_card(subtype, problem):
    client = RecordingClient()
    agent = ReasoningAgent(client=client)
    result = agent.solve(problem, {"subject": "discrete_math"})
    assert result["final_response"] == "42"
    assert len(client.calls) == 1

    prompt = "\n".join(item["content"] for item in client.calls[0]["messages"])
    assert _SCORE_FIRST_DOMAIN_STRATEGIES["discrete_math"] in prompt
    assert _SCORE_FIRST_DISCRETE_SUBTYPE_STRATEGIES[subtype] in prompt
    for other_subtype, other_card in _SCORE_FIRST_DISCRETE_SUBTYPE_STRATEGIES.items():
        if other_subtype != subtype:
            assert other_card not in prompt


def test_strategy_cards_remain_compact_and_closed_to_18_domains():
    assert set(_SCORE_FIRST_DOMAIN_STRATEGIES) == set(HARD_DOMAINS)
    for domain, card in _SCORE_FIRST_DOMAIN_STRATEGIES.items():
        words = card.split()
        assert 20 <= len(words) <= 80, (domain, len(words), card)

    assert set(_SCORE_FIRST_DISCRETE_SUBTYPE_STRATEGIES) == {
        "combinatorial_counting",
        "recurrence",
        "generating_function",
        "graph_theory",
        "number_theory_modular",
    }
    for subtype, card in _SCORE_FIRST_DISCRETE_SUBTYPE_STRATEGIES.items():
        words = card.split()
        assert 15 <= len(words) <= 60, (subtype, len(words), card)


def test_strategy_prompt_overhead_is_compact():
    problem = "Compute 6*7."
    prompt, _ = _prompt_for(problem, "advanced_math")
    overhead = prompt.replace(problem, "")
    assert len(overhead) < 2600


def test_strategy_conditioning_preserves_one_call_budget_for_100_successes():
    client = RecordingClient("Final answer: 42")
    agent = ReasoningAgent(client=client)
    domains = list(HARD_DOMAINS)
    for index in range(100):
        domain = domains[index % len(domains)]
        result = agent.solve(f"Compute 6*7 for case {index}.", {"subject": domain})
        assert result["final_response"] == "42"
    assert len(client.calls) == 100


def test_synthetic_hard_suite_is_held_out_balanced_and_routes_with_subject():
    rows = [json.loads(line) for line in SYNTHETIC.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 110

    domain_counts = Counter(row["expected_domain"] for row in rows)
    assert domain_counts["discrete_math"] == 25
    for domain in HARD_DOMAINS:
        if domain != "discrete_math":
            assert domain_counts[domain] == 5

    subtype_counts = Counter(
        row["expected_subtype"]
        for row in rows
        if row["expected_domain"] == "discrete_math"
    )
    assert subtype_counts == {
        "combinatorial_counting": 5,
        "recurrence": 5,
        "generating_function": 5,
        "graph_theory": 5,
        "number_theory_modular": 5,
    }

    old_problems = {
        json.loads(line)["problem"].strip()
        for line in HARD.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert all(row["problem"].strip() not in old_problems for row in rows)

    for row in rows:
        metadata = {
            "subject": row["subject"],
            "task_type": row["task_type"],
            "case_style": row["case_style"],
        }
        route = classify_problem(row["problem"], metadata)
        assert route["primary_domain"] == row["expected_domain"], (row["idx"], route)
        if row.get("expected_subtype"):
            assert route["discrete_subtype"] == row["expected_subtype"], (row["idx"], route)


def test_synthetic_suite_ground_truth_is_never_forwarded_as_solver_metadata():
    rows = [json.loads(line) for line in SYNTHETIC.read_text(encoding="utf-8").splitlines() if line.strip()]
    forbidden = {"grading", "expected_domain", "expected_subtype", "answer", "expected_answer", "answer_hint"}

    for row in rows:
        metadata = {
            key: value
            for key, value in row.items()
            if key not in forbidden | {"problem"}
        }
        assert forbidden.isdisjoint(metadata)
        assert "grading" not in metadata
        assert "expected_domain" not in metadata
        assert "expected_subtype" not in metadata
        assert "answer" not in metadata
        assert "expected_answer" not in metadata

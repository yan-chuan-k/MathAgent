from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from user_agent import (
    ReasoningAgent,
    _SCORE_FIRST_DOMAIN_FINAL_CHECKS,
    _SCORE_FIRST_DISCRETE_FINAL_CHECKS,
    _SCORE_FIRST_MICRO_FINAL_CHECKS,
    _SCORE_FIRST_HUMAN_DOMAIN_LABELS,
    _SCORE_FIRST_RESPONSE_MODE_ANSWER,
    _SCORE_FIRST_RESPONSE_MODE_DERIVATION,
)


ROOT = Path(__file__).resolve().parents[1]
VERIFICATION_FIXTURE = ROOT / "sample_data" / "score_recovery_v25_same_call_verification.jsonl"
ROUTING_FIXTURE = ROOT / "sample_data" / "score_recovery_v2_synthetic_hard.jsonl"


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


def _rows(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_verification_fixture_selects_exactly_one_expected_check():
    agent = ReasoningAgent(RecordingClient())
    failures = []

    for row in _rows(VERIFICATION_FIXTURE):
        context = agent._score_first_context(
            row["problem"],
            {"subject": row["subject"]},
        )
        if context["verification_key"] != row["expected_verification_key"]:
            failures.append(
                (
                    row["idx"],
                    row["expected_verification_key"],
                    context["verification_key"],
                )
            )
        assert context["verification_card"]
        assert isinstance(context["verification_card"], str)

    assert failures == []


def test_verification_fixture_labels_never_enter_solver_metadata():
    agent = ReasoningAgent(RecordingClient())
    for row in _rows(VERIFICATION_FIXTURE):
        metadata = {"subject": row["subject"]}
        assert "expected_verification_key" not in metadata
        context = agent._score_first_context(row["problem"], metadata)
        assert context["verification_key"] == row["expected_verification_key"]


def test_one_verification_block_appears_once_in_prompt():
    client = RecordingClient()
    agent = ReasoningAgent(client)

    for row in _rows(VERIFICATION_FIXTURE):
        client.calls.clear()
        agent.solve(row["problem"], {"subject": row["subject"]})
        assert len(client.calls) == 1
        user_prompt = client.calls[0]["messages"][1]["content"]
        assert user_prompt.count("Internal final check:") == 1

        context = agent._score_first_context(
            row["problem"],
            {"subject": row["subject"]},
        )
        assert user_prompt.count(context["verification_card"]) == 1
        assert "expected_verification_key" not in user_prompt


def test_verification_selection_hierarchy_micro_then_discrete_then_domain():
    agent = ReasoningAgent(RecordingClient())

    micro = agent._score_first_context(
        "Use Newton interpolation with divided differences to evaluate the interpolant at x=2.",
        {"subject": "Numerical Analysis"},
    )
    assert micro["verification_key"] == "micro:interpolation"

    discrete = agent._score_first_context(
        "Using a generating function, find the coefficient of x^5 in (1-x)^(-2).",
        {"subject": "Discrete Mathematics"},
    )
    assert discrete["verification_key"] == "discrete:generating_function"

    domain = agent._score_first_context(
        "Solve y''-3y'+2y=0 with y(0)=1 and y'(0)=0.",
        {"subject": "Ordinary Differential Equations"},
    )
    assert domain["verification_key"] == "domain:ode"


def test_all_18_domains_have_compact_fallback_checks():
    expected = {
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
    }
    assert set(_SCORE_FIRST_DOMAIN_FINAL_CHECKS) == expected
    assert set(_SCORE_FIRST_DISCRETE_FINAL_CHECKS) == {
        "combinatorial_counting",
        "recurrence",
        "generating_function",
        "graph_theory",
        "number_theory_modular",
    }
    assert all(len(card.split()) <= 35 for card in _SCORE_FIRST_DOMAIN_FINAL_CHECKS.values())
    assert all(len(card.split()) <= 45 for card in _SCORE_FIRST_DISCRETE_FINAL_CHECKS.values())
    assert all(len(card.split()) <= 45 for card in _SCORE_FIRST_MICRO_FINAL_CHECKS.values())


@pytest.mark.parametrize(
    ("problem", "expected_span"),
    [
        ("Differentiate f(x)=x^3 and evaluate at x=2.", "Differentiate f(x)=x^3 and evaluate at x=2."),
        ("Integrate x^2 from 0 to 1.", "Integrate x^2 from 0 to 1."),
        ("Simplify (x^2-1)/(x-1).", "Simplify (x^2-1)/(x-1)."),
        ("Factor x^4-1.", "Factor x^4-1."),
        ("Factorize x^4-1.", "Factorize x^4-1."),
        ("Expand (x+1)^5.", "Expand (x+1)^5."),
        ("Approximate sqrt(2) to four decimals.", "Approximate sqrt(2) to four decimals."),
        ("Estimate the integral numerically.", "Estimate the integral numerically."),
        ("Maximize x+y subject to x^2+y^2<=1.", "Maximize x+y subject to x^2+y^2<=1."),
        ("Minimize x^2+y^2 subject to x+y=1.", "Minimize x^2+y^2 subject to x+y=1."),
        ("Optimize the objective over the feasible set.", "Optimize the objective over the feasible set."),
        ("Diagonalize A and compute A^10.", "Diagonalize A and compute A^10."),
        ("Diagonalise A over C.", "Diagonalise A over C."),
        ("Invert A.", "Invert A."),
        ("Normalize the vector v.", "Normalize the vector v."),
        ("Normalise the vector v.", "Normalise the vector v."),
        ("Parameterize the curve.", "Parameterize the curve."),
        ("Parametrize the surface.", "Parametrize the surface."),
    ],
)
def test_high_frequency_english_math_actions_create_answer_value_targets(problem, expected_span):
    agent = ReasoningAgent(RecordingClient())
    context = agent._score_first_context(problem, {})
    assert context["request_spans"] == [expected_span]
    assert context["response_mode"] == _SCORE_FIRST_RESPONSE_MODE_ANSWER


@pytest.mark.parametrize(
    "problem",
    [
        "求导 f(x)=x^3。",
        "积分 x^2。",
        "化简该表达式。",
        "因式分解 x^4-1。",
        "展开 (x+1)^5。",
        "近似计算 sqrt(2)。",
        "估计该积分。",
        "最大化目标函数。",
        "最小化目标函数。",
        "优化该目标。",
        "对角化矩阵 A。",
        "求逆矩阵 A。",
        "归一化向量 v。",
        "参数化该曲线。",
    ],
)
def test_high_frequency_chinese_math_actions_use_frozen_request_positions(problem):
    agent = ReasoningAgent(RecordingClient())
    context = agent._score_first_context(problem, {})
    assert context["request_spans"] == [problem]
    assert context["response_mode"] == _SCORE_FIRST_RESPONSE_MODE_ANSWER


@pytest.mark.parametrize(
    "problem",
    [
        "Why does this sequence converge uniformly?",
        "Why is the operator continuous?",
        "How does the rank-nullity theorem imply the result?",
        "How do the hypotheses imply the bound?",
        "How can we derive the recurrence?",
        "How is the formula obtained?",
        "How is the recurrence derived?",
        "为什么该级数收敛？",
        "为何该映射连续？",
        "如何推出该递推式？",
        "怎么得到这个上界？",
    ],
)
def test_explanation_interrogatives_request_derivation(problem):
    agent = ReasoningAgent(RecordingClient())
    context = agent._score_first_context(problem, {})
    assert context["request_spans"]
    assert context["response_mode"] == _SCORE_FIRST_RESPONSE_MODE_DERIVATION


@pytest.mark.parametrize(
    "problem",
    [
        "How many iterations are required?",
        "How much probability mass remains?",
        "How large is the error?",
        "How fast does the sequence decay?",
    ],
)
def test_quantitative_how_questions_remain_answer_value(problem):
    agent = ReasoningAgent(RecordingClient())
    context = agent._score_first_context(problem, {})
    assert context["response_mode"] == _SCORE_FIRST_RESPONSE_MODE_ANSWER


@pytest.mark.parametrize(
    "problem",
    [
        "Show your work when computing the determinant.",
        "Show the steps for solving the system.",
        "Give the calculation for the integral.",
        "Show the calculation for the variance.",
        "Explain your calculation.",
        "写出计算过程。",
        "给出计算过程。",
        "写出步骤。",
        "说明计算步骤。",
    ],
)
def test_explicit_show_work_requests_are_derivations(problem):
    agent = ReasoningAgent(RecordingClient())
    context = agent._score_first_context(problem, {})
    assert context["response_mode"] == _SCORE_FIRST_RESPONSE_MODE_DERIVATION


def test_declarative_show_protection_remains_green():
    agent = ReasoningAgent(RecordingClient())
    context = agent._score_first_context(
        "The data show that X is normal. Compute E[X^2].",
        {"subject": "Probability Theory"},
    )
    assert context["request_spans"] == ["Compute E[X^2]."]
    assert context["response_mode"] == _SCORE_FIRST_RESPONSE_MODE_ANSWER


def test_exactness_and_precision_discipline_is_in_system_prompt():
    client = RecordingClient()
    agent = ReasoningAgent(client)
    agent.solve("Compute sqrt(2).", {"subject": "Advanced Mathematics"})
    system_prompt = client.calls[0]["messages"][0]["content"]

    assert "prefer an exact mathematical form" in system_prompt
    assert "honor the requested precision" in system_prompt
    assert "Preserve units, domains, moduli, multiplicities" in system_prompt


def test_answer_value_verification_remains_internal_and_one_line_contract_survives():
    client = RecordingClient("Final answer: 1/7")
    agent = ReasoningAgent(client)
    result = agent.solve("Compute 1/7 exactly.", {"subject": "Advanced Mathematics"})

    assert result["final_response"] == "1/7"
    system_prompt = client.calls[0]["messages"][0]["content"]
    user_prompt = client.calls[0]["messages"][1]["content"]
    assert "Output exactly ONE visible line" in system_prompt
    assert "Do not print the check unless the problem asks for reasoning." in user_prompt
    assert user_prompt.count("Internal final check:") == 1


def test_prompt_budget_on_110_case_suite_stays_under_1900_chars():
    rows = _rows(ROUTING_FIXTURE)
    agent = ReasoningAgent(RecordingClient())
    lengths = []

    for row in rows:
        messages = agent._build_score_first_prompt(
            row["problem"],
            {"subject": _SCORE_FIRST_HUMAN_DOMAIN_LABELS[row["expected_domain"]], "task_type": row["task_type"]},
        )
        lengths.append(sum(len(message["content"]) for message in messages))

    assert len(lengths) == 110
    assert max(lengths) <= 1900


def test_100_successful_score_first_problems_equal_100_model_calls():
    client = RecordingClient()
    agent = ReasoningAgent(client)

    for index in range(100):
        result = agent.solve(
            f"Compute 6*7 for same-call verification regression {index}.",
            {"subject": "Advanced Mathematics", "task_type": "calculation"},
        )
        assert result["final_response"] == "42"

    assert len(client.calls) == 100
    assert all(call["temperature"] == 0.8 for call in client.calls)
    assert all(call["max_tokens"] == 32768 for call in client.calls)
    assert all(call["thinking_mode"] is True for call in client.calls)

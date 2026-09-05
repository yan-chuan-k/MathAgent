from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from user_agent import (
    ReasoningAgent,
    _SCORE_FIRST_HUMAN_DOMAIN_LABELS,
    _SCORE_FIRST_MICRO_FINAL_CHECKS,
    _SCORE_FIRST_MICRO_STRATEGIES,
    _SCORE_FIRST_RESPONSE_MODE_ANSWER,
    _SCORE_FIRST_RESPONSE_MODE_DERIVATION,
    _SCORE_FIRST_RESPONSE_MODE_PROOF,
)


ROOT = Path(__file__).resolve().parents[1]
ADVERSARIAL = ROOT / "sample_data" / "score_recovery_v27_proof_verification_adversarial.jsonl"
INVENTORY = ROOT / "sample_data" / "score_recovery_v27_micro_final_check_inventory.jsonl"
ROUTING = ROOT / "sample_data" / "score_recovery_v2_synthetic_hard.jsonl"


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


def test_v27_adversarial_fixture_has_zero_semantic_failures():
    agent = ReasoningAgent(RecordingClient())
    failures = []

    for row in _rows(ADVERSARIAL):
        context = agent._score_first_context(
            row["problem"],
            {"subject": row["subject"]},
        )
        card = context["verification_card"]

        if "expected_response_mode" in row and context["response_mode"] != row["expected_response_mode"]:
            failures.append(
                (row["idx"], "response_mode", row["expected_response_mode"], context["response_mode"])
            )
        if "expected_request_spans" in row and context["request_spans"] != row["expected_request_spans"]:
            failures.append(
                (row["idx"], "request_spans", row["expected_request_spans"], context["request_spans"])
            )
        if "expected_micro" in row and context["micro_strategy"] != row["expected_micro"]:
            failures.append(
                (row["idx"], "micro_strategy", row["expected_micro"], context["micro_strategy"])
            )
        if "expected_verification_key" in row and context["verification_key"] != row["expected_verification_key"]:
            failures.append(
                (row["idx"], "verification_key", row["expected_verification_key"], context["verification_key"])
            )
        if "expected_verification_variant" in row and context["verification_variant"] != row["expected_verification_variant"]:
            failures.append(
                (
                    row["idx"],
                    "verification_variant",
                    row["expected_verification_variant"],
                    context["verification_variant"],
                )
            )

        for phrase in row.get("required", []):
            if phrase.lower() not in card.lower():
                failures.append((row["idx"], "missing", phrase, card))
        for phrase in row.get("forbidden", []):
            if phrase.lower() in card.lower():
                failures.append((row["idx"], "forbidden", phrase, card))

    assert failures == []


def test_exact_frozen_v2_fixed_point_case_uses_fixed_point_safe_verification():
    routing = _rows(ROUTING)
    frozen = next(row for row in routing if row["idx"] == "v2syn_numerical_analysis_05")
    assert frozen["problem"] == (
        "For fixed-point iteration x_{n+1}=g(x_n), give the standard local derivative "
        "condition ensuring convergence near a fixed point."
    )

    agent = ReasoningAgent(RecordingClient())
    context = agent._score_first_context(
        frozen["problem"],
        {"subject": frozen["expected_domain"], "task_type": frozen["task_type"]},
    )

    assert context["micro_strategy"] == "newton_fixed_point"
    assert context["verification_variant"] == "fixed_point_convergence"
    card = context["verification_card"]
    assert "x*=g(x*)" in card
    assert "|g'(x*)|" in card
    assert "simple-root" not in card
    assert "nonzero-derivative" not in card


@pytest.mark.parametrize(
    "problem",
    [
        "Show it is a homeomorphism.",
        "Show the estimator is unbiased.",
        "Show f is continuous on [0,1].",
        "Establish convergence of the series.",
        "Demonstrate the map is injective.",
    ],
)
def test_request_position_show_establish_demonstrate_are_proof_intents(problem):
    agent = ReasoningAgent(RecordingClient())
    context = agent._score_first_context(problem, {})
    assert context["request_spans"]
    assert context["response_mode"] == _SCORE_FIRST_RESPONSE_MODE_PROOF


@pytest.mark.parametrize(
    "problem",
    [
        "Show your work.",
        "Show the calculation.",
        "Show the steps.",
    ],
)
def test_show_work_forms_remain_derivation(problem):
    agent = ReasoningAgent(RecordingClient())
    context = agent._score_first_context(problem, {})
    assert context["response_mode"] == _SCORE_FIRST_RESPONSE_MODE_DERIVATION


@pytest.mark.parametrize(
    "problem",
    [
        "Show the resulting matrix.",
        "Show the final expression.",
    ],
)
def test_display_like_show_forms_remain_answer_value(problem):
    agent = ReasoningAgent(RecordingClient())
    context = agent._score_first_context(problem, {})
    assert context["response_mode"] == _SCORE_FIRST_RESPONSE_MODE_ANSWER


@pytest.mark.parametrize(
    ("problem", "expected_span"),
    [
        ("The data show that X is normal. Compute E[X^2].", "Compute E[X^2]."),
        (
            "The results show that the approximation is stable. Evaluate the error.",
            "Evaluate the error.",
        ),
        (
            "The proof shows that f is measurable. Compute the integral.",
            "Compute the integral.",
        ),
    ],
)
def test_declarative_show_protection_remains_exact(problem, expected_span):
    agent = ReasoningAgent(RecordingClient())
    context = agent._score_first_context(problem, {})
    assert context["request_spans"] == [expected_span]
    assert context["response_mode"] == _SCORE_FIRST_RESPONSE_MODE_ANSWER


def test_newton_and_fixed_point_convergence_checks_are_distinct():
    agent = ReasoningAgent(RecordingClient())

    fixed = agent._score_first_context(
        "For x_{n+1}=cos(x_n), determine the local fixed-point convergence condition.",
        {"subject": "Numerical Analysis"},
    )
    newton = agent._score_first_context(
        "Determine the local convergence order of Newton's method.",
        {"subject": "Numerical Analysis"},
    )

    assert fixed["verification_variant"] == "fixed_point_convergence"
    assert "x*=g(x*)" in fixed["verification_card"]
    assert "simple-root" not in fixed["verification_card"]

    assert newton["verification_variant"] == "newton_convergence"
    assert "simple-root/nonzero-derivative" in newton["verification_card"]


def test_verification_paraphrases_select_target_safe_variants():
    agent = ReasoningAgent(RecordingClient())

    count = agent._score_first_context(
        "Find the mean number of fixed points.",
        {"subject": "Probability Theory"},
    )
    assert count["verification_variant"] == "indicator_count"

    covariance = agent._score_first_context(
        "Under full rank, derive the variance of beta_hat.",
        {"subject": "Linear Regression"},
    )
    assert covariance["verification_variant"] == "covariance"
    assert "Var(beta_hat)=A Var(y) A^T" in covariance["verification_card"]

    limiting = agent._score_first_context(
        "Derive the limiting distribution of the MLE.",
        {"subject": "Statistics"},
    )
    assert limiting["verification_variant"] == "mle_asymptotic"
    assert "Fisher information" in limiting["verification_card"]

    asymptotic_proof = agent._score_first_context(
        "Show asymptotic normality of the MLE.",
        {"subject": "Statistics"},
    )
    assert asymptotic_proof["response_mode"] == _SCORE_FIRST_RESPONSE_MODE_PROOF
    assert asymptotic_proof["verification_variant"] == "mle_asymptotic_proof"
    assert "Fisher information" in asymptotic_proof["verification_card"]


def test_high_risk_domain_fallback_replacements_are_method_appropriate():
    agent = ReasoningAgent(RecordingClient())

    clt = agent._score_first_context(
        "Use the CLT to approximate the probability.",
        {"subject": "Probability Theory"},
    )
    assert clt["micro_strategy"] == "limit_theorem"
    assert clt["verification_key"] == "micro:limit_theorem"
    assert "centering/scaling" in clt["verification_card"]

    poisson = agent._score_first_context(
        "For a Poisson process N(t) with rate lambda, compute P(N(t)=k).",
        {"subject": "Stochastic Processes"},
    )
    assert poisson["micro_strategy"] == "poisson_process"
    assert poisson["verification_key"] == "micro:poisson_process"
    assert "rate times interval length" in poisson["verification_card"]
    assert "first-step" not in poisson["verification_card"]

    brownian = agent._score_first_context(
        "For Brownian motion B_t, compute E[B_t^2].",
        {"subject": "Stochastic Processes"},
    )
    assert brownian["verification_variant"] == "brownian_moment"
    assert "Var(B_t)=t" in brownian["verification_card"]
    assert "first-step" not in brownian["verification_card"]
    assert "stationarity" not in brownian["verification_card"]

    zeros = agent._score_first_context(
        "Use the argument principle to count zeros inside the contour.",
        {"subject": "Complex Analysis"},
    )
    assert zeros["micro_strategy"] == "zeros_argument"
    assert zeros["verification_key"] == "micro:zeros_argument"
    assert "zeros minus poles" in zeros["verification_card"]

    equilibrium = agent._score_first_context(
        "Determine the stability of the equilibrium y*=0.",
        {"subject": "ODE"},
    )
    assert equilibrium["micro_strategy"] == "equilibrium_stability"
    assert equilibrium["verification_key"] == "micro:equilibrium_stability"
    assert "actual stability criterion" in equilibrium["verification_card"]


def test_bias_variance_sufficiency_uses_only_requested_branch():
    agent = ReasoningAgent(RecordingClient())

    bias = agent._score_first_context(
        "Compute the bias of the estimator T.",
        {"subject": "Statistics"},
    )
    assert bias["micro_strategy"] == "bias_variance_sufficiency"
    assert bias["verification_variant"] == "bias_variance"
    assert "E[T]" in bias["verification_card"]
    assert "factorization" not in bias["verification_card"].lower()

    suff = agent._score_first_context(
        "Determine whether T is a sufficient statistic using the factorization theorem.",
        {"subject": "Statistics"},
    )
    assert suff["micro_strategy"] == "bias_variance_sufficiency"
    assert suff["verification_variant"] == "sufficiency"
    assert "factorization" in suff["verification_card"].lower()
    assert "Var(T)" not in suff["verification_card"]


def test_v27_inventory_documents_all_30_v26_domain_fallback_families():
    inventory = _rows(INVENTORY)
    assert len(inventory) == 30
    assert len({(row["domain"], row["micro"]) for row in inventory}) == 30

    class_a = [row for row in inventory if row["classification"] == "A"]
    class_b = [row for row in inventory if row["classification"] == "B"]
    class_c = [row for row in inventory if row["classification"] == "C"]

    assert len(class_a) == 24
    assert len(class_b) == 6
    assert class_c == []

    methods = {
        (domain, entry[0])
        for domain, entries in _SCORE_FIRST_MICRO_STRATEGIES.items()
        for entry in entries
    }
    assert len(methods) == 56

    for row in class_b:
        assert row["micro"] in _SCORE_FIRST_MICRO_FINAL_CHECKS
        assert row["v27_action"] == "dedicated_or_target_variant"

    for row in class_a:
        assert row["micro"] not in _SCORE_FIRST_MICRO_FINAL_CHECKS
        assert row["v27_action"] == "keep_domain_fallback"


def test_v27_fixture_has_one_internal_check_and_one_call_per_problem():
    client = RecordingClient()
    agent = ReasoningAgent(client)

    for row in _rows(ADVERSARIAL):
        client.calls.clear()
        agent.solve(row["problem"], {"subject": row["subject"]})
        assert len(client.calls) == 1
        prompt = client.calls[0]["messages"][1]["content"]
        assert prompt.count("Internal final check:") == 1
        assert "Internal final check:\n" in prompt
        assert "\\n" not in agent._score_first_verification_block(
            agent._score_first_context(row["problem"], {"subject": row["subject"]})
        )


def test_frozen_110_prompt_budget_remains_at_most_1900_chars():
    agent = ReasoningAgent(RecordingClient())
    lengths = []

    for row in _rows(ROUTING):
        metadata = {
            "subject": _SCORE_FIRST_HUMAN_DOMAIN_LABELS[row["expected_domain"]],
            "task_type": row["task_type"],
        }
        messages = agent._build_score_first_prompt(row["problem"], metadata)
        lengths.append(sum(len(message["content"]) for message in messages))

    assert len(lengths) == 110
    assert max(lengths) <= 1900


def test_100_successful_score_first_problems_still_equal_100_model_calls():
    client = RecordingClient()
    agent = ReasoningAgent(client)

    for index in range(100):
        result = agent.solve(
            f"Compute 6*7 for V2.7 single-call regression {index}.",
            {"subject": "Advanced Mathematics", "task_type": "calculation"},
        )
        assert result["final_response"] == "42"

    assert len(client.calls) == 100
    assert all(call["temperature"] == 0.8 for call in client.calls)
    assert all(call["max_tokens"] == 32768 for call in client.calls)
    assert all(call["thinking_mode"] is True for call in client.calls)

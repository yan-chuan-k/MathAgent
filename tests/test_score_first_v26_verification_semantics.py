from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from user_agent import (
    ReasoningAgent,
    _SCORE_FIRST_HUMAN_DOMAIN_LABELS,
)


ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_FIXTURE = ROOT / "sample_data" / "score_recovery_v26_verification_semantics.jsonl"
ROUTING_FIXTURE = ROOT / "sample_data" / "score_recovery_v2_synthetic_hard.jsonl"
V25_FIXTURE = ROOT / "sample_data" / "score_recovery_v25_same_call_verification.jsonl"


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


def test_verification_block_uses_real_newlines_and_required_shape():
    agent = ReasoningAgent(RecordingClient())
    context = agent._score_first_context(
        "Compute E[X^2].",
        {"subject": "Probability Theory"},
    )
    block = agent._score_first_verification_block(context)

    assert block.startswith("Internal final check:\n")
    assert "\\n" not in block
    assert block.count("Internal final check:") == 1
    assert "\nDo this silently; correct any failure before answering.\n" in block
    assert block.endswith(
        "Do not print the check unless the problem asks for reasoning.\n\n"
    )


@pytest.mark.parametrize(
    ("problem", "subject", "expected_completion_text"),
    [
        (
            "Compute 2+2.",
            "Advanced Mathematics",
            "all requested parts answered",
        ),
        (
            "Derive the recurrence formula.",
            "Discrete Mathematics",
            "every requested part has an answer",
        ),
        (
            "Prove that a continuous image of a compact space is compact.",
            "Topology",
            "every requested part has an answer",
        ),
        (
            "Prove or disprove: every bounded operator is compact.",
            "Functional Analysis",
            "every requested part has an answer",
        ),
        (
            "Construct a counterexample to the converse.",
            "Real Analysis",
            "every requested part has an answer",
        ),
    ],
)
def test_every_score_first_mode_has_universal_completeness_invariant(
    problem,
    subject,
    expected_completion_text,
):
    client = RecordingClient()
    agent = ReasoningAgent(client)
    agent.solve(problem, {"subject": subject})
    system_prompt = client.calls[0]["messages"][0]["content"]

    assert expected_completion_text in system_prompt
    assert "constraints/conditions" in system_prompt


def test_v26_semantic_fixture_has_zero_wrong_or_misleading_checks():
    agent = ReasoningAgent(RecordingClient())
    failures = []

    for row in _rows(SEMANTIC_FIXTURE):
        metadata = {"subject": row["subject"]}
        context = agent._score_first_context(row["problem"], metadata)
        card = context["verification_card"]

        if context["verification_key"] != row["expected_verification_key"]:
            failures.append(
                (
                    row["idx"],
                    "verification_key",
                    row["expected_verification_key"],
                    context["verification_key"],
                )
            )
        if context["verification_variant"] != row["expected_verification_variant"]:
            failures.append(
                (
                    row["idx"],
                    "verification_variant",
                    row["expected_verification_variant"],
                    context["verification_variant"],
                )
            )

        for phrase in row.get("required_verification_phrases", []):
            if phrase.lower() not in card.lower():
                failures.append((row["idx"], "missing", phrase, card))

        for phrase in row.get("forbidden_verification_phrases", []):
            if phrase.lower() in card.lower():
                failures.append((row["idx"], "forbidden", phrase, card))

    assert failures == []


def test_v26_fixture_expected_fields_never_enter_solver_metadata_or_prompt():
    client = RecordingClient()
    agent = ReasoningAgent(client)

    for row in _rows(SEMANTIC_FIXTURE):
        client.calls.clear()
        metadata = {"subject": row["subject"]}
        assert "expected_verification_key" not in metadata
        assert "expected_verification_variant" not in metadata
        agent.solve(row["problem"], metadata)

        prompt = "\n".join(
            message["content"]
            for message in client.calls[0]["messages"]
        )
        assert "expected_verification_key" not in prompt
        assert "expected_verification_variant" not in prompt
        assert "required_verification_phrases" not in prompt
        assert "forbidden_verification_phrases" not in prompt


def test_v26_each_prompt_has_exactly_one_verification_block_and_one_model_call():
    client = RecordingClient()
    agent = ReasoningAgent(client)

    for row in _rows(SEMANTIC_FIXTURE):
        client.calls.clear()
        agent.solve(row["problem"], {"subject": row["subject"]})
        assert len(client.calls) == 1

        user_prompt = client.calls[0]["messages"][1]["content"]
        assert user_prompt.count("Internal final check:") == 1
        assert "\n\nProblem:\n" in user_prompt
        assert "Internal final check:\n" in user_prompt


def test_general_moment_checks_do_not_force_indicator_verification():
    agent = ReasoningAgent(RecordingClient())

    for problem in (
        "For X~N(0,1), compute E[X^4].",
        "For X~Poisson(lambda), compute Var(X).",
    ):
        context = agent._score_first_context(
            problem,
            {"subject": "Probability Theory"},
        )
        assert context["verification_variant"] == "general_moment"
        card = context["verification_card"].lower()
        assert "pmf/pdf" in card
        assert "indicator event" not in card
        assert "success probability" not in card


def test_indicator_count_target_keeps_indicator_specific_check():
    agent = ReasoningAgent(RecordingClient())
    context = agent._score_first_context(
        "Compute the expected number of fixed points using indicator variables.",
        {"subject": "Probability Theory"},
    )
    assert context["verification_variant"] == "indicator_count"
    card = context["verification_card"].lower()
    assert "indicator event" in card
    assert "independence is not required" in card


def test_ols_estimator_and_covariance_receive_different_variants_same_family():
    agent = ReasoningAgent(RecordingClient())

    estimator = agent._score_first_context(
        "Under full column rank, derive beta_hat.",
        {"subject": "Linear Regression"},
    )
    covariance = agent._score_first_context(
        "Under full column rank, derive Var(beta_hat).",
        {"subject": "Linear Regression"},
    )

    assert estimator["verification_key"] == "micro:ols_full_rank"
    assert covariance["verification_key"] == "micro:ols_full_rank"
    assert estimator["verification_variant"] == "base"
    assert covariance["verification_variant"] == "covariance"
    assert "X^T(y-X beta_hat)=0" in estimator["verification_card"]
    assert "Var(beta_hat)=A Var(y) A^T" in covariance["verification_card"]


def test_newton_iteration_and_convergence_proof_receive_target_safe_checks():
    agent = ReasoningAgent(RecordingClient())

    iterate = agent._score_first_context(
        "Compute one Newton iterate for f(x)=0 from the supplied x_0.",
        {"subject": "Numerical Analysis"},
    )
    proof = agent._score_first_context(
        "Prove Newton's method is quadratically convergent near a simple root.",
        {"subject": "Numerical Analysis"},
    )

    assert iterate["verification_key"] == "micro:newton_fixed_point"
    assert iterate["verification_variant"] == "base"
    assert "iterate index" in iterate["verification_card"]

    assert proof["verification_key"] == "task:proof"
    assert proof["verification_variant"] == "newton_convergence_proof"
    assert "simple-root/nonzero-derivative" in proof["verification_card"]
    assert "local error relation" in proof["verification_card"]
    assert "iterate index was not shifted" not in proof["verification_card"]


def test_mle_estimation_and_asymptotic_targets_receive_different_checks():
    agent = ReasoningAgent(RecordingClient())

    mle = agent._score_first_context(
        "Derive the MLE for theta.",
        {"subject": "Statistics"},
    )
    asymptotic = agent._score_first_context(
        "Derive the asymptotic variance of the MLE theta_hat.",
        {"subject": "Statistics"},
    )
    multi = agent._score_first_context(
        "Compute the MLE and then derive its asymptotic variance.",
        {"subject": "Statistics"},
    )

    assert mle["verification_variant"] == "base"
    assert "boundary" in mle["verification_card"]

    assert asymptotic["verification_variant"] == "mle_asymptotic"
    assert "Fisher information" in asymptotic["verification_card"]
    assert "n I(theta)" in asymptotic["verification_card"]

    assert multi["verification_variant"] == "mle_asymptotic"
    assert "Fisher information" in multi["verification_card"]


def test_operator_invertibility_check_is_target_safe_for_T_and_I_minus_T():
    agent = ReasoningAgent(RecordingClient())

    for problem in (
        "Determine whether T is invertible.",
        "Determine whether I-T is invertible.",
    ):
        context = agent._score_first_context(
            problem,
            {"subject": "Functional Analysis"},
        )
        assert context["verification_key"] == "micro:spectrum_invertibility"
        assert context["verification_variant"] == "operator_invertibility"
        card = context["verification_card"]
        assert "exact operator named in the target" in card
        assert "lambda I-T" not in card


def test_proof_disproof_and_counterexample_checks_dominate_computational_micro():
    agent = ReasoningAgent(RecordingClient())

    proof = agent._score_first_context(
        "Prove or disprove: every compact operator on an infinite-dimensional Banach space is invertible.",
        {"subject": "Functional Analysis"},
    )
    construction = agent._score_first_context(
        "Construct a counterexample to the converse, satisfying all stated hypotheses.",
        {"subject": "Real Analysis"},
    )

    assert proof["verification_key"] == "task:proof_or_disproof"
    assert "truth value" in proof["verification_card"]
    assert "whole claim" in proof["verification_card"]

    assert construction["verification_key"] == "task:construction_counterexample"
    assert "every requested property" in construction["verification_card"]
    assert "all hypotheses" in construction["verification_card"]


def test_statistical_interval_construction_keeps_inference_micro_check():
    agent = ReasoningAgent(RecordingClient())

    confidence = agent._score_first_context(
        "Construct a 95% confidence interval using the t distribution.",
        {"subject": "Statistics"},
    )
    prediction = agent._score_first_context(
        "Construct a prediction interval for a new observation at x0.",
        {"subject": "Linear Regression"},
    )

    assert confidence["verification_key"] == "micro:hypothesis_test_ci"
    assert prediction["verification_key"] == "micro:sampling_inference"


def test_v25_verification_fixture_high_level_keys_remain_stable():
    agent = ReasoningAgent(RecordingClient())
    failures = []

    for row in _rows(V25_FIXTURE):
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

    assert failures == []


def test_frozen_110_prompt_budget_remains_at_most_1900_chars():
    agent = ReasoningAgent(RecordingClient())
    lengths = []

    for row in _rows(ROUTING_FIXTURE):
        metadata = {
            "subject": _SCORE_FIRST_HUMAN_DOMAIN_LABELS[row["expected_domain"]],
            "task_type": row["task_type"],
        }
        messages = agent._build_score_first_prompt(row["problem"], metadata)
        lengths.append(sum(len(message["content"]) for message in messages))

    assert len(lengths) == 110
    assert max(lengths) <= 1900


def test_same_call_verification_still_uses_one_call_per_problem():
    client = RecordingClient()
    agent = ReasoningAgent(client)

    for index in range(100):
        result = agent.solve(
            f"Compute 6*7 for V2.6 verification regression {index}.",
            {"subject": "Advanced Mathematics", "task_type": "calculation"},
        )
        assert result["final_response"] == "42"

    assert len(client.calls) == 100
    assert all(call["temperature"] == 0.8 for call in client.calls)
    assert all(call["max_tokens"] == 32768 for call in client.calls)
    assert all(call["thinking_mode"] is True for call in client.calls)

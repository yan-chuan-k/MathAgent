from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from user_agent import (
    ReasoningAgent,
    _SCORE_FIRST_HUMAN_DOMAIN_LABELS,
    _SCORE_FIRST_RESPONSE_MODE_ANSWER,
    _SCORE_FIRST_RESPONSE_MODE_CONSTRUCTION,
    _SCORE_FIRST_RESPONSE_MODE_DERIVATION,
    _SCORE_FIRST_RESPONSE_MODE_PROOF,
    _SCORE_FIRST_RESPONSE_MODE_PROOF_OR_DISPROOF,
)


ROOT = Path(__file__).resolve().parents[1]
INTENT_FIXTURE = ROOT / "sample_data" / "score_recovery_v23_request_intent_adversarial.jsonl"
TARGET_FIXTURE = ROOT / "sample_data" / "score_recovery_v23_target_micro.jsonl"
V22_ARBITRATION_FIXTURE = ROOT / "sample_data" / "score_recovery_v22_micro_arbitration.jsonl"
STRESS_FIXTURE = ROOT / "sample_data" / "score_recovery_v21_hidden_style_stress.jsonl"
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
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.mark.parametrize(
    ("problem", "expected_spans"),
    [
        (
            "Let T be a compact operator. Determine whether I-T is invertible.",
            ["Determine whether I-T is invertible."],
        ),
        (
            "Assume a normal model. Find the bias and variance of the MLE.",
            ["Find the bias and variance of the MLE."],
        ),
        (
            "Compute E[X^2] for X~N(0,1).",
            ["Compute E[X^2] for X~N(0,1)."],
        ),
        (
            "无需给出证明，只需说明使用哪个定理。",
            ["说明使用哪个定理。"],
        ),
    ],
)
def test_request_clause_extraction_separates_target_from_background(problem, expected_spans):
    agent = ReasoningAgent(RecordingClient())
    assert agent._score_first_request_spans(problem) == expected_spans


def test_context_text_keeps_background_outside_request_target():
    agent = ReasoningAgent(RecordingClient())
    context = agent._score_first_context(
        "Let T be a compact operator on a Banach space. Determine whether I-T is invertible.",
        {"subject": "Functional Analysis"},
    )
    assert context["target_text"] == "Determine whether I-T is invertible."
    assert "compact operator" in context["context_text"]
    assert "invertible" not in context["context_text"]


def test_router_reasoning_label_alone_cannot_force_visible_reasoning():
    agent = ReasoningAgent(RecordingClient())
    fake_route = {"task_type": "proof"}
    assert (
        agent._score_first_response_mode(
            "The proof uses dominated convergence. Which hypothesis fails?",
            {},
            fake_route,
        )
        == _SCORE_FIRST_RESPONSE_MODE_ANSWER
    )

    fake_route = {"task_type": "derivation"}
    assert (
        agent._score_first_response_mode(
            "The derivation above is irrelevant; compute the value at x=2.",
            {},
            fake_route,
        )
        == _SCORE_FIRST_RESPONSE_MODE_ANSWER
    )

    fake_route = {"task_type": "construction"}
    assert (
        agent._score_first_response_mode(
            "Constructing the interpolant is unnecessary; find its value at x=1/2.",
            {},
            fake_route,
        )
        == _SCORE_FIRST_RESPONSE_MODE_ANSWER
    )


@pytest.mark.parametrize(
    ("problem", "expected"),
    [
        ("Explain which option is correct.", _SCORE_FIRST_RESPONSE_MODE_DERIVATION),
        ("Which option is correct? Explain why.", _SCORE_FIRST_RESPONSE_MODE_DERIVATION),
        ("请选择正确选项并说明理由。", _SCORE_FIRST_RESPONSE_MODE_DERIVATION),
        ("以下哪个结论正确？请解释原因。", _SCORE_FIRST_RESPONSE_MODE_DERIVATION),
        ("Which of the following is a counterexample?", _SCORE_FIRST_RESPONSE_MODE_ANSWER),
    ],
)
def test_explanation_request_upgrades_choice_but_choice_only_stays_answer(problem, expected):
    agent = ReasoningAgent(RecordingClient())
    assert agent._score_first_context(problem, {})["response_mode"] == expected


def test_request_intent_adversarial_fixture_has_zero_failures():
    agent = ReasoningAgent(RecordingClient())
    failures = []
    for row in _rows(INTENT_FIXTURE):
        got = agent._score_first_context(row["problem"], {})["response_mode"]
        if got != row["expected_mode"]:
            failures.append((row["idx"], row["expected_mode"], got))
    assert failures == []


def test_proof_or_disproof_has_dedicated_prompt_and_preserves_full_response():
    response = (
        "False.\n"
        "Counterexample: the identity on an infinite-dimensional Banach space is bounded but not compact."
    )
    client = RecordingClient(response)
    agent = ReasoningAgent(client)
    result = agent.solve(
        "Prove or disprove: every bounded operator on a Banach space is compact.",
        {"subject": "Functional Analysis"},
    )
    assert result["final_response"] == response
    assert len(client.calls) == 1
    system_prompt = client.calls[0]["messages"][0]["content"]
    assert "If true, give a concise complete proof." in system_prompt
    assert "If false, give a concise disproof or counterexample" in system_prompt


def test_target_aware_micro_matrix_has_zero_wrong_cards():
    agent = ReasoningAgent(RecordingClient())
    wrong = []
    for row in _rows(TARGET_FIXTURE):
        context = agent._score_first_context(row["problem"], {"subject": row["domain"]})
        got = context["micro_strategy"]
        if got != row["expected_micro"]:
            wrong.append((row["idx"], row["expected_micro"], got))
    assert wrong == []


def test_background_only_method_does_not_create_micro_without_target_link():
    agent = ReasoningAgent(RecordingClient())
    context = agent._score_first_context(
        "Let T be a compact operator. Compute 2+2.",
        {"subject": "Functional Analysis"},
    )
    assert context["micro_strategy"] is None
    assert context["micro_card"] is None


def test_target_evidence_dominates_background_method():
    agent = ReasoningAgent(RecordingClient())
    context = agent._score_first_context(
        "For the heat equation on R^n, find its fundamental solution.",
        {"subject": "PDE"},
    )
    assert context["micro_strategy"] == "transform_fundamental_solution"
    assert context["micro_match"]["target_strong"] is True
    assert context["micro_match"]["context_strong"] is False


def test_regex_synonyms_inside_one_evidence_group_count_once():
    agent = ReasoningAgent(RecordingClient())
    context = agent._score_first_context(
        "Use KKT (Karush-Kuhn-Tucker) conditions and KKT complementary slackness to solve the problem.",
        {"subject": "Optimization"},
    )
    assert context["micro_strategy"] == "kkt_convex"
    # Several synonymous strong regexes can match, but the target strong group is
    # a single mathematical item and contributes exactly 8 once.
    assert context["micro_match"]["score"] == 8


def _audit_expected_micro(path: Path):
    agent = ReasoningAgent(RecordingClient())
    clear = selected = correct = 0
    wrong = []
    missing = []
    for row in _rows(path):
        if "expected_micro" not in row:
            continue
        expected = row["expected_micro"]
        got = agent._score_first_context(
            row["problem"],
            {"subject": row.get("domain") or row.get("expected_domain")},
        )["micro_strategy"]

        if expected is None:
            if got is not None:
                wrong.append((row["idx"], expected, got))
            continue

        clear += 1
        if got is not None:
            selected += 1
        if got == expected:
            correct += 1
        elif got is None:
            missing.append((row["idx"], expected))
        else:
            wrong.append((row["idx"], expected, got))
    return clear, selected, correct, wrong, missing


def test_v22_dedicated_arbitration_precision_is_preserved():
    clear, selected, correct, wrong, missing = _audit_expected_micro(V22_ARBITRATION_FIXTURE)
    assert clear == 68
    assert selected == 68
    assert correct == 68
    assert wrong == []
    assert missing == []


def test_hidden_style_method_clear_reaches_at_least_90_percent_with_zero_wrong():
    clear, selected, correct, wrong, missing = _audit_expected_micro(STRESS_FIXTURE)
    assert clear == 48
    assert wrong == []
    assert correct / clear >= 0.90
    assert correct == 48
    assert missing == []


def test_canonical_and_human_subject_routing_remain_110_of_110():
    rows = _rows(ROUTING_FIXTURE)
    agent = ReasoningAgent(RecordingClient())
    canonical = human = 0
    for row in rows:
        expected = row["expected_domain"]
        canonical += int(
            agent._score_first_context(
                row["problem"],
                {"subject": expected, "task_type": row["task_type"]},
            )["strategy_domain"]
            == expected
        )
        human += int(
            agent._score_first_context(
                row["problem"],
                {
                    "subject": _SCORE_FIRST_HUMAN_DOMAIN_LABELS[expected],
                    "task_type": row["task_type"],
                },
            )["strategy_domain"]
            == expected
        )
    assert canonical == 110
    assert human == 110


def test_no_subject_specialized_precision_remains_100_percent():
    rows = _rows(ROUTING_FIXTURE)
    agent = ReasoningAgent(RecordingClient())
    selected = correct = 0
    for row in rows:
        context = agent._score_first_context(
            row["problem"],
            {"task_type": row["task_type"]},
        )
        if context["strategy_is_specialized"]:
            selected += 1
            correct += int(context["strategy_domain"] == row["expected_domain"])
    assert selected == 11
    assert correct == 11


def test_score_first_call_budget_stays_one_per_problem():
    client = RecordingClient("Final answer: 42")
    agent = ReasoningAgent(client)
    for index in range(100):
        result = agent.solve(
            f"Compute 6*7 for regression case {index}.",
            {"subject": "Advanced Mathematics", "task_type": "calculation"},
        )
        assert result["final_response"] == "42"
    assert len(client.calls) == 100
    assert all(call["temperature"] == 0.1 for call in client.calls)
    assert all(call["max_tokens"] == 8192 for call in client.calls)
    assert all(call["thinking_mode"] is True for call in client.calls)


def test_v23_fixtures_are_nonempty_and_expected_labels_are_test_side_only():
    assert len(_rows(INTENT_FIXTURE)) >= 10
    assert len(_rows(TARGET_FIXTURE)) >= 10
    for row in _rows(TARGET_FIXTURE):
        solver_metadata = {"subject": row["domain"]}
        assert "expected_micro" not in solver_metadata
    for row in _rows(INTENT_FIXTURE):
        solver_metadata = {}
        assert "expected_mode" not in solver_metadata

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from user_agent import (
    ReasoningAgent,
    _SCORE_FIRST_HUMAN_DOMAIN_LABELS,
    _SCORE_FIRST_MICRO_STRATEGIES,
    _SCORE_FIRST_RESPONSE_MODE_ANSWER,
    _SCORE_FIRST_RESPONSE_MODE_CONSTRUCTION,
    _SCORE_FIRST_RESPONSE_MODE_DERIVATION,
    _SCORE_FIRST_RESPONSE_MODE_PROOF,
)


ROOT = Path(__file__).resolve().parents[1]
ROUTING_FIXTURE = ROOT / "sample_data" / "score_recovery_v2_synthetic_hard.jsonl"
STRESS_FIXTURE = ROOT / "sample_data" / "score_recovery_v21_hidden_style_stress.jsonl"
ARBITRATION_FIXTURE = ROOT / "sample_data" / "score_recovery_v22_micro_arbitration.jsonl"


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


@pytest.mark.parametrize(
    "problem",
    [
        "No proof is required; compute the determinant.",
        "You need not prove the theorem; just evaluate the integral.",
        "The proof uses dominated convergence. Which hypothesis fails?",
        "Do not construct the function explicitly; determine whether one exists.",
        "Which of the following is a counterexample?",
        "Rather than derive the formula, evaluate it at x=2.",
        "不必证明该定理，只需计算积分值。",
        "无需构造具体函数，只判断是否存在。",
        "不要求推导公式，只求 x=2 时的数值。",
        "在乘积测度空间上给定非负可测函数 g(x,y)。无需先证明绝对可积即可交换累次积分，应使用哪个定理？",
    ],
)
def test_negative_or_incidental_action_mentions_do_not_request_reasoning_mode(problem):
    agent = ReasoningAgent(RecordingClient())
    context = agent._score_first_context(problem, {})
    assert context["response_mode"] == _SCORE_FIRST_RESPONSE_MODE_ANSWER


@pytest.mark.parametrize(
    ("problem", "expected"),
    [
        ("Prove that every finite tree has n-1 edges.", _SCORE_FIRST_RESPONSE_MODE_PROOF),
        ("Provide a proof that the map is continuous.", _SCORE_FIRST_RESPONSE_MODE_PROOF),
        ("证明连续映射下紧致集的像仍紧致。", _SCORE_FIRST_RESPONSE_MODE_PROOF),
        ("Derive the OLS covariance formula.", _SCORE_FIRST_RESPONSE_MODE_DERIVATION),
        ("Explain why the normal equations are non-unique.", _SCORE_FIRST_RESPONSE_MODE_DERIVATION),
        ("Compute phi(36), but first factor 36 and justify the multiplicative formula you use.", _SCORE_FIRST_RESPONSE_MODE_DERIVATION),
        ("指出保留事件数的分布并说明理由。", _SCORE_FIRST_RESPONSE_MODE_DERIVATION),
        ("Construct a connected graph that is not Eulerian.", _SCORE_FIRST_RESPONSE_MODE_CONSTRUCTION),
        ("Give a counterexample to the converse.", _SCORE_FIRST_RESPONSE_MODE_CONSTRUCTION),
        ("给出一个反例说明该逆命题不成立。", _SCORE_FIRST_RESPONSE_MODE_CONSTRUCTION),
        ("Which of the following is a counterexample?", _SCORE_FIRST_RESPONSE_MODE_ANSWER),
    ],
)
def test_strong_request_intent_classifier(problem, expected):
    agent = ReasoningAgent(RecordingClient())
    assert agent._score_first_explicit_response_intent(problem) == expected


def test_strong_explicit_request_overrides_incorrect_task_metadata():
    agent = ReasoningAgent(RecordingClient())
    proof = agent._score_first_context(
        "Prove that every compact subset of a Hausdorff space is closed.",
        {"task_type": "calculation", "subject": "Topology"},
    )
    assert proof["response_mode"] == _SCORE_FIRST_RESPONSE_MODE_PROOF

    derivation = agent._score_first_context(
        "Derive the covariance of the OLS estimator.",
        {"task_type": "calculation", "subject": "Linear Regression"},
    )
    assert derivation["response_mode"] == _SCORE_FIRST_RESPONSE_MODE_DERIVATION


def test_72_stress_cases_are_mode_compatible_with_trusted_task_metadata():
    mapping = {
        "calculation": _SCORE_FIRST_RESPONSE_MODE_ANSWER,
        "choice": _SCORE_FIRST_RESPONSE_MODE_ANSWER,
        "derivation": _SCORE_FIRST_RESPONSE_MODE_DERIVATION,
        "proof": _SCORE_FIRST_RESPONSE_MODE_PROOF,
        "construction": _SCORE_FIRST_RESPONSE_MODE_CONSTRUCTION,
        "counterexample": _SCORE_FIRST_RESPONSE_MODE_CONSTRUCTION,
    }
    rows = [json.loads(line) for line in STRESS_FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 72

    agent = ReasoningAgent(RecordingClient())
    mismatches = []
    for row in rows:
        context = agent._score_first_context(
            row["problem"],
            {"subject": row["subject"], "task_type": row["task_type"]},
        )
        explicit = agent._score_first_explicit_response_intent(row["problem"])
        expected = explicit or mapping[row["task_type"]]
        if context["response_mode"] != expected:
            mismatches.append((row["idx"], expected, context["response_mode"]))
    assert mismatches == []


@pytest.mark.parametrize(
    "idx",
    ["v21stress_007", "v21stress_013", "v21stress_036", "v21stress_040", "v21stress_065"],
)
def test_subject_only_clear_explanation_requests_are_derivations(idx):
    rows = [json.loads(line) for line in STRESS_FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]
    row = next(item for item in rows if item["idx"] == idx)
    agent = ReasoningAgent(RecordingClient())
    context = agent._score_first_context(row["problem"], {"subject": row["subject"]})
    assert context["response_mode"] == _SCORE_FIRST_RESPONSE_MODE_DERIVATION


def test_score_first_subject_hint_is_emitted_only_for_trusted_domain_metadata():
    client = RecordingClient()
    agent = ReasoningAgent(client)

    agent.solve("Compute 1+1.", {"subject": "Probability Theory"})
    prompt = client.calls[-1]["messages"][1]["content"]
    assert "Subject hint: Probability Theory" in prompt

    agent.solve("Compute 1+1.", {"subject": "probability"})
    prompt = client.calls[-1]["messages"][1]["content"]
    assert "Subject hint: Probability Theory" in prompt

    agent.solve("Compute 1+1.", {"type": "derivation"})
    prompt = client.calls[-1]["messages"][1]["content"]
    assert "Subject hint:" not in prompt

    agent.solve("Compute 1+1.", {"category": "hard"})
    prompt = client.calls[-1]["messages"][1]["content"]
    assert "Subject hint:" not in prompt


def test_110_subject_views_and_no_subject_margin_regression():
    rows = [json.loads(line) for line in ROUTING_FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]
    agent = ReasoningAgent(RecordingClient())

    canonical = 0
    human = 0
    specialized = 0
    specialized_correct = 0
    fallback = 0

    for row in rows:
        expected = row["expected_domain"]

        canonical_context = agent._score_first_context(
            row["problem"],
            {"subject": expected, "task_type": row["task_type"]},
        )
        canonical += int(canonical_context["strategy_domain"] == expected)

        human_context = agent._score_first_context(
            row["problem"],
            {
                "subject": _SCORE_FIRST_HUMAN_DOMAIN_LABELS[expected],
                "task_type": row["task_type"],
            },
        )
        human += int(human_context["strategy_domain"] == expected)

        no_subject = agent._score_first_context(
            row["problem"],
            {"task_type": row["task_type"]},
        )
        if no_subject["strategy_is_specialized"]:
            specialized += 1
            specialized_correct += int(no_subject["strategy_domain"] == expected)
            assert no_subject["route_top_score"] >= 5.5
            assert no_subject["route_margin"] >= 4.0
        else:
            fallback += 1

    assert canonical == 110
    assert human == 110
    assert specialized == 11
    assert specialized_correct == 11
    assert fallback == 99


def test_cross_domain_hybrid_without_subject_falls_back_when_margin_is_small():
    agent = ReasoningAgent(RecordingClient())
    context = agent._score_first_context(
        "A random graph G(n,p) is chosen. Compute the probability that it is connected.",
        {},
    )
    assert context["strategy_domain"] == "advanced_math"
    assert context["strategy_is_specialized"] is False
    assert context["route_margin"] < 4.0


def _audit_micro_rows(rows):
    agent = ReasoningAgent(RecordingClient())
    selected = 0
    correct = 0
    wrong = []
    missing = []
    clear = 0

    for row in rows:
        if "expected_micro" not in row:
            continue
        expected = row["expected_micro"]
        context = agent._score_first_context(
            row["problem"],
            {"subject": row.get("domain") or row.get("expected_domain")},
        )
        got = context["micro_strategy"]

        if got is not None:
            selected += 1

        if expected is None:
            if got is not None:
                wrong.append((row["idx"], expected, got))
            continue

        clear += 1
        if got == expected:
            correct += 1
        elif got is None:
            missing.append((row["idx"], expected))
        else:
            wrong.append((row["idx"], expected, got))

    return {
        "selected": selected,
        "correct": correct,
        "wrong": wrong,
        "missing": missing,
        "clear": clear,
    }


def test_dedicated_micro_arbitration_has_zero_wrong_cards():
    rows = [json.loads(line) for line in ARBITRATION_FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]
    audit = _audit_micro_rows(rows)

    assert len(rows) >= 40
    assert audit["wrong"] == []
    assert audit["clear"] >= 60
    assert audit["correct"] == audit["clear"]
    assert audit["missing"] == []


@pytest.mark.parametrize(
    ("idx", "expected"),
    [
        ("micro_num_01", "interpolation"),
        ("micro_num_03", "stability_convergence"),
        ("micro_meas_01", "limit_integral"),
        ("micro_alg_01", "finite_field_galois"),
        ("micro_stoch_02", "markov_hitting"),
        ("micro_cx_02", "cauchy_formula"),
        ("micro_cx_04", "laurent_singularity"),
    ],
)
def test_known_micro_collisions_are_arbitrated_correctly(idx, expected):
    rows = [json.loads(line) for line in ARBITRATION_FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]
    row = next(item for item in rows if item["idx"] == idx)
    agent = ReasoningAgent(RecordingClient())
    context = agent._score_first_context(row["problem"], {"subject": row["domain"]})
    assert context["micro_strategy"] == expected
    assert context["micro_match"]["score"] >= 4
    assert context["micro_match"]["margin"] >= 2


def test_ambiguous_micro_cases_prefer_no_hint():
    rows = [json.loads(line) for line in ARBITRATION_FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]
    ambiguous = [row for row in rows if row.get("expected_micro", "missing") is None]
    assert ambiguous
    agent = ReasoningAgent(RecordingClient())
    for row in ambiguous:
        context = agent._score_first_context(row["problem"], {"subject": row["domain"]})
        assert context["micro_strategy"] is None, row["idx"]
        assert context["micro_card"] is None, row["idx"]


def test_stress_method_clear_subset_has_zero_wrong_and_at_least_70_percent_coverage():
    rows = [json.loads(line) for line in STRESS_FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]
    annotated = [row for row in rows if "expected_micro" in row]
    assert len(annotated) >= 36

    audit = _audit_micro_rows(annotated)
    assert audit["wrong"] == []
    assert audit["clear"] == len(annotated)
    assert audit["selected"] == audit["correct"]
    assert audit["correct"] / audit["clear"] >= 0.70


def test_scored_micro_rules_use_strong_and_weak_pattern_sets():
    assert len(_SCORE_FIRST_MICRO_STRATEGIES) >= 16
    for domain, entries in _SCORE_FIRST_MICRO_STRATEGIES.items():
        assert entries, domain
        for entry in entries:
            assert len(entry) == 4
            name, strong, weak, card = entry
            assert name
            assert strong
            assert isinstance(strong, tuple)
            assert isinstance(weak, tuple)
            assert card


def test_v22_preserves_one_model_call_per_successful_score_first_problem():
    client = RecordingClient("Final answer: 42")
    agent = ReasoningAgent(client)
    for i in range(100):
        result = agent.solve(
            f"Compute 6*7 for case {i}.",
            {"subject": "Advanced Mathematics", "task_type": "calculation"},
        )
        assert result["final_response"] == "42"
    assert len(client.calls) == 100
    assert all(call["max_tokens"] == 8192 for call in client.calls)
    assert all(call["temperature"] == 0.1 for call in client.calls)
    assert all(call["thinking_mode"] is True for call in client.calls)

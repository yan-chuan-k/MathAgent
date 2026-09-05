from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from user_agent import (
    ReasoningAgent,
    _SCORE_FIRST_RESPONSE_MODE_ANSWER,
    _SCORE_FIRST_RESPONSE_MODE_DERIVATION,
    _SCORE_FIRST_RESPONSE_MODE_PROOF,
)


ROOT = Path(__file__).resolve().parents[1]
PARSER_FIXTURE = ROOT / "sample_data" / "score_recovery_v24_request_parser_adversarial.jsonl"


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


def _rows():
    return [
        json.loads(line)
        for line in PARSER_FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_v24_parser_adversarial_fixture_has_zero_semantic_failures():
    agent = ReasoningAgent(RecordingClient())
    failures = []

    for row in _rows():
        metadata = {"subject": row["subject"]}
        if row.get("task_type"):
            metadata["task_type"] = row["task_type"]

        context = agent._score_first_context(row["problem"], metadata)

        if context["request_spans"] != row["expected_spans"]:
            failures.append(
                (
                    row["idx"],
                    "request_spans",
                    row["expected_spans"],
                    context["request_spans"],
                )
            )
        if context["response_mode"] != row["expected_mode"]:
            failures.append(
                (
                    row["idx"],
                    "response_mode",
                    row["expected_mode"],
                    context["response_mode"],
                )
            )
        if "expected_micro" in row and context["micro_strategy"] != row["expected_micro"]:
            failures.append(
                (
                    row["idx"],
                    "micro_strategy",
                    row["expected_micro"],
                    context["micro_strategy"],
                )
            )

    assert failures == []


@pytest.mark.parametrize(
    "value",
    ["0.25", "1.5", "3.14159"],
)
def test_decimal_periods_are_not_clause_boundaries(value):
    agent = ReasoningAgent(RecordingClient())
    problem = f"Evaluate f({value}) using Newton interpolation."
    clauses = agent._score_first_clause_records(problem)
    assert len(clauses) == 1
    assert value in clauses[0]["text"]
    assert agent._score_first_request_spans(problem) == [problem]


@pytest.mark.parametrize(
    ("abbreviation", "problem"),
    [
        ("a.e.", "Suppose f_n -> f a.e. on X, then compute the integral."),
        ("i.e.", "Use the special case i.e. x=1, then compute the value."),
        ("e.g.", "For e.g. x=1, compute the value."),
        ("w.r.t.", "Differentiate w.r.t. x and then evaluate at x=1."),
    ],
)
def test_protected_abbreviations_are_not_split_inside(abbreviation, problem):
    agent = ReasoningAgent(RecordingClient())
    clauses = agent._score_first_clause_records(problem)
    reconstructed = "".join(record["text"] for record in clauses)
    assert reconstructed == problem
    assert abbreviation in reconstructed
    assert not any(record["text"].strip() in {"a.", "e.", "i.", "g.", "w.", "r.", "t."} for record in clauses)


@pytest.mark.parametrize(
    "problem",
    [
        "The data show that X is normal. Compute E[X^2].",
        "The results show that the method converges. Evaluate the final error.",
        "We use Newton interpolation in the background. Calculate 2+2.",
        "The proof shows that f is measurable. Compute the integral.",
    ],
)
def test_declarative_english_action_verbs_do_not_become_request_starts(problem):
    agent = ReasoningAgent(RecordingClient())
    context = agent._score_first_context(problem, {})
    assert context["response_mode"] == _SCORE_FIRST_RESPONSE_MODE_ANSWER
    assert not context["request_spans"][0].lower().startswith(("show", "use", "proof"))


@pytest.mark.parametrize(
    ("problem", "expected_span"),
    [
        ("根据上述证明，计算该积分的值。", "计算该积分的值。"),
        ("该证明表明函数可测，求其积分。", "求其积分。"),
        ("已有推导使用 Taylor 展开，计算误差。", "计算误差。"),
        ("这个作用算子是紧的，判断 I-T 是否可逆。", "判断 I-T 是否可逆。"),
        ("设函数满足题目要求，计算其导数。", "计算其导数。"),
    ],
)
def test_chinese_noun_or_word_internal_action_characters_stay_context(problem, expected_span):
    agent = ReasoningAgent(RecordingClient())
    assert agent._score_first_request_spans(problem) == [expected_span]


@pytest.mark.parametrize(
    ("problem", "expected_prefix"),
    [
        ("使用普通生成函数求满足条件的序列数。", "使用"),
        ("利用牛顿插值计算 f(0.25)。", "利用"),
        ("采用梯形公式计算积分。", "采用"),
        ("应用留数定理求围道积分。", "应用"),
        ("用牛顿法求根。", "用"),
    ],
)
def test_chinese_method_directives_at_request_positions_remain_target_side(problem, expected_prefix):
    agent = ReasoningAgent(RecordingClient())
    spans = agent._score_first_request_spans(problem)
    assert len(spans) == 1
    assert spans[0].startswith(expected_prefix)


@pytest.mark.parametrize(
    "problem",
    [
        "Let T be compact. Is I-T invertible?",
        "Does f_n converge uniformly?",
        "How many iterations are required?",
        "该算子是否可逆？",
        "能否得到唯一解？",
        "需要多少次迭代？",
    ],
)
def test_interrogatives_create_request_spans_without_imperative_verbs(problem):
    agent = ReasoningAgent(RecordingClient())
    assert agent._score_first_request_spans(problem)


def test_intent_is_classified_per_span_then_aggregated():
    agent = ReasoningAgent(RecordingClient())
    context = agent._score_first_context(
        "Which theorem applies? Briefly justify your choice.",
        {"subject": "Advanced Mathematics"},
    )
    assert context["request_spans"] == [
        "Which theorem applies?",
        "justify your choice.",
    ]
    assert _SCORE_FIRST_RESPONSE_MODE_ANSWER in context["requested_actions"]
    assert _SCORE_FIRST_RESPONSE_MODE_DERIVATION in context["requested_actions"]
    assert context["response_mode"] == _SCORE_FIRST_RESPONSE_MODE_DERIVATION


def test_multiple_requested_actions_inside_one_clause_upgrade_response_mode():
    agent = ReasoningAgent(RecordingClient())

    derivation = agent._score_first_context(
        "Compute the MLE, then derive its asymptotic variance.",
        {"subject": "Statistics"},
    )
    assert _SCORE_FIRST_RESPONSE_MODE_ANSWER in derivation["requested_actions"]
    assert _SCORE_FIRST_RESPONSE_MODE_DERIVATION in derivation["requested_actions"]
    assert derivation["response_mode"] == _SCORE_FIRST_RESPONSE_MODE_DERIVATION

    proof = agent._score_first_context(
        "Compute the value and then prove the uniqueness claim.",
        {"subject": "Advanced Mathematics"},
    )
    assert _SCORE_FIRST_RESPONSE_MODE_PROOF in proof["requested_actions"]
    assert proof["response_mode"] == _SCORE_FIRST_RESPONSE_MODE_PROOF


def test_explicit_negative_request_blocks_contradictory_metadata():
    agent = ReasoningAgent(RecordingClient())

    english = agent._score_first_context(
        "No proof is required. Compute the determinant.",
        {"subject": "Linear Algebra", "task_type": "proof"},
    )
    assert english["response_mode"] == _SCORE_FIRST_RESPONSE_MODE_ANSWER

    chinese = agent._score_first_context(
        "无需证明，只计算积分。",
        {"subject": "Measure Theory", "task_type": "proof"},
    )
    assert chinese["response_mode"] == _SCORE_FIRST_RESPONSE_MODE_ANSWER


def test_request_diagnostics_do_not_leak_into_score_first_prompt():
    client = RecordingClient()
    agent = ReasoningAgent(client)
    context = agent._score_first_context(
        "Compute the MLE, then derive its asymptotic variance.",
        {"subject": "Statistics"},
    )
    assert context["request_spans"]
    assert context["requested_actions"]
    assert "target_text" in context
    assert "context_text" in context

    agent.solve(
        "Compute the MLE, then derive its asymptotic variance.",
        {"subject": "Statistics"},
    )
    prompt = "\n".join(message["content"] for message in client.calls[0]["messages"])
    assert "request_spans" not in prompt
    assert "requested_actions" not in prompt
    assert "target_text" not in prompt
    assert "context_text" not in prompt


def test_v24_preserves_one_model_call_per_successful_problem():
    client = RecordingClient()
    agent = ReasoningAgent(client)
    for index in range(100):
        result = agent.solve(
            f"Compute 6*7 for parser regression {index}.",
            {"subject": "Advanced Mathematics", "task_type": "calculation"},
        )
        assert result["final_response"] == "42"
    assert len(client.calls) == 100
    assert all(call["temperature"] == 0.8 for call in client.calls)
    assert all(call["max_tokens"] == 32768 for call in client.calls)
    assert all(call["thinking_mode"] is True for call in client.calls)

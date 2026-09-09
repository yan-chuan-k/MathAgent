from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

import math_agent_core.tools.augmented_tool as augmented_tool
from math_agent_core.tools.augmented_tool import ToolBatchResult
from user_agent import (
    ReasoningAgent,
    _SCORE_FIRST_DEFAULT_EXPERIMENT_PRESET,
    _SCORE_FIRST_HUMAN_DOMAIN_LABELS,
)


ROOT = Path(__file__).resolve().parents[1]
FROZEN_ROUTING = ROOT / "sample_data" / "score_recovery_v2_synthetic_hard.jsonl"
V305_PRESET = "tool_augmented_v305_full_final"


class SequenceClient:
    def __init__(self, responses: List[str]) -> None:
        self.responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        if not self.responses:
            raise AssertionError("unexpected extra model call")
        return self.responses.pop(0)


class NoToolClient:
    """Return NO_TOOL to every extractor invocation and a valid final surface."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        system = str(messages[0]["content"])
        if "exact deterministic tool extractor" in system:
            return "NO_TOOL"
        return "Final answer: 42"


class FailingExtractorClient:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        if "exact deterministic tool extractor" in str(messages[0]["content"]):
            raise RuntimeError("formalizer unavailable")
        return "Final answer: 2"


def _trace(result: Dict[str, Any]) -> Dict[str, str]:
    return {item["step"]: item["content"] for item in result["trace"]}


def _rows() -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in FROZEN_ROUTING.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "subject": _SCORE_FIRST_HUMAN_DOMAIN_LABELS[row["expected_domain"]],
        "task_type": row["task_type"],
    }


def _full_baseline(problem: str, metadata: Dict[str, Any]):
    baseline_agent = ReasoningAgent(
        SequenceClient([]),
        score_first_experiment_preset="full_thinking_on",
    )
    return baseline_agent._build_score_first_prompt(problem, metadata)


def _v305_agent(client: Any) -> ReasoningAgent:
    return ReasoningAgent(client, score_first_experiment_preset=V305_PRESET)


def _no_decisive_failures(*args, **kwargs):
    result = ([], [], [])
    return result if kwargs.get("with_metadata") else result[0]


def test_v305_remains_the_frozen_full_extended_tool_rollback_arm():
    agent = _v305_agent(SequenceClient([]))

    assert _SCORE_FIRST_DEFAULT_EXPERIMENT_PRESET == "tool_augmented_v307_cross_subject"
    assert agent.score_first_experiment_preset == V305_PRESET
    assert agent.production_mode == "tool_augmented"
    assert agent.score_first_prompt_profile == "full"
    assert agent.thinking_mode is True
    assert agent.tool_augmented_extended_tools is True
    assert agent.tool_augmented_verification_timeout_seconds == pytest.approx(5.0)
    assert augmented_tool.MAX_VERIFY_TIMEOUT_SECONDS == pytest.approx(5.0)


def test_no_tool_final_messages_match_full_baseline_for_frozen_110_cases(monkeypatch):
    """The extra extractor call may not perturb the known FULL prompt at all."""

    monkeypatch.setattr(
        ReasoningAgent,
        "_tool_augmented_decisive_failures",
        _no_decisive_failures,
    )
    rows = _rows()
    assert len(rows) == 110

    client = NoToolClient()
    agent = _v305_agent(client)
    matches = 0
    for row in rows:
        metadata = _metadata(row)
        result = agent.solve(row["problem"], metadata)
        final_messages = client.calls[-1]["messages"]
        assert final_messages == _full_baseline(row["problem"], metadata)
        assert "Optional deterministic computations:" not in final_messages[1]["content"]
        assert "Compact formalizer plan (advisory):" not in final_messages[1]["content"]
        trace = _trace(result)
        assert trace["final_prompt_profile"] == "full"
        assert trace["formalizer_plan_injected"] == "false"
        matches += 1

    assert matches == 110


def test_formalizer_failure_has_the_exact_full_baseline_fallback(monkeypatch):
    monkeypatch.setattr(
        ReasoningAgent,
        "_tool_augmented_decisive_failures",
        _no_decisive_failures,
    )
    problem = "Compute 1+1."
    metadata = {"subject": "Advanced Mathematics"}
    client = FailingExtractorClient()

    result = _v305_agent(client).solve(problem, metadata)

    assert result["final_response"] == "2"
    assert len(client.calls) == 2
    assert client.calls[-1]["messages"] == _full_baseline(problem, metadata)
    trace = _trace(result)
    assert trace["tool_worker_status"] == "no_requests"
    assert trace["formalizer_fallback"] == "RuntimeError"
    assert "error" not in trace


def test_malformed_tool_line_has_the_exact_full_baseline_fallback(monkeypatch):
    monkeypatch.setattr(
        ReasoningAgent,
        "_tool_augmented_decisive_failures",
        _no_decisive_failures,
    )
    problem = "Compute 1+1."
    metadata = {"subject": "Advanced Mathematics"}
    client = SequenceClient(
        ["TOOL: sympy|solve|this-is-not-valid", "Final answer: 2"]
    )

    result = _v305_agent(client).solve(problem, metadata)

    assert result["final_response"] == "2"
    assert len(client.calls) == 2
    assert client.calls[-1]["messages"] == _full_baseline(problem, metadata)
    trace = _trace(result)
    assert trace["tool_worker_status"] == "parse_rejected"
    assert trace["tool_successes"] == "0"


def test_plan_only_output_has_the_exact_full_baseline_fallback(monkeypatch):
    """Accepted legacy plan prefixes are trace-only in the V3.0.5 arm."""

    monkeypatch.setattr(
        ReasoningAgent,
        "_tool_augmented_decisive_failures",
        _no_decisive_failures,
    )
    problem = "Compute 1+1."
    metadata = {"subject": "Advanced Mathematics"}
    client = SequenceClient(
        ["TARGET: solve it\nMETHOD: arithmetic", "Final answer: 2"]
    )

    result = _v305_agent(client).solve(problem, metadata)

    assert result["final_response"] == "2"
    assert len(client.calls) == 2
    assert client.calls[-1]["messages"] == _full_baseline(problem, metadata)
    trace = _trace(result)
    assert trace["formalizer_plan_lines"] == "2"
    assert trace["formalizer_plan_injected"] == "false"
    assert trace["tool_worker_status"] == "no_requests"


def test_tool_timeout_has_the_exact_full_baseline_fallback(monkeypatch):
    monkeypatch.setattr(
        ReasoningAgent,
        "_tool_augmented_decisive_failures",
        _no_decisive_failures,
    )

    def timed_out(self, requests):
        return ToolBatchResult([], "timeout")

    monkeypatch.setattr(
        augmented_tool.SafeAugmentedToolExecutor,
        "execute_with_status",
        timed_out,
    )
    problem = "Compute 1+1."
    metadata = {"subject": "Advanced Mathematics"}
    client = SequenceClient(
        ["TOOL: sympy|simplify|expr=1+1", "Final answer: 2"]
    )

    result = _v305_agent(client).solve(problem, metadata)

    assert result["final_response"] == "2"
    assert len(client.calls) == 2
    assert client.calls[-1]["messages"] == _full_baseline(problem, metadata)
    trace = _trace(result)
    assert trace["tool_worker_status"] == "timeout"
    assert trace["tool_successes"] == "0"


def test_successful_evidence_only_augments_the_frozen_full_prompt(monkeypatch):
    monkeypatch.setattr(
        ReasoningAgent,
        "_tool_augmented_decisive_failures",
        _no_decisive_failures,
    )
    problem = "Solve x^2-5*x+6=0 for x."
    metadata = {"subject": "Advanced Mathematics"}
    client = SequenceClient(
        [
            "TARGET: ignored plan\n"
            "METHOD: do not use this\n"
            "TOOL: sympy|solve|equation=x^2-5*x+6=0|var=x|domain=real",
            "Final answer: x=2 or x=3",
        ]
    )

    result = _v305_agent(client).solve(problem, metadata)

    assert result["final_response"] == "x=2 or x=3"
    assert len(client.calls) == 2
    baseline = _full_baseline(problem, metadata)
    final_messages = client.calls[1]["messages"]
    extractor_call = client.calls[0]
    assert extractor_call["temperature"] == pytest.approx(0.2)
    assert extractor_call["top_p"] == pytest.approx(0.95)
    assert extractor_call["max_tokens"] == 1536
    assert extractor_call["thinking_mode"] is False
    assert "exact deterministic tool extractor" in extractor_call["messages"][0]["content"]
    assert "Do not output TARGET, METHOD" in extractor_call["messages"][0]["content"]
    assert client.calls[1]["temperature"] == pytest.approx(0.8)
    assert client.calls[1]["top_p"] == pytest.approx(0.95)
    assert client.calls[1]["max_tokens"] == 32768
    assert client.calls[1]["thinking_mode"] is True
    assert final_messages[0] == baseline[0]

    prefix, marker, suffix = baseline[1]["content"].partition(f"Problem:\n{problem}")
    assert marker
    expected_evidence = (
        "Optional deterministic computations:\n\n"
        "Deterministically computed facts from the formalizer's exact requests:\n"
        "- solve[x^2-5*x+6=0, x over Reals] -> {2, 3}\n\n"
        "Each computation is exact only for the input explicitly shown.\n"
        "Compare that shown input with the original problem before using it.\n"
        "Ignore any computation whose input does not match the requested object.\n\n"
    )
    assert final_messages[1]["content"] == f"{prefix}{expected_evidence}{marker}{suffix}"
    assert "Compact formalizer plan (advisory):" not in final_messages[1]["content"]
    assert "TARGET: ignored plan" not in final_messages[1]["content"]
    assert "METHOD: do not use this" not in final_messages[1]["content"]

    trace = _trace(result)
    assert trace["experiment_preset"] == V305_PRESET
    assert trace["final_prompt_profile"] == "full"
    assert trace["formalizer_plan_injected"] == "false"
    assert trace["tool_extractor_used"] == "true"
    assert trace["tool_successes"] == "1"
    assert trace["tool_worker_status"] == "completed"
    assert trace["model_calls"] == "2"
    assert trace["correction_call_used"] == "false"


@pytest.mark.parametrize(
    ("problem", "expected_mode"),
    [
        ("Prove that 1=1.", "PROOF"),
        ("Prove or disprove that 1=1.", "PROOF_OR_DISPROOF"),
        (
            "Construct a counterexample to the claim that every integer is even.",
            "CONSTRUCTION_COUNTEREXAMPLE",
        ),
    ],
)
def test_proof_oriented_modes_skip_the_tool_extractor(problem, expected_mode):
    metadata = {"subject": "Advanced Mathematics"}
    client = SequenceClient(["Conclusion: directly answered by the FULL solver."])

    result = _v305_agent(client).solve(problem, metadata)

    assert len(client.calls) == 1
    assert client.calls[0]["messages"] == _full_baseline(problem, metadata)
    trace = _trace(result)
    assert expected_mode in trace["mode"]
    assert trace["tool_extractor_used"] == "false"
    assert trace["formalizer_used"] == "false"
    assert trace["model_calls"] == "1"


def test_incomplete_equation_answer_triggers_correction_with_default_five_second_window():
    client = SequenceClient(
        [
            "NO_TOOL",
            "Final answer: x=2 only",
            "Final answer: x=2 or x=3",
        ]
    )

    result = _v305_agent(client).solve(
        "Given f(0)=1, solve x^2-5*x+6=0 for x.",
        {},
    )

    assert result["final_response"] == "x=2 or x=3"
    assert len(client.calls) == 3
    trace = _trace(result)
    assert trace["correction_call_used"] == "true"
    assert trace["correction_kept"] == "true"


def test_wrong_named_matrix_answer_corrects_against_B_only():
    problem = "Let A=[[1,0],[0,1]] and B=[[1,2],[3,4]]. Compute det(B)."
    client = SequenceClient(["NO_TOOL", "Final answer: -1", "Final answer: -2"])

    result = _v305_agent(client).solve(problem, {})

    assert result["final_response"] == "-2"
    assert len(client.calls) == 3
    correction = client.calls[2]["messages"][1]["content"]
    failure = correction.split("Exact failed deterministic check(s):", 1)[1].split(
        "ORIGINAL PROBLEM:",
        1,
    )[0]
    assert "matrix=[[1, 2], [3, 4]]" in failure
    assert "matrix=[[1, 0], [0, 1]]" not in failure
    assert _trace(result)["correction_call_used"] == "true"


def test_verifier_exception_retains_the_call_two_answer(monkeypatch):
    def failing_verifier(*args, **kwargs):
        raise RuntimeError("verification unavailable")

    monkeypatch.setattr(
        augmented_tool,
        "run_answer_verification_in_subprocess",
        failing_verifier,
    )
    client = SequenceClient(["NO_TOOL", "Final answer: 2"])

    result = _v305_agent(client).solve("Compute 1+1.", {})

    assert result["final_response"] == "2"
    assert len(client.calls) == 2
    trace = _trace(result)
    assert trace["correction_call_used"] == "false"
    assert "error" not in trace

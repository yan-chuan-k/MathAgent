from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pytest

import user_agent
from user_agent import (
    ReasoningAgent,
    _SCORE_FIRST_DEFAULT_EXPERIMENT_PRESET,
    _SCORE_FIRST_EXPERIMENT_PRESETS,
    _SCORE_FIRST_HUMAN_DOMAIN_LABELS,
)


ROOT = Path(__file__).resolve().parents[1]
ROUTING = ROOT / "sample_data" / "score_recovery_v2_synthetic_hard.jsonl"
V29_PROMPT_HASHES = (
    ROOT / "sample_data" / "score_recovery_v210_predev_v29_prompt_hashes.jsonl"
)


class RecordingClient:
    def __init__(self, response: str = "Final answer: 42"):
        self.response = response
        self.calls: List[Dict[str, Any]] = []

    def chat(
        self,
        messages,
        temperature=None,
        top_p=None,
        max_tokens=None,
        thinking_mode=None,
    ):
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
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


def _prompt_blob(messages) -> bytes:
    return json.dumps(
        messages,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _call_for(
    *,
    preset: str | None = None,
    prompt_profile: str | None = None,
    thinking_mode: bool | None = None,
    problem: str = "Compute 6*7.",
    metadata=None,
):
    kwargs = {}
    if preset is not None:
        kwargs["score_first_experiment_preset"] = preset
    if prompt_profile is not None:
        kwargs["score_first_prompt_profile"] = prompt_profile
    if thinking_mode is not None:
        kwargs["thinking_mode"] = thinking_mode

    client = RecordingClient()
    agent = ReasoningAgent(client, **kwargs)
    effective_metadata = (
        metadata if metadata is not None else {"subject": "Advanced Mathematics"}
    )
    result = agent.solve(problem, effective_metadata)
    assert result["final_response"] == "42"
    assert len(client.calls) == 1
    return agent, client.calls[0], result


def test_default_preset_is_valid_and_agent_matches_preset_table():
    assert _SCORE_FIRST_DEFAULT_EXPERIMENT_PRESET in _SCORE_FIRST_EXPERIMENT_PRESETS
    expected = _SCORE_FIRST_EXPERIMENT_PRESETS[_SCORE_FIRST_DEFAULT_EXPERIMENT_PRESET]

    agent = ReasoningAgent(RecordingClient())
    assert agent.score_first_experiment_preset == _SCORE_FIRST_DEFAULT_EXPERIMENT_PRESET
    assert agent.score_first_prompt_profile == expected["prompt_profile"]
    assert agent.thinking_mode is expected["thinking_mode"]
    assert agent.production_mode == expected.get("production_mode", "score_first")


def test_preset_table_is_complete_closed_world_release_matrix():
    assert _SCORE_FIRST_EXPERIMENT_PRESETS == {
        "v29_minimal": {
            "prompt_profile": "minimal",
            "thinking_mode": True,
        },
        "minimal_thinking_off": {
            "prompt_profile": "minimal",
            "thinking_mode": False,
        },
        "ultra_minimal": {
            "prompt_profile": "ultra_minimal",
            "thinking_mode": True,
        },
        "ultra_minimal_thinking_off": {
            "prompt_profile": "ultra_minimal",
            "thinking_mode": False,
        },
        "full_thinking_on": {
            "prompt_profile": "full",
            "thinking_mode": True,
        },
        "full_thinking_off": {
            "prompt_profile": "full",
            "thinking_mode": False,
        },
        "tool_augmented_v30": {
            "prompt_profile": "minimal",
            "thinking_mode": True,
            "production_mode": "tool_augmented",
            "extended_tools": False,
        },
        "tool_augmented_v31_extended": {
            "prompt_profile": "minimal",
            "thinking_mode": True,
            "production_mode": "tool_augmented",
            "extended_tools": True,
        },
        "tool_augmented_v305_full_final": {
            "prompt_profile": "full",
            "thinking_mode": True,
            "production_mode": "tool_augmented",
            "extended_tools": True,
        },
        "tool_augmented_v306_cross_subject": {
            "prompt_profile": "full",
            "thinking_mode": True,
            "production_mode": "tool_augmented",
            "extended_tools": True,
            "challenge_reasoning": True,
            "grounded_auto_tools": True,
        },
        "tool_augmented_v307_cross_subject": {
            "prompt_profile": "full",
            "thinking_mode": True,
            "production_mode": "tool_augmented",
            "extended_tools": True,
            "challenge_reasoning": True,
            "grounded_auto_tools": True,
            "max_tokens": 49152,
            "v307_tools": True,
        },
    }


@pytest.mark.parametrize("bad", ["unknown", "v30", "minimal-off"])
def test_unknown_experiment_preset_is_rejected(bad):
    with pytest.raises(ValueError):
        ReasoningAgent(
            RecordingClient(),
            score_first_experiment_preset=bad,
        )


def test_explicit_constructor_overrides_have_priority_over_preset():
    client = RecordingClient()
    agent = ReasoningAgent(
        client,
        score_first_experiment_preset="minimal_thinking_off",
        score_first_prompt_profile="ultra_minimal",
        thinking_mode=True,
        temperature=0.61,
        top_p=0.87,
        max_tokens=12345,
    )
    agent.solve(
        "Compute P(A|B).",
        {"subject": "Probability Theory"},
    )
    call = client.calls[0]

    assert agent.score_first_experiment_preset == "minimal_thinking_off"
    assert agent.score_first_prompt_profile == "ultra_minimal"
    assert agent.thinking_mode is True
    assert call["temperature"] == pytest.approx(0.61)
    assert call["top_p"] == pytest.approx(0.87)
    assert call["max_tokens"] == 12345
    assert call["thinking_mode"] is True
    assert "Subject:" not in call["messages"][1]["content"]


@pytest.mark.parametrize(
    ("preset", "profile", "thinking"),
    [
        ("v29_minimal", "minimal", True),
        ("minimal_thinking_off", "minimal", False),
        ("ultra_minimal", "ultra_minimal", True),
        ("ultra_minimal_thinking_off", "ultra_minimal", False),
        ("full_thinking_on", "full", True),
        ("full_thinking_off", "full", False),
        ("tool_augmented_v30", "minimal", True),
        ("tool_augmented_v31_extended", "minimal", True),
        ("tool_augmented_v305_full_final", "full", True),
        ("tool_augmented_v306_cross_subject", "full", True),
        ("tool_augmented_v307_cross_subject", "full", True),
    ],
)
def test_preset_effective_profile_and_thinking_mode(preset, profile, thinking):
    agent = ReasoningAgent(
        RecordingClient(),
        score_first_experiment_preset=preset,
    )
    assert agent.score_first_experiment_preset == preset
    assert agent.score_first_prompt_profile == profile
    assert agent.thinking_mode is thinking
    assert agent.temperature == pytest.approx(0.8)
    assert agent.top_p == pytest.approx(0.95)
    assert agent.max_tokens == (49152 if preset == "tool_augmented_v307_cross_subject" else 32768)


@pytest.mark.parametrize("preset", sorted(_SCORE_FIRST_EXPERIMENT_PRESETS))
def test_changing_only_default_preset_constant_selects_that_arm(monkeypatch, preset):
    monkeypatch.setattr(
        user_agent,
        "_SCORE_FIRST_DEFAULT_EXPERIMENT_PRESET",
        preset,
    )
    expected = _SCORE_FIRST_EXPERIMENT_PRESETS[preset]
    agent = ReasoningAgent(RecordingClient())
    assert agent.score_first_experiment_preset == preset
    assert agent.score_first_prompt_profile == expected["prompt_profile"]
    assert agent.thinking_mode is expected["thinking_mode"]
    assert agent.production_mode == expected.get("production_mode", "score_first")


def test_v29_minimal_110_prompts_are_byte_identical():
    reference = {row["idx"]: row for row in _rows(V29_PROMPT_HASHES)}
    routing = _rows(ROUTING)
    assert len(routing) == 110
    assert len(reference) == 110

    agent = ReasoningAgent(
        RecordingClient(),
        score_first_experiment_preset="v29_minimal",
    )

    matches = 0
    for row in routing:
        metadata = {
            "subject": _SCORE_FIRST_HUMAN_DOMAIN_LABELS[row["expected_domain"]],
            "task_type": row["task_type"],
        }
        messages = agent._build_score_first_prompt(row["problem"], metadata)
        digest = hashlib.sha256(_prompt_blob(messages)).hexdigest()
        assert digest == reference[row["idx"]]["v29_minimal_sha256"]
        matches += 1

    assert matches == 110


def test_v29_minimal_default_call_parameters_match_v29():
    agent, call, _ = _call_for(preset="v29_minimal")
    assert agent.score_first_prompt_profile == "minimal"
    assert call["temperature"] == pytest.approx(0.8)
    assert call["top_p"] == pytest.approx(0.95)
    assert call["max_tokens"] == 32768
    assert call["thinking_mode"] is True


def test_minimal_thinking_off_changes_only_thinking_mode():
    _, on_call, _ = _call_for(preset="v29_minimal")
    _, off_call, _ = _call_for(preset="minimal_thinking_off")

    assert on_call["messages"] == off_call["messages"]
    assert on_call["temperature"] == off_call["temperature"] == pytest.approx(0.8)
    assert on_call["top_p"] == off_call["top_p"] == pytest.approx(0.95)
    assert on_call["max_tokens"] == off_call["max_tokens"] == 32768

    differing = {
        key
        for key in on_call
        if on_call[key] != off_call[key]
    }
    assert differing == {"thinking_mode"}
    assert on_call["thinking_mode"] is True
    assert off_call["thinking_mode"] is False


def test_ultra_minimal_removes_only_trusted_subject_line():
    problem = "Compute P(A|B)."
    metadata = {"subject": "Probability Theory"}

    _, minimal_call, _ = _call_for(
        preset="v29_minimal",
        problem=problem,
        metadata=metadata,
    )
    _, ultra_call, _ = _call_for(
        preset="ultra_minimal",
        problem=problem,
        metadata=metadata,
    )

    minimal_messages = minimal_call["messages"]
    ultra_messages = ultra_call["messages"]

    assert minimal_messages[0] == ultra_messages[0]
    subject_prefix = "Subject: Probability Theory\n\n"
    assert minimal_messages[1]["content"].startswith(subject_prefix)
    assert (
        minimal_messages[1]["content"][len(subject_prefix):]
        == ultra_messages[1]["content"]
    )
    assert "Subject:" not in ultra_messages[1]["content"]


def test_ultra_minimal_without_trusted_subject_is_byte_identical_to_minimal():
    problem = "Compute P(A|B)."
    metadata = {}

    _, minimal_call, _ = _call_for(
        preset="v29_minimal",
        problem=problem,
        metadata=metadata,
    )
    _, ultra_call, _ = _call_for(
        preset="ultra_minimal",
        problem=problem,
        metadata=metadata,
    )
    assert minimal_call["messages"] == ultra_call["messages"]


def test_full_thinking_off_preserves_frozen_full_prompt_and_changes_only_thinking():
    problem = "Use KKT conditions to minimize x^2 subject to x>=1."
    metadata = {"subject": "Optimization"}

    _, full_on_call, _ = _call_for(
        preset="full_thinking_on",
        problem=problem,
        metadata=metadata,
    )
    _, full_off_call, _ = _call_for(
        preset="full_thinking_off",
        problem=problem,
        metadata=metadata,
    )

    assert full_on_call["messages"] == full_off_call["messages"]
    differing = {
        key
        for key in full_on_call
        if full_on_call[key] != full_off_call[key]
    }
    assert differing == {"thinking_mode"}
    assert full_on_call["thinking_mode"] is True
    assert full_off_call["thinking_mode"] is False


def test_full_110_prompts_still_match_v29_reference():
    reference = {row["idx"]: row for row in _rows(V29_PROMPT_HASHES)}
    routing = _rows(ROUTING)

    agent = ReasoningAgent(
        RecordingClient(),
        score_first_experiment_preset="full_thinking_on",
    )

    matches = 0
    for row in routing:
        metadata = {
            "subject": _SCORE_FIRST_HUMAN_DOMAIN_LABELS[row["expected_domain"]],
            "task_type": row["task_type"],
        }
        messages = agent._build_score_first_prompt(row["problem"], metadata)
        digest = hashlib.sha256(_prompt_blob(messages)).hexdigest()
        assert digest == reference[row["idx"]]["v29_full_sha256"]
        matches += 1
    assert matches == 110


@pytest.mark.parametrize(
    "preset",
    ["v29_minimal", "minimal_thinking_off", "ultra_minimal", "ultra_minimal_thinking_off"],
)
def test_nonfull_presets_keep_router_bypassed(monkeypatch, preset):
    def fail_router(*args, **kwargs):
        raise AssertionError("non-full preset must not call classify_problem")

    monkeypatch.setattr(user_agent, "classify_problem", fail_router)
    client = RecordingClient()
    agent = ReasoningAgent(
        client,
        score_first_experiment_preset=preset,
    )
    result = agent.solve(
        "Use Bayes' theorem to compute P(D|+).",
        {"subject": "Probability Theory"},
    )

    assert result["final_response"] == "42"
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    ("preset", "expected_thinking"),
    [
        ("v29_minimal", True),
        ("minimal_thinking_off", False),
        ("ultra_minimal", True),
        ("ultra_minimal_thinking_off", False),
        ("full_thinking_on", True),
        ("full_thinking_off", False),
    ],
)
def test_each_preset_100_successes_equal_100_calls(preset, expected_thinking):
    client = RecordingClient()
    agent = ReasoningAgent(
        client,
        score_first_experiment_preset=preset,
    )

    for index in range(100):
        result = agent.solve(
            f"Compute 6*7 for PREDEV preset regression {index}.",
            {"subject": "Advanced Mathematics", "task_type": "calculation"},
        )
        assert result["final_response"] == "42"

    assert len(client.calls) == 100
    assert all(call["temperature"] == pytest.approx(0.8) for call in client.calls)
    assert all(call["top_p"] == pytest.approx(0.95) for call in client.calls)
    assert all(call["max_tokens"] == 32768 for call in client.calls)
    assert all(call["thinking_mode"] is expected_thinking for call in client.calls)


def test_prompt_size_expectations_on_frozen_110_suite():
    routing = _rows(ROUTING)
    profiles = {
        "minimal": ReasoningAgent(
            RecordingClient(),
            score_first_experiment_preset="v29_minimal",
        ),
        "ultra": ReasoningAgent(
            RecordingClient(),
            score_first_experiment_preset="ultra_minimal",
        ),
        "full": ReasoningAgent(
            RecordingClient(),
            score_first_experiment_preset="full_thinking_on",
        ),
    }

    lengths = {key: [] for key in profiles}
    for row in routing:
        metadata = {
            "subject": _SCORE_FIRST_HUMAN_DOMAIN_LABELS[row["expected_domain"]],
            "task_type": row["task_type"],
        }
        for key, agent in profiles.items():
            messages = agent._build_score_first_prompt(row["problem"], metadata)
            lengths[key].append(sum(len(m["content"]) for m in messages))

    assert max(lengths["minimal"]) <= 556
    assert all(
        ultra <= minimal
        for ultra, minimal in zip(lengths["ultra"], lengths["minimal"])
    )
    assert max(lengths["full"]) <= 1900


def test_trace_reports_effective_experiment_settings():
    agent, _, result = _call_for(preset="minimal_thinking_off")
    trace = result["trace"]

    values = {step["step"]: step["content"] for step in trace}
    assert "experiment_preset=minimal_thinking_off" in values["mode"]
    assert "thinking_mode=false" in values["mode"]
    assert values["prompt_profile"] == "minimal"
    assert agent.thinking_mode is False
    assert len(trace) <= 5


def test_ultra_minimal_trace_does_not_claim_subject_was_injected():
    _, _, result = _call_for(
        preset="ultra_minimal",
        problem="Compute P(A|B).",
        metadata={"subject": "Probability Theory"},
    )
    assert not any(step["step"] == "subject_hint" for step in result["trace"])


def test_ultra_minimal_thinking_off_changes_only_thinking_mode():
    problem = "Compute P(A|B)."
    metadata = {"subject": "Probability Theory"}

    _, on_call, _ = _call_for(
        preset="ultra_minimal",
        problem=problem,
        metadata=metadata,
    )
    _, off_call, _ = _call_for(
        preset="ultra_minimal_thinking_off",
        problem=problem,
        metadata=metadata,
    )

    assert on_call["messages"] == off_call["messages"]
    assert on_call["temperature"] == off_call["temperature"] == pytest.approx(0.8)
    assert on_call["top_p"] == off_call["top_p"] == pytest.approx(0.95)
    assert on_call["max_tokens"] == off_call["max_tokens"] == 32768
    differing = {key for key in on_call if on_call[key] != off_call[key]}
    assert differing == {"thinking_mode"}
    assert on_call["thinking_mode"] is True
    assert off_call["thinking_mode"] is False


def test_minimal_off_vs_ultra_off_with_trusted_subject_differs_only_by_subject_line():
    problem = "Compute P(A|B)."
    metadata = {"subject": "Probability Theory"}

    _, minimal_call, _ = _call_for(
        preset="minimal_thinking_off",
        problem=problem,
        metadata=metadata,
    )
    _, ultra_call, _ = _call_for(
        preset="ultra_minimal_thinking_off",
        problem=problem,
        metadata=metadata,
    )

    assert minimal_call["temperature"] == ultra_call["temperature"] == pytest.approx(0.8)
    assert minimal_call["top_p"] == ultra_call["top_p"] == pytest.approx(0.95)
    assert minimal_call["max_tokens"] == ultra_call["max_tokens"] == 32768
    assert minimal_call["thinking_mode"] is ultra_call["thinking_mode"] is False

    assert minimal_call["messages"][0] == ultra_call["messages"][0]
    subject_prefix = "Subject: Probability Theory\n\n"
    assert minimal_call["messages"][1]["content"].startswith(subject_prefix)
    assert (
        minimal_call["messages"][1]["content"][len(subject_prefix):]
        == ultra_call["messages"][1]["content"]
    )


def test_minimal_off_vs_ultra_off_without_trusted_subject_is_byte_identical():
    problem = "Compute P(A|B)."
    metadata = {}

    _, minimal_call, _ = _call_for(
        preset="minimal_thinking_off",
        problem=problem,
        metadata=metadata,
    )
    _, ultra_call, _ = _call_for(
        preset="ultra_minimal_thinking_off",
        problem=problem,
        metadata=metadata,
    )
    assert minimal_call == ultra_call


def test_optional_release_default_check():
    expected = os.getenv("EXPECTED_SCORE_FIRST_PRESET")
    if not expected:
        return
    assert expected in _SCORE_FIRST_EXPERIMENT_PRESETS
    assert _SCORE_FIRST_DEFAULT_EXPERIMENT_PRESET == expected

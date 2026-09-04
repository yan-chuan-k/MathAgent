from __future__ import annotations

from typing import Any, Dict, List

import pytest

from user_agent import ReasoningAgent


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


class PromptSensitiveTruncationClient:
    """Simulates long incomplete JSON on the legacy orchestrated solver prompt."""

    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def chat(self, messages, temperature=0.1, max_tokens=8192, thinking_mode=True):
        prompt = "\n".join(str(item.get("content", "")) for item in messages)
        self.calls.append(
            {
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "thinking_mode": thinking_mode,
            }
        )
        if "OUTPUT_CONTRACT" in prompt or "Return exactly one valid JSON object" in prompt:
            return '{"problem_id":"truncated","solution":[{"step":1,"content":"' + ("x" * 7000)
        return "Final answer: 42"


def _solve(response: str, problem: str = "Compute 6*7.", metadata=None):
    client = RecordingClient(response)
    agent = ReasoningAgent(client=client)
    result = agent.solve(problem, metadata or {"subject": "mathematics"})
    return result, client, agent


def test_score_first_normal_calculation_one_call():
    result, client, agent = _solve("Final answer: 42")
    assert result["final_response"] == "42"
    assert len(client.calls) == 1
    assert agent.production_mode == "score_first"
    assert agent.orchestrator is None
    assert client.calls[0]["temperature"] == 0.1
    assert client.calls[0]["max_tokens"] == 8192
    assert client.calls[0]["thinking_mode"] is True


def test_score_first_expression_is_preserved():
    result, client, _ = _solve("Final answer: π(e^{-1}-e)")
    assert result["final_response"] == "π(e^{-1}-e)"
    assert len(client.calls) == 1


def test_score_first_multiple_roots_are_preserved():
    result, _, _ = _solve("Final answer: x = -2 or x = 2", "Solve x^2=4.")
    assert result["final_response"] == "x = -2 or x = 2"


def test_score_first_interval_is_preserved():
    result, _, _ = _solve("Final answer: (-∞,1] ∪ [3,∞)", "Solve the inequality.")
    assert result["final_response"] == "(-∞,1] ∪ [3,∞)"


def test_score_first_matrix_and_vector_are_not_collapsed_to_integer():
    matrix, _, _ = _solve("Final answer: [[1, 2], [3, 4]]", "Find the matrix A.")
    vector, _, _ = _solve("Final answer: (1, -2, 3)", "Find the vector v.")
    assert matrix["final_response"] == "[[1, 2], [3, 4]]"
    assert vector["final_response"] == "(1, -2, 3)"


def test_score_first_proof_keeps_concise_proof_text():
    response = "Final answer: The statement is true.\nProof: Since n is even, n=2k, so n^2=4k^2 is even."
    result, client, _ = _solve(response, "Prove that the square of an even integer is even.")
    assert "The statement is true." in result["final_response"]
    assert "Proof:" in result["final_response"]
    assert "n=2k" in result["final_response"]
    assert len(client.calls) == 1


def test_score_first_chinese_final_answer_extracts_cleanly():
    result, _, _ = _solve("最终答案：132", "计算该组合数。", {"subject": "离散数学"})
    assert result["final_response"] == "132"


def test_score_first_prompt_is_compact_free_text_not_internal_json_contract():
    result, client, _ = _solve("Final answer: 42", metadata={"subject": "离散数学"})
    assert result["final_response"] == "42"
    prompt = "\n".join(item["content"] for item in client.calls[0]["messages"])
    for forbidden in (
        "OUTPUT_CONTRACT",
        "Return exactly one valid JSON object",
        "requested_checks",
        "reasoning_plan",
    ):
        assert forbidden not in prompt
    assert "Do not output JSON." in prompt
    assert "Subject hint: 离散数学" in prompt
    assert "Final answer: <complete requested answer>" in prompt


def test_score_first_call_budget_is_exactly_one_per_successful_problem():
    client = RecordingClient("Final answer: 42")
    agent = ReasoningAgent(client=client)
    for index in range(100):
        result = agent.solve(f"Compute 6*7. Case {index}.", {"subject": "mathematics"})
        assert result["final_response"] == "42"
    assert len(client.calls) == 100


def test_score_first_caller_overrides_runtime_parameters():
    client = RecordingClient("Final answer: 42")
    agent = ReasoningAgent(client=client, max_tokens=6000, temperature=0.07, thinking_mode=False)
    agent.solve("Compute 6*7.")
    assert client.calls[0]["max_tokens"] == 6000
    assert client.calls[0]["temperature"] == 0.07
    assert client.calls[0]["thinking_mode"] is False


def test_score_first_trace_is_short_and_does_not_expose_prompt_or_raw_output():
    raw = "Final answer: 42\n" + ("private working " * 100)
    result, _, _ = _solve(raw)
    assert result["final_response"] == "42"
    assert len(result["trace"]) <= 4
    serialized_trace = repr(result["trace"])
    assert "private working" not in serialized_trace
    assert "OUTPUT_CONTRACT" not in serialized_trace


def test_truncation_simulation_shows_legacy_orchestration_cascades_but_score_first_does_not():
    legacy_client = PromptSensitiveTruncationClient()
    legacy = ReasoningAgent(
        client=legacy_client,
        production_mode="orchestrated",
        max_retries=1,
        max_candidates=2,
        enable_critic=True,
    )
    legacy_result = legacy.solve("Compute 6*7.", {"subject": "mathematics"})
    assert len(legacy_client.calls) > 1
    assert any(
        "OUTPUT_CONTRACT" in call["prompt"] or "Return exactly one valid JSON object" in call["prompt"]
        for call in legacy_client.calls
    )
    assert legacy_result["final_response"]

    score_client = PromptSensitiveTruncationClient()
    score_first = ReasoningAgent(client=score_client)
    score_result = score_first.solve("Compute 6*7.", {"subject": "mathematics"})
    assert score_result["final_response"] == "42"
    assert len(score_client.calls) == 1
    assert "OUTPUT_CONTRACT" not in score_client.calls[0]["prompt"]
    assert "Return exactly one valid JSON object" not in score_client.calls[0]["prompt"]


def test_orchestrated_mode_remains_explicitly_available_with_historical_defaults():
    client = RecordingClient("Final answer: 42")
    agent = ReasoningAgent(client=client, production_mode="orchestrated")
    assert agent.production_mode == "orchestrated"
    assert agent.orchestrator is not None
    assert agent.temperature == 0.2
    assert agent.max_tokens == 4096


@pytest.mark.parametrize(
    ("surface", "expected"),
    [
        ("Final answer: 42", "42"),
        ("Final answer is 42.", "42"),
        ("The final answer is 42.", "42"),
        ("Final result: 42;", "42"),
        ("Final result is 42.", "42"),
        ("Answer: 42", "42"),
        ("Answer is 42.", "42"),
        ("The answer is 42.", "42"),
        ("Result: 42", "42"),
        ("The result is 42.", "42"),
        ("Therefore, the answer is 42.", "42"),
        ("Thus, the answer is 42.", "42"),
        ("Hence, the answer is 42.", "42"),
        ("最终答案：42", "42"),
        ("最终答案是42。", "42"),
        ("最后答案：42；", "42"),
        ("答案：42", "42"),
        ("答案是42。", "42"),
        ("答案为42。", "42"),
        ("结果：42", "42"),
        ("结果是42。", "42"),
        ("因此答案是42。", "42"),
        ("所以答案为42。", "42"),
        ("故答案为42。", "42"),
    ],
)
def test_score_first_affirmative_wrappers_remove_copula_artifacts(surface, expected):
    result, client, _ = _solve(surface)
    assert result["final_response"] == expected
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    ("surface", "expected"),
    [
        (r"Final answer: \boxed{42}", "42"),
        ("Final answer: **42**", "42"),
        ("Final answer: __42__", "42"),
        ("Final answer: $42$", "42"),
        (r"Final answer: \(42\)", "42"),
        ("Final answer: {1,2,3}", "{1,2,3}"),
        ("Final answer: [1,2,3]", "[1,2,3]"),
        ("Final answer: (1,-2,3)", "(1,-2,3)"),
        ("Final answer: (-∞,1] ∪ [3,∞)", "(-∞,1] ∪ [3,∞)"),
        (r"Final answer: \frac{1}{2}", r"\frac{1}{2}"),
        ("Final answer: x ≡ 9 (mod 20)", "x ≡ 9 (mod 20)"),
        ("Final answer: 3.14159", "3.14159"),
        ("Final answer: 42.", "42"),
        ("Final answer: 42。", "42"),
        ("Final answer: 42;", "42"),
        ("Final answer: 42；", "42"),
    ],
)
def test_score_first_strips_only_known_outer_presentation_wrappers(surface, expected):
    result, _, _ = _solve(surface)
    assert result["final_response"] == expected


def test_score_first_long_answer_over_500_characters_is_preserved_complete():
    payload = "[" + ",".join(str(i) for i in range(400)) + "]"
    assert len(payload) > 500
    result, client, _ = _solve(f"Final answer: {payload}", "Return the requested vector.")
    assert result["final_response"] == payload
    assert len(result["final_response"]) == len(payload)
    assert result["final_response"].endswith("]")
    assert len(client.calls) == 1


def test_score_first_long_proof_preserves_opening_middle_and_final_sentence():
    opening = "Conclusion: the required statement holds.\n"
    middle = "Proof: " + ("For each intermediate step, the stated invariant is preserved. " * 95)
    ending = "\nTherefore the required statement follows."
    response = opening + middle + ending
    assert len(response) > 5000

    result, client, _ = _solve(response, "Prove that the stated invariant holds for every n.")
    output = result["final_response"]
    assert output == response
    assert output.startswith("Conclusion:")
    assert "For each intermediate step" in output
    assert output.endswith("Therefore the required statement follows.")
    assert len(output) > 5000
    assert len(client.calls) == 1


def test_score_first_first_line_answer_survives_truncated_incomplete_tail():
    response = "Final answer: 42\n" + ("incomplete trailing explanation " * 300)
    assert len(response) > 8000
    result, client, _ = _solve(response)
    assert result["final_response"] == "42"
    assert len(client.calls) == 1


def test_score_first_answer_and_proof_prompts_are_separate():
    answer_client = RecordingClient("Final answer: 42")
    answer_agent = ReasoningAgent(client=answer_client)
    answer_agent.solve("Compute 6*7.")
    answer_prompt = answer_client.calls[0]["messages"][0]["content"]
    assert "Output exactly ONE visible line" in answer_prompt
    assert "Then stop." in answer_prompt
    assert "Do not provide visible explanation or derivation." in answer_prompt
    assert "State the conclusion first" not in answer_prompt

    proof_client = RecordingClient("Conclusion: true.\nProof: direct.")
    proof_agent = ReasoningAgent(client=proof_client)
    proof_agent.solve("Prove that 2 is even.")
    proof_prompt = proof_client.calls[0]["messages"][0]["content"]
    assert "State the conclusion first" in proof_prompt
    assert "concise but complete proof" in proof_prompt
    assert "Do not omit necessary logical steps" in proof_prompt
    assert "Output exactly ONE visible line" not in proof_prompt


def test_score_first_long_matrix_payload_keeps_closing_delimiter_and_no_legacy_cap():
    rows = ["[" + ",".join(str(i + j * 20) for i in range(20)) + "]" for j in range(20)]
    payload = "[" + ",".join(rows) + "]"
    assert len(payload) > 500
    result, client, _ = _solve(f"Final answer: {payload}", "Find the matrix.")
    assert result["final_response"] == payload
    assert result["final_response"].endswith("]]")
    assert len(client.calls) == 1

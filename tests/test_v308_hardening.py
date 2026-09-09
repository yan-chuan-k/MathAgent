from __future__ import annotations

from typing import Any, Dict, List

from math_agent_core.output_guard import looks_truncated
from math_agent_core.tools.augmented_tool import SafeAugmentedToolExecutor, ToolRequest
from user_agent import ReasoningAgent


class SequenceClient:
    def __init__(self, responses: List[Any]) -> None:
        self.responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        if not self.responses:
            raise AssertionError("unexpected extra model call")
        return self.responses.pop(0)


def test_complete_answer_first_line_beats_truncated_tail():
    assert not looks_truncated(
        "Final answer: [1,2,3]\n"
        "The optional derivation was cut off because",
        response_mode="ANSWER_VALUE",
        raw_response={"finish_reason": "length"},
    )


def test_proof_with_provider_length_stop_gets_one_recovery_call():
    class LengthAwareClient(SequenceClient):
        def chat(self, messages, **kwargs):
            response = super().chat(messages, **kwargs)
            self.last_response_metadata = {
                "finish_reason": "length" if len(self.calls) == 1 else "stop"
            }
            return response

    client = LengthAwareClient(
        [
            "Conclusion: true.\nProof: The derivation was cut off",
            "Conclusion: true.\nProof: Let n=2k. Then n^2=2(2k^2), so it is even.",
        ]
    )
    agent = ReasoningAgent(client, score_first_experiment_preset="full_thinking_on")
    result = agent.solve("Prove that the square of an even integer is even.")
    assert len(client.calls) == 2
    assert result["final_response"].startswith("Conclusion:")


def test_v307_high_difficulty_proof_prompt_has_private_reasoning_card():
    client = SequenceClient(
        ["Conclusion: true. Proof: Let n=2k. Then n^2 is even."]
    )
    agent = ReasoningAgent(client)
    agent.solve(
        "Prove that every compact subset of a Hausdorff space is closed.",
        {"subject": "Topology"},
    )
    assert len(client.calls) == 1
    assert "High-difficulty solving discipline:" in client.calls[0]["messages"][1]["content"]


def test_v307_new_explicit_subfacts_are_parent_bound():
    agent = ReasoningAgent(SequenceClient([]))
    for problem, operation in (
        ("Compute 10!", "factorial"),
        ("Compute Fibonacci(20).", "fibonacci"),
        ("Is 97 prime?", "is_prime"),
        ("How many positive divisors does 60 have?", "divisor_count"),
        ("Find F_100 mod 1000.", "fibonacci_mod"),
    ):
        context = agent._score_first_context(problem, {})
        requests = agent._tool_augmented_v306_grounded_auto_tool_requests(
            problem,
            context,
            agent._tool_augmented_verification_requests(problem, context),
        )
        assert any(request.operation == operation for request in requests)


def test_v307_new_probability_and_arithmetic_tools_execute_exactly():
    requests = [
        ToolRequest("combinatorics", "factorial", {"n": 10}),
        ToolRequest("combinatorics", "fibonacci", {"n": 20}),
        ToolRequest("number_theory", "is_prime", {"n": 97}),
        ToolRequest("number_theory", "divisor_count", {"n": 60}),
        ToolRequest("number_theory", "fibonacci_mod", {"n": 100, "modulus": 1000}),
        ToolRequest("probability", "geometric_pmf", {"success": "1/2", "trials": 3}),
        ToolRequest("probability", "negative_binomial_pmf", {"successes": 2, "trials": 4, "p": "1/2"}),
        ToolRequest("probability", "binomial_cdf", {"trials": 4, "p": "1/2", "upper": 1}),
    ]
    executor = SafeAugmentedToolExecutor(extended_tools=True)
    facts = executor.execute(requests[:6]) + executor.execute(requests[6:])
    statements = [fact.statement for fact in facts]
    assert len(statements) == len(requests)
    assert "factorial[n=10] -> 3628800" in statements
    assert "fibonacci[n=20] -> 6765" in statements
    assert "is_prime[n=97] -> True" in statements
    assert "divisor_count[n=60] -> 12" in statements
    assert "fibonacci_mod[n=100; modulus=1000] -> 75" in statements
    assert "geometric_pmf[success=1/2; trials=3] -> 1/8" in statements
    assert "negative_binomial_pmf[successes=2; trials=4; p=1/2] -> 3/16" in statements
    assert "binomial_cdf[trials=4; p=1/2; upper=1] -> 5/16" in statements


def test_v307_does_not_materialize_factorial_used_only_as_modular_exponent():
    agent = ReasoningAgent(SequenceClient([]))
    context = agent._score_first_context(
        "Find the remainder when 2017^(2025!) - 1 is divided by 2025!.",
        {},
    )
    requests = agent._tool_augmented_v306_grounded_auto_tool_requests(
        "Find the remainder when 2017^(2025!) - 1 is divided by 2025!.",
        context,
        agent._tool_augmented_verification_requests(
            "Find the remainder when 2017^(2025!) - 1 is divided by 2025!.",
            context,
        ),
    )
    assert not any(request.operation == "factorial" for request in requests)

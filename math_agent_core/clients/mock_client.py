from __future__ import annotations

import json
from typing import Any, Dict, List


class MockClient:
    model = "mock-intern-s1"

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 8192,
        thinking_mode: bool = True,
    ) -> str:
        text = "\n".join(message.get("content", "") for message in messages)
        result = self._solve_from_text(text)
        return json.dumps(result, ensure_ascii=False)

    def _solve_from_text(self, text: str) -> Dict[str, Any]:
        problem_id = self._extract_problem_id(text)
        if "[[" in text and "]]" in text:
            return self._template_result(problem_id, "linear_algebra", "calculation")
        if "u_t" in text or "u_xx" in text:
            return self._template_result(problem_id, "pde", "derivation")
        if "f(x)" in text or "derivative" in text or "maximum" in text or "minimum" in text:
            return self._template_result(problem_id, "calculus", "calculation")
        return self._template_result(problem_id, "unknown", "unknown")

    def _extract_problem_id(self, text: str) -> str:
        marker = "Problem id:"
        if marker in text:
            return text.split(marker, 1)[1].splitlines()[0].strip() or "UNKNOWN"
        return "UNKNOWN"

    def _template_result(self, problem_id: str, problem_type: str, task_type: str) -> Dict[str, Any]:
        return {
            "problem_id": problem_id,
            "problem_type": problem_type,
            "task_type": task_type,
            "domain_candidates": [problem_type],
            "reasoning_plan": [
                "Parse the problem statement.",
                "Select a suitable math method.",
                "Return a schema-valid offline mock result.",
            ],
            "solution": [
                {
                    "step": 1,
                    "content": "MockClient is an offline test double and does not compute contest answers.",
                }
            ],
            "final_answer": {"answer": "mock_result", "answer_type": "unknown"},
            "verification": {
                "verification_result": "uncertain",
                "verification_process": "Offline mock output is only for pipeline and schema validation.",
                "confidence": 0.75,
            },
            "learning_hints": ["Use Intern-S1 for real solving; use MockClient for offline tests only."],
        }


class ScriptedClient:
    model = "scripted-client"

    def __init__(self, responses: List[Any] | Dict[str, List[Any]]):
        if isinstance(responses, dict):
            self.responses_by_role = {str(key): list(value) for key, value in responses.items()}
            self.responses = []
        else:
            self.responses_by_role = {}
            self.responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 8192,
        thinking_mode: bool = True,
    ) -> str:
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "thinking_mode": thinking_mode,
            }
        )
        role = self._detect_role(messages)
        role_queue = self.responses_by_role.get(role)
        if role_queue:
            response = role_queue.pop(0)
        else:
            default_queue = self.responses_by_role.get("default")
            if default_queue:
                response = default_queue.pop(0)
            elif self.responses:
                response = self.responses.pop(0)
            else:
                return json.dumps(self._default_result(), ensure_ascii=False)
        if isinstance(response, str):
            return response
        return json.dumps(response, ensure_ascii=False)

    def _detect_role(self, messages: List[Dict[str, str]]) -> str:
        system_text = " ".join(message.get("content", "") for message in messages if message.get("role") == "system")
        lowered = system_text.lower()
        if "mathematical critic" in lowered:
            return "critic"
        if "mathematical planning agent" in lowered:
            return "planner"
        if "final answer formatter" in lowered:
            return "finalizer"
        return "solver"

    def _default_result(self) -> Dict[str, Any]:
        return {
            "problem_id": "UNKNOWN",
            "problem_type": "unknown",
            "task_type": "calculation",
            "domain_candidates": ["unknown"],
            "reasoning_plan": ["scripted default"],
            "solution": [{"step": 1, "content": "No scripted response remained."}],
            "final_answer": {"answer": "", "answer_type": "unknown"},
            "verification": {"verification_result": "uncertain", "checks": [], "confidence": 0.0},
        }


class FaultInjectionClient(ScriptedClient):
    model = "fault-injection-client"

    @classmethod
    def invalid_json_then(cls, response: Any) -> "FaultInjectionClient":
        return cls(["{not valid json", response])

    @classmethod
    def wrong_then_fixed(cls, wrong_answer: str, fixed_answer: str) -> "FaultInjectionClient":
        return cls(
            [
                _calculation_response(wrong_answer, confidence=0.95),
                _calculation_response(fixed_answer, confidence=0.95),
            ]
        )


def _calculation_response(answer: str, confidence: float = 0.9) -> Dict[str, Any]:
    return {
        "problem_id": "UNKNOWN",
        "problem_type": "unknown",
        "task_type": "calculation",
        "domain_candidates": ["unknown"],
        "reasoning_plan": ["Compute directly."],
        "solution": [{"step": 1, "content": f"The candidate answer is {answer}."}],
        "final_answer": {"answer": answer, "answer_type": "numeric"},
        "verification": {
            "verification_result": "pass",
            "checks": ["Model self-check claimed the answer is consistent."],
            "confidence": confidence,
        },
    }

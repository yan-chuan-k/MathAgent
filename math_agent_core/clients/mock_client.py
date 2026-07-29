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

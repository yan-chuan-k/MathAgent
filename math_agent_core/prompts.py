from __future__ import annotations

import json
from typing import Any, Dict


BASE_SYSTEM_PROMPT = (
    "You are a rigorous math agent for higher mathematics. "
    "Understand the problem, classify the domain, plan the solution, solve step by step, "
    "verify the result, and output strict JSON only. "
    "Do not output Markdown fences or any text outside JSON. "
    "If the answer is uncertain, set verification_result to uncertain and explain why."
)


def build_solver_messages(problem: Dict[str, Any], problem_text: str) -> list:
    problem_id = problem.get("problem_id", "UNKNOWN")
    schema_hint = {
        "problem_id": str(problem_id),
        "problem_type": "string",
        "task_type": "calculation/proof/derivation/choice/unknown",
        "domain_candidates": ["string"],
        "reasoning_plan": ["string"],
        "solution": [{"step": 1, "content": "string"}],
        "final_answer": {
            "answer": "string",
            "answer_type": "numeric/expression/closed_form/proof/set/interval/matrix/unknown",
        },
        "verification": {
            "verification_result": "pass/fail/uncertain",
            "verification_process": "string",
            "confidence": 0.0,
        },
        "learning_hints": ["string"],
    }
    user_prompt = (
        f"Problem id: {problem_id}\n"
        f"Problem:\n{problem_text}\n\n"
        "Return one JSON object matching this shape. "
        "Keep verification concrete; do not claim pass without a check.\n"
        f"{json.dumps(schema_hint, ensure_ascii=False, indent=2)}"
    )
    return [
        {"role": "system", "content": BASE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

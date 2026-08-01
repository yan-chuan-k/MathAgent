import json
from pathlib import Path

from math_agent_core.json_utils import validate_result
from math_agent_core.schema import normalize_result


def test_normalize_result_accepts_verification_checks_and_new_types():
    raw = {
        "problem_id": "p1",
        "task_type": "construction",
        "domain_candidates": ["topology"],
        "reasoning_plan": ["Use the definition."],
        "solution": [{"step": 1, "content": "Construct the example."}],
        "final_answer": {"answer": "A counterexample", "answer_type": "text"},
        "verification": {
            "verification_result": "pass",
            "checks": ["The construction satisfies the stated conditions."],
            "confidence": 0.9,
        },
        "assumptions": [],
    }
    result = normalize_result(raw, problem_id="p1", model="mock", backend="simple")
    schema = json.loads(Path("result_schema.json").read_text(encoding="utf-8"))
    validation = validate_result(result, schema)

    assert validation.valid
    assert result["verification"]["checks"] == ["The construction satisfies the stated conditions."]
    assert result["verification"]["verification_process"] == "The construction satisfies the stated conditions."

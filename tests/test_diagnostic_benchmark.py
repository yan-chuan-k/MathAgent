import json

import diagnose_hard_cases
from math_agent_core.clients import ScriptedClient


def _correct_response():
    return {
        "problem_id": "0",
        "problem_type": "unknown",
        "task_type": "calculation",
        "domain_candidates": ["unknown"],
        "reasoning_plan": ["Compute directly."],
        "solution": [{"step": 1, "content": "1+1=2"}],
        "final_answer": {"answer": "2", "answer_type": "numeric"},
        "verification": {"verification_result": "pass", "checks": [], "confidence": 0.99},
        "assumptions": [],
        "learning_hints": [],
    }


def test_benchmark_reports_accuracy_and_model_calls_without_leaking_answer(tmp_path, monkeypatch):
    input_path = tmp_path / "benchmark.jsonl"
    output_path = tmp_path / "summary.json"
    input_path.write_text(
        json.dumps({"idx": 0, "problem": "1+1=?", "expected_domain": "unknown", "answer": "2"}) + "\n",
        encoding="utf-8",
    )
    scripted = ScriptedClient([_correct_response()])
    monkeypatch.setattr(diagnose_hard_cases, "build_client", lambda **kwargs: scripted)

    summary = diagnose_hard_cases.evaluate(
        input_file=input_path,
        output_file=output_path,
        use_mock=True,
        run_agent=True,
        thinking_mode=True,
    )

    assert summary["answer_accuracy"] == 1.0
    assert summary["model_calls"] == 1
    assert summary["model_calls_per_problem"] == 1.0
    user_content = scripted.calls[0]["messages"][1]["content"]
    input_payload = user_content.split("INPUT_PAYLOAD_BEGIN", 1)[1].split("INPUT_PAYLOAD_END", 1)[0]
    assert "expected_answer" not in input_payload
    assert '"answer"' not in input_payload

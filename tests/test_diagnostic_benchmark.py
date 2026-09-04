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
    scripted = ScriptedClient([_correct_response()] * 6)
    monkeypatch.setattr(diagnose_hard_cases, "build_client", lambda **kwargs: scripted)

    summary = diagnose_hard_cases.evaluate(
        input_file=input_path,
        output_file=output_path,
        use_mock=True,
        run_agent=True,
        thinking_mode=True,
        production_mode="orchestrated",
    )

    assert summary["answer_accuracy"] == 1.0
    assert summary["model_calls"] == 1
    assert summary["model_calls_per_problem"] == 1.0
    user_content = scripted.calls[0]["messages"][1]["content"]
    input_payload = user_content.split("INPUT_PAYLOAD_BEGIN", 1)[1].split("INPUT_PAYLOAD_END", 1)[0]
    assert "expected_answer" not in input_payload
    assert '"answer"' not in input_payload


def test_benchmark_ground_truth_is_loaded_from_fixture(tmp_path):
    input_path = tmp_path / "fixture.jsonl"
    input_path.write_text(
        json.dumps({"idx": "case", "problem": "1+1=?", "expected_answer": "2"}) + "\n",
        encoding="utf-8",
    )
    items = diagnose_hard_cases.load_jsonl(input_path)
    assert items[0]["expected_answer"] == "2"


def test_benchmark_accepts_grading_field_as_fixture_truth(tmp_path):
    input_path = tmp_path / "grading.jsonl"
    input_path.write_text(
        json.dumps({"idx": "case", "problem": "1+1=?", "grading": {"primary": "2"}}) + "\n",
        encoding="utf-8",
    )
    summary = diagnose_hard_cases.evaluate(
        input_file=input_path,
        output_file=tmp_path / "summary.json",
        use_mock=False,
        run_agent=False,
        thinking_mode=True,
    )
    assert summary["expected_answer_coverage"] == 1.0


def test_full_problem_accuracy_counts_single_claim_problems(tmp_path, monkeypatch):
    path = tmp_path / "single.jsonl"
    path.write_text(json.dumps({"idx": 1, "problem": "Compute 1+1.", "expected_answer": "2"}) + "\n", encoding="utf-8")
    scripted = ScriptedClient([_correct_response()] * 6)
    monkeypatch.setattr(diagnose_hard_cases, "build_client", lambda **kwargs: scripted)
    summary = diagnose_hard_cases.evaluate(
        path, tmp_path / "summary.json", True, True, True, production_mode="orchestrated"
    )
    assert summary["full_problem_accuracy"] == 1.0


def test_full_problem_accuracy_requires_all_gradable_claims(tmp_path, monkeypatch):
    path = tmp_path / "multi.jsonl"
    path.write_text(
        json.dumps({
            "idx": 1,
            "problem": "Find the estimate and state unbiasedness.",
            "expected_answer": {"primary": "2", "required_claims": ["UNBIASED"]},
        }) + "\n",
        encoding="utf-8",
    )
    response = _correct_response()
    response["final_answer"]["answer"] = "2"
    scripted = ScriptedClient([response] * 6)
    monkeypatch.setattr(diagnose_hard_cases, "build_client", lambda **kwargs: scripted)
    summary = diagnose_hard_cases.evaluate(
        path, tmp_path / "summary.json", True, True, True, production_mode="orchestrated"
    )
    assert summary["primary_answer_accuracy"] == 1.0
    assert summary["full_problem_accuracy"] == 0.0
    assert summary["required_claim_unresolved_count"] == 0

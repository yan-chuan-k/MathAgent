from __future__ import annotations

import json
from pathlib import Path

import evaluate_web_hard_stress as stress
from evaluate_web_hard_stress import evaluate, load_jsonl, safe_solver_metadata
from math_agent_core.clients import MockClient
from user_agent import ReasoningAgent


FIXTURE = Path(__file__).resolve().parents[1] / "sample_data" / "web_hard_stress_v1.jsonl"


def test_public_stress_fixture_is_broad_and_oracle_is_not_solver_metadata():
    rows = load_jsonl(FIXTURE)
    assert len(rows) >= 30
    assert len({row["idx"] for row in rows}) == len(rows)
    assert len({row["source_label"].split(" #")[0] for row in rows}) >= 3
    for row in rows:
        metadata = safe_solver_metadata(row)
        assert "grading" not in metadata
        assert "expected_answer" not in metadata
        assert "gold_summary" not in metadata
        assert row["source"].startswith("https://")


def test_public_stress_mock_run_is_explicitly_not_decision_grade():
    summary = evaluate(
        load_jsonl(FIXTURE)[:2],
        run_agent=True,
        use_mock=True,
        thinking_mode=True,
    )
    assert summary["evaluation_mode"] == "mock_pipeline_only"
    assert summary["decision_grade"] is False
    assert summary["total"] == 2


def test_public_stress_hard_hint_adds_discipline_without_gold_answer():
    rows = load_jsonl(FIXTURE)
    row = next(row for row in rows if row["idx"] == "hmmt25_an_09")
    agent = ReasoningAgent(MockClient())
    context = agent._score_first_context(row["problem"], safe_solver_metadata(row))
    assert context["difficulty_tier"] == "high"
    assert "target-and-hypotheses ledger" in context["high_difficulty_card"]
    prompt = agent._build_tool_augmented_v307_final_prompt(row["problem"], context, "")
    assert row["grading"]["primary"] not in prompt[0]["content"]
    assert row["grading"]["primary"] not in prompt[1]["content"]


def test_fixture_is_valid_jsonl_without_duplicate_oracle_shapes():
    for line in FIXTURE.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        assert item["problem"].strip()
        assert item["grading"]["primary_type"] in {"numeric", "symbolic", "manual"}


def test_transport_failure_is_unmeasured_and_not_a_wrong_answer(monkeypatch):
    row = load_jsonl(FIXTURE)[0]

    class ErrorAgent:
        def __init__(self, client, **kwargs):
            self.client = client

        def solve(self, problem, metadata):
            return {
                "final_response": "无法确定",
                "trace": [
                    {
                        "step": "error",
                        "content": "RuntimeError: Intern-S1 request failed after 3 attempts: Connection error.",
                    }
                ],
            }

    monkeypatch.setattr(stress, "_build_client", lambda **kwargs: object())
    monkeypatch.setattr(stress, "ReasoningAgent", ErrorAgent)

    summary = evaluate([row], run_agent=True, use_mock=False, thinking_mode=True)

    assert summary["evaluation_mode"] == "live_model_partial"
    assert summary["decision_grade"] is False
    assert summary["numeric_evaluated"] == 0
    assert summary["numeric_correct"] == 0
    assert summary["transport_error_items"] == 1
    assert summary["unmeasured_items"] == 1
    assert summary["rows"][0]["evaluation_status"] == "transport_error"
    assert summary["rows"][0]["grading"] is None


def test_completed_live_answer_is_graded_after_transport_guard(monkeypatch):
    row = load_jsonl(FIXTURE)[0]

    class AnswerAgent:
        def __init__(self, client, **kwargs):
            self.client = client

        def solve(self, problem, metadata):
            return {
                "final_response": "103 (9! = 362880)",
                "trace": [{"step": "answer", "content": "answer extracted"}],
            }

    monkeypatch.setattr(stress, "_build_client", lambda **kwargs: object())
    monkeypatch.setattr(stress, "ReasoningAgent", AnswerAgent)

    summary = evaluate([row], run_agent=True, use_mock=False, thinking_mode=True)

    assert summary["evaluation_mode"] == "live_model_decision_grade"
    assert summary["decision_grade"] is True
    assert summary["numeric_evaluated"] == 1
    assert summary["numeric_correct"] == 1
    assert summary["transport_error_items"] == 0
    assert summary["rows"][0]["evaluation_status"] == "completed"


def test_client_transport_settings_are_configurable_without_exposing_key(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("INTERN_API_KEY", "test-only-key")
    monkeypatch.setenv("INTERN_API_TIMEOUT", "240")
    monkeypatch.setenv("INTERN_API_RETRY", "5")
    monkeypatch.setattr(stress, "InternS1Client", FakeClient)

    stress._build_client(use_mock=False, thinking_mode=True)

    assert captured["timeout"] == 240
    assert captured["retry"] == 5
    assert "api_key" not in captured


def test_error_classifier_separates_transport_from_solver_errors():
    assert stress._classify_solver_error("Request timed out") == "transport_error"
    assert stress._classify_solver_error("Connection error") == "transport_error"
    assert stress._classify_solver_error("invalid response schema") == "solver_error"
    assert stress._classify_solver_error("") == ""

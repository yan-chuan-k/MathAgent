from __future__ import annotations

import json
from pathlib import Path

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

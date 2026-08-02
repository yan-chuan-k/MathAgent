import json

from math_agent_core import MathAgentOrchestrator
from math_agent_core.clients import FaultInjectionClient, ScriptedClient
from math_agent_core.schema import normalize_result
from user_agent import ReasoningAgent


def _response(answer, *, meta=None, checks=None, confidence=0.95):
    data = {
        "problem_id": "p1",
        "problem_type": "unknown",
        "task_type": "calculation",
        "domain_candidates": ["unknown"],
        "reasoning_plan": ["Compute directly."],
        "solution": [{"step": 1, "content": f"Candidate answer is {answer}."}],
        "final_answer": {"answer": str(answer), "answer_type": "numeric"},
        "verification": {
            "verification_result": "pass",
            "checks": checks or ["model self-check"],
            "confidence": confidence,
        },
        "assumptions": [],
        "learning_hints": [],
    }
    if meta is not None:
        data["_meta"] = meta
    return data


def test_model_meta_is_ignored_by_normalizer():
    raw = _response(
        "2",
        meta={
            "model": "attacker",
            "backend": "lagent",
            "attempts": 999,
            "schema_valid": True,
            "overall_status": "solved",
            "elapsed_seconds": 999,
        },
    )

    result = normalize_result(raw, problem_id="trusted", model="system-model", backend="simple", attempts=2)

    assert result["problem_id"] == "trusted"
    assert result["_meta"]["model"] == "system-model"
    assert result["_meta"]["backend"] == "simple"
    assert result["_meta"]["attempts"] == 2
    assert result["_meta"]["schema_valid"] is False
    assert result["_meta"]["overall_status"] == "uncertain"


def test_sympy_failure_feeds_targeted_repair_and_accepts_fixed_answer():
    client = FaultInjectionClient.wrong_then_fixed("3", "2")
    orchestrator = MathAgentOrchestrator(client=client, max_retries=1, enable_tool_verify=True)

    result = orchestrator.solve("1+1=?", {"idx": 0})

    assert result["final_answer"]["answer"] == "2"
    assert result["_meta"]["overall_status"] == "solved"
    assert result["_meta"]["answer_verified"] is True
    assert len(client.calls) >= 2
    second_prompt = json.dumps(client.calls[1]["messages"], ensure_ascii=False)
    assert "numeric_residual" in second_prompt or "symbolic_contradiction" in second_prompt
    assert "residual" in second_prompt


def test_reasoning_agent_does_not_recover_unverified_raw_output():
    client = ScriptedClient(
        [
            {
                "problem_id": "p1",
                "problem_type": "unknown",
                "task_type": "calculation",
                "domain_candidates": ["unknown"],
                "reasoning_plan": ["No answer."],
                "solution": [],
                "final_answer": {"answer": "", "answer_type": "unknown"},
                "verification": {"verification_result": "uncertain", "checks": [], "confidence": 0.0},
                "assumptions": [],
                "learning_hints": [],
            }
        ]
    )

    agent = ReasoningAgent(client=client, max_retries=0)
    result = agent.solve("1+1=?", {"idx": 0})

    assert result["final_response"]
    assert result["final_response"] != "2"


def test_inconclusive_tool_status_is_not_solved():
    client = ScriptedClient([_response("x^2+1", confidence=0.8)])
    orchestrator = MathAgentOrchestrator(client=client, max_retries=0, enable_tool_verify=True)

    result = orchestrator.solve("Find an antiderivative.", {"idx": 2})

    assert result["_meta"]["overall_status"] in {"probable", "uncertain"}
    assert result["_meta"]["answer_verified"] is False


def test_multi_candidate_ranking_prefers_verified_solution():
    client = ScriptedClient(
        {
            "solver": [
                _response("unverified_text", confidence=0.9),
                _response("2", confidence=0.95),
            ],
            "critic": [
                {"status": "pass", "failure_kind": "", "first_error": "", "missing_targets": [], "suggested_repair": ""},
            ],
        }
    )
    orchestrator = MathAgentOrchestrator(
        client=client,
        max_retries=0,
        enable_tool_verify=True,
        enable_critic=True,
        max_candidates=2,
    )

    result = orchestrator.solve("1+1=?", {"idx": 3})

    assert result["final_answer"]["answer"] == "2"
    assert orchestrator.last_log["route"]["candidate_count"] == 2
    assert orchestrator.last_log["candidates"][0]["score"] >= orchestrator.last_log["candidates"][1]["score"]


def test_critic_failure_rejects_first_candidate_and_uses_next_candidate():
    client = ScriptedClient(
        {
            "solver": [
                _response("2", confidence=0.95),
                _response("2", confidence=0.95),
            ],
            "critic": [
                {
                    "status": "fail",
                    "failure_kind": "missing_case",
                    "first_error": "The answer omits a required case.",
                    "missing_targets": ["case b"],
                    "suggested_repair": "Cover the second case.",
                },
                {"status": "pass", "failure_kind": "", "first_error": "", "missing_targets": [], "suggested_repair": ""},
            ],
        }
    )
    orchestrator = MathAgentOrchestrator(
        client=client,
        max_retries=0,
        enable_tool_verify=True,
        enable_critic=True,
        max_candidates=2,
    )

    result = orchestrator.solve("1+1=?", {"idx": 4})

    assert result["_meta"]["overall_status"] == "solved"
    assert orchestrator.last_log["candidates"][0]["critic_status"] == "pass"
    assert any(candidate["critic_status"] == "fail" for candidate in orchestrator.last_log["candidates"])


def test_finalizer_formats_selected_candidate_without_changing_verification():
    client = ScriptedClient(
        {
            "solver": [_response("2", confidence=0.95)],
            "critic": [{"status": "pass", "failure_kind": "", "first_error": "", "missing_targets": [], "suggested_repair": ""}],
            "finalizer": [{"final_response": "2"}],
        }
    )
    orchestrator = MathAgentOrchestrator(
        client=client,
        max_retries=0,
        enable_tool_verify=True,
        enable_critic=True,
        enable_finalizer=True,
        max_candidates=1,
    )

    result = orchestrator.solve("1+1=?", {"idx": 5})

    assert result["final_response"] == "2"
    assert result["_meta"]["overall_status"] == "solved"
    assert result["_meta"]["answer_verified"] is True

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
        enable_finalizer=False,
        max_candidates=2,
    )

    result = orchestrator.solve("1+1=?", {"idx": 3})

    assert result["final_answer"]["answer"] == "2"
    assert orchestrator.last_log["route"]["candidate_count"] == 2
    assert orchestrator.last_log["candidates"][0]["score"] >= orchestrator.last_log["candidates"][1]["score"]


def test_decisive_tool_pass_skips_critic():
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
        enable_finalizer=False,
        max_candidates=2,
    )

    result = orchestrator.solve("1+1=?", {"idx": 4})

    assert result["_meta"]["overall_status"] == "solved"
    assert len(client.calls) == 1
    assert orchestrator.last_log["candidates"][0]["critic_status"] is None
    evidence = result["verification"]["evidence"]
    assert any(item["verification_level"] == "exact_symbolic" and item["is_decisive"] for item in evidence)
    assert not any(item["verification_level"] == "model_critic" for item in evidence)


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


def _conflicting_response(answer):
    return {
        "problem_id": "p",
        "problem_type": "unknown",
        "task_type": "proof",
        "domain_candidates": ["unknown"],
        "reasoning_plan": ["Derive the claim."],
        "solution": [{"step": 1, "content": "A proof attempt."}],
        "final_answer": {"answer": answer, "answer_type": "proof"},
        "verification": {"verification_result": "uncertain", "checks": [], "confidence": 0.2},
        "assumptions": [],
        "learning_hints": [],
    }


def test_conflict_critic_preferred_b_controls_selection():
    client = ScriptedClient({
        "solver": [_conflicting_response("WRONG_A"), _conflicting_response("RIGHT_B")],
        "critic": [{"preferred_candidate": "B", "repair_candidate": "none", "repair_target": "", "confidence": 0.99}],
    })
    orchestrator = MathAgentOrchestrator(client=client, max_retries=0, max_candidates=2, enable_finalizer=False)
    result = orchestrator.solve("Prove a difficult theorem with many cases.", {"idx": "pref-b"})
    assert result["final_answer"]["answer"] == "RIGHT_B"


def test_conflict_critic_preferred_a_controls_selection():
    client = ScriptedClient({
        "solver": [_conflicting_response("RIGHT_A"), _conflicting_response("WRONG_B")],
        "critic": [{"preferred_candidate": "A", "repair_candidate": "none", "repair_target": "", "confidence": 0.99}],
    })
    orchestrator = MathAgentOrchestrator(client=client, max_retries=0, max_candidates=2, enable_finalizer=False)
    result = orchestrator.solve("Prove a difficult theorem with many cases.", {"idx": "pref-a"})
    assert result["final_answer"]["answer"] == "RIGHT_A"


def test_solver_runtime_configuration_reaches_client():
    client = ScriptedClient([_response("2")])
    orchestrator = MathAgentOrchestrator(
        client=client, max_retries=0, max_candidates=1, enable_finalizer=False,
        solver_max_tokens=4096, solver_temperature=0.37,
    )
    orchestrator.solve("Compute 1+1.", {"idx": "runtime"})
    assert client.calls[0]["max_tokens"] == 4096
    assert client.calls[0]["temperature"] == 0.37


def test_missing_root_cannot_be_solved_by_substitution_only():
    client = ScriptedClient([_response("x=2", confidence=0.95)])
    orchestrator = MathAgentOrchestrator(client=client, max_retries=0, max_candidates=1, enable_finalizer=False)
    result = orchestrator.solve("Solve x^2 - 5*x + 6 = 0 for x.", {"idx": "missing-root"})
    assert result["_meta"]["overall_status"] != "solved"


def test_multi_target_matrix_requires_all_claims():
    response = _response("-2", confidence=0.95)
    response["problem_type"] = "linear_algebra"
    client = ScriptedClient([response])
    orchestrator = MathAgentOrchestrator(client=client, max_retries=0, max_candidates=1, enable_finalizer=False)
    result = orchestrator.solve("Compute the determinant and rank of [[1,2],[3,4]].", {"idx": "multi-target"})
    assert result["_meta"]["overall_status"] != "solved"


def test_single_determinant_system_check_is_decisive():
    response = _response("-2", confidence=0.95)
    response["problem_type"] = "linear_algebra"
    client = ScriptedClient([response])
    orchestrator = MathAgentOrchestrator(client=client, max_retries=0, max_candidates=1, enable_finalizer=False)
    result = orchestrator.solve("Compute determinant of [[1,2],[3,4]].", {"idx": "single-det"})
    assert result["_meta"]["overall_status"] == "solved"
    assert len(client.calls) == 1

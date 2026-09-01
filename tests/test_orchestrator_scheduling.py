from math_agent_core import MathAgentOrchestrator
from math_agent_core.clients import ScriptedClient
from math_agent_core.search import choose_strategy_budget


def _uncertain_response(answer="candidate"):
    return {
        "problem_id": "p",
        "problem_type": "unknown",
        "task_type": "calculation",
        "domain_candidates": ["unknown"],
        "reasoning_plan": ["Attempt a direct solution."],
        "solution": [{"step": 1, "content": "No deterministic verifier applies."}],
        "final_answer": {"answer": answer, "answer_type": "text"},
        "verification": {"verification_result": "uncertain", "checks": [], "confidence": 0.5},
        "assumptions": [],
        "learning_hints": [],
    }


def test_adaptive_strategy_budget():
    assert choose_strategy_budget("calculation", max_candidates=3, verifiability="high") == 1
    assert choose_strategy_budget("calculation", max_candidates=3, verifiability="medium") == 2
    assert choose_strategy_budget("choice", max_candidates=3, verifiability="medium") == 1
    assert choose_strategy_budget("proof", max_candidates=3, verifiability="low") == 2
    assert choose_strategy_budget("unknown", max_candidates=3, verifiability="low") == 3


def test_planner_selection_precedes_default_pool_when_budget_exceeds_two():
    client = ScriptedClient({"planner": [{"selected_strategies": ["counterexample_search"]}]})
    orchestrator = MathAgentOrchestrator(client=client, max_candidates=3)
    route = {
        "primary_domain": "real_analysis",
        "domain_candidates": ["real_analysis"],
        "task_type": "unknown",
        "verifiability": "low",
    }
    problem = {"problem_id": "p", "_route_hint": route}

    selected = orchestrator._select_strategies(problem, "Classify this difficult statement.", route)

    assert selected[0] == "counterexample_search"
    assert len(selected) == 3


def test_planner_is_not_called_when_budget_is_two_or_less():
    client = ScriptedClient({"planner": [{"selected_strategies": ["counterexample_search"]}]})
    orchestrator = MathAgentOrchestrator(client=client, max_candidates=2)
    route = {
        "primary_domain": "real_analysis",
        "domain_candidates": ["real_analysis"],
        "task_type": "proof",
        "verifiability": "low",
    }

    selected = orchestrator._select_strategies({"problem_id": "p"}, "Prove the statement.", route)

    assert len(selected) == 2
    assert client.calls == []


def _call_roles(client):
    return [client._detect_role(call["messages"]) for call in client.calls]


def test_uncertain_high_verifiability_candidate_expands_before_repair_then_reviews_conflict():
    client = ScriptedClient(
        {
            "solver": [_uncertain_response("candidate A"), _uncertain_response("candidate B")],
            "critic": [
                {"status": "pass", "failure_kind": "", "first_error": "", "missing_targets": [], "suggested_repair": ""},
            ],
        }
    )
    orchestrator = MathAgentOrchestrator(
        client=client,
        max_retries=1,
        max_candidates=2,
        enable_critic=True,
        enable_finalizer=False,
    )

    orchestrator.solve("Compute the determinant of matrix A.", {"idx": 1, "subject": "linear algebra"})

    assert _call_roles(client) == ["solver", "solver", "critic"]
    assert orchestrator.last_log["state"]["budget"]["initial_candidate_budget"] == 1
    assert orchestrator.last_log["state"]["budget"]["expanded_after_uncertain"] is True
    assert orchestrator.last_log["repair_history"] == []
    assert orchestrator.last_log["route"]["candidate_count"] == 2
    assert orchestrator.last_log["model_calls"] == {
        "solver": 2,
        "planner": 0,
        "critic": 1,
        "finalizer": 0,
        "total": 3,
    }


def test_matching_uncertain_candidates_do_not_call_critic():
    client = ScriptedClient(
        {"solver": [_uncertain_response("same candidate"), _uncertain_response("same candidate")]}
    )
    orchestrator = MathAgentOrchestrator(
        client=client,
        max_retries=1,
        max_candidates=2,
        enable_critic=True,
        enable_finalizer=False,
    )

    orchestrator.solve("Compute the determinant of matrix A.", {"idx": 3, "subject": "linear algebra"})

    assert _call_roles(client) == ["solver", "solver"]
    assert orchestrator.last_log["repair_history"] == []


def test_uncertain_single_candidate_can_repair():
    client = ScriptedClient(
        {"solver": [_uncertain_response("candidate A"), _uncertain_response("candidate A revised")]}
    )
    orchestrator = MathAgentOrchestrator(
        client=client,
        max_retries=1,
        max_candidates=1,
        enable_critic=False,
        enable_finalizer=False,
    )

    orchestrator.solve("Determine the value of an unsupported expression.", {"idx": 2})

    assert len(client.calls) == 2
    assert len(orchestrator.last_log["repair_history"]) == 1

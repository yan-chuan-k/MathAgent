from math_agent_core import MathAgentOrchestrator
from math_agent_core.clients import ScriptedClient
from math_agent_core.memory import LemmaStore
from math_agent_core.state import SolveState
from math_agent_core.verifiers.completeness import check_completeness, extract_answer_targets


def _result(answer, task_type="calculation", solution=None):
    return {
        "problem_id": "p",
        "problem_type": "unknown",
        "task_type": task_type,
        "domain_candidates": ["unknown"],
        "reasoning_plan": ["plan"],
        "solution": solution if solution is not None else [{"step": 1, "content": "step"}],
        "final_answer": {"answer": answer, "answer_type": "text"},
        "verification": {"verification_result": "pass", "checks": ["self-check"], "confidence": 0.9},
        "assumptions": [],
        "learning_hints": [],
    }


def test_extract_answer_targets_from_multipart_problem():
    targets = extract_answer_targets("Find x. Compute y.")

    names = [target["name"].lower() for target in targets]
    assert any("x" in name for name in names)
    assert any("y" in name for name in names)


def test_completeness_detects_missing_multipart_target():
    evidence = check_completeness("Find x. Compute y.", _result("x = 2"))
    target_evidence = [item for item in evidence if item.claim_id == "answer_targets"][0]

    assert target_evidence.status == "fail"
    assert "y" in (target_evidence.residual or "").lower()


def test_short_proof_is_rejected_as_incomplete():
    evidence = check_completeness("Prove that n+n is even.", _result("True", task_type="proof", solution=[]))
    proof_evidence = [item for item in evidence if item.claim_id == "proof_body"][0]

    assert proof_evidence.status == "fail"


def test_lemma_store_tracks_verified_and_open_lemmas():
    store = LemmaStore()
    store.add_statement("l1", "n+n is even", status="verified", confidence=0.9)
    store.add_statement("l2", "n is arbitrary", status="open", confidence=0.2)
    store.mark_used("l1", "candidate_1")

    compact = store.compact()

    assert compact["verified_lemmas"][0]["lemma_id"] == "l1"
    assert compact["verified_lemmas"][0]["used_by"] == ["candidate_1"]
    assert compact["open_lemmas"][0]["lemma_id"] == "l2"


def test_solve_state_compacts_without_full_problem_text():
    state = SolveState(
        problem="long hidden problem text",
        route={"primary_domain": "unknown"},
        open_goals=["goal"],
        rejected_strategies=["bad_route"],
        budget={"max_candidates": 2},
    )

    compact = state.compact()

    assert "problem" not in compact
    assert compact["open_goals"] == ["goal"]
    assert compact["rejected_strategies"] == ["bad_route"]


def test_orchestrator_state_records_missing_target():
    client = ScriptedClient(
        {
            "solver": [_result("x = 2")],
            "critic": [{"status": "pass", "failure_kind": "", "first_error": "", "missing_targets": [], "suggested_repair": ""}],
        }
    )
    orchestrator = MathAgentOrchestrator(
        client=client,
        max_retries=0,
        max_candidates=1,
        enable_tool_verify=True,
        enable_critic=True,
    )

    result = orchestrator.solve("Find x. Compute y.", {"idx": 0})

    assert result["_meta"]["overall_status"] == "invalid"
    assert result["_meta"]["failure_kind"] == "missing_case"
    assert "y" in " ".join(orchestrator.last_log["state"]["open_goals"]).lower()

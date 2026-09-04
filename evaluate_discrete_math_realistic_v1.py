from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from diagnose_hard_cases import CountingClient
from intern_s1_client import InternS1Client
from math_agent_core.clients import MockClient
from math_agent_core.evaluation.grader import grade_full_problem
from math_agent_core.router import classify_problem
from math_agent_core.verifiers.discrete_math import (
    _extract_final_integer_assertion,
    run_discrete_math_verification,
)
from user_agent import ReasoningAgent


GROUND_TRUTH_FIELDS = {
    "expected_domain",
    "expected_subtype",
    "expected_answer",
    "answer",
    "answer_hint",
    "grading",
    "manual_failure_category",
    "manual_failure_notes",
}

DECISION_SET_NAME = "discrete_math_realistic_eval_v1"
V2_MIN_MEASURED_IMPACT = 2


FAILURE_TAXONOMY = {
    "ROUTING_ERROR",
    "SUBTYPE_ERROR",
    "SOLVER_MODELING_ERROR",
    "SOLVER_ALGEBRA_ARITHMETIC_ERROR",
    "SOLVER_METHOD_ERROR",
    "SOLVER_INCOMPLETE_ANSWER",
    "SOLVER_REPRESENTATION_ERROR",
    "VERIFIER_FALSE_REJECTION",
    "VERIFIER_COVERAGE_GAP",
    "VERIFIER_NONDECISIVE",
    "ACCEPTANCE_ERROR",
    "GRADER_UNRESOLVED",
    "DATASET_OR_GROUND_TRUTH_ISSUE",
    "UNKNOWN",
}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def safe_solver_metadata(item: Dict[str, Any], include_subject: bool = True) -> Dict[str, Any]:
    """Return only non-ground-truth metadata allowed to reach router/solver prompts."""
    metadata: Dict[str, Any] = {
        "idx": str(item.get("idx") or ""),
        "task_type": str(item.get("task_type") or "unknown"),
        "language": str(item.get("language") or ""),
    }
    if include_subject:
        metadata["subject"] = str(item.get("subject") or "")
    leaked = GROUND_TRUTH_FIELDS.intersection(metadata)
    if leaked:
        raise AssertionError(f"ground-truth metadata leak: {sorted(leaked)}")
    return metadata


def route_view(item: Dict[str, Any], include_subject: bool) -> Dict[str, Any]:
    metadata = safe_solver_metadata(item, include_subject=include_subject)
    route = classify_problem(str(item.get("problem") or ""), metadata)
    return {
        "primary_domain": route.get("primary_domain"),
        "domain_candidates": route.get("domain_candidates", []),
        "discrete_subtype": route.get("discrete_subtype"),
        "task_type": route.get("task_type"),
        "difficulty": route.get("difficulty"),
        "domain_correct": route.get("primary_domain") == item.get("expected_domain"),
        "subtype_correct": route.get("discrete_subtype") == item.get("expected_subtype"),
    }


def _empty_verifier_result() -> Dict[str, Any]:
    return {"problem_type": "discrete_math", "requested_checks": []}


def system_evidence(problem: str, answer: str) -> List[Any]:
    return run_discrete_math_verification(problem, answer, _empty_verifier_result())


def _evidence_summary(evidence: Iterable[Any]) -> Dict[str, Any]:
    items = list(evidence)
    decisive_pass = sum(bool(item.is_decisive and item.status == "pass") for item in items)
    decisive_fail = sum(bool(item.is_decisive and item.status == "fail") for item in items)
    nondecisive = sum(bool(not item.is_decisive) for item in items)
    return {
        "triggered": bool(items),
        "count": len(items),
        "decisive_pass_count": decisive_pass,
        "decisive_fail_count": decisive_fail,
        "nondecisive_count": nondecisive,
        "items": [item.to_dict() for item in items],
    }


def _numeric_primary(item: Dict[str, Any]) -> str | None:
    grading = item.get("grading") if isinstance(item.get("grading"), dict) else {}
    if grading.get("primary_type") != "numeric":
        return None
    primary = str(grading.get("primary") or "").strip()
    return primary or None


def theoretical_verifier_profile(item: Dict[str, Any]) -> Dict[str, Any]:
    primary = _numeric_primary(item)
    if primary is None:
        return {"numeric_primary": False, "eligible": False, "evidence": []}
    evidence = system_evidence(str(item.get("problem") or ""), primary)
    return {
        "numeric_primary": True,
        "eligible": bool(evidence),
        "evidence": [entry.to_dict() for entry in evidence],
    }


def answer_surface(answer: str) -> str:
    text = str(answer or "").strip()
    if re.fullmatch(r"[({[]*\s*-?\d+\s*[)}\].,;:!]*", text):
        return "bare_numeric"
    assertion = _extract_final_integer_assertion(text)
    if assertion is None:
        return "other_unsupported"
    if assertion.kind == "count_statement":
        return "count_statement"
    if assertion.kind == "assignment":
        return "assignment"
    if assertion.kind == "congruence":
        return "congruence_statement"
    if assertion.kind == "tree_edge_statement":
        return "graph_statement"
    if assertion.kind == "neutral_numeric":
        if re.search(r"\b(?:answer|result)\b", text, re.IGNORECASE):
            return "generic_answer"
        return "other_supported_numeric"
    return "other_unsupported"


def _build_client(use_mock: bool, thinking_mode: bool) -> Any:
    if use_mock:
        return MockClient()
    if not os.getenv("INTERN_API_KEY"):
        raise RuntimeError(
            "INTERN_API_KEY is not available. A real Solver evaluation cannot be run; "
            "use --mock only for pipeline/statistics validation."
        )
    model = os.getenv("INTERN_MODEL", "intern-s2-preview-397b")
    base_url = os.getenv("INTERN_API_BASE", "https://chat.intern-ai.org.cn/api/v1/")
    return InternS1Client(model=model, base_url=base_url, thinking_mode=thinking_mode)


def _acceptance_from_log(log: Dict[str, Any]) -> Dict[str, Any]:
    final_result = log.get("final_result") if isinstance(log.get("final_result"), dict) else {}
    meta = final_result.get("_meta") if isinstance(final_result.get("_meta"), dict) else {}
    return {
        "overall_status": str(meta.get("overall_status") or "unknown"),
        "answer_verified": bool(meta.get("answer_verified")),
        "content_complete": bool(meta.get("content_complete")),
        "failure_kind": meta.get("failure_kind"),
        "failure_details": str(meta.get("failure_details") or ""),
    }


def _failure_category(
    item: Dict[str, Any],
    route: Dict[str, Any],
    grading: Dict[str, Any] | None,
    acceptance: Dict[str, Any],
    verifier: Dict[str, Any],
    oracle_profile: Dict[str, Any],
    surface: str,
    final_response: str,
) -> Tuple[str | None, str]:
    """Classify end-to-end failure without treating routing diagnostics as causality.

    Route/subtype mismatches are always stored independently on the row. They only
    become the primary failure category when the Solver answer is actually wrong.
    """
    if grading is None:
        return None, "no Solver run; routing mismatches are diagnostic only"
    if grading.get("correct") is None:
        return "GRADER_UNRESOLVED", "frozen grader returned UNRESOLVED"

    correct = grading.get("correct") is True
    decisive_pass = verifier.get("decisive_pass_count", 0) > 0
    decisive_fail = verifier.get("decisive_fail_count", 0) > 0
    solved = acceptance.get("overall_status") == "solved" and bool(acceptance.get("answer_verified"))

    if correct:
        if decisive_fail:
            return "VERIFIER_FALSE_REJECTION", "correct candidate received decisive system FAIL"
        if solved:
            return None, "correct and accepted; any routing mismatch is diagnostic only"
        if decisive_pass:
            return "ACCEPTANCE_ERROR", "correct candidate had decisive PASS but was not accepted"
        if verifier.get("triggered"):
            return "VERIFIER_NONDECISIVE", "correct candidate had only nondecisive system evidence"
        if oracle_profile.get("eligible"):
            note = "correct answer used an unsupported final-answer surface" if surface == "other_unsupported" else "oracle-eligible problem did not trigger on the Solver answer"
            return "VERIFIER_COVERAGE_GAP", note
        return "VERIFIER_COVERAGE_GAP", "correct answer but frozen system verifier has no deterministic problem-template coverage"

    # A false decisive PASS is an acceptance/verifier failure regardless of routing.
    if decisive_pass:
        return "ACCEPTANCE_ERROR", "wrong candidate received decisive system PASS"

    # Only an actually incorrect Solver answer permits routing mismatch to be treated
    # as a plausible causal failure. Correct answers never enter these categories.
    if not route.get("domain_correct"):
        return "ROUTING_ERROR", f"incorrect answer with domain routed to {route.get('primary_domain')}"
    if not route.get("subtype_correct"):
        return "SUBTYPE_ERROR", f"incorrect answer with subtype routed to {route.get('discrete_subtype')}"

    manual = str(item.get("manual_failure_category") or "").strip()
    if manual:
        if manual not in FAILURE_TAXONOMY:
            return "DATASET_OR_GROUND_TRUTH_ISSUE", f"Unsupported manual taxonomy label: {manual}"
        return manual, str(item.get("manual_failure_notes") or "manual classification")

    if not str(final_response or "").strip():
        return "SOLVER_INCOMPLETE_ANSWER", "empty final response"
    return "UNKNOWN", "candidate graded incorrect with correct routing; manual mathematical diagnosis required"


def run_evaluation(
    items: List[Dict[str, Any]],
    run_agent: bool,
    use_mock: bool,
    thinking_mode: bool,
) -> Dict[str, Any]:
    route_with = [route_view(item, True) for item in items]
    route_without = [route_view(item, False) for item in items]
    oracle_profiles = [theoretical_verifier_profile(item) for item in items]

    client = None
    agent = None
    if run_agent:
        client = CountingClient(_build_client(use_mock=use_mock, thinking_mode=thinking_mode))
        agent = ReasoningAgent(client=client, thinking_mode=thinking_mode)

    rows: List[Dict[str, Any]] = []
    acceptance_states: Counter[str] = Counter()
    calls_by_role: Counter[str] = Counter()
    candidate_b = candidate_conflict = candidate_compare = critic = repair = targeted_repair = 0
    actual_trigger = decisive_pass = decisive_fail = nondecisive_cases = 0
    triggers_by_subtype: Counter[str] = Counter()
    surface_counts: Counter[str] = Counter()
    eligible = surface_misses = template_misses = 0
    confusion = Counter()
    failure_counts: Counter[str] = Counter()
    failure_by_subtype: Dict[str, Counter[str]] = defaultdict(Counter)

    for item, route_a, route_b, oracle in zip(items, route_with, route_without, oracle_profiles):
        final_response = ""
        grading = None
        acceptance = {
            "overall_status": "not_run",
            "answer_verified": False,
            "content_complete": False,
            "failure_kind": None,
            "failure_details": "",
        }
        model_calls = 0
        run_metrics: Dict[str, Any] = {}

        if agent is not None and client is not None:
            before = client.total_calls
            result = agent.solve(str(item.get("problem") or ""), safe_solver_metadata(item, include_subject=True))
            model_calls = client.total_calls - before
            final_response = str(result.get("final_response") or "").strip() if isinstance(result, dict) else ""
            grading = grade_full_problem(final_response, item.get("grading"))
            log = getattr(agent.orchestrator, "last_log", {}) if agent.orchestrator is not None else {}
            acceptance = _acceptance_from_log(log if isinstance(log, dict) else {})
            route_log = log.get("route", {}) if isinstance(log, dict) and isinstance(log.get("route"), dict) else {}
            run_metrics = route_log
            acceptance_states[acceptance["overall_status"]] += 1
            candidate_b += int(route_log.get("candidate_b_triggered", 0) or 0)
            critic += int(route_log.get("critic_triggered", 0) or 0)
            repair += int(route_log.get("repair_triggered", 0) or 0)
            targeted_repair += int(route_log.get("targeted_repair_triggered", 0) or 0)
            comparison = route_log.get("candidate_comparison") if isinstance(route_log.get("candidate_comparison"), dict) else {}
            if comparison and comparison.get("agreement") is not None:
                candidate_compare += 1
                candidate_conflict += int(not bool(comparison.get("agreement")))

        verifier_summary = _evidence_summary(system_evidence(str(item.get("problem") or ""), final_response)) if run_agent else {
            "triggered": False,
            "count": 0,
            "decisive_pass_count": 0,
            "decisive_fail_count": 0,
            "nondecisive_count": 0,
            "items": [],
        }
        surface = answer_surface(final_response) if run_agent else "not_run"
        if run_agent:
            surface_counts[surface] += 1
            if verifier_summary["triggered"]:
                actual_trigger += 1
                triggers_by_subtype[str(item.get("expected_subtype"))] += 1
            decisive_pass += int(verifier_summary["decisive_pass_count"] > 0)
            decisive_fail += int(verifier_summary["decisive_fail_count"] > 0)
            nondecisive_cases += int(verifier_summary["nondecisive_count"] > 0 and verifier_summary["decisive_pass_count"] == 0 and verifier_summary["decisive_fail_count"] == 0)

        if oracle.get("eligible"):
            eligible += 1
            if run_agent and not verifier_summary["triggered"]:
                surface_misses += 1
        elif oracle.get("numeric_primary"):
            template_misses += 1

        if run_agent and grading is not None and grading.get("correct") is not None:
            correct = grading.get("correct") is True
            if verifier_summary["decisive_pass_count"] > 0:
                key = "correct_decisive_pass" if correct else "wrong_decisive_pass"
            elif verifier_summary["decisive_fail_count"] > 0:
                key = "correct_decisive_fail" if correct else "wrong_decisive_fail"
            else:
                key = "correct_no_decisive" if correct else "wrong_no_decisive"
            confusion[key] += 1

        category, notes = _failure_category(
            item,
            route_a,
            grading,
            acceptance,
            verifier_summary,
            oracle,
            surface,
            final_response,
        )
        if category:
            failure_counts[category] += 1
            failure_by_subtype[str(item.get("expected_subtype"))][category] += 1

        solver_answer_correct = None if grading is None or grading.get("correct") is None else grading.get("correct") is True
        solver_answer_unresolved = None if grading is None else grading.get("correct") is None
        acceptance_solved = acceptance.get("overall_status") == "solved"
        answer_verified = bool(acceptance.get("answer_verified"))
        answer_surface_miss = bool(run_agent and oracle.get("eligible") and not verifier_summary.get("triggered"))
        problem_template_miss = bool(oracle.get("numeric_primary") and not oracle.get("eligible"))

        rows.append({
            "idx": item.get("idx"),
            "language": item.get("language"),
            "expected_subtype": item.get("expected_subtype"),
            "problem": item.get("problem"),
            "route_domain_correct": bool(route_a.get("domain_correct")),
            "route_subtype_correct": bool(route_a.get("subtype_correct")),
            "route_domain_mismatch": not bool(route_a.get("domain_correct")),
            "route_subtype_mismatch": not bool(route_a.get("subtype_correct")),
            "solver_answer_correct": solver_answer_correct,
            "solver_answer_unresolved": solver_answer_unresolved,
            "acceptance_solved": acceptance_solved,
            "answer_verified": answer_verified,
            "answer_surface_miss": answer_surface_miss,
            "problem_template_verifier_miss": problem_template_miss,
            "route_with_subject": route_a,
            "route_without_subject": route_b,
            "oracle_verifier": oracle,
            "final_response": final_response,
            "grading": grading,
            "acceptance": acceptance,
            "model_calls": model_calls,
            "system_verifier": verifier_summary,
            "answer_surface": surface,
            "failure_category": category,
            "failure_notes": notes,
        })

    total = len(items)
    language_counts = Counter(str(item.get("language")) for item in items)
    subtype_counts = Counter(str(item.get("expected_subtype")) for item in items)

    route_with_domain = sum(int(row["domain_correct"]) for row in route_with)
    route_without_domain = sum(int(row["domain_correct"]) for row in route_without)
    route_with_subtype = sum(int(row["subtype_correct"]) for row in route_with)
    route_without_subtype = sum(int(row["subtype_correct"]) for row in route_without)

    gradings = [row["grading"] for row in rows if row["grading"] is not None]
    primary_evaluable = [g for g in gradings if g["primary"]["correct"] is not None]
    full_evaluable = [g for g in gradings if g["correct"] is not None]
    primary_correct = sum(g["primary"]["correct"] is True for g in primary_evaluable)
    full_correct = sum(g["correct"] is True for g in full_evaluable)
    unresolved = sum(g["correct"] is None for g in gradings)

    by_subtype: Dict[str, Any] = {}
    for subtype in sorted(subtype_counts):
        subset = [row for row in rows if row["expected_subtype"] == subtype]
        subset_gradings = [row["grading"] for row in subset if row["grading"] is not None]
        evaluable = [g for g in subset_gradings if g["correct"] is not None]
        correct = sum(g["correct"] is True for g in evaluable)
        by_subtype[subtype] = {
            "cases": len(subset),
            "correct": correct if run_agent else None,
            "accuracy": (correct / len(evaluable)) if evaluable else None,
            "deterministic_triggers": sum(int(row["system_verifier"]["triggered"]) for row in subset) if run_agent else None,
            "deterministic_trigger_rate": (
                sum(int(row["system_verifier"]["triggered"]) for row in subset) / len(subset)
                if run_agent and subset else None
            ),
            "largest_failure_category": None,
            "largest_failure_count": 0,
        }
        fc = failure_by_subtype.get(subtype, Counter())
        if fc:
            label, count = fc.most_common(1)[0]
            by_subtype[subtype]["largest_failure_category"] = label
            by_subtype[subtype]["largest_failure_count"] = count

    model_calls_total = client.total_calls if client is not None else 0
    if client is not None:
        calls_by_role.update(client.calls_by_role)

    verified_correct = confusion["correct_decisive_pass"]
    caught_wrong = confusion["wrong_decisive_fail"]
    missed_correct = confusion["correct_no_decisive"]
    uncaught_wrong = confusion["wrong_no_decisive"]

    decision_rows = [row for row in rows if row["solver_answer_correct"] is not None]
    incorrect_and_domain_mismatch_count = sum(
        row["solver_answer_correct"] is False and row["route_domain_mismatch"] for row in decision_rows
    )
    incorrect_and_subtype_mismatch_count = sum(
        row["solver_answer_correct"] is False and row["route_subtype_mismatch"] for row in decision_rows
    )
    incorrect_and_route_or_subtype_mismatch_count = sum(
        row["solver_answer_correct"] is False
        and (row["route_domain_mismatch"] or row["route_subtype_mismatch"])
        for row in decision_rows
    )
    subtype_correct_rows = [row for row in decision_rows if row["route_subtype_correct"]]
    subtype_wrong_rows = [row for row in decision_rows if row["route_subtype_mismatch"]]
    accuracy_when_subtype_correct = (
        sum(row["solver_answer_correct"] is True for row in subtype_correct_rows) / len(subtype_correct_rows)
        if subtype_correct_rows else None
    )
    accuracy_when_subtype_wrong = (
        sum(row["solver_answer_correct"] is True for row in subtype_wrong_rows) / len(subtype_wrong_rows)
        if subtype_wrong_rows else None
    )

    required_claim_evaluable_count = sum(
        bool((item.get("grading") or {}).get("required_claims")) for item in items
    )
    method_compliance_evaluable_count = sum(
        bool((item.get("grading") or {}).get("method_checks")) for item in items
    )

    summary = {
        "name": "Discrete Math V1.1 Realistic Evaluation Report",
        "evaluation_mode": "mock_pipeline_validation" if run_agent and use_mock else "real_solver" if run_agent else "routing_and_oracle_only",
        "real_solver_available": bool(os.getenv("INTERN_API_KEY")),
        "solver_accuracy_is_decision_grade": bool(run_agent and not use_mock),
        "total_problems": total,
        "cases_per_subtype": dict(subtype_counts),
        "language_distribution": dict(language_counts),
        "route_accuracy_with_subject": route_with_domain / total if total else None,
        "route_accuracy_without_subject": route_without_domain / total if total else None,
        "subtype_accuracy_with_subject": route_with_subtype / total if total else None,
        "subtype_accuracy_without_subject": route_without_subtype / total if total else None,
        "route_domain_mismatch_count_with_subject": total - route_with_domain,
        "route_subtype_mismatch_count_with_subject": total - route_with_subtype,
        "incorrect_and_domain_mismatch_count": incorrect_and_domain_mismatch_count if run_agent and not use_mock else None,
        "incorrect_and_subtype_mismatch_count": incorrect_and_subtype_mismatch_count if run_agent and not use_mock else None,
        "incorrect_and_route_or_subtype_mismatch_count": incorrect_and_route_or_subtype_mismatch_count if run_agent and not use_mock else None,
        "accuracy_when_subtype_correct": accuracy_when_subtype_correct if run_agent and not use_mock else None,
        "accuracy_when_subtype_wrong": accuracy_when_subtype_wrong if run_agent and not use_mock else None,
        "primary_answer_accuracy": (primary_correct / len(primary_evaluable)) if primary_evaluable else None,
        "full_problem_accuracy": (full_correct / len(full_evaluable)) if full_evaluable else None,
        "strict_accuracy": (full_correct / len(gradings)) if gradings else None,
        "grader_unresolved_rate": (unresolved / len(gradings)) if gradings else None,
        "required_claim_evaluable_count": required_claim_evaluable_count,
        "method_compliance_evaluable_count": method_compliance_evaluable_count,
        "full_problem_metric_note": "full_problem_accuracy collapses to primary-answer grading when no required claims or method checks are encoded.",
        "accuracy_by_subtype": by_subtype,
        "acceptance_state_distribution": dict(acceptance_states),
        "model_calls_total": model_calls_total,
        "model_calls_per_problem": model_calls_total / total if run_agent and total else None,
        "calls_by_role": dict(calls_by_role),
        "candidate_b_trigger_rate": candidate_b / total if run_agent and total else None,
        "candidate_conflict_rate": candidate_conflict / candidate_compare if candidate_compare else 0.0 if run_agent else None,
        "critic_trigger_rate": critic / total if run_agent and total else None,
        "repair_trigger_rate": repair / total if run_agent and total else None,
        "targeted_repair_rate": targeted_repair / total if run_agent and total else None,
        "system_deterministic_trigger_rate": actual_trigger / total if run_agent and total else None,
        "system_deterministic_trigger_count": actual_trigger if run_agent else None,
        "deterministic_trigger_by_subtype": dict(triggers_by_subtype) if run_agent else None,
        "system_decisive_pass_rate": decisive_pass / total if run_agent and total else None,
        "system_decisive_fail_rate": decisive_fail / total if run_agent and total else None,
        "nondecisive_system_evidence_rate": nondecisive_cases / total if run_agent and total else None,
        "wrong_candidate_false_decisive_pass_count": confusion["wrong_decisive_pass"] if run_agent else None,
        "correct_candidate_false_decisive_fail_count": confusion["correct_decisive_fail"] if run_agent else None,
        "verifier_confusion": dict(confusion) if run_agent else None,
        "verified_correct_count": verified_correct if run_agent else None,
        "caught_wrong_count": caught_wrong if run_agent else None,
        "missed_correct_count": missed_correct if run_agent else None,
        "uncaught_wrong_count": uncaught_wrong if run_agent else None,
        "eligible_deterministic_cases": eligible,
        "actual_deterministic_triggers": actual_trigger if run_agent else None,
        "answer_surface_miss_count": surface_misses if run_agent else None,
        "problem_template_verifier_miss_count": template_misses,
        "answer_surface_distribution": dict(surface_counts) if run_agent else None,
        "failure_taxonomy_counts": dict(failure_counts),
        "failure_taxonomy_by_subtype": {key: dict(value) for key, value in failure_by_subtype.items()},
        "rows": rows,
    }
    return summary


def _fmt_pct(value: Any) -> str:
    return "N/A" if value is None else f"{100.0 * float(value):.1f}%"


def _measured_v2_impacts(summary: Dict[str, Any]) -> Dict[str, int]:
    """Return causal impact counts from decision-grade end-to-end rows only."""
    if not summary.get("solver_accuracy_is_decision_grade"):
        return {
            "routing vocabulary improvements": 0,
            "richer combinatorial strategy conditioning": 0,
            "explicit recurrence numeric verifier": 0,
            "generating-function coefficient verifier": 0,
            "CRT / phi / multiplicative-order deterministic checks": 0,
            "answer-surface expansion": 0,
        }

    rows = summary.get("rows") or []
    verifier_categories = {
        "VERIFIER_FALSE_REJECTION",
        "VERIFIER_NONDECISIVE",
        "VERIFIER_COVERAGE_GAP",
        "ACCEPTANCE_ERROR",
    }
    impacts = {
        "routing vocabulary improvements": sum(
            row.get("solver_answer_correct") is False
            and (row.get("route_domain_mismatch") or row.get("route_subtype_mismatch"))
            for row in rows
        ),
        "richer combinatorial strategy conditioning": sum(
            row.get("expected_subtype") == "combinatorial_counting"
            and row.get("solver_answer_correct") is False
            and row.get("failure_category") in {"SOLVER_MODELING_ERROR", "SOLVER_METHOD_ERROR"}
            for row in rows
        ),
        "explicit recurrence numeric verifier": sum(
            row.get("expected_subtype") == "recurrence"
            and row.get("solver_answer_correct") is True
            and not row.get("answer_verified")
            and row.get("failure_category") in verifier_categories
            for row in rows
        ),
        "generating-function coefficient verifier": sum(
            row.get("expected_subtype") == "generating_function"
            and row.get("solver_answer_correct") is True
            and not row.get("answer_verified")
            and row.get("failure_category") in verifier_categories
            for row in rows
        ),
        "CRT / phi / multiplicative-order deterministic checks": sum(
            row.get("expected_subtype") == "number_theory_modular"
            and row.get("solver_answer_correct") is True
            and not row.get("answer_verified")
            and row.get("failure_category") in verifier_categories
            for row in rows
        ),
        "answer-surface expansion": sum(
            row.get("solver_answer_correct") is True
            and not row.get("answer_verified")
            and bool(row.get("answer_surface_miss"))
            for row in rows
        ),
    }
    return {key: int(value) for key, value in impacts.items()}


def build_v2_candidate_table(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    impacts = _measured_v2_impacts(summary)
    specs = {
        "routing vocabulary improvements": ("medium", "low-medium", "none", "Counts only incorrect real rows with a domain/subtype mismatch; diagnostic route misses alone are not Accuracy failures."),
        "richer combinatorial strategy conditioning": ("low-medium", "medium", "none", "Counts only manually diagnosed combinatorial MODELING/METHOD failures."),
        "explicit recurrence numeric verifier": ("low if closed-world", "medium", "none", "Counts only correct-but-unverified recurrence rows with measured verifier/acceptance failure."),
        "generating-function coefficient verifier": ("low-medium", "medium-high", "none", "Counts only correct-but-unverified GF rows with measured verifier/acceptance failure."),
        "CRT / phi / multiplicative-order deterministic checks": ("low if tightly scoped", "medium", "none", "Counts only correct-but-unverified modular rows with measured verifier/acceptance failure."),
        "answer-surface expansion": ("medium-high", "medium", "none", "Counts only correct real answers lost on oracle-eligible problems because the frozen final-answer surface did not trigger."),
    }
    candidates = []
    for name, impact in impacts.items():
        risk, complexity, calls, note = specs[name]
        candidates.append({
            "candidate_improvement": name,
            "measured_failures_plausibly_addressed": impact,
            "estimated_precision_risk": risk,
            "implementation_complexity": complexity,
            "extra_model_call_cost": calls,
            "recommended": False,
            "evidence_note": note,
        })

    if summary.get("solver_accuracy_is_decision_grade"):
        ranked = sorted(candidates, key=lambda row: row["measured_failures_plausibly_addressed"], reverse=True)
        top = ranked[0]["measured_failures_plausibly_addressed"] if ranked else 0
        second = ranked[1]["measured_failures_plausibly_addressed"] if len(ranked) > 1 else 0
        if top >= V2_MIN_MEASURED_IMPACT and top > second:
            ranked[0]["recommended"] = True
        return ranked
    return sorted(candidates, key=lambda row: row["measured_failures_plausibly_addressed"], reverse=True)


def v2_decision(summary: Dict[str, Any]) -> str:
    candidates = summary.get("v2_candidates") or build_v2_candidate_table(summary)
    recommended = [row for row in candidates if row.get("recommended")]
    if not summary.get("solver_accuracy_is_decision_grade") or len(recommended) != 1:
        return "V2 NOT JUSTIFIED YET"
    return f"V2 CANDIDATE: {recommended[0]['candidate_improvement']}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_markdown(summary: Dict[str, Any]) -> str:
    decision_grade = summary.get("solver_accuracy_is_decision_grade")
    rows = summary.get("rows", [])
    failures = [row for row in rows if row.get("failure_category")]
    top10 = failures[:10]
    candidates = build_v2_candidate_table(summary)

    lines = [
        "# Discrete Math V1.1 Realistic Evaluation Report",
        "",
        "## Evaluation gate",
        "",
        f"- Mode: `{summary['evaluation_mode']}`",
        f"- Real Solver available in this environment: `{summary['real_solver_available']}`",
        f"- Solver-accuracy metrics decision-grade: `{decision_grade}`",
        f"- Decision-set SHA-256: `{summary.get('decision_set_sha256') or 'N/A'}`",
        "- Production behavior was not modified for this evaluation.",
    ]
    if not decision_grade:
        lines += [
            "- **Important:** the project MockClient is an offline schema/pipeline test double that returns `mock_result`; its answer accuracy is not used as evidence for a V2 implementation decision.",
            "- Therefore this run validates dataset balance, routing, grading isolation, reporting, acceptance telemetry, and frozen verifier instrumentation, but it does not establish the real Solver failure distribution.",
        ]
    lines += [
        "",
        "## Required metrics",
        "",
        f"- Dataset size: {summary['total_problems']}",
        f"- Cases per subtype: {summary['cases_per_subtype']}",
        f"- English / Chinese: {summary['language_distribution']}",
        f"- Route accuracy with subject: {_fmt_pct(summary['route_accuracy_with_subject'])}",
        f"- Route accuracy without subject: {_fmt_pct(summary['route_accuracy_without_subject'])}",
        f"- Subtype accuracy with subject: {_fmt_pct(summary['subtype_accuracy_with_subject'])}",
        f"- Subtype accuracy without subject: {_fmt_pct(summary['subtype_accuracy_without_subject'])}",
        f"- Route subtype diagnostic mismatches with subject: {summary.get('route_subtype_mismatch_count_with_subject')}",
        f"- Incorrect + domain mismatch count: {summary.get('incorrect_and_domain_mismatch_count')}",
        f"- Incorrect + subtype mismatch count: {summary.get('incorrect_and_subtype_mismatch_count')}",
        f"- Accuracy when subtype correct: {_fmt_pct(summary.get('accuracy_when_subtype_correct'))}",
        f"- Accuracy when subtype wrong: {_fmt_pct(summary.get('accuracy_when_subtype_wrong'))}",
        f"- Primary answer accuracy: {_fmt_pct(summary['primary_answer_accuracy'])}{' (mock-only; not decision-grade)' if not decision_grade and summary['primary_answer_accuracy'] is not None else ''}",
        f"- Full-problem accuracy: {_fmt_pct(summary['full_problem_accuracy'])}{' (mock-only; not decision-grade)' if not decision_grade and summary['full_problem_accuracy'] is not None else ''}",
        f"- Strict accuracy: {_fmt_pct(summary['strict_accuracy'])}{' (mock-only; not decision-grade)' if not decision_grade and summary['strict_accuracy'] is not None else ''}",
        f"- Grader unresolved rate: {_fmt_pct(summary['grader_unresolved_rate'])}",
        f"- Required-claim evaluable cases: {summary.get('required_claim_evaluable_count')}",
        f"- Method-compliance evaluable cases: {summary.get('method_compliance_evaluable_count')}",
        f"- Full-problem metric semantics: {summary.get('full_problem_metric_note')}",
        f"- Acceptance states: {summary['acceptance_state_distribution']}",
        f"- Model calls total: {summary['model_calls_total']}",
        f"- Model calls/problem: {summary['model_calls_per_problem']}",
        f"- Calls by role: {summary['calls_by_role']}",
        f"- Candidate B trigger rate: {_fmt_pct(summary['candidate_b_trigger_rate'])}",
        f"- Candidate conflict rate: {_fmt_pct(summary['candidate_conflict_rate'])}",
        f"- Critic trigger rate: {_fmt_pct(summary['critic_trigger_rate'])}",
        f"- Repair trigger rate: {_fmt_pct(summary['repair_trigger_rate'])}",
        f"- Targeted repair rate: {_fmt_pct(summary['targeted_repair_rate'])}",
        f"- Deterministic trigger rate overall: {_fmt_pct(summary['system_deterministic_trigger_rate'])}",
        f"- Deterministic triggers by subtype: {summary['deterministic_trigger_by_subtype']}",
        f"- Decisive PASS rate: {_fmt_pct(summary['system_decisive_pass_rate'])}",
        f"- Decisive FAIL rate: {_fmt_pct(summary['system_decisive_fail_rate'])}",
        f"- Wrong-candidate false decisive PASS count: {summary['wrong_candidate_false_decisive_pass_count']}",
        f"- Correct-candidate false decisive FAIL count: {summary['correct_candidate_false_decisive_fail_count']}",
        f"- Eligible deterministic cases: {summary['eligible_deterministic_cases']}",
        f"- Answer-surface miss count: {summary['answer_surface_miss_count']}",
        f"- Problem-template verifier miss count: {summary['problem_template_verifier_miss_count']}",
        f"- Failure taxonomy: {summary['failure_taxonomy_counts']}",
        "",
        "## Accuracy / failures by subtype",
        "",
        "| subtype | cases | correct | accuracy | largest failure category | count |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for subtype, data in summary["accuracy_by_subtype"].items():
        lines.append(
            f"| {subtype} | {data['cases']} | {data['correct'] if data['correct'] is not None else 'N/A'} | "
            f"{_fmt_pct(data['accuracy'])} | {data['largest_failure_category'] or '-'} | {data['largest_failure_count']} |"
        )
    lines += [
        "",
        "## Verifier value confusion table",
        "",
        f"`{summary.get('verifier_confusion')}`",
        "",
        f"- verified_correct_count: {summary.get('verified_correct_count')}",
        f"- caught_wrong_count: {summary.get('caught_wrong_count')}",
        f"- missed_correct_count: {summary.get('missed_correct_count')}",
        f"- uncaught_wrong_count: {summary.get('uncaught_wrong_count')}",
        "",
        "## Top 10 observed failure rows",
        "",
    ]
    if not top10:
        lines.append("No failure rows were produced in this view.")
    else:
        for row in top10:
            lines += [
                f"### {row['idx']} — {row['failure_category']}",
                "",
                f"Problem: {row['problem']}",
                "",
                f"Final response: `{row['final_response']}`",
                "",
                f"Diagnosis: {row['failure_notes']}",
                "",
            ]
    lines += [
        "## V2 candidate table",
        "",
        "| candidate improvement | measured failures plausibly addressed | precision risk | complexity | extra model-call cost | recommended? |",
        "|---|---:|---|---|---|---|",
    ]
    for row in candidates:
        lines.append(
            f"| {row['candidate_improvement']} | {row['measured_failures_plausibly_addressed']} | "
            f"{row['estimated_precision_risk']} | {row['implementation_complexity']} | "
            f"{row['extra_model_call_cost']} | {'yes' if row['recommended'] else 'no'} |"
        )
    lines += [
        "",
        "## Decision",
        "",
    ]
    decision = summary.get("v2_decision") or v2_decision(summary)
    if decision_grade:
        lines.append(
            "The decision below is computed only from measured decision-grade end-to-end rows. "
            "A candidate is selected only when it has a unique, non-sparse largest causal impact cluster."
        )
        lines.append("")
        lines.append(f"**{decision}**")
    else:
        lines.append(
            "The required causal input for a V2 choice — a real/near-real Solver failure distribution — was not available in this environment. "
            "MockClient failures are intentionally excluded from the decision. Routing measurements and verifier coverage gaps remain diagnostics, "
            "not Accuracy-failure counts."
        )
        lines.append("")
        lines.append("**V2 NOT JUSTIFIED YET**")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Frozen Discrete Math V1.1 realistic evaluation harness")
    parser.add_argument("--input_file", default="sample_data/discrete_math_realistic_eval_v1.jsonl")
    parser.add_argument("--output_json", default="sample_outputs/discrete_math_realistic_eval_v1_report.json")
    parser.add_argument("--output_md", default="sample_outputs/discrete_math_realistic_eval_v1_report.md")
    parser.add_argument("--run-agent", action="store_true")
    parser.add_argument("--mock", action="store_true", help="Pipeline-only MockClient run; not decision-grade Solver accuracy")
    parser.add_argument("--thinking-mode", action="store_true", default=True)
    args = parser.parse_args()

    input_path = Path(args.input_file)
    items = load_jsonl(input_path)
    summary = run_evaluation(items, run_agent=args.run_agent, use_mock=args.mock, thinking_mode=args.thinking_mode)
    summary["decision_set_name"] = DECISION_SET_NAME
    summary["decision_set_sha256"] = sha256_file(input_path)
    summary["decision_set_role"] = "V2 development/decision set; do not claim unbiased post-V2 generalization on this same set."
    summary["v2_candidates"] = build_v2_candidate_table(summary)
    summary["v2_decision"] = v2_decision(summary)

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(render_markdown(summary), encoding="utf-8")

    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "evaluation_mode",
                    "total_problems",
                    "route_accuracy_with_subject",
                    "route_accuracy_without_subject",
                    "subtype_accuracy_with_subject",
                    "subtype_accuracy_without_subject",
                    "primary_answer_accuracy",
                    "full_problem_accuracy",
                    "strict_accuracy",
                    "grader_unresolved_rate",
                    "model_calls_total",
                    "model_calls_per_problem",
                    "wrong_candidate_false_decisive_pass_count",
                    "correct_candidate_false_decisive_fail_count",
                    "eligible_deterministic_cases",
                    "answer_surface_miss_count",
                    "problem_template_verifier_miss_count",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

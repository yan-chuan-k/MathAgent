from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

from .state import OverallStatus


ALLOWED_VERIFICATION_RESULTS = {"pass", "fail", "uncertain"}
ALLOWED_TASK_TYPES = {
    "calculation",
    "proof",
    "derivation",
    "choice",
    "classification",
    "construction",
    "counterexample",
    "unknown",
}
ALLOWED_ANSWER_TYPES = {
    "numeric",
    "expression",
    "closed_form",
    "proof",
    "set",
    "interval",
    "matrix",
    "vector",
    "function",
    "distribution",
    "choice",
    "boolean",
    "text",
    "unknown",
}


def empty_result(problem_id: str, model: str, backend: str) -> Dict[str, Any]:
    return {
        "problem_id": str(problem_id),
        "problem_type": "unknown",
        "task_type": "unknown",
        "domain_candidates": ["unknown"],
        "reasoning_plan": [],
        "solution": [],
        "final_answer": {
            "answer": "",
            "answer_type": "unknown",
        },
        "verification": {
            "verification_result": "uncertain",
            "verification_process": "",
            "checks": [],
            "evidence": [],
            "confidence": 0.0,
        },
        "assumptions": [],
        "learning_hints": [],
        "_meta": {
            "model": model,
            "backend": backend,
            "attempts": 1,
            "schema_valid": False,
            "schema_error": None,
            "content_complete": False,
            "answer_verified": False,
            "proof_verified": False,
            "overall_status": OverallStatus.UNCERTAIN.value,
            "failure_kind": None,
            "failure_details": "",
            "elapsed_seconds": 0.0,
        },
    }


def normalize_result(
    result: Dict[str, Any],
    problem_id: str,
    model: str,
    backend: str,
    attempts: int = 1,
    elapsed_seconds: float = 0.0,
) -> Dict[str, Any]:
    normalized = empty_result(problem_id, model=model, backend=backend)
    if isinstance(result, dict):
        normalized.update({k: deepcopy(v) for k, v in result.items() if k != "_meta"})

    normalized["problem_id"] = str(problem_id)

    if "problem_type" not in normalized or not normalized["problem_type"]:
        normalized["problem_type"] = "unknown"
    normalized["problem_type"] = str(normalized["problem_type"])

    task_type = str(normalized.get("task_type") or "unknown")
    normalized["task_type"] = task_type if task_type in ALLOWED_TASK_TYPES else "unknown"

    domains = normalized.get("domain_candidates")
    if not isinstance(domains, list) or not domains:
        domains = [normalized["problem_type"] or "unknown"]
    normalized["domain_candidates"] = [str(item) for item in domains]

    plan = normalized.get("reasoning_plan")
    if isinstance(plan, str):
        plan = [plan]
    if not isinstance(plan, list):
        plan = []
    normalized["reasoning_plan"] = [str(item) for item in plan]

    solution = normalized.get("solution")
    if isinstance(solution, str):
        solution = [{"step": 1, "content": solution}]
    if not isinstance(solution, list):
        solution = []
    normalized_steps = []
    for index, step in enumerate(solution, start=1):
        if isinstance(step, dict):
            normalized_steps.append(
                {
                    "step": int(step.get("step") or index),
                    "content": str(step.get("content") or ""),
                }
            )
        else:
            normalized_steps.append({"step": index, "content": str(step)})
    normalized["solution"] = normalized_steps

    final_answer = normalized.get("final_answer")
    if not isinstance(final_answer, dict):
        final_answer = {"answer": str(final_answer or ""), "answer_type": "unknown"}
    answer_type = str(final_answer.get("answer_type") or "unknown")
    normalized["final_answer"] = {
        "answer": str(final_answer.get("answer") or ""),
        "answer_type": answer_type if answer_type in ALLOWED_ANSWER_TYPES else "unknown",
    }

    verification = normalized.get("verification")
    if not isinstance(verification, dict):
        verification = {}
    verification_result = str(verification.get("verification_result") or "uncertain")
    if verification_result not in ALLOWED_VERIFICATION_RESULTS:
        verification_result = "uncertain"
    try:
        confidence = float(verification.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(1.0, max(0.0, confidence))
    checks = verification.get("checks", [])
    if isinstance(checks, str):
        checks = [checks]
    if not isinstance(checks, list):
        checks = []
    verification_process = str(verification.get("verification_process") or "")
    if not verification_process and checks:
        verification_process = "; ".join(str(item) for item in checks[:3])
    normalized["verification"] = {
        "verification_result": verification_result,
        "verification_process": verification_process,
        "checks": [str(item) for item in checks],
        "confidence": confidence,
    }
    evidence = verification.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = []
    normalized["verification"]["evidence"] = [_normalize_evidence_item(item) for item in evidence[:10]]
    if "issues" in verification:
        normalized["verification"]["issues"] = verification["issues"]

    hints = normalized.get("learning_hints")
    if isinstance(hints, str):
        hints = [hints]
    if not isinstance(hints, list):
        hints = []
    normalized["learning_hints"] = [str(item) for item in hints]

    assumptions = normalized.get("assumptions")
    if isinstance(assumptions, str):
        assumptions = [assumptions]
    if not isinstance(assumptions, list):
        assumptions = []
    normalized["assumptions"] = [str(item) for item in assumptions]

    requested_checks = normalized.get("requested_checks", [])
    if not isinstance(requested_checks, list):
        requested_checks = []
    normalized["requested_checks"] = [_normalize_requested_check(item) for item in requested_checks[:5]]

    normalized["_meta"] = {
        "model": str(model),
        "backend": str(backend),
        "attempts": int(attempts),
        "schema_valid": False,
        "schema_error": None,
        "content_complete": bool(normalized["final_answer"]["answer"].strip()),
        "answer_verified": False,
        "proof_verified": False,
        "overall_status": OverallStatus.UNCERTAIN.value,
        "failure_kind": None,
        "failure_details": "",
        "elapsed_seconds": float(elapsed_seconds),
    }
    return normalized


def _normalize_evidence_item(item: Any) -> Dict[str, Any]:
    if not isinstance(item, dict):
        item = {"details": str(item)}
    status = str(item.get("status") or "inconclusive")
    if status not in {"pass", "fail", "inconclusive"}:
        status = "inconclusive"
    assumptions = item.get("assumptions", [])
    if isinstance(assumptions, str):
        assumptions = [assumptions]
    if not isinstance(assumptions, list):
        assumptions = []
    return {
        "verifier": str(item.get("verifier") or "unknown")[:80],
        "claim_id": str(item.get("claim_id") or "claim")[:80],
        "status": status,
        "method": str(item.get("method") or "unknown")[:120],
        "details": str(item.get("details") or "")[:500],
        "residual": None if item.get("residual") is None else str(item.get("residual"))[:300],
        "assumptions": [str(value)[:200] for value in assumptions[:5]],
    }


def _normalize_requested_check(item: Any) -> Dict[str, Any]:
    allowed_tools = {
        "symbolic_equivalence",
        "equation_solution",
        "numeric_arithmetic",
        "derivative_check",
        "integral_check",
    }
    if not isinstance(item, dict):
        return {"tool": "numeric_arithmetic", "arguments": {}}
    tool = str(item.get("tool") or "")
    if tool not in allowed_tools:
        tool = "numeric_arithmetic"
    arguments = item.get("arguments")
    if not isinstance(arguments, dict):
        arguments = {}
    safe_arguments = {str(key)[:80]: str(value)[:240] for key, value in arguments.items()}
    return {"tool": tool, "arguments": safe_arguments}

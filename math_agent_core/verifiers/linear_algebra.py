from __future__ import annotations

import ast
import re
from typing import Any, Dict, List

from math_agent_core.state import EvidenceStatus, VerificationEvidence, VerificationLevel
from math_agent_core.tools.matrix_tool import MatrixTool


MATRIX_TOOLS = {
    "matrix_determinant",
    "matrix_multiply",
    "matrix_inverse",
    "linear_system_residual",
    "matrix_rank",
    "eigenpair_residual",
    "vector_orthogonality",
    "vector_normalization",
    "matrix_equivalence",
    "vector_equivalence",
}


# Candidate-requested matrix checks are useful supporting evidence, but they
# must never be treated as an independent verification of the candidate's
# final answer.  Keep this marker stable so downstream traces and tests can
# identify the provenance of the evidence.
_CANDIDATE_AUXILIARY_DETAILS = "Candidate-proposed matrix check; treated as supporting evidence only."


def run_linear_algebra_verification(result: Dict[str, Any]) -> List[VerificationEvidence]:
    checks = _extract_matrix_checks(result)
    if not checks:
        return [
            VerificationEvidence(
                verifier="linear_algebra",
                claim_id="no_matrix_check",
                status=EvidenceStatus.INCONCLUSIVE.value,
                method="requested_check_dispatch",
                details="No matrix requested_checks were supplied.",
                verification_level=VerificationLevel.EXACT_SYMBOLIC.value,
                is_decisive=False,
            )
        ]
    tool = MatrixTool()
    evidence: List[VerificationEvidence] = []
    for index, check in enumerate(checks[:4], start=1):
        payload = {
            "tool": check["tool"],
            "arguments": check["arguments"],
            "claim_id": check.get("claim_id") or f"matrix_check_{index}",
        }
        item = tool.run(payload)
        # MatrixTool is also used directly for trusted/system-inferred checks
        # and therefore reports exact checks as decisive.  At this boundary,
        # however, every check came from the candidate's requested_checks and
        # is consequently auxiliary only.
        item.is_decisive = False
        details = str(item.details or "").strip()
        if _CANDIDATE_AUXILIARY_DETAILS not in details:
            item.details = f"{details} {_CANDIDATE_AUXILIARY_DETAILS}".strip()
        evidence.append(item)
    return evidence


def run_system_inferred_matrix_verification(problem_text: str, answer: str) -> List[VerificationEvidence]:
    """Infer only unambiguous matrix tasks from the problem text.

    Unlike ``run_linear_algebra_verification``, these checks are system-owned
    and may be decisive. Candidate-requested checks remain auxiliary.
    """
    text = str(problem_text or "")
    matrix_match = re.search(r"\[\s*\[[^\]]+\](?:\s*,\s*\[[^\]]+\])+\s*\]", text)
    if not matrix_match:
        return []
    try:
        matrix = ast.literal_eval(matrix_match.group(0))
    except Exception:
        return []
    normalized_answer = str(answer or "").strip()
    if not normalized_answer or any(ch.isalpha() for ch in normalized_answer):
        return []
    lower = text.lower()
    tool_name = None
    if "determinant" in lower or "det(" in lower or "行列式" in text:
        tool_name = "matrix_determinant"
        args = {"matrix": matrix, "expected": normalized_answer}
    elif "rank" in lower:
        tool_name = "matrix_rank"
        args = {"matrix": matrix, "expected": normalized_answer}
    else:
        return []
    evidence = MatrixTool().run({"tool": tool_name, "arguments": args, "claim_id": "system_matrix_check"})
    evidence.details = f"System-inferred {tool_name} check; exact verification."
    evidence.is_decisive = True
    return [evidence]


def _extract_matrix_checks(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    requested = result.get("requested_checks") if isinstance(result, dict) else None
    if not isinstance(requested, list):
        return []
    checks: List[Dict[str, Any]] = []
    for item in requested:
        if not isinstance(item, dict):
            continue
        tool = str(item.get("tool") or "")
        arguments = item.get("arguments")
        if tool in MATRIX_TOOLS and isinstance(arguments, dict):
            checks.append({"tool": tool, "arguments": arguments, "claim_id": item.get("claim_id")})
    return checks

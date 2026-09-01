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
    targets = []
    if "determinant" in lower or "det(" in lower or "行列式" in text:
        targets.append("determinant")
    if re.search(r"\brank\b", lower) or "秩" in text:
        targets.append("rank")
    if not targets:
        return []
    evidence: List[VerificationEvidence] = []
    if len(targets) > 1:
        missing = [target for target in targets if target not in lower]
        # A bare scalar cannot establish a multi-target response.
        mentioned = [target for target in targets if re.search(rf"{target}\s*[:=]", lower)]
        if len(mentioned) < len(targets):
            evidence.append(VerificationEvidence(
                verifier="matrix_tool", claim_id="system_matrix_targets",
                status=EvidenceStatus.FAIL.value, method="target_coverage",
                details=f"Multiple matrix targets requested: {targets}.", residual=", ".join(targets),
                verification_level=VerificationLevel.EXACT_SYMBOLIC.value,
                is_decisive=True, claim_scope="subclaim",
            ))
            return evidence
    all_pass = True
    if "determinant" in targets:
        expected = _extract_labeled_number(answer, "determinant") if len(targets) > 1 else normalized_answer
        if expected:
            item = MatrixTool().run({"tool": "matrix_determinant", "arguments": {"matrix": matrix, "expected": expected}, "claim_id": "system_matrix_determinant"})
            item.details = "System-inferred matrix determinant check; exact verification."
            item.is_decisive = True
            item.claim_scope = "full_answer" if len(targets) == 1 else "subclaim"
            evidence.append(item)
            all_pass = all_pass and item.status == EvidenceStatus.PASS.value
        else:
            all_pass = False
    if "rank" in targets:
        expected = _extract_labeled_number(answer, "rank") if len(targets) > 1 else normalized_answer
        if expected:
            item = MatrixTool().run({"tool": "matrix_rank", "arguments": {"matrix": matrix, "expected": expected}, "claim_id": "system_matrix_rank"})
            item.details = "System-inferred matrix rank check; exact verification."
            item.is_decisive = True
            item.claim_scope = "full_answer" if len(targets) == 1 else "subclaim"
            evidence.append(item)
            all_pass = all_pass and item.status == EvidenceStatus.PASS.value
        else:
            all_pass = False
    if len(targets) > 1 and all_pass:
        evidence.append(VerificationEvidence(
            verifier="matrix_tool", claim_id="system_matrix_targets_complete",
            status=EvidenceStatus.PASS.value, method="target_coverage",
            details="All system-inferred matrix targets were verified.", residual=None,
            verification_level=VerificationLevel.EXACT_SYMBOLIC.value,
            is_decisive=True, claim_scope="full_answer",
        ))
    return evidence


def _extract_labeled_number(answer: str, label: str) -> str:
    match = re.search(rf"{label}\s*[:=]\s*([+\-]?\d+(?:\.\d+)?)", str(answer or ""), flags=re.IGNORECASE)
    return match.group(1) if match else ""


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

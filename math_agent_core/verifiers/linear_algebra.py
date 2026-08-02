from __future__ import annotations

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
        evidence.append(tool.run(payload))
    return evidence


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

from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List, Sequence

from math_agent_core.state import EvidenceStatus, VerificationEvidence, VerificationLevel
from math_agent_core.tools.matrix_tool import MatrixTool
from .target_vocab import target_aliases, target_present


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


def run_system_inferred_matrix_verification(
    problem_text: str,
    answer: str,
    *,
    allowed_targets: Sequence[str] | None = None,
) -> List[VerificationEvidence]:
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
    if not normalized_answer:
        return []
    lower = text.lower()
    if allowed_targets is None:
        # Legacy callers retain historical text inference.  The active V3
        # worker supplies an explicit target enum so background mentions cannot
        # become answer targets.
        targets = []
        if target_present(text, "determinant") or "det(" in lower:
            targets.append("determinant")
        if target_present(text, "rank"):
            targets.append("rank")
    else:
        mapping = {
            "matrix_determinant": "determinant",
            "matrix_rank": "rank",
        }
        if not isinstance(allowed_targets, (list, tuple)):
            return []
        if any(target not in mapping for target in allowed_targets):
            return []
        targets = []
        for target in allowed_targets:
            resolved = mapping[target]
            if resolved not in targets:
                targets.append(resolved)
    if not targets:
        return []
    evidence: List[VerificationEvidence] = []
    matrix_tool = MatrixTool()

    def run_system_check(payload: Dict[str, Any]) -> VerificationEvidence:
        # The V3 worker passes ``allowed_targets`` and is itself under a
        # killable parent deadline.  Avoid a nested ThreadPoolExecutor there;
        # legacy text-inferred callers retain MatrixTool.run's local timeout.
        if allowed_targets is not None:
            return matrix_tool.run_under_parent_deadline(payload)
        return matrix_tool.run(payload)

    if len(targets) > 1:
        # A bare scalar cannot establish a multi-target response.
        answer_lower = normalized_answer.lower()
        mentioned = [target for target in targets if _extract_labeled_number(normalized_answer, target)]
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
        expected = _extract_labeled_number(answer, "determinant") if (len(targets) > 1 or re.search(r"[A-Za-z]", normalized_answer)) else normalized_answer
        if expected:
            item = run_system_check({"tool": "matrix_determinant", "arguments": {"matrix": matrix, "expected": expected}, "claim_id": "system_matrix_determinant"})
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
            item = run_system_check({"tool": "matrix_rank", "arguments": {"matrix": matrix, "expected": expected}, "claim_id": "system_matrix_rank"})
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


def run_grounded_matrix_verification(
    answer: str,
    *,
    requests: Sequence[Dict[str, Any]],
) -> List[VerificationEvidence]:
    """Verify matrices supplied explicitly by the parent binding layer.

    This active V3 path receives no problem text and never searches for a
    matrix.  Every determinant/rank calculation is tied to the exact nested
    list in its request.
    """

    if not isinstance(answer, str) or not isinstance(requests, (list, tuple)):
        return []
    matrix_tool = MatrixTool()
    requests = list(requests)[:4]
    evidence: List[VerificationEvidence] = []
    for index, request in enumerate(requests, start=1):
        if not isinstance(request, dict):
            continue
        kind = request.get("kind")
        if kind not in {"matrix_determinant", "matrix_rank"}:
            continue
        if len(requests) == 1:
            expected = _extract_strict_scalar(answer)
        else:
            expected = _extract_labeled_number(
                answer,
                "determinant" if kind == "matrix_determinant" else "rank",
            )
        if not expected:
            continue
        tool_name = "matrix_determinant" if kind == "matrix_determinant" else "matrix_rank"
        try:
            item = matrix_tool.run_under_parent_deadline(
                {
                    "tool": tool_name,
                    "arguments": {"matrix": request["matrix"], "expected": expected},
                    "claim_id": f"grounded_{kind}_{index}",
                }
            )
        except Exception as exc:
            item = VerificationEvidence(
                verifier="matrix_tool",
                claim_id=f"grounded_{kind}_{index}",
                status=EvidenceStatus.INCONCLUSIVE.value,
                method=tool_name,
                details=f"{type(exc).__name__}: {str(exc)[:220]}",
                verification_level=VerificationLevel.EXACT_SYMBOLIC.value,
                is_decisive=False,
            )
        item.details = (
            f"Exact {kind} check for matrix={request['matrix']} "
            f"(compact={json.dumps(request['matrix'], separators=(',', ':'))}). "
            "The matrix was bound by the parent request span."
        )
        item.is_decisive = True
        item.claim_scope = "full_answer" if len(requests) == 1 else "subclaim"
        evidence.append(item)
    return evidence[:8]


def _extract_strict_scalar(answer: str) -> str:
    text = str(answer or "").strip().rstrip(".")
    if re.fullmatch(r"[+\-]?\d+(?:\.\d+)?", text):
        return text
    return ""


def _extract_labeled_number(answer: str, label: str) -> str:
    aliases = "(?:" + "|".join(re.escape(alias) for alias in target_aliases(label)) + ")"
    match = re.search(rf"{aliases}\s*[:=]\s*([+\-]?\d+(?:\.\d+)?)", str(answer or ""), flags=re.IGNORECASE)
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

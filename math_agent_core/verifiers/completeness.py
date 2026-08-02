from __future__ import annotations

import re
from typing import Any, Dict, List

from math_agent_core.state import EvidenceStatus, VerificationEvidence, VerificationLevel


def check_completeness(problem_text: str, result: Dict[str, Any]) -> List[VerificationEvidence]:
    answer = _answer_text(result)
    task_type = str(result.get("task_type") or "unknown") if isinstance(result, dict) else "unknown"
    targets = extract_answer_targets(problem_text)
    evidence: List[VerificationEvidence] = []
    if targets:
        covered, missing = _target_coverage(answer, targets)
        status = EvidenceStatus.PASS.value if not missing else EvidenceStatus.FAIL.value
        evidence.append(
            VerificationEvidence(
                verifier="completeness",
                claim_id="answer_targets",
                status=status,
                method="target_coverage",
                details=f"covered={covered}; missing={missing}",
                residual=", ".join(missing) if missing else None,
                verification_level=VerificationLevel.COMPLETENESS_ONLY.value,
                is_decisive=bool(missing),
            )
        )
    else:
        evidence.append(
            VerificationEvidence(
                verifier="completeness",
                claim_id="answer_targets",
                status=EvidenceStatus.INCONCLUSIVE.value,
                method="target_extraction",
                details="No explicit answer targets were extracted.",
                verification_level=VerificationLevel.COMPLETENESS_ONLY.value,
                is_decisive=False,
            )
        )

    if task_type == "proof" or _looks_like_proof_problem(problem_text):
        evidence.append(_proof_completeness(problem_text, answer, result))
    return evidence


def extract_answer_targets(problem_text: str) -> List[Dict[str, str]]:
    text = str(problem_text or "")
    targets: List[Dict[str, str]] = []
    for index, part in enumerate(_split_multipart(text), start=1):
        target = _extract_target_from_part(part)
        if target:
            targets.append({"name": target, "symbol": target, "part": str(index)})
    if not targets:
        target = _extract_target_from_part(text)
        if target:
            targets.append({"name": target, "symbol": target, "part": "1"})
    return _dedupe_targets(targets)


def _split_multipart(text: str) -> List[str]:
    parts = re.split(
        r"(?:\(\s*[0-9ivxIVX]+\s*\)|\b[0-9]+\s*[.)]|[.;；]\s*(?=(?:find|compute|prove|show|determine|calculate)\b))",
        text,
        flags=re.IGNORECASE,
    )
    return [part.strip() for part in parts if part.strip()]


def _extract_target_from_part(part: str) -> str:
    patterns = [
        r"\b(?:find|compute|calculate|determine)\s+(?:the\s+)?([^.;,?\n]+)",
        r"\b(?:prove|show)\s+(?:that\s+)?([^.;,?\n]+)",
        r"(?:求|计算|证明|判断|确定)([^。；;，,\n]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, part, flags=re.IGNORECASE)
        if match:
            return _clean_target(match.group(1))
    symbol_match = re.search(r"\b([A-Z][A-Za-z0-9_]*|[a-z])\s*[=：:]", part)
    if symbol_match:
        return symbol_match.group(1)
    return ""


def _clean_target(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .,:;?，。；：")
    return text[:120]


def _dedupe_targets(targets: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    result = []
    for target in targets:
        key = target["name"].lower()
        if key not in seen:
            seen.add(key)
            result.append(target)
    return result[:8]


def _target_coverage(answer: str, targets: List[Dict[str, str]]) -> tuple[List[str], List[str]]:
    answer_lower = answer.lower()
    covered: List[str] = []
    missing: List[str] = []
    for target in targets:
        name = target["name"]
        symbol = target.get("symbol") or name
        tokens = _target_tokens(name, symbol)
        if any(token and token.lower() in answer_lower for token in tokens):
            covered.append(name)
        elif len(targets) == 1 and answer.strip():
            covered.append(name)
        else:
            missing.append(name)
    return covered, missing


def _target_tokens(name: str, symbol: str) -> List[str]:
    tokens = [name, symbol]
    tokens.extend(re.findall(r"[A-Za-z][A-Za-z0-9_]*", name))
    return [token for token in tokens if len(token) >= 1]


def _proof_completeness(problem_text: str, answer: str, result: Dict[str, Any]) -> VerificationEvidence:
    stripped = answer.strip()
    lowered = stripped.lower()
    weak_markers = {"true", "false", "holds", "成立", "正确", "yes", "no"}
    if not stripped or lowered in weak_markers:
        return VerificationEvidence(
            verifier="completeness",
            claim_id="proof_body",
            status=EvidenceStatus.FAIL.value,
            method="proof_length_and_markers",
            details="Proof task answer is empty or only states a conclusion.",
            verification_level=VerificationLevel.COMPLETENESS_ONLY.value,
            is_decisive=True,
        )
    if len(stripped) < 80 and not any(marker in lowered for marker in ("because", "therefore", "hence", "since", "proof", "因", "故", "所以")):
        return VerificationEvidence(
            verifier="completeness",
            claim_id="proof_body",
            status=EvidenceStatus.FAIL.value,
            method="proof_length_and_markers",
            details="Proof task answer is too short to establish the requested claim.",
            verification_level=VerificationLevel.COMPLETENESS_ONLY.value,
            is_decisive=True,
        )
    solution = result.get("solution") if isinstance(result, dict) else []
    if isinstance(solution, list) and len(solution) >= 1:
        return VerificationEvidence(
            verifier="completeness",
            claim_id="proof_body",
            status=EvidenceStatus.PASS.value,
            method="proof_structure",
            details="Proof answer includes a body and structured solution steps.",
            verification_level=VerificationLevel.COMPLETENESS_ONLY.value,
            is_decisive=False,
        )
    return VerificationEvidence(
        verifier="completeness",
        claim_id="proof_body",
        status=EvidenceStatus.INCONCLUSIVE.value,
        method="proof_structure",
        details="Could not confirm proof structure from the candidate.",
        verification_level=VerificationLevel.COMPLETENESS_ONLY.value,
        is_decisive=False,
    )


def _looks_like_proof_problem(problem_text: str) -> bool:
    lowered = str(problem_text or "").lower()
    return any(marker in lowered for marker in ("prove", "show that", "证明", "证得"))


def _answer_text(result: Dict[str, Any]) -> str:
    final_answer = result.get("final_answer") if isinstance(result, dict) else None
    if isinstance(final_answer, dict):
        return str(final_answer.get("answer") or "")
    return ""

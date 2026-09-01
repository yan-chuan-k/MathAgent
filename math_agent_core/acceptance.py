from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .state import EvidenceStatus, FailureKind, OverallStatus, VerificationEvidence, VerificationLevel


DECISIVE_PASS_LEVELS = {
    VerificationLevel.FORMAL.value,
    VerificationLevel.EXACT_SYMBOLIC.value,
    VerificationLevel.EXACT_ENUMERATION.value,
}

NUMERIC_PASS_LEVELS = {
    VerificationLevel.HIGH_PRECISION_NUMERIC.value,
}

MODEL_ONLY_LEVELS = {
    VerificationLevel.MODEL_CRITIC.value,
    VerificationLevel.COMPLETENESS_ONLY.value,
}


@dataclass
class AcceptanceDecision:
    overall_status: str
    answer_verified: bool
    proof_verified: bool
    failure_kind: Optional[str] = None
    failure_details: str = ""


class AcceptancePolicy:
    def decide(
        self,
        *,
        schema_valid: bool,
        content_complete: bool,
        task_type: str,
        answer_type: str,
        model_verification_pass: bool,
        evidence: List[VerificationEvidence],
        schema_error: Optional[str] = None,
    ) -> AcceptanceDecision:
        if not schema_valid:
            return AcceptanceDecision(
                overall_status=OverallStatus.INVALID.value,
                answer_verified=False,
                proof_verified=False,
                failure_kind=FailureKind.SCHEMA.value,
                failure_details=schema_error or "schema validation failed",
            )
        if not content_complete:
            return AcceptanceDecision(
                overall_status=OverallStatus.INVALID.value,
                answer_verified=False,
                proof_verified=False,
                failure_kind=FailureKind.WRONG_FINAL_ANSWER.value,
                failure_details="final answer or solution is empty",
            )

        decisive_fail = self._first_decisive_fail(evidence)
        if decisive_fail is not None:
            return AcceptanceDecision(
                overall_status=OverallStatus.INVALID.value,
                answer_verified=False,
                proof_verified=False,
                failure_kind=self._failure_kind_from_evidence(decisive_fail),
                failure_details=self._evidence_summary(decisive_fail),
            )

        completeness_fail = self._first_completeness_fail(evidence)
        if completeness_fail is not None:
            return AcceptanceDecision(
                overall_status=OverallStatus.INVALID.value,
                answer_verified=False,
                proof_verified=False,
                failure_kind=FailureKind.MISSING_CASE.value,
                failure_details=self._evidence_summary(completeness_fail),
            )

        decisive_pass = self._has_decisive_pass(evidence)
        numeric_pass = self._has_numeric_pass(evidence)
        critic_pass = self._has_level_pass(evidence, VerificationLevel.MODEL_CRITIC.value) or model_verification_pass
        completeness_pass = self._has_level_pass(evidence, VerificationLevel.COMPLETENESS_ONLY.value)

        if task_type == "proof" or answer_type == "proof":
            if completeness_pass and critic_pass:
                return AcceptanceDecision(
                    overall_status=OverallStatus.PROBABLE.value,
                    answer_verified=False,
                    proof_verified=True,
                    failure_kind=FailureKind.INCONCLUSIVE.value,
                    failure_details="proof has completeness and critic support but no formal verification",
                )
            return AcceptanceDecision(
                overall_status=OverallStatus.UNCERTAIN.value,
                answer_verified=False,
                proof_verified=False,
                failure_kind=FailureKind.INCONCLUSIVE.value,
                failure_details="proof lacks required completeness or independent critic support",
            )

        if decisive_pass:
            return AcceptanceDecision(
                overall_status=OverallStatus.SOLVED.value,
                answer_verified=True,
                proof_verified=False,
            )
        if numeric_pass:
            return AcceptanceDecision(
                overall_status=OverallStatus.PROBABLE.value,
                answer_verified=False,
                proof_verified=False,
                failure_kind=FailureKind.INCONCLUSIVE.value,
                failure_details="only high precision numeric evidence is available",
            )
        if critic_pass:
            return AcceptanceDecision(
                overall_status=OverallStatus.PROBABLE.value,
                answer_verified=False,
                proof_verified=False,
                failure_kind=FailureKind.INCONCLUSIVE.value,
                failure_details="only model critic or model self-check support is available",
            )
        return AcceptanceDecision(
            overall_status=OverallStatus.UNCERTAIN.value,
            answer_verified=False,
            proof_verified=False,
            failure_kind=FailureKind.INCONCLUSIVE.value,
            failure_details="no applicable decisive verifier evidence",
        )

    def _first_decisive_fail(self, evidence: List[VerificationEvidence]) -> Optional[VerificationEvidence]:
        for item in evidence:
            if item.status == EvidenceStatus.FAIL.value and item.is_decisive:
                return item
        return None

    def _first_completeness_fail(self, evidence: List[VerificationEvidence]) -> Optional[VerificationEvidence]:
        for item in evidence:
            if item.status == EvidenceStatus.FAIL.value and item.verification_level == VerificationLevel.COMPLETENESS_ONLY.value:
                return item
        return None

    def _has_decisive_pass(self, evidence: List[VerificationEvidence]) -> bool:
        return any(
            item.status == EvidenceStatus.PASS.value
            and item.is_decisive
            and (item.claim_scope == "full_answer" or getattr(item, "verifier", "") == "tool")
            and item.verification_level in DECISIVE_PASS_LEVELS
            for item in evidence
        )

    def _has_numeric_pass(self, evidence: List[VerificationEvidence]) -> bool:
        return any(
            item.status == EvidenceStatus.PASS.value
            and item.verification_level in NUMERIC_PASS_LEVELS
            for item in evidence
        )

    def _has_level_pass(self, evidence: List[VerificationEvidence], level: str) -> bool:
        return any(item.status == EvidenceStatus.PASS.value and item.verification_level == level for item in evidence)

    def _failure_kind_from_evidence(self, evidence: VerificationEvidence) -> str:
        if evidence.verifier == "completeness":
            return FailureKind.MISSING_CASE.value
        if evidence.method in {"equation_solution", "symbolic_equivalence", "derivative_check", "integral_check"}:
            return FailureKind.SYMBOLIC_CONTRADICTION.value
        if evidence.method == "numeric_arithmetic":
            return FailureKind.NUMERIC_RESIDUAL.value
        return FailureKind.WRONG_FINAL_ANSWER.value

    def _evidence_summary(self, evidence: VerificationEvidence) -> str:
        residual = f"; residual={evidence.residual}" if evidence.residual is not None else ""
        return f"{evidence.method}: {evidence.details}{residual}"[:500]

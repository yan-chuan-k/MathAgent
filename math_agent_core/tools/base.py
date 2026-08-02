from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from math_agent_core.state import EvidenceStatus, VerificationEvidence, VerificationLevel


class MathTool(ABC):
    name: str = "math_tool"

    @abstractmethod
    def validate_input(self, payload: Dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def run(self, payload: Dict[str, Any], timeout: float = 2.0) -> VerificationEvidence:
        raise NotImplementedError

    def inconclusive(self, claim_id: str, method: str, details: str) -> VerificationEvidence:
        return VerificationEvidence(
            verifier=self.name,
            claim_id=claim_id,
            status=EvidenceStatus.INCONCLUSIVE.value,
            method=method,
            details=details,
            verification_level=VerificationLevel.EXACT_SYMBOLIC.value,
            is_decisive=False,
        )

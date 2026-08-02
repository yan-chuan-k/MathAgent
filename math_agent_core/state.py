from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class OverallStatus(str, Enum):
    SOLVED = "solved"
    PROBABLE = "probable"
    UNCERTAIN = "uncertain"
    INVALID = "invalid"
    ERROR = "error"


class EvidenceStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


class FailureKind(str, Enum):
    JSON_PARSE = "json_parse"
    SCHEMA = "schema"
    WRONG_FINAL_ANSWER = "wrong_final_answer"
    SYMBOLIC_CONTRADICTION = "symbolic_contradiction"
    NUMERIC_RESIDUAL = "numeric_residual"
    MISSING_CASE = "missing_case"
    UNSUPPORTED_LEMMA = "unsupported_lemma"
    CIRCULAR_REASONING = "circular_reasoning"
    LOW_CONSENSUS = "low_consensus"
    INCONCLUSIVE = "inconclusive"


@dataclass
class VerificationEvidence:
    verifier: str
    claim_id: str
    status: str
    method: str
    details: str
    residual: Optional[str] = None
    assumptions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SolveAssessment:
    schema_valid: bool
    content_complete: bool
    answer_verified: bool
    proof_verified: bool
    overall_status: str
    failure_kind: Optional[str] = None
    failure_details: str = ""
    evidence: List[VerificationEvidence] = field(default_factory=list)

    def to_meta_fields(self) -> Dict[str, Any]:
        return {
            "content_complete": bool(self.content_complete),
            "answer_verified": bool(self.answer_verified),
            "proof_verified": bool(self.proof_verified),
            "overall_status": str(self.overall_status),
            "failure_kind": self.failure_kind,
            "failure_details": self.failure_details[:500],
        }

    def evidence_dicts(self) -> List[Dict[str, Any]]:
        return [item.to_dict() for item in self.evidence]

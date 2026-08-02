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


class VerificationLevel(str, Enum):
    FORMAL = "formal"
    EXACT_SYMBOLIC = "exact_symbolic"
    EXACT_ENUMERATION = "exact_enumeration"
    HIGH_PRECISION_NUMERIC = "high_precision_numeric"
    RANDOMIZED_SANITY = "randomized_sanity"
    MODEL_CRITIC = "model_critic"
    COMPLETENESS_ONLY = "completeness_only"


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
    verification_level: str = VerificationLevel.MODEL_CRITIC.value
    is_decisive: bool = False

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


@dataclass
class SolveState:
    problem: str
    route: Dict[str, Any]
    open_goals: List[str] = field(default_factory=list)
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    verified_lemmas: List[str] = field(default_factory=list)
    rejected_attempts: List[Dict[str, Any]] = field(default_factory=list)
    verification_evidence: List[Dict[str, Any]] = field(default_factory=list)
    rejected_strategies: List[str] = field(default_factory=list)
    current_strategy: str = ""
    known_counterexamples: List[str] = field(default_factory=list)
    budget: Dict[str, Any] = field(default_factory=dict)

    def compact(self) -> Dict[str, Any]:
        return {
            "route": self.route,
            "open_goals": self.open_goals[:8],
            "verified_lemmas": self.verified_lemmas[:8],
            "rejected_attempts": self.rejected_attempts[-5:],
            "verification_evidence": self.verification_evidence[-8:],
            "rejected_strategies": self.rejected_strategies[-8:],
            "current_strategy": self.current_strategy,
            "known_counterexamples": self.known_counterexamples[-5:],
            "budget": self.budget,
        }

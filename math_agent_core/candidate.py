from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .state import SolveAssessment, VerificationEvidence


@dataclass
class CandidateSolution:
    candidate_id: str
    strategy: str
    result: Dict[str, Any]
    assessment: SolveAssessment
    evidence: List[VerificationEvidence] = field(default_factory=list)
    critic: Optional[Dict[str, Any]] = None
    normalized_answer: str = ""
    cluster_id: str = ""
    score: float = 0.0
    profile: str = "direct"
    judge_preference_score: float = 0.0

    def to_trace_dict(self) -> Dict[str, Any]:
        meta = self.result.get("_meta", {}) if isinstance(self.result, dict) else {}
        return {
            "candidate_id": self.candidate_id,
            "strategy": self.strategy,
            "profile": self.profile,
            "answer": self.normalized_answer[:300],
            "overall_status": meta.get("overall_status"),
            "failure_kind": meta.get("failure_kind"),
            "score": round(self.score, 4),
            "cluster_id": self.cluster_id,
            "critic_status": (self.critic or {}).get("status"),
            "judge_preference_score": round(self.judge_preference_score, 4),
        }

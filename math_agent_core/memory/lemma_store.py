from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from math_agent_core.state import VerificationEvidence


@dataclass
class Lemma:
    lemma_id: str
    statement: str
    dependencies: List[str] = field(default_factory=list)
    status: str = "open"
    evidence: List[VerificationEvidence] = field(default_factory=list)
    used_by: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["evidence"] = [item.to_dict() for item in self.evidence]
        return data


class LemmaStore:
    def __init__(self) -> None:
        self._lemmas: Dict[str, Lemma] = {}

    def add_or_update(self, lemma: Lemma) -> None:
        self._lemmas[lemma.lemma_id] = lemma

    def add_statement(self, lemma_id: str, statement: str, status: str = "open", confidence: float = 0.0) -> Lemma:
        lemma = Lemma(lemma_id=lemma_id, statement=statement, status=status, confidence=confidence)
        self.add_or_update(lemma)
        return lemma

    def mark_used(self, lemma_id: str, candidate_id: str) -> None:
        lemma = self._lemmas.get(lemma_id)
        if lemma is None:
            return
        if candidate_id not in lemma.used_by:
            lemma.used_by.append(candidate_id)

    def verified(self) -> List[Lemma]:
        return [lemma for lemma in self._lemmas.values() if lemma.status == "verified"]

    def open(self) -> List[Lemma]:
        return [lemma for lemma in self._lemmas.values() if lemma.status not in {"verified", "rejected"}]

    def compact(self) -> Dict[str, Any]:
        return {
            "verified_lemmas": [lemma.to_dict() for lemma in self.verified()[:8]],
            "open_lemmas": [lemma.to_dict() for lemma in self.open()[:8]],
        }

from __future__ import annotations

from typing import Iterable, List

from math_agent_core.candidate import CandidateSolution


def rank_candidates(candidates: Iterable[CandidateSolution]) -> List[CandidateSolution]:
    ranked = list(candidates)
    cluster_sizes = _cluster_sizes(ranked)
    for candidate in ranked:
        candidate.score = _score_candidate(candidate, cluster_sizes.get(candidate.cluster_id, 1))
    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked


def _cluster_sizes(candidates: List[CandidateSolution]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        key = candidate.cluster_id or candidate.normalized_answer
        counts[key] = counts.get(key, 0) + 1
    return counts


def _score_candidate(candidate: CandidateSolution, cluster_size: int) -> float:
    meta = candidate.result.get("_meta", {}) if isinstance(candidate.result, dict) else {}
    status = meta.get("overall_status")
    score = 0.0
    if status == "solved":
        score += 100.0
    elif status == "probable":
        score += 70.0
    elif status == "uncertain":
        score += 30.0
    elif status == "invalid":
        score -= 30.0
    elif status == "error":
        score -= 50.0

    if meta.get("answer_verified"):
        score += 20.0
    if meta.get("proof_verified"):
        score += 15.0
    score += min(cluster_size - 1, 3) * 5.0

    critic = candidate.critic or {}
    if critic.get("status") == "pass":
        score += 8.0
    elif critic.get("status") == "fail":
        score -= 25.0
    elif critic.get("status") == "inconclusive":
        score -= 2.0
    return score

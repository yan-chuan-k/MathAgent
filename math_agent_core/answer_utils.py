from __future__ import annotations

import json
import re
from typing import Any, Optional


DEFAULT_FALLBACK = "无法确定"


def extract_final_answer(text: str, problem: Optional[str] = None) -> str:
    if text is None:
        return DEFAULT_FALLBACK
    value = str(text).strip()
    if not value:
        return DEFAULT_FALLBACK

    value = _strip_markdown_fences(value)

    parsed = _try_parse_json(value)
    if isinstance(parsed, dict):
        for key in ("final_response", "answer", "final_answer", "content", "text"):
            candidate = parsed.get(key)
            if isinstance(candidate, dict):
                candidate = candidate.get("answer")
            if isinstance(candidate, str) and candidate.strip():
                return normalize_final_response(candidate, problem=problem)

    lines = [line.strip() for line in value.splitlines() if line.strip()]
    markers = (
        "final answer",
        "final_response",
        "answer",
        "答案",
        "最终答案",
        "结论",
    )
    for line in reversed(lines):
        lowered = line.lower()
        if any(marker in lowered for marker in markers):
            cleaned = re.sub(
                r"^\s*(final\s*answer|final_response|answer|答案|最终答案|结论)\s*[:：]\s*",
                "",
                line,
                flags=re.IGNORECASE,
            )
            if cleaned.strip():
                return normalize_final_response(cleaned, problem=problem)

    if len(lines) > 1:
        short_tail = lines[-1]
        if 0 < len(short_tail) <= 500:
            return normalize_final_response(short_tail, problem=problem)

    return normalize_final_response(value, problem=problem)


def normalize_final_response(answer: str, problem: Optional[str] = None) -> str:
    if answer is None:
        return DEFAULT_FALLBACK
    normalized = str(answer).strip()
    normalized = _strip_markdown_fences(normalized)
    normalized = re.sub(r"^\s*(final\s*answer|final_response|answer|答案|最终答案|结论)\s*[:：]\s*", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = normalized.strip("`")

    if not normalized:
        return DEFAULT_FALLBACK

    max_length = 3000 if _looks_like_proof(problem or normalized) else 500
    if len(normalized) > max_length:
        normalized = _truncate_answer(normalized, max_length=max_length)
    return normalized or DEFAULT_FALLBACK


def _strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json|text|markdown)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return re.sub(r"^```(?:json|text|markdown)?|```$", "", stripped, flags=re.IGNORECASE).strip()


def _try_parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def _looks_like_proof(text: str) -> bool:
    lowered = text.lower()
    proof_markers = ("prove", "proof", "show that", "证明", "证得", "命题")
    return any(marker in lowered for marker in proof_markers)


def _truncate_answer(text: str, max_length: int) -> str:
    tail_markers = ("therefore", "thus", "hence", "so", "答案", "最终", "结论", "所以", "因此")
    sentences = re.split(r"(?<=[。.!?])\s+", text)
    for sentence in reversed(sentences):
        lowered = sentence.lower()
        if any(marker in lowered for marker in tail_markers) and len(sentence) <= max_length:
            return sentence.strip()
    return text[:max_length].rstrip()

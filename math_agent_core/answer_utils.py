from __future__ import annotations

import json
import re
from typing import Any, Optional


DEFAULT_FALLBACK = "无法确定"

_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"final\s*answer|final_response|answer|result|conclusion|"
    r"最终答案|最后答案|答案|答|结果|结论|所以答案为|因此答案为"
    r")\s*[:：]\s*",
    flags=re.IGNORECASE,
)

_ANSWER_MARKERS = (
    "final answer",
    "final_response",
    "answer",
    "result",
    "conclusion",
    "最终答案",
    "最后答案",
    "答案",
    "结果",
    "结论",
)


def extract_final_answer(text: str, problem: Optional[str] = None) -> str:
    if text is None:
        return DEFAULT_FALLBACK
    value = str(text).strip()
    if not value:
        return DEFAULT_FALLBACK

    value = _strip_markdown_fences(value)

    parsed = _try_parse_json(value)
    extracted = _extract_from_json_value(parsed)
    if extracted:
        return normalize_final_response(extracted, problem=problem)

    lines = [line.strip() for line in value.splitlines() if line.strip()]
    for line in reversed(lines):
        lowered = line.lower()
        if any(marker in lowered for marker in _ANSWER_MARKERS):
            cleaned = _PREFIX_RE.sub("", line).strip()
            if cleaned:
                return normalize_final_response(cleaned, problem=problem)

    latex_boxed = _extract_latex_boxed(value)
    if latex_boxed:
        return normalize_final_response(latex_boxed, problem=problem)

    if len(lines) > 1 and 0 < len(lines[-1]) <= 500:
        return normalize_final_response(lines[-1], problem=problem)

    return normalize_final_response(value, problem=problem)


def normalize_final_response(answer: str, problem: Optional[str] = None) -> str:
    if answer is None:
        return DEFAULT_FALLBACK
    normalized = str(answer).strip()
    normalized = _strip_markdown_fences(normalized)
    normalized = _PREFIX_RE.sub("", normalized).strip()
    normalized = normalized.strip("`")
    normalized = re.sub(r"\s+", " ", normalized).strip()

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


def _extract_from_json_value(value: Any) -> Optional[str]:
    if isinstance(value, dict):
        for key in ("final_response", "answer", "final_answer", "content", "text"):
            candidate = value.get(key)
            if isinstance(candidate, dict):
                nested = _extract_from_json_value(candidate)
                if nested:
                    return nested
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return None


def _extract_latex_boxed(text: str) -> Optional[str]:
    match = re.search(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if match:
        return match.group(1).strip()
    return None


def _looks_like_proof(text: str) -> bool:
    lowered = text.lower()
    proof_markers = ("prove", "proof", "show that", "证明", "证得", "命题", "成立")
    return any(marker in lowered for marker in proof_markers)


def _truncate_answer(text: str, max_length: int) -> str:
    tail_markers = ("therefore", "thus", "hence", "so", "所以", "因此", "故", "结论", "最终答案")
    sentences = re.split(r"(?<=[。.!?；;])\s+", text)
    for sentence in reversed(sentences):
        lowered = sentence.lower()
        if any(marker in lowered for marker in tail_markers) and len(sentence) <= max_length:
            return sentence.strip()
    return text[:max_length].rstrip()

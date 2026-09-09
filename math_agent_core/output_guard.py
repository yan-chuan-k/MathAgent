"""Small, dependency-free guards for incomplete model responses.

The official answer surface is deliberately free-form, so a generic JSON
parser cannot reliably tell a complete proof from a response cut off by a
transport or completion limit.  This module only detects high-confidence
incompleteness.  It never rewrites a mathematical answer and it is not a
correctness grader.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


_LENGTH_FINISH_REASONS = frozenset(
    {
        "length",
        "max_tokens",
        "max_output_tokens",
        "token_limit",
        "truncated",
        "incomplete",
    }
)

_VISIBLE_ANSWER_RE = re.compile(
    r"^\s*(?:final\s+(?:answer|result)|the\s+final\s+(?:answer|result)|"
    r"answer|result|最终答案|最后答案|答案|结果)\s*"
    r"(?::|=|\bis\b|是|为)\s*(?P<payload>\S.*)$",
    flags=re.IGNORECASE | re.DOTALL,
)

_TRAILING_CONNECTOR_RE = re.compile(
    r"(?:,|，|:|：|=|＝|->|→|⇒|\band|\bor|\bbecause|\bsince|\bwhich|\bthat|"
    r"\bwhere|\bsuch\s+as|\btherefore|\bthus|\bhence|因此|所以|故|由于|并且|以及)\s*$",
    flags=re.IGNORECASE,
)


def finish_reason(value: Any) -> str:
    """Extract a provider finish reason from dict- or SDK-style responses."""

    if isinstance(value, Mapping):
        direct = value.get("finish_reason") or value.get("stop_reason")
        if direct:
            return str(direct).strip().lower()
        choices = value.get("choices")
        if isinstance(choices, (list, tuple)) and choices:
            return finish_reason(choices[0])
        for key in ("response", "completion", "result", "output"):
            nested = value.get(key)
            if nested is not None:
                reason = finish_reason(nested)
                if reason:
                    return reason
        return ""

    direct = getattr(value, "finish_reason", None) or getattr(value, "stop_reason", None)
    if direct:
        return str(direct).strip().lower()
    choices = getattr(value, "choices", None)
    if isinstance(choices, (list, tuple)) and choices:
        return finish_reason(choices[0])
    return ""


def has_length_finish_reason(value: Any) -> bool:
    reason = finish_reason(value)
    return reason in _LENGTH_FINISH_REASONS or reason.startswith("length")


def _balanced_delimiters(text: str) -> bool:
    """Check only obvious delimiter damage; strings are not parsed as Python."""

    pairs = (("(", ")"), ("[", "]"), ("{", "}"))
    for opening, closing in pairs:
        depth = 0
        for char in text:
            if char == opening:
                depth += 1
            elif char == closing:
                depth -= 1
                if depth < 0:
                    return False
        if depth:
            return False
    if text.count("`") % 2:
        return False
    if text.count(r"\left") != text.count(r"\right"):
        return False
    if text.count(r"\begin{") != text.count(r"\end{"):
        return False
    return True


def looks_truncated(
    text: Any,
    *,
    response_mode: str = "",
    raw_response: Any = None,
) -> bool:
    """Return True only for a strong, actionable incompleteness signal.

    ANSWER_VALUE responses get special treatment: a complete first visible line
    is sufficient even when an optional explanation after it was cut off.  For
    proof/derivation/construction responses, a provider length reason or an
    unfinished delimiter/connector is enough to request one recovery call.
    """

    value = str(text or "").strip()
    if not value:
        return True

    mode = str(response_mode or "").strip().upper()
    first_line = next((line.strip() for line in value.splitlines() if line.strip()), "")
    answer_match = _VISIBLE_ANSWER_RE.fullmatch(first_line)
    if mode == "ANSWER_VALUE" and answer_match:
        payload = str(answer_match.group("payload") or "").strip()
        if payload and _balanced_delimiters(payload) and not _TRAILING_CONNECTOR_RE.search(payload):
            return False

    if has_length_finish_reason(raw_response):
        return True

    if not _balanced_delimiters(value):
        return True
    if value.endswith(("...", "…", "\\")):
        return True
    if _TRAILING_CONNECTOR_RE.search(value):
        return True
    # A long proof/derivation that ends in a bare word or half-written clause
    # is a common symptom when a provider omits finish metadata.  Keep this
    # deliberately narrow so short answers such as ``Proof: direct`` are not
    # turned into needless second calls.
    if mode in {"PROOF", "PROOF_OR_DISPROOF", "DERIVATION", "CONSTRUCTION_COUNTEREXAMPLE"}:
        last_line = value.splitlines()[-1].strip() if value.splitlines() else value
        has_explicit_completion = bool(
            re.search(r"(?:\bQED\b|\bCQFD\b|\\square|□|证毕|证明完毕|得证)\s*$", value, flags=re.IGNORECASE)
        )
        if (
            len(value) >= 240
            and last_line
            and not has_explicit_completion
            and not re.search(r"[.!?。！？;；:)\]}$]$", value)
        ):
            return True
    return False


def response_content(value: Any) -> str:
    """Extract text from common SDK/dict wrappers without importing an SDK."""

    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        for key in ("final_response", "content", "text", "answer"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        choices = value.get("choices")
        if isinstance(choices, (list, tuple)) and choices:
            return response_content(choices[0])
        message = value.get("message")
        if message is not None:
            return response_content(message)
        return ""
    if isinstance(value, (list, tuple)):
        parts = []
        for item in value:
            part = response_content(item)
            if part:
                parts.append(part)
            elif isinstance(item, str) and item.strip():
                parts.append(item.strip())
        return "".join(parts).strip()
    choices = getattr(value, "choices", None)
    if isinstance(choices, (list, tuple)) and choices:
        return response_content(choices[0])
    content = getattr(value, "content", None)
    if isinstance(content, (list, tuple)):
        return response_content(content)
    if isinstance(content, str) and content.strip():
        return content.strip()
    message = getattr(value, "message", None)
    if message is not None:
        return response_content(message)
    return ""

from __future__ import annotations

import json
import sys
import time
from typing import Any, Dict, List

from .augmented_tool import (
    MAX_TOOL_REQUESTS,
    ToolRequest,
    _InlineToolExecutor,
    _fact_statement_is_complete,
    _normalize_verification_requests,
)


_MAX_STDIN_CHARS = 262_144
_MAX_TEXT_FIELD_CHARS = 500
_ANSWER_VALUE_MODE = "ANSWER_VALUE"


def _compact_text(value: Any, limit: int = _MAX_TEXT_FIELD_CHARS) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit]


def _apply_test_delay(payload: Dict[str, Any]) -> None:
    """Apply a bounded delay supplied only by repository timeout tests."""

    if "_test_sleep_seconds" not in payload:
        return
    seconds = float(payload["_test_sleep_seconds"])
    if seconds < 0 or seconds > 60:
        raise ValueError("test delay outside bound")
    time.sleep(seconds)


def _tool_facts(payload: Dict[str, Any]) -> Dict[str, Any]:
    _apply_test_delay(payload)
    serialized = payload.get("requests")
    if not isinstance(serialized, list):
        return {"status": "error"}

    executor = _InlineToolExecutor(extended_tools=bool(payload.get("extended_tools", False)))
    facts: List[Dict[str, str]] = []
    for item in serialized[:MAX_TOOL_REQUESTS]:
        if not isinstance(item, dict):
            continue
        arguments = item.get("arguments")
        request = ToolRequest(
            family=str(item.get("family") or ""),
            operation=str(item.get("operation") or ""),
            arguments=arguments if isinstance(arguments, dict) else {},
            source=str(item.get("source") or ""),
        )
        fact = executor.execute_one(request)
        if fact is not None and _fact_statement_is_complete(fact.statement):
            facts.append(
                {
                    "family": _compact_text(fact.family, 40),
                    "operation": _compact_text(fact.operation, 40),
                    "source": _compact_text(item.get("source"), 40),
                    # Exact facts are never shortened in transit.  The parent
                    # independently rejects any oversized or ellipsis-ended
                    # statement before prompt injection.
                    "statement": fact.statement,
                }
            )
    return {"status": "ok", "facts": facts}


def _compact_verification_evidence(item: Any) -> Dict[str, Any]:
    status = str(getattr(item, "status", "inconclusive") or "inconclusive")
    if status not in {"pass", "fail", "inconclusive"}:
        status = "inconclusive"
    residual = getattr(item, "residual", None)
    claim_scope = str(getattr(item, "claim_scope", "subclaim") or "subclaim")
    if claim_scope not in {"full_answer", "subclaim"}:
        claim_scope = "subclaim"
    return {
        "status": status,
        "method": _compact_text(getattr(item, "method", "deterministic_check"), 120),
        "details": _compact_text(getattr(item, "details", ""), 500),
        "residual": None if residual in {None, ""} else _compact_text(residual, 240),
        "is_decisive": bool(getattr(item, "is_decisive", False)),
        "claim_scope": claim_scope,
    }


def _answer_verification(payload: Dict[str, Any]) -> Dict[str, Any]:
    if str(payload.get("response_mode") or "") != _ANSWER_VALUE_MODE:
        return {"status": "ok", "evidence": []}

    # The production parent never derives the test delay from problem/model data.
    _apply_test_delay(payload)

    final_response = payload.get("final_response")
    if not isinstance(final_response, str):
        return {"status": "error"}

    requests = _normalize_verification_requests(payload.get("requests"))
    if not requests:
        return {"status": "ok", "evidence": []}

    evidence: List[Any] = []
    try:
        from .sympy_tool import run_grounded_sympy_verification

        evidence.extend(
            run_grounded_sympy_verification(
                final_response,
                requests=requests,
            )
        )
    except Exception:
        pass

    try:
        from math_agent_core.verifiers.linear_algebra import run_grounded_matrix_verification

        matrix_requests = [
            request
            for request in requests
            if request.get("kind") in {"matrix_determinant", "matrix_rank"}
        ]
        if matrix_requests:
            evidence.extend(
                run_grounded_matrix_verification(
                    final_response,
                    requests=matrix_requests,
                )
            )
    except Exception:
        pass

    return {
        "status": "ok",
        "evidence": [_compact_verification_evidence(item) for item in evidence[:8]],
    }


def _dispatch(payload: Dict[str, Any]) -> Dict[str, Any]:
    mode = str(payload.get("mode") or "")
    if mode == "execute_tools":
        return _tool_facts(payload)
    if mode == "verify_answer":
        return _answer_verification(payload)
    return {"status": "error"}


def main() -> int:
    try:
        raw = sys.stdin.read(_MAX_STDIN_CHARS + 1)
        if len(raw) > _MAX_STDIN_CHARS:
            response = {"status": "error"}
        else:
            payload = json.loads(raw)
            response = _dispatch(payload) if isinstance(payload, dict) else {"status": "error"}
    except Exception:
        response = {"status": "error"}

    sys.stdout.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from typing import Any, Dict, Iterable, List


MAX_TRACE_CONTENT = 1000


def make_trace_step(step: str, content: Any) -> Dict[str, str]:
    return {
        "step": str(step or "info")[:80],
        "content": sanitize_trace_content(content),
    }


def sanitize_trace(trace: Any) -> List[Dict[str, str]]:
    if not isinstance(trace, list):
        return []
    cleaned = []
    for item in trace[:20]:
        if isinstance(item, dict):
            cleaned.append(make_trace_step(item.get("step", "info"), item.get("content", "")))
        else:
            cleaned.append(make_trace_step("info", item))
    return cleaned


def trace_from_orchestrator_result(result: Dict[str, Any], log: Dict[str, Any] | None = None) -> List[Dict[str, str]]:
    trace = []
    if isinstance(result.get("trace"), list):
        trace.extend(sanitize_trace(result["trace"]))
    if result.get("problem_type"):
        trace.append(make_trace_step("route", result.get("problem_type")))
    if log and log.get("settings"):
        trace.append(make_trace_step("settings", log.get("settings")))
    if log and log.get("state"):
        trace.append(make_trace_step("state", log.get("state")))
    if log and log.get("candidates"):
        trace.append(make_trace_step("candidates", log.get("candidates")))
    if result.get("reasoning_plan"):
        trace.append(make_trace_step("plan", result.get("reasoning_plan")))
    if result.get("verification"):
        trace.append(make_trace_step("verify", result.get("verification")))
    if result.get("_meta"):
        meta = result["_meta"]
        trace.append(
            make_trace_step(
                "finalize",
                {
                    "attempts": meta.get("attempts"),
                    "schema_valid": meta.get("schema_valid"),
                    "overall_status": meta.get("overall_status"),
                    "content_complete": meta.get("content_complete"),
                    "answer_verified": meta.get("answer_verified"),
                    "proof_verified": meta.get("proof_verified"),
                    "failure_kind": meta.get("failure_kind"),
                },
            )
        )
    if log and log.get("errors"):
        trace.append(make_trace_step("error", log.get("errors")))
    return trace[:20]


def sanitize_trace_content(content: Any) -> str:
    if isinstance(content, str):
        text = content
    else:
        text = repr(_json_safe(content))
    text = text.replace("\r", " ").replace("\n", " ")
    for marker in ("sk-", "INTERN_API_KEY", "api_key", "authorization", "Authorization"):
        text = text.replace(marker, "[redacted]")
    if len(text) > MAX_TRACE_CONTENT:
        text = text[:MAX_TRACE_CONTENT].rstrip()
    return text


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)

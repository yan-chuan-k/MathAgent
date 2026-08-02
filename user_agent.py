from __future__ import annotations

import re
from typing import Any, Dict, List

from math_agent_core.answer_utils import DEFAULT_FALLBACK, extract_final_answer, normalize_final_response
from math_agent_core.trace_utils import make_trace_step, sanitize_trace, trace_from_orchestrator_result


class ReasoningAgent:
    def __init__(self, client, *args, **kwargs):
        self.client = client
        self.max_retries = int(kwargs.get("max_retries", 1))
        self.temperature = float(kwargs.get("temperature", 0.2))
        self.max_tokens = int(kwargs.get("max_tokens", 4096))
        self.thinking_mode = bool(kwargs.get("thinking_mode", True))
        self.orchestrator = None

        try:
            from math_agent_core.orchestrator import MathAgentOrchestrator

            self.orchestrator = MathAgentOrchestrator(
                client=self.client,
                max_retries=self.max_retries,
                enable_repair=True,
                enable_tool_verify=True,
                backend="simple",
                thinking_mode=self.thinking_mode,
            )
        except Exception:
            self.orchestrator = None

    def solve(self, problem: str, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
        try:
            if not isinstance(problem, str) or not problem.strip():
                return self._fallback_result("problem is empty or not a string")

            safe_metadata = metadata if isinstance(metadata, dict) else {}

            if self.orchestrator is not None:
                result = self.orchestrator.solve(problem=problem, metadata=safe_metadata)
                final_response = self._extract_final_response(result, problem)
                trace = trace_from_orchestrator_result(result, getattr(self.orchestrator, "last_log", None))
                return self._json_safe_result(final_response, trace)

            response = self._direct_model_call(problem, safe_metadata)
            final_response = extract_final_answer(self._normalize_model_response(response), problem=problem)
            trace = [
                make_trace_step(
                    "fallback",
                    {"mode": "direct client.chat call", "thinking_mode": self.thinking_mode},
                )
            ]
            return self._json_safe_result(final_response, trace)
        except Exception as exc:
            return self._fallback_result(f"{type(exc).__name__}: {str(exc)[:300]}")

    def _direct_model_call(self, problem: str, metadata: Dict[str, Any]) -> Any:
        messages = self._build_direct_prompt(problem, metadata)
        try:
            return self.client.chat(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                thinking_mode=self.thinking_mode,
            )
        except TypeError:
            return self.client.chat(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

    def _build_direct_prompt(self, problem: str, metadata: Dict[str, Any]) -> List[Dict[str, str]]:
        subject = metadata.get("subject") or metadata.get("type") or metadata.get("category") or ""
        return [
            {
                "role": "system",
                "content": (
                    "You are a rigorous math problem solver. Solve the problem and return a concise, "
                    "judgeable final answer. For calculation, output only the final value or expression. "
                    "For proof, output a concise complete proof. Do not use any provided reference answer."
                ),
            },
            {
                "role": "user",
                "content": f"Subject hint: {subject}\nProblem:\n{problem}\n\nGive the final answer.",
            },
        ]

    def _normalize_model_response(self, response: Any) -> str:
        if response is None:
            return ""
        if isinstance(response, str):
            return response.strip()
        if isinstance(response, dict):
            for key in ("final_response", "content", "text", "answer"):
                value = response.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return str(response).strip()

    def _extract_final_response(self, result: Any, problem: str) -> str:
        if isinstance(result, dict):
            if not self._is_acceptable_orchestrator_result(result):
                return DEFAULT_FALLBACK
            value = result.get("final_response")
            if isinstance(value, str) and value.strip():
                normalized = normalize_final_response(value, problem=problem)
                return self._repair_missing_requested_value(normalized, result, problem)
            final_answer = result.get("final_answer")
            if isinstance(final_answer, dict):
                answer = final_answer.get("answer")
                if isinstance(answer, str) and answer.strip():
                    normalized = normalize_final_response(answer, problem=problem)
                    return self._repair_missing_requested_value(normalized, result, problem)
            if isinstance(final_answer, str) and final_answer.strip():
                normalized = normalize_final_response(final_answer, problem=problem)
                return self._repair_missing_requested_value(normalized, result, problem)
        return DEFAULT_FALLBACK

    def _is_acceptable_orchestrator_result(self, result: Dict[str, Any]) -> bool:
        meta = result.get("_meta") if isinstance(result.get("_meta"), dict) else {}
        status = meta.get("overall_status")
        if status != "solved":
            return False
        if not bool(meta.get("content_complete")):
            return False
        if status == "solved" and not (meta.get("answer_verified") or meta.get("proof_verified")):
            return False
        return True

    def _repair_missing_requested_value(self, final_response: str, result: Any, problem: str) -> str:
        problem_text = str(problem or "").lower()
        final_text = str(final_response or "").strip()
        if not final_text or not isinstance(result, dict):
            return final_response

        asks_gaussian_curvature = any(marker in problem_text for marker in ("高斯曲率", "gaussian curvature"))
        final_has_curvature_value = bool(
            re.search(r"\bK\s*=", final_text)
            or re.search(r"(?:curvature|曲率)[^0-9+\-]*[+\-]?\d+(?:\.\d+)?", final_text, flags=re.IGNORECASE)
        )
        if asks_gaussian_curvature and not final_has_curvature_value:
            evidence = self._collect_result_text(result)
            match = re.search(r"\bK\s*=[^.;。；]*?=\s*([+-]?\d+(?:\.\d+)?)", evidence)
            if match is None:
                match = re.search(r"\bK\s*=\s*([+-]?\d+(?:\.\d+)?)", evidence)
            if match:
                value = match.group(1).rstrip(".;,，。")
                return normalize_final_response(f"K = {value}. {final_text}", problem=problem)
        return final_response

    def _collect_result_text(self, value: Any) -> str:
        if isinstance(value, dict):
            return " ".join(self._collect_result_text(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return " ".join(self._collect_result_text(item) for item in value)
        if isinstance(value, str):
            return value
        return ""

    def _fallback_result(self, reason: str) -> Dict[str, Any]:
        return self._json_safe_result(DEFAULT_FALLBACK, [make_trace_step("error", reason)])

    def _json_safe_result(self, final_response: str, trace: Any) -> Dict[str, Any]:
        final_text = normalize_final_response(final_response)
        return {
            "final_response": final_text or DEFAULT_FALLBACK,
            "trace": sanitize_trace(trace),
        }

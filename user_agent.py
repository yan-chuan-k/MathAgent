from __future__ import annotations

from typing import Any, Dict, List

from math_agent_core.answer_utils import DEFAULT_FALLBACK, extract_final_answer, normalize_final_response
from math_agent_core.trace_utils import make_trace_step, sanitize_trace, trace_from_orchestrator_result


class ReasoningAgent:
    def __init__(self, client, *args, **kwargs):
        self.client = client
        self.max_retries = int(kwargs.get("max_retries", 1))
        self.temperature = float(kwargs.get("temperature", 0.2))
        self.max_tokens = int(kwargs.get("max_tokens", 4096))
        self.orchestrator = None

        try:
            from math_agent_core.orchestrator import MathAgentOrchestrator

            self.orchestrator = MathAgentOrchestrator(
                client=self.client,
                max_retries=self.max_retries,
                enable_repair=True,
                enable_tool_verify=False,
                backend="simple",
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
                if final_response == DEFAULT_FALLBACK:
                    raw_output = getattr(self.orchestrator, "last_log", {}).get("solver_raw_output", "")
                    final_response = extract_final_answer(raw_output, problem=problem)
                trace = trace_from_orchestrator_result(result, getattr(self.orchestrator, "last_log", None))
                return self._json_safe_result(final_response, trace)

            response = self._direct_model_call(problem, safe_metadata)
            final_response = extract_final_answer(self._normalize_model_response(response), problem=problem)
            trace = [make_trace_step("fallback", "direct client.chat call")]
            return self._json_safe_result(final_response, trace)
        except Exception as exc:
            return self._fallback_result(f"{type(exc).__name__}: {str(exc)[:300]}")

    def _direct_model_call(self, problem: str, metadata: Dict[str, Any]) -> Any:
        messages = self._build_direct_prompt(problem, metadata)
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
            value = result.get("final_response")
            if isinstance(value, str) and value.strip():
                return normalize_final_response(value, problem=problem)
            final_answer = result.get("final_answer")
            if isinstance(final_answer, dict):
                answer = final_answer.get("answer")
                if isinstance(answer, str) and answer.strip():
                    return normalize_final_response(answer, problem=problem)
            if isinstance(final_answer, str) and final_answer.strip():
                return normalize_final_response(final_answer, problem=problem)
        return DEFAULT_FALLBACK

    def _fallback_result(self, reason: str) -> Dict[str, Any]:
        return self._json_safe_result(DEFAULT_FALLBACK, [make_trace_step("error", reason)])

    def _json_safe_result(self, final_response: str, trace: Any) -> Dict[str, Any]:
        final_text = normalize_final_response(final_response)
        return {
            "final_response": final_text or DEFAULT_FALLBACK,
            "trace": sanitize_trace(trace),
        }

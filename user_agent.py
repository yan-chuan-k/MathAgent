from __future__ import annotations

import inspect
import re
from typing import Any, Dict, List

from math_agent_core.answer_utils import DEFAULT_FALLBACK, extract_final_answer, normalize_final_response
from math_agent_core.trace_utils import make_trace_step, sanitize_trace, trace_from_orchestrator_result


_SCORE_FIRST_ANSWER_SYSTEM_PROMPT = """You are a high-accuracy mathematics competition solver.

Solve the problem carefully using internal reasoning.

Output exactly ONE visible line:

Final answer: <complete requested answer>

Then stop.

Do not output JSON.
Do not repeat the problem.
Do not provide visible explanation or derivation.

Preserve all requested roots, conditions, intervals,
moduli, matrices, vectors, sets, and multiple answer parts.
""".strip()

_SCORE_FIRST_PROOF_SYSTEM_PROMPT = """You are a high-accuracy mathematics competition solver.

State the conclusion first, then give a concise but complete proof.

Do not output JSON.
Do not repeat the problem.

Do not omit necessary logical steps merely to shorten the response.
""".strip()

_SCORE_FIRST_WRAPPER_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE | re.DOTALL)
    for pattern in (
        r"^\s*(?:final\s+answer|the\s+final\s+answer)\s*(?::|=|\bis\b)\s*(?P<payload>.+?)\s*$",
        r"^\s*(?:final\s+result|the\s+final\s+result)\s*(?::|=|\bis\b)\s*(?P<payload>.+?)\s*$",
        r"^\s*(?:answer|the\s+answer)\s*(?::|=|\bis\b)\s*(?P<payload>.+?)\s*$",
        r"^\s*(?:result|the\s+result)\s*(?::|=|\bis\b)\s*(?P<payload>.+?)\s*$",
        r"^\s*(?:therefore|thus|hence)\s*,?\s*(?:the\s+)?answer\s*(?::|=|\bis\b)\s*(?P<payload>.+?)\s*$",
        r"^\s*最终答案\s*(?:[:：=]|是)\s*(?P<payload>.+?)\s*$",
        r"^\s*最后答案\s*(?:[:：=]|是)\s*(?P<payload>.+?)\s*$",
        r"^\s*答案\s*(?:[:：=]|是|为)\s*(?P<payload>.+?)\s*$",
        r"^\s*结果\s*(?:[:：=]|是|为)\s*(?P<payload>.+?)\s*$",
        r"^\s*(?:因此|所以|故)\s*答案\s*(?:[:：=]|是|为)\s*(?P<payload>.+?)\s*$",
    )
)

_PROOF_TASK_MARKERS = (
    "prove",
    "proof",
    "show that",
    "demonstrate that",
    "证明",
    "证实",
    "说明.*成立",
)


class ReasoningAgent:
    def __init__(self, client, *args, **kwargs):
        self.client = client
        self.production_mode = str(kwargs.get("production_mode", "score_first") or "score_first").strip().lower()
        if self.production_mode not in {"score_first", "orchestrated"}:
            raise ValueError("production_mode must be 'score_first' or 'orchestrated'")

        self.max_retries = int(kwargs.get("max_retries", 1))
        default_temperature = 0.1 if self.production_mode == "score_first" else 0.2
        default_max_tokens = 8192 if self.production_mode == "score_first" else 4096
        self.temperature = float(kwargs.get("temperature", default_temperature))
        self.max_tokens = int(kwargs.get("max_tokens", default_max_tokens))
        self.thinking_mode = bool(kwargs.get("thinking_mode", True))
        self.max_candidates = int(kwargs.get("max_candidates", 2))
        self.orchestrator = None

        # The full orchestrator remains available, but only as an explicit opt-in.
        # ScoreFirst deliberately does not construct or invoke the orchestration path.
        if self.production_mode == "orchestrated":
            try:
                from math_agent_core.orchestrator import MathAgentOrchestrator

                self.orchestrator = MathAgentOrchestrator(
                    client=self.client,
                    max_retries=self.max_retries,
                    enable_repair=True,
                    enable_tool_verify=True,
                    backend="simple",
                    thinking_mode=self.thinking_mode,
                    max_candidates=self.max_candidates,
                    solver_max_tokens=self.max_tokens,
                    solver_temperature=self.temperature,
                    enable_critic=bool(kwargs.get("enable_critic", True)),
                    enable_finalizer=bool(kwargs.get("enable_finalizer", False)),
                )
            except Exception:
                self.orchestrator = None

    def solve(self, problem: str, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
        try:
            if not isinstance(problem, str) or not problem.strip():
                return self._fallback_result("problem is empty or not a string")

            safe_metadata = metadata if isinstance(metadata, dict) else {}

            if self.production_mode == "score_first":
                return self._solve_score_first(problem, safe_metadata)

            if self.orchestrator is not None:
                result = self.orchestrator.solve(problem=problem, metadata=safe_metadata)
                final_response = self._extract_final_response(result, problem)
                trace = trace_from_orchestrator_result(result, getattr(self.orchestrator, "last_log", None))
                if final_response == DEFAULT_FALLBACK:
                    raw_output = self._extract_last_raw_output()
                    final_response = extract_final_answer(raw_output, problem=problem)
                if final_response == DEFAULT_FALLBACK:
                    response = self._direct_model_call(problem, safe_metadata)
                    final_response = extract_final_answer(self._normalize_model_response(response), problem=problem)
                    trace.append(make_trace_step("fallback", "orchestrator did not produce a usable answer; used direct client.chat"))
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

    def _solve_score_first(self, problem: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        response = self._score_first_model_call(problem, metadata)
        raw_output = self._normalize_model_response(response)
        final_response = self._extract_score_first_response(raw_output, problem)
        subject_hint = self._subject_hint(metadata)
        trace = [
            make_trace_step("mode", "score_first"),
            make_trace_step("model_call", "primary client.chat call: 1"),
            make_trace_step("answer", "answer extracted" if final_response != DEFAULT_FALLBACK else "empty/unusable answer"),
        ]
        if subject_hint:
            trace.insert(1, make_trace_step("subject_hint", subject_hint))
        return self._score_first_json_result(final_response, trace)

    def _score_first_model_call(self, problem: str, metadata: Dict[str, Any]) -> Any:
        messages = self._build_score_first_prompt(problem, metadata)
        return self._chat_once(messages)

    def _direct_model_call(self, problem: str, metadata: Dict[str, Any]) -> Any:
        messages = self._build_direct_prompt(problem, metadata)
        return self._chat_once(messages)

    def _chat_once(self, messages: List[Dict[str, str]]) -> Any:
        kwargs = {
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self._client_supports_thinking_mode():
            kwargs["thinking_mode"] = self.thinking_mode
        return self.client.chat(**kwargs)

    def _client_supports_thinking_mode(self) -> bool:
        try:
            signature = inspect.signature(self.client.chat)
        except (TypeError, ValueError):
            return True
        if "thinking_mode" in signature.parameters:
            return True
        return any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())

    def _build_score_first_prompt(self, problem: str, metadata: Dict[str, Any]) -> List[Dict[str, str]]:
        subject = self._subject_hint(metadata)
        subject_line = f"Subject hint: {subject}\n\n" if subject else ""
        is_proof = self._is_proof_task(problem)
        system_prompt = _SCORE_FIRST_PROOF_SYSTEM_PROMPT if is_proof else _SCORE_FIRST_ANSWER_SYSTEM_PROMPT
        user_instruction = (
            "Give the requested proof."
            if is_proof
            else "Give the requested mathematical answer."
        )
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"{subject_line}Problem:\n{problem}\n\n{user_instruction}",
            },
        ]

    def _build_direct_prompt(self, problem: str, metadata: Dict[str, Any]) -> List[Dict[str, str]]:
        subject = self._subject_hint(metadata)
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

    def _subject_hint(self, metadata: Dict[str, Any]) -> str:
        value = metadata.get("subject") or metadata.get("type") or metadata.get("category") or ""
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        return text[:120]

    def _extract_score_first_response(self, raw_output: str, problem: str) -> str:
        text = str(raw_output or "").strip()
        if not text:
            return DEFAULT_FALLBACK

        # Proofs are intentionally preserved in full. The legacy answer normalizer
        # has answer-value truncation/tail-sentence heuristics that are destructive
        # for proofs and therefore must not participate in the ScoreFirst proof path.
        if self._is_proof_task(problem):
            return text

        lines = [line.strip() for line in text.splitlines() if line.strip()]

        # The answer-first contract makes this branch robust even if the transport
        # truncates a later explanation: the complete first-line answer has already
        # arrived and does not depend on the remainder of the response.
        if lines:
            first_answer = self._parse_score_first_answer_wrapper(lines[0])
            if first_answer is not None:
                return self._normalize_score_first_answer_payload(first_answer)

        # Closed-world fallback only: accept a complete supported wrapper on the
        # whole response or on an individual line. This fixes English/Chinese
        # copula forms without arbitrary substring searches.
        whole_answer = self._parse_score_first_answer_wrapper(text)
        if whole_answer is not None:
            return self._normalize_score_first_answer_payload(whole_answer)

        for line in reversed(lines):
            wrapped = self._parse_score_first_answer_wrapper(line)
            if wrapped is not None:
                return self._normalize_score_first_answer_payload(wrapped)

        # If the model emits a bare one-line answer, preserve it without routing
        # through the legacy 500-character cap. For an unwrapped multiline
        # answer-value response, retain the final nonempty line as a conservative
        # compatibility fallback; proof tasks never enter this branch.
        if len(lines) == 1:
            return self._normalize_score_first_answer_payload(lines[0])
        if lines:
            return self._normalize_score_first_answer_payload(lines[-1])
        return DEFAULT_FALLBACK

    def _parse_score_first_answer_wrapper(self, text: str) -> str | None:
        value = str(text or "").strip()
        if not value:
            return None
        for pattern in _SCORE_FIRST_WRAPPER_PATTERNS:
            match = pattern.fullmatch(value)
            if match:
                payload = str(match.group("payload") or "").strip()
                return payload or None
        return None

    def _normalize_score_first_answer_payload(self, payload: str) -> str:
        value = str(payload or "").strip()
        if not value:
            return DEFAULT_FALLBACK

        # Only remove complete, known presentation wrappers. Mathematical grouping
        # such as {1,2,3}, [1,2,3], tuples, intervals, and LaTeX braces is untouched.
        # Strip one clearly external terminal punctuation mark first so surfaces like
        # "$42$." and "\\boxed{42}。" can expose their actual outer wrapper.
        value = self._strip_score_first_terminal_punctuation(value)
        changed = True
        while changed and value:
            changed = False
            stripped = self._strip_score_first_presentation_wrapper(value)
            if stripped != value:
                value = stripped.strip()
                value = self._strip_score_first_terminal_punctuation(value)
                changed = True

        return value.strip() or DEFAULT_FALLBACK

    def _strip_score_first_presentation_wrapper(self, text: str) -> str:
        value = str(text or "").strip()

        if len(value) >= 4 and value.startswith("**") and value.endswith("**"):
            return value[2:-2].strip()
        if len(value) >= 4 and value.startswith("__") and value.endswith("__"):
            return value[2:-2].strip()
        if len(value) >= 2 and value.startswith("$") and value.endswith("$"):
            return value[1:-1].strip()
        if len(value) >= 4 and value.startswith(r"\(") and value.endswith(r"\)"):
            return value[2:-2].strip()
        if len(value) >= 4 and value.startswith(r"\[") and value.endswith(r"\]"):
            return value[2:-2].strip()

        boxed_prefix = r"\boxed{"
        if value.startswith(boxed_prefix):
            inner = self._extract_complete_braced_wrapper(value, boxed_prefix)
            if inner is not None:
                return inner
        return value

    def _extract_complete_braced_wrapper(self, text: str, prefix: str) -> str | None:
        if not text.startswith(prefix):
            return None
        depth = 1
        chars: List[str] = []
        index = len(prefix)
        while index < len(text):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    if text[index + 1 :].strip():
                        return None
                    return "".join(chars).strip()
            chars.append(char)
            index += 1
        return None

    def _strip_score_first_terminal_punctuation(self, text: str) -> str:
        value = str(text or "").rstrip()
        if not value:
            return value

        # One terminal presentation punctuation mark is safe to remove. A decimal
        # point inside a number is unaffected because it is not terminal.
        if value.endswith(("。", ";", "；")):
            return value[:-1].rstrip()
        if value.endswith(".") and not value.endswith("..."):
            return value[:-1].rstrip()
        return value

    def _is_proof_task(self, problem: str) -> bool:
        lowered = str(problem or "").lower()
        if any(marker in lowered for marker in _PROOF_TASK_MARKERS[:-1]):
            return True
        return bool(re.search(_PROOF_TASK_MARKERS[-1], str(problem or "")))

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
            solution = result.get("solution")
            if isinstance(solution, list) and solution:
                for item in reversed(solution):
                    content = item.get("content") if isinstance(item, dict) else item
                    if isinstance(content, str) and content.strip():
                        return extract_final_answer(content, problem=problem)
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

    def _extract_last_raw_output(self) -> str:
        log = getattr(self.orchestrator, "last_log", None)
        if isinstance(log, dict):
            raw = log.get("solver_raw_output")
            if isinstance(raw, str) and raw.strip():
                return raw
        return ""

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

    def _score_first_json_result(self, final_response: str, trace: Any) -> Dict[str, Any]:
        # ScoreFirst extraction has already normalized answer-value surfaces, while
        # proof responses must remain complete. Do not reapply the legacy
        # normalize_final_response() 500/3000-character caps here.
        final_text = str(final_response or "").strip()
        return {
            "final_response": final_text or DEFAULT_FALLBACK,
            "trace": sanitize_trace(trace),
        }

    def _json_safe_result(self, final_response: str, trace: Any) -> Dict[str, Any]:
        final_text = normalize_final_response(final_response)
        return {
            "final_response": final_text or DEFAULT_FALLBACK,
            "trace": sanitize_trace(trace),
        }

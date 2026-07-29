from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .json_utils import extract_json_from_text, repair_json_locally, validate_result
from .prompts import build_solver_messages
from .schema import empty_result


class MathAgentOrchestrator:
    def __init__(
        self,
        client: Any,
        max_retries: int = 2,
        enable_repair: bool = True,
        enable_tool_verify: bool = True,
        backend: str = "simple",
        schema_path: Optional[Path] = None,
    ):
        self.client = client
        self.max_retries = max_retries
        self.enable_repair = enable_repair
        self.enable_tool_verify = enable_tool_verify
        self.backend = self._resolve_backend(backend)
        self.model = getattr(client, "model", "intern-s1")
        self.schema = self._load_schema(schema_path)
        self.last_log: Dict[str, Any] = {}

    def solve(self, problem: Any, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        started = time.time()
        problem = self._normalize_problem_input(problem, metadata)
        problem_id = str(problem.get("problem_id") or "UNKNOWN")
        problem_text = self._get_problem_text(problem)
        log = {
            "problem_id": problem_id,
            "input": problem,
            "parsed": {"problem_text": problem_text},
            "route": {},
            "plan": [],
            "solver_raw_output": "",
            "solver_result": {},
            "verification": {},
            "repair_history": [],
            "final_result": {},
            "timing": {"start_time": started, "end_time": None, "elapsed_seconds": 0.0},
            "errors": [],
        }

        result: Dict[str, Any] = empty_result(problem_id, model=self.model, backend=self.backend)
        for attempt in range(1, self.max_retries + 2):
            try:
                raw_output = self._call_solver(problem, problem_text)
                log["solver_raw_output"] = raw_output
                parsed = extract_json_from_text(raw_output)
                result = repair_json_locally(
                    parsed,
                    problem_id=problem_id,
                    model=self.model,
                    backend=self.backend,
                    attempts=attempt,
                    elapsed_seconds=time.time() - started,
                )
                validation = validate_result(result, self.schema)
                result["_meta"]["schema_valid"] = validation.valid
                result["_meta"]["schema_error"] = validation.error
                log["solver_result"] = result
                log["route"] = {
                    "primary_domain": result.get("problem_type", "unknown"),
                    "domain_candidates": result.get("domain_candidates", ["unknown"]),
                    "task_type": result.get("task_type", "unknown"),
                    "needs_tool_verification": self.enable_tool_verify,
                }
                log["plan"] = result.get("reasoning_plan", [])
                log["verification"] = result.get("verification", {})

                if self._needs_repair(result, validation.valid) and attempt <= self.max_retries:
                    log["repair_history"].append(
                        {
                            "attempt": attempt + 1,
                            "previous_error": validation.error or self._verification_error(result),
                            "repair_strategy": "retry solver with same structured JSON requirements",
                        }
                    )
                    continue
                break
            except Exception as exc:
                log["errors"].append({"attempt": attempt, "error": str(exc)})
                result = empty_result(problem_id, model=self.model, backend=self.backend)
                result["_meta"]["attempts"] = attempt
                result["_meta"]["schema_error"] = str(exc)
                if attempt > self.max_retries:
                    break

        elapsed = time.time() - started
        result["_meta"]["elapsed_seconds"] = elapsed
        log["final_result"] = result
        log["timing"]["end_time"] = time.time()
        log["timing"]["elapsed_seconds"] = elapsed
        self.last_log = log
        return result

    def _call_solver(self, problem: Dict[str, Any], problem_text: str) -> str:
        messages = build_solver_messages(problem, problem_text)
        try:
            return self.client.chat(
                messages=messages,
                temperature=0.1,
                max_tokens=8192,
                thinking_mode=True,
            )
        except TypeError:
            return self.client.chat(
                messages=messages,
                temperature=0.1,
                max_tokens=8192,
            )

    def _normalize_problem_input(self, problem: Any, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(problem, dict):
            normalized = dict(problem)
        else:
            safe_metadata = metadata if isinstance(metadata, dict) else {}
            problem_id = safe_metadata.get("problem_id", safe_metadata.get("idx", safe_metadata.get("id", "UNKNOWN")))
            normalized = {
                "problem_id": str(problem_id),
                "problem_text": str(problem or ""),
            }
            for key in ("subject", "type", "category"):
                if key in safe_metadata:
                    normalized[key] = safe_metadata[key]
        return normalized

    def _get_problem_text(self, problem: Dict[str, Any]) -> str:
        text = problem.get("problem_text")
        if text is None:
            text = problem.get("problem")
        return str(text or "")

    def _resolve_backend(self, backend: str) -> str:
        if backend == "lagent":
            try:
                __import__("lagent")
                return "lagent"
            except ImportError:
                return "simple"
        return "simple"

    def _load_schema(self, schema_path: Optional[Path]) -> Dict[str, Any]:
        if schema_path is None:
            schema_path = Path(__file__).resolve().parents[1] / "result_schema.json"
        with open(schema_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _needs_repair(self, result: Dict[str, Any], schema_valid: bool) -> bool:
        if not self.enable_repair:
            return False
        verification = result.get("verification", {})
        return (
            not schema_valid
            or verification.get("verification_result") == "fail"
            or float(verification.get("confidence", 0.0)) < 0.75
            or not result.get("final_answer", {}).get("answer")
        )

    def _verification_error(self, result: Dict[str, Any]) -> str:
        verification = result.get("verification", {})
        if verification.get("verification_result") == "fail":
            return verification.get("verification_process", "verification failed")
        if float(verification.get("confidence", 0.0)) < 0.75:
            return "confidence below threshold"
        return "empty final answer or uncertain result"

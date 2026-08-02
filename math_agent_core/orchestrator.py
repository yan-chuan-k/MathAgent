from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .json_utils import ValidationResult, extract_json_from_text, repair_json_locally, validate_result
from .prompts import build_solver_messages
from .router import classify_problem
from .schema import empty_result
from .state import EvidenceStatus, FailureKind, OverallStatus, SolveAssessment, VerificationEvidence
from .tools import run_sympy_verification


class MathAgentOrchestrator:
    def __init__(
        self,
        client: Any,
        max_retries: int = 2,
        enable_repair: bool = True,
        enable_tool_verify: bool = True,
        backend: str = "simple",
        schema_path: Optional[Path] = None,
        thinking_mode: bool = True,
    ):
        self.client = client
        self.max_retries = max_retries
        self.enable_repair = enable_repair
        self.enable_tool_verify = enable_tool_verify
        self.thinking_mode = thinking_mode
        self.backend = self._resolve_backend(backend)
        self.model = getattr(client, "model", "intern-s2-preview-397b")
        self.schema = self._load_schema(schema_path)
        self.last_log: Dict[str, Any] = {}

    def solve(self, problem: Any, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        started = time.time()
        problem = self._normalize_problem_input(problem, metadata)
        problem_id = str(problem.get("problem_id") or "UNKNOWN")
        problem_text = self._get_problem_text(problem)
        route_hint = classify_problem(problem_text, problem)
        problem["_route_hint"] = route_hint
        log = {
            "problem_id": problem_id,
            "input": problem,
            "parsed": {"problem_text": problem_text},
            "route": route_hint,
            "plan": [],
            "solver_raw_output": "",
            "solver_result": {},
            "verification": {},
            "verification_evidence": [],
            "settings": {"thinking_mode": self.thinking_mode},
            "repair_history": [],
            "final_result": {},
            "timing": {"start_time": started, "end_time": None, "elapsed_seconds": 0.0},
            "errors": [],
        }

        result: Dict[str, Any] = empty_result(problem_id, model=self.model, backend=self.backend)
        repair_context: Dict[str, Any] | None = None
        for attempt in range(1, self.max_retries + 2):
            try:
                raw_output = self._call_solver(problem, problem_text, repair_context=repair_context)
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
                if result.get("problem_type") in ("", "unknown") and route_hint["primary_domain"] != "unknown":
                    result["problem_type"] = route_hint["primary_domain"]
                if result.get("domain_candidates") in ([], ["unknown"]) and route_hint["domain_candidates"] != ["unknown"]:
                    result["domain_candidates"] = route_hint["domain_candidates"]
                if result.get("problem_type") in ("", "unknown") and result.get("domain_candidates"):
                    result["problem_type"] = str(result["domain_candidates"][0])
                validation = validate_result(result, self.schema)
                result["_meta"]["schema_valid"] = validation.valid
                result["_meta"]["schema_error"] = validation.error
                log["solver_result"] = result
                log["route"] = {
                    "primary_domain": (result.get("domain_candidates") or [result.get("problem_type") or route_hint["primary_domain"]])[0],
                    "domain_candidates": result.get("domain_candidates") or route_hint["domain_candidates"],
                    "local_route_hint": route_hint,
                    "task_type": result.get("task_type", "unknown"),
                    "needs_tool_verification": self.enable_tool_verify,
                    "thinking_mode": self.thinking_mode,
                }
                log["plan"] = result.get("reasoning_plan", [])
                evidence = self._run_verifiers(problem_text, result) if self.enable_tool_verify else []
                assessment = self._assess_result(result, validation, evidence)
                self._apply_assessment(result, assessment)
                final_validation = validate_result(result, self.schema)
                result["_meta"]["schema_valid"] = final_validation.valid
                result["_meta"]["schema_error"] = final_validation.error
                if not final_validation.valid:
                    assessment = self._assess_result(result, final_validation, evidence)
                    self._apply_assessment(result, assessment)
                log["verification"] = result.get("verification", {})
                log["verification_evidence"] = assessment.evidence_dicts()

                if self._needs_repair(result, assessment) and attempt <= self.max_retries:
                    repair_context = self._build_repair_context(result, assessment, validation)
                    log["repair_history"].append(
                        {
                            "attempt": attempt + 1,
                            "failure_kind": assessment.failure_kind,
                            "previous_error": assessment.failure_details or validation.error or self._verification_error(result),
                            "repair_strategy": "targeted retry with concrete validation evidence",
                        }
                    )
                    continue
                break
            except Exception as exc:
                log["errors"].append({"attempt": attempt, "error": str(exc)})
                result = empty_result(problem_id, model=self.model, backend=self.backend)
                result["_meta"]["attempts"] = attempt
                result["_meta"]["schema_error"] = str(exc)
                result["_meta"]["overall_status"] = OverallStatus.ERROR.value
                result["_meta"]["failure_kind"] = FailureKind.JSON_PARSE.value
                result["_meta"]["failure_details"] = str(exc)[:500]
                if attempt <= self.max_retries:
                    repair_context = {
                        "failure_kind": FailureKind.JSON_PARSE.value,
                        "failure_details": str(exc)[:500],
                        "instruction": "Return exactly one valid JSON object matching the schema. Do not change the math unless needed.",
                    }
                    continue
                if attempt > self.max_retries:
                    break

        elapsed = time.time() - started
        result["_meta"]["elapsed_seconds"] = elapsed
        log["final_result"] = result
        log["timing"]["end_time"] = time.time()
        log["timing"]["elapsed_seconds"] = elapsed
        self.last_log = log
        return result

    def _call_solver(
        self,
        problem: Dict[str, Any],
        problem_text: str,
        repair_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        messages = build_solver_messages(problem, problem_text, repair_context=repair_context)
        try:
            return self.client.chat(
                messages=messages,
                temperature=0.1,
                max_tokens=8192,
                thinking_mode=self.thinking_mode,
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

    def _run_verifiers(self, problem_text: str, result: Dict[str, Any]) -> list[VerificationEvidence]:
        answer = ""
        final_answer = result.get("final_answer")
        if isinstance(final_answer, dict):
            answer = str(final_answer.get("answer") or "")
        try:
            return run_sympy_verification(problem_text=problem_text, answer=answer, result=result)
        except Exception as exc:
            return [
                VerificationEvidence(
                    verifier="safe_sympy",
                    claim_id="verifier_error",
                    status=EvidenceStatus.INCONCLUSIVE.value,
                    method="exception_boundary",
                    details=f"{type(exc).__name__}: {str(exc)[:220]}",
                )
            ]

    def _assess_result(
        self,
        result: Dict[str, Any],
        validation: ValidationResult,
        evidence: list[VerificationEvidence],
    ) -> SolveAssessment:
        answer = ""
        final_answer = result.get("final_answer")
        if isinstance(final_answer, dict):
            answer = str(final_answer.get("answer") or "").strip()
        task_type = str(result.get("task_type") or "unknown")
        solution = result.get("solution")
        has_solution = bool(solution) if isinstance(solution, list) else bool(solution)
        content_complete = bool(answer) and (has_solution or task_type != "proof")
        model_verification = result.get("verification", {}) if isinstance(result.get("verification"), dict) else {}
        model_pass = model_verification.get("verification_result") == "pass"

        if not validation.valid:
            return SolveAssessment(
                schema_valid=False,
                content_complete=content_complete,
                answer_verified=False,
                proof_verified=False,
                overall_status=OverallStatus.INVALID.value,
                failure_kind=FailureKind.SCHEMA.value,
                failure_details=validation.error or "schema validation failed",
                evidence=evidence,
            )
        if not content_complete:
            return SolveAssessment(
                schema_valid=True,
                content_complete=False,
                answer_verified=False,
                proof_verified=False,
                overall_status=OverallStatus.INVALID.value,
                failure_kind=FailureKind.WRONG_FINAL_ANSWER.value,
                failure_details="final answer or solution is empty",
                evidence=evidence,
            )

        statuses = [item.status for item in evidence]
        failed = [item for item in evidence if item.status == EvidenceStatus.FAIL.value]
        passed = [item for item in evidence if item.status == EvidenceStatus.PASS.value]
        supported = [item for item in evidence if item.claim_id != "no_supported_check"]
        if failed:
            return SolveAssessment(
                schema_valid=True,
                content_complete=True,
                answer_verified=False,
                proof_verified=False,
                overall_status=OverallStatus.INVALID.value,
                failure_kind=self._failure_kind_from_evidence(failed[0]),
                failure_details=self._evidence_summary(failed[0]),
                evidence=evidence,
            )
        if passed and supported:
            proof_verified = task_type == "proof" and model_pass
            return SolveAssessment(
                schema_valid=True,
                content_complete=True,
                answer_verified=True,
                proof_verified=proof_verified,
                overall_status=OverallStatus.SOLVED.value,
                evidence=evidence,
            )
        if model_pass:
            return SolveAssessment(
                schema_valid=True,
                content_complete=True,
                answer_verified=False,
                proof_verified=task_type == "proof",
                overall_status=OverallStatus.PROBABLE.value,
                failure_kind=FailureKind.INCONCLUSIVE.value,
                failure_details="no conclusive tool evidence; relying on structured model verification only",
                evidence=evidence,
            )
        return SolveAssessment(
            schema_valid=True,
            content_complete=True,
            answer_verified=False,
            proof_verified=False,
            overall_status=OverallStatus.UNCERTAIN.value,
            failure_kind=FailureKind.INCONCLUSIVE.value,
            failure_details="verification did not pass and no conclusive tool evidence is available",
            evidence=evidence,
        )

    def _apply_assessment(self, result: Dict[str, Any], assessment: SolveAssessment) -> None:
        verification = result.get("verification")
        if not isinstance(verification, dict):
            verification = {}
            result["verification"] = verification
        verification["evidence"] = assessment.evidence_dicts()
        if assessment.overall_status == OverallStatus.SOLVED.value:
            verification["verification_result"] = "pass"
            verification["confidence"] = max(float(verification.get("confidence") or 0.0), 0.9)
        elif assessment.overall_status == OverallStatus.INVALID.value:
            verification["verification_result"] = "fail"
            verification["confidence"] = min(float(verification.get("confidence") or 0.0), 0.4)
        elif assessment.overall_status == OverallStatus.PROBABLE.value:
            verification["verification_result"] = "uncertain"
            verification["confidence"] = min(max(float(verification.get("confidence") or 0.0), 0.65), 0.84)
        else:
            verification["verification_result"] = "uncertain"
            verification["confidence"] = min(float(verification.get("confidence") or 0.0), 0.6)
        if assessment.failure_details and not verification.get("verification_process"):
            verification["verification_process"] = assessment.failure_details
        result["_meta"].update(assessment.to_meta_fields())

    def _needs_repair(self, result: Dict[str, Any], assessment: SolveAssessment) -> bool:
        if not self.enable_repair:
            return False
        return assessment.overall_status in {OverallStatus.INVALID.value, OverallStatus.UNCERTAIN.value}

    def _build_repair_context(
        self,
        result: Dict[str, Any],
        assessment: SolveAssessment,
        validation: ValidationResult,
    ) -> Dict[str, Any]:
        final_answer = result.get("final_answer") if isinstance(result.get("final_answer"), dict) else {}
        instruction = "Repair the previous answer using the failure evidence."
        if assessment.failure_kind == FailureKind.SCHEMA.value:
            instruction = "Fix only the schema and missing required fields unless the math answer is empty."
        elif assessment.failure_kind == FailureKind.SYMBOLIC_CONTRADICTION.value:
            instruction = "The candidate answer contradicts a symbolic substitution check. Recompute the affected step and answer."
        elif assessment.failure_kind == FailureKind.NUMERIC_RESIDUAL.value:
            instruction = "The numeric residual is nonzero. Recalculate and return the corrected final answer."
        elif assessment.failure_kind == FailureKind.INCONCLUSIVE.value:
            instruction = "Provide a clearer answer and include concrete requested_checks that can be safely verified."
        return {
            "failure_kind": assessment.failure_kind,
            "failure_details": assessment.failure_details,
            "schema_error": validation.error,
            "previous_answer": final_answer.get("answer", ""),
            "evidence": assessment.evidence_dicts(),
            "instruction": instruction,
        }

    def _failure_kind_from_evidence(self, evidence: VerificationEvidence) -> str:
        if evidence.method in {"equation_solution", "symbolic_equivalence", "derivative_check", "integral_check"}:
            return FailureKind.SYMBOLIC_CONTRADICTION.value
        if evidence.method == "numeric_arithmetic":
            return FailureKind.NUMERIC_RESIDUAL.value
        return FailureKind.WRONG_FINAL_ANSWER.value

    def _evidence_summary(self, evidence: VerificationEvidence) -> str:
        residual = f"; residual={evidence.residual}" if evidence.residual is not None else ""
        return f"{evidence.method}: {evidence.details}{residual}"[:500]

    def _verification_error(self, result: Dict[str, Any]) -> str:
        verification = result.get("verification", {})
        if verification.get("verification_result") == "fail":
            if verification.get("checks"):
                return "; ".join(str(item) for item in verification.get("checks", [])[:3])
            return verification.get("verification_process", "verification failed")
        if float(verification.get("confidence", 0.0)) < 0.75:
            return "confidence below threshold"
        return "empty final answer or uncertain result"

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .answer_utils import normalize_final_response
from .candidate import CandidateSolution
from .evaluation import answer_cluster_key
from .json_utils import ValidationResult, extract_json_from_text, repair_json_locally, validate_result
from .prompts import build_critic_messages, build_finalizer_messages, build_planner_messages, build_solver_messages
from .router import classify_problem
from .schema import empty_result
from .search import choose_strategy_budget, rank_candidates, strategies_for_domain
from .state import EvidenceStatus, FailureKind, OverallStatus, SolveAssessment, SolveState, VerificationEvidence
from .tools import run_sympy_verification
from .verifiers import check_completeness


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
        max_candidates: int = 1,
        enable_critic: bool = True,
        enable_finalizer: bool = True,
    ):
        self.client = client
        self.max_retries = max_retries
        self.enable_repair = enable_repair
        self.enable_tool_verify = enable_tool_verify
        self.thinking_mode = thinking_mode
        self.max_candidates = max(1, int(max_candidates))
        self.enable_critic = enable_critic
        self.enable_finalizer = enable_finalizer
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
            "candidates": [],
            "state": {},
            "verification": {},
            "verification_evidence": [],
            "settings": {"thinking_mode": self.thinking_mode},
            "repair_history": [],
            "final_result": {},
            "timing": {"start_time": started, "end_time": None, "elapsed_seconds": 0.0},
            "errors": [],
        }
        solve_state = SolveState(
            problem=problem_text,
            route=route_hint,
            open_goals=[],
            budget={"max_candidates": self.max_candidates, "max_retries": self.max_retries},
        )

        strategies = self._select_strategies(problem, problem_text, route_hint)
        candidates: list[CandidateSolution] = []
        for candidate_index, strategy in enumerate(strategies, start=1):
            solve_state.current_strategy = strategy
            candidate = self._solve_candidate(
                problem=problem,
                problem_text=problem_text,
                problem_id=problem_id,
                route_hint=route_hint,
                strategy=strategy,
                candidate_index=candidate_index,
                started=started,
                log=log,
            )
            candidates.append(candidate)
            self._update_solve_state(solve_state, candidate)
            if candidate.assessment.overall_status == OverallStatus.SOLVED.value and candidate_index == 1:
                break

        candidates = self._cluster_and_rank_candidates(candidates)
        selected = candidates[0] if candidates else None
        result = selected.result if selected is not None else empty_result(problem_id, model=self.model, backend=self.backend)
        if selected is not None and self.enable_finalizer:
            final_response = self._call_finalizer(problem_text, selected.result)
            if final_response:
                result["final_response"] = final_response
        if selected is None:
            result["_meta"]["overall_status"] = OverallStatus.ERROR.value
            result["_meta"]["failure_kind"] = FailureKind.INCONCLUSIVE.value
            result["_meta"]["failure_details"] = "no candidate could be generated"
        log["candidates"] = [candidate.to_trace_dict() for candidate in candidates]
        log["state"] = solve_state.compact()
        log["solver_result"] = result
        log["verification"] = result.get("verification", {})
        log["verification_evidence"] = selected.assessment.evidence_dicts() if selected is not None else []
        log["plan"] = result.get("reasoning_plan", [])
        log["route"] = {
            "primary_domain": (result.get("domain_candidates") or [result.get("problem_type") or route_hint["primary_domain"]])[0],
            "domain_candidates": result.get("domain_candidates") or route_hint["domain_candidates"],
            "local_route_hint": route_hint,
            "task_type": result.get("task_type", "unknown"),
            "needs_tool_verification": self.enable_tool_verify,
            "thinking_mode": self.thinking_mode,
            "candidate_count": len(candidates),
            "enable_critic": self.enable_critic,
            "enable_finalizer": self.enable_finalizer,
        }

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
        strategy: Optional[str] = None,
    ) -> str:
        messages = build_solver_messages(problem, problem_text, repair_context=repair_context, strategy=strategy)
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

    def _select_strategies(
        self,
        problem: Dict[str, Any],
        problem_text: str,
        route_hint: Dict[str, Any],
    ) -> list[str]:
        primary_domain = route_hint.get("primary_domain") or "unknown"
        pool = strategies_for_domain(str(primary_domain))
        budget = choose_strategy_budget(str(problem.get("task_type") or "unknown"), self.max_candidates)
        selected = pool[:budget]
        if self.max_candidates > 2:
            planner_selected = self._call_planner(problem, problem_text, pool)
            for strategy in planner_selected:
                if strategy in pool and strategy not in selected:
                    selected.append(strategy)
                if len(selected) >= budget:
                    break
        return selected[:budget] or ["direct_computation"]

    def _call_planner(self, problem: Dict[str, Any], problem_text: str, strategies: list[str]) -> list[str]:
        try:
            messages = build_planner_messages(problem, problem_text, strategies)
            raw = self._chat(messages, temperature=0.0, max_tokens=1024)
            parsed = extract_json_from_text(raw)
            selected = parsed.get("selected_strategies")
            if isinstance(selected, list):
                return [str(item) for item in selected]
        except Exception:
            return []
        return []

    def _solve_candidate(
        self,
        problem: Dict[str, Any],
        problem_text: str,
        problem_id: str,
        route_hint: Dict[str, Any],
        strategy: str,
        candidate_index: int,
        started: float,
        log: Dict[str, Any],
    ) -> CandidateSolution:
        result: Dict[str, Any] = empty_result(problem_id, model=self.model, backend=self.backend)
        assessment = SolveAssessment(
            schema_valid=False,
            content_complete=False,
            answer_verified=False,
            proof_verified=False,
            overall_status=OverallStatus.ERROR.value,
            failure_kind=FailureKind.JSON_PARSE.value,
            failure_details="candidate not attempted",
        )
        repair_context: Dict[str, Any] | None = None
        last_evidence: list[VerificationEvidence] = []
        for attempt in range(1, self.max_retries + 2):
            try:
                raw_output = self._call_solver(problem, problem_text, repair_context=repair_context, strategy=strategy)
                log["solver_raw_output"] = raw_output
                parsed = extract_json_from_text(raw_output)
                result, assessment, last_evidence, validation = self._normalize_validate_assess(
                    parsed=parsed,
                    problem_id=problem_id,
                    route_hint=route_hint,
                    attempt=attempt,
                    started=started,
                    problem_text=problem_text,
                )
                if self.enable_critic and assessment.overall_status in {OverallStatus.SOLVED.value, OverallStatus.PROBABLE.value}:
                    critic = self._call_critic(problem_text, result, assessment.evidence_dicts())
                    if critic:
                        self._apply_critic(result, assessment, critic)
                else:
                    critic = None

                if self._needs_repair(result, assessment) and attempt <= self.max_retries:
                    repair_context = self._build_repair_context(result, assessment, validation)
                    log["repair_history"].append(
                        {
                            "candidate": candidate_index,
                            "strategy": strategy,
                            "attempt": attempt + 1,
                            "failure_kind": assessment.failure_kind,
                            "previous_error": assessment.failure_details or validation.error or self._verification_error(result),
                            "repair_strategy": "targeted retry with concrete validation evidence",
                        }
                    )
                    continue
                return self._make_candidate(candidate_index, strategy, result, assessment, last_evidence, critic)
            except Exception as exc:
                result = empty_result(problem_id, model=self.model, backend=self.backend)
                result["_meta"]["attempts"] = attempt
                result["_meta"]["schema_error"] = str(exc)
                result["_meta"]["overall_status"] = OverallStatus.ERROR.value
                result["_meta"]["failure_kind"] = FailureKind.JSON_PARSE.value
                result["_meta"]["failure_details"] = str(exc)[:500]
                assessment = SolveAssessment(
                    schema_valid=False,
                    content_complete=False,
                    answer_verified=False,
                    proof_verified=False,
                    overall_status=OverallStatus.ERROR.value,
                    failure_kind=FailureKind.JSON_PARSE.value,
                    failure_details=str(exc)[:500],
                )
                log["errors"].append({"candidate": candidate_index, "strategy": strategy, "attempt": attempt, "error": str(exc)})
                if attempt <= self.max_retries:
                    repair_context = {
                        "failure_kind": FailureKind.JSON_PARSE.value,
                        "failure_details": str(exc)[:500],
                        "instruction": "Return exactly one valid JSON object matching the schema. Do not change the math unless needed.",
                    }
                    continue
                break
        return self._make_candidate(candidate_index, strategy, result, assessment, last_evidence, None)

    def _normalize_validate_assess(
        self,
        parsed: Dict[str, Any],
        problem_id: str,
        route_hint: Dict[str, Any],
        attempt: int,
        started: float,
        problem_text: str,
    ) -> tuple[Dict[str, Any], SolveAssessment, list[VerificationEvidence], ValidationResult]:
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
        evidence = self._run_verifiers(problem_text, result) if self.enable_tool_verify else []
        assessment = self._assess_result(result, validation, evidence)
        self._apply_assessment(result, assessment)
        final_validation = validate_result(result, self.schema)
        result["_meta"]["schema_valid"] = final_validation.valid
        result["_meta"]["schema_error"] = final_validation.error
        if not final_validation.valid:
            validation = final_validation
            assessment = self._assess_result(result, final_validation, evidence)
            self._apply_assessment(result, assessment)
        return result, assessment, evidence, validation

    def _call_critic(self, problem_text: str, result: Dict[str, Any], evidence: list[Dict[str, Any]]) -> Dict[str, Any] | None:
        try:
            messages = build_critic_messages(problem_text, result, evidence)
            raw = self._chat(messages, temperature=0.0, max_tokens=1024)
            parsed = extract_json_from_text(raw)
            status = str(parsed.get("status") or "inconclusive")
            if status not in {"pass", "fail", "inconclusive"}:
                status = "inconclusive"
            return {
                "status": status,
                "failure_kind": str(parsed.get("failure_kind") or ""),
                "first_error": str(parsed.get("first_error") or "")[:500],
                "missing_targets": parsed.get("missing_targets", []),
                "suggested_repair": str(parsed.get("suggested_repair") or "")[:500],
            }
        except Exception:
            return None

    def _call_finalizer(self, problem_text: str, selected_candidate: Dict[str, Any]) -> str:
        try:
            messages = build_finalizer_messages(problem_text, selected_candidate)
            raw = self._chat(messages, temperature=0.0, max_tokens=512)
            parsed = extract_json_from_text(raw)
            value = parsed.get("final_response")
            if isinstance(value, str) and value.strip():
                return normalize_final_response(value, problem=problem_text)
        except Exception:
            return ""
        return ""

    def _apply_critic(self, result: Dict[str, Any], assessment: SolveAssessment, critic: Dict[str, Any]) -> None:
        if critic.get("status") != "fail":
            return
        assessment.overall_status = OverallStatus.INVALID.value
        assessment.failure_kind = critic.get("failure_kind") or FailureKind.INCONCLUSIVE.value
        assessment.failure_details = critic.get("first_error") or critic.get("suggested_repair") or "critic rejected candidate"
        self._apply_assessment(result, assessment)

    def _make_candidate(
        self,
        candidate_index: int,
        strategy: str,
        result: Dict[str, Any],
        assessment: SolveAssessment,
        evidence: list[VerificationEvidence],
        critic: Dict[str, Any] | None,
    ) -> CandidateSolution:
        answer = ""
        final_answer = result.get("final_answer") if isinstance(result, dict) else None
        if isinstance(final_answer, dict):
            answer = str(final_answer.get("answer") or "")
        normalized = normalize_final_response(answer)
        return CandidateSolution(
            candidate_id=f"candidate_{candidate_index}",
            strategy=strategy,
            result=result,
            assessment=assessment,
            evidence=evidence,
            critic=critic,
            normalized_answer=normalized,
            cluster_id=answer_cluster_key(normalized),
        )

    def _cluster_and_rank_candidates(self, candidates: list[CandidateSolution]) -> list[CandidateSolution]:
        for candidate in candidates:
            candidate.cluster_id = candidate.cluster_id or answer_cluster_key(candidate.normalized_answer)
        return rank_candidates(candidates)

    def _chat(self, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> str:
        try:
            return self.client.chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                thinking_mode=self.thinking_mode,
            )
        except TypeError:
            return self.client.chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
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
        evidence: list[VerificationEvidence] = []
        try:
            evidence.extend(run_sympy_verification(problem_text=problem_text, answer=answer, result=result))
        except Exception as exc:
            evidence.append(
                VerificationEvidence(
                    verifier="safe_sympy",
                    claim_id="verifier_error",
                    status=EvidenceStatus.INCONCLUSIVE.value,
                    method="exception_boundary",
                    details=f"{type(exc).__name__}: {str(exc)[:220]}",
                )
            )
        try:
            evidence.extend(check_completeness(problem_text, result))
        except Exception as exc:
            evidence.append(
                VerificationEvidence(
                    verifier="completeness",
                    claim_id="verifier_error",
                    status=EvidenceStatus.INCONCLUSIVE.value,
                    method="exception_boundary",
                    details=f"{type(exc).__name__}: {str(exc)[:220]}",
                )
            )
        return evidence

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
        verifier_passed = [item for item in passed if item.verifier != "completeness"]
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
        if verifier_passed and supported:
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
        if evidence.verifier == "completeness":
            return FailureKind.MISSING_CASE.value
        if evidence.method in {"equation_solution", "symbolic_equivalence", "derivative_check", "integral_check"}:
            return FailureKind.SYMBOLIC_CONTRADICTION.value
        if evidence.method == "numeric_arithmetic":
            return FailureKind.NUMERIC_RESIDUAL.value
        return FailureKind.WRONG_FINAL_ANSWER.value

    def _update_solve_state(self, solve_state: SolveState, candidate: CandidateSolution) -> None:
        solve_state.candidates.append(candidate.to_trace_dict())
        solve_state.verification_evidence.extend(candidate.assessment.evidence_dicts())
        meta = candidate.result.get("_meta", {}) if isinstance(candidate.result, dict) else {}
        if meta.get("overall_status") in {OverallStatus.INVALID.value, OverallStatus.ERROR.value}:
            solve_state.rejected_attempts.append(candidate.to_trace_dict())
            solve_state.rejected_strategies.append(candidate.strategy)
        for item in candidate.assessment.evidence:
            if item.verifier == "completeness" and item.status == EvidenceStatus.FAIL.value and item.residual:
                solve_state.open_goals.extend(goal.strip() for goal in item.residual.split(",") if goal.strip())

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

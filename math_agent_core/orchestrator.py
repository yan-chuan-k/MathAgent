from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .acceptance import AcceptancePolicy
from .answer_utils import normalize_final_response
from .candidate import CandidateSolution
from .evaluation import answer_cluster_key
from .json_utils import ValidationResult, extract_json_from_text, repair_json_locally, validate_result
from .prompts import build_critic_messages, build_finalizer_messages, build_planner_messages, build_solver_messages
from .router import classify_problem
from .schema import empty_result
from .search import choose_strategy_budget, rank_candidates, strategies_for_domain
from .state import EvidenceStatus, FailureKind, OverallStatus, SolveAssessment, SolveState, VerificationEvidence, VerificationLevel
from .tools import run_sympy_verification
from .verifiers import check_completeness, run_linear_algebra_verification


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
        self.acceptance_policy = AcceptancePolicy()
        self.last_log: Dict[str, Any] = {}
        self._model_call_counts: Dict[str, int] = {
            "solver": 0,
            "planner": 0,
            "critic": 0,
            "finalizer": 0,
        }

    def solve(self, problem: Any, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        started = time.time()
        self._model_call_counts = {key: 0 for key in self._model_call_counts}
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
        strategies = self._select_strategies(problem, problem_text, route_hint)
        solve_state = SolveState(
            problem=problem_text,
            route=route_hint,
            open_goals=[],
            budget={
                "max_candidates": self.max_candidates,
                "initial_candidate_budget": len(strategies),
                "max_retries": self.max_retries,
            },
        )

        candidates: list[CandidateSolution] = []
        candidate_index = 0
        while candidate_index < len(strategies):
            strategy = strategies[candidate_index]
            candidate_index += 1
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
            if (
                candidate_index == 1
                and len(strategies) == 1
                and self.max_candidates >= 2
                and candidate.assessment.overall_status == OverallStatus.UNCERTAIN.value
            ):
                primary_domain = str(route_hint.get("primary_domain") or "unknown")
                alternate = next(
                    (item for item in strategies_for_domain(primary_domain) if item not in strategies),
                    None,
                )
                if alternate:
                    strategies.append(alternate)
                    solve_state.budget["expanded_after_uncertain"] = True

        self._review_conflicting_candidates(problem_text, candidates)
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
            "task_type": result.get("task_type") or route_hint.get("task_type", "unknown"),
            "verifiability": route_hint.get("verifiability", "low"),
            "needs_tool_verification": self.enable_tool_verify,
            "thinking_mode": self.thinking_mode,
            "candidate_count": len(candidates),
            "enable_critic": self.enable_critic,
            "enable_finalizer": self.enable_finalizer,
        }
        log["model_calls"] = {
            **self._model_call_counts,
            "total": sum(self._model_call_counts.values()),
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
        self._increment_model_call("solver")
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
        task_type = str(route_hint.get("task_type") or problem.get("task_type") or "unknown")
        verifiability = str(route_hint.get("verifiability") or "low")
        budget = choose_strategy_budget(task_type, self.max_candidates, verifiability)
        if budget <= 2:
            return pool[:budget] or ["direct_computation"]

        selected: list[str] = []
        planner_selected = self._call_planner(problem, problem_text, pool)
        for strategy in planner_selected:
            if strategy in pool and strategy not in selected:
                selected.append(strategy)
        for strategy in pool:
            if strategy not in selected:
                selected.append(strategy)
            if len(selected) >= budget:
                break
        return selected[:budget] or ["direct_computation"]

    def _call_planner(self, problem: Dict[str, Any], problem_text: str, strategies: list[str]) -> list[str]:
        try:
            messages = build_planner_messages(problem, problem_text, strategies)
            self._increment_model_call("planner")
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
                if self.enable_critic and assessment.overall_status == OverallStatus.PROBABLE.value:
                    critic = self._call_critic(problem_text, result, assessment.evidence_dicts())
                    if critic:
                        self._apply_critic(result, assessment, critic)
                        assessment = self._reassess_with_current_evidence(result, validation)
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
        if result.get("task_type") in ("", "unknown") and route_hint.get("task_type") != "unknown":
            result["task_type"] = str(route_hint["task_type"])

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
            self._increment_model_call("critic")
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
            self._increment_model_call("finalizer")
            raw = self._chat(messages, temperature=0.0, max_tokens=512)
            parsed = extract_json_from_text(raw)
            value = parsed.get("final_response")
            if isinstance(value, str) and value.strip():
                return normalize_final_response(value, problem=problem_text)
        except Exception:
            return ""
        return ""

    def _review_conflicting_candidates(self, problem_text: str, candidates: list[CandidateSolution]) -> None:
        if not self.enable_critic or len(candidates) < 2:
            return
        if any(candidate.assessment.overall_status == OverallStatus.SOLVED.value for candidate in candidates):
            return
        if len({candidate.cluster_id for candidate in candidates if candidate.cluster_id}) < 2:
            return

        review_target = next(
            (
                candidate
                for candidate in candidates
                if candidate.critic is None
                and candidate.assessment.overall_status in {OverallStatus.UNCERTAIN.value, OverallStatus.PROBABLE.value}
            ),
            None,
        )
        if review_target is None:
            return
        critic = self._call_critic(problem_text, review_target.result, review_target.assessment.evidence_dicts())
        if critic is None:
            return
        review_target.critic = critic
        self._apply_critic(review_target.result, review_target.assessment, critic)
        validation = validate_result(review_target.result, self.schema)
        review_target.assessment = self._reassess_with_current_evidence(review_target.result, validation)
        review_target.evidence = list(review_target.assessment.evidence)

    def _apply_critic(self, result: Dict[str, Any], assessment: SolveAssessment, critic: Dict[str, Any]) -> None:
        assessment.evidence.append(self._critic_to_evidence(critic))
        self._apply_assessment(result, assessment)

    def _critic_to_evidence(self, critic: Dict[str, Any]) -> VerificationEvidence:
        status = str(critic.get("status") or EvidenceStatus.INCONCLUSIVE.value)
        if status not in {EvidenceStatus.PASS.value, EvidenceStatus.FAIL.value, EvidenceStatus.INCONCLUSIVE.value}:
            status = EvidenceStatus.INCONCLUSIVE.value
        details = critic.get("first_error") or critic.get("suggested_repair") or "critic review"
        return VerificationEvidence(
            verifier="critic",
            claim_id="critic_review",
            status=status,
            method="model_critic",
            details=str(details)[:500],
            residual=None,
            assumptions=[],
            verification_level=VerificationLevel.MODEL_CRITIC.value,
            is_decisive=False,
        )

    def _reassess_with_current_evidence(self, result: Dict[str, Any], validation: ValidationResult) -> SolveAssessment:
        verification = result.get("verification") if isinstance(result.get("verification"), dict) else {}
        evidence: list[VerificationEvidence] = []
        raw_evidence = verification.get("evidence", [])
        if isinstance(raw_evidence, list):
            for item in raw_evidence:
                if not isinstance(item, dict):
                    continue
                assumptions = item.get("assumptions", [])
                if isinstance(assumptions, str):
                    assumptions = [assumptions]
                if not isinstance(assumptions, list):
                    assumptions = []
                evidence.append(
                    VerificationEvidence(
                        verifier=str(item.get("verifier") or "unknown"),
                        claim_id=str(item.get("claim_id") or "claim"),
                        status=str(item.get("status") or EvidenceStatus.INCONCLUSIVE.value),
                        method=str(item.get("method") or "unknown"),
                        details=str(item.get("details") or ""),
                        residual=None if item.get("residual") is None else str(item.get("residual")),
                        assumptions=[str(value) for value in assumptions],
                        verification_level=str(item.get("verification_level") or VerificationLevel.MODEL_CRITIC.value),
                        is_decisive=bool(item.get("is_decisive", False)),
                    )
                )
        assessment = self._assess_result(result, validation, evidence)
        self._apply_assessment(result, assessment)
        return assessment

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

    def _increment_model_call(self, role: str) -> None:
        if role not in self._model_call_counts:
            self._model_call_counts[role] = 0
        self._model_call_counts[role] += 1

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
            for key in ("subject", "type", "category", "task_type"):
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
                    verification_level=VerificationLevel.EXACT_SYMBOLIC.value,
                    is_decisive=False,
                )
            )
        try:
            linear_evidence = run_linear_algebra_verification(result)
            if not (len(linear_evidence) == 1 and linear_evidence[0].claim_id == "no_matrix_check"):
                evidence.extend(linear_evidence)
        except Exception as exc:
            evidence.append(
                VerificationEvidence(
                    verifier="linear_algebra",
                    claim_id="verifier_error",
                    status=EvidenceStatus.INCONCLUSIVE.value,
                    method="exception_boundary",
                    details=f"{type(exc).__name__}: {str(exc)[:220]}",
                    verification_level=VerificationLevel.EXACT_SYMBOLIC.value,
                    is_decisive=False,
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
                    verification_level=VerificationLevel.COMPLETENESS_ONLY.value,
                    is_decisive=False,
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
        answer_type = "unknown"
        final_answer_payload = result.get("final_answer") if isinstance(result.get("final_answer"), dict) else {}
        if isinstance(final_answer_payload, dict):
            answer_type = str(final_answer_payload.get("answer_type") or "unknown")
        decision = self.acceptance_policy.decide(
            schema_valid=validation.valid,
            content_complete=content_complete,
            task_type=task_type,
            answer_type=answer_type,
            model_verification_pass=model_pass,
            evidence=evidence,
            schema_error=validation.error,
        )
        return SolveAssessment(
            schema_valid=validation.valid,
            content_complete=content_complete,
            answer_verified=decision.answer_verified,
            proof_verified=decision.proof_verified,
            overall_status=decision.overall_status,
            failure_kind=decision.failure_kind,
            failure_details=decision.failure_details,
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
        if assessment.overall_status == OverallStatus.INVALID.value:
            return True
        return assessment.overall_status == OverallStatus.UNCERTAIN.value and self.max_candidates == 1

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

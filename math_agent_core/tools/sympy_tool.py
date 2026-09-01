from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from fractions import Fraction
from typing import Any, Dict, List, Optional

from math_agent_core.state import EvidenceStatus, VerificationEvidence, VerificationLevel


MAX_EXPR_LENGTH = 240
MAX_TOOL_TIMEOUT_SECONDS = 2.0
_DANGEROUS_TOKENS = ("__", "import", "exec", "eval", "lambda", "open(", "read(", "write(", "os.", "sys.")
_CANDIDATE_CHECK_NOTICE = "Candidate-proposed auxiliary check; not sufficient to verify final_answer."


class SafeSympyTool:
    """Small whitelist wrapper around SymPy checks used by the orchestrator."""

    verifier_name = "safe_sympy"

    def verify(self, problem_text: str, answer: str, result: Dict[str, Any]) -> List[VerificationEvidence]:
        if not _sympy_available():
            return [
                VerificationEvidence(
                    verifier=self.verifier_name,
                    claim_id="sympy_available",
                    status=EvidenceStatus.INCONCLUSIVE.value,
                    method="dependency_check",
                    details="SymPy is not available.",
                    verification_level=VerificationLevel.EXACT_SYMBOLIC.value,
                    is_decisive=False,
                )
            ]

        problem_text = str(problem_text or "")
        answer = str(answer or "").strip()
        inferred_checks = self._build_checks(problem_text, answer)
        requested_checks = _extract_structured_tool_checks(result)
        if not inferred_checks and not requested_checks:
            return [
                VerificationEvidence(
                    verifier=self.verifier_name,
                    claim_id="no_supported_check",
                    status=EvidenceStatus.INCONCLUSIVE.value,
                    method="heuristic_dispatch",
                    details="No safe symbolic or numeric check matched this problem and answer.",
                    verification_level=VerificationLevel.EXACT_SYMBOLIC.value,
                    is_decisive=False,
                )
            ]

        evidence: List[VerificationEvidence] = []
        for index, check in enumerate(inferred_checks, start=1):
            evidence.append(self.run_check(check, claim_id=f"check_{index}"))
        for index, check in enumerate(requested_checks, start=1):
            requested_evidence = self.run_check(check, claim_id=f"requested_check_{index}")
            requested_evidence.is_decisive = False
            requested_evidence.details = f"{requested_evidence.details.rstrip()} {_CANDIDATE_CHECK_NOTICE}"
            evidence.append(requested_evidence)
        return evidence

    def run_check(self, spec: Dict[str, Any], claim_id: str = "tool_check") -> VerificationEvidence:
        tool_name = str(spec.get("tool") or "")
        arguments = spec.get("arguments") if isinstance(spec.get("arguments"), dict) else {}
        if tool_name not in {
            "symbolic_equivalence",
            "equation_solution",
            "equation_solution_set",
            "numeric_arithmetic",
            "derivative_check",
            "integral_check",
        }:
            return VerificationEvidence(
                verifier=self.verifier_name,
                claim_id=claim_id,
                status=EvidenceStatus.INCONCLUSIVE.value,
                method=tool_name or "unknown_tool",
                details="Unsupported tool requested.",
                verification_level=VerificationLevel.EXACT_SYMBOLIC.value,
                is_decisive=False,
            )

        try:
            return _run_with_timeout(lambda: self._run_check_now(tool_name, arguments, claim_id))
        except TimeoutError:
            return VerificationEvidence(
                verifier=self.verifier_name,
                claim_id=claim_id,
                status=EvidenceStatus.INCONCLUSIVE.value,
                method=tool_name,
                details="SymPy check timed out within the configured budget.",
                verification_level=VerificationLevel.EXACT_SYMBOLIC.value,
                is_decisive=False,
            )
        except Exception as exc:
            return VerificationEvidence(
                verifier=self.verifier_name,
                claim_id=claim_id,
                status=EvidenceStatus.INCONCLUSIVE.value,
                method=tool_name,
                details=f"{type(exc).__name__}: {str(exc)[:220]}",
                verification_level=VerificationLevel.EXACT_SYMBOLIC.value,
                is_decisive=False,
            )

    def _run_check_now(self, tool_name: str, arguments: Dict[str, Any], claim_id: str) -> VerificationEvidence:
        if tool_name == "symbolic_equivalence":
            return self._symbolic_equivalence(arguments, claim_id)
        if tool_name == "equation_solution":
            return self._equation_solution(arguments, claim_id)
        if tool_name == "equation_solution_set":
            return self._equation_solution_set(arguments, claim_id)
        if tool_name == "numeric_arithmetic":
            return self._numeric_arithmetic(arguments, claim_id)
        if tool_name == "derivative_check":
            return self._derivative_check(arguments, claim_id)
        if tool_name == "integral_check":
            return self._integral_check(arguments, claim_id)
        raise ValueError(f"unsupported tool {tool_name}")

    def _build_checks(self, problem_text: str, answer: str) -> List[Dict[str, Any]]:
        inferred: List[Dict[str, Any]] = []
        equation = _extract_first_equation(problem_text)
        answer_values = _extract_answer_values(answer)
        variable = _infer_variable(problem_text, answer)
        if equation and answer_values:
            for value in answer_values[:4]:
                inferred.append(
                    {
                        "tool": "equation_solution",
                        "arguments": {
                            "equation": equation,
                            "variable": variable,
                            "value": value,
                        },
                    }
                )
            if re.search(r"\b(?:solve|solutions?|roots?)\b", problem_text, flags=re.IGNORECASE):
                inferred.append(
                    {
                        "tool": "equation_solution_set",
                        "arguments": {"equation": equation, "variable": variable, "answer": answer},
                    }
                )

        arithmetic = _extract_simple_arithmetic(problem_text)
        numeric_answer = _extract_single_number(answer)
        if arithmetic and numeric_answer is not None:
            inferred.append(
                {
                    "tool": "numeric_arithmetic",
                    "arguments": {
                        "expression": arithmetic,
                        "expected": numeric_answer,
                    },
                }
            )

        derivative = _extract_derivative_claim(problem_text, answer)
        if derivative:
            inferred.append({"tool": "derivative_check", "arguments": derivative})

        integral = _extract_integral_claim(problem_text, answer)
        if integral:
            inferred.append({"tool": "integral_check", "arguments": integral})

        return inferred[:4]

    def _symbolic_equivalence(self, arguments: Dict[str, Any], claim_id: str) -> VerificationEvidence:
        left = _parse_expr(arguments.get("left"))
        right = _parse_expr(arguments.get("right"))
        residual = _safe_simplify(left - right)
        status = EvidenceStatus.PASS.value if residual == 0 else EvidenceStatus.FAIL.value
        return VerificationEvidence(
            verifier=self.verifier_name,
            claim_id=claim_id,
            status=status,
            method="symbolic_equivalence",
            details="Compared simplified difference of both expressions.",
            residual=str(residual),
            verification_level=VerificationLevel.EXACT_SYMBOLIC.value,
            is_decisive=True,
            claim_scope="subclaim",
        )

    def _equation_solution(self, arguments: Dict[str, Any], claim_id: str) -> VerificationEvidence:
        equation = str(arguments.get("equation") or "")
        variable_name = str(arguments.get("variable") or "x")
        value = _parse_expr(arguments.get("value"))
        left_text, right_text = _split_equation(equation)
        symbol = _symbol(variable_name)
        residual = _safe_simplify((_parse_expr(left_text) - _parse_expr(right_text)).subs(symbol, value))
        status = EvidenceStatus.PASS.value if residual == 0 else EvidenceStatus.FAIL.value
        return VerificationEvidence(
            verifier=self.verifier_name,
            claim_id=claim_id,
            status=status,
            method="equation_solution",
            details=f"Substituted {variable_name}={value} into the candidate equation.",
            residual=str(residual),
            verification_level=VerificationLevel.EXACT_SYMBOLIC.value,
            is_decisive=True,
        )

    def _numeric_arithmetic(self, arguments: Dict[str, Any], claim_id: str) -> VerificationEvidence:
        expression = _parse_expr(arguments.get("expression"))
        expected = _parse_expr(arguments.get("expected"))
        residual = _safe_simplify(expression - expected)
        status = EvidenceStatus.PASS.value if residual == 0 else EvidenceStatus.FAIL.value
        return VerificationEvidence(
            verifier=self.verifier_name,
            claim_id=claim_id,
            status=status,
            method="numeric_arithmetic",
            details="Recomputed the arithmetic expression exactly.",
            residual=str(residual),
            verification_level=VerificationLevel.EXACT_SYMBOLIC.value,
            is_decisive=True,
            claim_scope="full_answer",
        )

    def _equation_solution_set(self, arguments: Dict[str, Any], claim_id: str) -> VerificationEvidence:
        import sympy as sp

        equation = str(arguments.get("equation") or "")
        variable_name = str(arguments.get("variable") or "x")
        answer = str(arguments.get("answer") or "")
        left_text, right_text = _split_equation(equation)
        symbol = _symbol(variable_name)
        expression = _parse_expr(left_text) - _parse_expr(right_text)
        if expression.has(sp.Derivative) or len(str(expression)) > 240:
            raise ValueError("equation is outside conservative solution-set scope")
        expected = sp.solveset(expression, symbol, domain=sp.S.Complexes)
        if not isinstance(expected, sp.FiniteSet):
            return VerificationEvidence(
                verifier=self.verifier_name, claim_id=claim_id,
                status=EvidenceStatus.INCONCLUSIVE.value, method="equation_solution_set",
                details="Safe solveset did not return a finite solution set.",
                verification_level=VerificationLevel.EXACT_SYMBOLIC.value, is_decisive=False,
            )
        candidate_values = []
        for value in re.findall(r"(?:[A-Za-z]\s*=\s*)?([+\-]?\d+(?:/\d+)?(?:\.\d+)?)", answer):
            candidate_values.append(_parse_expr(value))
        if not candidate_values:
            return VerificationEvidence(
                verifier=self.verifier_name, claim_id=claim_id,
                status=EvidenceStatus.INCONCLUSIVE.value, method="equation_solution_set",
                details="Could not safely extract candidate roots.",
                verification_level=VerificationLevel.EXACT_SYMBOLIC.value, is_decisive=False,
            )
        candidate_set = sp.FiniteSet(*candidate_values)
        residual = sp.simplify(candidate_set.symmetric_difference(expected))
        status = EvidenceStatus.PASS.value if candidate_set == expected else EvidenceStatus.FAIL.value
        return VerificationEvidence(
            verifier=self.verifier_name, claim_id=claim_id, status=status,
            method="equation_solution_set", details="Compared candidate roots with SymPy solveset.",
            residual=str(residual), verification_level=VerificationLevel.EXACT_SYMBOLIC.value,
            is_decisive=True, claim_scope="full_answer" if status == EvidenceStatus.PASS.value else "subclaim",
        )

    def _derivative_check(self, arguments: Dict[str, Any], claim_id: str) -> VerificationEvidence:
        import sympy as sp

        variable_name = str(arguments.get("variable") or "x")
        symbol = _symbol(variable_name)
        function_expr = _parse_expr(arguments.get("function"))
        derivative_expr = _parse_expr(arguments.get("derivative"))
        residual = _safe_simplify(sp.diff(function_expr, symbol) - derivative_expr)
        status = EvidenceStatus.PASS.value if residual == 0 else EvidenceStatus.FAIL.value
        return VerificationEvidence(
            verifier=self.verifier_name,
            claim_id=claim_id,
            status=status,
            method="derivative_check",
            details=f"Differentiated with respect to {variable_name} and compared the result.",
            residual=str(residual),
            verification_level=VerificationLevel.EXACT_SYMBOLIC.value,
            is_decisive=True,
        )

    def _integral_check(self, arguments: Dict[str, Any], claim_id: str) -> VerificationEvidence:
        import sympy as sp

        variable_name = str(arguments.get("variable") or "x")
        symbol = _symbol(variable_name)
        integrand = _parse_expr(arguments.get("integrand"))
        antiderivative = _parse_expr(arguments.get("antiderivative"))
        residual = _safe_simplify(sp.diff(antiderivative, symbol) - integrand)
        status = EvidenceStatus.PASS.value if residual == 0 else EvidenceStatus.FAIL.value
        return VerificationEvidence(
            verifier=self.verifier_name,
            claim_id=claim_id,
            status=status,
            method="integral_check",
            details=f"Differentiated the proposed antiderivative with respect to {variable_name}.",
            residual=str(residual),
            verification_level=VerificationLevel.EXACT_SYMBOLIC.value,
            is_decisive=True,
        )


def run_sympy_verification(problem_text: str, answer: str, result: Dict[str, Any]) -> List[VerificationEvidence]:
    return SafeSympyTool().verify(problem_text=problem_text, answer=answer, result=result)


def _run_with_timeout(callback):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(callback)
        return future.result(timeout=MAX_TOOL_TIMEOUT_SECONDS)


def _sympy_available() -> bool:
    try:
        import sympy  # noqa: F401

        return True
    except Exception:
        return False


def _parse_expr(value: Any):
    import sympy as sp
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    text = _clean_math_text(str(value or ""))
    _validate_expr_text(text)
    local_dict = {name: sp.Symbol(name) for name in "abcdefghijklmnopqrstuvwxyz"}
    local_dict.update(
        {
            "pi": sp.pi,
            "E": sp.E,
            "e": sp.E,
            "I": sp.I,
            "sqrt": sp.sqrt,
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "exp": sp.exp,
            "log": sp.log,
            "ln": sp.log,
            "Abs": sp.Abs,
        }
    )
    transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
    safe_globals = dict(sp.__dict__)
    safe_globals["__builtins__"] = {}
    return parse_expr(
        text,
        local_dict=local_dict,
        global_dict=safe_globals,
        transformations=transformations,
        evaluate=True,
    )


def _safe_simplify(expr: Any) -> Any:
    import sympy as sp

    simplified = sp.simplify(expr)
    if simplified == 0:
        return 0
    numeric = sp.N(simplified, 40)
    if abs(complex(numeric)) < 1e-30:
        return 0
    return simplified


def _symbol(name: str):
    import sympy as sp

    cleaned = re.sub(r"[^A-Za-z]", "", str(name or "x"))[:1] or "x"
    return sp.Symbol(cleaned)


def _validate_expr_text(text: str) -> None:
    if not text or len(text) > MAX_EXPR_LENGTH:
        raise ValueError("expression is empty or too long")
    lowered = text.lower()
    if any(token in lowered for token in _DANGEROUS_TOKENS):
        raise ValueError("expression contains a disallowed token")
    if not re.fullmatch(r"[A-Za-z0-9_+\-*/^().,= <>{}\[\]\\|:]+", text):
        raise ValueError("expression contains unsupported characters")


def _clean_math_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.replace("\\pi", "pi").replace("\\cdot", "*").replace("\\times", "*")
    cleaned = cleaned.replace("{", "(").replace("}", ")")
    cleaned = cleaned.replace("\u2212", "-").replace("\u00f7", "/")
    cleaned = re.sub(r"^\$|\$$", "", cleaned)
    cleaned = re.sub(r"\\left|\\right", "", cleaned)
    return cleaned.strip()


def _extract_structured_tool_checks(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    checks = result.get("requested_checks") if isinstance(result, dict) else None
    if not isinstance(checks, list):
        return []
    safe_checks: List[Dict[str, Any]] = []
    for item in checks:
        if isinstance(item, dict) and isinstance(item.get("tool"), str) and isinstance(item.get("arguments"), dict):
            safe_checks.append({"tool": item["tool"], "arguments": dict(item["arguments"])})
    return safe_checks[:3]


def _extract_first_equation(text: str) -> Optional[str]:
    # Match mathematical bodies only; do not absorb natural-language prefixes
    # ("Solve", "Find x if") or suffixes ("for x").
    pattern = re.compile(
        r"(?<![A-Za-z0-9_])"
        r"([A-Za-z0-9_().]+(?:\s*[+\-*/^]\s*[A-Za-z0-9_().]+)*)"
        r"\s*=\s*"
        r"([A-Za-z0-9_().]+)"
    )
    for match in pattern.finditer(str(text or "")):
        left, right = match.group(1).strip(), match.group(2).strip().rstrip(".,;:!?\u3002")
        if len(left) >= 1 and len(right) >= 1 and (any(ch.isalpha() for ch in left) or any(ch.isalpha() for ch in right)):
            return f"{left}={right}"
    return None


def _split_equation(equation: str) -> tuple[str, str]:
    if "=" not in equation:
        raise ValueError("equation must contain '='")
    left, right = equation.split("=", 1)
    return left.strip(), right.strip()


def _extract_answer_values(answer: str) -> List[str]:
    text = str(answer or "")
    values: List[str] = []
    for match in re.finditer(r"[A-Za-z]\s*=\s*([+\-]?\d+(?:/\d+)?(?:\.\d+)?)", text):
        values.append(match.group(1))
    if values:
        return values
    if re.fullmatch(r"\s*[+\-]?\d+(?:/\d+)?(?:\.\d+)?\s*", text):
        return [text.strip()]
    return re.findall(r"[+\-]?\d+(?:/\d+)?(?:\.\d+)?", text)[:4]


def _infer_variable(problem_text: str, answer: str) -> str:
    match = re.search(r"([A-Za-z])\s*=", answer)
    if match:
        return match.group(1)
    match = re.search(r"\bsolve\s+for\s+([A-Za-z])\b", problem_text, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    variables = re.findall(r"\b([a-z])\b", problem_text)
    return variables[0] if variables else "x"


def _extract_simple_arithmetic(text: str) -> Optional[str]:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^(?:compute|calculate|evaluate|\u8ba1\u7b97|\u6c42\u503c)\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"=\s*\?\s*$", "", cleaned)
    cleaned = cleaned.rstrip(" .;:!?\u3002")
    if "?" in cleaned or re.search(r"[A-Za-z_]", cleaned):
        return None
    if not re.fullmatch(r"[0-9()+\-*/^.\s]+", cleaned) or not re.search(r"[+\-*/^]", cleaned):
        return None
    return cleaned


def _extract_single_number(text: str) -> Optional[str]:
    matches = re.findall(r"[+\-]?\d+(?:/\d+)?(?:\.\d+)?", str(text or ""))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        return matches[-1]
    try:
        return str(Fraction(str(text).strip()))
    except Exception:
        return None


def _extract_derivative_claim(problem_text: str, answer: str) -> Optional[Dict[str, Any]]:
    combined = f"{problem_text}\n{answer}"
    match = re.search(
        r"(?:derivative|differentiate)\s+([A-Za-z0-9+\-*/^().\s]+?)\s+(?:is|=)\s+([A-Za-z0-9+\-*/^().\s]+)",
        combined,
        flags=re.IGNORECASE,
    )
    if match:
        return {"function": match.group(1), "derivative": match.group(2), "variable": "x"}
    return None


def _extract_integral_claim(problem_text: str, answer: str) -> Optional[Dict[str, Any]]:
    combined = f"{problem_text}\n{answer}"
    match = re.search(
        r"(?:integral|antiderivative)\s+of\s+([A-Za-z0-9+\-*/^().\s]+?)\s+(?:is|=)\s+([A-Za-z0-9+\-*/^().\s]+)",
        combined,
        flags=re.IGNORECASE,
    )
    if match:
        return {"integrand": match.group(1), "antiderivative": match.group(2), "variable": "x"}
    return None

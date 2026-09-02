"""Deterministic, benchmark-only grading helpers.

This module is intentionally separate from production verification and never
participates in Agent acceptance decisions.
"""
from __future__ import annotations

import re
from typing import Any

from .answer_equivalence import answers_equivalent
from math_agent_core.search.candidate_compare import compare_candidate_answers


def _result(correct: bool | None, **extra: Any) -> dict[str, Any]:
    status = "UNRESOLVED" if correct is None else "CORRECT" if correct else "INCORRECT"
    return {"correct": correct, "status": status, **extra}


def grade_primary_answer(response: str, grading_spec: Any) -> dict[str, Any]:
    expected, kind, aliases = _primary_spec(grading_spec)
    if not expected:
        return _result(None, reason="missing_primary_answer")
    text = str(response or "").strip()
    if not text:
        return _result(None, reason="empty_response", expected=expected, primary_type=kind)
    statements = _answer_statements(text)
    if kind == "text_alias":
        matched = _match_text_alias(statements, expected, aliases)
        return _result(matched, expected=expected, primary_type=kind)
    if _negative_final_statement(statements, expected, kind):
        return _result(False, expected=expected, primary_type=kind)
    candidates = list(reversed(statements))
    candidates.append(text)
    for candidate in candidates:
        if kind == "numeric":
            fragment = _explicit_value_fragment(candidate)
            if fragment and _numeric_equal(fragment, expected):
                return _result(True, expected=expected, primary_type=kind)
        elif kind in {"symbolic", "solution_set"}:
            if _symbolic_assignment_match(candidate, expected) or answers_equivalent(candidate, expected) or _symbolic_expression_equivalent(candidate, expected):
                return _result(True, expected=expected, primary_type=kind)
    return _result(False, expected=expected, primary_type=kind)


def grade_required_claims(response: str, grading_spec: Any) -> dict[str, Any]:
    claims = grading_spec.get("required_claims", []) if isinstance(grading_spec, dict) else []
    if not isinstance(claims, list):
        claims = [claims]
    results = []
    for claim in claims:
        results.append(_grade_claim(response, str(claim)))
    gradable = [item for item in results if item["correct"] is not None]
    correct = None if any(item["correct"] is None for item in results) else all(item["correct"] for item in results)
    return {
        "correct": correct,
        "status": "UNRESOLVED" if correct is None else "CORRECT" if correct else "INCORRECT",
        "claims": results,
        "gradable_count": len(gradable),
    }


def grade_full_problem(response: str, grading_spec: Any) -> dict[str, Any]:
    primary = grade_primary_answer(response, grading_spec)
    claims = grade_required_claims(response, grading_spec)
    if primary["correct"] is None or claims["correct"] is None:
        correct = None
    else:
        correct = bool(primary["correct"] and claims["correct"])
    return {
        "correct": correct,
        "status": "UNRESOLVED" if correct is None else "CORRECT" if correct else "INCORRECT",
        "primary": primary,
        "required_claims": claims,
    }


def _primary_spec(spec: Any) -> tuple[str, str, list[str]]:
    if isinstance(spec, dict):
        expected = spec.get("primary", spec.get("answer", ""))
        kind = str(spec.get("primary_type") or "").lower()
        aliases = spec.get("aliases", [])
        aliases = aliases if isinstance(aliases, list) else [aliases]
    else:
        expected, kind, aliases = spec, "", []
    expected = str(expected or "").strip()
    if not kind:
        kind = "numeric" if re.fullmatch(r"[+\-]?\d+(?:\.\d+)?", expected) else "symbolic"
    return expected, kind, [str(item) for item in aliases if str(item).strip()]


def _answer_statements(text: str) -> list[str]:
    chunks = [chunk.strip() for chunk in re.split(r"(?<=[.!?;])\s+|\n+", text) if chunk.strip()]
    return chunks or [text.strip()]


def _negative_final_statement(statements: list[str], expected: str, kind: str) -> bool:
    expected_norm = re.sub(r"\s+", "", expected.lower())
    for statement in reversed(statements):
        lowered = statement.lower()
        if expected_norm and expected_norm in re.sub(r"\s+", "", lowered):
            if re.search(r"\b(?:not|incorrect|impossible|false|does\s+not\s+equal|is\s+not)\b|[≠]", lowered):
                return True
            return False
    return False


def _explicit_value_fragment(statement: str) -> str:
    assignments = list(re.finditer(r"\b[A-Za-z][A-Za-z0-9_]*\s*=\s*([^.;,]+)", statement))
    if assignments:
        return assignments[-1].group(1).strip()
    matches = re.findall(r"(?:final\s+answer|answer|conclusion|therefore|thus|hence)\s*(?:is|=|:)?\s*([+\-]?\d+(?:\.\d+)?)", statement, flags=re.IGNORECASE)
    return matches[-1] if matches else (statement.strip() if re.fullmatch(r"[+\-]?\d+(?:\.\d+)?", statement.strip()) else "")


def _numeric_equal(left: str, right: str) -> bool:
    try:
        return answers_equivalent(left, right)
    except Exception:
        return False


def _symbolic_assignment_match(response: str, expected: str) -> bool:
    expected_rhs = _rhs(expected)
    if expected_rhs is None:
        return False
    for match in re.finditer(r"=\s*([^.;,]+)", response):
        candidate = match.group(1).strip()
        normalized_candidate = candidate.replace("lambda", "lam")
        normalized_expected = expected_rhs.replace("lambda", "lam")
        if answers_equivalent(normalized_candidate, normalized_expected) or compare_candidate_answers(normalized_candidate, normalized_expected)["agreement"]:
            return True
    return False


def _rhs(value: str) -> str | None:
    match = re.search(r"=\s*(.+)$", str(value or "").strip())
    return match.group(1).strip().rstrip(".") if match else str(value or "").strip()


def _symbolic_expression_equivalent(left: str, right: str) -> bool:
    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import convert_xor, implicit_multiplication_application, parse_expr, standard_transformations
        transforms = standard_transformations + (implicit_multiplication_application, convert_xor)
        local = {name: sp.Symbol(name) for name in "abcdefghijklmnopqrstuvwxyz"}
        local.update({"sin": sp.sin, "exp": sp.exp, "e": sp.E, "pi": sp.pi})
        glob = dict(sp.__dict__)
        glob["__builtins__"] = {}
        l = parse_expr(_rhs(left) or left, local_dict=local, global_dict=glob, transformations=transforms)
        r = parse_expr(_rhs(right) or right, local_dict=local, global_dict=glob, transformations=transforms)
        return sp.simplify(l - r) == 0
    except Exception:
        return False


def _match_text_alias(statements: list[str], expected: str, aliases: list[str]) -> bool | None:
    groups = {
        "sample_mean": ("sample mean", "sample average", "arithmetic mean", "样本均值"),
        "uniform_convergence": ("uniformly convergent", "converges uniformly", "uniform convergence", "一致收敛"),
        "cauchy_schwarz": ("cauchy-schwarz", "cauchy schwarz"),
    }
    accepted = tuple(aliases) or groups.get(expected.lower(), ())
    if not accepted:
        return None
    for statement in reversed(statements):
        if any(alias.lower() in statement.lower() for alias in accepted):
            return not bool(re.search(r"\b(?:not|incorrect|false|does\s+not|cannot)\b", statement.lower()))
    return False


def _grade_claim(response: str, claim: str) -> dict[str, Any]:
    text = str(response or "").lower()
    compact = re.sub(r"[^a-z0-9]+", "", text)
    key = re.sub(r"[^a-z0-9]+", "", claim.lower())
    specs = {
        "unbiased": (("unbiased", "is unbiased", "无偏"), ("not unbiased", "is biased", "biased estimator", "有偏")),
        "quadraticconvergence": (("quadratic convergence", "converges quadratically", "quadratic rate"), ("not quadratically", "not quadratic")),
        "dctnotapplicable": (("dct does not apply", "dominated convergence does not apply", "dct not applicable"), ("dct applies", "dominated convergence applies")),
        "weierstrassmtest": (("weierstrass m-test", "m-test"), ("m-test does not apply",)),
        "equalityifflineardependence": (("equality iff linearly dependent", "linearly dependent", "scalar multiple", "proportional", "线性相关"), ("not linearly dependent",)),
    }
    spec = specs.get(key)
    if spec is None:
        return _result(None, claim=claim)
    positives, negatives = spec
    if any(_alias_present(text, compact, alias) for alias in negatives):
        return _result(False, claim=claim)
    if any(_alias_present(text, compact, alias) for alias in positives):
        return _result(True, claim=claim)
    return _result(False, claim=claim)


def _alias_present(text: str, compact: str, alias: str) -> bool:
    value = str(alias).lower()
    if value.isascii():
        normalized = re.sub(r"[^a-z0-9]+", "", value)
        return normalized in compact
    return value in text

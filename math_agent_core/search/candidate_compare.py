from __future__ import annotations

import re
from fractions import Fraction
from typing import Any, Dict

from math_agent_core.evaluation.answer_equivalence import normalize_answer_for_comparison
from math_agent_core.answer_utils import normalize_final_response


def compare_candidate_answers(candidate_a: Any, candidate_b: Any) -> Dict[str, Any]:
    """Compare candidate final answers without consulting an LLM."""
    left = _extract_answer(candidate_a)
    right = _extract_answer(candidate_b)
    left = normalize_final_response(left)
    right = normalize_final_response(right)
    left_norm = normalize_answer_for_comparison(left)
    right_norm = normalize_answer_for_comparison(right)

    if left and right and left == right:
        return {"agreement": True, "agreement_type": "normalized_exact", "confidence": 1.0}

    if _numeric_equivalent(left, right):
        return {"agreement": True, "agreement_type": "numeric", "confidence": 0.99}

    symbolic = _symbolic_equivalent(left, right)
    if symbolic is True:
        return {"agreement": True, "agreement_type": "symbolic", "confidence": 0.97}

    if left_norm and left_norm == right_norm:
        return {"agreement": True, "agreement_type": "normalized_exact", "confidence": 0.98}

    return {"agreement": False, "agreement_type": "conflict", "confidence": 0.95}


def _extract_answer(candidate: Any) -> str:
    if isinstance(candidate, dict):
        final_answer = candidate.get("final_answer")
        if isinstance(final_answer, dict):
            return str(final_answer.get("answer") or "").strip()
        for key in ("final_response", "answer"):
            value = candidate.get(key)
            if isinstance(value, str):
                return value.strip()
    return str(candidate or "").strip()


def _numeric_equivalent(left: str, right: str) -> bool:
    try:
        return Fraction(left.strip()) == Fraction(right.strip())
    except Exception:
        pass
    try:
        left_value = float(left.strip())
        right_value = float(right.strip())
    except Exception:
        return False
    scale = max(1.0, abs(left_value), abs(right_value))
    return abs(left_value - right_value) <= 1e-12 * scale


def _symbolic_equivalent(left: str, right: str) -> bool | None:
    if not _safe_expression(left) or not _safe_expression(right):
        return None
    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import (
            convert_xor,
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )

        transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
        local_dict = {name: sp.Symbol(name) for name in "abcdefghijklmnopqrstuvwxyz"}
        safe_globals = dict(sp.__dict__)
        safe_globals["__builtins__"] = {}
        left_expr = parse_expr(left.replace("^", "**"), local_dict=local_dict, global_dict=safe_globals, transformations=transformations)
        right_expr = parse_expr(right.replace("^", "**"), local_dict=local_dict, global_dict=safe_globals, transformations=transformations)
        return sp.simplify(left_expr - right_expr) == 0
    except Exception:
        return None


def _safe_expression(value: str) -> bool:
    text = str(value or "").strip()
    if not text or len(text) > 240:
        return False
    lowered = text.lower()
    if any(token in lowered for token in ("__", "import", "exec", "eval", "lambda", "open(")):
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9_+\-*/^()., ]+", text))


compare_candidates = compare_candidate_answers

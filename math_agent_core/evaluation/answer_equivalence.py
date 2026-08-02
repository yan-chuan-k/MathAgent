from __future__ import annotations

import re
from fractions import Fraction
from typing import Optional


def normalize_answer_for_comparison(answer: str) -> str:
    text = str(answer or "").strip()
    text = re.sub(r"\s+", "", text)
    text = text.strip("`$.,;:")
    choice = re.fullmatch(r"(?i)(?:option)?([A-D])", text)
    if choice:
        return choice.group(1).upper()
    boolean = text.lower()
    if boolean in {"true", "yes", "correct"}:
        return "true"
    if boolean in {"false", "no", "incorrect"}:
        return "false"
    number = _normalize_number(text)
    if number is not None:
        return number
    expression = _normalize_expression(text)
    return expression or text.lower()


def answers_equivalent(left: str, right: str) -> bool:
    left_norm = normalize_answer_for_comparison(left)
    right_norm = normalize_answer_for_comparison(right)
    if left_norm == right_norm:
        return True
    return _expressions_equivalent(left_norm, right_norm)


def answer_cluster_key(answer: str) -> str:
    return normalize_answer_for_comparison(answer)


def _normalize_number(text: str) -> Optional[str]:
    try:
        return str(Fraction(text))
    except Exception:
        pass
    try:
        value = float(text)
    except Exception:
        return None
    return format(value, ".12g")


def _normalize_expression(text: str) -> Optional[str]:
    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import (
            convert_xor,
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )

        if len(text) > 200 or any(token in text.lower() for token in ("__", "import", "exec", "eval", "lambda")):
            return None
        transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
        safe_globals = dict(sp.__dict__)
        safe_globals["__builtins__"] = {}
        parsed = parse_expr(text.replace("^", "**"), global_dict=safe_globals, transformations=transformations)
        return str(sp.simplify(parsed))
    except Exception:
        return None


def _expressions_equivalent(left: str, right: str) -> bool:
    try:
        import sympy as sp

        left_expr = sp.sympify(left)
        right_expr = sp.sympify(right)
        return sp.simplify(left_expr - right_expr) == 0
    except Exception:
        return False

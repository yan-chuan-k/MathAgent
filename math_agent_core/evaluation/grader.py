"""Deterministic, benchmark-only grading helpers.

This module is intentionally separate from production verification and never
participates in Agent acceptance decisions.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

from .answer_equivalence import answers_equivalent
from math_agent_core.search.candidate_compare import compare_candidate_answers


_SUPPORTED_PRIMARY_TYPES = {"numeric", "symbolic", "text_alias", "solution_set", "structured"}


_CORRECTION_MARKER_RE = re.compile(
    r"\b(?:actually|instead|rather|however|correction|in\s+fact|but)\b",
    flags=re.IGNORECASE,
)
_REJECTION_RE = re.compile(
    r"\b(?:but|however|actually|instead|rather|correction|in\s+fact)\b"
    r"[^.!?;\n]{0,100}\b(?:that|this)(?:\s+(?:conclusion|claim|statement))?\s+"
    r"(?:is|was)\s+(?:false|wrong|incorrect)\b",
    flags=re.IGNORECASE,
)


def _balanced_brace_content(text: str, start: int) -> tuple[str, int] | None:
    """Return the contents/end of a balanced {...} group starting at ``start``."""
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:index], index + 1
    return None


def _latex_atom(text: str, start: int) -> tuple[str, int] | None:
    """Consume one tiny supported TeX atom for benchmark-only macro rewriting."""
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text):
        return None
    if text[start] == "{":
        return _balanced_brace_content(text, start)
    if text[start] == "\\":
        command = re.match(r"\\[A-Za-z]+", text[start:])
        if command:
            token = command.group(0)
            return token, start + len(token)
        if start + 1 < len(text):
            return text[start:start + 2], start + 2
    # TeX shorthand such as \frac12 consumes one token for each side.
    return text[start], start + 1


def _format_fraction_part(value: str) -> str:
    token = value.strip()
    if re.fullmatch(r"[+\-]?(?:[A-Za-z0-9_.]+|\\[A-Za-z]+)", token):
        return token
    return f"({token})"


def _rewrite_supported_latex_macros(text: str) -> str:
    """Rewrite only the small macro vocabulary used by the frozen benchmark."""
    value = text
    # A fixed bound keeps malformed/pathological model text from causing an
    # unbounded normalization loop while still allowing nested supported forms.
    for _ in range(8):
        previous = value

        # Balanced wrappers: \boxed{\frac{1}{2}} -> \frac{1}{2}.
        macro = r"\boxed"
        cursor = 0
        replacements = 0
        while replacements < 32:
            pos = value.find(macro, cursor)
            if pos < 0:
                break
            atom = _latex_atom(value, pos + len(macro))
            if atom is None:
                cursor = pos + len(macro)
                continue
            inner, end = atom
            value = value[:pos] + inner + value[end:]
            cursor = pos + len(inner)
            replacements += 1

        # Estimator notation used by hard_linear_regression.
        cursor = 0
        replacements = 0
        while replacements < 32:
            pos = value.find(r"\hat", cursor)
            if pos < 0:
                break
            atom = _latex_atom(value, pos + len(r"\hat"))
            if atom is None:
                cursor = pos + len(r"\hat")
                continue
            inner, end = atom
            if inner.strip() not in {r"\beta", "beta", "β"}:
                cursor = end
                continue
            value = value[:pos] + "beta_hat" + value[end:]
            cursor = pos + len("beta_hat")
            replacements += 1

        # Harmless operator typography used by the variance/covariance claim.
        cursor = 0
        replacements = 0
        macro = r"\operatorname"
        while replacements < 32:
            pos = value.find(macro, cursor)
            if pos < 0:
                break
            atom = _latex_atom(value, pos + len(macro))
            if atom is None:
                cursor = pos + len(macro)
                continue
            inner, end = atom
            if inner.strip() not in {"Var", "Cov"}:
                cursor = end
                continue
            replacement = inner.strip()
            value = value[:pos] + replacement + value[end:]
            cursor = pos + len(replacement)
            replacements += 1

        # Fractions support both braced form and the common single-token
        # shorthand: \frac{1}{2}, \frac{1}{\lambda+\mu}, and \frac12.
        cursor = 0
        replacements = 0
        macro = r"\frac"
        while replacements < 64:
            pos = value.find(macro, cursor)
            if pos < 0:
                break
            numerator = _latex_atom(value, pos + len(macro))
            if numerator is None:
                cursor = pos + len(macro)
                continue
            num, after_num = numerator
            denominator = _latex_atom(value, after_num)
            if denominator is None:
                cursor = after_num
                continue
            den, end = denominator
            replacement = f"{_format_fraction_part(num)}/{_format_fraction_part(den)}"
            value = value[:pos] + replacement + value[end:]
            cursor = pos + len(replacement)
            replacements += 1

        if value == previous:
            break
    return value


def _normalize_benchmark_text(value: str) -> str:
    """Bounded deterministic LaTeX surface normalization for benchmark grading only."""
    text = str(value or "")
    text = text.replace(r"\left", "").replace(r"\right", "")
    text = text.replace(r"\(", "").replace(r"\)", "")
    text = text.replace(r"\[", "").replace(r"\]", "")
    text = text.replace("$", "")

    text = _rewrite_supported_latex_macros(text)

    commands = {
        r"\lambda": "lam",
        r"\mu": "mu",
        r"\sigma": "sigma",
        r"\pi": "pi",
        r"\beta": "beta",
        r"\sin": "sin",
        r"\cos": "cos",
        r"\exp": "exp",
        r"\min": "min",
        r"\leq": "<=",
        r"\le": "<=",
        r"\geq": ">=",
        r"\ge": ">=",
        r"\infty": "infinity",
        r"\|": "|",
        r"\qquad": " ",
        r"\quad": " ",
        r"\,": "",
        r"\;": "",
        r"\!": "",
    }
    # Longest commands first so \leq is not partially rewritten as \le.
    for command in sorted(commands, key=len, reverse=True):
        text = text.replace(command, commands[command])

    text = text.replace("λ", "lam").replace("μ", "mu").replace("σ", "sigma")
    text = text.replace("π", "pi").replace("β", "beta")

    # e^{-x} -> e^(-x); preserve subscript braces such as x_{n+1}.
    exponent_group = re.compile(r"\^\{([^{}]+)\}")
    for _ in range(4):
        updated = exponent_group.sub(r"^(\1)", text)
        if updated == text:
            break
        text = updated
    simple_group = re.compile(r"(?<!_)\{([^{}]+)\}")
    for _ in range(4):
        updated = simple_group.sub(r"(\1)", text)
        if updated == text:
            break
        text = updated

    # pi(...) is multiplication, unlike ordinary function calls such as sin(...).
    text = re.sub(r"\bpi\s*(?=\()", "pi*", text)
    text = text.replace("×", "*").replace("·", "*").replace("--", "-")
    return text

def _phrase_matches(text: str, phrase: str) -> list[re.Match[str]]:
    value = str(phrase or "").strip()
    if not value:
        return []
    if value.isascii():
        escaped = re.escape(value).replace(r"\ ", r"\s+")
        return list(re.finditer(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", text, flags=re.IGNORECASE))
    return list(re.finditer(re.escape(value), text, flags=re.IGNORECASE))


def _last_semantic_polarity(text: str, positives: Iterable[str], negatives: Iterable[str] = ()) -> bool | None:
    events: list[tuple[int, bool]] = []
    for alias in positives:
        for match in _phrase_matches(text, alias):
            polarity = not _match_is_negated(text, match.start())
            events.append((match.start(), polarity))
    for alias in negatives:
        for match in _phrase_matches(text, alias):
            polarity = False
            if _match_is_negated(text, match.start()):
                polarity = True
            events.append((match.start(), polarity))
    events.sort(key=lambda item: item[0])
    if not events:
        return None

    # An explicit anaphoric rejection reverses the most recent assertion.
    for rejection in _REJECTION_RE.finditer(text):
        prior = [item for item in events if item[0] < rejection.start()]
        if prior:
            events.append((rejection.start(), not prior[-1][1]))
            events.sort(key=lambda item: item[0])
    return events[-1][1]


def _correction_positions(text: str) -> list[int]:
    return [match.start() for match in _CORRECTION_MARKER_RE.finditer(text)]


def _has_unparsed_trailing_correction(text: str, selected_position: int, candidate_positions: Iterable[int]) -> bool:
    later_markers = [position for position in _correction_positions(text) if position > selected_position]
    if not later_markers:
        return False
    last_marker = max(later_markers)
    return not any(position > last_marker for position in candidate_positions)


def _canonical_lhs(value: str) -> str:
    text = _normalize_benchmark_text(str(value or "")).strip(" `$,;:.\n\t")
    # Strip ordinary prose prefixes from forms such as 'The solution is y'.
    if re.search(r"\s", text):
        match = re.search(r"([^\s]+(?:\([^=]*\))?)\s*$", text)
        if match:
            text = match.group(1)
    return re.sub(r"\s+", "", text).lower()


def _result(correct: bool | None, **extra: Any) -> dict[str, Any]:
    status = "UNRESOLVED" if correct is None else "CORRECT" if correct else "INCORRECT"
    return {"correct": correct, "status": status, **extra}


def grade_primary_answer(response: str, grading_spec: Any) -> dict[str, Any]:
    expected, kind, aliases = _primary_spec(grading_spec)
    text = _normalize_benchmark_text(str(response or "").strip())
    if not text:
        return _result(None, reason="empty_response", expected=expected, primary_type=kind)
    if kind == "unresolved":
        return _result(None, reason="unreliable_legacy_primary_type", expected=expected, primary_type=kind)
    if kind not in _SUPPORTED_PRIMARY_TYPES:
        return _result(None, reason="unsupported_primary_type", expected=expected, primary_type=kind)
    if kind == "structured":
        return _grade_structured_primary(text, grading_spec)
    if not expected:
        return _result(None, reason="missing_primary_answer", primary_type=kind)

    if kind == "text_alias":
        matched = _match_text_alias(text, expected, aliases)
        return _result(matched, expected=expected, primary_type=kind)
    if kind == "numeric":
        candidate, ambiguous = _last_numeric_conclusion(text)
        if ambiguous:
            return _result(None, reason="ambiguous_trailing_correction", expected=expected, primary_type=kind)
        if candidate is None:
            return _result(False, expected=expected, primary_type=kind)
        return _result(_numeric_equal(candidate, expected), expected=expected, primary_type=kind, extracted=candidate)
    if kind == "symbolic":
        candidate, ambiguous = _select_symbolic_conclusion(text, expected)
        if ambiguous:
            return _result(None, reason="ambiguous_trailing_correction", expected=expected, primary_type=kind)
        if candidate is None:
            return _result(False, expected=expected, primary_type=kind)
        return _result(_symbolic_equivalent(candidate, expected), expected=expected, primary_type=kind, extracted=candidate)
    if kind == "solution_set":
        candidates = _solution_set_fragments(text)
        if not candidates:
            return _result(False, expected=expected, primary_type=kind)
        candidate = candidates[0]
        return _result(_solution_set_equal(candidate, expected), expected=expected, primary_type=kind, extracted=candidate)
    return _result(None, reason="unsupported_primary_type", expected=expected, primary_type=kind)


def grade_required_claims(response: str, grading_spec: Any) -> dict[str, Any]:
    claims = grading_spec.get("required_claims", []) if isinstance(grading_spec, dict) else []
    if not isinstance(claims, list):
        claims = [claims]
    results = [_grade_claim(response, str(claim)) for claim in claims]
    gradable = [item for item in results if item["correct"] is not None]
    correct = None if any(item["correct"] is None for item in results) else all(item["correct"] for item in results)
    return {
        "correct": correct,
        "status": "UNRESOLVED" if correct is None else "CORRECT" if correct else "INCORRECT",
        "claims": results,
        "gradable_count": len(gradable),
        "unresolved_count": sum(item["correct"] is None for item in results),
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
        explicit_kind = spec.get("primary_type")
        kind = str(explicit_kind or "").strip().lower()
        aliases = spec.get("aliases", [])
        aliases = aliases if isinstance(aliases, list) else [aliases]
    else:
        expected, kind, aliases = spec, "", []
    expected = str(expected or "").strip()
    if not kind:
        kind = _infer_legacy_primary_type(expected)
    return expected, kind, [str(item) for item in aliases if str(item).strip()]


def _infer_legacy_primary_type(expected: str) -> str:
    value = str(expected or "").strip()
    if _looks_numeric_literal(value):
        return "numeric"
    if _looks_reliably_symbolic(value):
        return "symbolic"
    return "unresolved"


def _looks_numeric_literal(value: str) -> bool:
    return bool(re.fullmatch(r"[+\-]?(?:\d+(?:\.\d+)?|\d+\s*/\s*\d+)", value.strip()))


def _looks_reliably_symbolic(value: str) -> bool:
    text = value.strip()
    if not text or re.fullmatch(r"[A-Za-z_]+(?:\s+[A-Za-z_]+)*", text):
        return False
    if re.search(r"\b(?:if|when|otherwise|the|is|and|or|compact|mean|convergence)\b", text.lower()):
        return False
    return bool(re.search(r"[=+\-*/^()\[\]{}]", text))


def _answer_statements(text: str) -> list[str]:
    chunks = [chunk.strip() for chunk in re.split(r"(?<=[.!?;])\s+|\n+", text) if chunk.strip()]
    return chunks or [text.strip()]


def _last_numeric_conclusion(text: str) -> tuple[str | None, bool]:
    candidates: list[tuple[int, str]] = []
    for assignment in _top_level_assignments(text):
        rhs = assignment["rhs"]
        if _looks_numeric_literal(rhs):
            candidates.append((assignment["position"], rhs))

    number = r"[+\-]?(?:\d+\s*/\s*\d+|\d+(?:\.\d+)?)"
    patterns = (
        rf"\b(?:final\s+answer|answer|conclusion|limit|integral|value|result|optimum\s+value|optimal\s+value|maximum\s+value|minimum\s+value)\s*(?:is|equals|=|:)?\s*({number})",
        rf"\b(?:therefore|hence|thus)\s*(?:the\s+)?(?:answer|limit|integral|value|result)?\s*(?:is|equals|=|:)?\s*({number})",
        rf"\bequals\s*({number})",
        rf"\b(?:actually|instead|rather|in\s+fact)\s*(?:it\s+)?(?:is|equals|=|:)?\s*({number})",
        rf"\bcorrection\s*(?:is|equals|=|:)?\s*({number})",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            candidates.append((match.start(1), match.group(1).strip()))

    stripped = text.strip().strip("`$.,;: ")
    if _looks_numeric_literal(stripped):
        candidates.append((len(text), stripped))
    if not candidates:
        return None, False
    candidates.sort(key=lambda item: item[0])
    position, value = candidates[-1]
    ambiguous = _has_unparsed_trailing_correction(text, position, [item[0] for item in candidates])
    return value, ambiguous


def _numeric_equal(left: str, right: str) -> bool:
    try:
        return answers_equivalent(left, right)
    except Exception:
        return False


def _symbolic_conclusion_candidates(text: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for assignment in _top_level_assignments(text):
        rhs = assignment["rhs"]
        if rhs:
            candidates.append({
                "position": assignment["position"],
                "fragment": rhs,
                "lhs": _canonical_lhs(assignment["lhs"]),
                "source": "assignment",
            })

    conclusion_patterns = (
        r"\b(?:final\s+answer|answer|conclusion|result|solution|expectation|contour\s+integral|minimal\s+polynomial|iteration|estimate|estimator)\s+(?:is\s+given\s+by|is|equals)\s+",
        r"\b(?:is\s+given\s+by|equals)\s+",
        r"\b(?:actually|instead|rather|in\s+fact)\s+(?:it|this)?\s*(?:is\s+given\s+by|is|equals)\s+",
        r"\bcorrection\s+(?:is\s+given\s+by|is|equals|:)\s*",
    )
    for pattern in conclusion_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            fragment = _read_math_fragment(text, match.end())
            if fragment:
                candidates.append({"position": match.end(), "fragment": fragment, "lhs": None, "source": "conclusion"})

    stripped = text.strip().strip("`$ ")
    if _looks_like_standalone_math(stripped):
        candidates.append({"position": len(text), "fragment": stripped.rstrip("."), "lhs": None, "source": "standalone"})

    candidates.sort(key=lambda item: item["position"], reverse=True)
    seen: set[tuple[str | None, str]] = set()
    ordered: list[dict[str, Any]] = []
    for item in candidates:
        key = (item.get("lhs"), re.sub(r"\s+", "", item["fragment"]))
        if key not in seen:
            seen.add(key)
            ordered.append(item)
    return ordered


def _symbolic_conclusion_fragments(text: str) -> list[str]:
    return [item["fragment"] for item in _symbolic_conclusion_candidates(text)]


def _select_symbolic_conclusion(text: str, expected: str) -> tuple[str | None, bool]:
    candidates = _symbolic_conclusion_candidates(text)
    if not candidates:
        return None, False

    expected_text = _normalize_benchmark_text(expected)
    expected_assignments = _top_level_assignments(expected_text)
    if expected_assignments:
        expected_lhs = _canonical_lhs(expected_assignments[-1]["lhs"])
        matching = [item for item in candidates if item.get("lhs") == expected_lhs]
        if not matching:
            return None, False
        # Assignment-valued expected answers are LHS-aware; unrelated later assignments
        # (e.g. Var(beta_hat)=...) do not replace the primary target.
        selected = matching[0]
        return selected["fragment"], False

    selected = candidates[0]
    positions = [int(item["position"]) for item in candidates]
    ambiguous = _has_unparsed_trailing_correction(text, int(selected["position"]), positions)
    return selected["fragment"], ambiguous


def _solution_set_fragments(text: str) -> list[str]:
    candidates: list[tuple[int, str]] = []
    for assignment in _top_level_assignments(text):
        rhs = assignment["rhs"]
        if rhs:
            candidates.append((assignment["position"], rhs))
    for match in re.finditer(r"\b(?:optimizer|maximizer|minimizer|solution|optimal\s+point)\s+(?:is|equals)\s+", text, flags=re.IGNORECASE):
        fragment = _read_math_fragment(text, match.end())
        if fragment:
            assignments = _top_level_assignments(fragment)
            candidates.append((match.end(), assignments[-1]["rhs"] if assignments else fragment))
    stripped = text.strip().strip("`$.,;: ")
    if re.fullmatch(r"[\(\[\{].*[\)\]\}]", stripped):
        candidates.append((len(text), stripped))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [fragment for _, fragment in candidates]


def _top_level_assignments(text: str) -> list[dict[str, Any]]:
    value = str(text or "")
    positions: list[int] = []
    depth = 0
    pairs = {"(": ")", "[": "]", "{": "}"}
    closing = set(pairs.values())
    for index, char in enumerate(value):
        if char in pairs:
            depth += 1
            continue
        if char in closing:
            depth = max(0, depth - 1)
            continue
        if char != "=" or depth != 0:
            continue
        previous = value[index - 1] if index > 0 else ""
        following = value[index + 1] if index + 1 < len(value) else ""
        if previous in "<>!=" or following in "=>":
            continue
        positions.append(index)

    assignments: list[dict[str, Any]] = []
    for number, position in enumerate(positions):
        lhs_start = _assignment_lhs_start(value, position)
        next_position = positions[number + 1] if number + 1 < len(positions) else None
        rhs_end = _assignment_rhs_end(value, position + 1, next_position)
        lhs = value[lhs_start:position].strip(" ,;:\n\t")
        rhs = value[position + 1:rhs_end].strip(" `$,;:.\n\t")
        if lhs and rhs:
            assignments.append({"lhs": lhs, "rhs": rhs, "position": position})
    return assignments


def _assignment_lhs_start(text: str, position: int) -> int:
    start = 0
    for delimiter in (";", ".", "\n"):
        found = text.rfind(delimiter, 0, position)
        if found >= start:
            start = found + 1
    prefix = text[start:position]
    connectors = list(re.finditer(r"\b(?:and|but|then|while)\b", prefix, flags=re.IGNORECASE))
    if connectors:
        start += connectors[-1].end()
    comma = text.rfind(",", start, position)
    if comma >= start:
        start = comma + 1
    return start


def _assignment_rhs_end(text: str, start: int, next_assignment: int | None) -> int:
    hard_end = len(text)
    depth = 0
    index = start
    while index < len(text):
        char = text[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif depth == 0:
            if char in ";\n":
                hard_end = index
                break
            if char == "." and not (index > 0 and index + 1 < len(text) and text[index - 1].isdigit() and text[index + 1].isdigit()):
                hard_end = index
                break
            if char == ",":
                hard_end = index
                break
        index += 1

    if next_assignment is not None and next_assignment < hard_end:
        between = text[start:next_assignment]
        connector_matches = list(re.finditer(r"\b(?:and|but|then|while)\b", between, flags=re.IGNORECASE))
        if connector_matches:
            hard_end = start + connector_matches[-1].start()
        else:
            comma = between.rfind(",")
            hard_end = start + comma if comma >= 0 else next_assignment
    else:
        fragment = text[start:hard_end]
        connector = re.search(r"\s+(?:and|but|with|hence|therefore)\s+", fragment, flags=re.IGNORECASE)
        if connector:
            hard_end = start + connector.start()
    return hard_end


def _read_math_fragment(text: str, start: int) -> str:
    end = _assignment_rhs_end(text, start, None)
    fragment = text[start:end].strip(" `$,;:.\n\t")
    return fragment


def _looks_like_standalone_math(text: str) -> bool:
    if not text or len(text) > 240:
        return False
    if re.search(r"\b(?:not|incorrect|false|because|therefore|hence|the|is|equals)\b", text.lower()):
        return False
    return bool(re.search(r"[=+\-*/^()\[\]{}]", text))


def _symbolic_equivalent(left: str, right: str) -> bool:
    left_expr = _rhs_of_expression(left)
    right_expr = _rhs_of_expression(right)
    normalized_left = _normalize_symbolic_text(left_expr)
    normalized_right = _normalize_symbolic_text(right_expr)
    try:
        if answers_equivalent(normalized_left, normalized_right):
            return True
    except Exception:
        pass
    try:
        if compare_candidate_answers(normalized_left, normalized_right)["agreement"]:
            return True
    except Exception:
        pass
    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import (
            convert_xor,
            factorial_notation,
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )

        transformations = standard_transformations + (
            implicit_multiplication_application,
            convert_xor,
            factorial_notation,
        )
        local = {name: sp.Symbol(name) for name in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"}
        local.update({"lam": sp.Symbol("lam"), "mu": sp.Symbol("mu"), "sigma": sp.Symbol("sigma")})
        local.update({"sin": sp.sin, "cos": sp.cos, "exp": sp.exp, "e": sp.E, "pi": sp.pi, "factorial": sp.factorial})
        glob = dict(sp.__dict__)
        glob["__builtins__"] = {}
        lhs = parse_expr(normalized_left, local_dict=local, global_dict=glob, transformations=transformations)
        rhs = parse_expr(normalized_right, local_dict=local, global_dict=glob, transformations=transformations)
        return sp.simplify(lhs - rhs) == 0
    except Exception:
        return False


def _rhs_of_expression(value: str) -> str:
    assignments = _top_level_assignments(str(value or ""))
    if assignments:
        return assignments[-1]["rhs"]
    return str(value or "").strip().rstrip(".")


def _normalize_symbolic_text(value: str) -> str:
    text = _normalize_benchmark_text(str(value or "")).strip().strip("`$")
    text = text.replace("λ", "lam").replace("μ", "mu").replace("σ", "sigma")
    text = re.sub(r"\blambda\b", "lam", text, flags=re.IGNORECASE)
    text = text.replace("×", "*").replace("·", "*")
    text = re.sub(r"\s+", " ", text)
    return text


def _solution_set_equal(left: str, right: str) -> bool:
    left_norm = _normalize_solution_set(left)
    right_norm = _normalize_solution_set(right)
    if left_norm == right_norm:
        return True
    try:
        return answers_equivalent(left_norm, right_norm)
    except Exception:
        return False


def _normalize_solution_set(value: str) -> str:
    text = _normalize_benchmark_text(str(value or "")).strip().strip("`$.,;: ")
    assignments = _top_level_assignments(text)
    if assignments:
        text = assignments[-1]["rhs"]
    return re.sub(r"\s+", "", text)


def _match_text_alias(text: str, expected: str, aliases: list[str]) -> bool | None:
    groups = {
        "sample_mean": ("sample mean", "sample average", "arithmetic mean", "样本均值"),
        "uniform_convergence": ("uniformly convergent", "converges uniformly", "uniform convergence", "一致收敛"),
        "cauchy_schwarz": ("cauchy-schwarz", "cauchy schwarz", "cauchy-schwarz inequality"),
        "compact": ("compact", "is compact", "remains compact", "紧致"),
    }
    accepted = tuple(aliases) or groups.get(expected.lower(), ())
    if not accepted:
        return None
    polarity = _last_semantic_polarity(str(text or ""), accepted)
    return False if polarity is None else polarity


def _phrase_match(text: str, phrase: str) -> re.Match[str] | None:
    value = str(phrase or "").strip()
    if not value:
        return None
    if value.isascii():
        escaped = re.escape(value).replace(r"\ ", r"\s+")
        return re.search(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", text, flags=re.IGNORECASE)
    return re.search(re.escape(value), text, flags=re.IGNORECASE)


def _match_is_negated(text: str, start: int) -> bool:
    prefix = text[max(0, start - 64):start].lower()
    return bool(
        re.search(
            r"(?:\bnot\b|\bnever\b|\bno\b|\bfails?\s+to\b|\bdoes\s+not\b|\bisn['’]?t\b|\bfailure\s+of\b|\bnon[-\s]?)\s*(?:\w+\s+){0,3}$",
            prefix,
        )
    )


def _grade_claim(response: str, claim: str) -> dict[str, Any]:
    key = re.sub(r"[^a-z0-9]+", "", claim.lower())
    if key in {"varbetahat", "varbetahatsigma2xtx1"}:
        return _grade_var_beta_hat(response, claim)
    if key == "normconjugateindicatordualunitball":
        return _grade_norm_conjugate_indicator(response, claim)
    if key == "equalityifflineardependence":
        return _grade_equality_linear_dependence(response, claim)

    specs = {
        "unbiased": (
            ("unbiased", "is unbiased", "无偏"),
            ("not unbiased", "is biased", "biased estimator", "有偏"),
        ),
        "quadraticconvergence": (
            ("quadratic convergence", "convergence is quadratic", "converges quadratically", "quadratically convergent", "quadratic rate"),
            ("not quadratically", "not quadratic", "fails to converge quadratically"),
        ),
        "dctnotapplicable": (
            ("dct does not apply", "dominated convergence does not apply", "dominated convergence theorem does not apply", "the dominated convergence theorem does not apply", "dct not applicable", "dominated convergence is not applicable"),
            ("dct applies", "dominated convergence applies", "dominated convergence theorem applies"),
        ),
        "weierstrassmtest": (
            ("weierstrass m-test", "m-test"),
            ("m-test does not apply", "weierstrass m-test does not apply"),
        ),
    }
    spec = specs.get(key)
    if spec is None:
        return _result(None, claim=claim)
    positives, negatives = spec
    polarity = _last_semantic_polarity(str(response or ""), positives, negatives)
    return _result(False if polarity is None else polarity, claim=claim)


def _grade_var_beta_hat(response: str, claim: str) -> dict[str, Any]:
    expected = "sigma^2*(X^T X)^(-1)"
    text = _normalize_benchmark_text(str(response or ""))
    targets: list[dict[str, Any]] = []
    for assignment in _top_level_assignments(text):
        lhs_key = re.sub(r"[^a-z0-9]+", "", _canonical_lhs(assignment["lhs"]))
        if lhs_key in {"varbetahat", "covbetahat", "variancebetahat", "covariancebetahat"}:
            targets.append(assignment)
    if targets:
        target = max(targets, key=lambda item: item["position"])
        return _result(
            _symbolic_equivalent(target["rhs"], expected),
            claim=claim,
            extracted=target["rhs"],
            reason=None if _symbolic_equivalent(target["rhs"], expected) else "wrong_variance",
        )

    scoped_candidates: list[tuple[int, str]] = []
    offset = 0
    for statement in _answer_statements(text):
        position = text.find(statement, offset)
        offset = max(offset, position + len(statement)) if position >= 0 else offset
        if not re.search(r"\b(?:var|variance|cov|covariance)\b", statement, flags=re.IGNORECASE):
            continue
        if not re.search(r"beta[_\s]*hat|beta\s*\^?\s*hat", statement, flags=re.IGNORECASE):
            continue
        fragments = _symbolic_conclusion_fragments(statement)
        if fragments:
            scoped_candidates.append((max(position, 0), fragments[0]))
    if not scoped_candidates:
        return _result(False, claim=claim, reason="missing_variance")
    _, fragment = max(scoped_candidates, key=lambda item: item[0])
    return _result(_symbolic_equivalent(fragment, expected), claim=claim, extracted=fragment, reason="wrong_variance")


def _grade_equality_linear_dependence(response: str, claim: str) -> dict[str, Any]:
    text = str(response or "").lower()
    events: list[tuple[int, bool]] = []
    negative_patterns = (
        r"equality.{0,60}(?:iff|if\s+and\s+only\s+if|exactly\s+when).{0,60}(?:linearly\s+independent|not\s+linearly\s+dependent)",
    )
    positive_patterns = (
        r"equality.{0,60}(?:iff|if\s+and\s+only\s+if|exactly\s+when).{0,60}(?:linearly\s+dependent|scalar\s+multiple|proportional)",
        r"(?:linearly\s+dependent|scalar\s+multiple|proportional).{0,60}(?:iff|if\s+and\s+only\s+if).{0,60}equality",
    )
    for pattern in negative_patterns:
        events.extend((match.start(), False) for match in re.finditer(pattern, text))
    for pattern in positive_patterns:
        events.extend((match.start(), True) for match in re.finditer(pattern, text))
    if not events:
        return _result(False, claim=claim)
    events.sort(key=lambda item: item[0])
    return _result(events[-1][1], claim=claim)


def _grade_norm_conjugate_indicator(response: str, claim: str) -> dict[str, Any]:
    text = _normalize_benchmark_text(str(response or "")).lower().replace("∞", "infinity")
    if re.search(r"\bnot\b.{0,24}indicator", text):
        return _result(False, claim=claim)
    if re.search(r"indicator(?:\s+function)?\s+of\s+the\s+dual(?:\s+norm)?\s+unit\s+ball", text):
        return _result(True, claim=claim)
    has_inside_condition = bool(
        re.search(r"(?:dual\s+norm|\|[^\n]{0,24}\|_?\s*\*)[^\n]{0,40}(?:<=|≤)\s*1", text)
        or re.search(r"(?:inside|in)\s+the\s+dual(?:\s+norm)?\s+unit\s+ball", text)
    )
    has_zero = bool(re.search(r"(?:\b0\b.{0,60}(?:when|if|for).{0,60}(?:<=|≤)\s*1)|(?:(?:<=|≤)\s*1.{0,60}(?:=|is|gives)\s*0\b)", text))
    has_infinity = bool(re.search(r"(?:\+?infinity|\+?inf\b)", text))
    has_outside = "otherwise" in text or bool(re.search(r"outside\s+the\s+dual(?:\s+norm)?\s+unit\s+ball", text))
    return _result(bool(has_inside_condition and has_zero and has_infinity and has_outside), claim=claim)


def _grade_structured_primary(response: str, grading_spec: Any) -> dict[str, Any]:
    if not isinstance(grading_spec, dict):
        return _result(None, reason="structured_spec_must_be_object", primary_type="structured")
    targets = grading_spec.get("targets")
    if not isinstance(targets, dict) or not targets:
        return _result(None, reason="missing_structured_targets", primary_type="structured")

    target_results: dict[str, dict[str, Any]] = {}
    for name, raw_target in targets.items():
        target = raw_target if isinstance(raw_target, dict) else {"value": raw_target, "type": "symbolic"}
        target_type = str(target.get("type") or "symbolic").lower()
        target_value = str(target.get("value", target.get("primary", "")) or "").strip()
        target_aliases = target.get("aliases", [])
        target_aliases = target_aliases if isinstance(target_aliases, list) else [target_aliases]
        if target_type == "canonical_claim":
            target_results[str(name)] = _grade_claim(response, target_value)
            continue
        scoped_text = _target_scoped_text(response, [str(alias) for alias in target_aliases if str(alias).strip()])
        sub_spec = {"primary": target_value, "primary_type": target_type, "aliases": target_aliases, "required_claims": []}
        target_results[str(name)] = grade_primary_answer(scoped_text, sub_spec)

    values = [item["correct"] for item in target_results.values()]
    correct = None if any(value is None for value in values) else all(value is True for value in values)
    return _result(
        correct,
        expected=str(grading_spec.get("primary") or ""),
        primary_type="structured",
        targets=target_results,
    )


def _target_scoped_text(response: str, aliases: Iterable[str]) -> str:
    alias_values = [alias for alias in aliases if alias]
    if not alias_values:
        return str(response or "")
    statements = _answer_statements(str(response or ""))
    for statement in reversed(statements):
        if any(_phrase_match(statement, alias) for alias in alias_values):
            return statement
    return str(response or "")

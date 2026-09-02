from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, List

from math_agent_core.state import EvidenceStatus, VerificationEvidence, VerificationLevel


DISCRETE_TOOLS = {"small_case_enumeration", "recurrence_check", "modular_check"}
_MAX_ENUMERATION_STATES = 200_000
_MAX_SUBSET_N = 200
_MAX_PERMUTATION_N = 100
_CANDIDATE_CHECK_NOTICE = "Candidate-proposed auxiliary check; not sufficient by itself to verify final_answer."

_SMALL_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
}


@dataclass(frozen=True)
class FinalIntegerAssertion:
    """Structured semantics for the closed-world V1 final-integer answer surfaces."""

    value: int
    kind: str
    metadata: Dict[str, Any]



def run_discrete_math_verification(problem_text: str, answer: str, result: Dict[str, Any]) -> List[VerificationEvidence]:
    """Run conservative exact checks for high-value discrete-math subtypes."""
    problem_text = str(problem_text or "")
    answer = str(answer or "").strip()
    evidence: List[VerificationEvidence] = []

    for index, spec in enumerate(_infer_system_checks(problem_text, answer), start=1):
        item = _run_check(spec, claim_id=f"system_discrete_{index}")
        default_decisive = item.status in {EvidenceStatus.PASS.value, EvidenceStatus.FAIL.value}
        item.is_decisive = bool(spec.get("decisive", default_decisive)) and default_decisive
        item.claim_scope = str(spec.get("claim_scope") or ("full_answer" if item.is_decisive else "subclaim"))
        item.details = f"System-inferred discrete-math check. {item.details}"
        evidence.append(item)

    for index, spec in enumerate(_extract_requested_checks(result), start=1):
        item = _run_check(spec, claim_id=str(spec.get("claim_id") or f"requested_discrete_{index}"))
        item.is_decisive = False
        item.claim_scope = "subclaim"
        item.details = f"{item.details.rstrip()} {_CANDIDATE_CHECK_NOTICE}"
        evidence.append(item)

    return evidence


def _run_check(spec: Dict[str, Any], claim_id: str) -> VerificationEvidence:
    tool = str(spec.get("tool") or "")
    arguments = spec.get("arguments") if isinstance(spec.get("arguments"), dict) else {}
    try:
        if tool == "modular_check":
            return _modular_check(arguments, claim_id)
        if tool == "recurrence_check":
            return _recurrence_check(arguments, claim_id)
        if tool == "small_case_enumeration":
            return _small_case_enumeration(arguments, claim_id)
        if tool == "_graph_invariant_check":
            return _graph_invariant_check(arguments, claim_id)
        return _inconclusive(tool or "unknown_tool", claim_id, "Unsupported discrete-math tool requested.")
    except Exception as exc:
        return _inconclusive(tool or "discrete_check", claim_id, f"{type(exc).__name__}: {str(exc)[:220]}")


def _modular_check(arguments: Dict[str, Any], claim_id: str) -> VerificationEvidence:
    modulus = _require_int(arguments.get("modulus"), "modulus")
    if modulus <= 0:
        raise ValueError("modulus must be positive")

    if all(key in arguments for key in ("coefficient", "solution", "rhs")):
        coefficient = _require_int(arguments.get("coefficient"), "coefficient")
        solution = _require_int(arguments.get("solution"), "solution")
        rhs = _require_int(arguments.get("rhs"), "rhs")
        actual = (coefficient * solution) % modulus
        expected = rhs % modulus
        representation = str(arguments.get("solution_representation") or "").strip().lower()
        congruence_valid = actual == expected
        if representation == "least_nonnegative":
            range_valid = 0 <= solution < modulus
            status = EvidenceStatus.PASS.value if congruence_valid and range_valid else EvidenceStatus.FAIL.value
            description = (
                f"Checked {coefficient}*{solution} ≡ {rhs} (mod {modulus}) and "
                f"least-nonnegative range 0 <= {solution} < {modulus}."
            )
            residual = None if status == EvidenceStatus.PASS.value else (
                "congruence mismatch" if not congruence_valid else "candidate is outside canonical residue range"
            )
            return VerificationEvidence(
                verifier="discrete_math",
                claim_id=claim_id,
                status=status,
                method="modular_check",
                details=description,
                residual=residual,
                verification_level=VerificationLevel.EXACT_ENUMERATION.value,
                is_decisive=False,
            )
        description = f"Checked {coefficient}*{solution} ≡ {rhs} (mod {modulus})."
    else:
        expected = _require_int(arguments.get("expected"), "expected") % modulus
        if "base" in arguments and "exponent" in arguments:
            base = _require_int(arguments.get("base"), "base")
            exponent = _require_int(arguments.get("exponent"), "exponent")
            if exponent < 0:
                raise ValueError("negative exponents are not supported")
            actual = pow(base, exponent, modulus)
            representation = str(arguments.get("expected_representation") or "").strip().lower()
            if representation == "canonical_remainder":
                expected = _require_int(arguments.get("expected"), "expected")
                status = EvidenceStatus.PASS.value if expected == actual else EvidenceStatus.FAIL.value
                description = (
                    f"Computed canonical remainder pow({base}, {exponent}, {modulus})={actual} "
                    f"and compared it directly with candidate {expected}."
                )
                return VerificationEvidence(
                    verifier="discrete_math",
                    claim_id=claim_id,
                    status=status,
                    method="modular_check",
                    details=description,
                    residual=str(expected - actual),
                    verification_level=VerificationLevel.EXACT_ENUMERATION.value,
                    is_decisive=False,
                )
            description = f"Computed pow({base}, {exponent}, {modulus})."
        else:
            value = _require_int(arguments.get("value"), "value")
            actual = value % modulus
            description = f"Reduced {value} modulo {modulus}."

    status = EvidenceStatus.PASS.value if actual == expected else EvidenceStatus.FAIL.value
    return VerificationEvidence(
        verifier="discrete_math",
        claim_id=claim_id,
        status=status,
        method="modular_check",
        details=description,
        residual=str(actual - expected),
        verification_level=VerificationLevel.EXACT_ENUMERATION.value,
        is_decisive=False,
    )


def _recurrence_check(arguments: Dict[str, Any], claim_id: str) -> VerificationEvidence:
    initial_values = _fraction_list(arguments.get("initial_values"), "initial_values")
    coefficients = _fraction_list(arguments.get("coefficients"), "coefficients")
    claimed_values = _fraction_list(arguments.get("claimed_values"), "claimed_values")
    constant = _to_fraction(arguments.get("constant", 0))
    if not initial_values or not coefficients or not claimed_values:
        raise ValueError("initial_values, coefficients, and claimed_values are required")
    order = len(coefficients)
    if len(initial_values) < order:
        raise ValueError("initial_values must contain at least recurrence-order terms")
    if len(initial_values) + len(claimed_values) > 200:
        raise ValueError("recurrence check is limited to 200 terms")

    generated = list(initial_values)
    while len(generated) < len(initial_values) + len(claimed_values):
        previous = generated[-order:]
        next_value = constant + sum(coeff * value for coeff, value in zip(coefficients, reversed(previous)))
        generated.append(next_value)
    actual_values = generated[len(initial_values):]
    mismatch = next(
        (idx for idx, (actual, claimed) in enumerate(zip(actual_values, claimed_values)) if actual != claimed),
        None,
    )
    status = EvidenceStatus.PASS.value if mismatch is None else EvidenceStatus.FAIL.value
    residual = (
        None
        if mismatch is None
        else f"index {mismatch}: expected {actual_values[mismatch]}, got {claimed_values[mismatch]}"
    )
    return VerificationEvidence(
        verifier="discrete_math",
        claim_id=claim_id,
        status=status,
        method="recurrence_check",
        details=f"Generated {len(claimed_values)} exact term(s) from a linear recurrence of order {order}.",
        residual=residual,
        verification_level=VerificationLevel.EXACT_ENUMERATION.value,
        is_decisive=False,
    )


def _small_case_enumeration(arguments: Dict[str, Any], claim_id: str) -> VerificationEvidence:
    kind = str(arguments.get("kind") or "binary_strings").strip().lower()
    expected = _require_int(arguments.get("expected"), "expected")
    if kind == "binary_strings":
        actual, state_count = _enumerate_binary_strings(arguments)
    elif kind == "integer_tuples":
        actual, state_count = _enumerate_integer_tuples(arguments)
    elif kind == "subsets":
        actual, state_count = _count_subsets(arguments)
    elif kind == "permutations":
        actual, state_count = _count_permutations(arguments)
    else:
        raise ValueError("kind must be binary_strings, integer_tuples, subsets, or permutations")
    status = EvidenceStatus.PASS.value if actual == expected else EvidenceStatus.FAIL.value
    return VerificationEvidence(
        verifier="discrete_math",
        claim_id=claim_id,
        status=status,
        method="small_case_enumeration",
        details=f"Exact finite check for {kind}; states/objects={state_count}, matching objects={actual}.",
        residual=str(actual - expected),
        verification_level=VerificationLevel.EXACT_ENUMERATION.value,
        is_decisive=False,
    )


def _enumerate_binary_strings(arguments: Dict[str, Any]) -> tuple[int, int]:
    length = _require_int(arguments.get("length"), "length")
    if length < 0 or length > 18:
        raise ValueError("binary-string enumeration requires 0 <= length <= 18")
    ones = arguments.get("ones")
    ones = None if ones is None else _require_int(ones, "ones")
    if ones is not None and not 0 <= ones <= length:
        return 0, 1 << length
    no_adjacent_ones = bool(arguments.get("no_adjacent_ones", False))
    state_count = 1 << length
    count = 0
    for bits in itertools.product((0, 1), repeat=length):
        if ones is not None and sum(bits) != ones:
            continue
        if no_adjacent_ones and any(bits[i] == bits[i + 1] == 1 for i in range(max(0, length - 1))):
            continue
        count += 1
    return count, state_count


def _enumerate_integer_tuples(arguments: Dict[str, Any]) -> tuple[int, int]:
    length = _require_int(arguments.get("length"), "length")
    min_value = _require_int(arguments.get("min_value", 0), "min_value")
    max_value = _require_int(arguments.get("max_value"), "max_value")
    if length < 0 or length > 8 or max_value < min_value:
        raise ValueError("invalid integer-tuple bounds")
    width = max_value - min_value + 1
    state_count = width ** length
    if state_count > _MAX_ENUMERATION_STATES:
        raise ValueError(f"enumeration exceeds {_MAX_ENUMERATION_STATES} states")
    target_sum = arguments.get("sum")
    target_sum = None if target_sum is None else _require_int(target_sum, "sum")
    distinct = bool(arguments.get("distinct", False))
    count = 0
    for values in itertools.product(range(min_value, max_value + 1), repeat=length):
        if target_sum is not None and sum(values) != target_sum:
            continue
        if distinct and len(set(values)) != len(values):
            continue
        count += 1
    return count, state_count


def _count_subsets(arguments: Dict[str, Any]) -> tuple[int, int]:
    n = _require_int(arguments.get("n"), "n")
    k = _require_int(arguments.get("k"), "k")
    if n < 0 or n > _MAX_SUBSET_N or k < 0:
        raise ValueError(f"subsets requires 0 <= n <= {_MAX_SUBSET_N} and k >= 0")
    actual = math.comb(n, k) if k <= n else 0
    return actual, actual


def _count_permutations(arguments: Dict[str, Any]) -> tuple[int, int]:
    n = _require_int(arguments.get("n"), "n")
    k = _require_int(arguments.get("k"), "k")
    if n < 0 or n > _MAX_PERMUTATION_N or k < 0:
        raise ValueError(f"permutations requires 0 <= n <= {_MAX_PERMUTATION_N} and k >= 0")
    actual = math.perm(n, k) if k <= n else 0
    return actual, actual


def _graph_invariant_check(arguments: Dict[str, Any], claim_id: str) -> VerificationEvidence:
    kind = str(arguments.get("kind") or "").strip().lower()
    expected = _require_int(arguments.get("expected"), "expected")
    if kind == "tree_edges":
        n = _require_int(arguments.get("n"), "n")
        if n < 1:
            raise ValueError("tree vertex count must be positive")
        actual = n - 1
        description = f"Used |E|=|V|-1 for a tree with {n} vertices."
    elif kind == "complete_graph_edges":
        n = _require_int(arguments.get("n"), "n")
        if n < 0:
            raise ValueError("complete-graph vertex count must be nonnegative")
        actual = n * (n - 1) // 2
        description = f"Used |E(K_{n})|=n(n-1)/2."
    elif kind == "complete_bipartite_edges":
        m = _require_int(arguments.get("m"), "m")
        n = _require_int(arguments.get("n"), "n")
        if m < 0 or n < 0:
            raise ValueError("bipartition sizes must be nonnegative")
        actual = m * n
        description = f"Used |E(K_{{{m},{n}}})|=mn."
    elif kind == "handshake_edges":
        degrees = arguments.get("degrees")
        if not isinstance(degrees, list) or not degrees:
            raise ValueError("degrees must be a nonempty list")
        degree_values = [_require_int(value, "degree") for value in degrees]
        if any(value < 0 for value in degree_values):
            raise ValueError("degrees must be nonnegative")
        total = sum(degree_values)
        if total % 2:
            raise ValueError("degree sum is odd, so no integer edge count follows")
        actual = total // 2
        description = f"Used the handshake lemma on degree sum {total}."
    else:
        raise ValueError("unsupported graph invariant kind")
    status = EvidenceStatus.PASS.value if actual == expected else EvidenceStatus.FAIL.value
    return VerificationEvidence(
        verifier="discrete_math",
        claim_id=claim_id,
        status=status,
        method="graph_invariant_check",
        details=description,
        residual=str(actual - expected),
        verification_level=VerificationLevel.EXACT_ENUMERATION.value,
        is_decisive=False,
    )


def _infer_system_checks(problem_text: str, answer: str) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    assertion = _extract_final_integer_assertion(answer)
    if assertion is None:
        return checks

    inferers = (
        _infer_modular_power,
        _infer_linear_congruence,
        _infer_binary_string_count,
        _infer_subset_count,
        _infer_permutation_count,
        _infer_integer_tuple_count,
        _infer_graph_invariant,
    )
    for inferer in inferers:
        spec = inferer(problem_text, assertion.value)
        if spec is not None and _system_answer_contract_allows(assertion, spec):
            checks.append(spec)
    return checks[:3]


def _system_answer_contract_allows(assertion: FinalIntegerAssertion, spec: Dict[str, Any]) -> bool:
    """Require assertion syntax to match the semantics of a decisive system-inferred check."""
    kind = assertion.kind
    tool = str(spec.get("tool") or "")
    arguments = spec.get("arguments") if isinstance(spec.get("arguments"), dict) else {}

    if tool == "small_case_enumeration":
        # V1 finite-enumeration checks may consume a neutral number, but an
        # explicit counted-object noun must match the closed-world problem kind.
        if kind == "neutral_numeric":
            return True
        if kind != "count_statement":
            return False

        enumeration_kind = str(arguments.get("kind") or "").strip().lower()
        noun = str(assertion.metadata.get("noun") or "").strip().lower()
        compatible_count_nouns = {
            "subsets": {"way", "ways", "subset", "subsets", "selection", "selections"},
            "permutations": {
                "way",
                "ways",
                "permutation",
                "permutations",
                "selection",
                "selections",
            },
            "binary_strings": {"way", "ways", "string", "strings"},
            "integer_tuples": {"way", "ways", "solution", "solutions"},
        }
        return noun in compatible_count_nouns.get(enumeration_kind, set())

    if tool == "_graph_invariant_check":
        if kind == "neutral_numeric":
            return True
        if kind != "tree_edge_statement" or str(arguments.get("kind") or "") != "tree_edges":
            return False
        return assertion.metadata.get("vertices") == arguments.get("n")

    if tool == "modular_check":
        if "base" in arguments and "exponent" in arguments:
            # V1 canonical-remainder questions accept only neutral answer assertions.
            return kind == "neutral_numeric"

        if all(key in arguments for key in ("coefficient", "solution", "rhs")):
            if kind == "neutral_numeric":
                return True
            if kind == "assignment":
                return str(assertion.metadata.get("lhs") or "").strip().lower() == "x"
            if kind == "congruence":
                lhs = str(assertion.metadata.get("lhs") or "").strip().lower()
                return lhs == "x" and assertion.metadata.get("modulus") == arguments.get("modulus")
            return False

    return False


def _extract_requested_checks(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    requested = result.get("requested_checks") if isinstance(result, dict) else None
    if not isinstance(requested, list):
        return []
    checks: List[Dict[str, Any]] = []
    for item in requested:
        if not isinstance(item, dict):
            continue
        tool = str(item.get("tool") or "")
        arguments = item.get("arguments")
        if tool in DISCRETE_TOOLS and isinstance(arguments, dict):
            checks.append({"tool": tool, "arguments": arguments, "claim_id": item.get("claim_id")})
    return checks[:5]


def _extract_single_integer(answer: str) -> int | None:
    """Backward-compatible alias for the V1 discrete final-integer extractor."""
    return _extract_final_integer(answer)


def _top_level_assignment_rhs(text: str) -> List[str]:
    depths = {"(": 0, "[": 0, "{": 0}
    closing = {")": "(", "]": "[", "}": "{"}
    positions: List[int] = []
    for index, char in enumerate(text):
        if char in depths:
            depths[char] += 1
            continue
        if char in closing:
            opener = closing[char]
            depths[opener] = max(0, depths[opener] - 1)
            continue
        if char != "=" or any(depths.values()):
            continue
        previous = text[index - 1] if index else ""
        following = text[index + 1] if index + 1 < len(text) else ""
        if previous in "<>=!" or following == "=":
            continue
        positions.append(index)
    rhs_values: List[str] = []
    for offset, position in enumerate(positions):
        end = positions[offset + 1] if offset + 1 < len(positions) else len(text)
        rhs_values.append(text[position + 1:end].strip())
    return rhs_values


def _small_integer_token(token: str) -> int:
    value = str(token or "").strip().lower()
    if value.isdigit():
        return int(value)
    if value in _SMALL_NUMBER_WORDS:
        return _SMALL_NUMBER_WORDS[value]
    raise ValueError("unsupported small integer token")


def _require_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        integer = int(str(value).strip())
    except Exception as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if str(integer) != str(value).strip() and not isinstance(value, int):
        try:
            if Fraction(str(value).strip()) != integer:
                raise ValueError(f"{name} must be an integer")
        except Exception as exc:
            raise ValueError(f"{name} must be an integer") from exc
    return integer


def _to_fraction(value: Any) -> Fraction:
    return Fraction(str(value).strip())


def _fraction_list(value: Any, name: str) -> List[Fraction]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return [_to_fraction(item) for item in value]


def _inconclusive(method: str, claim_id: str, details: str) -> VerificationEvidence:
    return VerificationEvidence(
        verifier="discrete_math",
        claim_id=claim_id,
        status=EvidenceStatus.INCONCLUSIVE.value,
        method=method,
        details=details,
        verification_level=VerificationLevel.EXACT_ENUMERATION.value,
        is_decisive=False,
    )

# --- Discrete Math V1.1 closed-world system-inference safety layer ---
# These definitions intentionally replace the earlier V1 inferers at module load
# time. Full-answer decisiveness now requires an anchored complete-template match;
# unknown residual mathematical content is never treated as harmless.

_REQUEST_LEAST_NONNEGATIVE_RESIDUE = "LEAST_NONNEGATIVE_RESIDUE"
_REQUEST_RESIDUE_MOD_M = "RESIDUE_MOD_M"
_REQUEST_ALL_INTEGER_SOLUTIONS = "ALL_INTEGER_SOLUTIONS"
_REQUEST_UNSPECIFIED = "UNSPECIFIED"
_REQUEST_UNKNOWN_CONSTRAINED = "UNKNOWN_CONSTRAINED"


def _closed_surface(text: str) -> str:
    value = str(text or "").strip().lower()
    value = value.replace("\\equiv", "≡").replace("\\pmod", " mod ")
    value = value.replace("\\{", "{").replace("\\}", "}")
    value = value.replace("−", "-").replace("，", ",").replace("？", "?").replace("。", ".")
    value = re.sub(r"\s+", " ", value)
    return re.sub(r"[\s?.!。！？]+$", "", value).strip()


def _closed_fullmatch(text: str, patterns: tuple[str, ...]) -> re.Match[str] | None:
    value = _closed_surface(text)
    for pattern in patterns:
        match = re.fullmatch(pattern, value, re.IGNORECASE)
        if match:
            return match
    return None


def _infer_modular_power(problem_text: str, answer_value: int) -> Dict[str, Any] | None:
    match = _closed_fullmatch(problem_text, (
        r"(?:compute|calculate|evaluate|find)\s+(-?\d+)\s*(?:\^|\*\*)\s*(\d+)\s+(?:mod(?:ulo)?|mod)\s+(\d+)",
        r"what\s+is\s+(-?\d+)\s*(?:\^|\*\*)\s*(\d+)\s+(?:mod(?:ulo)?|mod)\s+(\d+)",
        r"计算\s*(-?\d+)\s*(?:\^|\*\*)\s*(\d+)\s*模\s*(\d+)\s*的余数",
    ))
    if not match:
        return None
    base, exponent, modulus = map(int, match.groups())
    if modulus <= 0:
        return None
    return {"tool": "modular_check", "arguments": {
        "base": base, "exponent": exponent, "modulus": modulus, "expected": answer_value,
        "expected_representation": "canonical_remainder",
    }}


def _congruence_mode(text: str, coefficient: int, rhs: int, modulus: int) -> str:
    value = _closed_surface(text)
    core = (
        rf"{re.escape(str(coefficient))}\s*\*?\s*x\s*≡\s*{re.escape(str(rhs))}"
        rf"\s*\(?\s*(?:mod(?:ulo)?|mod|模)\s*{re.escape(str(modulus))}\s*\)?"
    )
    families = (
        (_REQUEST_LEAST_NONNEGATIVE_RESIDUE, (
            rf"find the (?:least|smallest) nonnegative (?:solution|residue)(?: to)? {core}",
            rf"{core}(?:\s*[,;.]\s*|\s+and\s+)give the (?:least|smallest) nonnegative (?:solution|residue)",
        )),
        (_REQUEST_RESIDUE_MOD_M, (
            rf"solve {core} for x modulo {modulus}",
            rf"solve {core} for the (?:unique )?residue x modulo {modulus}",
            rf"find the (?:unique )?residue x modulo {modulus} satisfying {core}",
        )),
        (_REQUEST_ALL_INTEGER_SOLUTIONS, (
            rf"find all integers? x satisfying {core}",
            rf"find all integer solutions? (?:to )?{core}",
            rf"solve {core} in integers",
            rf"solve {core} over the integers",
            rf"solve {core} for integer x",
            rf"find the integer solutions? (?:to )?{core}",
            rf"求{core}的?所有整数解",
            rf"在整数范围内求解 {core}",
        )),
        (_REQUEST_UNSPECIFIED, (
            rf"solve {core}", rf"解同余(?:方程)? {core}", rf"求解 {core}",
        )),
    )
    for mode, patterns in families:
        if any(re.fullmatch(pattern, value, re.IGNORECASE) for pattern in patterns):
            return mode
    return _REQUEST_UNKNOWN_CONSTRAINED


def _infer_linear_congruence(problem_text: str, answer_value: int) -> Dict[str, Any] | None:
    inverse = _closed_fullmatch(problem_text, (
        r"find the modular inverse of (-?\d+) modulo (\d+)",
        r"find the modular inverse of (-?\d+) mod (\d+)",
    ))
    if inverse:
        coefficient, modulus = map(int, inverse.groups())
        if modulus <= 0:
            return None
        decisive = math.gcd(coefficient, modulus) == 1
        return {"tool": "modular_check", "arguments": {
            "coefficient": coefficient, "solution": answer_value, "rhs": 1, "modulus": modulus,
        }, "decisive": decisive, "claim_scope": "full_answer" if decisive else "subclaim",
        "request_mode": _REQUEST_RESIDUE_MOD_M}

    value = _closed_surface(problem_text)
    core_match = re.search(
        r"(-?\d+)\s*\*?\s*x\s*≡\s*(-?\d+)\s*\(?\s*(?:mod(?:ulo)?|mod|模)\s*(\d+)\s*\)?",
        value, re.IGNORECASE,
    )
    if not core_match:
        return None
    coefficient, rhs, modulus = map(int, core_match.groups())
    if modulus <= 0:
        return None
    mode = _congruence_mode(problem_text, coefficient, rhs, modulus)
    decisive = math.gcd(coefficient, modulus) == 1 and mode in {
        _REQUEST_LEAST_NONNEGATIVE_RESIDUE, _REQUEST_RESIDUE_MOD_M, _REQUEST_UNSPECIFIED,
    }
    arguments = {
        "coefficient": coefficient, "solution": answer_value, "rhs": rhs, "modulus": modulus,
    }
    if mode == _REQUEST_LEAST_NONNEGATIVE_RESIDUE:
        arguments["solution_representation"] = "least_nonnegative"
    return {"tool": "modular_check", "arguments": arguments,
        "decisive": decisive, "claim_scope": "full_answer" if decisive else "subclaim",
        "request_mode": mode}


def _infer_binary_string_count(problem_text: str, answer_value: int) -> Dict[str, Any] | None:
    number = (r"(?:\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
              r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen)")
    patterns = (
        rf"how many binary strings of length (?P<n>\d+) (?:contain|containing|have|with) exactly (?P<k>{number}) (?:ones?|1s?)",
        rf"how many length[- ](?P<n2>\d+) binary strings (?:contain|containing|have|with) exactly (?P<k2>{number}) (?:ones?|1s?)",
        r"how many binary strings of length (?P<n3>\d+) (?:have|with|containing) no (?:consecutive|adjacent) (?:ones?|1s?)",
        rf"how many binary strings of length (?P<n4>\d+) (?:contain|containing|have|with) exactly (?P<k4>{number}) (?:ones?|1s?) and no (?:consecutive|adjacent) (?:ones?|1s?)",
        rf"how many binary strings of length (?P<n5>\d+) (?:contain|containing|have|with) exactly (?P<k5>{number}) (?:ones?|1s?) with no (?:consecutive|adjacent) (?:ones?|1s?)",
    )
    match = _closed_fullmatch(problem_text, patterns)
    if not match:
        return None
    groups = match.groupdict()
    length = next(int(v) for key, v in groups.items() if key.startswith("n") and v is not None)
    if length < 0 or length > 18:
        return None
    token = next((v for key, v in groups.items() if key.startswith("k") and v is not None), None)
    ones = _small_integer_token(token) if token is not None else None
    no_adjacent = bool(re.search(r"no (?:consecutive|adjacent)", _closed_surface(problem_text)))
    arguments: Dict[str, Any] = {"kind": "binary_strings", "length": length,
                                 "no_adjacent_ones": no_adjacent, "expected": answer_value}
    if ones is not None:
        arguments["ones"] = ones
    return {"tool": "small_case_enumeration", "arguments": arguments}


def _infer_subset_count(problem_text: str, answer_value: int) -> Dict[str, Any] | None:
    value = _closed_surface(problem_text)
    templates = (
        (r"choose (\d+) (?:distinct )?(?:objects?|items?|people|students) from (\d+)(?: distinct)?(?: objects?|items?|people|students)?", "kn"),
        (r"choose (\d+) from (\d+)", "kn"),
        (r"how many ways can (\d+) (?:distinct )?(?:objects?|items?|people|students) be chosen from (\d+)(?: distinct)?(?: objects?|items?|people|students)?", "kn"),
        (r"how many subsets of size (\d+) can be chosen from (\d+) distinct objects", "kn"),
        (r"how many subsets of size (\d+) can be chosen from (\d+) objects", "kn"),
    )
    for pattern, order in templates:
        match = re.fullmatch(pattern, value, re.IGNORECASE)
        if not match:
            continue
        first, second = map(int, match.groups())
        k, n = (first, second) if order == "kn" else (second, first)
        if n > _MAX_SUBSET_N:
            return None
        return {"tool": "small_case_enumeration", "arguments": {
            "kind": "subsets", "n": n, "k": k, "expected": answer_value,
        }}
    return None


def _infer_permutation_count(problem_text: str, answer_value: int) -> Dict[str, Any] | None:
    value = _closed_surface(problem_text)
    templates = (
        (r"how many ordered selections of (\d+) from (\d+) without repetition", "kn"),
        (r"how many ordered selections of (\d+) objects from (\d+) distinct objects are possible without repetition", "kn"),
        (r"ordered selections? of (\d+) (?:distinct )?(?:objects?|items?) from (\d+) without repetition", "kn"),
        (r"permutations? of (\d+) objects taken (\d+)", "nk"),
    )
    for pattern, order in templates:
        match = re.fullmatch(pattern, value, re.IGNORECASE)
        if not match:
            continue
        first, second = map(int, match.groups())
        k, n = (first, second) if order == "kn" else (second, first)
        if n > _MAX_PERMUTATION_N:
            return None
        return {"tool": "small_case_enumeration", "arguments": {
            "kind": "permutations", "n": n, "k": k, "expected": answer_value,
        }}
    return None


def _infer_integer_tuple_count(problem_text: str, answer_value: int) -> Dict[str, Any] | None:
    match = _closed_fullmatch(problem_text, (
        r"how many (nonnegative|non-negative|positive) integer (pairs?|triples?) satisfy ([a-z](?:\s*\+\s*[a-z]){1,7})\s*=\s*(\d+)",
    ))
    if not match:
        return None
    domain_token, shape, equation_vars, target_token = match.groups()
    min_value = 0 if domain_token in {"nonnegative", "non-negative"} else 1
    expected_arity = 2 if shape in {"pair", "pairs"} else 3
    variables = [part.strip() for part in equation_vars.split("+")]
    if len(variables) != expected_arity or len(set(variables)) != len(variables):
        return None
    length, target = len(variables), int(target_token)
    width = target - min_value + 1
    if width < 0 or length > 8 or width ** length > _MAX_ENUMERATION_STATES:
        return None
    return {"tool": "small_case_enumeration", "arguments": {
        "kind": "integer_tuples", "length": length, "min_value": min_value,
        "max_value": target, "sum": target, "expected": answer_value,
    }}


def _infer_graph_invariant(problem_text: str, answer_value: int) -> Dict[str, Any] | None:
    bipartite = _closed_fullmatch(problem_text, (
        r"how many edges does (?:the complete bipartite graph )?k_?\{?(\d+),\s*(\d+)\}? have",
        r"how many edges are in (?:the complete bipartite graph )?k_?\{?(\d+),\s*(\d+)\}?",
    ))
    if bipartite:
        m, n = map(int, bipartite.groups())
        return {"tool": "_graph_invariant_check", "arguments": {"kind": "complete_bipartite_edges", "m": m, "n": n, "expected": answer_value}}
    maximum_simple = _closed_fullmatch(problem_text, (
        r"what is the maximum number of edges in a simple graph on (\d+) vertices",
        r"what is the maximum number of edges in a simple graph with (\d+) vertices",
    ))
    if maximum_simple:
        return {"tool": "_graph_invariant_check", "arguments": {"kind": "complete_graph_edges", "n": int(maximum_simple.group(1)), "expected": answer_value}}
    complete = _closed_fullmatch(problem_text, (
        r"how many edges does (?:the complete graph )?k_?\{?(\d+)\}? have",
        r"how many edges are in (?:the complete graph )?k_?\{?(\d+)\}?",
    ))
    if complete:
        return {"tool": "_graph_invariant_check", "arguments": {"kind": "complete_graph_edges", "n": int(complete.group(1)), "expected": answer_value}}
    tree = _closed_fullmatch(problem_text, (
        r"a tree with (\d+) vertices has how many edges",
        r"a tree on (\d+) vertices has how many edges",
        r"how many edges does a tree with (\d+) vertices have",
        r"how many edges does a tree on (\d+) vertices have",
    ))
    if tree:
        return {"tool": "_graph_invariant_check", "arguments": {"kind": "tree_edges", "n": int(tree.group(1)), "expected": answer_value}}
    degree = _closed_fullmatch(problem_text, (
        r"a graph has degree sequence \[([^\]]+)\]\.? how many edges does it have",
        r"given degree sequence \[([^\]]+)\],? how many edges does the graph have",
    ))
    if degree:
        raw = degree.group(1)
        if not re.fullmatch(r"\s*-?\d+(?:\s*,\s*-?\d+)*\s*", raw):
            return None
        degrees = [int(part.strip()) for part in raw.split(",")]
        if any(value < 0 for value in degrees) or sum(degrees) % 2:
            return None
        return {"tool": "_graph_invariant_check", "arguments": {"kind": "handshake_edges", "degrees": degrees, "expected": answer_value}}
    return None


def _extract_final_integer_assertion(answer: str) -> FinalIntegerAssertion | None:
    """Extract a supported final integer assertion without discarding its answer semantics."""
    text = re.sub(r"\s+", " ", str(answer or "").strip().replace("$", ""))
    if not text:
        return None

    match = re.fullmatch(r"\s*[({[]*\s*(-?\d+)\s*[)}\].,;:!]*\s*", text)
    if match:
        return FinalIntegerAssertion(int(match.group(1)), "neutral_numeric", {})

    match = re.fullmatch(
        r"([A-Za-z][A-Za-z0-9_]*)\s*(?:≡|\\equiv)\s*(-?\d+)\s*\(?\s*(?:mod(?:ulo)?)\s+(\d+)\s*\)?[.\s]*",
        text,
        re.IGNORECASE,
    )
    if match:
        return FinalIntegerAssertion(
            int(match.group(2)),
            "congruence",
            {"lhs": match.group(1), "modulus": int(match.group(3))},
        )

    # Final markers are affirmative only on explicitly supported surfaces:
    # either the assertion starts the answer, or it begins a distinct terminal sentence.
    final_surfaces = (
        r"(?:final answer|final result)\s*(?:is|equals|=|:)?\s*(-?\d+)\s*[\])}.,;:!]*",
        r".+[.!?]\s+(?:final answer|final result)\s*(?:is|equals|=|:)?\s*(-?\d+)\s*[\])}.,;:!]*",
        r"after checking the cases,\s*(?:final answer|final result)\s*(?:is|equals|=|:)?\s*(-?\d+)\s*[\])}.,;:!]*",
        r"after the derivation above,\s*(?:final answer|final result)\s*(?:is|equals|=|:)?\s*(-?\d+)\s*[\])}.,;:!]*",
    )
    for pattern in final_surfaces:
        match = re.fullmatch(pattern, text, re.IGNORECASE)
        if match:
            return FinalIntegerAssertion(int(match.group(1)), "neutral_numeric", {})

    # Positive correction conclusions are also closed-world sentence families.
    match = re.fullmatch(
        r"there are\s+(-?\d+)\s+ways?\s+before excluding one invalid case,\s+so\s+(-?\d+)\s+remain[.!]?",
        text,
        re.IGNORECASE,
    )
    if match:
        return FinalIntegerAssertion(
            int(match.group(2)),
            "count_statement",
            {"noun": "ways", "surface": "remaining_count"},
        )

    correction_surfaces = (
        r"the answer is\s+(-?\d+)\s+is incorrect;\s*actually\s+(-?\d+)[.!]?",
        r"actually\s+(-?\d+)[.!]?",
        r"instead\s+(?:the\s+answer\s+is\s+)?(-?\d+)[.!]?",
    )
    for pattern in correction_surfaces:
        match = re.fullmatch(pattern, text, re.IGNORECASE)
        if match:
            return FinalIntegerAssertion(int(match.groups()[-1]), "neutral_numeric", {})

    match = re.fullmatch(
        r"so\s+(-?\d+)\s+(remain|remains|ways?|solutions?)[.!]?",
        text,
        re.IGNORECASE,
    )
    if match:
        return FinalIntegerAssertion(
            int(match.group(1)),
            "count_statement",
            {"noun": match.group(2).lower()},
        )

    # Any conditional branch with competing numerical conclusions is unsafe.
    if re.search(r"\b(?:if|unless|otherwise|in which case|provided)\b", text, re.IGNORECASE) and len(
        re.findall(r"(?<![\w.])-?\d+\b", text)
    ) >= 2:
        return None

    neutral_patterns = (
        r"(?:answer|the\s+answer|the\s+result)\s*(?:is|equals|=|:)\s*(-?\d+)[.!]?",
        r"(?:therefore|hence|thus|consequently)\s+(?:the\s+)?(?:answer|result)\s*(?:is|equals|=|:)?\s*(-?\d+)[.!]?",
        r"(?:therefore|hence|thus|consequently)\s*(-?\d+)[.!]?",
        r"\\boxed\s*\{\s*(-?\d+)\s*\}[.!]?",
    )
    for pattern in neutral_patterns:
        match = re.fullmatch(pattern, text, re.IGNORECASE)
        if match:
            return FinalIntegerAssertion(int(match.group(1)), "neutral_numeric", {})

    match = re.fullmatch(
        r"([A-Za-z][A-Za-z0-9_{}(),]*)\s*=\s*(-?\d+)[.!]?",
        text,
        re.IGNORECASE,
    )
    if match:
        return FinalIntegerAssertion(
            int(match.group(2)),
            "assignment",
            {"lhs": match.group(1)},
        )

    match = re.fullmatch(
        r"there are\s+([A-Za-z][A-Za-z0-9_{}(),]*)\s*=\s*(-?\d+)\s+(?:such\s+)?"
        r"(strings?|ways?|objects?|subsets?|selections?)[.!]?",
        text,
        re.IGNORECASE,
    )
    if match:
        return FinalIntegerAssertion(
            int(match.group(2)),
            "count_statement",
            {"lhs": match.group(1), "noun": match.group(3).lower()},
        )

    match = re.fullmatch(
        r"there are\s+(?:exactly\s+)?(-?\d+)\s+"
        r"(ways?|strings?|solutions?|subsets?|permutations?|selections?)[.!]?",
        text,
        re.IGNORECASE,
    )
    if match:
        return FinalIntegerAssertion(
            int(match.group(1)),
            "count_statement",
            {"noun": match.group(2).lower()},
        )

    match = re.fullmatch(
        r"(?:a|the)\s+tree with (\d+) vertices has\s+(-?\d+)\s+edges[.!]?",
        text,
        re.IGNORECASE,
    )
    if match:
        return FinalIntegerAssertion(
            int(match.group(2)),
            "tree_edge_statement",
            {"vertices": int(match.group(1))},
        )

    return None


def _extract_final_integer(answer: str) -> int | None:
    """Backward-compatible V1 helper returning only the structured assertion's integer value."""
    assertion = _extract_final_integer_assertion(answer)
    return None if assertion is None else assertion.value


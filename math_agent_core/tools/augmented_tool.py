from __future__ import annotations

import ast
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .sympy_tool import _parse_expr as _safe_parse_expr


MAX_PROTOCOL_CHARS = 6000
MAX_PROTOCOL_LINES = 32
MAX_TOOL_REQUESTS = 6
MAX_TOOL_ARG_CHARS = 700
MAX_EVIDENCE_CHARS = 1500
MAX_RESULT_CHARS = 520
MAX_BATCH_TIMEOUT_SECONDS = 7.0
MAX_VERIFY_TIMEOUT_SECONDS = 5.0
MAX_MATRIX_DIM = 6
MAX_MATRIX_ENTRIES = MAX_MATRIX_DIM * MAX_MATRIX_DIM
MAX_WORKER_INPUT_CHARS = 262_144
MAX_WORKER_OUTPUT_CHARS = 32_768
MAX_VERIFY_PROBLEM_CHARS = 32_768
MAX_VERIFY_RESPONSE_CHARS = 131_072

# These names are system-owned dispatch labels.  They are never parsed from a
# formalizer response and are the only verification targets a worker accepts.
_VERIFICATION_KINDS = frozenset(
    {
        "pure_arithmetic",
        "equation_solution",
        "equation_solution_set",
        "matrix_determinant",
        "matrix_rank",
    }
)
_VERIFICATION_REQUEST_MAX = 4
_VERIFICATION_REQUEST_KEYS = {
    "pure_arithmetic": frozenset({"kind", "expression"}),
    "equation_solution": frozenset({"kind", "equation", "variable"}),
    "equation_solution_set": frozenset({"kind", "equation", "variable", "domain"}),
    "matrix_determinant": frozenset({"kind", "matrix"}),
    "matrix_rank": frozenset({"kind", "matrix"}),
}
_VERIFICATION_DOMAINS = frozenset(
    {
        "complex",
        "real",
        "positive_real",
        "nonnegative_real",
        "positive_integer",
        "nonnegative_integer",
        "integer",
    }
)
_TOOL_WORKER_STATUSES = frozenset(
    {
        "no_requests",
        "parse_rejected",
        "completed",
        "partial_success",
        "timeout",
        "worker_error",
    }
)

_WORKER_COMMAND = (sys.executable, "-m", "math_agent_core.tools.augmented_worker")
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

_ALLOWED_PLAN_KEYS = {
    "TARGET",
    "METHOD",
    "FACT",
    "CHECK",
    "CLAIM",
    "STATUS",
    "HYPOTHESES",
    "SUBGOAL",
    "THEOREM",
    "FAILURE_TRAP",
    "COUNTEREXAMPLE",
}
_CORE_FAMILIES = {"sympy", "matrix"}
_EXTENDED_FAMILIES = {
    "number_theory",
    "combinatorics",
    "recurrence",
    "graph",
    "numerical",
    "probability",
}
_ALLOWED_ARG_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,31}$")

_PARAMETER_NORMALIZATIONS = {
    "λ": "lam",
    "θ": "theta",
    "μ": "mu",
    "σ": "sigma",
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "ρ": "rho",
    "φ": "phi",
    "ϕ": "phi",
    "ω": "omega",
    "∞": "oo",
}
_PROTECTED_PARAMETER_NAMES = (
    "lam",
    "theta",
    "mu",
    "sigma",
    "alpha",
    "beta",
    "gamma",
    "rho",
    "phi",
    "omega",
)


@dataclass(frozen=True)
class ToolRequest:
    family: str
    operation: str
    arguments: Dict[str, Any]
    source: str = ""


@dataclass(frozen=True)
class ToolFact:
    family: str
    operation: str
    statement: str
    # Provenance is intentionally compact and system-owned.  It lets the
    # parent distinguish a target-bound fact from an advisory formalizer fact
    # without exposing raw requests or changing the public statement format.
    source: str = ""


@dataclass(frozen=True)
class ToolBatchResult:
    """A bounded batch result plus a sanitized worker outcome label."""

    facts: List[ToolFact]
    worker_status: str


@dataclass(frozen=True)
class EvidenceBlock:
    """Final-solver evidence and the exact facts that actually fit."""

    text: str
    included_facts: tuple[ToolFact, ...]

    @property
    def included_count(self) -> int:
        return len(self.included_facts)


@dataclass(frozen=True)
class _WorkerResponse:
    payload: Dict[str, Any] | None
    worker_status: str


@dataclass
class FormalizerProtocol:
    plan_lines: List[str]
    tool_requests: List[ToolRequest]

    def compact_plan(self, max_chars: int = 2400) -> str:
        if not self.plan_lines:
            return "(no usable formalizer plan)"
        text = "\n".join(self.plan_lines)
        return text if len(text) <= max_chars else text[: max_chars - 3].rstrip() + "..."


class SafeAugmentedToolExecutor:
    """Closed-world deterministic tools with one hard-timeout subprocess per batch."""

    def __init__(
        self,
        *,
        extended_tools: bool = False,
        timeout_seconds: float = MAX_BATCH_TIMEOUT_SECONDS,
    ) -> None:
        self.extended_tools = bool(extended_tools)
        self.timeout_seconds = max(0.05, float(timeout_seconds))
        self.last_worker_status = "no_requests"

    def execute(self, requests: Sequence[ToolRequest]) -> List[ToolFact]:
        return self.execute_with_status(requests).facts

    def execute_with_status(self, requests: Sequence[ToolRequest]) -> ToolBatchResult:
        bounded = list(requests)[:MAX_TOOL_REQUESTS]
        if not bounded:
            self.last_worker_status = "no_requests"
            return ToolBatchResult([], self.last_worker_status)
        result = _run_batch_with_status_in_subprocess(
            bounded,
            extended_tools=self.extended_tools,
            timeout_seconds=self.timeout_seconds,
        )
        self.last_worker_status = result.worker_status
        return result

    def execute_one(self, request: ToolRequest) -> ToolFact | None:
        facts = self.execute([request])
        return facts[0] if facts else None


class _InlineToolExecutor:
    """Child-process-only implementation. Never exposed to model-generated code."""

    def __init__(self, extended_tools: bool) -> None:
        self.extended_tools = bool(extended_tools)

    def execute_one(self, request: ToolRequest) -> ToolFact | None:
        allowed = set(_CORE_FAMILIES)
        if self.extended_tools:
            allowed.update(_EXTENDED_FAMILIES)
        if request.family not in allowed:
            return None
        try:
            if request.family == "sympy":
                return self._run_sympy(request.operation, request.arguments)
            if request.family == "matrix":
                return self._run_matrix(request.operation, request.arguments)
            if request.family == "number_theory" and self.extended_tools:
                return self._run_number_theory(request.operation, request.arguments)
            if request.family == "combinatorics" and self.extended_tools:
                return self._run_combinatorics(request.operation, request.arguments)
            if request.family == "recurrence" and self.extended_tools:
                return self._run_recurrence(request.operation, request.arguments)
            if request.family == "graph" and self.extended_tools:
                return self._run_graph(request.operation, request.arguments)
            if request.family == "numerical" and self.extended_tools:
                return self._run_numerical(request.operation, request.arguments)
            if request.family == "probability" and self.extended_tools:
                return self._run_probability(request.operation, request.arguments)
        except Exception:
            return None
        return None

    def _run_sympy(self, operation: str, args: Dict[str, Any]) -> ToolFact | None:
        import sympy as sp

        op = operation.lower()

        if op in {"simplify", "factor", "expand"}:
            raw = _arg(args, "expr")
            expr = _parse_expr(raw)
            value = {
                "simplify": sp.simplify,
                "factor": sp.factor,
                "expand": sp.expand,
            }[op](expr)
            return _fact("sympy", op, f"{op}[expr={_prov(raw)}] -> {_display(value)}")

        if op in {"solve", "roots"}:
            raw_var = args.get("var") or args.get("variable") or "x"
            variable = _symbol(raw_var)
            expression = _equation_expression(args)
            if op == "roots":
                raw_expr = args.get("equation", args.get("expr", ""))
                poly = sp.Poly(expression, variable)
                roots = sp.roots(poly, variable)
                try:
                    multiplicity_total = sum(int(multiplicity) for multiplicity in roots.values())
                except Exception as exc:
                    raise ValueError("root multiplicities were not explicit") from exc
                # ``sympy.roots`` may return an incomplete mapping when it cannot
                # express every root.  An empty mapping is not evidence that a
                # nonconstant polynomial has no roots.
                if multiplicity_total != int(poly.degree()):
                    raise ValueError("roots result is incomplete")
                rendered = "{" + ", ".join(
                    f"{_display(root)}: multiplicity {mult}"
                    for root, mult in roots.items()
                ) + "}"
                return _fact(
                    "sympy",
                    op,
                    f"roots[{_prov(raw_expr)}; var={_prov(raw_var)}] -> {rendered}",
                )
            raw_eq = args.get("equation", args.get("expr", ""))
            domain_name, domain = _sympy_domain(args.get("domain"))
            solution = sp.solveset(expression, variable, domain=domain)
            return _fact(
                "sympy",
                op,
                f"solve[{_prov(raw_eq)}, {_prov(raw_var)} over {domain_name}] -> {_display(solution)}",
            )

        if op == "solve_system":
            equations = _as_sequence(args.get("equations"), max_items=5)
            variables_raw = _as_sequence(args.get("variables"), max_items=5)
            variables = _variable_names(variables_raw, max_items=5)
            if not equations or not variables:
                raise ValueError("solve_system requires equations and variables")
            symbols = [_symbol(name) for name in variables]
            exprs = [_equation_text_to_expr(item) for item in equations]
            solution = sp.solve(exprs, symbols, dict=True, simplify=True)
            if solution == []:
                # ``solve`` uses [] for both no solution and an unresolved
                # system in several cases, so it is not usable exact evidence.
                raise ValueError("solve_system did not return explicit solutions")
            return _fact(
                "sympy",
                op,
                "solve_system["
                f"equations={_compact_json(equations)}; "
                f"variables={_compact_json(variables_raw)}"
                f"] -> {_display(solution)}",
            )

        if op in {"differentiate", "diff"}:
            raw_expr = _arg(args, "expr")
            raw_var = args.get("var") or args.get("variable") or "x"
            expr = _parse_expr(raw_expr)
            variable = _symbol(raw_var)
            order = _bounded_int(args.get("order", 1), 1, 4)
            value = sp.diff(expr, variable, order)
            return _fact(
                "sympy",
                "differentiate",
                f"differentiate[expr={_prov(raw_expr)}; var={_prov(raw_var)}; order={order}] -> {_display(value)}",
            )

        if op == "integrate":
            raw_expr = _arg(args, "expr")
            raw_var = args.get("var") or args.get("variable") or "x"
            expr = _parse_expr(raw_expr)
            variable = _symbol(raw_var)
            if "lower" in args and "upper" in args:
                raw_lower = args["lower"]
                raw_upper = args["upper"]
                lower = _parse_expr(raw_lower)
                upper = _parse_expr(raw_upper)
                value = sp.integrate(expr, (variable, lower, upper))
                return _fact(
                    "sympy",
                    op,
                    f"integral[{_prov(raw_expr)}, {_prov(raw_var)}={_prov(raw_lower)}..{_prov(raw_upper)}] -> {_display(value)}",
                )
            value = sp.integrate(expr, variable)
            return _fact(
                "sympy",
                op,
                f"integral[{_prov(raw_expr)}, d{_prov(raw_var)}] -> {_display(value)}",
            )

        if op == "limit":
            raw_expr = _arg(args, "expr")
            raw_var = args.get("var") or args.get("variable") or "x"
            raw_point = args.get("point", 0)
            expr = _parse_expr(raw_expr)
            variable = _symbol(raw_var)
            point = _parse_expr(raw_point)
            direction = str(args.get("direction") or "+-")
            if direction not in {"+", "-", "+-"}:
                direction = "+-"
            value = sp.limit(expr, variable, point, dir=direction)
            return _fact(
                "sympy",
                op,
                f"limit[{_prov(raw_expr)}; {_prov(raw_var)}->{_prov(raw_point)}; dir={direction}] -> {_display(value)}",
            )

        if op == "series":
            raw_expr = _arg(args, "expr")
            raw_var = args.get("var") or args.get("variable") or "x"
            raw_point = args.get("point", 0)
            order = _bounded_int(args.get("order", 6), 2, 20)
            expr = _parse_expr(raw_expr)
            variable = _symbol(raw_var)
            point = _parse_expr(raw_point)
            value = sp.series(expr, variable, point, order)
            return _fact(
                "sympy",
                op,
                f"series[{_prov(raw_expr)}; var={_prov(raw_var)}; point={_prov(raw_point)}; order={order}] -> {_display(value)}",
            )

        if op == "coefficient" and self.extended_tools:
            raw_expr = _arg(args, "expr")
            raw_var = args.get("var") or args.get("variable") or "x"
            power = _bounded_int(_arg(args, "power"), 0, 120)
            expr = _parse_expr(raw_expr)
            variable = _symbol(raw_var)
            expanded = sp.series(expr, variable, 0, power + 1).removeO().expand()
            value = sp.simplify(expanded.coeff(variable, power))
            return _fact(
                "sympy",
                op,
                f"coefficient[{_prov(raw_var)}^{power} of {_prov(raw_expr)}] -> {_display(value)}",
            )

        if op == "residue":
            raw_expr = _arg(args, "expr")
            raw_var = args.get("var") or args.get("variable") or "z"
            raw_point = args.get("point", 0)
            expr = _parse_expr(raw_expr)
            variable = _symbol(raw_var)
            point = _parse_expr(raw_point)
            value = sp.residue(expr, variable, point)
            return _fact(
                "sympy",
                op,
                f"residue[{_prov(raw_expr)}, {_prov(raw_var)}={_prov(raw_point)}] -> {_display(value)}",
            )

        if op in {"numeric", "evalf"}:
            raw_expr = _arg(args, "expr")
            digits = _bounded_int(args.get("digits", 16), 6, 40)
            expr = _parse_expr(raw_expr)
            value = sp.N(expr, digits)
            return _fact(
                "sympy",
                "numeric",
                f"numeric[expr={_prov(raw_expr)}; digits={digits}] -> {_display(value)}",
            )

        if op == "sum":
            raw_expr = _arg(args, "expr")
            raw_var = args.get("var") or args.get("variable") or "k"
            raw_lower = _arg(args, "lower")
            raw_upper = _arg(args, "upper")
            expr = _parse_expr(raw_expr)
            variable = _symbol(raw_var)
            lower = _parse_expr(raw_lower)
            upper = _parse_expr(raw_upper)
            value = sp.summation(expr, (variable, lower, upper))
            return _fact(
                "sympy",
                op,
                f"sum[{_prov(raw_expr)}; {_prov(raw_var)}={_prov(raw_lower)}..{_prov(raw_upper)}] -> {_display(value)}",
            )

        if op in {"residual", "equivalence"}:
            raw_left = _arg(args, "left")
            raw_right = _arg(args, "right")
            left = _parse_expr(raw_left)
            right = _parse_expr(raw_right)
            residual = sp.simplify(left - right)
            return _fact(
                "sympy",
                "residual",
                f"residual[left={_prov(raw_left)}; right={_prov(raw_right)}] -> {_display(residual)}",
            )

        if op == "mod":
            raw_expr = _arg(args, "expr")
            raw_modulus = _arg(args, "modulus")
            expr = _parse_expr(raw_expr)
            modulus = _bounded_int(raw_modulus, 1, 10**12)
            if not bool(expr.is_integer):
                raise ValueError("mod expression must be an explicit integer")
            value = int(expr) % modulus
            return _fact(
                "sympy",
                op,
                f"mod[expr={_prov(raw_expr)}; modulus={_prov(raw_modulus)}] -> {value}",
            )

        if op == "congruence":
            raw_left = _arg(args, "left")
            raw_right = _arg(args, "right")
            raw_modulus = _arg(args, "modulus")
            left = _parse_expr(raw_left)
            right = _parse_expr(raw_right)
            modulus = _bounded_int(raw_modulus, 1, 10**12)
            difference = sp.simplify(left - right)
            if not bool(difference.is_integer):
                raise ValueError("congruence operands must be explicit integers")
            residual = int(difference) % modulus
            return _fact(
                "sympy",
                op,
                f"congruence[left={_prov(raw_left)}; right={_prov(raw_right)}; modulus={_prov(raw_modulus)}] -> residual {residual}",
            )

        if op == "ode_residual":
            raw_var = args.get("var") or args.get("variable") or "x"
            raw_state = args.get("state") or "y"
            raw_candidate = _arg(args, "candidate")
            raw_rhs = _arg(args, "rhs")
            variable = _symbol(raw_var)
            state = _symbol(raw_state)
            candidate = _parse_expr(raw_candidate)
            rhs = _parse_expr(raw_rhs)
            rhs_sub = rhs.subs(state, candidate)
            residual = sp.simplify(sp.diff(candidate, variable) - rhs_sub)
            return _fact(
                "sympy",
                op,
                "ode_residual["
                f"candidate={_prov(raw_candidate)}; rhs={_prov(raw_rhs)}; "
                f"var={_prov(raw_var)}; state={_prov(raw_state)}"
                f"] -> {_display(residual)}",
            )

        if op == "stationary":
            raw_expr = _arg(args, "expr")
            raw_variables = args.get("variables") or args.get("var") or "x"
            expr = _parse_expr(raw_expr)
            variables = _variable_names(raw_variables, max_items=4)
            symbols = [_symbol(name) for name in variables]
            equations = [sp.diff(expr, symbol) for symbol in symbols]
            solution = sp.solve(equations, symbols, dict=True, simplify=True)
            if solution == []:
                # Do not claim an empty stationary set when the solver may have
                # simply failed to resolve the system.
                raise ValueError("stationary solver did not return explicit solutions")
            return _fact(
                "sympy",
                op,
                f"stationary[{_prov(raw_expr)}; variables={_compact_json(_as_sequence(raw_variables, 4))}] -> {_display(solution)}",
            )

        return None

    def _run_matrix(self, operation: str, args: Dict[str, Any]) -> ToolFact | None:
        import sympy as sp

        op = operation.lower()
        raw_matrix = _arg(args, "matrix")
        matrix = _parse_matrix_safe(raw_matrix)
        matrix_text = _compact_json(raw_matrix)

        if op in {"det", "determinant"}:
            return _fact("matrix", "determinant", f"det({matrix_text}) -> {_display(matrix.det())}")

        if op == "rank":
            return _fact("matrix", op, f"rank({matrix_text}) -> {matrix.rank()}")

        if op == "inverse":
            if matrix.rows != matrix.cols or matrix.det() == 0:
                return _fact("matrix", op, f"inverse({matrix_text}) -> does not exist (singular)")
            return _fact("matrix", op, f"inverse({matrix_text}) -> {_display(matrix.inv())}")

        if op in {"eigen", "eigenvalues"}:
            values = matrix.eigenvals()
            try:
                multiplicity_total = sum(int(multiplicity) for multiplicity in values.values())
            except Exception as exc:
                raise ValueError("eigenvalue multiplicities were not explicit") from exc
            if matrix.rows != matrix.cols or multiplicity_total != matrix.rows:
                raise ValueError("eigenvalue result is incomplete")
            rendered = "{" + ", ".join(
                f"{_display(value)}: multiplicity {mult}"
                for value, mult in values.items()
            ) + "}"
            return _fact("matrix", "eigenvalues", f"eigenvalues({matrix_text}) -> {rendered}")

        if op in {"solve", "linear_solve"}:
            raw_rhs = _arg(args, "rhs")
            rhs = _parse_vector_safe(raw_rhs)
            if matrix.rows != rhs.rows:
                raise ValueError("matrix/rhs dimension mismatch")
            solution = sp.linsolve((matrix, rhs))
            return _fact(
                "matrix",
                "linear_solve",
                f"linear_solve[matrix={matrix_text}; rhs={_compact_json(raw_rhs)}] -> {_display(solution)}",
            )

        if self.extended_tools and op == "nullspace":
            value = matrix.nullspace()
            return _fact("matrix", op, f"nullspace({matrix_text}) -> {_display(value)}")

        if self.extended_tools and op == "charpoly":
            raw_var = args.get("var") or "lam"
            variable = _symbol(raw_var)
            value = matrix.charpoly(variable).as_expr()
            return _fact(
                "matrix",
                op,
                f"charpoly[matrix={matrix_text}; var={_prov(raw_var)}] -> {_display(value)}",
            )

        if self.extended_tools and op == "eigenvectors":
            value = matrix.eigenvects()
            return _fact("matrix", op, f"eigenvectors({matrix_text}) -> {_display(value)}")

        if self.extended_tools and op == "rref":
            rref_matrix, pivots = matrix.rref()
            return _fact(
                "matrix",
                op,
                f"rref({matrix_text}) -> matrix={_display(rref_matrix)}; pivots={pivots}",
            )

        if self.extended_tools and op == "trace":
            return _fact("matrix", op, f"trace({matrix_text}) -> {_display(matrix.trace())}")

        if self.extended_tools and op == "power":
            exponent = _bounded_int(_arg(args, "exponent"), 0, 512)
            if matrix.rows != matrix.cols:
                raise ValueError("matrix power requires a square matrix")
            return _fact(
                "matrix",
                op,
                f"power[matrix={matrix_text}; exponent={exponent}] -> {_display(matrix ** exponent)}",
            )

        return None

    def _run_number_theory(self, operation: str, args: Dict[str, Any]) -> ToolFact | None:
        import sympy as sp
        from sympy.ntheory.modular import crt

        op = operation.lower()

        if op == "gcd":
            a = _bounded_whole(_arg(args, "a"))
            b = _bounded_whole(_arg(args, "b"))
            value = math.gcd(a, b)
            return _fact("number_theory", op, f"gcd[a={a}; b={b}] -> {value}")

        if op == "extended_gcd":
            a = _bounded_whole(_arg(args, "a"))
            b = _bounded_whole(_arg(args, "b"))
            g, x, y = _extended_gcd(a, b)
            return _fact(
                "number_theory",
                op,
                f"extended_gcd[a={a}; b={b}] -> gcd={g}; x={x}; y={y}",
            )

        if op == "mod_inverse":
            a = _bounded_whole(_arg(args, "a"))
            modulus = _bounded_positive(_arg(args, "modulus"))
            value = int(sp.mod_inverse(a, modulus))
            return _fact(
                "number_theory",
                op,
                f"mod_inverse[a={a}; modulus={modulus}] -> {value}",
            )

        if op == "pow_mod":
            base = _bounded_whole(_arg(args, "base"))
            exponent = _bounded_int(_arg(args, "exponent"), 0, 10**12)
            modulus = _bounded_positive(_arg(args, "modulus"))
            value = pow(base, exponent, modulus)
            return _fact(
                "number_theory",
                op,
                f"pow_mod[base={base}; exponent={exponent}; modulus={modulus}] -> {value}",
            )

        if op == "crt":
            residues = [_bounded_whole(v) for v in _as_sequence(_arg(args, "residues"), 8)]
            moduli = [_bounded_positive(v) for v in _as_sequence(_arg(args, "moduli"), 8)]
            if not residues or len(residues) != len(moduli):
                raise ValueError("CRT requires equally sized residue/modulus lists")
            result = crt(moduli, residues)
            if result is None:
                rendered = "no simultaneous solution"
            else:
                value, modulus = result
                rendered = f"x={value} mod {modulus}"
            return _fact(
                "number_theory",
                op,
                f"crt[residues={_compact_json(residues)}; moduli={_compact_json(moduli)}] -> {rendered}",
            )

        if op == "totient":
            n = _bounded_positive(_arg(args, "n"))
            value = int(sp.totient(n))
            return _fact("number_theory", op, f"totient[n={n}] -> {value}")

        if op == "factorint":
            n = _bounded_positive(_arg(args, "n"), limit=10**12)
            value = sp.factorint(n)
            return _fact("number_theory", op, f"factorint[n={n}] -> {_display(value)}")

        if op == "multiplicative_order":
            a = _bounded_whole(_arg(args, "a"), limit=10**9)
            modulus = _bounded_positive(_arg(args, "modulus"), limit=10**9)
            if math.gcd(a, modulus) != 1:
                rendered = "undefined (gcd(a, modulus) != 1)"
            else:
                rendered = str(int(sp.n_order(a, modulus)))
            return _fact(
                "number_theory",
                op,
                f"multiplicative_order[a={a}; modulus={modulus}] -> {rendered}",
            )

        if op == "linear_congruence":
            a = _bounded_whole(_arg(args, "a"), limit=10**9)
            b = _bounded_whole(_arg(args, "b"), limit=10**9)
            modulus = _bounded_positive(_arg(args, "modulus"), limit=10**9)
            g = math.gcd(a, modulus)
            if b % g:
                rendered = "no solution"
            else:
                reduced_a, reduced_b, reduced_m = a // g, b // g, modulus // g
                base = (pow(reduced_a, -1, reduced_m) * reduced_b) % reduced_m if reduced_m > 1 else 0
                rendered = f"x={base} mod {reduced_m}; solution_count={g}"
            return _fact(
                "number_theory",
                op,
                f"linear_congruence[a={a}; b={b}; modulus={modulus}] -> {rendered}",
            )

        if op == "unit_square_solution_count":
            modulus = _bounded_positive(_arg(args, "modulus"), limit=1000000)
            value = sum(1 for x in range(modulus) if (x * x - 1) % modulus == 0)
            return _fact(
                "number_theory",
                op,
                f"unit_square_solution_count[modulus={modulus}] -> {value}",
            )

        if op == "primitive_root":
            modulus = _bounded_positive(_arg(args, "modulus"), limit=10**6)
            value = sp.primitive_root(modulus)
            if value is None:
                raise ValueError("primitive root does not exist")
            return _fact("number_theory", op, f"primitive_root[modulus={modulus}] -> {value}")

        if op == "count_divisible_union":
            upper = _bounded_int(_arg(args, "upper"), 0, 10**9)
            divisors = [_bounded_positive(v, limit=10**6) for v in _as_sequence(_arg(args, "divisors"), 8)]
            if not divisors:
                raise ValueError("divisors cannot be empty")
            value = 0
            for mask in range(1, 1 << len(divisors)):
                lcm = 1
                bits = 0
                for index, divisor in enumerate(divisors):
                    if mask & (1 << index):
                        bits += 1
                        lcm = math.lcm(lcm, divisor)
                term = upper // lcm
                value += term if bits % 2 else -term
            return _fact(
                "number_theory",
                op,
                f"count_divisible_union[upper={upper}; divisors={_compact_json(divisors)}] -> {value}",
            )

        if op == "is_prime":
            n = _bounded_whole(_arg(args, "n"), limit=10**12)
            value = bool(sp.isprime(n)) if n >= 0 else False
            return _fact("number_theory", op, f"is_prime[n={n}] -> {value}")

        if op == "divisor_count":
            n = _bounded_positive(_arg(args, "n"), limit=10**12)
            factors = sp.factorint(n)
            value = math.prod(int(exponent) + 1 for exponent in factors.values())
            return _fact("number_theory", op, f"divisor_count[n={n}] -> {value}")

        if op == "fibonacci_mod":
            n = _bounded_int(_arg(args, "n"), 0, 10**18)
            modulus = _bounded_positive(_arg(args, "modulus"), limit=10**12)

            def fast_doubling(index: int) -> tuple[int, int]:
                if index == 0:
                    return 0, 1
                first, second = fast_doubling(index // 2)
                c = (first * ((2 * second - first) % modulus)) % modulus
                d = (first * first + second * second) % modulus
                return (d, (c + d) % modulus) if index & 1 else (c, d)

            value = fast_doubling(n)[0]
            return _fact(
                "number_theory",
                op,
                f"fibonacci_mod[n={n}; modulus={modulus}] -> {value}",
            )

        if op == "primitive_root_check":
            from sympy.ntheory.residue_ntheory import is_primitive_root

            a = _bounded_whole(_arg(args, "a"), limit=10**9)
            modulus = _bounded_positive(_arg(args, "modulus"), limit=10**9)
            value = bool(is_primitive_root(a, modulus))
            return _fact(
                "number_theory",
                op,
                f"primitive_root_check[a={a}; modulus={modulus}] -> {value}",
            )

        return None

    def _run_combinatorics(self, operation: str, args: Dict[str, Any]) -> ToolFact | None:
        op = operation.lower()

        if op == "factorial":
            n = _bounded_int(_arg(args, "n"), 0, 5000)
            value = math.factorial(n)
            return _fact("combinatorics", op, f"factorial[n={n}] -> {value}")

        if op == "fibonacci":
            n = _bounded_int(_arg(args, "n"), 0, 5000)
            first, second = 0, 1
            for _ in range(n):
                first, second = second, first + second
            return _fact("combinatorics", op, f"fibonacci[n={n}] -> {first}")

        if op == "binomial":
            n = _bounded_int(_arg(args, "n"), 0, 5000)
            k = _bounded_int(_arg(args, "k"), 0, n)
            value = math.comb(n, k)
            return _fact("combinatorics", op, f"binomial[n={n}; k={k}] -> {value}")

        if op == "multinomial":
            parts = [_bounded_int(v, 0, 500) for v in _as_sequence(_arg(args, "parts"), 12)]
            if not parts or sum(parts) > 500:
                raise ValueError("multinomial input too large")
            total = sum(parts)
            value = math.factorial(total)
            for part in parts:
                value //= math.factorial(part)
            return _fact(
                "combinatorics",
                op,
                f"multinomial[parts={_compact_json(parts)}] -> {value}",
            )

        if op == "derangement":
            n = _bounded_int(_arg(args, "n"), 0, 500)
            if n == 0:
                value = 1
            elif n == 1:
                value = 0
            else:
                d0, d1 = 1, 0
                for k in range(2, n + 1):
                    d0, d1 = d1, (k - 1) * (d0 + d1)
                value = d1
            return _fact("combinatorics", op, f"derangement[n={n}] -> {value}")

        if op == "stirling2":
            n = _bounded_int(_arg(args, "n"), 0, 160)
            k = _bounded_int(_arg(args, "k"), 0, n)
            value = _stirling2(n, k)
            return _fact("combinatorics", op, f"stirling2[n={n}; k={k}] -> {value}")

        if op == "catalan":
            n = _bounded_int(_arg(args, "n"), 0, 2000)
            value = math.comb(2 * n, n) // (n + 1)
            return _fact("combinatorics", op, f"catalan[n={n}] -> {value}")

        if op == "stars_bars":
            total = _bounded_int(_arg(args, "total"), 0, 10000)
            variables = _bounded_int(_arg(args, "variables"), 1, 1000)
            minimum_each = _bounded_int(_arg(args, "minimum_each"), 0, 1)
            remaining = total - minimum_each * variables
            value = (
                0
                if remaining < 0
                else math.comb(remaining + variables - 1, variables - 1)
            )
            return _fact(
                "combinatorics",
                op,
                "stars_bars["
                f"total={total}; variables={variables}; minimum_each={minimum_each}"
                f"] -> {value}",
            )

        if op == "onto_functions":
            domain_size = _bounded_int(_arg(args, "domain_size"), 0, 160)
            codomain_size = _bounded_int(_arg(args, "codomain_size"), 0, 160)
            if codomain_size == 0:
                value = 1 if domain_size == 0 else 0
            elif codomain_size > domain_size:
                value = 0
            else:
                value = math.factorial(codomain_size) * _stirling2(domain_size, codomain_size)
            return _fact(
                "combinatorics",
                op,
                f"onto_functions[domain_size={domain_size}; codomain_size={codomain_size}] -> {value}",
            )

        if op == "partition_exact_parts":
            total = _bounded_int(_arg(args, "total"), 0, 500)
            parts = _bounded_int(_arg(args, "parts"), 0, total)
            value = _partition_exact_parts(total, parts)
            return _fact(
                "combinatorics",
                op,
                f"partition_exact_parts[total={total}; parts={parts}] -> {value}",
            )

        if op == "partition_total":
            n = _bounded_int(_arg(args, "n"), 0, 500)
            value = _partition_total(n)
            return _fact("combinatorics", op, f"partition_total[n={n}] -> {value}")

        if op == "distinct_partition":
            n = _bounded_int(_arg(args, "n"), 0, 500)
            value = _distinct_partition_total(n)
            return _fact("combinatorics", op, f"distinct_partition[n={n}] -> {value}")

        if op == "composition_positive":
            total = _bounded_int(_arg(args, "total"), 0, 10000)
            parts = _bounded_int(_arg(args, "parts"), 1, 1000)
            value = 0 if total < parts else math.comb(total - 1, parts - 1)
            return _fact(
                "combinatorics",
                op,
                f"composition_positive[total={total}; parts={parts}] -> {value}",
            )

        if op == "bounded_compositions":
            total = _bounded_int(_arg(args, "total"), 0, 10000)
            parts = _bounded_int(_arg(args, "parts"), 1, 1000)
            lower = _bounded_int(args.get("lower", 0), 0, 10000)
            upper = _bounded_int(args.get("upper", total), lower, 10000)
            value = _bounded_compositions(total, parts, lower, upper)
            return _fact(
                "combinatorics",
                op,
                f"bounded_compositions[total={total}; parts={parts}; lower={lower}; upper={upper}] -> {value}",
            )

        if op == "one_coordinate_lower":
            total = _bounded_int(_arg(args, "total"), 0, 10000)
            variables = _bounded_int(_arg(args, "variables"), 1, 1000)
            lower = _bounded_int(_arg(args, "lower"), 0, total)
            value = 0 if total < lower else math.comb(total - lower + variables - 1, variables - 1)
            return _fact(
                "combinatorics",
                op,
                f"one_coordinate_lower[total={total}; variables={variables}; lower={lower}] -> {value}",
            )

        if op == "one_coordinate_upper":
            total = _bounded_int(_arg(args, "total"), 0, 10000)
            variables = _bounded_int(_arg(args, "variables"), 1, 1000)
            upper = _bounded_int(_arg(args, "upper"), 0, total)
            unrestricted = math.comb(total + variables - 1, variables - 1)
            violating = 0 if total <= upper else math.comb(total - upper - 1 + variables - 1, variables - 1)
            return _fact(
                "combinatorics",
                op,
                f"one_coordinate_upper[total={total}; variables={variables}; upper={upper}] -> {unrestricted - violating}",
            )

        if op == "binary_no_adjacent":
            length = _bounded_int(_arg(args, "length"), 0, 10000)
            ones = _bounded_int(_arg(args, "ones"), 0, length)
            value = 0 if ones and length - ones + 1 < ones else math.comb(length - ones + 1, ones)
            return _fact(
                "combinatorics",
                op,
                f"binary_no_adjacent[length={length}; ones={ones}] -> {value}",
            )

        if op == "circular_adjacent_block":
            people = _bounded_int(_arg(args, "people"), 2, 500)
            value = 2 * math.factorial(people - 2)
            return _fact("combinatorics", op, f"circular_adjacent_block[people={people}] -> {value}")

        if op == "choose_excluding_pair":
            total = _bounded_int(_arg(args, "total"), 0, 5000)
            choose = _bounded_int(_arg(args, "choose"), 0, total)
            forbidden = math.comb(total - 2, choose - 2) if choose >= 2 and total >= 2 else 0
            value = math.comb(total, choose) - forbidden
            return _fact(
                "combinatorics",
                op,
                f"choose_excluding_pair[total={total}; choose={choose}] -> {value}",
            )

        if op == "multiset_permutations":
            counts = [_bounded_int(v, 0, 500) for v in _as_sequence(_arg(args, "counts"), 32)]
            total = sum(counts)
            if not counts or total > 500:
                raise ValueError("multiset input too large")
            value = math.factorial(total)
            for count in counts:
                value //= math.factorial(count)
            return _fact(
                "combinatorics",
                op,
                f"multiset_permutations[counts={_compact_json(counts)}] -> {value}",
            )

        if op == "triangulations":
            vertices = _bounded_int(_arg(args, "vertices"), 3, 500)
            value = math.comb(2 * (vertices - 2), vertices - 2) // (vertices - 1)
            return _fact("combinatorics", op, f"triangulations[vertices={vertices}] -> {value}")

        if op == "full_parenthesizations":
            factors = _bounded_int(_arg(args, "factors"), 1, 500)
            n = factors - 1
            value = math.comb(2 * n, n) // (n + 1)
            return _fact("combinatorics", op, f"full_parenthesizations[factors={factors}] -> {value}")

        if op == "ballot_diagonal_paths":
            steps = _bounded_int(_arg(args, "steps"), 0, 500)
            value = math.comb(2 * steps, steps) // (steps + 1)
            return _fact("combinatorics", op, f"ballot_diagonal_paths[steps={steps}] -> {value}")

        if op == "tilings_parts":
            total = _bounded_int(_arg(args, "total"), 0, 10000)
            pieces = sorted({_bounded_int(v, 1, 100) for v in _as_sequence(_arg(args, "pieces"), 8)})
            if not pieces:
                raise ValueError("piece lengths cannot be empty")
            values = [0] * (total + 1)
            values[0] = 1
            for amount in range(1, total + 1):
                values[amount] = sum(values[amount - piece] for piece in pieces if piece <= amount)
            return _fact("combinatorics", op, f"tilings_parts[total={total}; pieces={_compact_json(pieces)}] -> {values[total]}")

        if op == "steps_no_consecutive_two":
            total = _bounded_int(_arg(args, "total"), 0, 10000)
            # State is (remaining length, whether the previous step was 2).
            free, after_two = 1, 0
            for _ in range(total):
                free, after_two = free + after_two, free
            return _fact("combinatorics", op, f"steps_no_consecutive_two[total={total}] -> {free + after_two}")

        return None

    def _run_recurrence(self, operation: str, args: Dict[str, Any]) -> ToolFact | None:
        import sympy as sp

        if operation.lower() != "linear_eval":
            return None
        raw_initial = _as_sequence(_arg(args, "initial"), 8)
        raw_coefficients = _as_sequence(_arg(args, "coefficients"), 8)
        if not raw_initial or len(raw_initial) != len(raw_coefficients):
            raise ValueError("initial and coefficients must have same nonzero length")
        if len(raw_initial) > 8:
            raise ValueError("recurrence order exceeds 8")
        target_n = _bounded_int(_arg(args, "target_n"), 0, 2000)
        initial_start = _bounded_int(args.get("initial_start", 0), 0, 2000)
        sequence_symbol = str(args.get("sequence_symbol", "a") or "a")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,8}", sequence_symbol):
            raise ValueError("invalid recurrence symbol")
        initial = [_parse_expr(value) for value in raw_initial]
        coefficients = [_parse_expr(value) for value in raw_coefficients]
        constant = _parse_expr(args.get("constant", 0))
        forcing_raw = args.get("forcing")
        forcing_expr = None
        n_symbol = None
        if forcing_raw not in (None, ""):
            n_symbol = sp.Symbol("n")
            forcing_expr = _parse_expr(forcing_raw)
            if not forcing_expr.free_symbols.issubset({n_symbol}):
                raise ValueError("forcing term must depend only on n")
        if target_n < initial_start:
            raise ValueError("target precedes initial recurrence index")
        if target_n < initial_start + len(initial):
            value = initial[target_n - initial_start]
        else:
            terms = list(initial)
            order = len(initial)
            for _n in range(initial_start + order, target_n + 1):
                next_value = constant
                if forcing_expr is not None and n_symbol is not None:
                    next_value = forcing_expr.subs(n_symbol, _n)
                for j, coefficient in enumerate(coefficients):
                    next_value += coefficient * terms[-1 - j]
                terms.append(sp.simplify(next_value))
            value = terms[target_n - initial_start]
        recurrence_terms: List[str] = []
        coefficient_order: List[str] = []
        for index, coefficient in enumerate(coefficients, start=1):
            is_negative = bool(coefficient.could_extract_minus_sign())
            magnitude = -coefficient if is_negative else coefficient
            lag = f"{sequence_symbol}_{{n-{index}}}"
            body = lag if magnitude == 1 else f"{_display(magnitude)}*{lag}"
            if not recurrence_terms:
                recurrence_terms.append(f"-{body}" if is_negative else body)
            else:
                recurrence_terms.append(f"-{body}" if is_negative else f"+{body}")
            coefficient_order.append(f"c_{index}->{lag}")
        constant_negative = bool(constant.could_extract_minus_sign())
        constant_magnitude = -constant if constant_negative else constant
        recurrence_terms.append(
            f"-{_display(constant_magnitude)}"
            if constant_negative
            else f"+{_display(constant_magnitude)}"
        )
        recurrence_text = f"{sequence_symbol}_n=" + "".join(recurrence_terms)
        initial_bindings = ",".join(
            f"{sequence_symbol}_{index}={_prov(raw_value)}"
            for index, raw_value in enumerate(raw_initial)
        )
        return _fact(
            "recurrence",
            "linear_eval",
            "linear_eval["
            f"{recurrence_text}; initial={_compact_json(raw_initial)} as {initial_bindings}; "
            f"coefficients={_compact_json(raw_coefficients)}; "
            f"coefficient_order={','.join(coefficient_order)}; "
            f"initial_start={initial_start}; "
            f"constant={_prov(args.get('constant', 0))}; "
            f"forcing={_prov(forcing_raw) if forcing_raw not in (None, '') else 'none'}; target_n={target_n}"
            f"] -> {_display(value)}",
        )

    def _run_graph(self, operation: str, args: Dict[str, Any]) -> ToolFact | None:
        op = operation.lower()

        if op == "edge_count_from_degrees":
            degrees = _degree_list(_arg(args, "degrees"))
            total = sum(degrees)
            if total % 2:
                rendered = "invalid degree sum (odd)"
            else:
                rendered = str(total // 2)
            return _fact(
                "graph",
                op,
                f"edge_count_from_degrees[degrees={_compact_json(degrees)}] -> {rendered}",
            )

        if op == "tree_edge_count":
            vertices = _bounded_int(_arg(args, "vertices"), 1, 10**9)
            return _fact("graph", op, f"tree_edge_count[vertices={vertices}] -> {vertices - 1}")

        if op == "complete_graph_edges":
            n = _bounded_int(_arg(args, "n"), 0, 10**9)
            value = n * (n - 1) // 2
            return _fact("graph", op, f"complete_graph_edges[n={n}] -> {value}")

        if op == "complete_bipartite_edges":
            m = _bounded_int(_arg(args, "m"), 0, 10**9)
            n = _bounded_int(_arg(args, "n"), 0, 10**9)
            return _fact(
                "graph",
                op,
                f"complete_bipartite_edges[m={m}; n={n}] -> {m * n}",
            )

        if op == "cycle_space_dimension":
            vertices = _bounded_int(_arg(args, "vertices"), 0, 10**9)
            edges = _bounded_int(_arg(args, "edges"), 0, 10**12)
            components = _bounded_int(args.get("components", 1), 0, max(vertices, 1))
            value = edges - vertices + components
            return _fact(
                "graph",
                op,
                f"cycle_space_dimension[vertices={vertices}; edges={edges}; components={components}] -> {value}",
            )

        if op == "eulerian_degree_check":
            degrees = _degree_list(_arg(args, "degrees"))
            all_even = all(degree % 2 == 0 for degree in degrees)
            if "connected" in args:
                connected = _strict_bool(args["connected"])
                result = bool(connected and all_even)
                request_text = (
                    f"degrees={_compact_json(degrees)}; "
                    f"connected={_compact_json(connected)}"
                )
                rendered = f"eulerian_circuit_criterion={result}; derived_all_even={all_even}"
            else:
                request_text = f"degrees={_compact_json(degrees)}; connected=not supplied"
                rendered = f"all_even={all_even}; connectivity not supplied"
            return _fact(
                "graph",
                op,
                f"eulerian_degree_check[{request_text}] -> {rendered}",
            )

        if op == "degree_sequence_graphical":
            degrees = _degree_list(_arg(args, "degrees"))
            result = _havel_hakimi(degrees)
            return _fact(
                "graph",
                op,
                f"degree_sequence_graphical[degrees={_compact_json(degrees)}] -> {result}",
            )

        if op == "cycle_chromatic":
            vertices = _bounded_int(_arg(args, "n"), 1, 10**9)
            value = 1 if vertices == 1 else 2 if vertices % 2 == 0 else 3
            return _fact("graph", op, f"cycle_chromatic[n={vertices}] -> {value}")

        if op == "complete_graph_chromatic":
            vertices = _bounded_int(_arg(args, "n"), 0, 10**9)
            return _fact("graph", op, f"complete_graph_chromatic[n={vertices}] -> {vertices}")

        if op == "complete_bipartite_chromatic":
            m = _bounded_int(_arg(args, "m"), 0, 10**9)
            n = _bounded_int(_arg(args, "n"), 0, 10**9)
            value = 2 if m > 0 and n > 0 else 0
            return _fact("graph", op, f"complete_bipartite_chromatic[m={m}; n={n}] -> {value}")

        if op == "path_matching":
            vertices = _bounded_int(_arg(args, "vertices"), 0, 10**9)
            return _fact("graph", op, f"path_matching[vertices={vertices}] -> {vertices // 2}")

        if op == "cycle_matching":
            vertices = _bounded_int(_arg(args, "vertices"), 0, 10**9)
            return _fact("graph", op, f"cycle_matching[vertices={vertices}] -> {vertices // 2}")

        if op == "planar_faces":
            vertices = _bounded_int(_arg(args, "vertices"), 0, 10**9)
            edges = _bounded_int(_arg(args, "edges"), 0, 10**12)
            components = _bounded_int(args.get("components", 1), 1, max(vertices, 1))
            value = edges - vertices + components + 1
            return _fact(
                "graph",
                op,
                f"planar_faces[vertices={vertices}; edges={edges}; components={components}] -> {value}",
            )

        if op == "turan_edges":
            vertices = _bounded_int(_arg(args, "vertices"), 0, 10**6)
            forbidden_clique = _bounded_int(_arg(args, "forbidden_clique"), 2, 1000)
            if forbidden_clique <= 2:
                value = 0
            else:
                parts = forbidden_clique - 1
                q, r = divmod(vertices, parts)
                sum_squares = r * (q + 1) ** 2 + (parts - r) * q**2
                value = (vertices**2 - sum_squares) // 2
            return _fact(
                "graph",
                op,
                f"turan_edges[vertices={vertices}; forbidden_clique={forbidden_clique}] -> {value}",
            )

        if op == "tree_remaining_degree":
            vertices = _bounded_int(_arg(args, "vertices"), 1, 10**6)
            leaves = _bounded_int(_arg(args, "leaves"), 0, vertices)
            degree_two = _bounded_int(_arg(args, "degree_two"), 0, vertices - leaves)
            value = 2 * (vertices - 1) - leaves - 2 * degree_two
            return _fact(
                "graph",
                op,
                f"tree_remaining_degree[vertices={vertices}; leaves={leaves}; degree_two={degree_two}] -> {value}",
            )

        if op == "connected_edge_guarantee":
            vertices = _bounded_int(_arg(args, "vertices"), 1, 10**6)
            value = math.comb(vertices - 1, 2) + 1 if vertices >= 2 else 0
            return _fact("graph", op, f"connected_edge_guarantee[vertices={vertices}] -> {value}")

        if op == "euler_trail_possible":
            odd_vertices = _bounded_int(_arg(args, "odd_vertices"), 0, 10**6)
            connected = _strict_bool(_arg(args, "connected"))
            value = connected and odd_vertices in {0, 2}
            return _fact("graph", op, f"euler_trail_possible[connected={connected}; odd_vertices={odd_vertices}] -> {value}")

        if op == "euler_circuit_possible":
            odd_vertices = _bounded_int(_arg(args, "odd_vertices"), 0, 10**6)
            connected = _strict_bool(_arg(args, "connected"))
            value = connected and odd_vertices == 0
            return _fact("graph", op, f"euler_circuit_possible[connected={connected}; odd_vertices={odd_vertices}] -> {value}")

        if op == "complete_bipartite_hamiltonian":
            m = _bounded_int(_arg(args, "m"), 0, 10**6)
            n = _bounded_int(_arg(args, "n"), 0, 10**6)
            value = m == n and m >= 2
            return _fact("graph", op, f"complete_bipartite_hamiltonian[m={m}; n={n}] -> {value}")

        if op == "regular_graph_edges":
            vertices = _bounded_int(_arg(args, "vertices"), 0, 10**6)
            degree = _bounded_int(_arg(args, "degree"), 0, max(vertices - 1, 0))
            total = vertices * degree
            rendered = str(total // 2) if total % 2 == 0 else "impossible (odd degree sum)"
            return _fact("graph", op, f"regular_graph_edges[vertices={vertices}; degree={degree}] -> {rendered}")

        if op == "hall_matching_size":
            left_vertices = _bounded_int(_arg(args, "left_vertices"), 0, 10**6)
            return _fact("graph", op, f"hall_matching_size[left_vertices={left_vertices}] -> {left_vertices}")

        if op == "triangulated_planar_edges":
            vertices = _bounded_int(_arg(args, "vertices"), 3, 10**6)
            return _fact("graph", op, f"triangulated_planar_edges[vertices={vertices}] -> {3 * (vertices - 2)}")

        if op == "tree_cut_edges":
            components = _bounded_int(_arg(args, "components"), 1, 10**6)
            return _fact("graph", op, f"tree_cut_edges[components={components}] -> {components - 1}")

        return None

    def _run_numerical(self, operation: str, args: Dict[str, Any]) -> ToolFact | None:
        import sympy as sp

        op = operation.lower()

        if op == "newton_step":
            raw_f = _arg(args, "f")
            raw_var = args.get("variable") or args.get("var") or "x"
            raw_x0 = _arg(args, "x0")
            variable = _symbol(raw_var)
            f = _parse_expr(raw_f)
            x0 = _parse_expr(raw_x0)
            derivative = sp.diff(f, variable)
            denom = sp.simplify(derivative.subs(variable, x0))
            if denom == 0:
                rendered = "undefined (derivative is zero at x0)"
            else:
                value = sp.simplify(x0 - f.subs(variable, x0) / denom)
                rendered = _display(value)
            return _fact(
                "numerical",
                op,
                f"newton_step[f={_prov(raw_f)}; variable={_prov(raw_var)}; x0={_prov(raw_x0)}] -> {rendered}",
            )

        if op == "interpolate_eval":
            raw_points = _arg(args, "points")
            raw_eval = _arg(args, "evaluate_at")
            points = _parse_points(raw_points)
            if len(points) > 12:
                raise ValueError("too many interpolation points")
            x = sp.Symbol("x")
            polynomial = sp.interpolate(points, x)
            evaluate_at = _parse_expr(raw_eval)
            value = sp.simplify(polynomial.subs(x, evaluate_at))
            return _fact(
                "numerical",
                op,
                f"interpolate_eval[points={_compact_json(raw_points)}; evaluate_at={_prov(raw_eval)}] -> {value}",
            )

        return None

    def _run_probability(self, operation: str, args: Dict[str, Any]) -> ToolFact | None:
        """Exact closed-world distribution facts for explicit finite inputs."""

        import sympy as sp

        op = operation.lower()
        if op == "binomial_pmf":
            trials = _bounded_int(_arg(args, "trials"), 0, 500)
            successes = _bounded_int(_arg(args, "successes"), 0, trials)
            raw_p = _arg(args, "p")
            p = _bounded_probability(raw_p)
            value = sp.simplify(sp.binomial(trials, successes) * p**successes * (1 - p) ** (trials - successes))
            return _fact(
                "probability",
                op,
                f"binomial_pmf[trials={trials}; successes={successes}; p={_prov(raw_p)}] -> {_display(value)}",
            )

        if op == "poisson_pmf":
            raw_rate = _arg(args, "rate")
            rate = _bounded_nonnegative_number(raw_rate)
            k = _bounded_int(_arg(args, "k"), 0, 500)
            value = sp.simplify(sp.exp(-rate) * rate**k / sp.factorial(k))
            return _fact(
                "probability",
                op,
                f"poisson_pmf[rate={_prov(raw_rate)}; k={k}] -> {_display(value)}",
            )

        if op == "hypergeometric_pmf":
            population = _bounded_int(_arg(args, "population"), 1, 500)
            success_states = _bounded_int(_arg(args, "success_states"), 0, population)
            draws = _bounded_int(_arg(args, "draws"), 0, population)
            successes = _bounded_int(_arg(args, "successes"), 0, draws)
            if successes > success_states or draws - successes > population - success_states:
                value = 0
            else:
                value = sp.Rational(
                    math.comb(success_states, successes)
                    * math.comb(population - success_states, draws - successes),
                    math.comb(population, draws),
                )
            return _fact(
                "probability",
                op,
                "hypergeometric_pmf["
                f"population={population}; success_states={success_states}; draws={draws}; successes={successes}"
                f"] -> {_display(value)}",
            )

        if op == "geometric_pmf":
            raw_p = _arg(args, "success")
            p = _bounded_probability(raw_p)
            trials = _bounded_int(_arg(args, "trials"), 1, 500)
            value = sp.simplify((1 - p) ** (trials - 1) * p)
            return _fact(
                "probability",
                op,
                f"geometric_pmf[success={_prov(raw_p)}; trials={trials}] -> {_display(value)}",
            )

        if op == "negative_binomial_pmf":
            raw_p = args.get("success_probability")
            if raw_p in (None, ""):
                raw_p = args.get("p")
            if raw_p in (None, ""):
                raise ValueError("missing argument: success_probability or p")
            p = _bounded_probability(raw_p)
            successes = _bounded_int(_arg(args, "successes"), 1, 500)
            trials = _bounded_int(_arg(args, "trials"), successes, 500)
            value = sp.simplify(
                sp.binomial(trials - 1, successes - 1)
                * p ** successes
                * (1 - p) ** (trials - successes)
            )
            return _fact(
                "probability",
                op,
                "negative_binomial_pmf["
                f"successes={successes}; trials={trials}; p={_prov(raw_p)}"
                f"] -> {_display(value)}",
            )

        if op == "binomial_cdf":
            trials = _bounded_int(_arg(args, "trials"), 0, 500)
            upper = _bounded_int(_arg(args, "upper"), 0, trials)
            raw_p = _arg(args, "p")
            p = _bounded_probability(raw_p)
            value = sp.simplify(
                sum(
                    sp.binomial(trials, successes)
                    * p ** successes
                    * (1 - p) ** (trials - successes)
                    for successes in range(upper + 1)
                )
            )
            return _fact(
                "probability",
                op,
                f"binomial_cdf[trials={trials}; p={_prov(raw_p)}; upper={upper}] -> {_display(value)}",
            )

        return None


def parse_formalizer_protocol(
    text: str,
    *,
    extended_tools: bool = False,
) -> FormalizerProtocol:
    raw = str(text or "")[:MAX_PROTOCOL_CHARS]
    plan_lines: List[str] = []
    requests: List[ToolRequest] = []
    for line in raw.splitlines()[:MAX_PROTOCOL_LINES]:
        clean = line.strip()
        if not clean:
            continue
        key, sep, payload = clean.partition(":")
        if not sep:
            continue
        key_upper = key.strip().upper()
        payload = payload.strip()
        if key_upper == "TOOL":
            request = _parse_tool_request(payload, extended_tools=extended_tools)
            if request is not None and len(requests) < MAX_TOOL_REQUESTS:
                requests.append(request)
            continue
        if key_upper in _ALLOWED_PLAN_KEYS and payload:
            plan_lines.append(f"{key_upper}: {payload[:450]}")
    return FormalizerProtocol(plan_lines=plan_lines, tool_requests=requests)


def build_evidence_block(
    facts: Sequence[ToolFact],
    max_chars: int = MAX_EVIDENCE_CHARS,
    *,
    return_metadata: bool = False,
) -> Any:
    """Build final-solver evidence from complete facts only.

    The public default remains a string for existing callers.  V3 telemetry can
    request metadata to distinguish successful worker output from facts that
    were actually small enough to inject.
    """
    result = _build_evidence_block(facts, max_chars=max_chars)
    return result if return_metadata else result.text


def build_evidence_block_with_metadata(
    facts: Sequence[ToolFact],
    max_chars: int = MAX_EVIDENCE_CHARS,
) -> EvidenceBlock:
    """Return evidence together with the exact facts included in it."""

    return _build_evidence_block(facts, max_chars=max_chars)


def _build_evidence_block(
    facts: Sequence[ToolFact],
    *,
    max_chars: int,
) -> EvidenceBlock:
    if not facts:
        return EvidenceBlock("(no deterministic tool facts)", ())
    lines = ["Deterministically computed facts from the formalizer's exact requests:"]
    included: List[ToolFact] = []
    for fact in facts:
        if not _fact_statement_is_complete(fact.statement):
            continue
        line = f"- {fact.statement}"
        if len("\n".join(lines + [line])) > max_chars:
            # An oversized fact must not hide a later short fact.
            continue
        lines.append(line)
        included.append(fact)
    if not included:
        return EvidenceBlock("(no deterministic tool facts)", ())
    return EvidenceBlock("\n".join(lines), tuple(included))


def _run_batch_in_subprocess(
    requests: Sequence[ToolRequest],
    *,
    extended_tools: bool,
    timeout_seconds: float,
    _test_sleep_seconds: float | None = None,
) -> List[ToolFact]:
    """Execute one closed-world batch through the fixed repository worker."""

    return _run_batch_with_status_in_subprocess(
        requests,
        extended_tools=extended_tools,
        timeout_seconds=timeout_seconds,
        _test_sleep_seconds=_test_sleep_seconds,
    ).facts


def _run_batch_with_status_in_subprocess(
    requests: Sequence[ToolRequest],
    *,
    extended_tools: bool,
    timeout_seconds: float,
    _test_sleep_seconds: float | None = None,
) -> ToolBatchResult:
    """Execute a batch and retain only a bounded, sanitized outcome label."""

    serialized = [
        {
            "family": request.family,
            "operation": request.operation,
            "arguments": request.arguments,
            "source": request.source,
        }
        for request in list(requests)[:MAX_TOOL_REQUESTS]
    ]
    if not serialized:
        return ToolBatchResult([], "no_requests")
    worker_request: Dict[str, Any] = {
        "mode": "execute_tools",
        "extended_tools": bool(extended_tools),
        "requests": serialized,
    }
    if _test_sleep_seconds is not None:
        worker_request["_test_sleep_seconds"] = float(_test_sleep_seconds)
    response = _run_fixed_worker(worker_request, timeout_seconds=timeout_seconds)
    if response.worker_status != "completed":
        return ToolBatchResult([], response.worker_status)
    payload = response.payload
    if not isinstance(payload, dict) or payload.get("status") != "ok":
        return ToolBatchResult([], "worker_error")
    raw_facts = payload.get("facts")
    if not isinstance(raw_facts, list):
        return ToolBatchResult([], "worker_error")
    facts: List[ToolFact] = []
    for item in raw_facts[:MAX_TOOL_REQUESTS]:
        if not isinstance(item, dict):
            continue
        statement = str(item.get("statement") or "")
        if _fact_statement_is_complete(statement):
            facts.append(
                ToolFact(
                    family=str(item.get("family") or "")[:40],
                    operation=str(item.get("operation") or "")[:40],
                    statement=statement,
                    source=str(item.get("source") or "")[:40],
                )
            )
    worker_status = "completed" if len(facts) == len(serialized) else "partial_success"
    return ToolBatchResult(facts, worker_status)


def run_answer_verification_in_subprocess(
    final_response: str,
    response_mode: str,
    *,
    requests: Sequence[Dict[str, Any]] = (),
    timeout_seconds: float = MAX_VERIFY_TIMEOUT_SECONDS,
    _test_sleep_seconds: float | None = None,
) -> List[Dict[str, Any]]:
    """Run exact object-grounded checks in a killable fixed worker.

    ``requests`` are constructed by the parent from grounded request spans.
    The old type-only ``targets`` API is intentionally gone: accepting a type
    without its object would reintroduce first-equation/first-matrix bugs.
    """

    accepted_requests = _normalize_verification_requests(requests)
    if response_mode != "ANSWER_VALUE" or not accepted_requests:
        return []
    if not isinstance(final_response, str):
        return []
    if len(final_response) > MAX_VERIFY_RESPONSE_CHARS:
        return []
    request: Dict[str, Any] = {
        "mode": "verify_answer",
        "final_response": final_response,
        "response_mode": response_mode,
        "requests": accepted_requests,
    }
    if _test_sleep_seconds is not None:
        request["_test_sleep_seconds"] = float(_test_sleep_seconds)
    worker_response = _run_fixed_worker(request, timeout_seconds=timeout_seconds)
    if worker_response.worker_status != "completed":
        return []
    payload = worker_response.payload
    if not isinstance(payload, dict) or payload.get("status") != "ok":
        return []
    raw_evidence = payload.get("evidence")
    if not isinstance(raw_evidence, list):
        return []

    evidence: List[Dict[str, Any]] = []
    for item in raw_evidence[:8]:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or "inconclusive")
        if status not in {"pass", "fail", "inconclusive"}:
            status = "inconclusive"
        claim_scope = str(item.get("claim_scope") or "subclaim")
        if claim_scope not in {"full_answer", "subclaim"}:
            claim_scope = "subclaim"
        residual = item.get("residual")
        evidence.append(
            {
                "status": status,
                "method": str(item.get("method") or "deterministic_check")[:120],
                "details": str(item.get("details") or "")[:500],
                "residual": None if residual in {None, ""} else str(residual)[:240],
                "is_decisive": bool(item.get("is_decisive", False)),
                "claim_scope": claim_scope,
            }
        )
    return evidence


def _run_fixed_worker(
    request: Dict[str, Any],
    *,
    timeout_seconds: float,
) -> _WorkerResponse:
    """Exchange one JSON request with the fixed module under one deadline."""

    deadline = time.monotonic() + max(0.05, float(timeout_seconds))
    try:
        serialized = json.dumps(
            request,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    except Exception:
        return _WorkerResponse(None, "worker_error")
    if len(serialized) > MAX_WORKER_INPUT_CHARS:
        return _WorkerResponse(None, "worker_error")

    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(
            list(_WORKER_COMMAND),
            cwd=str(_REPOSITORY_ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
        )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _terminate_worker(process)
            return _WorkerResponse(None, "timeout")
        stdout, _stderr = process.communicate(input=serialized, timeout=remaining)
        if time.monotonic() > deadline or process.returncode != 0:
            return _WorkerResponse(None, "timeout" if time.monotonic() > deadline else "worker_error")
        if len(stdout) > MAX_WORKER_OUTPUT_CHARS:
            return _WorkerResponse(None, "worker_error")
        payload = json.loads(stdout)
        return _WorkerResponse(
            payload if isinstance(payload, dict) else None,
            "completed" if isinstance(payload, dict) else "worker_error",
        )
    except subprocess.TimeoutExpired:
        if process is not None:
            _terminate_worker(process)
        return _WorkerResponse(None, "timeout")
    except Exception:
        if process is not None and process.poll() is None:
            _terminate_worker(process)
        return _WorkerResponse(None, "worker_error")
    finally:
        if process is not None:
            if process.poll() is None:
                _terminate_worker(process)
            for stream in (process.stdin, process.stdout, process.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except Exception:
                    pass


def _terminate_worker(process: subprocess.Popen[str]) -> None:
    """Reap a timed-out worker; escalate from terminate to kill."""

    if process.poll() is not None:
        return
    try:
        process.terminate()
    except Exception:
        pass
    try:
        process.wait(timeout=0.2)
        return
    except Exception:
        pass
    try:
        process.kill()
    except Exception:
        pass
    try:
        process.wait(timeout=0.2)
    except Exception:
        pass


def _parse_tool_request(payload: str, *, extended_tools: bool) -> ToolRequest | None:
    if not payload or len(payload) > 2200:
        return None
    parts = [part.strip() for part in payload.split("|")]
    if len(parts) < 2:
        return None
    family = parts[0].lower()
    operation = parts[1].lower()
    allowed = set(_CORE_FAMILIES)
    if extended_tools:
        allowed.update(_EXTENDED_FAMILIES)
    if family not in allowed or not re.fullmatch(r"[a-z_]{2,32}", operation):
        return None

    core_sympy = {
        "simplify", "factor", "expand", "solve", "roots", "solve_system",
        "differentiate", "diff", "integrate", "limit", "series", "residue",
        "numeric", "evalf", "sum", "residual", "equivalence", "mod",
        "congruence", "ode_residual", "stationary",
    }
    extended_sympy = {"coefficient"}
    core_matrix = {"det", "determinant", "rank", "inverse", "eigen", "eigenvalues", "solve", "linear_solve"}
    extended_matrix = {"nullspace", "charpoly", "eigenvectors", "rref", "trace", "power"}
    family_operations = {
        "sympy": core_sympy | (extended_sympy if extended_tools else set()),
        "matrix": core_matrix | (extended_matrix if extended_tools else set()),
        "number_theory": {
            "gcd", "extended_gcd", "mod_inverse", "pow_mod", "crt", "totient", "factorint",
            "multiplicative_order", "primitive_root_check", "linear_congruence",
            "unit_square_solution_count", "primitive_root", "count_divisible_union",
            "is_prime", "divisor_count", "fibonacci_mod",
        },
        "combinatorics": {
            "binomial", "multinomial", "derangement", "stirling2", "catalan", "stars_bars",
            "onto_functions", "partition_exact_parts", "partition_total", "distinct_partition",
            "composition_positive", "bounded_compositions", "binary_no_adjacent",
            "one_coordinate_lower", "one_coordinate_upper",
            "circular_adjacent_block", "choose_excluding_pair", "multiset_permutations",
            "triangulations", "full_parenthesizations", "ballot_diagonal_paths",
            "tilings_parts", "steps_no_consecutive_two", "factorial", "fibonacci",
        },
        "recurrence": {"linear_eval"},
        "graph": {
            "edge_count_from_degrees", "tree_edge_count", "complete_graph_edges",
            "complete_bipartite_edges", "cycle_space_dimension",
            "eulerian_degree_check", "degree_sequence_graphical",
            "cycle_chromatic", "complete_graph_chromatic", "complete_bipartite_chromatic",
            "path_matching", "cycle_matching", "planar_faces", "turan_edges",
            "tree_remaining_degree", "connected_edge_guarantee", "euler_trail_possible",
            "euler_circuit_possible", "complete_bipartite_hamiltonian", "hall_matching_size",
            "triangulated_planar_edges", "tree_cut_edges", "regular_graph_edges",
        },
        "numerical": {"newton_step", "interpolate_eval"},
        "probability": {
            "binomial_pmf", "poisson_pmf", "hypergeometric_pmf",
            "geometric_pmf", "negative_binomial_pmf", "binomial_cdf",
        },
    }
    if operation not in family_operations.get(family, set()):
        return None

    args: Dict[str, Any] = {}
    for part in parts[2:]:
        key, sep, value = part.partition("=")
        key = key.strip()
        value = value.strip()
        if not sep or not _ALLOWED_ARG_KEY.fullmatch(key) or len(value) > MAX_TOOL_ARG_CHARS:
            return None
        args[key] = _parse_arg_value(value)

    return ToolRequest(
        family=family,
        operation=operation,
        arguments=args,
        source=payload[:700],
    )


def _parse_arg_value(value: str) -> Any:
    text = str(value or "").strip()
    if not text:
        return ""
    if (
        text[0] in "[{\"'"
        or text.lower() in {"true", "false", "null"}
        or re.fullmatch(r"[+\-]?\d+(?:\.\d+)?", text)
    ):
        try:
            return json.loads(text)
        except Exception:
            try:
                return ast.literal_eval(text)
            except Exception:
                return text
    return text


def _normalize_math_parameter_text(value: Any) -> str:
    text = str(value)
    for source, target in _PARAMETER_NORMALIZATIONS.items():
        text = text.replace(source, target)
    text = re.sub(r"\blambda\b", "lam", text)
    return text


def _parse_expr(value: Any):
    import sympy as sp

    if isinstance(value, (int, float)):
        value = str(value)
    normalized = _normalize_math_parameter_text(value)
    protected = normalized
    replacements: Dict[Any, Any] = {}
    for name in _PROTECTED_PARAMETER_NAMES:
        placeholder = f"ma_param_{name}"
        pattern = rf"\b{re.escape(name)}\b"
        if re.search(pattern, protected):
            protected = re.sub(pattern, placeholder, protected)
            replacements[sp.Symbol(placeholder)] = sp.Symbol(name)
    expr = _safe_parse_expr(protected)
    if replacements:
        expr = expr.xreplace(replacements)
    return expr


def _arg(args: Dict[str, Any], key: str) -> Any:
    if key not in args:
        raise ValueError(f"missing argument: {key}")
    value = args[key]
    if value is None or value == "":
        raise ValueError(f"missing argument: {key}")
    return value


def _equation_expression(args: Dict[str, Any]):
    if "equation" in args:
        return _equation_text_to_expr(args["equation"])
    return _parse_expr(_arg(args, "expr"))


def _equation_text_to_expr(value: Any):
    text = str(value or "").strip()
    if "=" not in text:
        return _parse_expr(text)
    if text.count("=") != 1:
        raise ValueError("equation must contain one equals sign")
    left, right = text.split("=", 1)
    return _parse_expr(left) - _parse_expr(right)


def _symbol(name: Any):
    import sympy as sp

    text = _normalize_math_parameter_text(name).strip()
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,24}", text):
        raise ValueError("invalid symbol")
    return sp.Symbol(text)


def _sympy_domain(value: Any):
    import sympy as sp

    name = str(value or "complex").strip().lower()
    mapping = {
        "real": ("Reals", sp.S.Reals),
        "reals": ("Reals", sp.S.Reals),
        "integer": ("Integers", sp.S.Integers),
        "integers": ("Integers", sp.S.Integers),
        "natural": ("Naturals", sp.S.Naturals),
        "naturals": ("Naturals", sp.S.Naturals),
        "complex": ("Complexes", sp.S.Complexes),
        "complexes": ("Complexes", sp.S.Complexes),
    }
    return mapping.get(name, ("Complexes", sp.S.Complexes))


def _parse_matrix_safe(value: Any):
    import sympy as sp

    if not isinstance(value, list) or not value or not all(isinstance(row, list) for row in value):
        raise ValueError("matrix must be a non-empty nested list")
    if len(value) > MAX_MATRIX_DIM or any(len(row) > MAX_MATRIX_DIM for row in value):
        raise ValueError("matrix exceeds dimension limit")
    row_lengths = {len(row) for row in value}
    if len(row_lengths) != 1:
        raise ValueError("matrix rows must have equal length")
    if len(value) * next(iter(row_lengths)) > MAX_MATRIX_ENTRIES:
        raise ValueError("matrix exceeds entry limit")
    return sp.Matrix([[_parse_expr(entry) for entry in row] for row in value])


def _parse_vector_safe(value: Any):
    import sympy as sp

    if isinstance(value, list) and value and all(not isinstance(item, list) for item in value):
        if len(value) > MAX_MATRIX_DIM:
            raise ValueError("vector exceeds dimension limit")
        return sp.Matrix([_parse_expr(item) for item in value])
    matrix = _parse_matrix_safe(value)
    if matrix.cols != 1:
        raise ValueError("vector must be a flat list or a column matrix")
    return matrix


def _parse_points(value: Any):
    points = _as_sequence(value, 12)
    parsed = []
    for item in points:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("points must be [x,y] pairs")
        parsed.append((_parse_expr(item[0]), _parse_expr(item[1])))
    if not parsed:
        raise ValueError("no interpolation points")
    return parsed


def _as_sequence(value: Any, max_items: int) -> List[Any]:
    """Parse a sequence and reject overflow instead of truncating math input."""

    if isinstance(value, (list, tuple)):
        values = list(value)
    else:
        text = str(value or "").strip()
        if not text:
            return []
        parsed_values: List[Any] | None = None
        if text.startswith("["):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple)):
                    parsed_values = list(parsed)
            except Exception:
                parsed_values = None
        values = (
            parsed_values
            if parsed_values is not None
            else [item.strip() for item in text.split(";") if item.strip()]
        )
    if len(values) > max_items:
        raise ValueError("sequence exceeds safe item bound")
    return values


def _variable_names(value: Any, max_items: int) -> List[str]:
    values = _as_sequence(value, max_items=max_items)
    names: List[str] = []
    for item in values:
        for candidate in str(item).split(","):
            name = _normalize_math_parameter_text(candidate).strip()
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,24}", name):
                names.append(name)
            else:
                raise ValueError("invalid variable name")
            if len(names) > max_items:
                raise ValueError("variable list exceeds safe item bound")
    return names


def _strict_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ValueError("expected boolean")


def _bounded_int(value: Any, low: int, high: int) -> int:
    parsed = int(value)
    if parsed < low or parsed > high:
        raise ValueError("integer outside safe bound")
    return parsed


def _bounded_whole(value: Any, *, limit: int = 10**12) -> int:
    parsed = int(value)
    if abs(parsed) > limit:
        raise ValueError("integer outside safe bound")
    return parsed


def _bounded_positive(value: Any, *, limit: int = 10**12) -> int:
    parsed = int(value)
    if parsed <= 0 or parsed > limit:
        raise ValueError("positive integer outside safe bound")
    return parsed


def _bounded_probability(value: Any):
    """Return an exact explicit probability in [0, 1], rejecting symbols."""

    import sympy as sp

    parsed = _parse_expr(value)
    if getattr(parsed, "free_symbols", set()):
        raise ValueError("probability must be explicit")
    numeric = float(sp.N(parsed, 30))
    if not math.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
        raise ValueError("probability outside [0, 1]")
    return parsed


def _bounded_nonnegative_number(value: Any):
    """Return an explicit nonnegative real number under a conservative cap."""

    import sympy as sp

    parsed = _parse_expr(value)
    if getattr(parsed, "free_symbols", set()):
        raise ValueError("numeric parameter must be explicit")
    numeric = float(sp.N(parsed, 30))
    if not math.isfinite(numeric) or numeric < 0.0 or numeric > 10**6:
        raise ValueError("numeric parameter outside safe bound")
    return parsed


def _degree_list(value: Any) -> List[int]:
    values = _as_sequence(value, 200)
    if not values:
        raise ValueError("empty degree list")
    degrees = [_bounded_int(item, 0, 10**6) for item in values]
    if len(degrees) > 200:
        raise ValueError("degree list too long")
    return degrees


def _extended_gcd(a: int, b: int):
    old_r, r = abs(a), abs(b)
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    x = old_s if a >= 0 else -old_s
    y = old_t if b >= 0 else -old_t
    return old_r, x, y


def _stirling2(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    row = [0] * (k + 1)
    row[0] = 1
    for i in range(1, n + 1):
        upper = min(i, k)
        for j in range(upper, 0, -1):
            row[j] = row[j - 1] + j * row[j]
        row[0] = 0
    return row[k]


def _partition_exact_parts(total: int, parts: int) -> int:
    """Count partitions of ``total`` into exactly ``parts`` positive parts."""

    if total == 0:
        return 1 if parts == 0 else 0
    if parts <= 0 or parts > total:
        return 0
    values = [[0] * (parts + 1) for _ in range(total + 1)]
    values[0][0] = 1
    for n in range(1, total + 1):
        for k in range(1, min(n, parts) + 1):
            values[n][k] = values[n - 1][k - 1]
            if n >= k:
                values[n][k] += values[n - k][k]
    return values[total][parts]


def _partition_total(total: int) -> int:
    """Count unrestricted integer partitions p(total) with a small DP."""

    values = [0] * (total + 1)
    values[0] = 1
    for part in range(1, total + 1):
        for amount in range(part, total + 1):
            values[amount] += values[amount - part]
    return values[total]


def _bounded_compositions(total: int, parts: int, lower: int, upper: int) -> int:
    """Count ordered integer compositions with identical lower/upper bounds."""

    if total < parts * lower or total > parts * upper:
        return 0
    shifted = total - parts * lower
    width = upper - lower
    value = 0
    # Inclusion-exclusion over coordinates that exceed the shifted upper bound.
    for j in range(0, min(parts, shifted // (width + 1)) + 1):
        value += (-1) ** j * math.comb(parts, j) * math.comb(
            shifted - j * (width + 1) + parts - 1,
            parts - 1,
        )
    return value


def _distinct_partition_total(total: int) -> int:
    """Count partitions into distinct positive parts."""

    values = [0] * (total + 1)
    values[0] = 1
    for part in range(1, total + 1):
        for amount in range(total, part - 1, -1):
            values[amount] += values[amount - part]
    return values[total]


def _havel_hakimi(degrees: Sequence[int]) -> bool:
    seq = list(degrees)
    n = len(seq)
    if any(degree >= n for degree in seq):
        return False
    while True:
        seq = sorted((degree for degree in seq if degree > 0), reverse=True)
        if not seq:
            return True
        d = seq.pop(0)
        if d > len(seq):
            return False
        for i in range(d):
            seq[i] -= 1
            if seq[i] < 0:
                return False


def _raw_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, dict, tuple)):
        return _compact_json(value)
    return str(value)


def _prov(value: Any) -> str:
    raw = _raw_text(value)
    normalized = _normalize_math_parameter_text(raw)
    if normalized != raw:
        return f"{raw} [normalized={normalized}]"
    return raw


def _compact_json(value: Any) -> str:
    if isinstance(value, tuple):
        value = list(value)
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(value)


def _display(value: Any) -> str:
    _reject_unresolved_symbolic_value(value)
    text = str(value)
    return re.sub(r"\s+", " ", text).strip()


def _reject_unresolved_symbolic_value(value: Any) -> None:
    """Reject symbolic placeholders that are not completed deterministic output."""

    try:
        import sympy as sp
    except Exception:
        return

    forbidden = (sp.Integral, sp.Sum, sp.Limit, sp.Derivative)

    def contains_unresolved(item: Any) -> bool:
        if isinstance(item, dict):
            return any(
                contains_unresolved(key) or contains_unresolved(entry)
                for key, entry in item.items()
            )
        if isinstance(item, (list, tuple, set)):
            return any(contains_unresolved(entry) for entry in item)
        if isinstance(item, sp.MatrixBase):
            return any(contains_unresolved(entry) for entry in item)
        condition_set = getattr(sp, "ConditionSet", ())
        if condition_set and isinstance(item, condition_set):
            return True
        has = getattr(item, "has", None)
        try:
            return bool(callable(has) and has(*forbidden))
        except Exception:
            return False

    if contains_unresolved(value):
        raise ValueError("symbolic result was not fully evaluated")


def _fact_statement_is_complete(statement: Any) -> bool:
    text = str(statement or "")
    if not text or len(text) > MAX_RESULT_CHARS:
        return False
    # Ellipses in exact evidence are indistinguishable from truncation to the
    # final solver, so never inject them as a completed deterministic fact.
    if text.rstrip().endswith("..."):
        return False
    return not any(
        marker in text
        for marker in ("Integral(", "Sum(", "Limit(", "Derivative(", "ConditionSet(")
    )


def _normalize_verification_requests(requests: Sequence[Dict[str, Any]] | Any) -> List[Dict[str, Any]]:
    """Validate the closed, object-grounded verification request schema.

    This boundary is shared by the parent and the fixed worker.  It accepts
    only objects whose mathematical input is already explicit; the worker
    never receives a problem string from which it could rediscover a target.
    Any malformed, oversized, ambiguous, or mixed-schema batch is rejected in
    full (fail closed).
    """

    if not isinstance(requests, (list, tuple)):
        return []
    if not requests or len(requests) > _VERIFICATION_REQUEST_MAX:
        return []

    normalized: List[Dict[str, Any]] = []
    for item in requests:
        if not isinstance(item, dict):
            return []
        kind = item.get("kind")
        if not isinstance(kind, str) or kind not in _VERIFICATION_KINDS:
            return []
        if frozenset(item.keys()) != _VERIFICATION_REQUEST_KEYS[kind]:
            return []

        if kind == "pure_arithmetic":
            expression = _verification_text(item.get("expression"), max_chars=240)
            if expression is None or not re.fullmatch(r"[0-9()+\-*/^.\s]+", expression):
                return []
            if not re.search(r"[+\-*/^]", expression):
                return []
            try:
                parsed = _safe_parse_expr(expression)
                if getattr(parsed, "free_symbols", set()):
                    return []
            except Exception:
                return []
            normalized.append({"kind": kind, "expression": expression})
            continue

        if kind in {"equation_solution", "equation_solution_set"}:
            equation = _verification_text(item.get("equation"), max_chars=240)
            variable = item.get("variable")
            if equation is None or not isinstance(variable, str) or not re.fullmatch(r"[A-Za-z]", variable):
                return []
            if equation.count("=") != 1:
                return []
            left, right = (part.strip() for part in equation.split("=", 1))
            if not left or not right:
                return []
            try:
                left_expr = _safe_parse_expr(left)
                right_expr = _safe_parse_expr(right)
                symbols = getattr(left_expr - right_expr, "free_symbols", set())
                if not any(str(symbol) == variable for symbol in symbols):
                    return []
            except Exception:
                return []
            if kind == "equation_solution_set":
                domain = item.get("domain")
                if not isinstance(domain, str) or domain not in _VERIFICATION_DOMAINS:
                    return []
                normalized.append(
                    {"kind": kind, "equation": equation, "variable": variable, "domain": domain}
                )
            else:
                normalized.append({"kind": kind, "equation": equation, "variable": variable})
            continue

        matrix = _normalize_verification_matrix(item.get("matrix"))
        if matrix is None:
            return []
        normalized.append({"kind": kind, "matrix": matrix})

    # Duplicate requests are harmless but needlessly amplify evidence and
    # correction prompts; collapse them deterministically.
    deduped: List[Dict[str, Any]] = []
    for request in normalized:
        if request not in deduped:
            deduped.append(request)
    return deduped


def _verification_text(value: Any, *, max_chars: int) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split()).strip()
    if not text or len(text) > max_chars:
        return None
    return text


def _normalize_verification_matrix(value: Any) -> List[List[Any]] | None:
    if not isinstance(value, list) or not value or len(value) > MAX_MATRIX_DIM:
        return None
    if any(not isinstance(row, list) or not row or len(row) > MAX_MATRIX_DIM for row in value):
        return None
    widths = {len(row) for row in value}
    if len(widths) != 1:
        return None
    if len(value) * next(iter(widths)) > MAX_MATRIX_ENTRIES:
        return None
    normalized: List[List[Any]] = []
    for row in value:
        normalized_row: List[Any] = []
        for entry in row:
            if isinstance(entry, bool) or not isinstance(entry, (int, float)):
                return None
            if isinstance(entry, float) and not math.isfinite(entry):
                return None
            normalized_row.append(entry)
        normalized.append(normalized_row)
    return normalized


def _fact(family: str, operation: str, statement: str) -> ToolFact:
    complete_statement = _display(statement)
    if not _fact_statement_is_complete(complete_statement):
        raise ValueError("complete fact exceeds the evidence bound")
    return ToolFact(
        family=family,
        operation=operation,
        statement=complete_statement,
    )

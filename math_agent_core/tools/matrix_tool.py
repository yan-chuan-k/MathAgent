from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any, Dict, List

from math_agent_core.state import EvidenceStatus, VerificationEvidence, VerificationLevel
from math_agent_core.tools.base import MathTool


MAX_MATRIX_DIM = 6
MAX_MATRIX_ENTRIES = MAX_MATRIX_DIM * MAX_MATRIX_DIM
MAX_TIMEOUT_SECONDS = 2.0


class MatrixTool(MathTool):
    name = "matrix_tool"

    def validate_input(self, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            raise ValueError("payload must be a dict")
        tool = str(payload.get("tool") or "")
        if tool not in {
            "matrix_determinant",
            "matrix_multiply",
            "matrix_inverse",
            "linear_system_residual",
            "matrix_rank",
            "eigenpair_residual",
            "vector_orthogonality",
            "vector_normalization",
            "matrix_equivalence",
            "vector_equivalence",
        }:
            raise ValueError("unsupported matrix tool")

    def run(self, payload: Dict[str, Any], timeout: float = MAX_TIMEOUT_SECONDS) -> VerificationEvidence:
        try:
            self.validate_input(payload)
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._run_now, payload)
                return future.result(timeout=timeout)
        except TimeoutError:
            return self.inconclusive(str(payload.get("claim_id") or "matrix_check"), str(payload.get("tool") or "matrix"), "matrix check timed out")
        except Exception as exc:
            return self.inconclusive(str(payload.get("claim_id") or "matrix_check"), str(payload.get("tool") or "matrix"), f"{type(exc).__name__}: {str(exc)[:220]}")

    def _run_now(self, payload: Dict[str, Any]) -> VerificationEvidence:
        tool = str(payload.get("tool") or "")
        args = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {}
        claim_id = str(payload.get("claim_id") or "matrix_check")
        if tool == "matrix_determinant":
            return self._matrix_determinant(args, claim_id)
        if tool == "matrix_multiply":
            return self._matrix_multiply(args, claim_id)
        if tool == "matrix_inverse":
            return self._matrix_inverse(args, claim_id)
        if tool == "linear_system_residual":
            return self._linear_system_residual(args, claim_id)
        if tool == "matrix_rank":
            return self._matrix_rank(args, claim_id)
        if tool == "eigenpair_residual":
            return self._eigenpair_residual(args, claim_id)
        if tool == "vector_orthogonality":
            return self._vector_orthogonality(args, claim_id)
        if tool == "vector_normalization":
            return self._vector_normalization(args, claim_id)
        if tool == "matrix_equivalence":
            return self._matrix_equivalence(args, claim_id)
        if tool == "vector_equivalence":
            return self._vector_equivalence(args, claim_id)
        raise ValueError("unsupported matrix tool")

    def _matrix_determinant(self, args: Dict[str, Any], claim_id: str) -> VerificationEvidence:
        matrix = _parse_matrix(args.get("matrix"))
        expected = _parse_expr(args.get("expected"))
        residual = _simplify(matrix.det() - expected)
        return _evidence(claim_id, "matrix_determinant", "Recomputed determinant exactly.", residual)

    def _matrix_multiply(self, args: Dict[str, Any], claim_id: str) -> VerificationEvidence:
        left = _parse_matrix(args.get("left"))
        right = _parse_matrix(args.get("right"))
        expected = _parse_matrix(args.get("expected"))
        residual = _simplify_matrix(left * right - expected)
        return _evidence(claim_id, "matrix_multiply", "Recomputed matrix product exactly.", residual)

    def _matrix_inverse(self, args: Dict[str, Any], claim_id: str) -> VerificationEvidence:
        matrix = _parse_matrix(args.get("matrix"))
        inverse = _parse_matrix(args.get("inverse"))
        identity = _identity(matrix.rows)
        residual = _simplify_matrix(matrix * inverse - identity)
        if residual == "0":
            residual = _simplify_matrix(inverse * matrix - identity)
        return _evidence(claim_id, "matrix_inverse", "Checked both inverse products against identity.", residual)

    def _linear_system_residual(self, args: Dict[str, Any], claim_id: str) -> VerificationEvidence:
        matrix = _parse_matrix(args.get("matrix"))
        vector = _parse_vector(args.get("vector"))
        rhs = _parse_vector(args.get("rhs"))
        residual = _simplify_matrix(matrix * vector - rhs)
        return _evidence(claim_id, "linear_system_residual", "Computed A*x-b exactly.", residual)

    def _matrix_rank(self, args: Dict[str, Any], claim_id: str) -> VerificationEvidence:
        matrix = _parse_matrix(args.get("matrix"))
        expected = int(args.get("expected"))
        residual = str(matrix.rank() - expected)
        return _evidence(claim_id, "matrix_rank", "Recomputed matrix rank exactly.", residual)

    def _eigenpair_residual(self, args: Dict[str, Any], claim_id: str) -> VerificationEvidence:
        matrix = _parse_matrix(args.get("matrix"))
        vector = _parse_vector(args.get("vector"))
        eigenvalue = _parse_expr(args.get("eigenvalue"))
        residual = _simplify_matrix(matrix * vector - eigenvalue * vector)
        return _evidence(claim_id, "eigenpair_residual", "Computed A*v-lambda*v exactly.", residual)

    def _vector_orthogonality(self, args: Dict[str, Any], claim_id: str) -> VerificationEvidence:
        left = _parse_vector(args.get("left"))
        right = _parse_vector(args.get("right"))
        residual = _simplify((left.T * right)[0])
        return _evidence(claim_id, "vector_orthogonality", "Computed dot product exactly.", residual)

    def _vector_normalization(self, args: Dict[str, Any], claim_id: str) -> VerificationEvidence:
        vector = _parse_vector(args.get("vector"))
        expected_norm_sq = _parse_expr(args.get("expected_norm_squared", 1))
        residual = _simplify((vector.T * vector)[0] - expected_norm_sq)
        return _evidence(claim_id, "vector_normalization", "Computed squared norm exactly.", residual)

    def _matrix_equivalence(self, args: Dict[str, Any], claim_id: str) -> VerificationEvidence:
        left = _parse_matrix(args.get("left"))
        right = _parse_matrix(args.get("right"))
        residual = _simplify_matrix(left - right)
        return _evidence(claim_id, "matrix_equivalence", "Compared matrix entries exactly.", residual)

    def _vector_equivalence(self, args: Dict[str, Any], claim_id: str) -> VerificationEvidence:
        left = _parse_vector(args.get("left"))
        right = _parse_vector(args.get("right"))
        residual = _simplify_matrix(left - right)
        return _evidence(claim_id, "vector_equivalence", "Compared vector entries exactly.", residual)


def _parse_matrix(value: Any):
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


def _parse_vector(value: Any):
    import sympy as sp

    if isinstance(value, list) and value and all(not isinstance(item, list) for item in value):
        if len(value) > MAX_MATRIX_DIM:
            raise ValueError("vector exceeds dimension limit")
        return sp.Matrix([_parse_expr(item) for item in value])
    matrix = _parse_matrix(value)
    if matrix.cols != 1:
        raise ValueError("vector must be a flat list or a column matrix")
    return matrix


def _parse_expr(value: Any):
    import sympy as sp
    from sympy.parsing.sympy_parser import parse_expr

    text = str(value)
    if len(text) > 80 or any(token in text.lower() for token in ("__", "import", "exec", "eval", "lambda", "open")):
        raise ValueError("unsafe expression")
    safe_globals = dict(sp.__dict__)
    safe_globals["__builtins__"] = {}
    return parse_expr(text, global_dict=safe_globals, evaluate=True)


def _identity(size: int):
    import sympy as sp

    return sp.eye(size)


def _simplify(value: Any) -> str:
    import sympy as sp

    simplified = sp.simplify(value)
    return "0" if simplified == 0 else str(simplified)


def _simplify_matrix(matrix: Any) -> str:
    simplified = matrix.applyfunc(_simplify_entry)
    if all(item == 0 for item in simplified):
        return "0"
    return str(simplified)


def _simplify_entry(value: Any) -> Any:
    import sympy as sp

    return sp.simplify(value)


def _evidence(claim_id: str, method: str, details: str, residual: str) -> VerificationEvidence:
    status = EvidenceStatus.PASS.value if residual == "0" else EvidenceStatus.FAIL.value
    return VerificationEvidence(
        verifier="matrix_tool",
        claim_id=claim_id,
        status=status,
        method=method,
        details=details,
        residual=residual,
        verification_level=VerificationLevel.EXACT_SYMBOLIC.value,
        is_decisive=True,
    )

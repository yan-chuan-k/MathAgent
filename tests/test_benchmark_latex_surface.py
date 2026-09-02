import json
from pathlib import Path

import pytest

from math_agent_core.evaluation.grader import (
    _normalize_benchmark_text,
    grade_full_problem,
    grade_primary_answer,
    grade_required_claims,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "sample_data" / "hard_diagnostics.jsonl"


REALISTIC_LATEX_RESPONSES = {
    "hard_discrete_math": r"Therefore, $C_6=\boxed{132}$.",
    "hard_numerical_analysis": (
        r"The Newton iteration is $x_{n+1}=\frac12(x_n+\frac{2}{x_n})$, "
        r"and the convergence is quadratic."
    ),
    "hard_measure_integration": (
        r"The limit is $\boxed{\frac{1}{2}}$. "
        r"The dominated convergence theorem does not apply."
    ),
    "hard_differential_geometry": r"The Gaussian curvature is $K=\boxed{1}$.",
    "hard_probability": r"$\mathbb E[\min(X,Y)]=\boxed{\frac{1}{\lambda+\mu}}$.",
    "hard_abstract_algebra": r"Therefore the answer is $\boxed{72}$.",
    "hard_stochastic_process": r"$P(N(3)-N(1)=2)=e^{-2\lambda}\frac{(2\lambda)^2}{2!}$.",
    "hard_complex_analysis": r"The contour integral equals $\boxed{\pi(e^{-1}-e)}$.",
    "hard_ode": r"The solution is $\boxed{y=(x+C)e^{-x}}$.",
    "hard_statistics": r"The MLE is the sample mean $\bar X$, and it is unbiased.",
    "hard_functional_analysis": (
        r"By Cauchy--Schwarz, equality holds iff the vectors are linearly dependent."
    ),
    "hard_linear_regression": (
        r"$\hat\beta=(X^T X)^{-1}X^T y$, and "
        r"$\operatorname{Var}(\hat\beta)=\sigma^2(X^T X)^{-1}$."
    ),
    "hard_pde": r"$u(x,t)=\sin(\pi x)e^{-\pi^2 t}$.",
    "hard_advanced_math": r"$f^*(y)=0$ for $\|y\|_*\le1$, and $+\infty$ otherwise.",
    "hard_linear_algebra": r"The minimal polynomial is $\boxed{(t-2)^2}$.",
    "hard_optimization": r"The optimizer is $(x,y)=(2,0)$, and the optimum value is $\boxed{6}$.",
    "hard_real_analysis": r"The series converges uniformly on $\mathbb R$ by the Weierstrass M-test.",
    "hard_topology": r"The continuous image is compact.",
}


def _load_cases():
    return [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.mark.parametrize("case", _load_cases(), ids=lambda case: case["idx"])
def test_realistic_latex_hard_diagnostic_response_is_correct(case):
    result = grade_full_problem(REALISTIC_LATEX_RESPONSES[case["idx"]], case["grading"])
    assert result["status"] == "CORRECT", result


def test_realistic_latex_required_claims_are_fully_gradable():
    unresolved = 0
    for case in _load_cases():
        result = grade_full_problem(REALISTIC_LATEX_RESPONSES[case["idx"]], case["grading"])
        unresolved += result["required_claims"]["unresolved_count"]
    assert unresolved == 0


@pytest.mark.parametrize(
    ("expected", "primary_type", "response"),
    [
        ("1/2", "numeric", r"The limit is $\boxed{\frac{1}{2}}$."),
        ("1/(lambda+mu)", "symbolic", r"The expectation is $\boxed{\frac{1}{\lambda+\mu}}$."),
        (
            "x_{n+1}=1/2*(x_n+2/x_n)",
            "symbolic",
            r"The Newton iteration is $x_{n+1}=\frac12(x_n+\frac{2}{x_n})$.",
        ),
        (
            "e^(-2*lambda)*(2*lambda)^2/2!",
            "symbolic",
            r"$P(N(3)-N(1)=2)=e^{-2\lambda}\frac{(2\lambda)^2}{2!}$.",
        ),
        (
            "u(x,t)=e^(-pi^2*t)sin(pi*x)",
            "symbolic",
            r"$u(x,t)=e^{-\pi^2 t}\sin(\pi x)$.",
        ),
        (
            "beta_hat=(X^T X)^(-1)X^T y",
            "symbolic",
            r"$\hat{\beta}=(X^T X)^{-1}X^T y$.",
        ),
    ],
)
def test_requested_compositional_surface_forms(expected, primary_type, response):
    result = grade_primary_answer(response, {"primary": expected, "primary_type": primary_type})
    assert result["status"] == "CORRECT", result


def test_relation_infinity_spacing_and_typography_normalization_is_bounded_and_explicit():
    normalized = _normalize_benchmark_text(
        r"Cauchy--Schwarz\quad \|y\|_*\leq1,\qquad z\ge 0,\; +\infty"
    )
    assert "Cauchy-Schwarz" in normalized
    assert "|y|_*<=1" in normalized
    assert "z>= 0" in normalized or "z>=0" in normalized
    assert "+infinity" in normalized


def test_var_beta_hat_latex_required_claim():
    response = r"$\operatorname{Var}(\hat\beta)=\sigma^2(X^T X)^{-1}$."
    result = grade_required_claims(response, {"required_claims": ["VAR_BETA_HAT"]})
    assert result["status"] == "CORRECT", result


def test_requested_required_claim_surface_aliases():
    assert grade_required_claims(
        "The convergence is quadratic.",
        {"required_claims": ["QUADRATIC_CONVERGENCE"]},
    )["status"] == "CORRECT"
    assert grade_required_claims(
        "The dominated convergence theorem does not apply.",
        {"required_claims": ["DCT_NOT_APPLICABLE"]},
    )["status"] == "CORRECT"
    assert grade_required_claims(
        "The dominated convergence theorem does not apply at first, but actually DCT applies.",
        {"required_claims": ["DCT_NOT_APPLICABLE"]},
    )["status"] == "INCORRECT"


@pytest.mark.parametrize(
    ("spec", "response"),
    [
        (
            {"primary": "132", "primary_type": "numeric"},
            r"The answer is $\boxed{132}$, but actually the answer is 131.",
        ),
        (
            {"primary": "1/(lambda+mu)", "primary_type": "symbolic"},
            r"The expectation is $\frac{1}{\lambda+\mu}$, but actually the expectation is $\frac{2}{\lambda+\mu}$.",
        ),
        (
            {"primary": "beta_hat=(X^T X)^(-1)X^T y", "primary_type": "symbolic"},
            r"$\hat\beta=(X^T X)^{-1}X^T y$, but actually $\hat\beta=X^T y$.",
        ),
    ],
)
def test_latex_primary_correction_negatives_remain_incorrect(spec, response):
    result = grade_primary_answer(response, spec)
    assert result["status"] == "INCORRECT", result


def test_latex_dct_correction_negative_remains_incorrect():
    result = grade_required_claims(
        "DCT does not apply, but actually DCT applies.",
        {"required_claims": ["DCT_NOT_APPLICABLE"]},
    )
    assert result["status"] == "INCORRECT", result

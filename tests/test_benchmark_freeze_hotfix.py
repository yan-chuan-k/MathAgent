import pytest

from math_agent_core.evaluation.grader import (
    grade_primary_answer,
    grade_required_claims,
)


@pytest.mark.parametrize(
    ("expected", "primary_type", "response"),
    [
        ("1/2", "numeric", r"The limit is $\frac{1}{2}$."),
        ("132", "numeric", r"Therefore $C_6=\boxed{132}$ ."),
        ("1", "numeric", r"The Gaussian curvature is $K=\boxed{1}$ ."),
        ("1/(lambda+mu)", "symbolic", r"The expectation is $\frac{1}{\lambda+\mu}$ ."),
        ("y=(x+C)e^(-x)", "symbolic", r"The solution is $y=(x+C)e^{-x}$ ."),
        ("pi*(e^(-1)-e)", "symbolic", r"The contour integral equals $\pi(e^{-1}-e)$ ."),
    ],
)
def test_lightweight_latex_positive_controls(expected, primary_type, response):
    result = grade_primary_answer(response, {"primary": expected, "primary_type": primary_type})
    assert result["status"] == "CORRECT", result


def test_symbolic_uses_last_decisive_assertion_negative():
    spec = {"primary": "pi*(e^(-1)-e)", "primary_type": "symbolic"}
    response = "The contour integral equals pi*(e^(-1)-e), but that is wrong; actually it equals 0."
    assert grade_primary_answer(response, spec)["status"] == "INCORRECT"


def test_minimal_polynomial_uses_last_decisive_assertion_negative():
    spec = {"primary": "(t-2)^2", "primary_type": "symbolic"}
    response = "The minimal polynomial is (t-2)^2, but this is false; actually it is (t-2)^3."
    assert grade_primary_answer(response, spec)["status"] == "INCORRECT"


def test_numeric_trailing_correction_negative():
    spec = {"primary": "132", "primary_type": "numeric"}
    response = "The answer is 132, but that is wrong; actually it is 131."
    assert grade_primary_answer(response, spec)["status"] == "INCORRECT"


def test_text_alias_trailing_rejection_negative():
    compact = {"primary": "compact", "primary_type": "text_alias", "aliases": ["compact"]}
    uniform = {
        "primary": "uniform_convergence",
        "primary_type": "text_alias",
        "aliases": ["uniformly convergent", "converges uniformly", "uniform convergence"],
    }
    assert grade_primary_answer("The continuous image is compact, but that conclusion is false.", compact)["status"] == "INCORRECT"
    assert grade_primary_answer("The series converges uniformly. Actually, that is false.", uniform)["status"] == "INCORRECT"


def test_required_claims_use_last_semantic_polarity_negative():
    assert grade_required_claims(
        "The estimator is unbiased at first glance, but actually it is biased.",
        {"required_claims": ["UNBIASED"]},
    )["status"] == "INCORRECT"
    assert grade_required_claims(
        "DCT does not apply at first glance, but in fact DCT applies.",
        {"required_claims": ["DCT_NOT_APPLICABLE"]},
    )["status"] == "INCORRECT"


def test_correction_order_positive_controls():
    symbolic = {"primary": "pi*(e^(-1)-e)", "primary_type": "symbolic"}
    numeric = {"primary": "132", "primary_type": "numeric"}
    assert grade_primary_answer(
        "The contour integral equals 0, but actually it equals pi*(e^(-1)-e).", symbolic
    )["status"] == "CORRECT"
    assert grade_primary_answer("The answer is 131, but actually it is 132.", numeric)["status"] == "CORRECT"
    assert grade_required_claims(
        "The estimator is biased at first glance, but actually it is unbiased.",
        {"required_claims": ["UNBIASED"]},
    )["status"] == "CORRECT"
    assert grade_required_claims(
        "DCT applies? No. DCT does not apply because no integrable dominator exists.",
        {"required_claims": ["DCT_NOT_APPLICABLE"]},
    )["status"] == "CORRECT"


def test_text_alias_positive_correction_order():
    compact = {"primary": "compact", "primary_type": "text_alias", "aliases": ["compact"]}
    uniform = {
        "primary": "uniform_convergence",
        "primary_type": "text_alias",
        "aliases": ["uniformly convergent", "converges uniformly", "uniform convergence"],
    }
    assert grade_primary_answer(
        "At first this may look non-compact, but the continuous image is compact.", compact
    )["status"] == "CORRECT"
    assert grade_primary_answer(
        "One might suspect failure of uniform convergence; however, the series converges uniformly.", uniform
    )["status"] == "CORRECT"


def test_assignment_expected_is_lhs_aware():
    spec = {"primary": "beta_hat=(X^T X)^(-1)X^T y", "primary_type": "symbolic"}
    response = "beta_hat=(X^T X)^(-1)X^T y and Var(beta_hat)=sigma^2*(X^T X)^(-1)."
    assert grade_primary_answer(response, spec)["status"] == "CORRECT"


@pytest.mark.parametrize(
    ("expected", "response"),
    [
        ("y=(x+C)e^(-x)", "The solution is y=(x+C)e^(-x) and C=0."),
        ("x_{n+1}=1/2*(x_n+2/x_n)", "The iteration is x_{n+1}=1/2*(x_n+2/x_n) and the order=2."),
        ("u(x,t)=sin(pi*x)*exp(-pi^2*t)", "The solution is u(x,t)=sin(pi*x)*exp(-pi^2*t) and t=1."),
    ],
)
def test_assignment_lhs_awareness_generalizes(expected, response):
    assert grade_primary_answer(response, {"primary": expected, "primary_type": "symbolic"})["status"] == "CORRECT"


def test_unparsed_trailing_correction_is_unresolved_not_early_correct():
    spec = {"primary": "132", "primary_type": "numeric"}
    result = grade_primary_answer("The answer is 132, but correction: see revised derivation below.", spec)
    assert result["status"] == "UNRESOLVED"

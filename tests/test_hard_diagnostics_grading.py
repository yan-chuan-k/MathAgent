import json
from pathlib import Path

import pytest

from math_agent_core.evaluation.grader import grade_full_problem


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "sample_data" / "hard_diagnostics.jsonl"

CORRECT_RESPONSES = {
    "hard_discrete_math": "Therefore C_6=132.",
    "hard_numerical_analysis": "The Newton iteration is x_{n+1}=1/2*(x_n+2/x_n), and it converges quadratically.",
    "hard_measure_integration": "The limit is 1/2. DCT does not apply because there is no integrable dominator.",
    "hard_differential_geometry": "The Gaussian curvature is K=1.",
    "hard_probability": "The expectation is 1/(mu+lambda).",
    "hard_abstract_algebra": "The answer is 72.",
    "hard_stochastic_process": "P(N(3)-N(1)=2)=e^(-2*lambda)*(2*lambda)^2/2!.",
    "hard_complex_analysis": "The contour integral equals pi*(e^(-1)-e).",
    "hard_ode": "The solution is y=(x+C)e^(-x).",
    "hard_statistics": "The MLE is the sample mean, and it is unbiased.",
    "hard_functional_analysis": "By the Cauchy-Schwarz inequality, equality holds iff the vectors are linearly dependent.",
    "hard_linear_regression": "beta_hat=(X^T X)^(-1)X^T y and Var(beta_hat)=sigma^2*(X^T X)^(-1).",
    "hard_pde": "The solution is u(x,t)=sin(pi*x)*exp(-pi^2*t).",
    "hard_advanced_math": "The conjugate is 0 when the dual norm is <= 1 and +infinity otherwise.",
    "hard_linear_algebra": "The minimal polynomial is (t-2)^2.",
    "hard_optimization": "The optimizer is (x,y)=(2,0), and the optimum value is 6.",
    "hard_real_analysis": "The series converges uniformly on the real line.",
    "hard_topology": "The continuous image is compact.",
}

NEGATIVE_RESPONSES = {
    "hard_discrete_math": "The final answer is 131; 132 is only a nearby value.",
    "hard_numerical_analysis": "The Newton iteration is x_{n+1}=1/2*(x_n+2/x_n), but it is not quadratically convergent.",
    "hard_measure_integration": "The limit is 1/2, and DCT applies.",
    "hard_differential_geometry": "The calculation mentions 1, but the final value is 2.",
    "hard_probability": "The expectation is not 1/(lambda+mu); it equals 2/(lambda+mu).",
    "hard_abstract_algebra": "The answer is 71; 72 occurs only in an intermediate count.",
    "hard_stochastic_process": "P(N(3)-N(1)=2)=e^(-lambda)*(2*lambda)^2/2!.",
    "hard_complex_analysis": "The contour integral is not pi*(e^(-1)-e); it equals 0.",
    "hard_ode": "The solution is y=(x+C)e^(x).",
    "hard_statistics": "The MLE is the sample mean, but it is biased.",
    "hard_functional_analysis": "By Cauchy-Schwarz, equality holds iff the vectors are linearly independent.",
    "hard_linear_regression": "beta_hat=(X^T X)^(-1)X^T y and Var(beta_hat)=sigma^2*X^T X.",
    "hard_pde": "The solution is u(x,t)=sin(pi*x)*exp(-pi*t).",
    "hard_advanced_math": "The conjugate is 0 when the dual norm is <= 1.",
    "hard_linear_algebra": "The minimal polynomial is not (t-2)^2; it is (t-2)^3.",
    "hard_optimization": "The optimizer is (x,y)=(2,0), but the optimum value is 5.",
    "hard_real_analysis": "The series is not uniformly convergent on the real line.",
    "hard_topology": "The continuous image is not compact.",
}


def _load_cases():
    return [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_all_hard_diagnostics_have_explicit_grading_specs():
    cases = _load_cases()
    assert len(cases) == 18
    assert set(CORRECT_RESPONSES) == {case["idx"] for case in cases}
    for case in cases:
        grading = case.get("grading")
        assert isinstance(grading, dict)
        assert grading.get("primary")
        assert grading.get("primary_type") in {"numeric", "symbolic", "text_alias", "solution_set", "structured"}
        assert isinstance(grading.get("aliases"), list)
        assert isinstance(grading.get("required_claims"), list)


@pytest.mark.parametrize("case", _load_cases(), ids=lambda case: case["idx"])
def test_representative_natural_correct_response_is_correct(case):
    result = grade_full_problem(CORRECT_RESPONSES[case["idx"]], case["grading"])
    assert result["status"] == "CORRECT", result


@pytest.mark.parametrize("case", _load_cases(), ids=lambda case: case["idx"])
def test_hard_diagnostic_negative_control_is_incorrect(case):
    result = grade_full_problem(NEGATIVE_RESPONSES[case["idx"]], case["grading"])
    assert result["status"] == "INCORRECT", result


def test_structured_optimization_requires_both_obligations():
    case = next(case for case in _load_cases() if case["idx"] == "hard_optimization")
    spec = case["grading"]
    assert grade_full_problem("The optimizer is (2,0).", spec)["status"] == "INCORRECT"
    assert grade_full_problem("The optimum value is 6.", spec)["status"] == "INCORRECT"
    assert grade_full_problem("The optimizer is (1,1), and the optimum value is 6.", spec)["status"] == "INCORRECT"
    assert grade_full_problem("The optimizer is (2,0), and the optimum value is 5.", spec)["status"] == "INCORRECT"


def test_hard_diagnostics_required_claims_are_fully_gradable():
    unresolved = 0
    for case in _load_cases():
        result = grade_full_problem(CORRECT_RESPONSES[case["idx"]], case["grading"])
        unresolved += result["required_claims"]["unresolved_count"]
    assert unresolved == 0

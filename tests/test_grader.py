from math_agent_core.evaluation.grader import grade_full_problem, grade_primary_answer, grade_required_claims


def test_numeric_primary_and_negations():
    assert grade_primary_answer("The answer is 132.", {"primary": "132", "primary_type": "numeric"})["correct"] is True
    assert grade_primary_answer("C_6=132", {"primary": "132", "primary_type": "numeric"})["correct"] is True
    assert grade_primary_answer("The answer is not 132.", {"primary": "132", "primary_type": "numeric"})["correct"] is False
    assert grade_primary_answer("132 is incorrect; the answer is 131.", {"primary": "132", "primary_type": "numeric"})["correct"] is False
    assert grade_primary_answer("K=1 is impossible; actually K=2", {"primary": "1", "primary_type": "numeric"})["correct"] is False


def test_symbolic_assignments_and_text_aliases():
    assert grade_primary_answer("The Gaussian curvature is K=1.", {"primary": "K = 1", "primary_type": "symbolic"})["correct"] is True
    assert grade_primary_answer("E[min(X,Y)] = 1/(mu+lambda).", {"primary": "1/(lambda+mu)", "primary_type": "symbolic"})["correct"] is True
    assert grade_primary_answer("y=e^(-x)*(x+C)", {"primary": "y=(x+C)e^(-x)", "primary_type": "symbolic"})["correct"] is True
    assert grade_primary_answer("u(x,t)=sin(pi*x)*exp(-pi^2*t)", {"primary": "u(x,t)=e^(-pi^2*t)sin(pi*x)", "primary_type": "symbolic"})["correct"] is True
    assert grade_primary_answer("The series converges uniformly.", {"primary": "uniform_convergence", "primary_type": "text_alias"})["correct"] is True
    assert grade_primary_answer("The sample average is used.", {"primary": "sample_mean", "primary_type": "text_alias"})["correct"] is True


def test_claim_polarity():
    assert grade_required_claims("The estimator is unbiased.", {"required_claims": ["UNBIASED"]})["correct"] is True
    assert grade_required_claims("The estimator is not unbiased.", {"required_claims": ["UNBIASED"]})["correct"] is False
    assert grade_required_claims("The method converges quadratically.", {"required_claims": ["QUADRATIC_CONVERGENCE"]})["correct"] is True
    assert grade_required_claims("The method is not quadratically convergent.", {"required_claims": ["QUADRATIC_CONVERGENCE"]})["correct"] is False
    assert grade_required_claims("DCT does not apply.", {"required_claims": ["DCT_NOT_APPLICABLE"]})["correct"] is True
    assert grade_required_claims("DCT applies.", {"required_claims": ["DCT_NOT_APPLICABLE"]})["correct"] is False
    assert grade_required_claims("The Weierstrass M-test does not apply.", {"required_claims": ["Weierstrass M-test"]})["correct"] is False


def test_full_problem_requires_primary_and_claims():
    spec = {"primary": "2", "primary_type": "numeric", "required_claims": ["UNBIASED"]}
    assert grade_full_problem("answer=2; estimator is not unbiased", spec)["correct"] is False
    assert grade_full_problem("answer=3; estimator is unbiased", spec)["correct"] is False
    assert grade_full_problem("answer=2", {"primary": "2", "primary_type": "numeric"})["correct"] is True
    assert grade_full_problem("answer=2", {"primary": "2", "primary_type": "numeric", "required_claims": ["UNKNOWN_CLAIM"]})["correct"] is None


def test_numeric_conclusion_patterns_do_not_scan_arbitrary_numbers():
    spec = {"primary": "1/2", "primary_type": "numeric"}
    assert grade_primary_answer("The integral is 1/2 for every n, hence the limit is 1/2.", spec)["correct"] is True
    assert grade_primary_answer("The proof uses 1/2 in an intermediate bound, but the final value is 1/3.", spec)["correct"] is False


def test_symbolic_prose_conclusion_extraction():
    assert grade_primary_answer(
        "The contour integral equals pi*(e^(-1)-e).",
        {"primary": "pi*(e^(-1)-e)", "primary_type": "symbolic"},
    )["correct"] is True
    assert grade_primary_answer(
        "The minimal polynomial is (t-2)^2.",
        {"primary": "(t-2)^2", "primary_type": "symbolic"},
    )["correct"] is True
    assert grade_primary_answer(
        "The expectation is 1/(mu+lambda).",
        {"primary": "1/(lambda+mu)", "primary_type": "symbolic"},
    )["correct"] is True
    assert grade_primary_answer(
        "The minimal polynomial is not (t-2)^2; it is (t-2)^3.",
        {"primary": "(t-2)^2", "primary_type": "symbolic"},
    )["correct"] is False


def test_nested_equals_uses_top_level_assignment_rhs():
    spec = {"primary": "e^(-2*lambda)*(2*lambda)^2/2!", "primary_type": "symbolic"}
    response = "P(N(3)-N(1)=2)=e^(-2*lambda)*(2*lambda)^2/2!"
    assert grade_primary_answer(response, spec)["correct"] is True


def test_explicit_primary_type_wins_and_plain_legacy_phrase_is_unresolved():
    explicit = {"primary": "compact", "primary_type": "text_alias", "aliases": ["compact"]}
    assert grade_primary_answer("The image is compact.", explicit)["status"] == "CORRECT"
    assert grade_primary_answer("The image is compact.", {"primary": "compact"})["status"] == "UNRESOLVED"


def test_variance_claim_uses_symbolic_rhs_not_substring():
    spec = {"required_claims": ["VAR_BETA_HAT"]}
    assert grade_required_claims("Var(beta_hat)=sigma^2*(X^T X)^(-1)", spec)["status"] == "CORRECT"
    assert grade_required_claims("Cov(beta_hat)=sigma^2*(X^T X)^(-1)", spec)["status"] == "CORRECT"
    assert grade_required_claims("Var(beta_hat)=sigma^2*X^T X", spec)["status"] == "INCORRECT"


def test_text_alias_negative_controls_use_polarity():
    compact = {"primary": "compact", "primary_type": "text_alias", "aliases": ["compact"]}
    assert grade_primary_answer("The image is not compact.", compact)["status"] == "INCORRECT"
    mean = {"primary": "sample_mean", "primary_type": "text_alias", "aliases": ["sample mean"]}
    assert grade_primary_answer("The MLE is not the sample mean.", mean)["status"] == "INCORRECT"

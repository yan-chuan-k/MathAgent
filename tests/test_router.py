from math_agent_core.router import (
    BENCHMARK_DOMAIN_PRIORS,
    classify_problem,
    classify_task_type,
    estimate_verifiability,
)


def test_router_uses_subject_hint():
    route = classify_problem("设 X 为随机变量，求期望。", {"subject": "概率论"})

    assert route["primary_domain"] == "probability"


def test_router_detects_discrete_math_keywords():
    route = classify_problem("用容斥原理计算满足条件的排列个数。", {})

    assert route["primary_domain"] == "discrete_math"


def test_router_separates_stochastic_process_from_probability():
    route = classify_problem("设 {X_n} 为马尔可夫链，求两步转移概率。", {"subject": "随机过程"})

    assert route["primary_domain"] == "stochastic_process"


def test_router_detects_linear_regression():
    route = classify_problem("在线性回归模型中用最小二乘估计回归系数。", {})

    assert route["primary_domain"] == "linear_regression"


def test_router_detects_pde():
    route = classify_problem("求热方程 u_t=u_xx 在给定边值条件下的解。", {})

    assert route["primary_domain"] == "pde"


def test_router_exposes_benchmark_priors():
    route = classify_problem("求一个图的染色数。", {})

    assert BENCHMARK_DOMAIN_PRIORS["discrete_math"] == 21.43
    assert route["priors"]["discrete_math"] == 21.43


def test_router_classifies_task_types_in_chinese_and_english():
    assert classify_task_type("证明每个有限子群都满足该性质。") == "proof"
    assert classify_task_type("Find a counterexample to the claim.") == "counterexample"
    assert classify_task_type("构造一个满足条件的函数。") == "construction"
    assert classify_task_type("Which of the following is correct?") == "choice"
    assert classify_task_type("Derive the recurrence relation.") == "derivation"
    assert classify_task_type("计算 1+1。") == "calculation"


def test_router_returns_task_type_and_verifiability():
    route = classify_problem("计算矩阵 [[1,2],[3,4]] 的行列式。", {})

    assert route["task_type"] == "calculation"
    assert route["verifiability"] == "high"


def test_proof_and_construction_have_low_verifiability():
    assert estimate_verifiability("abstract_algebra", "proof") == "low"
    assert estimate_verifiability("linear_algebra", "construction") == "low"


def test_router_honors_explicit_task_type_metadata():
    route = classify_problem("Determine whether the statement holds.", {"task_type": "proof"})

    assert route["task_type"] == "proof"
    assert route["verifiability"] == "low"


def test_verifiability_normalizes_labels_and_distinguishes_domains():
    assert estimate_verifiability(" LINEAR_ALGEBRA ", " CALCULATION ") == "high"
    assert estimate_verifiability("probability", "calculation") == "medium"

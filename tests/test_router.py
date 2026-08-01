from math_agent_core.router import BENCHMARK_DOMAIN_PRIORS, classify_problem


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

from math_agent_core.router import classify_problem


def test_router_uses_subject_hint():
    route = classify_problem("设 X 为随机变量，求期望。", {"subject": "概率论"})

    assert route["primary_domain"] == "probability"


def test_router_detects_discrete_math_keywords():
    route = classify_problem("用容斥原理计算满足条件的排列个数。", {})

    assert route["primary_domain"] == "discrete_math"

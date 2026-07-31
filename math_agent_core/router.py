from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple


_DOMAIN_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "discrete_math": (
        "离散", "组合", "图", "树", "匹配", "染色", "递推", "生成函数", "鸽巢", "容斥",
        "graph", "combin", "recurrence", "generating function", "pigeonhole",
    ),
    "numerical_analysis": (
        "数值", "插值", "迭代", "牛顿法", "欧拉法", "龙格", "误差", "收敛阶", "稳定性",
        "interpolation", "iteration", "newton method", "runge", "error", "stability",
    ),
    "measure_integration": (
        "测度", "可测", "积分", "勒贝格", "几乎处处", "单调收敛", "控制收敛", "fatou",
        "measure", "measurable", "lebesgue", "almost everywhere", "dominated convergence",
    ),
    "differential_geometry": (
        "微分几何", "流形", "切空间", "度量", "联络", "曲率", "测地线", "第二基本形式",
        "manifold", "tangent", "metric", "connection", "curvature", "geodesic",
    ),
    "probability": (
        "概率", "随机变量", "期望", "方差", "分布", "条件概率", "独立", "马尔可夫", "布朗",
        "probability", "random variable", "expectation", "variance", "distribution", "markov", "brownian",
    ),
    "statistics": (
        "统计", "估计", "置信", "检验", "似然", "回归", "最小二乘", "无偏", "方差分析",
        "statistics", "estimator", "confidence", "hypothesis test", "likelihood", "regression",
    ),
    "abstract_algebra": (
        "群", "环", "域", "理想", "同态", "商群", "有限域", "galois", "伽罗瓦", "多项式环",
        "group", "ring", "field", "ideal", "homomorphism", "quotient", "finite field",
    ),
    "linear_algebra": (
        "矩阵", "行列式", "特征值", "特征向量", "秩", "线性空间", "线性变换", "二次型",
        "matrix", "determinant", "eigen", "rank", "linear transformation", "quadratic form",
    ),
    "complex_analysis": (
        "复", "解析", "全纯", "留数", "围道", "奇点", "洛朗", "共形", "调和函数",
        "complex", "analytic", "holomorphic", "residue", "contour", "singularity", "laurent",
    ),
    "ode": (
        "常微分", "微分方程", "初值问题", "通解", "特解", "wronski", "ode",
        "ordinary differential", "initial value",
    ),
    "pde": (
        "偏微分", "热方程", "波方程", "laplace", "边值", "分离变量", "pde",
        "partial differential", "heat equation", "wave equation", "boundary value",
    ),
    "functional_analysis": (
        "泛函", "banach", "hilbert", "算子", "范数", "弱收敛", "有界线性", "谱",
        "functional analysis", "operator", "normed space", "weak convergence",
    ),
    "topology": (
        "拓扑", "开集", "闭集", "紧致", "连通", "同胚", "基本群", "覆盖",
        "topology", "compact", "connected", "homeomorphism", "fundamental group",
    ),
    "optimization": (
        "运筹", "优化", "线性规划", "单纯形", "对偶", "kkt", "凸", "约束",
        "optimization", "linear programming", "simplex", "dual", "convex",
    ),
    "real_analysis": (
        "数学分析", "极限", "连续", "一致收敛", "级数", "可微", "riemann", "实分析",
        "real analysis", "uniform convergence", "series", "differentiable",
    ),
}

_SUBJECT_ALIASES: Dict[str, str] = {
    "离散数学": "discrete_math",
    "数值分析": "numerical_analysis",
    "测度积分": "measure_integration",
    "微分几何": "differential_geometry",
    "概率论": "probability",
    "随机过程": "probability",
    "统计推断": "statistics",
    "线性回归": "statistics",
    "抽象代数": "abstract_algebra",
    "高等代数": "linear_algebra",
    "复分析": "complex_analysis",
    "常微分方程": "ode",
    "偏微分方程": "pde",
    "泛函分析": "functional_analysis",
    "拓扑学": "topology",
    "运筹学": "optimization",
    "数学分析": "real_analysis",
}


def classify_problem(problem_text: str, metadata: Dict[str, Any] | None = None, limit: int = 4) -> Dict[str, Any]:
    metadata = metadata if isinstance(metadata, dict) else {}
    text = " ".join(str(part or "") for part in _iter_hint_parts(problem_text, metadata)).lower()
    scores: Dict[str, int] = {}

    for alias, domain in _SUBJECT_ALIASES.items():
        if alias.lower() in text:
            scores[domain] = scores.get(domain, 0) + 8

    for domain, keywords in _DOMAIN_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            score += len(re.findall(re.escape(keyword.lower()), text))
        if score:
            scores[domain] = scores.get(domain, 0) + score

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    domains = [domain for domain, _ in ranked[:limit]] or ["unknown"]
    return {
        "primary_domain": domains[0],
        "domain_candidates": domains,
        "scores": {domain: score for domain, score in ranked[:limit]},
    }


def _iter_hint_parts(problem_text: str, metadata: Dict[str, Any]) -> Iterable[str]:
    yield problem_text
    for key in ("subject", "type", "category", "source"):
        if key in metadata:
            yield str(metadata[key])

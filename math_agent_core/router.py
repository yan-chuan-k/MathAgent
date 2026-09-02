from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Tuple


BENCHMARK_DOMAIN_PRIORS: Dict[str, float] = {
    "discrete_math": 21.43,
    "numerical_analysis": 11.61,
    "measure_integration": 9.82,
    "differential_geometry": 8.04,
    "probability": 7.14,
    "abstract_algebra": 7.14,
    "stochastic_process": 6.25,
    "complex_analysis": 6.25,
    "ode": 4.46,
    "statistics": 3.57,
    "functional_analysis": 3.57,
    "linear_regression": 2.68,
    "pde": 2.68,
    "advanced_math": 1.79,
    "linear_algebra": 0.89,
    "optimization": 0.89,
    "real_analysis": 0.89,
    "topology": 0.89,
}

_PRIOR_SCALE = 0.12
_SUBJECT_ALIAS_WEIGHT = 12.0

_DOMAIN_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "discrete_math": (
        "离散", "组合", "图论", "图", "树", "匹配", "染色", "递推", "递归", "生成函数", "母函数", "鸽巢",
        "容斥", "排列", "计数", "整数分拆", "同余", "模", "余数", "整除", "最大公约数", "欧拉函数", "费马", "中国剩余", "burnside",
        "graph", "graphs", "vertex", "vertices", "edge", "edges", "tree", "spanning tree",
        "matching", "coloring", "binary string", "subset", "permutation", "combin", "choose", "in how many ways", "recurrence", "generating function",
        "coefficient of", "pigeonhole", "inclusion-exclusion", "degree sequence", "complete graph", "bipartite", "congruence", "≡", "mod ", "modulo", "remainder",
        "divisibility", "gcd", "chinese remainder", "crt", "euler phi", "phi function", "totient", "euclidean algorithm", "multiplicative order", "primitive root", "derangement", "surjection", "onto function", "stirling", "catalan", "composition", "integer partition", "ramsey", "chromatic", "euler trail", "euler circuit", "hamiltonian", "hall's theorem", "graphical degree sequence", "多少种方法",
    ),
    "numerical_analysis": (
        "数值", "插值", "迭代", "牛顿法", "欧拉法", "龙格", "runge-kutta", "误差", "收敛阶",
        "稳定性", "差分", "求积", "辛普森", "梯形公式", "舍入", "条件数",
        "interpolation", "iteration", "newton method", "runge", "error", "stability", "quadrature",
    ),
    "measure_integration": (
        "测度", "可测", "勒贝格", "lebesgue", "积分", "几乎处处", "单调收敛", "控制收敛",
        "fatou", "tonelli", "fubini", "可积", "sigma", "σ代数", "外测度",
        "measure", "measurable", "almost everywhere", "dominated convergence",
    ),
    "differential_geometry": (
        "微分几何", "流形", "切空间", "余切", "度量", "联络", "曲率", "测地线", "第二基本形式",
        "黎曼", "christoffel", "ricci", "gauss", "曲面", "第一基本形式",
        "manifold", "tangent", "metric", "connection", "curvature", "geodesic",
    ),
    "probability": (
        "概率", "随机变量", "期望", "方差", "分布", "条件概率", "独立", "贝叶斯", "协方差",
        "大数定律", "中心极限定理", "矩母函数", "特征函数",
        "probability", "random variable", "expectation", "variance", "distribution", "bayes",
    ),
    "stochastic_process": (
        "随机过程", "马尔可夫", "布朗", "泊松过程", "更新过程", "平稳过程", "转移概率",
        "鞅", "停时", "kolmogorov", "markov", "brownian", "poisson process", "martingale",
    ),
    "statistics": (
        "统计推断", "统计", "估计", "置信", "检验", "似然", "最大似然", "无偏", "方差分析",
        "充分统计量", "一致估计", "假设检验", "statistics", "estimator", "confidence",
        "hypothesis test", "likelihood",
    ),
    "linear_regression": (
        "线性回归", "回归", "最小二乘", "ols", "残差", "设计矩阵", "正规方程", "回归系数",
        "linear regression", "least squares", "normal equation", "residual",
    ),
    "abstract_algebra": (
        "抽象代数", "群", "环", "域", "理想", "同态", "同构", "商群", "正规子群", "有限域",
        "galois", "伽罗瓦", "多项式环", "单群", "陪集", "group", "ring", "field", "ideal",
        "homomorphism", "quotient", "finite field",
    ),
    "linear_algebra": (
        "高等代数", "线性代数", "矩阵", "行列式", "特征值", "特征向量", "秩", "线性空间",
        "线性变换", "二次型", "相似", "对角化", "jordan", "matrix", "determinant", "eigen",
        "rank", "linear transformation", "quadratic form",
    ),
    "complex_analysis": (
        "复分析", "复变", "解析", "全纯", "留数", "围道", "奇点", "洛朗", "共形", "调和函数",
        "最大模", "柯西积分", "rouche", "complex", "analytic", "holomorphic", "residue",
        "contour", "singularity", "laurent",
    ),
    "ode": (
        "常微分", "微分方程", "初值问题", "通解", "特解", "wronski", "稳定解", "线性系统",
        "ordinary differential", "initial value", "ode",
    ),
    "pde": (
        "偏微分", "热方程", "波方程", "laplace", "poisson方程", "边值", "分离变量", "初边值",
        "特征线", "partial differential", "heat equation", "wave equation", "boundary value", "pde",
    ),
    "functional_analysis": (
        "泛函", "banach", "hilbert", "算子", "范数", "弱收敛", "有界线性", "谱", "紧算子",
        "内积空间", "开映射", "一致有界", "functional analysis", "operator", "normed space",
        "weak convergence",
    ),
    "topology": (
        "拓扑", "开集", "闭集", "紧致", "连通", "同胚", "基本群", "覆盖", "商空间", "分离公理",
        "topology", "compact", "connected", "homeomorphism", "fundamental group",
    ),
    "optimization": (
        "运筹", "优化", "线性规划", "单纯形", "对偶", "kkt", "凸", "约束", "可行域",
        "最优性", "optimization", "linear programming", "simplex", "dual", "convex",
    ),
    "real_analysis": (
        "数学分析", "实分析", "极限", "连续", "一致收敛", "级数", "可微", "riemann", "导数",
        "闭区间", "一致连续", "real analysis", "uniform convergence", "series", "differentiable",
    ),
    "advanced_math": (
        "非基础", "进阶课程", "高级", "综合", "交叉", "advanced", "graduate",
    ),
}

DISCRETE_SUBTYPE_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "combinatorial_counting": (
        "组合计数", "计数", "排列", "组合", "容斥", "鸽巢", "双计数", "bijection", "双射",
        "inclusion-exclusion", "pigeonhole", "permutation", "combination", "counting", "choose",
        "binary string", "how many ways", "in how many ways", "subset", "subsets", "ordered selection", "stars and bars",
        "derangement", "surjection", "onto function", "stirling", "catalan", "composition", "integer partition",
    ),
    "recurrence": (
        "递推", "递归", "递推式", "递推关系", "recurrence", "recursive", "characteristic equation",
    ),
    "generating_function": (
        "生成函数", "母函数", "generating function", "ordinary generating function", "exponential generating function",
        "coefficient extraction", "系数提取", "coefficient of", "[x^",
    ),
    "graph_theory": (
        "图论", "图", "顶点", "边", "树", "生成树", "匹配", "染色", "连通", "路径", "回路", "度数",
        "graph", "vertex", "vertices", "edge", "edges", "tree", "spanning tree", "matching", "coloring",
        "connected", "path", "cycle", "degree sequence", "complete graph", "bipartite",
        "ramsey", "chromatic", "euler trail", "euler circuit", "hamiltonian", "hall's theorem", "graphical degree sequence",
    ),
    "number_theory_modular": (
        "数论", "同余", "模", "余数", "整除", "最大公约数", "欧拉函数", "费马", "中国剩余",
        "congruence", "≡", "modulo", "mod ", "remainder", "divisibility", "gcd", "totient", "fermat",
        "chinese remainder", "crt", "euler phi", "phi function", "totient", "euclidean algorithm", "multiplicative order", "primitive root",
    ),
}

_DISCRETE_SUBTYPE_PRIORITY: Tuple[str, ...] = (
    "generating_function",
    "recurrence",
    "graph_theory",
    "number_theory_modular",
    "combinatorial_counting",
)


_SUBJECT_ALIASES: Dict[str, str] = {
    "离散数学": "discrete_math",
    "组合数学": "discrete_math",
    "图论": "discrete_math",
    "数论": "discrete_math",
    "数值分析": "numerical_analysis",
    "测度积分": "measure_integration",
    "测度论": "measure_integration",
    "实变函数": "measure_integration",
    "微分几何": "differential_geometry",
    "概率论": "probability",
    "随机过程": "stochastic_process",
    "统计推断": "statistics",
    "统计学": "statistics",
    "线性回归": "linear_regression",
    "抽象代数": "abstract_algebra",
    "近世代数": "abstract_algebra",
    "高等代数": "linear_algebra",
    "线性代数": "linear_algebra",
    "复分析": "complex_analysis",
    "复变函数": "complex_analysis",
    "常微分方程": "ode",
    "偏微分方程": "pde",
    "泛函分析": "functional_analysis",
    "拓扑学": "topology",
    "运筹学": "optimization",
    "数学分析": "real_analysis",
    "非基础及进阶课程": "advanced_math",
}


_TASK_TYPE_PATTERNS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("counterexample", ("反例", "counterexample", "disprove")),
    ("construction", ("构造", "construct")),
    ("choice", ("选择题", "which of the following", "正确的是", "不正确的是", "选项")),
    ("proof", ("证明", "prove", "show that")),
    ("derivation", ("推导", "derive", "deduce")),
    (
        "calculation",
        (
            "计算",
            "求解",
            "求出",
            "求值",
            "求",
            "compute",
            "calculate",
            "evaluate",
            "solve",
            "find",
            "determine",
        ),
    ),
)

_TASK_TYPES = {task_type for task_type, _ in _TASK_TYPE_PATTERNS} | {"unknown"}


def classify_task_type(problem_text: str) -> str:
    text = str(problem_text or "").strip().lower()
    if not text:
        return "unknown"
    for task_type, markers in _TASK_TYPE_PATTERNS:
        if any(marker in text for marker in markers):
            return task_type
    if re.search(r"\d\s*[+\-*/^]\s*\d", text) or "=?" in text.replace(" ", ""):
        return "calculation"
    return "unknown"


def classify_discrete_subtype(problem_text: str, metadata: Dict[str, Any] | None = None) -> str:
    """Classify a discrete-math problem into an existing Phase-B subtype.

    Keyword scoring remains the base mechanism. Narrow structural boosts prevent
    generic counting words from overriding more diagnostic mathematical forms.
    """
    metadata = metadata if isinstance(metadata, dict) else {}
    text = " ".join(str(part or "") for part in _iter_hint_parts(problem_text, metadata)).lower()
    # Explicitly requested methods outrank object vocabulary within the existing
    # five discrete subtypes. No generic routing architecture changes are made.
    if re.search(
        r"\b(?:using|use|via|derive|with)\s+(?:an?\s+)?(?:ordinary\s+|exponential\s+)?generating\s+functions?\b"
        r"|用生成函数|使用生成函数|通过生成函数",
        text,
    ):
        return "generating_function"
    if re.search(
        r"\b(?:using|use|via|derive|solve\s+by|with)\s+(?:an?\s+)?recurrence\b"
        r"|用递推|使用递推|通过递推",
        text,
    ):
        return "recurrence"

    scores: Dict[str, float] = {}
    for subtype, keywords in DISCRETE_SUBTYPE_KEYWORDS.items():
        score = 0.0
        for keyword in keywords:
            occurrences = len(re.findall(re.escape(keyword.lower()), text))
            if occurrences:
                score += occurrences * _keyword_weight(keyword)
        if score:
            scores[subtype] = score

    structural_boosts = {
        "generating_function": (
            r"(?:ordinary|exponential)?\s*generating\s+function",
            r"coefficient\s+of\s+x\^?\{?\d+\}?",
            r"\[x\^",
            r"生成函数|母函数|求.*?系数",
        ),
        "recurrence": (
            r"\b[a-zA-Z]_[{]?n[}]?\s*=.*?[a-zA-Z]_[{]?n[-−]1[}]?",
            r"closed\s+form.*?recurrence|recurrence.*?closed\s+form",
            r"递推(?:关系|式)?.*?[a-zA-Z]?_?n",
        ),
        "graph_theory": (
            r"spanning\s+tree|complete\s+(?:bi)?partite\s+graph|complete\s+graph",
            r"\bgraph\b|\bvertices?\b|\bedges?\b|\bmatching\b|\bcoloring\b",
            r"图论|生成树|顶点|匹配|染色|度数",
        ),
        "number_theory_modular": (
            r"≡|\\equiv|\bcongruence\b|\bmod(?:ulo)?\b|chinese\s+remainder|\bcrt\b",
            r"同余|中国剩余|模\s*\d+|余数",
        ),
        "combinatorial_counting": (
            r"binary\s+strings?|choose\s+\d+.*?from\s+\d+|in\s+how\s+many\s+ways",
            r"ordered\s+selections?|inclusion[- ]exclusion|pigeonhole|permutations?|subsets?|stars[- ]and[- ]bars",
            r"(?:nonnegative|non-negative|positive)\s+integer\s+(?:tuples?|pairs?|triples?|solutions?)",
            r"二进制(?:字符串|串)|多少种|多少个|容斥|排列|组合",
        ),
    }
    for subtype, patterns in structural_boosts.items():
        hits = sum(1 for pattern in patterns if re.search(pattern, text))
        if hits:
            scores[subtype] = scores.get(subtype, 0.0) + 5.0 * hits

    if not scores:
        return "general_discrete"
    ranked = sorted(
        scores.items(),
        key=lambda item: (-item[1], _DISCRETE_SUBTYPE_PRIORITY.index(item[0])),
    )
    return ranked[0][0]


def estimate_verifiability(domain: str, task_type: str) -> str:
    domain = str(domain or "unknown").strip().lower()
    task_type = str(task_type or "unknown").strip().lower()
    if task_type in {"proof", "construction", "counterexample"}:
        return "low"
    if task_type == "choice":
        return "medium"
    if task_type == "calculation":
        if domain in {"linear_algebra", "calculus", "numerical_analysis"}:
            return "high"
        if domain in {"ode", "optimization"}:
            return "medium"
        return "medium"
    if task_type == "derivation":
        return "medium"
    return "low"


def estimate_difficulty(problem_text: str, domain: str, task_type: str, verifiability: str) -> str:
    """Deterministic difficulty heuristic; intentionally no model call."""
    text = str(problem_text or "")
    lowered = text.lower()
    complexity = 0
    if len(text) > 500:
        complexity += 1
    if len(re.findall(r"\b(?:prove|show|derive|case|suppose|whereas|if and only if)\b", lowered)) >= 2:
        complexity += 1
    if task_type in {"proof", "construction", "counterexample"}:
        complexity += 2
    if domain in {"measure_integration", "functional_analysis", "topology", "differential_geometry", "stochastic_process"}:
        complexity += 1
    if verifiability == "low":
        complexity += 1
    if complexity >= 3:
        return "hard"
    if complexity >= 1 or verifiability == "medium":
        return "medium"
    return "easy"


def classify_problem(problem_text: str, metadata: Dict[str, Any] | None = None, limit: int = 4) -> Dict[str, Any]:
    metadata = metadata if isinstance(metadata, dict) else {}
    text = " ".join(str(part or "") for part in _iter_hint_parts(problem_text, metadata)).lower()
    scores: Dict[str, float] = {}

    # Explicit metadata labels are trusted routing hints when they name a known domain.
    for key in ("subject", "type", "category"):
        value = str(metadata.get(key) or "").strip().lower()
        if value in _DOMAIN_KEYWORDS:
            scores[value] = scores.get(value, 0.0) + _SUBJECT_ALIAS_WEIGHT

    for alias, domain in _SUBJECT_ALIASES.items():
        if alias.lower() in text:
            scores[domain] = scores.get(domain, 0.0) + _SUBJECT_ALIAS_WEIGHT

    for domain, keywords in _DOMAIN_KEYWORDS.items():
        score = 0.0
        for keyword in keywords:
            occurrences = len(re.findall(re.escape(keyword.lower()), text))
            if occurrences:
                score += occurrences * _keyword_weight(keyword)
        if score:
            scores[domain] = scores.get(domain, 0.0) + score

    for domain in list(scores):
        scores[domain] += BENCHMARK_DOMAIN_PRIORS.get(domain, 0.0) * _PRIOR_SCALE

    ranked = sorted(
        scores.items(),
        key=lambda item: (-item[1], -BENCHMARK_DOMAIN_PRIORS.get(item[0], 0.0), item[0]),
    )
    domains = [domain for domain, _ in ranked[:limit]] or ["unknown"]
    explicit_task_type = str(metadata.get("task_type") or "").strip().lower()
    task_type = explicit_task_type if explicit_task_type in _TASK_TYPES else classify_task_type(problem_text)
    verifiability = estimate_verifiability(domains[0], task_type)
    difficulty = estimate_difficulty(problem_text, domains[0], task_type, verifiability)
    discrete_subtype = classify_discrete_subtype(problem_text, metadata) if domains[0] == "discrete_math" else None
    return {
        "primary_domain": domains[0],
        "domain_candidates": domains,
        "task_type": task_type,
        "verifiability": verifiability,
        "difficulty": difficulty,
        "discrete_subtype": discrete_subtype,
        "scores": {domain: round(score, 3) for domain, score in ranked[:limit]},
        "priors": {domain: BENCHMARK_DOMAIN_PRIORS.get(domain, 0.0) for domain in domains},
    }


def _keyword_weight(keyword: str) -> float:
    if len(keyword) >= 4 or " " in keyword or "-" in keyword:
        return 2.0
    return 1.0


def _iter_hint_parts(problem_text: str, metadata: Dict[str, Any]) -> Iterable[str]:
    yield problem_text
    for key in ("subject", "type", "category", "source"):
        if key in metadata:
            yield str(metadata[key])

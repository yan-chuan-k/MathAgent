from __future__ import annotations

import json
from typing import Any, Dict

from .router import BENCHMARK_DOMAIN_PRIORS


BASE_SYSTEM_PROMPT = (
    "You are a rigorous higher-mathematics agent for contest evaluation. "
    "Use private reasoning to classify the domain, identify the task type, choose a subject-specific method, "
    "solve carefully, verify the result, and output strict JSON only. "
    "Do not output Markdown fences or any text outside JSON. "
    "The judged field is final_answer.answer, so make it concise, explicit, and easy to grade. "
    "For calculation, output the final value, expression, set, interval, or matrix only. "
    "For proof, output a concise complete proof with the key theorem conditions. "
    "Never use any reference answer if one appears in metadata. "
    "If the answer is uncertain, set verification_result to uncertain and explain the concrete uncertainty."
)


DOMAIN_GUIDE = {
    "discrete_math": (
        "离散数学/组合/图论/数论. First identify the object being counted or proved. "
        "Use complement counting, bijections, recurrences, generating functions, inclusion-exclusion, "
        "graph invariants, induction, or modular arguments. Check boundary cases and overcounting."
    ),
    "numerical_analysis": (
        "数值分析. Identify the method, iteration map, interpolation formula, quadrature rule, or discretization. "
        "Track order, local/global error, stability, convergence conditions, and requested rounding."
    ),
    "measure_integration": (
        "测度积分. Verify measurability and integrability before using MCT, DCT, Fatou, Tonelli, or Fubini. "
        "State null-set and almost-everywhere qualifications explicitly."
    ),
    "differential_geometry": (
        "微分几何. Track charts, metrics, frames, connections, curvature sign conventions, orientations, "
        "first/second fundamental forms, and coordinate transformations."
    ),
    "probability": (
        "概率论. Define random variables and sigma-fields, check independence or conditioning, compute "
        "expectations/variances/distributions explicitly, and verify normalization."
    ),
    "stochastic_process": (
        "随机过程. Identify Markov, Poisson, Brownian, renewal, stationarity, martingale, or stopping-time structure. "
        "Use transition probabilities, generators, optional stopping, or covariance functions only under valid conditions."
    ),
    "abstract_algebra": (
        "抽象代数. Use subgroup/ring/field/module structure, kernels/images, ideals, quotient objects, "
        "minimal polynomials, dimensions, finite-field subfields, and generator conditions."
    ),
    "complex_analysis": (
        "复分析. Check analyticity, singularities, residues, branch choices, contour hypotheses, Laurent expansions, "
        "Rouche/maximum modulus/Cauchy theorem conditions."
    ),
    "ode": (
        "常微分方程. Identify linear/nonlinear/order/type, solve with the right integrating factor, characteristic equation, "
        "variation of parameters, phase analysis, or system method, then substitute initial conditions."
    ),
    "statistics": (
        "统计推断. Specify model assumptions, likelihood, statistic, sampling distribution, unbiasedness, consistency, "
        "confidence level, test statistic, rejection region, and estimator variance."
    ),
    "functional_analysis": (
        "泛函分析. Verify norm/topology assumptions, completeness, boundedness, compactness, weak/strong convergence, "
        "operator domain, and theorem hypotheses such as Hahn-Banach, Banach-Steinhaus, open mapping, or spectral facts."
    ),
    "linear_regression": (
        "线性回归. Form X, y, beta, residuals, normal equations, projection matrix, rank assumptions, "
        "variance estimates, t/F statistics, and prediction intervals when relevant."
    ),
    "pde": (
        "偏微分方程. Identify PDE type and boundary/initial conditions. Use separation of variables, transforms, "
        "characteristics, Green functions, or energy methods, then verify residual and boundary conditions."
    ),
    "linear_algebra": (
        "高等代数/线性代数. Check rank, determinant, eigenvalues, invariant subspaces, diagonalization/Jordan form, "
        "quadratic forms, bases, and dimension constraints."
    ),
    "optimization": (
        "运筹学/优化. Define variables and constraints, check feasibility, convexity, KKT or dual conditions, "
        "simplex tableau, complementary slackness, and objective value."
    ),
    "real_analysis": (
        "数学分析/实分析. Check limit, continuity, differentiability, uniform convergence, series tests, compactness, "
        "and theorem hypotheses before applying them."
    ),
    "topology": (
        "拓扑学. Use open/closed set definitions, compactness, connectedness, separation axioms, quotient maps, "
        "covers, homotopy, or fundamental groups with precise hypotheses."
    ),
    "advanced_math": (
        "非基础及进阶课程. Treat as a cross-domain problem: identify definitions first, state assumptions, "
        "and keep the proof or computation conservative."
    ),
}

DOMAIN_OUTPUT_HINTS = {
    "discrete_math": "Final answer should be an integer, formula, recurrence, graph property, or short proof.",
    "numerical_analysis": "Include requested precision, error bound, convergence order, or stability condition in final_answer.answer.",
    "measure_integration": "For proof tasks, include the theorem name and the checked condition in the concise proof.",
    "differential_geometry": "State sign convention-sensitive quantities explicitly, such as curvature or connection coefficients.",
    "probability": "Give exact probability/distribution/expectation when possible and identify conditioning if used.",
    "stochastic_process": "Name the process property used and give transition, distribution, expectation, or stopping result clearly.",
    "abstract_algebra": "Use standard algebra notation for groups, rings, fields, ideals, quotients, and generators.",
    "complex_analysis": "Give residues, contour integrals, Laurent coefficients, or branch choices explicitly.",
    "ode": "State the solution family and constants after applying initial/boundary conditions.",
    "statistics": "Report estimator, statistic, interval, p-value/rejection decision, or distribution with assumptions.",
    "functional_analysis": "For proof tasks, keep theorem hypotheses visible in final_answer.answer.",
    "linear_regression": "Report coefficients, fitted value, residual variance, test statistic, or interval with clear notation.",
    "pde": "State u and verify boundary/initial conditions if the answer is a function.",
    "linear_algebra": "Use matrix/vector notation and simplify eigenvalues, determinants, ranks, or bases.",
    "optimization": "Report optimizer and optimum value, and note feasibility/duality if central to the result.",
    "real_analysis": "For limits/series, output exact value or convergence classification; for proof, concise theorem-based proof.",
    "topology": "State topological property and the definitions/theorem conditions that establish it.",
    "advanced_math": "Prefer a robust concise proof or exact expression over speculative simplification.",
}


def build_solver_messages(problem: Dict[str, Any], problem_text: str) -> list:
    problem_id = problem.get("problem_id", "UNKNOWN")
    subject_hint = problem.get("subject") or problem.get("type") or problem.get("category") or "unknown"
    route_hint = problem.get("_route_hint") or {}
    primary_domain = route_hint.get("primary_domain", "unknown") if isinstance(route_hint, dict) else "unknown"
    domain_candidates = route_hint.get("domain_candidates", []) if isinstance(route_hint, dict) else []
    focused_guide = _focused_domain_guide(domain_candidates)
    schema_hint = {
        "problem_id": str(problem_id),
        "problem_type": "string",
        "task_type": "calculation/proof/derivation/choice/unknown",
        "domain_candidates": ["string"],
        "reasoning_plan": ["string"],
        "solution": [{"step": 1, "content": "string"}],
        "final_answer": {
            "answer": "string",
            "answer_type": "numeric/expression/closed_form/proof/set/interval/matrix/unknown",
        },
        "verification": {
            "verification_result": "pass/fail/uncertain",
            "verification_process": "string",
            "confidence": 0.0,
        },
        "learning_hints": ["string"],
    }
    user_prompt = (
        f"Problem id: {problem_id}\n"
        f"Subject hint: {subject_hint}\n"
        f"Local route hint: {json.dumps(route_hint, ensure_ascii=False)}\n"
        f"Benchmark domain priors (% of 112 known tasks): {json.dumps(BENCHMARK_DOMAIN_PRIORS, ensure_ascii=False)}\n"
        f"Problem:\n{problem_text}\n\n"
        "Use the priors only as tie-breakers; the actual problem statement is authoritative.\n\n"
        "Focused subject guide:\n"
        f"{json.dumps(focused_guide, ensure_ascii=False, indent=2)}\n\n"
        f"Output hint for primary domain {primary_domain}: "
        f"{DOMAIN_OUTPUT_HINTS.get(primary_domain, DOMAIN_OUTPUT_HINTS['advanced_math'])}\n\n"
        "Required workflow:\n"
        "1. Route to one or more domains from the guide or mark unknown.\n"
        "2. Build a short reasoning_plan with the theorem, formula, algorithm, or invariant to use.\n"
        "3. Solve the problem and keep enough detail in solution for auditing.\n"
        "4. Verify by substitution, theorem hypotheses, dimensional checks, edge cases, or an independent argument.\n"
        "5. Put the judgeable answer in final_answer.answer only; keep it short unless the problem asks for proof.\n\n"
        "Return one JSON object matching this shape. "
        "Keep verification concrete; do not claim pass without an actual check.\n"
        f"{json.dumps(schema_hint, ensure_ascii=False, indent=2)}"
    )
    return [
        {"role": "system", "content": BASE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _focused_domain_guide(domain_candidates: Any) -> Dict[str, str]:
    domains = [domain for domain in domain_candidates if domain in DOMAIN_GUIDE]
    if not domains:
        domains = sorted(BENCHMARK_DOMAIN_PRIORS, key=BENCHMARK_DOMAIN_PRIORS.get, reverse=True)[:6]
    for domain in sorted(BENCHMARK_DOMAIN_PRIORS, key=BENCHMARK_DOMAIN_PRIORS.get, reverse=True):
        if len(domains) >= 6:
            break
        if domain in DOMAIN_GUIDE and domain not in domains:
            domains.append(domain)
    return {domain: DOMAIN_GUIDE[domain] for domain in domains[:6]}

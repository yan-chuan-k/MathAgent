from __future__ import annotations

import json
from typing import Any, Dict


BASE_SYSTEM_PROMPT = (
    "You are a rigorous higher-mathematics agent for contest evaluation. "
    "Use internal deep reasoning before answering: classify the domain, identify the task type, "
    "choose a subject-specific method, solve carefully, verify the result, and output strict JSON only. "
    "Do not output Markdown fences or any text outside JSON. "
    "The judged field is final_answer.answer, so make it concise, explicit, and easy to grade. "
    "For calculation, output the final value/expression only. For proof, output a concise complete proof. "
    "Do not use any reference answer if one appears in metadata. "
    "If the answer is uncertain, set verification_result to uncertain and explain the concrete uncertainty."
)


DOMAIN_GUIDE = {
    "discrete_math": (
        "离散数学/组合/图论/数论: check definitions, count complements when useful, "
        "state bijections or induction invariants, and verify edge cases."
    ),
    "numerical_analysis": (
        "数值分析: identify the algorithm, order, stability, convergence condition, "
        "error term, and required rounding or norm."
    ),
    "measure_integration": (
        "测度积分: verify measurability, integrability, limit-exchange conditions, "
        "and use MCT/DCT/Fatou/Tonelli/Fubini only when hypotheses hold."
    ),
    "differential_geometry": (
        "微分几何: track charts, metrics, connections, curvature sign conventions, "
        "orientations, and tensor coordinate transformations."
    ),
    "probability": (
        "概率论/随机过程/统计推断/线性回归: define random variables and sigma-fields, "
        "check independence/conditioning, compute expectations and variances explicitly, "
        "and verify distributional assumptions."
    ),
    "abstract_algebra": (
        "抽象代数/高等代数: use subgroup/ring/field/module structure, homomorphism kernels, "
        "minimal polynomials, dimensions, and generator conditions."
    ),
    "complex_analysis": (
        "复分析: check analyticity, singularities, residues, branch choices, contours, "
        "and theorem hypotheses."
    ),
    "ode_pde": (
        "常微分方程/偏微分方程: identify type, boundary or initial conditions, "
        "solve and substitute back to verify residuals."
    ),
    "functional_topology_analysis": (
        "泛函分析/拓扑/数学分析: verify norm/topology assumptions, compactness, completeness, "
        "continuity, convergence mode, and theorem hypotheses."
    ),
    "optimization": (
        "运筹学/优化: state variables, constraints, convexity or KKT/duality conditions, "
        "and verify feasibility and optimality."
    ),
}


def build_solver_messages(problem: Dict[str, Any], problem_text: str) -> list:
    problem_id = problem.get("problem_id", "UNKNOWN")
    subject_hint = problem.get("subject") or problem.get("type") or problem.get("category") or "unknown"
    route_hint = problem.get("_route_hint") or {}
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
        f"Problem:\n{problem_text}\n\n"
        "High-frequency subject guide for this benchmark:\n"
        f"{json.dumps(DOMAIN_GUIDE, ensure_ascii=False, indent=2)}\n\n"
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

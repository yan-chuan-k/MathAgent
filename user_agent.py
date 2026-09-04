from __future__ import annotations

import inspect
import re
from typing import Any, Dict, List

from math_agent_core.answer_utils import DEFAULT_FALLBACK, extract_final_answer, normalize_final_response
from math_agent_core.router import classify_problem
from math_agent_core.trace_utils import make_trace_step, sanitize_trace, trace_from_orchestrator_result


_SCORE_FIRST_ANSWER_SYSTEM_PROMPT = """You are a high-accuracy mathematics competition solver.

Solve the problem carefully using internal reasoning.

Output exactly ONE visible line:

Final answer: <complete requested answer>

Then stop.

Do not output JSON.
Do not repeat the problem.
Do not provide visible explanation or derivation.

Preserve all requested roots, conditions, intervals,
moduli, matrices, vectors, sets, and multiple answer parts.

The subject/strategy hint is advisory. If the mathematical structure of the problem
indicates a different method, follow the problem itself.

""".strip()

_SCORE_FIRST_PROOF_SYSTEM_PROMPT = """You are a high-accuracy mathematics competition solver.

State the conclusion first, then give a concise but complete proof.

Do not output JSON.
Do not repeat the problem.

Do not omit necessary logical steps merely to shorten the response.

The subject/strategy hint is advisory. If the mathematical structure of the problem
indicates a different method, follow the problem itself.

""".strip()

_SCORE_FIRST_DOMAIN_STRATEGIES = {
    "discrete_math": (
        "Identify the exact combinatorial, graph, recurrence, generating-function, or number-theoretic "
        "structure before calculating. Track every restriction explicitly and check small cases when possible. "
        "Avoid solving an unrestricted version and forgetting a constraint."
    ),
    "numerical_analysis": (
        "Identify approximation, convergence, error order, interpolation, quadrature, or iteration first. "
        "Write the exact error expression or iteration before calculating. Distinguish local error, global error, "
        "and stability; for Newton, interpolation, quadrature, and finite differences check their defining assumptions."
    ),
    "measure_integration": (
        "Separate pointwise facts from measure-theoretic facts. Check measurability and integrability before "
        "manipulating integrals. Before exchanging limits and integrals, verify the hypotheses of MCT, Fatou, "
        "DCT, or Fubini/Tonelli rather than invoking them from convergence alone."
    ),
    "differential_geometry": (
        "Identify the geometric objects and coordinate convention first. For surfaces compute the metric or "
        "first fundamental form before curvature. Distinguish intrinsic quantities from embedding-dependent ones, "
        "track signs and orientation, and verify the applicable curvature formula before substitution."
    ),
    "probability": (
        "Define the random variables and conditioning events first. Prefer exact distributions, symmetry, "
        "indicator variables, and conditioning before direct enumeration. Check independence rather than assuming it, "
        "and verify normalization; for expectations consider linearity before deriving a full distribution."
    ),
    "abstract_algebra": (
        "Identify the algebraic structure and the property being tested. Use kernels, images, normality, ideals, "
        "quotients, orders, and homomorphism theorems before element-wise computation. Never assume commutativity; "
        "for finite groups use Lagrange and order arguments early."
    ),
    "stochastic_process": (
        "Identify the process, time parameter or filtration, and requested quantity. For Markov chains write the "
        "transition structure explicitly; for hitting or stopping questions derive first-step equations. Check "
        "recurrence, stationarity, and independence assumptions before using them."
    ),
    "complex_analysis": (
        "Locate singularities and determine their types before integrating. Choose Cauchy formulas, residues, "
        "Laurent series, or parametrization according to the geometry. Track contour orientation and which "
        "singularities lie inside; use the simplest exact residue formula available."
    ),
    "ode": (
        "Classify the ODE before manipulating it: separable, linear, exact, Bernoulli, constant-coefficient, "
        "Euler-Cauchy, system, or qualitative. Obtain the general solution before applying all initial or boundary "
        "conditions, and check whether division discarded singular solutions."
    ),
    "statistics": (
        "Write the likelihood, estimator, or test target explicitly. Distinguish estimator, estimate, bias, variance, "
        "and sampling distribution. For tests identify null and alternative, statistic, and reference distribution; "
        "check regularity assumptions before asymptotic results."
    ),
    "functional_analysis": (
        "Identify the normed or topological structure before using finite-dimensional intuition. Check boundedness, "
        "completeness, compactness, and continuity separately. For operators distinguish norm, spectrum, eigenvalues, "
        "and invertibility; use major functional-analysis theorems only when their hypotheses hold."
    ),
    "linear_regression": (
        "Write the model as y = Xβ + ε. Check rank assumptions before using (X^T X)^(-1). Distinguish fitted values, "
        "residuals, coefficient estimates, sampling variance, and prediction variance; use projection geometry when "
        "it simplifies the calculation."
    ),
    "pde": (
        "Classify the PDE together with boundary and initial conditions first. Choose separation of variables, "
        "characteristics, transforms, energy methods, or fundamental solutions according to equation type. Never "
        "ignore boundary conditions, and use the correct boundary eigenbasis in expansions."
    ),
    "advanced_math": (
        "Translate the problem into precise mathematical objects and requested outputs. Identify the governing "
        "theorem or structure before calculation, check assumptions and edge cases, and preserve every requested "
        "answer component. Prefer exact symbolic reasoning when feasible without forcing an unsupported specialized theorem."
    ),
    "linear_algebra": (
        "Identify systems, rank, basis, maps, eigenstructure, quadratic forms, or decompositions before computing. "
        "Exploit invariants before expanding determinants, check algebraic and geometric multiplicities separately, "
        "and use rank-nullity before brute-force elimination when appropriate."
    ),
    "optimization": (
        "Determine feasibility and convexity first. For smooth unconstrained problems solve stationary conditions "
        "and classify them; for constrained problems use KKT or Lagrange conditions and check boundary cases. "
        "A stationary point is not automatically a global optimum."
    ),
    "real_analysis": (
        "Read all quantifiers before choosing a theorem. Distinguish pointwise, uniform, absolute, and Lp convergence, "
        "and check compactness, boundedness, or completeness hypotheses explicitly. For proofs start from the exact "
        "epsilon-delta or sequence criterion required."
    ),
    "topology": (
        "Work from definitions when uncertain. Track whether the claim concerns open, closed, compact, connected, "
        "path-connected, Hausdorff, or continuous properties. Do not import metric-space facts into arbitrary spaces; "
        "for continuous maps use inverse images and preservation properties carefully."
    ),
}

_SCORE_FIRST_DISCRETE_SUBTYPE_STRATEGIES = {
    "combinatorial_counting": (
        "Determine whether order and repetition matter, and translate every restriction before choosing a formula. "
        "Check overcounting, inclusion-exclusion, bijections, or a recurrence before brute-force expansion."
    ),
    "recurrence": (
        "Write the recurrence together with every initial condition and verify the first few terms before solving it. "
        "Watch index shifts and repeated characteristic roots."
    ),
    "generating_function": (
        "Define exactly what coefficient represents the requested quantity and fix the index convention first. "
        "Track shifts carefully before manipulating or extracting coefficients."
    ),
    "graph_theory": (
        "Identify whether the question concerns degree, paths, cycles, coloring, matching, planarity, or extremal "
        "structure. Check every theorem hypothesis before applying a graph invariant."
    ),
    "number_theory_modular": (
        "Reduce modulo m early, check gcd before dividing or taking an inverse, and distinguish a residue class from "
        "its least nonnegative representative. Use CRT only after checking compatibility."
    ),
}

_SCORE_FIRST_EXACTNESS_DISCIPLINE = (
    "By default, prefer an exact mathematical form; if approximation is requested, honor the requested precision. "
    "Preserve units, domains, moduli, multiplicities, and all requested parts."
)

_SCORE_FIRST_DOMAIN_FINAL_CHECKS = {
    "discrete_math": "Recheck one cheap small case, invariant, or direct substitution appropriate to the discrete structure.",
    "numerical_analysis": "Recompute one residual, error order, interpolation condition, quadrature scaling, or stability quantity.",
    "measure_integration": "Verify every hypothesis of the measure/integration theorem actually used.",
    "differential_geometry": "Recheck the metric, convention, sign, and one known special case when available.",
    "probability": "Check support, normalization, probability range, or expectation bounds against the result.",
    "abstract_algebra": "Recheck the relevant closure, divisibility, kernel, ideal, quotient, or order condition.",
    "stochastic_process": "Recheck normalization, boundary values, or the first-step/stationarity equation used.",
    "complex_analysis": "Recheck singularities, contour inclusion/orientation, and the relevant Laurent/residue coefficient.",
    "ode": "Substitute the solution into the ODE and every supplied initial or boundary condition.",
    "statistics": "Recheck model support, parameterization, reference distribution, and any boundary case.",
    "functional_analysis": "Recheck theorem hypotheses and avoid any conclusion valid only in finite dimensions.",
    "linear_regression": "Recheck dimensions, rank assumptions, and the relevant normal-equation or projection identity.",
    "pde": "Substitute the result into the PDE and verify every supplied initial and boundary condition.",
    "advanced_math": "Use the cheapest independent substitution, inverse operation, special case, or range/dimensional sanity check.",
    "linear_algebra": "Recheck dimensions and one useful rank, trace, determinant, or eigenvalue invariant.",
    "optimization": "Recheck feasibility and compare objective values or the applicable optimality/KKT conditions.",
    "real_analysis": "Recheck quantifiers, endpoint behavior, and every hypothesis of the theorem used.",
    "topology": "Recheck the exact definitions and the direction and hypotheses of every preservation theorem used.",
}

_SCORE_FIRST_DISCRETE_FINAL_CHECKS = {
    "combinatorial_counting": (
        "Recheck every restriction and disjointness; when cheap compare a small case or unrestricted total. "
        "The count must be a nonnegative integer."
    ),
    "recurrence": "Substitute the claimed recurrence or closed form into the first few indices and every supplied initial condition.",
    "generating_function": "Re-expand the first few coefficients and verify the requested coefficient index and every x^k shift.",
    "graph_theory": "Check one cheap invariant: degree-sum parity, vertex-edge bounds, or the exact theorem hypothesis.",
    "number_theory_modular": "Substitute back into the original congruence; check gcd/invertibility and the requested residue representative.",
}

_SCORE_FIRST_MICRO_FINAL_CHECKS = {
    "newton_fixed_point": "Check the residual or defining iteration and verify the iterate index was not shifted.",
    "interpolation": "Evaluate the polynomial at enough supplied interpolation nodes to catch coefficient or index errors.",
    "quadrature": "Recheck interval scaling, weights, and nodes; when cheap test a polynomial within the rule's exactness degree.",
    "finite_difference_error": "Check the first uncancelled Taylor power of h against the claimed truncation order.",
    "stability_convergence": "Recompute the amplification/root/contraction condition and separate stability, consistency, and global order.",
    "conditioning_bayes": "Recompute the conditioning denominator explicitly and verify the probability lies in [0,1].",
    "expectation_indicator": "Check expectation bounds/support and independently verify each indicator success probability used.",
    "named_distribution": "Check support and parameterization, then verify a known normalization, mean, or variance when cheap.",
    "group_order_sylow": "Check proposed orders divide the group order and reapply both Sylow divisibility and congruence conditions.",
    "homomorphism_quotient": "Verify kernel normality or ideal conditions and cross-check kernel-image sizes when finite.",
    "finite_field_galois": "Check multiplicative orders divide q-1 and extension/subfield degrees satisfy divisibility.",
    "markov_stationary": "Verify pi P = pi and sum(pi)=1.",
    "markov_hitting": "Verify the target-state boundary value and substitute the result into the first-step equation.",
    "residue_contour": "Relist singularities inside the contour, check orientation, and verify the residue sum in 2*pi*i*sum Res.",
    "cauchy_formula": "Check analyticity, contour inclusion of the evaluation point, and the derivative/factorial order.",
    "laurent_singularity": "Check the Laurent principal part and that the residue is the coefficient of (z-z0)^(-1).",
    "likelihood_mle": "Check support and boundary cases as well as stationary equations; compare likelihood values when cheap.",
    "hypothesis_test_ci": "Recheck reference distribution, degrees of freedom, tail direction, and test-versus-interval target.",
    "ols_full_rank": "Check dimensions and verify X^T(y-X beta_hat)=0.",
    "sampling_inference": "Distinguish mean-response from new-observation uncertainty; recheck degrees of freedom and leverage.",
    "eigen_jordan": "Check algebraic multiplicities and use trace, determinant, or eigenspace dimensions as a consistency check.",
    "kkt_convex": "Verify primal feasibility, dual feasibility, stationarity, and complementary slackness.",
    "lagrange_boundary": "Recheck feasibility and compare objective values at every stationary and boundary or endpoint candidate.",
    "unconstrained_hessian": "Substitute into the gradient and verify the Hessian or global argument supports the claimed extremum.",
    "uniform_pointwise": "Check the supremum error over the whole domain, including endpoints or moving-peak locations.",
    "spectrum_invertibility": "Check the exact operator named in the target. Verify the applicable injectivity/surjectivity, bounded-inverse, Neumann-series, or spectral criterion; absence of eigenvectors alone is not enough.",
}

_SCORE_FIRST_TASK_FINAL_CHECKS = {
    "proof": (
        "Verify every stated hypothesis is available and the argument reaches exactly the requested conclusion; "
        "do not assume a converse or stronger statement."
    ),
    "proof_or_disproof": (
        "Verify the truth value is established. If false, the counterexample must satisfy every hypothesis and violate "
        "the conclusion; if true, the proof must cover the whole claim."
    ),
    "construction_counterexample": (
        "Verify the constructed object satisfies every requested property. For a counterexample, all hypotheses must "
        "hold while the claimed conclusion fails."
    ),
}

_SCORE_FIRST_TARGET_FINAL_CHECKS = {
    "newton_convergence": (
        "Recheck simple-root/nonzero-derivative and regularity assumptions; verify error recursion gives "
        "the claimed convergence order."
    ),
    "expectation_indicator_count": (
        "Verify each indicator event and its success probability, then re-sum by linearity of expectation; "
        "independence is not required for linearity."
    ),
    "expectation_moment": (
        "Independently recompute the requested moment from the pmf/pdf, MGF, or a known moment identity; "
        "check support, sign/range, and a known special case."
    ),
    "ols_covariance": (
        "Write beta_hat=A y and recompute Var(beta_hat)=A Var(y) A^T. Check dimensions and the sigma^2 I assumption, "
        "then simplify to the claimed covariance matrix."
    ),
    "mle_asymptotic": (
        "Recompute the score/Fisher information or variance formula under the stated parameterization and sample size; "
        "check whether the result uses I(theta) or n I(theta) and verify reciprocal/scaling factors."
    ),
    "pde_characteristics_derivation": (
        "Substitute the derived characteristic equations back into the PDE/characteristic relation and verify the "
        "resulting invariant or reduced equation."
    ),
    "generic_derivation": (
        "Independently validate the derived quantity by a cheap substitution, differentiation, dimension check, "
        "initial condition, coefficient comparison, or known special case appropriate to the target."
    ),
}

_SCORE_FIRST_RESPONSE_MODE_ANSWER = "ANSWER_VALUE"
_SCORE_FIRST_RESPONSE_MODE_DERIVATION = "DERIVATION"
_SCORE_FIRST_RESPONSE_MODE_PROOF = "PROOF"
_SCORE_FIRST_RESPONSE_MODE_PROOF_OR_DISPROOF = "PROOF_OR_DISPROOF"
_SCORE_FIRST_RESPONSE_MODE_CONSTRUCTION = "CONSTRUCTION_COUNTEREXAMPLE"

_SCORE_FIRST_DERIVATION_SYSTEM_PROMPT = """You are a high-accuracy mathematics competition solver.

Give the final result first.

Then provide a concise derivation containing the essential mathematical steps.

Do not output JSON.
Do not repeat the problem.
Do not omit a requested derivation merely to shorten the answer.

The subject/strategy hint is advisory. If the mathematical structure of the problem
indicates a different method, follow the problem itself.

""".strip()

_SCORE_FIRST_PROOF_OR_DISPROOF_SYSTEM_PROMPT = """You are a high-accuracy mathematics competition solver.

State whether the claim is true or false first.

If true, give a concise complete proof.
If false, give a concise disproof or counterexample and verify why it invalidates the claim.

Do not output JSON.
Do not repeat the problem.
Do not assume in advance that the statement is true.

The subject/strategy hint is advisory. If the mathematical structure of the problem
indicates a different method, follow the problem itself.

""".strip()

_SCORE_FIRST_CONSTRUCTION_SYSTEM_PROMPT = """You are a high-accuracy mathematics competition solver.

State the constructed object or counterexample first.

Then give the minimal verification needed to show that it satisfies the requested
properties or invalidates the statement.

Do not output JSON.
Do not repeat the problem.

The subject/strategy hint is advisory. If the mathematical structure of the problem
indicates a different method, follow the problem itself.

""".strip()

_SCORE_FIRST_REQUEST_INTENT_PATTERNS = {
    "proof_or_disproof": (
        r"\bprove\s+or\s+disprove\b",
        r"\bprove\s+or\s+refute\b",
        r"\bestablish\s+or\s+disprove\b",
        r"证明或反驳",
        r"证明或否定",
    ),
    "construction": (
        r"(?:^|[.!?;:\n]\s*)\s*(?:please\s+)?construct\b",
        r"\b(?:give|provide|construct|find)\s+(?:an?\s+)?counterexample\b",
        r"\bexhibit\s+(?:an?\s+)?example\b",
        r"\b(?:give|find|exhibit)\s+(?:an?\s+)?example\s+(?:showing|for\s+which)\b[^.\n]{0,100}\b(?:fails?|false|not\s+hold)\b",
        r"\bexample\s+showing\s+the\s+converse\s+fails\b",
        r"\b(?:and|then|also)\s+(?:construct|give|provide|find)\s+(?:an?\s+)?counterexample\b",
        r"(?:^|[。！？；：\n]\s*)\s*(?:请)?构造",
        r"给出(?:一个|一)?反例",
        r"举(?:一个|一)?反例",
        r"举例说明[^。；\n]{0,50}(?:不成立|失败)",
        r"给出(?:一个|一)?使[^。；\n]{0,50}(?:失败|不成立)的例子",
    ),
    "proof": (
        r"(?:^|[.!?;:\n]\s*)\s*(?:please\s+)?prove(?:\s+that)?\b",
        r"\b(?:give|provide)\s+(?:a\s+)?proof\b",
        r"(?:^|[.!?;:\n]\s*)\s*show\s+that\b",
        r"(?:^|[.!?;:\n]\s*)\s*demonstrate\s+that\b",
        r"(?:^|[.!?;:\n]\s*)\s*establish\s+that\b",
        r"\b(?:and|then|also)\s+(?:prove|show|establish)\b",
        r"(?:^|[。！？；：\n]\s*)\s*(?:请)?证明",
        r"给出证明",
        r"(?:^|[。！？；：\n]\s*)\s*(?:请)?证实",
    ),
    "derivation": (
        r"(?:^|[.!?;:\n]\s*)\s*(?:please\s+)?derive\b",
        r"^\s*why\b",
        r"^\s*how\s+(?:does|do|can)\b",
        r"^\s*how\s+is\b[^?\n]{0,120}\b(?:obtained|derived)\b",
        r"(?:^|[.!?;:\n]\s*)\s*(?:please\s+)?deduce\b",
        r"\bshow\s+how\s+to\s+obtain\b",
        r"(?:^|[.!?;:\n]\s*)\s*explain\b",
        r"\b(?:and|then|first|also)\s+explain\b",
        r"\bexplaining\s+(?:why|what|how)\b",
        r"(?:^|[,;:]\s*|\b(?:and|then|first|but\s+first)\s+)justify\b",
        r"\bjustify\s+your\s+answer\b",
        r"\bgive\s+(?:(?:a|the)\s+)?(?:reason|justification)\b",
        r"\bshow\s+the\s+calculation\b",
        r"\bshow\s+your\s+work\b",
        r"\bshow\s+the\s+steps\b",
        r"\bgive\s+the\s+calculation\b",
        r"\bexplain\s+your\s+calculation\b",
        r"\b(?:and|then|also)\s+(?:derive|deduce|explain|justify|verify)\b",
        r"(?:^|[.!?;:\n]\s*)\s*verify(?:\s+numerically|\s+that)?\b",
        r"(?:^|[。！？；：\n]\s*)\s*(?:请)?推导",
        r"^\s*(?:为什么|为何|如何|怎么)",
        r"写出计算过程",
        r"给出计算过程",
        r"写出步骤",
        r"说明计算步骤",
        r"(?:^|[。！？；：\n]\s*)\s*(?:请)?推演",
        r"(?:^|[。！？；：\n]\s*)\s*(?:请)?解释",
        r"解释为什么",
        r"解释原因",
        r"并解释",
        r"并推导",
        r"然后推导",
        r"并证明",
        r"然后证明",
        r"并验证",
        r"并给出理由",
        r"说明理由",
        r"说明原因",
        r"给出理由",
        r"并说明",
        r"说明使用哪个定理",
        r"说明[^。；\n]{0,12}使用[^。；\n]{0,12}定理",
        r"(?:^|[。！？；：\n]\s*)\s*(?:请)?验证",
    ),
    "choice": (
        r"\bwhich\s+of\s+the\s+following\b",
        r"\bwhich\s+option\b",
        r"\bselect\s+(?:the\s+)?(?:correct|best)\b",
        r"\bchoose\s+(?:the\s+)?(?:correct|best)\b",
        r"以下(?:哪|哪个)",
        r"哪个选项",
        r"选择(?:正确|最合适)",
    ),
}

_SCORE_FIRST_PROTECTED_DOT_ABBREVIATIONS = (
    "a.e.",
    "i.e.",
    "e.g.",
    "w.r.t.",
)

_SCORE_FIRST_ENGLISH_REQUEST_VERB_RE = re.compile(
    r"""
    \b(?:compute|calculate|evaluate|find|determine|solve|classify|identify|state|answer|
       give|provide|prove|show|establish|derive|deduce|explain|justify|verify|
       construct|disprove|select|choose|exhibit|use|apply|differentiate|integrate|
       simplify|factor|factorize|expand|approximate|estimate|maximize|minimize|
       optimize|diagonalize|diagonalise|invert|normalize|normalise)\b
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

_SCORE_FIRST_ENGLISH_DEFERRED_REQUEST_VERB_RE = re.compile(
    r"\b(?:parameterize|parametrize)\b",
    flags=re.IGNORECASE,
)

_SCORE_FIRST_ENGLISH_INTERROGATIVE_RE = re.compile(
    r"""
    \b(?:
        what
        |which
        |why
        |how\s+(?:many|much|large|fast|does|do|can)
        |how\s+is\b
        |is|are|does|do|can|could|will
    )\b
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

_SCORE_FIRST_CHINESE_REQUEST_PHRASES = (
    "使用",
    "利用",
    "采用",
    "应用",
    "因式分解",
    "最大化",
    "最小化",
    "对角化",
    "归一化",
    "参数化",
    "求导",
    "积分",
    "化简",
    "展开",
    "近似",
    "估计",
    "优化",
    "求逆",
    "求出",
    "计算",
    "确定",
    "判断",
    "指出",
    "写出",
    "给出",
    "选择",
    "证明",
    "推导",
    "说明",
    "解释",
    "验证",
    "构造",
    "举出",
    "反驳",
    "用",
    "求",
)

_SCORE_FIRST_CHINESE_INTERROGATIVE_RE = re.compile(
    r"是否|能否|可否|多少|几个|几次|几步|几项|几种|哪个|哪一个|为何|为什么|如何|怎么"
)

_SCORE_FIRST_NONREQUEST_ACTION_PATTERNS = (
    r"\bno\s+proof\s+is\s+required\b",
    r"\bdo\s+not\s+prove\b",
    r"\bneed\s+not\s+prove\b",
    r"\bwithout\s+proving\b",
    r"\byou\s+need\s+not\s+prove\b",
    r"\bdo\s+not\s+construct\b",
    r"\bneed\s+not\s+construct\b",
    r"\bwithout\s+constructing\b",
    r"\bdo\s+not\s+derive\b",
    r"\brather\s+than\s+derive\b",
    r"\bno\s+derivation\s+is\s+required\b",
    r"\bthe\s+proof\s+(?:uses|above|shows|states)\b",
    r"\bthe\s+construction\s+(?:above|uses|shows)\b",
    r"\bthe\s+derivation\s+(?:above|uses|shows)\b",
    r"\bis\s+it\s+possible\s+to\s+prove\b",
    r"\bpossible\s+to\s+prove\b",
    r"\bconstructing\b[^.;!?\n]{0,80}\b(?:is|was)\s+unnecessary\b",
    r"无需给出证明",
    r"不必给出证明",
    r"无需(?:先)?证明",
    r"不必(?:先)?证明",
    r"不用(?:先)?证明",
    r"不要(?:先)?证明",
    r"无需(?:先)?构造",
    r"不必(?:先)?构造",
    r"不要(?:先)?构造",
    r"无需(?:先)?推导",
    r"不要求(?:先)?推导",
    r"不必(?:先)?推导",
)

_SCORE_FIRST_TRUSTED_DOMAIN_ALIASES = {
    # Canonical labels.
    "discrete math": "discrete_math",
    "numerical analysis": "numerical_analysis",
    "measure integration": "measure_integration",
    "differential geometry": "differential_geometry",
    "probability": "probability",
    "abstract algebra": "abstract_algebra",
    "stochastic process": "stochastic_process",
    "complex analysis": "complex_analysis",
    "ode": "ode",
    "statistics": "statistics",
    "functional analysis": "functional_analysis",
    "linear regression": "linear_regression",
    "pde": "pde",
    "advanced math": "advanced_math",
    "linear algebra": "linear_algebra",
    "optimization": "optimization",
    "real analysis": "real_analysis",
    "topology": "topology",

    # Human-readable English labels.
    "discrete mathematics": "discrete_math",
    "measure theory": "measure_integration",
    "measure and integration": "measure_integration",
    "measure theory and integration": "measure_integration",
    "probability theory": "probability",
    "stochastic processes": "stochastic_process",
    "ordinary differential equation": "ode",
    "ordinary differential equations": "ode",
    "partial differential equation": "pde",
    "partial differential equations": "pde",
    "advanced mathematics": "advanced_math",

    # Existing Chinese aliases used by the deterministic router.
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

_SCORE_FIRST_NO_SUBJECT_SPECIALIZED_SCORE_MIN = 5.5
_SCORE_FIRST_NO_SUBJECT_SPECIALIZED_MARGIN_MIN = 4.0

_SCORE_FIRST_HUMAN_DOMAIN_LABELS = {
    "discrete_math": "Discrete Mathematics",
    "numerical_analysis": "Numerical Analysis",
    "measure_integration": "Measure Theory and Integration",
    "differential_geometry": "Differential Geometry",
    "probability": "Probability Theory",
    "abstract_algebra": "Abstract Algebra",
    "stochastic_process": "Stochastic Processes",
    "complex_analysis": "Complex Analysis",
    "ode": "Ordinary Differential Equations",
    "statistics": "Statistics",
    "functional_analysis": "Functional Analysis",
    "linear_regression": "Linear Regression",
    "pde": "Partial Differential Equations",
    "advanced_math": "Advanced Mathematics",
    "linear_algebra": "Linear Algebra",
    "optimization": "Optimization",
    "real_analysis": "Real Analysis",
    "topology": "Topology",
}

_SCORE_FIRST_MICRO_STRONG_SCORE = 4
_SCORE_FIRST_MICRO_WEAK_SCORE = 1
_SCORE_FIRST_MICRO_MIN_SCORE = 4
_SCORE_FIRST_MICRO_MIN_MARGIN = 2

_SCORE_FIRST_MICRO_TARGET_STRONG_SCORE = 8
_SCORE_FIRST_MICRO_TARGET_WEAK_SCORE = 3
_SCORE_FIRST_MICRO_CONTEXT_STRONG_SCORE = 2
_SCORE_FIRST_MICRO_CONTEXT_WEAK_SCORE = 0

# Each rule is (name, strong_patterns, weak_patterns, card). Arbitration scores every
# rule; tuple order has no semantic priority.
_SCORE_FIRST_MICRO_STRATEGIES = {
    "numerical_analysis": (
        (
            "newton_fixed_point",
            (
                r"\bnewton\s+(?:iteration|iterations|step|iterate)\b",
                r"\bnewton(?:'s)?\s+method\b[^.\n]{0,80}\b(?:root|zero|f\s*\([^)]*\)\s*=\s*0)\b",
                r"(?<!not )(?<!without )\broot[- ]finding\b",
                r"\bfixed[- ]point\s+iteration\b",
                r"\bfixed[- ]point\b[^.\n]{0,60}\blocal\s+conver",
                r"x_?\{?n\+1\}?\s*=\s*",
                r"不动点[^。；\n]{0,50}局部收敛",
                r"局部收敛[^。；\n]{0,50}不动点",
                r"牛顿迭代",
                r"牛顿法求根",
                r"不动点迭代",
            ),
            (r"\bnewton\b", r"\bfixed[- ]point\b", r"牛顿"),
            "Write the iteration explicitly and evaluate all derivatives at the correct iterate. After computing the "
            "step, check it by substitution and distinguish convergence order from the numerical value of the iterate.",
        ),
        (
            "interpolation",
            (
                r"\bnewton\s+interpolation\b",
                r"\bnewton\s+divided\s+differences?\b",
                r"\bdivided\s+differences?\b",
                r"\binterpolating\s+polynomial\b",
                r"\blagrange\s+interpolation\b",
                r"插值",
                r"差商",
            ),
            (r"\binterpol", r"\blagrange\b"),
            "Identify the interpolation nodes and degree before expanding. Use the interpolation form that preserves "
            "the node structure and check that the resulting polynomial matches the supplied data.",
        ),
        (
            "quadrature",
            (
                r"\bsimpson(?:'s)?\s+(?:rule|formula)\b",
                r"\btrapezoid(?:al)?\s+(?:rule|formula)\b",
                r"\bgauss(?:ian)?\s+quadrature\b",
                r"\bquadrature\s+rule\b",
                r"辛普森",
                r"梯形公式",
                r"高斯求积",
                r"求积公式",
            ),
            (r"\bquadrature\b", r"\btrapez", r"求积"),
            "Identify the quadrature rule and its polynomial exactness/error term before substituting. Track interval "
            "scaling factors carefully.",
        ),
        (
            "finite_difference_error",
            (
                r"\bcentral[- ]difference\b",
                r"\bforward[- ]difference\b",
                r"\bbackward[- ]difference\b",
                r"\bfinite[- ]difference\b",
                r"f\s*\(\s*x\s*\+\s*h\s*\)[^.\n]{0,120}f\s*\(\s*x\s*-\s*h\s*\)",
                r"f\s*\(\s*x\s*-\s*h\s*\)[^.\n]{0,120}f\s*\(\s*x\s*\+\s*h\s*\)",
                r"中心差分",
                r"前向差分",
                r"后向差分",
                r"有限差分",
            ),
            (r"\btruncation\s+error\b", r"截断误差", r"差分"),
            "Expand the finite-difference formula at the correct point, track each power of h, and distinguish the "
            "leading truncation term from the stated order.",
        ),
        (
            "stability_convergence",
            (
                r"\bzero[- ]stable\b",
                r"\bzero\s+stability\b",
                r"\babsolute\s+stability\b",
                r"\ba[- ]stable\b",
                r"\bstability\s+region\b",
                r"\bglobal\s+order\b",
                r"\bconvergence\s+order\b",
                r"零稳定",
                r"绝对稳定",
                r"稳定域",
                r"全局误差阶",
                r"收敛阶",
            ),
            (r"\bstability\b", r"\bstable\b", r"\bconvergence\b", r"稳定", r"收敛"),
            "Separate consistency, convergence, and stability. Write the relevant amplification, error, or contraction "
            "condition explicitly before drawing a conclusion.",
        ),
    ),
    "measure_integration": (
        (
            "limit_integral",
            (
                r"\bdominated\s+convergence\b",
                r"\bdct\b",
                r"\bmonotone\s+convergence\b",
                r"\bmct\b",
                r"\bfatou\b",
                r"f_?n[^.\n]{0,40}(?:↑|\\uparrow)[^.\n]{0,40}f",
                r"\blim(?:it)?[^.\n]{0,50}\bintegral\b",
                r"\blim(?:it)?[^.\n]{0,50}∫",
                r"\binterchang(?:e|ing)[^.\n]{0,40}\blimit\b[^.\n]{0,40}\bintegral\b",
                r"控制收敛",
                r"单调收敛",
                r"Fatou",
                r"极限[^。；\n]{0,30}积分",
                r"交换[^。；\n]{0,20}极限[^。；\n]{0,20}积分",
            ),
            (r"\bconvergen", r"\blimit\b", r"收敛"),
            "Before interchanging limit and integral, decide specifically among MCT, Fatou, DCT, or no applicable "
            "theorem. Verify every hypothesis, especially domination/integrability and nonnegativity.",
        ),
        (
            "fubini_tonelli",
            (
                r"\bfubini\b",
                r"\btonelli\b",
                r"\b(?:swap|exchange|interchange)\s+(?:the\s+)?order\s+of\s+integration\b",
                r"\biterated\s+integrals?\b",
                r"\babsolute(?:ly)?\s+integrab[^.\n]{0,60}\bproduct\s+(?:space|measure)\b",
                r"交换积分次序",
                r"交换累次积分",
                r"累次积分",
                r"Fubini",
                r"Tonelli",
            ),
            (r"\bproduct\s+(?:space|measure)\b", r"\bdouble\s+integral\b", r"乘积测度", r"二重积分"),
            "Check nonnegativity or absolute integrability before swapping integration order. Distinguish Tonelli's "
            "nonnegative case from Fubini's integrable case.",
        ),
        (
            "measurability",
            (
                r"\b(?:prove|show|determine|decide)\b[^.\n]{0,60}\bmeasurable\b",
                r"\bmeasurability\b",
                r"证明[^。；\n]{0,40}可测",
                r"判断[^。；\n]{0,40}可测",
                r"可测性",
            ),
            (r"\bmeasurable\b", r"可测"),
            "Reduce measurability to inverse images or closure properties of measurable functions. Keep measurability "
            "separate from integrability and almost-everywhere statements.",
        ),
        (
            "lp_integrability",
            (
                r"\bL\s*\^\s*[p12]\b",
                r"\bL[_ ]?[p12]\b",
                r"\bholder(?:'s)?\b",
                r"\bminkowski\b",
                r"\bnorm\s+inequalit",
                r"\|\|[^|\n]+\|\|[_^]?\s*[12p]",
                r"L\^?[p12][^.\n]{0,60}范数",
                r"Holder",
                r"Minkowski",
            ),
            (r"\bintegrab", r"\bnorm\b", r"可积", r"范数"),
            "Identify the exact Lp exponent and underlying measure. Check finiteness of the defining integral before "
            "using norm inequalities or inclusions.",
        ),
    ),
    "differential_geometry": (
        (
            "first_fundamental_form",
            (
                r"\bfirst\s+fundamental\s+form\b",
                r"\bmetric\s+coefficients?\b",
                r"\bcompute\b[^.\n]{0,30}\bE\s*,\s*F\s*,\s*G\b",
                r"第一基本形式",
                r"度量系数",
                r"求[^。；\n]{0,20}E\s*[,，]\s*F\s*[,，]\s*G",
            ),
            (r"\bmetric\b", r"度量"),
            "Differentiate the parametrization first and compute the metric coefficients from inner products. Check "
            "regularity and coordinate order before using the first fundamental form.",
        ),
        (
            "curvature",
            (
                r"\bgaussian\s+curvature\b",
                r"\bmean\s+curvature\b",
                r"\bsectional\s+curvature\b",
                r"\bcurvature\s+formula\b",
                r"高斯曲率",
                r"平均曲率",
                r"截面曲率",
            ),
            (r"\bcurvature\b", r"曲率"),
            "Fix the metric and convention first. For a parametrized surface compute the necessary first/second "
            "fundamental-form data before using a curvature formula, then check sign and normalization.",
        ),
        (
            "geodesic_connection",
            (
                r"\bgeodesic\s+equation\b",
                r"\bchristoffel\s+symbols?\b",
                r"\bconnection\s+coefficients?\b",
                r"Γ\s*\^",
                r"Gamma\s*\^",
                r"测地线方程",
                r"Christoffel",
                r"联络系数",
            ),
            (r"\bgeodesic\b", r"\bconnection\b", r"测地", r"联络"),
            "Write the metric or connection coefficients in the chosen coordinates before the geodesic equation. Track "
            "parameterization and sign conventions consistently.",
        ),
        (
            "intrinsic_geometry",
            (
                r"\bintrinsic\s+(?:geometry|quantity|property)\b",
                r"\bisometr(?:y|ic|ically)\b",
                r"\bunrolled\s+isometrically\b",
                r"\bgauss[- ]bonnet\b",
                r"内蕴",
                r"等距",
                r"Gauss[- ]Bonnet",
            ),
            (r"\bintrinsic\b", r"\bisometry\b"),
            "Separate intrinsic metric information from embedding-dependent quantities. Use only data invariant under "
            "the relevant isometry or coordinate change.",
        ),
    ),
    "probability": (
        (
            "conditioning_bayes",
            (
                r"\bconditional\s+probability\b",
                r"\bgiven\s+that\b",
                r"\bbayes(?:['’]s?)?(?:\s+theorem)?\b",
                r"\bcondition(?:ing|ed)\s+on\b",
                r"\bprevalence\b[^.\n]{0,100}\b(?:sensitivity|specificity|positive\s+test)\b",
                r"\bpositive\s+test\b[^.\n]{0,100}\b(?:prevalence|sensitivity|specificity)\b",
                r"患病率[^。；\n]{0,80}(?:灵敏度|特异度|阳性)",
                r"阳性[^。；\n]{0,80}(?:患病率|灵敏度|特异度)",
                r"阳性时[^。；\n]{0,50}概率",
                r"\bpositive\b[^.\n]{0,50}\bprobability\b",
                r"条件概率",
                r"已知[^。；\n]{0,50}求[^。；\n]{0,20}概率",
                r"贝叶斯",
            ),
            (r"\bconditional\b", r"\bgiven\b", r"条件"),
            "Write the conditioning event and denominator explicitly. Restrict the sample space before counting, and "
            "do not assume independence after conditioning.",
        ),
        (
            "expectation_indicator",
            (
                r"\bexpected\s+(?:number|value)\b",
                r"\bexpectation\b",
                r"\bindicator\s+variables?\b",
                r"\bE\s*\[[^\]]+\]",
                r"\bVar\s*\([^\)]+\)",
                r"期望",
                r"指示变量",
            ),
            (r"\bexpected\b", r"\bmean\b"),
            "Try linearity of expectation and indicator variables before deriving a full distribution. Verify each "
            "indicator's success probability and avoid unnecessary independence assumptions.",
        ),
        (
            "named_distribution",
            (
                r"\b(?:identify|find|determine|state)\b[^.\n]{0,60}\bdistribution\b",
                r"\bdistribution\s+of\b",
                r"\bthinn(?:ed|ing)\b[^.\n]{0,80}\bpoisson\b",
                r"\bpoisson\b[^.\n]{0,100}\b(?:retained|kept)\b[^.\n]{0,50}\bprobability\b",
                r"指出[^。；\n]{0,50}分布",
                r"求[^。；\n]{0,50}分布",
                r"泊松[^。；\n]{0,80}(?:保留|稀疏|抽稀)[^。；\n]{0,50}概率",
            ),
            (r"\bbinomial\b", r"\bpoisson\b", r"\bnormal\b", r"\bexponential\b", r"二项", r"泊松", r"正态", r"分布"),
            "Write the exact distribution and its parameters before applying a formula. Check support, parameterization, "
            "and whether a transformation changes the distribution family.",
        ),
        (
            "limit_theorem",
            (
                r"\bcentral\s+limit\s+theorem\b",
                r"\blaw\s+of\s+large\s+numbers\b",
                r"\bclt\b",
                r"中心极限定理",
                r"大数定律",
            ),
            (r"\basymptotic\b", r"极限分布"),
            "State the normalization and assumptions of the limit theorem before using it. Distinguish convergence in "
            "distribution, probability, and almost sure convergence.",
        ),
    ),
    "abstract_algebra": (
        (
            "group_order_sylow",
            (
                r"\bsylow\b",
                r"\bp[- ]subgroup\b",
                r"\bnumber\s+of\s+sylow\b",
                r"\bgroup\s+(?:G\s+)?has\s+order\s+\d+\b",
                r"\bgroup\s+of\s+order\s+\d+\b",
                r"Sylow",
                r"群[^。；\n]{0,15}阶为\s*\d+",
            ),
            (r"\blagrange(?:'s)?\b", r"\belement\s+order\b", r"拉格朗日"),
            "Use divisibility and index constraints first. For Sylow questions combine the congruence and divisibility "
            "conditions before attempting structural classification.",
        ),
        (
            "homomorphism_quotient",
            (
                r"\bhomomorphism\b",
                r"\bkernel\b",
                r"\bker\s*(?:\(?\s*phi\s*\)?|\b)",
                r"\bimage\b[^.\n]{0,30}\bhomomorphism\b",
                r"\bquotient\s+group\b",
                r"同态",
                r"核与像",
                r"商群",
            ),
            (r"\bkernel\b", r"\bimage\b", r"\bquotient\b", r"核", r"像"),
            "Compute kernel and image first and use the homomorphism/isomorphism theorems. Check normality or ideal "
            "conditions before forming a quotient.",
        ),
        (
            "ring_ideal_field",
            (
                r"\bquotient\s+ring\b",
                r"\bmaximal\s+ideal\b",
                r"\bprime\s+ideal\b",
                r"\bintegral\s+domain\b",
                r"\bzero\s+divisor\b",
                r"(?:R|F)\s*\[x\]\s*/\s*\(",
                r"\b(?:is|whether)\b[^.\n]{0,60}\bfield\b",
                r"是否为域",
                r"商环",
                r"极大理想",
                r"素理想",
                r"整环",
                r"零因子",
            ),
            (r"\bring\b", r"\bideal\b", r"\bfield\b", r"环", r"理想", r"域"),
            "Identify units, zero divisors, ideals, and quotient conditions before declaring a ring a field or domain. "
            "Use ideal structure rather than element-wise guesses.",
        ),
        (
            "finite_field_galois",
            (
                r"\bfinite\s+field\b",
                r"\bGF\s*\(\s*q\s*\)",
                r"\bGF\s*\(\s*\d+\s*\)",
                r"\bF[_ ]?\{?\d+\}?\b",
                r"\bfrobenius\b",
                r"\bgalois\s+field\b",
                r"\bmultiplicative\s+(?:group\s+)?generator\b",
                r"乘法群生成元",
                r"有限域",
                r"Frobenius",
                r"伽罗瓦域",
            ),
            (r"\bfield\b", r"域"),
            "Use finite-field cardinality, subfield divisibility, and Frobenius structure first. Check extension degrees "
            "before counting elements or automorphisms.",
        ),
    ),
    "stochastic_process": (
        (
            "markov_stationary",
            (
                r"\bstationary\s+distribution\b",
                r"\bstationarity\b",
                r"\binvariant\s+distribution\b",
                r"\bpi\s*P\s*=\s*pi\b",
                r"平稳分布",
                r"不变分布",
            ),
            (r"\bmarkov\s+chain\b", r"\btransition\s+matrix\b", r"马尔可夫链", r"转移矩阵"),
            "Write the relevant transition structure explicitly. For stationarity solve pi P = pi together with "
            "normalization and check extra chain assumptions only when the requested conclusion needs them.",
        ),
        (
            "markov_hitting",
            (
                r"\bhitting\s+time\b",
                r"\bexpected\s+hitting\b",
                r"\bfirst\s+passage\b",
                r"\babsorption\b",
                r"\breturn\s+time\b",
                r"\bhit\s+(?:state|level)\b",
                r"首次到达",
                r"击中时间",
                r"吸收",
                r"返回时间",
                r"先到[^。；\n]{0,30}(?:而非|之前)[^。；\n]{0,30}概率",
            ),
            (r"\bhitting\b", r"\bhit\b", r"击中"),
            "Write the first-step equation explicitly. For hitting times condition on the first transition and keep "
            "boundary or absorbing states explicit.",
        ),
        (
            "poisson_process",
            (
                r"\bpoisson\s+process\b",
                r"\bindependent\s+increments\b[^.\n]{0,60}\bpoisson\b",
                r"\bN\s*\(\s*t\s*\)\b[^.\n]{0,80}\bpoisson\b",
                r"\b(?:law|distribution)\s+of\s+N\s*\(",
                r"泊松过程",
            ),
            (r"\bpoisson\b", r"\bN\s*\(\s*t\s*\)"),
            "Translate the requested increment into its interval length and Poisson mean first. Use independent "
            "increments only for disjoint intervals.",
        ),
        (
            "brownian_martingale",
            (
                r"\bbrownian\s+motion\b",
                r"\bmartingale\b",
                r"布朗运动",
                r"鞅",
            ),
            (r"\bbrownian\b", r"布朗"),
            "Use the defining increment, covariance, or conditional-expectation property directly. Check filtration "
            "and stopping assumptions before applying martingale results.",
        ),
    ),
    "complex_analysis": (
        (
            "residue_contour",
            (
                r"\bresidue\s+theorem\b",
                r"\bevaluate\b[^.\n]{0,50}\bby\s+residues\b",
                r"\busing\s+residues\b",
                r"\bsum\s+of\s+residues\b",
                r"\benclosed\s+residues\b",
                r"\blist\b[^.\n]{0,60}\bresidues\b[^.\n]{0,40}\b(?:inside|enclosed)\b",
                r"留数定理",
                r"用留数计算",
                r"利用留数",
            ),
            (r"\bresidue\b", r"\bcontour\b", r"留数", r"围道"),
            "List singularities first and determine which are inside the contour. Check contour orientation, then compute "
            "only the required residues using the simplest applicable formula.",
        ),
        (
            "cauchy_formula",
            (
                r"\bcauchy(?:'s)?\s+integral\s+formula\b",
                r"\bcauchy(?:'s)?\s+differentiation\s+formula\b",
                r"柯西积分公式",
                r"柯西求导公式",
            ),
            (r"\bcauchy\b", r"柯西"),
            "Match the integrand to the correct Cauchy integral formula and derivative order. Verify analyticity inside "
            "the contour and the location of the evaluation point.",
        ),
        (
            "laurent_singularity",
            (
                r"\blaurent\b",
                r"\bprincipal\s+part\b",
                r"\bclassif(?:y|ication)\b[^.\n]{0,60}\bsingularit",
                r"\bclassify\s+z\s*=",
                r"\bclassify\b[^.\n]{0,80}\band\s+compute\s+(?:the\s+)?residue\b",
                r"\b(?:removable|essential)\s+singularit",
                r"\bpole\s+of\s+order\b",
                r"洛朗",
                r"主部",
                r"判断[^。；\n]{0,30}奇点类型",
                r"奇点类型",
            ),
            (r"\bsingularit", r"\bpole\b", r"奇点"),
            "Choose the annulus before expanding the Laurent series. Read the principal part to classify the singularity "
            "and identify the residue coefficient.",
        ),
        (
            "zeros_argument",
            (
                r"\bargument\s+principle\b",
                r"\brouch[eé](?:'s)?\b",
                r"\bcount\b[^.\n]{0,30}\bzeros?\b",
                r"辐角原理",
                r"Rouch[eé]",
                r"儒歇",
                r"计算[^。；\n]{0,20}零点个数",
            ),
            (r"\bzeros?\b", r"零点"),
            "Choose the contour and compare magnitudes on the boundary before using Rouche or the argument principle. "
            "Count zeros and poles with multiplicity.",
        ),
    ),
    "ode": (
        (
            "separable_linear_bernoulli",
            (
                r"\bseparable\s+(?:ode|equation)\b",
                r"\bfirst[- ]order\s+linear\s+(?:ode|equation)\b",
                r"\bbernoulli\s+(?:ode|equation)\b",
                r"可分离变量",
                r"一阶线性",
                r"Bernoulli",
            ),
            (r"\bseparable\b", r"\bbernoulli\b", r"变量分离"),
            "Classify the first-order equation before dividing or integrating. For linear or Bernoulli form, write the "
            "standard transformed equation explicitly and preserve any singular solution lost by division.",
        ),
        (
            "constant_coefficient_euler_cauchy",
            (
                r"\bconstant[- ]coefficient\b",
                r"\bcharacteristic\s+(?:equation|roots?)\b",
                r"\beuler[- ]cauchy\b",
                r"\bequidimensional\b",
                r"常系数",
                r"特征方程",
                r"Euler[- ]Cauchy",
            ),
            (r"\bcharacteristic\b", r"欧拉方程"),
            "Write the characteristic equation in the correct variable. Check repeated or complex roots, and for "
            "Euler-Cauchy equations use the power-law ansatz before applying conditions.",
        ),
        (
            "equilibrium_stability",
            (
                r"\bequilibrium\s+(?:solution|point|points)\b",
                r"\bphase\s+line\b",
                r"\bstability\s+of\s+(?:an?\s+)?equilibrium\b",
                r"平衡解",
                r"平衡点",
                r"相线",
                r"稳定性",
            ),
            (r"\bequilibrium\b", r"\bstability\b", r"平衡"),
            "Find equilibria before linearizing or drawing a phase line. Classify stability from the local sign or "
            "linearization and keep semistable cases separate.",
        ),
    ),
    "statistics": (
        (
            "likelihood_mle",
            (
                r"\b(?:derive|find|compute|obtain)\s+(?:the\s+)?(?:MLE|maximum\s+likelihood\s+estimator)\b",
                r"\bwrite\s+(?:the\s+)?likelihood\s+function\b",
                r"\bmaximize\b[^.\n]{0,50}\blikelihood\b",
                r"(?:推导|求出?|计算)[^。；\n]{0,20}(?:MLE|最大似然估计)",
                r"写出[^。；\n]{0,20}似然函数",
            ),
            (r"\bmle\b", r"\bmaximum\s+likelihood\b", r"\blikelihood\b", r"最大似然", r"似然"),
            "Write the likelihood on the correct support, then optimize the log-likelihood with boundary cases included. "
            "Check whether the maximizer depends on an order statistic rather than a stationary equation.",
        ),
        (
            "bias_variance_sufficiency",
            (
                r"\bunbiased\b",
                r"\bbias\b[^.\n]{0,40}\bvariance\b",
                r"\bbias\s+of\s+(?:the\s+)?estimator\b",
                r"\bsufficient\s+statistic\b",
                r"\bfactorization\s+theorem\b",
                r"无偏",
                r"偏差",
                r"充分统计量",
                r"因子分解定理",
            ),
            (r"\bvariance\b", r"\bestimator\b", r"方差", r"估计量"),
            "Compute expectation or factor the joint model according to the requested property. Keep unbiasedness, "
            "variance, efficiency, and sufficiency logically separate.",
        ),
        (
            "hypothesis_test_ci",
            (
                r"\bhypothesis\s+test\b",
                r"\bconfidence\s+interval\b",
                r"\bp[- ]value\b",
                r"\bz[- ]test\b",
                r"\bt[- ]test\b",
                r"\bz\s*检验",
                r"\bnull\s+hypothesis\b",
                r"假设检验",
                r"置信区间",
                r"p值",
            ),
            (r"\bH0\b", r"\bH_0\b", r"检验统计量"),
            "State the null and alternative, the statistic, and its reference distribution before computing. Distinguish "
            "a test decision from an interval estimate and check the assumptions behind the reference law.",
        ),
    ),
    "functional_analysis": (
        (
            "bounded_operator_continuity",
            (
                r"\bbounded\s+linear\s+(?:map|operator)\b",
                r"\boperator\s+norm\b",
                r"\bcontinuous\s+at\s+0\b",
                r"\b(?:prove|show|determine)\b[^.\n]{0,50}\bcontinuous\b",
                r"\bis\s+continuous\b",
                r"有界线性算子",
                r"算子范数",
                r"在0处连续",
            ),
            (r"\bbounded\b", r"\bcontinuous\b", r"有界", r"连续"),
            "Use linearity to relate boundedness and continuity, and estimate the operator norm directly from the defining "
            "inequality. Do not infer compactness from boundedness.",
        ),
        (
            "compactness",
            (
                r"\bcompact\s+operator\b",
                r"\bcompletely\s+continuous\b",
                r"\bidentity\s+operator\b[^.\n]{0,50}\bcompact\b",
                r"紧算子",
                r"完全连续",
            ),
            (r"\bcompact\b", r"紧"),
            "Test compactness through images of bounded sequences or the unit ball. In infinite dimensions, distinguish "
            "boundedness from relative compactness and use a separated sequence when disproving compactness.",
        ),
        (
            "spectrum_invertibility",
            (
                r"\bspectrum\b",
                r"\bresolvent\b",
                r"\binvertib(?:le|ility)\b[^.\n]{0,50}\boperator\b",
                r"\b(?:determine|decide|whether)\b[^.\n]{0,60}\binvertib(?:le|ility)\b",
                r"\bI\s*-\s*T\b[^.\n]{0,40}\binvertib(?:le|ility)\b",
                r"谱",
                r"预解集",
                r"可逆算子",
            ),
            (r"\beigenvalue\b", r"\binvertib", r"特征值", r"可逆"),
            "Separate spectrum from point spectrum. Check whether lambda I-T is bijective with bounded inverse rather than "
            "treating absence of eigenvectors as invertibility.",
        ),
    ),
    "linear_regression": (
        (
            "ols_full_rank",
            (
                r"\bols\b",
                r"\bordinary\s+least\s+squares\b",
                r"\bleast[- ]squares\s+estimator\b",
                r"\bnormal\s+equations?\b[^.\n]{0,80}\bfull\s+(?:column\s+)?rank\b",
                r"\bvar\s*\(\s*beta_?hat\s*\)",
                r"普通最小二乘",
                r"最小二乘估计",
                r"满列秩",
            ),
            (r"\bnormal\s+equations?\b", r"\bbeta_?hat\b", r"正规方程"),
            "Write beta_hat as the projection solution and use the full-rank assumption exactly where inversion is needed. "
            "For covariance, propagate epsilon through the linear estimator rather than memorizing the formula.",
        ),
        (
            "rank_deficiency_pseudoinverse",
            (
                r"\brank[- ]deficient\b",
                r"\brank\s+deficien",
                r"\bpseudoinverse\b",
                r"\bmoore[- ]penrose\b",
                r"\bnonunique\s+(?:coefficient|solution)",
                r"\bX\^T\s*X\b[^.\n]{0,40}\bsingular\b",
                r"秩亏",
                r"伪逆",
                r"Moore[- ]Penrose",
                r"系数不唯一",
            ),
            (r"\bsingular\b", r"\bnonunique\b", r"不唯一"),
            "Use rank and null-space geometry first. State why the normal equations are non-unique and use the Moore-Penrose "
            "pseudoinverse only as a chosen canonical solution, not as proof of identifiability.",
        ),
        (
            "sampling_inference",
            (
                r"\bconfidence\s+interval\b[^.\n]{0,60}\bregression\b",
                r"\bstandard\s+error\b[^.\n]{0,60}\bcoefficient\b",
                r"\bt[- ]statistic\b",
                r"\bprediction\s+interval\b",
                r"回归系数[^。；\n]{0,30}置信区间",
                r"预测区间",
            ),
            (r"\bstandard\s+error\b", r"\binference\b", r"标准误"),
            "Separate coefficient uncertainty from prediction uncertainty. Use the correct residual variance estimate, "
            "degrees of freedom, and leverage term for the requested inferential quantity.",
        ),
    ),
    "pde": (
        (
            "heat_wave_separation",
            (
                r"\bheat\s+equation\b",
                r"\bwave\s+equation\b",
                r"\bseparation\s+of\s+variables\b",
                r"\beigenfunction\s+expansion\b",
                r"热方程",
                r"波动方程",
                r"分离变量",
                r"特征函数展开",
            ),
            (r"\bdirichlet\b", r"\bneumann\b", r"边界条件"),
            "Match the boundary conditions to the correct eigenbasis before expanding the initial data. Keep time factors "
            "and eigenvalues paired with the same spatial mode.",
        ),
        (
            "transport_characteristics",
            (
                r"\btransport\s+equation\b",
                r"\badvection\s+equation\b",
                r"\bmethod\s+of\s+characteristics\b",
                r"\bcharacteristic\s+curves?\b",
                r"\bcharacteristic\s+form\b",
                r"\bderive\s+(?:the\s+)?characteristics?\b",
                r"输运方程",
                r"对流方程",
                r"特征线法",
                r"特征曲线",
            ),
            (r"\bcharacteristics?\b", r"特征线"),
            "Write the characteristic ODEs first and identify the invariant along them. Apply initial or boundary data only "
            "after tracing each point back to the data surface.",
        ),
        (
            "transform_fundamental_solution",
            (
                r"\bfourier\s+transform\b",
                r"\blaplace\s+transform\b",
                r"\bfundamental\s+solution\b",
                r"\bgreen(?:'s)?\s+function\b",
                r"傅里叶变换",
                r"拉普拉斯变换",
                r"基本解",
                r"Green函数",
            ),
            (r"\btransform\b", r"\bgreen\b", r"变换"),
            "Transform the PDE together with its data, solve the transformed algebraic/ODE problem, and invert with the "
            "correct normalization. For fundamental solutions, verify the singular source and boundary behavior.",
        ),
    ),
    "linear_algebra": (
        (
            "rank_system",
            (
                r"\brank[- ]nullity\b",
                r"\bnullity\b",
                r"\bnull\s*space\b",
                r"\bsolve\s+(?:the\s+)?linear\s+system\b",
                r"\brank\s+of\s+(?:the\s+)?matrix\b",
                r"秩-零化度",
                r"零空间",
                r"解线性方程组",
                r"矩阵的秩",
            ),
            (r"\brank\b", r"\bsystem\b", r"秩", r"方程组"),
            "Use row-space/null-space structure and rank-nullity before brute-force elimination. Track free variables and "
            "consistency conditions explicitly.",
        ),
        (
            "eigen_jordan",
            (
                r"\bjordan\s+(?:form|blocks?)\b",
                r"\bminimal\s+polynomial\b",
                r"\bdiagonaliz",
                r"\beigenvalues?\b[^.\n]{0,60}\bmultiplicit",
                r"Jordan",
                r"最小多项式",
                r"对角化",
                r"特征值[^。；\n]{0,30}重数",
            ),
            (r"\beigenvalue\b", r"\beigenvector\b", r"特征值", r"特征向量"),
            "Separate algebraic and geometric multiplicities. Use the minimal polynomial or eigenspace dimensions to "
            "constrain Jordan blocks before constructing a basis.",
        ),
        (
            "quadratic_form_spectral",
            (
                r"\bquadratic\s+form\b",
                r"\bpositive\s+definite\b",
                r"\bspectral\s+theorem\b[^.\n]{0,40}\bsymmetric\b",
                r"二次型",
                r"正定",
                r"谱定理[^。；\n]{0,20}对称",
            ),
            (r"\bsymmetric\s+matrix\b", r"对称矩阵"),
            "Exploit symmetry and inertia before expanding determinants. Use orthogonal diagonalization for symmetric forms "
            "and distinguish eigenvalue signs from mere nonsingularity.",
        ),
    ),
    "optimization": (
        (
            "kkt_convex",
            (
                r"\bkkt\b",
                r"\bkarush[- ]kuhn[- ]tucker\b",
                r"\bcomplementary\s+slackness\b",
                r"\bconvex\b[^.\n]{0,60}\binequality\s+constraint",
                r"KKT",
                r"互补松弛",
                r"凸[^。；\n]{0,30}不等式约束",
            ),
            (r"\bconvex\b", r"\binequality\s+constraint", r"凸", r"不等式约束"),
            "Write primal feasibility, dual feasibility, stationarity, and complementary slackness together. Use convexity "
            "only after checking it, and inspect active-set boundary cases.",
        ),
        (
            "lagrange_boundary",
            (
                r"\blagrange\s+multipliers?\b",
                r"\bequality\s+constraint\b",
                r"\bboundary\s+case\b",
                r"拉格朗日乘子",
                r"等式约束",
                r"边界情形",
            ),
            (r"\blagrange\b", r"\bboundary\b", r"约束"),
            "Form the Lagrangian for equality constraints and solve all stationary equations, then compare feasible "
            "boundary or endpoint cases rather than assuming every stationary point is optimal.",
        ),
        (
            "unconstrained_hessian",
            (
                r"\bunconstrained\b[^.\n]{0,50}\b(?:minimum|maximum|optimization)\b",
                r"\bhessian\b",
                r"\bsecond[- ]order\s+condition",
                r"无约束[^。；\n]{0,30}(?:最小|最大|优化)",
                r"Hessian",
                r"二阶条件",
            ),
            (r"\bstationary\s+point\b", r"驻点"),
            "Solve the gradient equations first and use the Hessian or another global argument to classify candidates. "
            "A stationary point alone is not a minimum.",
        ),
    ),
    "real_analysis": (
        (
            "sequence_series_convergence",
            (
                r"\bseries\b[^.\n]{0,50}\bconver",
                r"\bconvergence\s+test\b",
                r"\bcauchy\s+sequence\b",
                r"\babsolutely\s+convergent\b",
                r"级数[^。；\n]{0,30}收敛",
                r"收敛判别",
                r"Cauchy序列",
                r"绝对收敛",
            ),
            (r"\bsequence\b", r"\bconvergen", r"序列", r"收敛"),
            "Identify the exact convergence notion and use a criterion matched to it. For series, separate absolute from "
            "conditional convergence and avoid importing uniform-convergence conclusions.",
        ),
        (
            "uniform_pointwise",
            (
                r"\buniform(?:ly)?\s+conver",
                r"\bconver\w*\s+uniformly\b",
                r"\bpointwise\s+conver",
                r"\bnot\s+uniform(?:ly)?\b",
                r"\bneed\s+not\s+be\s+uniform\b",
                r"\bnot\s+(?:be\s+)?uniform(?:ly)?\b",
                r"一致收敛",
                r"点态收敛",
                r"非一致收敛",
            ),
            (r"\bsup\s*norm\b", r"上确界范数"),
            "Compare pointwise and uniform convergence using the supremum error on the whole domain. Check endpoint or "
            "moving-peak behavior before applying limit-interchange theorems.",
        ),
        (
            "compactness_continuity",
            (
                r"\buniformly\s+continuous\b",
                r"\bheine[- ]borel\b",
                r"\bbolzano[- ]weierstrass\b",
                r"连续[^。；\n]{0,30}紧",
                r"一致连续",
                r"Heine[- ]Borel",
            ),
            (r"\bcontinuous\b[^.\n]{0,60}\bcompact\s+(?:set|interval|space)\b", r"\bcompact\b", r"\bcontinuous\b", r"紧致", r"连续"),
            "Use compactness only for conclusions it actually supports, such as bounded extrema or uniform continuity. "
            "Track whether the domain is compact and which continuity notion is requested.",
        ),
    ),
    "topology": (
        (
            "compact_hausdorff",
            (
                r"\bcompact\b[^.\n]{0,60}\bhausdorff\b",
                r"\bhausdorff\b[^.\n]{0,60}\bcompact\b",
                r"\bcontinuous\s+bijection\b[^.\n]{0,80}\bhomeomorphism\b",
                r"紧致[^。；\n]{0,30}Hausdorff",
                r"Hausdorff[^。；\n]{0,30}紧致",
            ),
            (r"\bcompact\b", r"\bhausdorff\b", r"\bhomeomorphism\b", r"紧致", r"同胚"),
            "Use compactness plus Hausdorff separation only where both hypotheses are present. For continuous bijections, "
            "prove the inverse is continuous via closed or compact images rather than metric intuition.",
        ),
        (
            "connectedness",
            (
                r"\bpath[- ]connected\b",
                r"\bconnected\s+(?:space|subset|set|image)\b",
                r"\bcomponents?\b",
                r"道路连通",
                r"连通空间",
                r"连通子集",
                r"连通分支",
            ),
            (r"\bconnected\b", r"连通"),
            "Use separation or continuous-image arguments according to the requested property. Keep connectedness and "
            "path-connectedness distinct unless the space supplies extra structure.",
        ),
        (
            "continuity_quotient",
            (
                r"\bquotient\s+topology\b",
                r"\bquotient\s+map\b",
                r"\binverse\s+images?\s+of\s+open\s+sets?\b",
                r"\bcontinuous\s+map\b[^.\n]{0,60}\binverse\s+image\b",
                r"商拓扑",
                r"商映射",
                r"开集的原像",
            ),
            (r"\bcontinuous\s+map\b", r"\bhomeomorphism\b", r"连续映射", r"同胚"),
            "Use inverse images for continuity and the defining saturated-open condition for quotient maps. Do not assume "
            "metric or open-map properties unless they are given or proved.",
        ),
    ),

}

_SCORE_FIRST_WRAPPER_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE | re.DOTALL)
    for pattern in (
        r"^\s*(?:final\s+answer|the\s+final\s+answer)\s*(?::|=|\bis\b)\s*(?P<payload>.+?)\s*$",
        r"^\s*(?:final\s+result|the\s+final\s+result)\s*(?::|=|\bis\b)\s*(?P<payload>.+?)\s*$",
        r"^\s*(?:answer|the\s+answer)\s*(?::|=|\bis\b)\s*(?P<payload>.+?)\s*$",
        r"^\s*(?:result|the\s+result)\s*(?::|=|\bis\b)\s*(?P<payload>.+?)\s*$",
        r"^\s*(?:therefore|thus|hence)\s*,?\s*(?:the\s+)?answer\s*(?::|=|\bis\b)\s*(?P<payload>.+?)\s*$",
        r"^\s*最终答案\s*(?:[:：=]|是)\s*(?P<payload>.+?)\s*$",
        r"^\s*最后答案\s*(?:[:：=]|是)\s*(?P<payload>.+?)\s*$",
        r"^\s*答案\s*(?:[:：=]|是|为)\s*(?P<payload>.+?)\s*$",
        r"^\s*结果\s*(?:[:：=]|是|为)\s*(?P<payload>.+?)\s*$",
        r"^\s*(?:因此|所以|故)\s*答案\s*(?:[:：=]|是|为)\s*(?P<payload>.+?)\s*$",
    )
)

_PROOF_TASK_MARKERS = (
    "prove",
    "proof",
    "show that",
    "demonstrate that",
    "证明",
    "证实",
    "说明.*成立",
)


class ReasoningAgent:
    def __init__(self, client, *args, **kwargs):
        self.client = client
        self.production_mode = str(kwargs.get("production_mode", "score_first") or "score_first").strip().lower()
        if self.production_mode not in {"score_first", "orchestrated"}:
            raise ValueError("production_mode must be 'score_first' or 'orchestrated'")

        self.max_retries = int(kwargs.get("max_retries", 1))
        default_temperature = 0.1 if self.production_mode == "score_first" else 0.2
        default_max_tokens = 8192 if self.production_mode == "score_first" else 4096
        self.temperature = float(kwargs.get("temperature", default_temperature))
        self.max_tokens = int(kwargs.get("max_tokens", default_max_tokens))
        self.thinking_mode = bool(kwargs.get("thinking_mode", True))
        self.max_candidates = int(kwargs.get("max_candidates", 2))
        self.orchestrator = None

        # The full orchestrator remains available, but only as an explicit opt-in.
        # ScoreFirst deliberately does not construct or invoke the orchestration path.
        if self.production_mode == "orchestrated":
            try:
                from math_agent_core.orchestrator import MathAgentOrchestrator

                self.orchestrator = MathAgentOrchestrator(
                    client=self.client,
                    max_retries=self.max_retries,
                    enable_repair=True,
                    enable_tool_verify=True,
                    backend="simple",
                    thinking_mode=self.thinking_mode,
                    max_candidates=self.max_candidates,
                    solver_max_tokens=self.max_tokens,
                    solver_temperature=self.temperature,
                    enable_critic=bool(kwargs.get("enable_critic", True)),
                    enable_finalizer=bool(kwargs.get("enable_finalizer", False)),
                )
            except Exception:
                self.orchestrator = None

    def solve(self, problem: str, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
        try:
            if not isinstance(problem, str) or not problem.strip():
                return self._fallback_result("problem is empty or not a string")

            safe_metadata = metadata if isinstance(metadata, dict) else {}

            if self.production_mode == "score_first":
                return self._solve_score_first(problem, safe_metadata)

            if self.orchestrator is not None:
                result = self.orchestrator.solve(problem=problem, metadata=safe_metadata)
                final_response = self._extract_final_response(result, problem)
                trace = trace_from_orchestrator_result(result, getattr(self.orchestrator, "last_log", None))
                if final_response == DEFAULT_FALLBACK:
                    raw_output = self._extract_last_raw_output()
                    final_response = extract_final_answer(raw_output, problem=problem)
                if final_response == DEFAULT_FALLBACK:
                    response = self._direct_model_call(problem, safe_metadata)
                    final_response = extract_final_answer(self._normalize_model_response(response), problem=problem)
                    trace.append(make_trace_step("fallback", "orchestrator did not produce a usable answer; used direct client.chat"))
                return self._json_safe_result(final_response, trace)

            response = self._direct_model_call(problem, safe_metadata)
            final_response = extract_final_answer(self._normalize_model_response(response), problem=problem)
            trace = [
                make_trace_step(
                    "fallback",
                    {"mode": "direct client.chat call", "thinking_mode": self.thinking_mode},
                )
            ]
            return self._json_safe_result(final_response, trace)
        except Exception as exc:
            return self._fallback_result(f"{type(exc).__name__}: {str(exc)[:300]}")

    def _solve_score_first(self, problem: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        context = self._score_first_context(problem, metadata)
        response = self._score_first_model_call(problem, metadata, context=context)
        raw_output = self._normalize_model_response(response)
        final_response = self._extract_score_first_response(
            raw_output,
            problem,
            response_mode=context["response_mode"],
        )
        subject_hint = context.get("subject_hint") or ""
        trace = [
            make_trace_step("mode", "score_first"),
            make_trace_step("model_call", "primary client.chat call: 1"),
            make_trace_step("answer", "answer extracted" if final_response != DEFAULT_FALLBACK else "empty/unusable answer"),
        ]
        if subject_hint:
            trace.insert(1, make_trace_step("subject_hint", subject_hint))
        return self._score_first_json_result(final_response, trace)

    def _score_first_model_call(
        self,
        problem: str,
        metadata: Dict[str, Any],
        context: Dict[str, Any] | None = None,
    ) -> Any:
        messages = self._build_score_first_prompt(problem, metadata, context=context)
        return self._chat_once(messages)

    def _direct_model_call(self, problem: str, metadata: Dict[str, Any]) -> Any:
        messages = self._build_direct_prompt(problem, metadata)
        return self._chat_once(messages)

    def _chat_once(self, messages: List[Dict[str, str]]) -> Any:
        kwargs = {
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self._client_supports_thinking_mode():
            kwargs["thinking_mode"] = self.thinking_mode
        return self.client.chat(**kwargs)

    def _client_supports_thinking_mode(self) -> bool:
        try:
            signature = inspect.signature(self.client.chat)
        except (TypeError, ValueError):
            return True
        if "thinking_mode" in signature.parameters:
            return True
        return any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())

    def _build_score_first_prompt(
        self,
        problem: str,
        metadata: Dict[str, Any],
        context: Dict[str, Any] | None = None,
    ) -> List[Dict[str, str]]:
        context = context or self._score_first_context(problem, metadata)
        subject = str(context.get("subject_hint") or "").strip()
        subject_line = f"Subject hint: {subject}\n\n" if subject else ""
        response_mode = context["response_mode"]
        system_prompt = self._score_first_system_prompt(response_mode)
        strategy_block = self._score_first_strategy_block(context)
        verification_block = self._score_first_verification_block(context)
        user_instruction = {
            _SCORE_FIRST_RESPONSE_MODE_ANSWER: "Give the requested mathematical answer.",
            _SCORE_FIRST_RESPONSE_MODE_DERIVATION: "Give the requested result and derivation.",
            _SCORE_FIRST_RESPONSE_MODE_PROOF: "Give the requested proof.",
            _SCORE_FIRST_RESPONSE_MODE_PROOF_OR_DISPROOF: "Determine the claim and prove or disprove it as requested.",
            _SCORE_FIRST_RESPONSE_MODE_CONSTRUCTION: "Give the requested construction or counterexample.",
        }[response_mode]
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"{subject_line}{strategy_block}{verification_block}"
                    f"Problem:\n{problem}\n\n{user_instruction}"
                ),
            },
        ]

    def _score_first_system_prompt(self, response_mode: str) -> str:
        if response_mode == _SCORE_FIRST_RESPONSE_MODE_DERIVATION:
            base = _SCORE_FIRST_DERIVATION_SYSTEM_PROMPT
        elif response_mode == _SCORE_FIRST_RESPONSE_MODE_PROOF:
            base = _SCORE_FIRST_PROOF_SYSTEM_PROMPT
        elif response_mode == _SCORE_FIRST_RESPONSE_MODE_PROOF_OR_DISPROOF:
            base = _SCORE_FIRST_PROOF_OR_DISPROOF_SYSTEM_PROMPT
        elif response_mode == _SCORE_FIRST_RESPONSE_MODE_CONSTRUCTION:
            base = _SCORE_FIRST_CONSTRUCTION_SYSTEM_PROMPT
        else:
            base = _SCORE_FIRST_ANSWER_SYSTEM_PROMPT

        if response_mode == _SCORE_FIRST_RESPONSE_MODE_ANSWER:
            completeness = (
                "Before emitting the final answer, internally check: all requested parts answered; "
                "all stated constraints/conditions preserved."
            )
        else:
            completeness = (
                "Before finalizing, ensure every requested part has an answer and all stated "
                "constraints/conditions are preserved."
            )
        return (
            base
            + "\n\n"
            + _SCORE_FIRST_EXACTNESS_DISCIPLINE
            + "\n"
            + completeness
        )

    def _score_first_context(self, problem: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        trusted_domain, trusted_key = self._trusted_score_first_domain(metadata)
        router_metadata = self._canonical_score_first_router_metadata(
            metadata,
            trusted_domain=trusted_domain,
            trusted_key=trusted_key,
        )
        route = classify_problem(problem, router_metadata)
        request_records = self._score_first_request_span_records(problem)
        request_spans = [record["text"] for record in request_records]
        requested_actions = self._score_first_requested_actions(request_spans)
        target_text = " ".join(request_spans).strip()
        context_text = self._score_first_context_text(problem, request_records)
        response_mode = self._score_first_response_mode(
            problem,
            metadata,
            route,
            request_spans=request_spans,
        )

        route_primary = str(route.get("primary_domain") or "").strip().lower()
        route_scores = route.get("scores") if isinstance(route.get("scores"), dict) else {}
        top_score = float(route_scores.get(route_primary, 0.0) or 0.0)
        second_score = max(
            (
                float(score or 0.0)
                for domain, score in route_scores.items()
                if str(domain).strip().lower() != route_primary
            ),
            default=0.0,
        )
        route_margin = top_score - second_score

        if trusted_domain:
            strategy_domain = trusted_domain
            strategy_is_specialized = strategy_domain != "advanced_math"
            domain_source = f"trusted:{trusted_key}"
        elif (
            route_primary in _SCORE_FIRST_DOMAIN_STRATEGIES
            and top_score >= _SCORE_FIRST_NO_SUBJECT_SPECIALIZED_SCORE_MIN
            and route_margin >= _SCORE_FIRST_NO_SUBJECT_SPECIALIZED_MARGIN_MIN
        ):
            strategy_domain = route_primary
            strategy_is_specialized = strategy_domain != "advanced_math"
            domain_source = "router_high_confidence"
        else:
            strategy_domain = "advanced_math"
            strategy_is_specialized = False
            domain_source = "general_fallback"

        discrete_subtype = None
        if strategy_domain == "discrete_math":
            subtype = str(route.get("discrete_subtype") or "").strip().lower()
            if subtype in _SCORE_FIRST_DISCRETE_SUBTYPE_STRATEGIES:
                discrete_subtype = subtype

        micro_name = None
        micro_card = None
        micro_match = None
        if strategy_domain != "discrete_math":
            micro_match = self._score_first_micro_strategy(
                strategy_domain,
                problem,
                target_text=target_text,
                context_text=context_text,
                has_request_spans=bool(request_spans),
            )
            if micro_match:
                micro_name = micro_match["name"]
                micro_card = micro_match["card"]

        verification_key, verification_variant, verification_card = self._score_first_final_check(
            strategy_domain=strategy_domain,
            discrete_subtype=discrete_subtype,
            micro_strategy=micro_name,
            target_text=target_text,
            response_mode=response_mode,
            requested_actions=requested_actions,
        )

        subject_hint = self._trusted_score_first_subject_hint(
            metadata,
            trusted_domain=trusted_domain,
            trusted_key=trusted_key,
        )

        return {
            "response_mode": response_mode,
            "request_spans": request_spans,
            "requested_actions": requested_actions,
            "target_text": target_text,
            "context_text": context_text,
            "route": route,
            "trusted_domain": trusted_domain,
            "trusted_key": trusted_key,
            "strategy_domain": strategy_domain,
            "strategy_is_specialized": strategy_is_specialized,
            "domain_source": domain_source,
            "route_top_score": top_score,
            "route_second_score": second_score,
            "route_margin": route_margin,
            "subject_hint": subject_hint,
            "discrete_subtype": discrete_subtype,
            "micro_strategy": micro_name,
            "micro_card": micro_card,
            "micro_match": micro_match,
            "verification_key": verification_key,
            "verification_variant": verification_variant,
            "verification_card": verification_card,
        }

    def _score_first_response_mode(
        self,
        problem: str,
        metadata: Dict[str, Any],
        route: Dict[str, Any],
        request_spans: List[str] | None = None,
    ) -> str:
        spans = request_spans if request_spans is not None else self._score_first_request_spans(problem)
        requested_actions = self._score_first_requested_actions(spans)
        explicit_intent = self._score_first_aggregate_requested_actions(requested_actions)
        if explicit_intent is not None:
            return explicit_intent

        # Explicit exclusions outrank contradictory trusted metadata. Router
        # task_type remains diagnostic only.
        excluded_modes = self._score_first_explicit_excluded_modes(problem)
        metadata_task = str(metadata.get("task_type") or "").strip().lower()
        task_modes = {
            "proof": _SCORE_FIRST_RESPONSE_MODE_PROOF,
            "derivation": _SCORE_FIRST_RESPONSE_MODE_DERIVATION,
            "construction": _SCORE_FIRST_RESPONSE_MODE_CONSTRUCTION,
            "counterexample": _SCORE_FIRST_RESPONSE_MODE_CONSTRUCTION,
            "calculation": _SCORE_FIRST_RESPONSE_MODE_ANSWER,
            "choice": _SCORE_FIRST_RESPONSE_MODE_ANSWER,
        }
        if metadata_task in task_modes:
            metadata_mode = task_modes[metadata_task]
            if metadata_mode in excluded_modes:
                return _SCORE_FIRST_RESPONSE_MODE_ANSWER
            return metadata_mode
        return _SCORE_FIRST_RESPONSE_MODE_ANSWER

    def _score_first_request_spans(self, problem: str) -> List[str]:
        return [
            record["text"]
            for record in self._score_first_request_span_records(problem)
        ]

    def _score_first_clause_records(self, problem: str) -> List[Dict[str, Any]]:
        text = str(problem or "")
        if not text:
            return []

        protected_abbreviations: List[tuple[int, int]] = []
        for abbreviation in _SCORE_FIRST_PROTECTED_DOT_ABBREVIATIONS:
            protected_abbreviations.extend(
                (match.start(), match.end())
                for match in re.finditer(
                    re.escape(abbreviation),
                    text,
                    flags=re.IGNORECASE,
                )
            )

        records: List[Dict[str, Any]] = []
        start = 0
        index = 0
        while index < len(text):
            char = text[index]
            boundary = False
            if char in "?!;？！；。\n":
                boundary = True
            elif char == ".":
                boundary = self._score_first_period_is_clause_boundary(
                    text,
                    index,
                    protected_abbreviations,
                )

            if boundary:
                end = index + 1
                clause = text[start:end]
                if clause.strip():
                    records.append({"start": start, "end": end, "text": clause})
                start = end
            index += 1

        if start < len(text):
            clause = text[start:]
            if clause.strip():
                records.append({"start": start, "end": len(text), "text": clause})
        return records

    def _score_first_period_is_clause_boundary(
        self,
        text: str,
        index: int,
        protected_abbreviations: List[tuple[int, int]],
    ) -> bool:
        if index > 0 and index + 1 < len(text):
            if text[index - 1].isdigit() and text[index + 1].isdigit():
                return False

        for start, end in protected_abbreviations:
            if start <= index < end:
                # Internal dots in a.e./i.e./e.g./w.r.t. never split. The final dot
                # can also terminate a sentence if followed by end-of-input or a
                # clearly new capitalized/request clause.
                if index < end - 1:
                    return False
                cursor = index + 1
                while cursor < len(text) and text[cursor] in " \t\r":
                    cursor += 1
                if cursor >= len(text) or text[cursor] == "\n":
                    return True
                next_char = text[cursor]
                if next_char.isupper() or re.match(r"[\u4e00-\u9fff]", next_char):
                    return True
                return False

        cursor = index + 1
        while cursor < len(text) and text[cursor] in "\"'”’)]}":
            cursor += 1
        if cursor >= len(text):
            return True
        return text[cursor].isspace()

    def _score_first_request_span_records(self, problem: str) -> List[Dict[str, Any]]:
        text = str(problem or "")
        if not text.strip():
            return []

        records: List[Dict[str, Any]] = []
        for clause_record in self._score_first_clause_records(text):
            clause = str(clause_record["text"])
            clause_start = int(clause_record["start"])
            anchor_start = self._score_first_first_request_anchor(clause)
            if anchor_start is None:
                continue
            start = clause_start + anchor_start
            end = int(clause_record["end"])
            span = text[start:end].strip()
            if span:
                records.append({"start": start, "end": end, "text": span})
        return records

    def _score_first_first_request_anchor(self, clause: str) -> int | None:
        if not clause.strip():
            return None

        # Interrogatives do not need an imperative verb.
        question_like = clause.rstrip().endswith(("?", "？"))
        if question_like:
            english_q = self._score_first_first_valid_english_interrogative(clause)
            if english_q is not None:
                return english_q
            chinese_q = _SCORE_FIRST_CHINESE_INTERROGATIVE_RE.search(clause)
            if chinese_q is not None:
                # The whole Chinese question is the requested object; retaining its
                # subject is safer for mathematics than starting at one character
                # inside "是否/多少/哪个".
                return 0

        candidates: List[int] = []

        for match in _SCORE_FIRST_ENGLISH_REQUEST_VERB_RE.finditer(clause):
            if self._score_first_is_english_request_position(clause, match.start()):
                candidates.append(match.start())

        if not candidates:
            for match in _SCORE_FIRST_ENGLISH_DEFERRED_REQUEST_VERB_RE.finditer(clause):
                if self._score_first_is_english_request_position(clause, match.start()):
                    candidates.append(match.start())

        for phrase in _SCORE_FIRST_CHINESE_REQUEST_PHRASES:
            search_start = 0
            while True:
                position = clause.find(phrase, search_start)
                if position < 0:
                    break
                if self._score_first_is_chinese_request_position(
                    clause,
                    position,
                    phrase=phrase,
                ):
                    candidates.append(position)
                search_start = position + max(1, len(phrase))

        return min(candidates) if candidates else None

    def _score_first_first_valid_english_interrogative(self, clause: str) -> int | None:
        for match in _SCORE_FIRST_ENGLISH_INTERROGATIVE_RE.finditer(clause):
            if self._score_first_is_english_request_position(
                clause,
                match.start(),
                allow_interrogative_subject=True,
            ):
                return match.start()
        return None

    def _score_first_is_english_request_position(
        self,
        clause: str,
        start: int,
        *,
        allow_interrogative_subject: bool = False,
    ) -> bool:
        prefix = clause[:start]
        stripped = prefix.strip()
        if not stripped:
            return True
        if re.fullmatch(
            r"(?:please|briefly|first|then|next|finally|now)"
            r"(?:\s+(?:please|briefly|first|then|next|finally|now))*",
            stripped,
            flags=re.IGNORECASE,
        ):
            return True
        if re.search(
            r"(?:[,:\uFF0C\uFF1A]\s*|\b(?:and|then|also)\s+)"
            r"(?:(?:please|briefly|first|now)\s+)*$",
            prefix,
            flags=re.IGNORECASE,
        ):
            return True

        # For an English yes/no or wh-question, a subject may precede an embedded
        # Chinese-style marker, but English auxiliaries themselves should still
        # begin a grammatical question or follow a comma/coordination.
        if allow_interrogative_subject and clause.rstrip().endswith("?"):
            return False
        return False

    def _score_first_is_chinese_request_position(
        self,
        clause: str,
        start: int,
        *,
        phrase: str = "",
    ) -> bool:
        prefix = clause[:start]
        stripped = prefix.strip()
        if not stripped:
            return True

        # Request particles/coordinators can directly introduce the action.
        if re.search(r"(?:请|试|并|再|然后|接着|只|仅|只需|仅需)\s*$", prefix):
            return True

        # A comma/colon separates background from the requested action. Optional
        # request particles may follow the delimiter.
        if re.search(
            r"[，,:：]\s*(?:(?:请|试|并|再|然后|接着|只|仅|只需|仅需)\s*)*$",
            prefix,
        ):
            return True

        # Common imperative prepositional form: "对 <object> 使用/利用/采用/应用 ...".
        # Restrict this exception to multi-character method directives so a raw 用
        # inside words such as 作用 cannot become a request anchor.
        if phrase in {"使用", "利用", "采用", "应用"} and re.fullmatch(
            r"\s*对[^，,。；;！？!?]{1,80}",
            prefix,
        ):
            return True
        return False

    def _score_first_context_text(
        self,
        problem: str,
        request_records: List[Dict[str, Any]],
    ) -> str:
        text = str(problem or "")
        if not request_records:
            return ""
        pieces: List[str] = []
        cursor = 0
        for record in request_records:
            start = int(record["start"])
            end = int(record["end"])
            if cursor < start:
                pieces.append(text[cursor:start])
            cursor = max(cursor, end)
        if cursor < len(text):
            pieces.append(text[cursor:])
        return " ".join(piece.strip() for piece in pieces if piece.strip()).strip()

    def _score_first_explicit_response_intent(
        self,
        problem: str,
        request_spans: List[str] | None = None,
    ) -> str | None:
        spans = request_spans if request_spans is not None else self._score_first_request_spans(problem)
        return self._score_first_aggregate_requested_actions(
            self._score_first_requested_actions(spans)
        )

    def _score_first_requested_actions(self, request_spans: List[str]) -> List[str]:
        actions: List[str] = []
        for span in request_spans:
            for action in self._score_first_requested_actions_for_span(span):
                if action not in actions:
                    actions.append(action)
        return actions

    def _score_first_requested_actions_for_span(self, span: str) -> List[str]:
        text = str(span or "").strip()
        if not text:
            return []

        matched = {
            family: any(
                re.search(pattern, text, flags=re.IGNORECASE)
                for pattern in patterns
            )
            for family, patterns in _SCORE_FIRST_REQUEST_INTENT_PATTERNS.items()
        }

        actions: List[str] = []
        if matched.get("choice") or self._score_first_span_requests_answer_value(text):
            actions.append(_SCORE_FIRST_RESPONSE_MODE_ANSWER)
        if matched.get("derivation"):
            actions.append(_SCORE_FIRST_RESPONSE_MODE_DERIVATION)
        if matched.get("proof"):
            actions.append(_SCORE_FIRST_RESPONSE_MODE_PROOF)
        if matched.get("construction"):
            actions.append(_SCORE_FIRST_RESPONSE_MODE_CONSTRUCTION)
        if matched.get("proof_or_disproof"):
            actions.append(_SCORE_FIRST_RESPONSE_MODE_PROOF_OR_DISPROOF)

        # Any successfully grounded request span still requests an answer even if its
        # action family is not one of the rich visible-reasoning modes.
        if not actions:
            actions.append(_SCORE_FIRST_RESPONSE_MODE_ANSWER)
        return actions

    def _score_first_span_requests_answer_value(self, span: str) -> bool:
        text = str(span or "")
        return bool(
            re.search(
                r"""
                ^\s*(?:
                    compute|calculate|evaluate|find|determine|solve|classify|identify|
                    state|answer|differentiate|integrate|simplify|factor|factorize|expand|
                    approximate|estimate|maximize|minimize|optimize|diagonalize|diagonalise|
                    invert|normalize|normalise|parameterize|parametrize|
                    which|what|how\s+(?:many|much|large|fast)|
                    is|are|does|do|can|could|will|select|choose
                )\b
                |^(?:求导|积分|化简|因式分解|展开|近似|估计|最大化|最小化|优化|对角化|求逆|归一化|参数化|求出|计算|确定|判断|指出|写出|选择|哪个|是否|能否|可否|多少|几)
                """,
                text,
                flags=re.IGNORECASE | re.VERBOSE,
            )
        )

    def _score_first_aggregate_requested_actions(self, actions: List[str]) -> str | None:
        if not actions:
            return None
        priority = (
            _SCORE_FIRST_RESPONSE_MODE_PROOF_OR_DISPROOF,
            _SCORE_FIRST_RESPONSE_MODE_CONSTRUCTION,
            _SCORE_FIRST_RESPONSE_MODE_PROOF,
            _SCORE_FIRST_RESPONSE_MODE_DERIVATION,
            _SCORE_FIRST_RESPONSE_MODE_ANSWER,
        )
        for mode in priority:
            if mode in actions:
                return mode
        return None

    def _score_first_explicit_excluded_modes(self, problem: str) -> set[str]:
        text = str(problem or "")
        excluded: set[str] = set()

        if re.search(
            r"\b(?:no\s+proof(?:\s+is\s+required)?|do\s+not\s+prove|need\s+not\s+prove|without\s+proving)\b"
            r"|(?:无需|不必|不用|不要|不要求)(?:先|给出)?证明",
            text,
            flags=re.IGNORECASE,
        ):
            excluded.add(_SCORE_FIRST_RESPONSE_MODE_PROOF)
            excluded.add(_SCORE_FIRST_RESPONSE_MODE_PROOF_OR_DISPROOF)

        if re.search(
            r"\b(?:no\s+derivation(?:\s+is\s+required)?|do\s+not\s+derive|need\s+not\s+derive|without\s+deriving|rather\s+than\s+derive)\b"
            r"|(?:无需|不必|不用|不要|不要求)(?:先|给出)?推导",
            text,
            flags=re.IGNORECASE,
        ):
            excluded.add(_SCORE_FIRST_RESPONSE_MODE_DERIVATION)

        if re.search(
            r"\b(?:do\s+not\s+construct|need\s+not\s+construct|without\s+constructing|constructing\b[^.;!?\n]{0,80}\b(?:is|was)\s+unnecessary)\b"
            r"|(?:无需|不必|不用|不要|不要求)(?:先|给出)?构造",
            text,
            flags=re.IGNORECASE,
        ):
            excluded.add(_SCORE_FIRST_RESPONSE_MODE_CONSTRUCTION)

        return excluded

    def _score_first_has_nonrequest_action_cue(self, problem: str) -> bool:
        text = re.sub(r"\s+", " ", str(problem or "")).strip()
        return any(
            re.search(pattern, text, flags=re.IGNORECASE)
            for pattern in _SCORE_FIRST_NONREQUEST_ACTION_PATTERNS
        )

    def _trusted_score_first_domain(self, metadata: Dict[str, Any]) -> tuple[str | None, str | None]:
        for key in ("subject", "type", "category"):
            domain = self._canonical_score_first_domain_label(metadata.get(key))
            if domain:
                return domain, key
        return None, None

    def _canonical_score_first_domain_label(self, value: Any) -> str | None:
        raw = str(value or "").strip()
        if not raw:
            return None
        normalized = re.sub(r"[\s_\-]+", " ", raw.casefold()).strip()
        return _SCORE_FIRST_TRUSTED_DOMAIN_ALIASES.get(normalized)

    def _trusted_score_first_subject_hint(
        self,
        metadata: Dict[str, Any],
        trusted_domain: str | None,
        trusted_key: str | None,
    ) -> str:
        if not trusted_domain or not trusted_key:
            return ""
        raw = re.sub(r"\s+", " ", str(metadata.get(trusted_key) or "")).strip()
        if not raw:
            return _SCORE_FIRST_HUMAN_DOMAIN_LABELS.get(trusted_domain, trusted_domain)
        # Canonical snake-case labels are implementation identifiers; render their
        # stable human-readable label. Recognized human/Chinese labels may remain
        # visible because they are already trusted domain labels.
        normalized_raw = raw.casefold()
        if "_" in raw or normalized_raw == trusted_domain.casefold():
            return _SCORE_FIRST_HUMAN_DOMAIN_LABELS.get(trusted_domain, trusted_domain)
        return raw[:120]

    def _canonical_score_first_router_metadata(
        self,
        metadata: Dict[str, Any],
        trusted_domain: str | None,
        trusted_key: str | None,
    ) -> Dict[str, Any]:
        canonical = dict(metadata)
        if not trusted_domain or not trusted_key:
            return canonical

        # First trusted subject/type/category wins. Remove only conflicting trusted
        # domain labels from lower-priority keys; ordinary metadata remains intact.
        found = False
        for key in ("subject", "type", "category"):
            if key == trusted_key:
                canonical[key] = trusted_domain
                found = True
                continue
            if found and self._canonical_score_first_domain_label(metadata.get(key)):
                canonical.pop(key, None)
        return canonical

    def _score_first_strategy_block(self, context: Dict[str, Any]) -> str:
        domain = str(context.get("strategy_domain") or "advanced_math")
        domain_card = _SCORE_FIRST_DOMAIN_STRATEGIES.get(
            domain,
            _SCORE_FIRST_DOMAIN_STRATEGIES["advanced_math"],
        )
        blocks = [f"Domain strategy:\n{domain_card}"]

        subtype = context.get("discrete_subtype")
        if subtype:
            blocks.append(
                "Subtype hint:\n"
                + _SCORE_FIRST_DISCRETE_SUBTYPE_STRATEGIES[str(subtype)]
            )

        micro_card = context.get("micro_card")
        if micro_card:
            blocks.append(f"Method hint:\n{micro_card}")

        return "\n\n".join(blocks) + "\n\n"

    def _score_first_final_check(
        self,
        *,
        strategy_domain: str,
        discrete_subtype: str | None,
        micro_strategy: str | None,
        target_text: str,
        response_mode: str,
        requested_actions: List[str] | None,
    ) -> tuple[str, str, str]:
        target = str(target_text or "")
        requested = list(requested_actions or [])

        def has(pattern: str) -> bool:
            return bool(re.search(pattern, target, flags=re.IGNORECASE))

        newton_convergence_target = has(
            r"\b(?:quadratic(?:ally)?|convergence\s+order|order\s+of\s+convergence|"
            r"local\s+convergence|convergen\w*)\b|二次收敛|收敛阶|局部收敛"
        )
        indicator_count_target = has(
            r"\bindicator(?:\s+variables?)?\b|\bexpected\s+(?:number|count)\b|"
            r"\bsum\s+of\s+(?:events|indicators)\b|指示变量|期望(?:个数|数量|数目)"
        )
        moment_target = has(
            r"\bE\s*\[[^\]]+\]|\bVar\s*\(|\bmoment\b|\bmean\b|\bexpectation\b|"
            r"\bexpected\s+value\b|期望|方差|矩"
        )
        ols_covariance_target = has(
            r"\bVar\s*\(\s*beta_?hat|\bCov\s*\(\s*beta_?hat|"
            r"\bcovariance\b[^.\n]{0,60}\b(?:OLS|beta|coefficient)|"
            r"\bsampling\s+variance\b|协方差|(?:OLS|beta|系数)[^。；\n]{0,30}方差"
        )
        mle_asymptotic_target = has(
            r"\basymptotic\s+(?:variance|distribution)\b|\bFisher\s+information\b|"
            r"\bstandard\s+error\b[^.\n]{0,50}\bMLE\b|渐近方差|渐近分布|"
            r"Fisher信息|费舍尔信息|标准误"
        )
        pde_characteristics_target = has(
            r"\b(?:derive|find|obtain)\b[^.\n]{0,60}\bcharacteristic(?:s|\s+equations?)\b|"
            r"推导[^。；\n]{0,40}特征(?:线|方程)"
        )

        # Task semantics dominate computational micro checks.
        if response_mode == _SCORE_FIRST_RESPONSE_MODE_PROOF_OR_DISPROOF:
            return (
                "task:proof_or_disproof",
                "proof_or_disproof",
                _SCORE_FIRST_TASK_FINAL_CHECKS["proof_or_disproof"],
            )

        construction_is_inferential_interval = has(
            r"\b(?:confidence|prediction)\s+interval\b|置信区间|预测区间"
        )
        if (
            response_mode == _SCORE_FIRST_RESPONSE_MODE_CONSTRUCTION
            and not construction_is_inferential_interval
        ):
            return (
                "task:construction_counterexample",
                "construction_counterexample",
                _SCORE_FIRST_TASK_FINAL_CHECKS["construction_counterexample"],
            )

        if response_mode == _SCORE_FIRST_RESPONSE_MODE_PROOF:
            if micro_strategy == "newton_fixed_point" and newton_convergence_target:
                return (
                    "task:proof",
                    "newton_convergence_proof",
                    (
                        "Verify every stated hypothesis and the exact conclusion. For Newton convergence, check the "
                        "simple-root/nonzero-derivative and regularity assumptions, then confirm the local error relation "
                        "implies the claimed order, e.g. e_(n+1)=O(e_n^2)."
                    ),
                )
            return (
                "task:proof",
                "proof",
                _SCORE_FIRST_TASK_FINAL_CHECKS["proof"],
            )

        # Target-aware variants within an already-selected high-precision micro family.
        if micro_strategy == "newton_fixed_point" and newton_convergence_target:
            return (
                "micro:newton_fixed_point",
                "newton_convergence",
                _SCORE_FIRST_TARGET_FINAL_CHECKS["newton_convergence"],
            )

        if micro_strategy == "expectation_indicator":
            if indicator_count_target:
                return (
                    "micro:expectation_indicator",
                    "indicator_count",
                    _SCORE_FIRST_TARGET_FINAL_CHECKS["expectation_indicator_count"],
                )
            if moment_target:
                return (
                    "micro:expectation_indicator",
                    "general_moment",
                    _SCORE_FIRST_TARGET_FINAL_CHECKS["expectation_moment"],
                )

        if micro_strategy == "ols_full_rank" and ols_covariance_target:
            return (
                "micro:ols_full_rank",
                "covariance",
                _SCORE_FIRST_TARGET_FINAL_CHECKS["ols_covariance"],
            )

        if strategy_domain == "statistics" and mle_asymptotic_target:
            key = (
                "micro:likelihood_mle"
                if micro_strategy == "likelihood_mle"
                else "domain:statistics"
            )
            return (
                key,
                "mle_asymptotic",
                _SCORE_FIRST_TARGET_FINAL_CHECKS["mle_asymptotic"],
            )

        if micro_strategy == "spectrum_invertibility":
            return (
                "micro:spectrum_invertibility",
                "operator_invertibility",
                _SCORE_FIRST_MICRO_FINAL_CHECKS["spectrum_invertibility"],
            )

        # Deriving PDE characteristics should validate the derived characteristic
        # relation even though transport_characteristics has no dedicated V2.5 card.
        if (
            response_mode == _SCORE_FIRST_RESPONSE_MODE_DERIVATION
            and strategy_domain == "pde"
            and micro_strategy == "transport_characteristics"
            and pde_characteristics_target
        ):
            return (
                "domain:pde",
                "characteristics_derivation",
                _SCORE_FIRST_TARGET_FINAL_CHECKS["pde_characteristics_derivation"],
            )

        # Preserve V2.5 hierarchy for ordinary calculation/derivation targets.
        if micro_strategy and micro_strategy in _SCORE_FIRST_MICRO_FINAL_CHECKS:
            return (
                f"micro:{micro_strategy}",
                "base",
                _SCORE_FIRST_MICRO_FINAL_CHECKS[micro_strategy],
            )

        if discrete_subtype and discrete_subtype in _SCORE_FIRST_DISCRETE_FINAL_CHECKS:
            return (
                f"discrete:{discrete_subtype}",
                "base",
                _SCORE_FIRST_DISCRETE_FINAL_CHECKS[discrete_subtype],
            )

        domain = (
            strategy_domain
            if strategy_domain in _SCORE_FIRST_DOMAIN_FINAL_CHECKS
            else "advanced_math"
        )

        # For an otherwise-generic derivation, validate the derived target itself
        # rather than injecting a calculation-only residual.
        if response_mode == _SCORE_FIRST_RESPONSE_MODE_DERIVATION and not requested:
            return (
                f"domain:{domain}",
                "generic_derivation",
                _SCORE_FIRST_TARGET_FINAL_CHECKS["generic_derivation"],
            )

        return (
            f"domain:{domain}",
            "base",
            _SCORE_FIRST_DOMAIN_FINAL_CHECKS[domain],
        )

    def _score_first_verification_block(self, context: Dict[str, Any]) -> str:
        card = str(context.get("verification_card") or "").strip()
        if not card:
            card = _SCORE_FIRST_DOMAIN_FINAL_CHECKS["advanced_math"]
        return (
            "Internal final check:\n"
            + card
            + "\nDo this silently; correct any failure before answering.\n"
              "Do not print the check unless the problem asks for reasoning.\n\n"
        )

    def _score_first_micro_strategy(
        self,
        domain: str,
        problem: str,
        *,
        target_text: str = "",
        context_text: str = "",
        has_request_spans: bool = False,
    ) -> Dict[str, Any] | None:
        full_text = str(problem or "")
        target = str(target_text or "").strip()
        context = str(context_text or "").strip()

        # If request extraction failed, preserve a conservative whole-problem
        # fallback. If it succeeded, target-side evidence is normally mandatory.
        if not has_request_spans:
            target = full_text
            context = ""

        scored: List[Dict[str, Any]] = []
        context_only: List[Dict[str, Any]] = []
        for name, strong_patterns, weak_patterns, card in _SCORE_FIRST_MICRO_STRATEGIES.get(domain, ()):
            target_strong = self._score_first_evidence_group_hit(strong_patterns, target)
            target_weak = (
                False
                if target_strong
                else self._score_first_evidence_group_hit(weak_patterns, target)
            )
            context_strong = self._score_first_evidence_group_hit(strong_patterns, context)
            context_weak = (
                False
                if context_strong
                else self._score_first_evidence_group_hit(weak_patterns, context)
            )

            if has_request_spans and not (target_strong or target_weak):
                if context_strong:
                    context_only.append(
                        {
                            "name": name,
                            "score": _SCORE_FIRST_MICRO_CONTEXT_STRONG_SCORE,
                            "target_strong": False,
                            "target_weak": False,
                            "context_strong": True,
                            "context_weak": context_weak,
                            "card": card,
                        }
                    )
                continue

            if has_request_spans:
                score = (
                    (_SCORE_FIRST_MICRO_TARGET_STRONG_SCORE if target_strong else 0)
                    + (_SCORE_FIRST_MICRO_TARGET_WEAK_SCORE if target_weak else 0)
                    + (_SCORE_FIRST_MICRO_CONTEXT_STRONG_SCORE if context_strong else 0)
                    + (_SCORE_FIRST_MICRO_CONTEXT_WEAK_SCORE if context_weak else 0)
                )
            else:
                score = (
                    (_SCORE_FIRST_MICRO_STRONG_SCORE if target_strong else 0)
                    + (_SCORE_FIRST_MICRO_WEAK_SCORE if target_weak else 0)
                )

            if score:
                scored.append(
                    {
                        "name": name,
                        "score": int(score),
                        "target_strong": target_strong,
                        "target_weak": target_weak,
                        "context_strong": context_strong,
                        "context_weak": context_weak,
                        "card": card,
                    }
                )

        # Narrow anaphoric bridge: if the requested target explicitly points back to
        # prior context and that context supports exactly one strong method, allow
        # that one method. This does not reactivate general background-only scoring.
        if (
            has_request_spans
            and not scored
            and len(context_only) == 1
            and self._score_first_target_links_to_context(target)
        ):
            top = context_only[0]
            return {
                **top,
                "score": _SCORE_FIRST_MICRO_MIN_SCORE,
                "target_link": True,
                "second_score": 0,
                "margin": _SCORE_FIRST_MICRO_MIN_SCORE,
            }

        if not scored:
            return None

        scored.sort(key=lambda item: (-int(item["score"]), str(item["name"])))
        top = scored[0]
        second_score = int(scored[1]["score"]) if len(scored) > 1 else 0
        margin = int(top["score"]) - second_score
        minimum_score = (
            _SCORE_FIRST_MICRO_TARGET_WEAK_SCORE
            if has_request_spans
            else _SCORE_FIRST_MICRO_MIN_SCORE
        )
        if int(top["score"]) < minimum_score:
            return None
        if margin < _SCORE_FIRST_MICRO_MIN_MARGIN:
            return None
        return {
            **top,
            "target_link": False,
            "second_score": second_score,
            "margin": margin,
        }

    def _score_first_target_links_to_context(self, target_text: str) -> bool:
        target = str(target_text or "")
        if not target:
            return False
        return bool(
            re.search(
                r"""
                \b(?:it|this|that|its|theorem|result)\b
                |\bwhich\s+theorem\b
                |\bx_?\{?\d+\}?\b
                |\bprobability\b
                |(?:哪个定理|概率|精确值)
                """,
                target,
                flags=re.IGNORECASE | re.VERBOSE,
            )
        )

    def _score_first_evidence_group_hit(
        self,
        patterns: tuple[str, ...],
        text: str,
    ) -> bool:
        value = str(text or "")
        if not value or not patterns:
            return False
        # All regexes inside the tuple are synonyms/variants of one evidence group.
        # Matching several variants therefore contributes once, preventing duplicate
        # regex synonyms from manufacturing artificial score margins.
        return any(
            re.search(pattern, value, flags=re.IGNORECASE)
            for pattern in patterns
        )

    def _build_direct_prompt(self, problem: str, metadata: Dict[str, Any]) -> List[Dict[str, str]]:
        subject = self._subject_hint(metadata)
        return [
            {
                "role": "system",
                "content": (
                    "You are a rigorous math problem solver. Solve the problem and return a concise, "
                    "judgeable final answer. For calculation, output only the final value or expression. "
                    "For proof, output a concise complete proof. Do not use any provided reference answer."
                ),
            },
            {
                "role": "user",
                "content": f"Subject hint: {subject}\nProblem:\n{problem}\n\nGive the final answer.",
            },
        ]

    def _subject_hint(self, metadata: Dict[str, Any]) -> str:
        value = metadata.get("subject") or metadata.get("type") or metadata.get("category") or ""
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        return text[:120]

    def _extract_score_first_response(
        self,
        raw_output: str,
        problem: str,
        response_mode: str = _SCORE_FIRST_RESPONSE_MODE_ANSWER,
    ) -> str:
        text = str(raw_output or "").strip()
        if not text:
            return DEFAULT_FALLBACK

        # Derivations, proofs, constructions, and counterexamples are complete
        # judgeable responses. Do not collapse them through answer-value parsing.
        if response_mode != _SCORE_FIRST_RESPONSE_MODE_ANSWER:
            return text

        lines = [line.strip() for line in text.splitlines() if line.strip()]

        # The answer-first contract makes this branch robust even if the transport
        # truncates a later explanation: the complete first-line answer has already
        # arrived and does not depend on the remainder of the response.
        if lines:
            first_answer = self._parse_score_first_answer_wrapper(lines[0])
            if first_answer is not None:
                return self._normalize_score_first_answer_payload(first_answer)

        # Closed-world fallback only: accept a complete supported wrapper on the
        # whole response or on an individual line. This fixes English/Chinese
        # copula forms without arbitrary substring searches.
        whole_answer = self._parse_score_first_answer_wrapper(text)
        if whole_answer is not None:
            return self._normalize_score_first_answer_payload(whole_answer)

        for line in reversed(lines):
            wrapped = self._parse_score_first_answer_wrapper(line)
            if wrapped is not None:
                return self._normalize_score_first_answer_payload(wrapped)

        # If the model emits a bare one-line answer, preserve it without routing
        # through the legacy 500-character cap. For an unwrapped multiline
        # answer-value response, retain the final nonempty line as a conservative
        # compatibility fallback; proof tasks never enter this branch.
        if len(lines) == 1:
            return self._normalize_score_first_answer_payload(lines[0])
        if lines:
            return self._normalize_score_first_answer_payload(lines[-1])
        return DEFAULT_FALLBACK

    def _parse_score_first_answer_wrapper(self, text: str) -> str | None:
        value = str(text or "").strip()
        if not value:
            return None
        for pattern in _SCORE_FIRST_WRAPPER_PATTERNS:
            match = pattern.fullmatch(value)
            if match:
                payload = str(match.group("payload") or "").strip()
                return payload or None
        return None

    def _normalize_score_first_answer_payload(self, payload: str) -> str:
        value = str(payload or "").strip()
        if not value:
            return DEFAULT_FALLBACK

        # Only remove complete, known presentation wrappers. Mathematical grouping
        # such as {1,2,3}, [1,2,3], tuples, intervals, and LaTeX braces is untouched.
        # Strip one clearly external terminal punctuation mark first so surfaces like
        # "$42$." and "\\boxed{42}。" can expose their actual outer wrapper.
        value = self._strip_score_first_terminal_punctuation(value)
        changed = True
        while changed and value:
            changed = False
            stripped = self._strip_score_first_presentation_wrapper(value)
            if stripped != value:
                value = stripped.strip()
                value = self._strip_score_first_terminal_punctuation(value)
                changed = True

        return value.strip() or DEFAULT_FALLBACK

    def _strip_score_first_presentation_wrapper(self, text: str) -> str:
        value = str(text or "").strip()

        if len(value) >= 4 and value.startswith("**") and value.endswith("**"):
            return value[2:-2].strip()
        if len(value) >= 4 and value.startswith("__") and value.endswith("__"):
            return value[2:-2].strip()
        if len(value) >= 2 and value.startswith("$") and value.endswith("$"):
            return value[1:-1].strip()
        if len(value) >= 4 and value.startswith(r"\(") and value.endswith(r"\)"):
            return value[2:-2].strip()
        if len(value) >= 4 and value.startswith(r"\[") and value.endswith(r"\]"):
            return value[2:-2].strip()

        boxed_prefix = r"\boxed{"
        if value.startswith(boxed_prefix):
            inner = self._extract_complete_braced_wrapper(value, boxed_prefix)
            if inner is not None:
                return inner
        return value

    def _extract_complete_braced_wrapper(self, text: str, prefix: str) -> str | None:
        if not text.startswith(prefix):
            return None
        depth = 1
        chars: List[str] = []
        index = len(prefix)
        while index < len(text):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    if text[index + 1 :].strip():
                        return None
                    return "".join(chars).strip()
            chars.append(char)
            index += 1
        return None

    def _strip_score_first_terminal_punctuation(self, text: str) -> str:
        value = str(text or "").rstrip()
        if not value:
            return value

        # One terminal presentation punctuation mark is safe to remove. A decimal
        # point inside a number is unaffected because it is not terminal.
        if value.endswith(("。", ";", "；")):
            return value[:-1].rstrip()
        if value.endswith(".") and not value.endswith("..."):
            return value[:-1].rstrip()
        return value

    def _is_proof_task(self, problem: str) -> bool:
        return self._score_first_explicit_response_intent(problem) in {
            _SCORE_FIRST_RESPONSE_MODE_PROOF,
            _SCORE_FIRST_RESPONSE_MODE_PROOF_OR_DISPROOF,
        }

    def _normalize_model_response(self, response: Any) -> str:
        if response is None:
            return ""
        if isinstance(response, str):
            return response.strip()
        if isinstance(response, dict):
            for key in ("final_response", "content", "text", "answer"):
                value = response.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return str(response).strip()

    def _extract_final_response(self, result: Any, problem: str) -> str:
        if isinstance(result, dict):
            value = result.get("final_response")
            if isinstance(value, str) and value.strip():
                normalized = normalize_final_response(value, problem=problem)
                return self._repair_missing_requested_value(normalized, result, problem)
            final_answer = result.get("final_answer")
            if isinstance(final_answer, dict):
                answer = final_answer.get("answer")
                if isinstance(answer, str) and answer.strip():
                    normalized = normalize_final_response(answer, problem=problem)
                    return self._repair_missing_requested_value(normalized, result, problem)
            if isinstance(final_answer, str) and final_answer.strip():
                normalized = normalize_final_response(final_answer, problem=problem)
                return self._repair_missing_requested_value(normalized, result, problem)
            solution = result.get("solution")
            if isinstance(solution, list) and solution:
                for item in reversed(solution):
                    content = item.get("content") if isinstance(item, dict) else item
                    if isinstance(content, str) and content.strip():
                        return extract_final_answer(content, problem=problem)
        return DEFAULT_FALLBACK

    def _is_acceptable_orchestrator_result(self, result: Dict[str, Any]) -> bool:
        meta = result.get("_meta") if isinstance(result.get("_meta"), dict) else {}
        status = meta.get("overall_status")
        if status != "solved":
            return False
        if not bool(meta.get("content_complete")):
            return False
        if status == "solved" and not (meta.get("answer_verified") or meta.get("proof_verified")):
            return False
        return True

    def _extract_last_raw_output(self) -> str:
        log = getattr(self.orchestrator, "last_log", None)
        if isinstance(log, dict):
            raw = log.get("solver_raw_output")
            if isinstance(raw, str) and raw.strip():
                return raw
        return ""

    def _repair_missing_requested_value(self, final_response: str, result: Any, problem: str) -> str:
        problem_text = str(problem or "").lower()
        final_text = str(final_response or "").strip()
        if not final_text or not isinstance(result, dict):
            return final_response

        asks_gaussian_curvature = any(marker in problem_text for marker in ("高斯曲率", "gaussian curvature"))
        final_has_curvature_value = bool(
            re.search(r"\bK\s*=", final_text)
            or re.search(r"(?:curvature|曲率)[^0-9+\-]*[+\-]?\d+(?:\.\d+)?", final_text, flags=re.IGNORECASE)
        )
        if asks_gaussian_curvature and not final_has_curvature_value:
            evidence = self._collect_result_text(result)
            match = re.search(r"\bK\s*=[^.;。；]*?=\s*([+-]?\d+(?:\.\d+)?)", evidence)
            if match is None:
                match = re.search(r"\bK\s*=\s*([+-]?\d+(?:\.\d+)?)", evidence)
            if match:
                value = match.group(1).rstrip(".;,，。")
                return normalize_final_response(f"K = {value}. {final_text}", problem=problem)
        return final_response

    def _collect_result_text(self, value: Any) -> str:
        if isinstance(value, dict):
            return " ".join(self._collect_result_text(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return " ".join(self._collect_result_text(item) for item in value)
        if isinstance(value, str):
            return value
        return ""

    def _fallback_result(self, reason: str) -> Dict[str, Any]:
        return self._json_safe_result(DEFAULT_FALLBACK, [make_trace_step("error", reason)])

    def _score_first_json_result(self, final_response: str, trace: Any) -> Dict[str, Any]:
        # ScoreFirst extraction has already normalized answer-value surfaces, while
        # proof responses must remain complete. Do not reapply the legacy
        # normalize_final_response() 500/3000-character caps here.
        final_text = str(final_response or "").strip()
        return {
            "final_response": final_text or DEFAULT_FALLBACK,
            "trace": sanitize_trace(trace),
        }

    def _json_safe_result(self, final_response: str, trace: Any) -> Dict[str, Any]:
        final_text = normalize_final_response(final_response)
        return {
            "final_response": final_text or DEFAULT_FALLBACK,
            "trace": sanitize_trace(trace),
        }

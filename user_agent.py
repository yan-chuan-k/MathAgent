from __future__ import annotations

import ast
import inspect
import re
from typing import Any, Dict, List

from math_agent_core.answer_utils import DEFAULT_FALLBACK, extract_final_answer, normalize_final_response
from math_agent_core.output_guard import looks_truncated, response_content
from math_agent_core.router import classify_problem
from math_agent_core.trace_utils import make_trace_step, sanitize_trace, trace_from_orchestrator_result


_SCORE_FIRST_ANSWER_SYSTEM_PROMPT = """You are a high-accuracy mathematics competition solver.
Output exactly ONE visible line:
Final answer: <complete requested answer>
Then stop.
Do not output JSON. Do not repeat the problem.
Do not provide visible explanation or derivation.
Preserve all requested roots, conditions, intervals, moduli, matrices, vectors, sets, and answer parts.
The subject/strategy hint is advisory.
""".strip()

_SCORE_FIRST_PROOF_SYSTEM_PROMPT = """You are a high-accuracy mathematics competition solver.
State the conclusion first, then give a concise but complete proof.
No JSON or problem restatement. Do not omit necessary logical steps.
The subject/strategy hint is advisory.
""".strip()

_SCORE_FIRST_MINIMAL_ANSWER_SYSTEM_PROMPT = """You are a high-accuracy mathematics solver.
Solve the problem carefully.
Output exactly ONE visible line:
Final answer: <complete requested answer>
Then stop.
Answer every requested part.
Preserve exact conditions, domains, units, moduli, and multiplicities.
Use an exact form unless an approximation is requested.
Do not output JSON. Do not repeat the problem.
""".strip()

_SCORE_FIRST_MINIMAL_DERIVATION_SYSTEM_PROMPT = """You are a high-accuracy mathematics solver.
Give the final result first, then a concise derivation containing the mathematical steps explicitly requested.
Answer every requested part.
Do not output JSON or repeat the problem.
""".strip()

_SCORE_FIRST_MINIMAL_PROOF_SYSTEM_PROMPT = """You are a high-accuracy mathematics solver.
State the conclusion first, then give a concise complete proof.
Use all necessary hypotheses and prove exactly the requested claim.
Do not output JSON or repeat the problem.
""".strip()

_SCORE_FIRST_MINIMAL_PROOF_OR_DISPROOF_SYSTEM_PROMPT = """You are a high-accuracy mathematics solver.
Determine whether the claim is true or false.
If true, give a concise proof.
If false, give a counterexample or disproof and verify it.
Do not assume the statement is true.
Do not output JSON or repeat the problem.
""".strip()

_SCORE_FIRST_MINIMAL_CONSTRUCTION_SYSTEM_PROMPT = """You are a high-accuracy mathematics solver.
State the requested object or counterexample first, then give only the verification needed to show it satisfies the requested properties.
Do not output JSON or repeat the problem.
""".strip()

_SCORE_FIRST_EXPERIMENT_PRESETS = {
    "v29_minimal": {
        "prompt_profile": "minimal",
        "thinking_mode": True,
    },
    "minimal_thinking_off": {
        "prompt_profile": "minimal",
        "thinking_mode": False,
    },
    "ultra_minimal": {
        "prompt_profile": "ultra_minimal",
        "thinking_mode": True,
    },
    "ultra_minimal_thinking_off": {
        "prompt_profile": "ultra_minimal",
        "thinking_mode": False,
    },
    "full_thinking_on": {
        "prompt_profile": "full",
        "thinking_mode": True,
    },
    "full_thinking_off": {
        "prompt_profile": "full",
        "thinking_mode": False,
    },
    "tool_augmented_v30": {
        "prompt_profile": "minimal",
        "thinking_mode": True,
        "production_mode": "tool_augmented",
        "extended_tools": False,
    },
    "tool_augmented_v31_extended": {
        "prompt_profile": "minimal",
        "thinking_mode": True,
        "production_mode": "tool_augmented",
        "extended_tools": True,
    },
    "tool_augmented_v305_full_final": {
        "prompt_profile": "full",
        "thinking_mode": True,
        "production_mode": "tool_augmented",
        "extended_tools": True,
    },
    "tool_augmented_v306_cross_subject": {
        "prompt_profile": "full",
        "thinking_mode": True,
        "production_mode": "tool_augmented",
        "extended_tools": True,
        "challenge_reasoning": True,
        "grounded_auto_tools": True,
    },
    "tool_augmented_v307_cross_subject": {
        "prompt_profile": "full",
        "thinking_mode": True,
        "production_mode": "tool_augmented",
        "extended_tools": True,
        "challenge_reasoning": True,
        "grounded_auto_tools": True,
        # The previous 32768-token ceiling was a known truncation pressure
        # point on proof-heavy and multi-part competition items.  Keep older
        # arms frozen and give only this controlled experiment the larger
        # completion budget.
        "max_tokens": 49152,
        "v307_tools": True,
    },
}

_SCORE_FIRST_DEFAULT_EXPERIMENT_PRESET = "tool_augmented_v307_cross_subject"

_TOOL_AUGMENTED_V305_FULL_FINAL_PRESET = "tool_augmented_v305_full_final"
_TOOL_AUGMENTED_V306_CROSS_SUBJECT_PRESET = "tool_augmented_v306_cross_subject"
_TOOL_AUGMENTED_V307_CROSS_SUBJECT_PRESET = "tool_augmented_v307_cross_subject"

_TOOL_AUGMENTED_FORMALIZER_MAX_TOKENS = 1536
_TOOL_AUGMENTED_FORMALIZER_TEMPERATURE = 0.2
_TOOL_AUGMENTED_FORMALIZER_TOP_P = 0.95

_TOOL_AUGMENTED_V305_EXTRACTOR_SYSTEM_PROMPT = """You are an exact deterministic tool extractor, not a solver.
For ANSWER_VALUE and DERIVATION tasks, extract only exact deterministic computations that may help solve the ORIGINAL problem.
Output TOOL lines only when the mathematical input is explicit in the problem.
If no safe deterministic computation is useful, output exactly:
NO_TOOL
Do not solve the problem. Do not recommend a method. Do not write a proof.
Do not output TARGET, METHOD, FACT, CHECK, CLAIM, STATUS, HYPOTHESES, SUBGOAL, THEOREM, FAILURE_TRAP, or COUNTEREXAMPLE lines.

Inside TOOL arguments:
- use ASCII identifiers when possible
- use lam for λ/lambda; use theta, mu, sigma, alpha, beta, gamma, rho, phi, omega for Greek parameters
- use Abs(x), never |x|
- use * for multiplication; use ^ or ** for powers
- use oo for infinity

Safe core TOOL syntax:
TOOL: sympy|simplify|expr=...
TOOL: sympy|solve|equation=...|var=x|domain=real
TOOL: sympy|differentiate|expr=...|var=x
TOOL: sympy|integrate|expr=...|var=x|lower=...|upper=...
TOOL: sympy|limit|expr=...|var=x|point=...
TOOL: sympy|series|expr=...|var=x|point=0|order=6
TOOL: sympy|roots|expr=...|var=x
TOOL: sympy|residue|expr=...|var=z|point=...
TOOL: sympy|residual|left=...|right=...
TOOL: sympy|mod|expr=...|modulus=...
TOOL: sympy|congruence|left=...|right=...|modulus=...
TOOL: sympy|sum|expr=...|var=k|lower=...|upper=...
TOOL: sympy|solve_system|equations=["...","..."]|variables=["x","y"]
TOOL: sympy|stationary|expr=...|variables=x,y
TOOL: sympy|ode_residual|candidate=...|rhs=...|var=x|state=y
TOOL: matrix|determinant|matrix=[[...],[...]]
TOOL: matrix|rank|matrix=[[...],[...]]
TOOL: matrix|inverse|matrix=[[...],[...]]
TOOL: matrix|eigenvalues|matrix=[[...],[...]]
TOOL: matrix|linear_solve|matrix=[[...],[...]]|rhs=[...]
Do not request Python, shell, filesystem, network, arbitrary code execution, or unlisted tool families.
""".strip()

_TOOL_AUGMENTED_FORMALIZER_SYSTEM_PROMPT = """You are a compact mathematical formalizer, not the final solver.
Return a short line-oriented protocol only. Do not write a full proof or long derivation.
Use only these optional prefixes: TARGET, METHOD, FACT, CHECK, CLAIM, STATUS, HYPOTHESES, SUBGOAL, THEOREM, FAILURE_TRAP, COUNTEREXAMPLE, TOOL.
For proof/disproof tasks, give only truth status if confident, critical hypotheses, 2-5 subgoals, a theorem/lemma candidate, and a counterexample shape if relevant.
For explicit computations, request deterministic tools only when the expression is precise enough. Never invent a tool call.

Inside TOOL arguments:
- use ASCII identifiers when possible
- use lam for λ/lambda; use theta, mu, sigma, alpha, beta, gamma, rho, phi, omega for Greek parameters
- use Abs(x), never |x|
- use * for multiplication; use ^ or ** for powers
- use oo for infinity

Safe core TOOL syntax:
TOOL: sympy|simplify|expr=...
TOOL: sympy|solve|equation=...|var=x|domain=real
TOOL: sympy|differentiate|expr=...|var=x
TOOL: sympy|integrate|expr=...|var=x|lower=...|upper=...
TOOL: sympy|limit|expr=...|var=x|point=...
TOOL: sympy|series|expr=...|var=x|point=0|order=6
TOOL: sympy|roots|expr=...|var=x
TOOL: sympy|residue|expr=...|var=z|point=...
TOOL: sympy|residual|left=...|right=...
TOOL: sympy|mod|expr=...|modulus=...
TOOL: sympy|congruence|left=...|right=...|modulus=...
TOOL: sympy|sum|expr=...|var=k|lower=...|upper=...
TOOL: sympy|solve_system|equations=["...","..."]|variables=["x","y"]
TOOL: sympy|stationary|expr=...|variables=x,y
TOOL: sympy|ode_residual|candidate=...|rhs=...|var=x|state=y
TOOL: matrix|determinant|matrix=[[...],[...]]
TOOL: matrix|rank|matrix=[[...],[...]]
TOOL: matrix|inverse|matrix=[[...],[...]]
TOOL: matrix|eigenvalues|matrix=[[...],[...]]
TOOL: matrix|linear_solve|matrix=[[...],[...]]|rhs=[...]
Do not request Python, shell, filesystem, network, arbitrary code execution, or unlisted tool families.
Keep the protocol compact and finish after the useful structure is captured.
""".strip()

_TOOL_AUGMENTED_EXTENDED_TOOL_SYNTAX = """Extended deterministic TOOL syntax enabled for this preset:
TOOL: number_theory|gcd|a=...|b=...
TOOL: number_theory|extended_gcd|a=...|b=...
TOOL: number_theory|mod_inverse|a=...|modulus=...
TOOL: number_theory|pow_mod|base=...|exponent=...|modulus=...
TOOL: number_theory|crt|residues=[...]|moduli=[...]
TOOL: number_theory|linear_congruence|a=...|b=...|modulus=...
TOOL: number_theory|unit_square_solution_count|modulus=...
TOOL: number_theory|primitive_root|modulus=...
TOOL: number_theory|count_divisible_union|upper=...|divisors=[...]
TOOL: number_theory|totient|n=...
TOOL: number_theory|factorint|n=...
TOOL: combinatorics|binomial|n=...|k=...
TOOL: combinatorics|multinomial|parts=[...]
TOOL: combinatorics|derangement|n=...
TOOL: combinatorics|stirling2|n=...|k=...
TOOL: combinatorics|catalan|n=...
TOOL: combinatorics|stars_bars|total=...|variables=...|minimum_each=0|1
For stars_bars, minimum_each is required: 0 means every ordered variable is
nonnegative; 1 means every ordered variable is positive.
TOOL: combinatorics|onto_functions|domain_size=...|codomain_size=...
TOOL: recurrence|linear_eval|initial=[a0,...,a{k-1}]|coefficients=[c1,...,ck]|constant=...|target_n=...
Here initial=[a0,...,a{k-1}] and coefficients=[c1,...,ck] mean
a_n=c1*a_{n-1}+c2*a_{n-2}+...+ck*a_{n-k}+constant.
TOOL: sympy|coefficient|expr=...|var=x|power=...
TOOL: graph|edge_count_from_degrees|degrees=[...]
TOOL: graph|tree_edge_count|vertices=...
TOOL: graph|complete_graph_edges|n=...
TOOL: graph|complete_bipartite_edges|m=...|n=...
TOOL: graph|cycle_space_dimension|vertices=...|edges=...|components=...
TOOL: graph|eulerian_degree_check|degrees=[...]|connected=true
TOOL: graph|degree_sequence_graphical|degrees=[...]
TOOL: numerical|newton_step|f=...|variable=x|x0=...
TOOL: numerical|interpolate_eval|points=[[x0,y0],...]|evaluate_at=...
TOOL: matrix|nullspace|matrix=[[...],[...]]
TOOL: matrix|charpoly|matrix=[[...],[...]]|var=lam
TOOL: matrix|eigenvectors|matrix=[[...],[...]]
TOOL: matrix|rref|matrix=[[...],[...]]
""".strip()

_TOOL_AUGMENTED_V306_CROSS_SUBJECT_TOOL_SYNTAX = """Additional cross-subject deterministic TOOL syntax for this preset:
TOOL: number_theory|multiplicative_order|a=...|modulus=...
TOOL: number_theory|primitive_root_check|a=...|modulus=...
TOOL: combinatorics|partition_exact_parts|total=...|parts=...
TOOL: probability|binomial_pmf|trials=...|successes=...|p=...
TOOL: probability|poisson_pmf|rate=...|k=...
TOOL: probability|hypergeometric_pmf|population=...|success_states=...|draws=...|successes=...
TOOL: matrix|trace|matrix=[[...],[...]]
TOOL: matrix|power|matrix=[[...],[...]]|exponent=...
TOOL: sympy|factor|expr=...
TOOL: sympy|expand|expr=...
TOOL: sympy|numeric|expr=...|digits=16
Only request an operation when every argument is explicitly present. These tools compute a subfact; they do not replace solving the original problem.
""".strip()

_TOOL_AUGMENTED_V307_CROSS_SUBJECT_TOOL_SYNTAX = """Additional V3.0.7 high-frequency exact TOOL syntax:
TOOL: number_theory|mod_inverse|a=...|modulus=...
TOOL: number_theory|pow_mod|base=...|exponent=...|modulus=...
TOOL: number_theory|crt|residues=[...]|moduli=[...]
TOOL: recurrence|linear_eval|initial=[...]|coefficients=[...]|constant=...|target_n=...
TOOL: combinatorics|partition_total|n=...
TOOL: combinatorics|distinct_partition|n=...
TOOL: combinatorics|composition_positive|total=...|parts=...
TOOL: combinatorics|bounded_compositions|total=...|parts=...|upper=...|lower=...
TOOL: combinatorics|one_coordinate_lower|total=...|variables=...|lower=...
TOOL: combinatorics|one_coordinate_upper|total=...|variables=...|upper=...
TOOL: combinatorics|binary_no_adjacent|length=...|ones=...
TOOL: combinatorics|circular_adjacent_block|people=...
TOOL: combinatorics|choose_excluding_pair|total=...|choose=...
TOOL: combinatorics|multiset_permutations|counts=[...]
TOOL: combinatorics|triangulations|vertices=...
TOOL: combinatorics|full_parenthesizations|factors=...
TOOL: combinatorics|ballot_diagonal_paths|steps=...
TOOL: combinatorics|tilings_parts|total=...|pieces=[...]
TOOL: combinatorics|steps_no_consecutive_two|total=...
TOOL: graph|cycle_chromatic|n=...
TOOL: graph|complete_graph_chromatic|n=...
TOOL: graph|complete_bipartite_chromatic|m=...|n=...
TOOL: graph|path_matching|vertices=...
TOOL: graph|cycle_matching|vertices=...
TOOL: graph|planar_faces|vertices=...|edges=...|components=...
TOOL: graph|turan_edges|vertices=...|forbidden_clique=...
TOOL: graph|tree_remaining_degree|vertices=...|leaves=...|degree_two=...
TOOL: graph|connected_edge_guarantee|vertices=...
TOOL: graph|euler_trail_possible|connected=true|odd_vertices=...
TOOL: graph|euler_circuit_possible|connected=true|odd_vertices=...
TOOL: graph|complete_bipartite_hamiltonian|m=...|n=...
TOOL: graph|hall_matching_size|left_vertices=...
TOOL: graph|triangulated_planar_edges|vertices=...
TOOL: graph|tree_cut_edges|components=...
TOOL: matrix|inverse|matrix=[[...],[...]]
TOOL: matrix|eigenvalues|matrix=[[...],[...]]
TOOL: matrix|rref|matrix=[[...],[...]]
TOOL: matrix|nullspace|matrix=[[...],[...]]
TOOL: matrix|linear_solve|matrix=[[...],[...]]|rhs=[...]
TOOL: sympy|differentiate|expr=...|var=x
TOOL: sympy|integrate|expr=...|var=x|lower=...|upper=...
TOOL: sympy|limit|expr=...|var=x|point=...
TOOL: sympy|coefficient|expr=...|var=x|power=...
TOOL: combinatorics|factorial|n=...
TOOL: combinatorics|fibonacci|n=...
TOOL: number_theory|is_prime|n=...
TOOL: number_theory|divisor_count|n=...
TOOL: number_theory|fibonacci_mod|n=...|modulus=...
TOOL: probability|geometric_pmf|success=...|trials=...
TOOL: probability|negative_binomial_pmf|successes=...|trials=...
TOOL: probability|binomial_cdf|trials=...|p=...|upper=...
Only request these when the exact expression, matrix, recurrence data, graph parameters, or distribution parameters are explicitly present in the original problem. They are subfacts, not a substitute for the full solution.
""".strip()

_TOOL_AUGMENTED_V306_DOMAIN_TOOL_GUIDANCE = {
    "discrete_math": (
        "Priority only when exact inputs are present: number_theory, combinatorics, recurrence, graph, "
        "or sympy coefficient. Preserve every stated combinatorial restriction."
    ),
    "probability": (
        "Priority only for an explicitly parameterized finite distribution: probability pmf tools or exact SymPy "
        "algebra. Do not infer a distribution or independence that was not stated."
    ),
    "statistics": (
        "Priority only for explicitly parameterized distribution subcomputations. Do not use a tool to infer an "
        "estimator, likelihood, or asymptotic theorem."
    ),
    "stochastic_process": (
        "Priority only for an explicitly displayed finite-state transition calculation or exact distribution "
        "subexpression. Do not infer Markov, independence, stopping, or martingale assumptions."
    ),
    "linear_regression": (
        "Priority only for a displayed finite matrix/vector or explicit likelihood algebra. Do not use a tool to "
        "choose a model, identify assumptions, or claim an inferential conclusion."
    ),
    "linear_algebra": (
        "Priority only for an explicitly displayed matrix/vector: determinant, rank, rref, eigenstructure, trace, "
        "matrix power, or a linear system."
    ),
    "abstract_algebra": (
        "Priority only for explicit finite modular arithmetic or a displayed polynomial expression. Do not use a "
        "tool to infer group, ring, field, normality, or homomorphism hypotheses."
    ),
    "optimization": (
        "Priority only for an explicit symbolic objective whose derivative, stationary points, or residual can be "
        "computed exactly. Constraints still need mathematical interpretation."
    ),
    "numerical_analysis": (
        "Priority only for an explicitly supplied iteration, polynomial data, or exact formula: numerical step, "
        "interpolation evaluation, or a SymPy residual."
    ),
    "complex_analysis": (
        "Priority only for an explicitly supplied integrand/point/contour-related subexpression: residue, series, "
        "limit, or symbolic simplification."
    ),
    "real_analysis": (
        "Priority only for an explicit symbolic limit, derivative, integral, series, or counterexample candidate. "
        "The validity of a theorem's hypotheses remains a reasoning task."
    ),
    "measure_integration": (
        "Priority only for explicit integrands or finite algebraic subexpressions. Never use a tool as evidence for "
        "measurability, integrability, or an interchange-of-limit theorem."
    ),
    "functional_analysis": (
        "Priority only for displayed finite-dimensional or symbolic subexpressions. Norm, completeness, boundedness, "
        "and operator-theoretic assumptions must be established mathematically."
    ),
    "topology": (
        "Priority only for explicit finite combinatorial or symbolic subexpressions. Separation, compactness, and "
        "continuity claims require their stated hypotheses, not a tool guess."
    ),
    "differential_geometry": (
        "Priority only for an explicitly supplied coordinate expression whose derivative or simplification is exact. "
        "Do not infer geometric conventions or hypotheses."
    ),
    "ode": (
        "Priority only for an explicitly supplied expression or candidate solution: differentiate, integrate, solve "
        "a displayed algebraic subproblem, or evaluate an ODE residual."
    ),
    "pde": (
        "Priority only for explicit algebraic/separation subexpressions. Do not turn boundary-value interpretation "
        "into a tool request."
    ),
}

_TOOL_AUGMENTED_V306_HIGH_DIFFICULTY_DISCIPLINE = (
    "Before finalizing, silently make a target-and-hypotheses ledger, choose a theorem or representation only after "
    "matching its assumptions, and perform one independent boundary, invariant, substitution, or counterexample check. "
    "Resolve every requested part, then commit rather than continuing open-ended exploration."
)

_TOOL_AUGMENTED_V307_HIGH_DIFFICULTY_DISCIPLINE = (
    "Use the full internal reasoning budget as needed, but keep the visible solution compact. "
    "First make a private target-and-hypotheses ledger, choose one governing representation or theorem, "
    "and solve all requested parts. Perform one independent invariant, boundary, substitution, or counterexample check. "
    "Do not print scratch work, abandoned routes, repeated derivations, or meta-commentary; emit only a complete judgeable solution."
)

_TOOL_AUGMENTED_FINAL_SYSTEM_BASE = """You are the final high-accuracy mathematics solver.
Solve the ORIGINAL problem yourself.
The formalizer plan is advisory and may be wrong.
Parent-bound target evidence, when present, is system-derived and authoritative
for the exact requested object; treat unrelated formalizer facts as advisory.
Each deterministic computation is exact only for the input explicitly shown in that fact.
You MUST compare the shown input with the ORIGINAL problem before using the result.
Ignore any fact whose shown input does not match the original task.
Correct any mistake in the plan.
Answer every requested part and preserve all hypotheses, domains, units, moduli, and multiplicities.
Do not output JSON and do not repeat the problem.
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
    "Unless approximating, prefer an exact mathematical form; otherwise honor the requested precision. "
    "Preserve units, domains, moduli, multiplicities."
)

_SCORE_FIRST_DECISIVE_REASONING = (
    "Use a direct path. Restart only for a concrete error. "
    "If the independent check fails, correct once; if it succeeds and all parts are covered, commit. "
    "Do not re-derive a verified solution."
)

_SCORE_FIRST_COMPACT_OUTPUT_DISCIPLINE = (
    "Use the available internal reasoning budget as needed, but emit only a compact complete solution: "
    "no scratch work, repeated checks, abandoned alternatives, or meta-commentary."
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
    "limit_theorem": "Recheck centering/scaling and theorem assumptions; confirm the limiting/reference distribution and any continuity correction used.",
    "poisson_process": "Recheck rate times interval length, Poisson support k>=0, and normalization or mean=variance; for thinning/superposition verify the transformed rate.",
    "brownian_martingale": "Check the defining Brownian or martingale property actually requested; do not import Markov-chain first-step or stationarity reasoning.",
    "zeros_argument": "Verify no zero/pole lies on the contour, track orientation and multiplicity, and check argument-principle count = zeros minus poles.",
    "equilibrium_stability": "Recheck the equilibrium and the actual stability criterion: linearization/eigenvalues, phase-line sign, or Lyapunov argument; solving the ODE alone is insufficient.",
    "bias_variance_sufficiency": "Check only the estimator property requested from the sampling model or the sufficiency criterion actually invoked.",
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
    "fixed_point_convergence": (
        "Verify x*=g(x*) and the contraction/local derivative condition, e.g. |g'(x*)|<1 near x*; "
        "use only the fixed-point hypotheses actually stated."
    ),
    "brownian_moment": (
        "For Brownian motion, recheck E[B_t]=0, Var(B_t)=t and Cov(B_s,B_t)=min(s,t) when relevant."
    ),
    "martingale": (
        "For a martingale claim, check integrability and the conditional-expectation or independent-increment step."
    ),
    "bias_variance": (
        "Independently recompute the requested E[T] and/or Var(T) under the stated sampling model; compare E[T] "
        "with the target parameter when bias/unbiasedness is requested."
    ),
    "sufficiency": (
        "Verify the factorization or conditional-distribution criterion actually used establishes sufficiency."
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
       give|provide|prove|show|establish|demonstrate|derive|deduce|explain|justify|verify|
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

_SCORE_FIRST_SMALL_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
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

        preset_name = str(
            kwargs.get("score_first_experiment_preset", _SCORE_FIRST_DEFAULT_EXPERIMENT_PRESET)
            or _SCORE_FIRST_DEFAULT_EXPERIMENT_PRESET
        ).strip().lower()
        if preset_name not in _SCORE_FIRST_EXPERIMENT_PRESETS:
            raise ValueError(
                "score_first_experiment_preset must be one of: "
                + ", ".join(sorted(_SCORE_FIRST_EXPERIMENT_PRESETS))
            )
        self.score_first_experiment_preset = preset_name
        preset = _SCORE_FIRST_EXPERIMENT_PRESETS[preset_name]

        preset_production_mode = str(preset.get("production_mode") or "score_first")
        self.production_mode = str(
            kwargs.get("production_mode", preset_production_mode)
            or preset_production_mode
        ).strip().lower()
        if self.production_mode not in {"score_first", "orchestrated", "tool_augmented"}:
            raise ValueError(
                "production_mode must be 'score_first', 'orchestrated', or 'tool_augmented'"
            )

        explicit_prompt_profile = kwargs.get("score_first_prompt_profile")
        if explicit_prompt_profile:
            prompt_profile = str(explicit_prompt_profile).strip().lower()
        else:
            prompt_profile = str(preset["prompt_profile"]).strip().lower()
        self.score_first_prompt_profile = prompt_profile
        if self.score_first_prompt_profile not in {"minimal", "ultra_minimal", "full"}:
            raise ValueError(
                "score_first_prompt_profile must be 'minimal', 'ultra_minimal', or 'full'"
            )

        self.max_retries = int(kwargs.get("max_retries", 1))
        is_modern_solver = self.production_mode in {"score_first", "tool_augmented"}
        default_temperature = 0.8 if is_modern_solver else 0.2
        # An explicit production_mode is the legacy constructor surface.  If
        # no preset was selected alongside it, preserve that surface's
        # 32768-token default instead of unexpectedly inheriting the release
        # preset's experimental 49152-token budget.  The normal constructor
        # path (no explicit mode) still selects the V3.0.7 release budget.
        explicit_preset = kwargs.get("score_first_experiment_preset") is not None
        explicit_mode_without_preset = (
            not explicit_preset and "production_mode" in kwargs
        )
        if is_modern_solver and explicit_mode_without_preset:
            default_max_tokens = 32768
        else:
            default_max_tokens = int(preset.get("max_tokens", 32768)) if is_modern_solver else 4096
        default_top_p = 0.95 if is_modern_solver else None
        self.temperature = float(kwargs.get("temperature", default_temperature))
        self.max_tokens = int(kwargs.get("max_tokens", default_max_tokens))
        top_p_value = kwargs.get("top_p", default_top_p)
        self.top_p = None if top_p_value is None else float(top_p_value)
        if "thinking_mode" in kwargs:
            self.thinking_mode = bool(kwargs["thinking_mode"])
        elif is_modern_solver:
            self.thinking_mode = bool(preset["thinking_mode"])
        else:
            self.thinking_mode = True

        self.tool_augmented_formalizer_temperature = float(
            kwargs.get("tool_augmented_formalizer_temperature", _TOOL_AUGMENTED_FORMALIZER_TEMPERATURE)
        )
        self.tool_augmented_formalizer_top_p = float(
            kwargs.get("tool_augmented_formalizer_top_p", _TOOL_AUGMENTED_FORMALIZER_TOP_P)
        )
        self.tool_augmented_formalizer_max_tokens = int(
            kwargs.get("tool_augmented_formalizer_max_tokens", _TOOL_AUGMENTED_FORMALIZER_MAX_TOKENS)
        )
        self.tool_augmented_verification_timeout_seconds = max(
            0.05,
            float(kwargs.get("tool_augmented_verification_timeout_seconds", 5.0)),
        )
        self.tool_augmented_extended_tools = bool(
            kwargs.get("tool_augmented_extended_tools", preset.get("extended_tools", False))
        )
        # V3.0.6 is deliberately opt-in at the preset level.  Keeping these
        # as independent flags lets the frozen V3.0.5 FULL+tools arm remain a
        # meaningful rollback control even though it shares the same executor.
        self.tool_augmented_challenge_reasoning = bool(
            kwargs.get(
                "tool_augmented_challenge_reasoning",
                preset.get("challenge_reasoning", False),
            )
        )
        self.tool_augmented_grounded_auto_tools = bool(
            kwargs.get(
                "tool_augmented_grounded_auto_tools",
                preset.get("grounded_auto_tools", False),
            )
        )
        self.tool_augmented_v307_tools = bool(
            kwargs.get("tool_augmented_v307_tools", preset.get("v307_tools", False))
        )
        self.max_candidates = int(kwargs.get("max_candidates", 2))
        self.orchestrator = None
        # Only bounded transport metadata is retained here.  It is used to
        # distinguish a provider length stop from a genuinely short answer;
        # raw SDK responses and hidden reasoning are never put in trace.
        self._last_response_metadata: Dict[str, Any] = {}

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

            if self.production_mode == "tool_augmented":
                return self._solve_tool_augmented(problem, safe_metadata)

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

    def _solve_tool_augmented(self, problem: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        from math_agent_core.tools.augmented_tool import (
            SafeAugmentedToolExecutor,
            build_evidence_block_with_metadata,
            parse_formalizer_protocol,
        )

        v305_full_final = self._tool_augmented_is_v305_full_final()
        v306_cross_subject = self._tool_augmented_is_v306_cross_subject()
        v307_cross_subject = self._tool_augmented_is_v307_cross_subject()
        full_final_mode = v305_full_final or v306_cross_subject or v307_cross_subject
        # Both FULL-final arms deliberately begin from the frozen FULL
        # ScoreFirst context.  The legacy V3.0/V3.1 arms remain available
        # unchanged for rollback comparisons.
        context = (
            self._score_first_context(problem, metadata)
            if full_final_mode
            else self._score_first_minimal_context(problem, metadata)
        )
        response_mode = context["response_mode"]
        model_calls = 0
        formalizer_used = False
        tool_extractor_used = False
        protocol = parse_formalizer_protocol("", extended_tools=self.tool_augmented_extended_tools)
        formalizer_error = ""
        formalizer_raw = ""
        protocol_parse_rejected = False
        tool_worker_status = "no_requests"

        target_verification_requests = self._tool_augmented_verification_requests(problem, context)
        target_tool_requests: List[Any] = []
        grounded_auto_tool_requests: List[Any] = []

        # The low-thinking first call is useful only for explicit computations
        # or derivations.  Proof/disproof/construction tasks go directly to the
        # frozen FULL solver and therefore retain a one-call normal path.
        tool_extractor_allowed = (
            not full_final_mode
            or self._tool_augmented_v305_tool_extraction_allowed(response_mode)
        )
        if tool_extractor_allowed:
            tool_extractor_used = full_final_mode
            try:
                formalizer_messages = self._build_tool_augmented_formalizer_prompt(
                    problem,
                    context,
                )
                model_calls += 1
                formalizer_response = self._chat_with_settings(
                    formalizer_messages,
                    temperature=self.tool_augmented_formalizer_temperature,
                    top_p=self.tool_augmented_formalizer_top_p,
                    max_tokens=self.tool_augmented_formalizer_max_tokens,
                    thinking_mode=False,
                )
                formalizer_raw = self._normalize_model_response(formalizer_response)
                protocol = parse_formalizer_protocol(
                    formalizer_raw,
                    extended_tools=self.tool_augmented_extended_tools,
                )
                formalizer_used = True
                protocol_parse_rejected = self._tool_augmented_protocol_parse_rejected(
                    formalizer_raw,
                    protocol,
                )
            except Exception as exc:
                # Call 1 is advisory.  Its failure must have no influence on
                # FULL-final Call 2, which falls back to the frozen FULL
                # ScoreFirst prompt below unless V3.0.6 has separate,
                # parent-grounded evidence or a high-difficulty discipline.
                formalizer_error = type(exc).__name__

        if (v306_cross_subject or v307_cross_subject) and self.tool_augmented_grounded_auto_tools:
            grounded_auto_tool_requests = self._tool_augmented_v306_grounded_auto_tool_requests(
                problem,
                context,
                target_verification_requests,
            )

        if full_final_mode:
            # V3.0.5 intentionally does not synthesize parent tool calls when
            # the extractor emitted NO_TOOL or no valid request.  This is the
            # no-regret invariant for the V3.0.5 rollback arm: no successful
            # extractor fact means no augmentation of the frozen FULL prompt.
            # V3.0.6 may add only independently parent-grounded requests from
            # explicit target spans; it never copies a model plan downstream.
            formalizer_requests = (
                []
                if protocol_parse_rejected
                else self._tool_augmented_requests_for_arm(
                    protocol.tool_requests,
                    allow_v306_only=v306_cross_subject or v307_cross_subject,
                    allow_v307_only=v307_cross_subject,
                )
            )
            merged_tool_requests = self._tool_augmented_merge_tool_requests(
                grounded_auto_tool_requests,
                formalizer_requests,
            )
        else:
            # Preserve the V3.0.4 target-first parent binding behavior in the
            # explicitly selectable rollback arms.
            target_tool_requests = self._tool_augmented_target_tool_requests(
                target_verification_requests
            )
            merged_tool_requests = self._tool_augmented_merge_tool_requests(
                target_tool_requests,
                self._tool_augmented_requests_for_arm(
                    protocol.tool_requests,
                    allow_v306_only=False,
                ),
            )

        tool_facts: List[Any] = []
        if protocol_parse_rejected and not (
            v306_cross_subject and grounded_auto_tool_requests
        ):
            tool_worker_status = "parse_rejected"
        elif merged_tool_requests:
            try:
                executor = SafeAugmentedToolExecutor(
                    extended_tools=self.tool_augmented_extended_tools
                )
                tool_batch = executor.execute_with_status(merged_tool_requests)
                tool_facts = tool_batch.facts
                tool_worker_status = tool_batch.worker_status
            except Exception:
                # A deterministic worker is optional evidence.  Treat an
                # unexpected parent-side worker failure exactly like a batch
                # with no usable facts, so the FULL-final arms still reach
                # their final solver rather than the global fallback.
                tool_facts = []
                tool_worker_status = "worker_error"

        target_tool_facts = [
            fact for fact in tool_facts if getattr(fact, "source", "") == "parent_target"
        ]
        grounded_auto_tool_facts = [
            fact for fact in tool_facts if getattr(fact, "source", "") == "parent_explicit"
        ]
        formalizer_tool_facts = [
            fact
            for fact in tool_facts
            if getattr(fact, "source", "") not in {"parent_target", "parent_explicit"}
        ]
        evidence = build_evidence_block_with_metadata(tool_facts)
        evidence_block = evidence.text
        evidence_is_injectable = bool(evidence.included_facts)
        injected_grounded_auto_tool_facts = [
            fact
            for fact in evidence.included_facts
            if getattr(fact, "source", "") == "parent_explicit"
        ]
        if (v306_cross_subject or v307_cross_subject) and grounded_auto_tool_facts and evidence_is_injectable:
            # A V3.0.6 batch can contain both parent-extracted and
            # formalizer-extracted facts.  The heading must therefore not
            # overstate that every fact came from the low-thinking extractor.
            evidence_block = evidence_block.replace(
                "Deterministically computed facts from the formalizer's exact requests:",
                "Deterministically computed facts from explicit mathematical inputs:",
                1,
            )
        if not full_final_mode and target_tool_requests:
            # Keep the legacy evidence heading for compatibility while making
            # the provenance ordering explicit to the final solver.  The
            # target-bound requests are always the first merged requests and
            # therefore their successful facts appear before advisory facts.
            evidence_block = (
                "Parent-bound target facts are listed first when available; "
                "they are authoritative for the requested object.\n"
                + evidence_block
            )

        if v305_full_final:
            final_messages = self._build_tool_augmented_v305_final_prompt(
                problem=problem,
                context=context,
                evidence_block=evidence.text if evidence_is_injectable else "",
            )
        elif v307_cross_subject:
            final_messages = self._build_tool_augmented_v307_final_prompt(
                problem=problem,
                context=context,
                evidence_block=evidence_block if evidence_is_injectable else "",
            )
        elif v306_cross_subject:
            final_messages = self._build_tool_augmented_v306_final_prompt(
                problem=problem,
                context=context,
                evidence_block=evidence_block if evidence_is_injectable else "",
            )
        else:
            final_messages = self._build_tool_augmented_final_prompt(
                problem=problem,
                context=context,
                plan_text=protocol.compact_plan(),
                evidence_block=evidence_block,
            )
        model_calls += 1
        final_response_obj = self._chat_with_settings(
            final_messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            thinking_mode=self.thinking_mode,
        )
        final_raw = self._normalize_model_response(final_response_obj)
        final_response = self._extract_score_first_response(
            final_raw,
            problem,
            response_mode=response_mode,
        )

        output_truncated = self._output_is_likely_truncated(
            final_raw,
            final_response,
            response_mode,
        )

        failed_checks, correction_targets, verification_failure_methods = self._tool_augmented_decisive_failures(
            problem,
            final_response,
            response_mode,
            context,
            with_metadata=True,
        )
        grounded_fact_failures = self._tool_augmented_v306_grounded_fact_failures(
            final_response,
            response_mode,
            injected_grounded_auto_tool_facts,
            has_answer_contract=self._tool_augmented_has_answer_contract(final_raw),
        )
        if grounded_fact_failures:
            failed_checks = self._tool_augmented_combine_failed_checks(
                grounded_fact_failures,
                failed_checks,
            )
            if "parent_grounded_exact_fact" not in correction_targets:
                correction_targets.append("parent_grounded_exact_fact")
            if "parent_grounded_exact_fact" not in verification_failure_methods:
                verification_failure_methods.insert(0, "parent_grounded_exact_fact")
        correction_used = False
        correction_reverified = False
        correction_kept = False
        recovery_call_used = False
        recovery_kept = False
        correction_error = ""
        if failed_checks or output_truncated:
            current_answer_before_correction = final_response
            if output_truncated and "transport_truncation" not in verification_failure_methods:
                verification_failure_methods.append("transport_truncation")
            if output_truncated and "output_completion" not in correction_targets:
                correction_targets.append("output_completion")
            if failed_checks:
                correction_messages = self._build_tool_augmented_correction_prompt(
                    problem=problem,
                    context=context,
                    current_answer=final_response,
                    failed_checks=failed_checks,
                    evidence_block=evidence_block,
                    truncation_recovery=output_truncated,
                )
            else:
                correction_messages = self._build_tool_augmented_recovery_prompt(
                    problem=problem,
                    context=context,
                    evidence_block=evidence_block,
                )
            model_calls += 1
            correction_used = True
            recovery_call_used = output_truncated
            try:
                corrected_obj = self._chat_with_settings(
                    correction_messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    thinking_mode=self.thinking_mode,
                )
                corrected_raw = self._normalize_model_response(corrected_obj)
                corrected_response = self._extract_score_first_response(
                    corrected_raw,
                    problem,
                    response_mode=response_mode,
                )

                corrected_truncated = self._output_is_likely_truncated(
                    corrected_raw,
                    corrected_response,
                    response_mode,
                )

                # A correction call is allowed to replace Call 2 only when the
                # same bounded deterministic checks no longer find a decisive
                # failure (or find strictly fewer failures).  This prevents a
                # repair response from silently turning one known-bad answer into
                # another answer that is even less grounded.
                corrected_failures, _corrected_targets, _corrected_methods = (
                    self._tool_augmented_decisive_failures(
                        problem,
                        corrected_response,
                        response_mode,
                        context,
                        with_metadata=True,
                    )
                )
                corrected_grounded_fact_failures = self._tool_augmented_v306_grounded_fact_failures(
                    corrected_response,
                    response_mode,
                    injected_grounded_auto_tool_facts,
                    has_answer_contract=self._tool_augmented_has_answer_contract(corrected_raw),
                )
                if corrected_grounded_fact_failures:
                    corrected_failures = self._tool_augmented_combine_failed_checks(
                        corrected_grounded_fact_failures,
                        corrected_failures,
                    )
                correction_reverified = bool(failed_checks)
                if output_truncated and not corrected_truncated:
                    recovery_kept = True
                if not corrected_truncated and (
                    not corrected_failures or len(corrected_failures) <= len(failed_checks)
                ):
                    final_response = corrected_response
                    correction_kept = True
                else:
                    final_response = current_answer_before_correction
            except Exception as exc:
                correction_error = type(exc).__name__
                final_response = current_answer_before_correction

        trace = [
            make_trace_step(
                "mode",
                (
                    "tool_augmented; "
                    f"experiment_preset={self.score_first_experiment_preset}; "
                    f"response_mode={response_mode}"
                ),
            ),
            make_trace_step("experiment_preset", self.score_first_experiment_preset),
            make_trace_step("model_calls", str(model_calls)),
            make_trace_step(
                "final_prompt_profile",
                "full" if full_final_mode else self.score_first_prompt_profile,
            ),
            make_trace_step("tool_extractor_used", str(tool_extractor_used).lower()),
            make_trace_step(
                "formalizer_plan_injected",
                "false" if full_final_mode else "true",
            ),
            make_trace_step("formalizer_used", str(formalizer_used).lower()),
            make_trace_step("formalizer_plan_lines", str(len(protocol.plan_lines))),
            make_trace_step(
                "formalizer_protocol_usable",
                str(
                    bool(protocol.tool_requests)
                    if full_final_mode
                    else bool(protocol.plan_lines or protocol.tool_requests)
                ).lower(),
            ),
            make_trace_step("tool_requests", str(len(protocol.tool_requests))),
            make_trace_step("target_tool_requests", str(len(target_tool_requests))),
            make_trace_step(
                "grounded_auto_tool_requests",
                str(len(grounded_auto_tool_requests)),
            ),
            # Keep the historical keys formalizer-scoped for compatibility;
            # target-bound activity has explicit parallel telemetry below.
            make_trace_step("tool_successes", str(len(formalizer_tool_facts))),
            make_trace_step(
                "tool_requested_ops",
                self._tool_augmented_trace_operations(protocol.tool_requests),
            ),
            make_trace_step(
                "tool_succeeded_ops",
                self._tool_augmented_trace_operations(formalizer_tool_facts),
            ),
            make_trace_step(
                "tool_injected_ops",
                self._tool_augmented_trace_operations(formalizer_tool_facts),
            ),
            make_trace_step(
                "tool_all_requested_ops",
                self._tool_augmented_trace_operations(merged_tool_requests),
            ),
            make_trace_step(
                "tool_all_succeeded_ops",
                self._tool_augmented_trace_operations(tool_facts),
            ),
            make_trace_step(
                "tool_all_injected_ops",
                self._tool_augmented_trace_operations(evidence.included_facts),
            ),
            make_trace_step("target_tool_successes", str(len(target_tool_facts))),
            make_trace_step(
                "grounded_auto_tool_successes",
                str(len(grounded_auto_tool_facts)),
            ),
            make_trace_step(
                "target_tool_requested_ops",
                self._tool_augmented_trace_operations(target_tool_requests),
            ),
            make_trace_step(
                "target_tool_succeeded_ops",
                self._tool_augmented_trace_operations(target_tool_facts),
            ),
            make_trace_step("tool_worker_status", tool_worker_status),
            make_trace_step("extended_tools", str(self.tool_augmented_extended_tools).lower()),
            make_trace_step("v307_tools", str(self.tool_augmented_v307_tools).lower()),
            make_trace_step("completion_budget", str(self.max_tokens)),
            make_trace_step("output_truncated_detected", str(output_truncated).lower()),
            make_trace_step("recovery_call_used", str(recovery_call_used).lower()),
            make_trace_step("recovery_kept", str(recovery_kept).lower()),
            make_trace_step(
                "challenge_reasoning",
                str(self.tool_augmented_challenge_reasoning).lower(),
            ),
            make_trace_step("difficulty_tier", str(context.get("difficulty_tier") or "standard")),
            make_trace_step("correction_targets", ",".join(correction_targets) or "none"),
            make_trace_step(
                "verification_failure_methods",
                ",".join(verification_failure_methods) or "none",
            ),
            make_trace_step("decisive_check_failure", str(bool(failed_checks)).lower()),
            make_trace_step("correction_call_used", str(correction_used).lower()),
            make_trace_step("correction_reverified", str(correction_reverified).lower()),
            make_trace_step("correction_kept", str(correction_kept).lower()),
        ]
        if formalizer_error:
            trace.append(make_trace_step("formalizer_fallback", formalizer_error))
        if correction_error:
            trace.append(make_trace_step("correction_fallback", correction_error))
        return self._score_first_json_result(final_response, trace)

    def _tool_augmented_is_v305_full_final(self) -> bool:
        """Whether this instance is running the controlled V3.0.5 arm."""

        return (
            self.score_first_experiment_preset == _TOOL_AUGMENTED_V305_FULL_FINAL_PRESET
            and self.score_first_prompt_profile == "full"
        )

    def _tool_augmented_is_v306_cross_subject(self) -> bool:
        """Whether the V3.0.6 cross-subject safety layer is active.

        V3.0.7 is a strict superset of the V3.0.6 arm: the default preset
        keeps the earlier parent-grounded facts, domain selection guidance,
        and scalar consistency gate while adding its newer tools and output
        discipline.  Treating the layer as active for both presets prevents
        the default upgrade from silently dropping V3.0.6 protections.
        """

        return (
            self.score_first_experiment_preset
            in {
                _TOOL_AUGMENTED_V306_CROSS_SUBJECT_PRESET,
                _TOOL_AUGMENTED_V307_CROSS_SUBJECT_PRESET,
            }
            and self.score_first_prompt_profile == "full"
        )

    def _tool_augmented_is_v307_cross_subject(self) -> bool:
        """Whether this instance is running the V3.0.7 high-budget arm."""

        return (
            self.score_first_experiment_preset == _TOOL_AUGMENTED_V307_CROSS_SUBJECT_PRESET
            and self.score_first_prompt_profile == "full"
            and self.tool_augmented_v307_tools
        )

    def _tool_augmented_requests_for_arm(
        self,
        requests: Any,
        *,
        allow_v306_only: bool,
        allow_v307_only: bool = False,
    ) -> List[Any]:
        """Keep pre-V3.0.6 rollback surfaces frozen as tool families grow.

        The parser knows the newer V3.0.6 operations so it can safely parse a
        V3.0.6 extractor response.  A model running an older rollback arm
        must nevertheless not get a newly added operation merely because it
        happened to emit its name.  This filter preserves each arm's
        deterministic evidence surface.
        """

        v306_only_operations = {
            ("number_theory", "multiplicative_order"),
            ("number_theory", "primitive_root_check"),
            ("combinatorics", "partition_exact_parts"),
            ("probability", "binomial_pmf"),
            ("probability", "poisson_pmf"),
            ("probability", "hypergeometric_pmf"),
            ("matrix", "trace"),
            ("matrix", "power"),
        }
        v307_only_operations = {
            ("number_theory", "mod_inverse"),
            ("number_theory", "pow_mod"),
            ("number_theory", "crt"),
            ("recurrence", "linear_eval"),
            ("combinatorics", "partition_total"),
            ("combinatorics", "distinct_partition"),
            ("combinatorics", "composition_positive"),
            ("graph", "cycle_chromatic"),
            ("graph", "complete_graph_chromatic"),
            ("graph", "complete_bipartite_chromatic"),
            ("graph", "path_matching"),
            ("graph", "cycle_matching"),
            ("graph", "planar_faces"),
            ("graph", "turan_edges"),
            ("sympy", "coefficient"),
            ("matrix", "rref"),
            ("matrix", "nullspace"),
            ("combinatorics", "factorial"),
            ("combinatorics", "fibonacci"),
            ("number_theory", "is_prime"),
            ("number_theory", "divisor_count"),
            ("number_theory", "fibonacci_mod"),
            ("probability", "geometric_pmf"),
            ("probability", "negative_binomial_pmf"),
            ("probability", "binomial_cdf"),
        }
        result: List[Any] = []
        for request in list(requests or []):
            key = (
                str(getattr(request, "family", "") or ""),
                str(getattr(request, "operation", "") or ""),
            )
            if (not allow_v306_only and key in v306_only_operations) or (
                not allow_v307_only and key in v307_only_operations
            ):
                continue
            result.append(request)
        return result

    def _tool_augmented_v305_tool_extraction_allowed(self, response_mode: str) -> bool:
        """Keep the first low-thinking call off theorem-heavy response modes."""

        return response_mode in {
            _SCORE_FIRST_RESPONSE_MODE_ANSWER,
            _SCORE_FIRST_RESPONSE_MODE_DERIVATION,
        }

    def _build_tool_augmented_formalizer_prompt(
        self,
        problem: str,
        context: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        subject = str(context.get("subject_hint") or "").strip()
        subject_line = f"Trusted subject: {subject}\n" if subject else ""
        response_mode = str(context.get("response_mode") or _SCORE_FIRST_RESPONSE_MODE_ANSWER)

        if (
            self._tool_augmented_is_v305_full_final()
            or self._tool_augmented_is_v306_cross_subject()
            or self._tool_augmented_is_v307_cross_subject()
        ):
            formalizer_system = _TOOL_AUGMENTED_V305_EXTRACTOR_SYSTEM_PROMPT
            if self.tool_augmented_extended_tools:
                formalizer_system = (
                    f"{formalizer_system}\n\n{_TOOL_AUGMENTED_EXTENDED_TOOL_SYNTAX}"
                )
            if self._tool_augmented_is_v306_cross_subject():
                formalizer_system = (
                    f"{formalizer_system}\n\n{_TOOL_AUGMENTED_V306_CROSS_SUBJECT_TOOL_SYNTAX}"
                )
                domain_guidance = self._tool_augmented_v306_domain_tool_guidance(context)
                if domain_guidance:
                    formalizer_system = (
                        f"{formalizer_system}\n\nSubject-specific priority:\n{domain_guidance}"
                    )
            if self._tool_augmented_is_v307_cross_subject():
                formalizer_system = (
                    f"{formalizer_system}\n\n{_TOOL_AUGMENTED_V307_CROSS_SUBJECT_TOOL_SYNTAX}"
                )
            return [
                {"role": "system", "content": formalizer_system},
                {
                    "role": "user",
                    "content": (
                        f"{subject_line}Response mode: {response_mode}\n"
                        "Extract only exact deterministic computations that may help solve "
                        "the ORIGINAL problem.\n"
                        "Output TOOL lines only when the mathematical input is explicit.\n"
                        "If no safe deterministic computation is useful, output: NO_TOOL\n"
                        "Do not solve the problem. Do not recommend a method. Do not write a proof.\n"
                        f"ORIGINAL problem:\n{problem}"
                    ),
                },
            ]

        proof_note = ""
        if response_mode in {
            _SCORE_FIRST_RESPONSE_MODE_PROOF,
            _SCORE_FIRST_RESPONSE_MODE_PROOF_OR_DISPROOF,
        }:
            proof_note = (
                "This is a proof-oriented task. Prefer CLAIM/HYPOTHESES/SUBGOAL/"
                "THEOREM/FAILURE_TRAP lines. Do not write the proof.\n"
            )
        else:
            proof_note = (
                "For explicit computational structure, request only safe deterministic "
                "TOOL operations that can be parsed exactly.\n"
            )
        formalizer_system = _TOOL_AUGMENTED_FORMALIZER_SYSTEM_PROMPT
        if self.tool_augmented_extended_tools:
            formalizer_system = (
                f"{formalizer_system}\n\n{_TOOL_AUGMENTED_EXTENDED_TOOL_SYNTAX}"
            )
        return [
            {"role": "system", "content": formalizer_system},
            {
                "role": "user",
                "content": (
                    f"{subject_line}Response mode: {response_mode}\n"
                    f"{proof_note}"
                    f"Original problem:\n{problem}"
                ),
            },
        ]

    def _tool_augmented_v306_domain_tool_guidance(
        self,
        context: Dict[str, Any],
    ) -> str:
        """Return a bounded, domain-specific tool-selection reminder.

        It is intentionally a selection aid for Call 1 only.  It never enters
        the final solver prompt, where the model must reason from the frozen
        FULL conditioning and any successfully computed provenance-bearing
        facts.
        """

        domain = str(context.get("strategy_domain") or "").strip().lower()
        return _TOOL_AUGMENTED_V306_DOMAIN_TOOL_GUIDANCE.get(domain, "")

    def _build_tool_augmented_v305_final_prompt(
        self,
        problem: str,
        context: Dict[str, Any],
        evidence_block: str,
    ) -> List[Dict[str, str]]:
        """Return FULL ScoreFirst unchanged unless usable tool evidence exists."""

        baseline = self._build_full_score_first_prompt(problem, context)
        if not evidence_block:
            return baseline

        user_content = baseline[1]["content"]
        problem_marker = f"Problem:\n{problem}"
        prefix, marker, suffix = user_content.partition(problem_marker)
        if not marker:
            # This should be unreachable for the frozen full builder.  Failing
            # closed here preserves the exact baseline rather than risking a
            # misplaced evidence block.
            return baseline

        evidence_instruction = (
            "Optional deterministic computations:\n\n"
            f"{evidence_block}\n\n"
            "Each computation is exact only for the input explicitly shown.\n"
            "Compare that shown input with the original problem before using it.\n"
            "Ignore any computation whose input does not match the requested object.\n\n"
        )
        return [
            baseline[0],
            {
                "role": "user",
                "content": f"{prefix}{evidence_instruction}{marker}{suffix}",
            },
        ]

    def _build_tool_augmented_v306_final_prompt(
        self,
        problem: str,
        context: Dict[str, Any],
        evidence_block: str,
    ) -> List[Dict[str, str]]:
        """Build V3.0.6 from the frozen FULL prompt with bounded additions.

        A standard task with no successful deterministic fact intentionally
        gets the exact frozen FULL messages.  For an objectively high-complexity
        task, the only non-tool addition is a short discipline card; it is not
        a strategy replacement, a plan, or ungrounded mathematical advice.
        """

        baseline = self._build_full_score_first_prompt(problem, context)
        high_difficulty_card = ""
        if self.tool_augmented_challenge_reasoning:
            high_difficulty_card = str(context.get("high_difficulty_card") or "").strip()
        if not evidence_block and not high_difficulty_card:
            return baseline

        user_content = baseline[1]["content"]
        problem_marker = f"Problem:\n{problem}"
        prefix, marker, suffix = user_content.partition(problem_marker)
        if not marker:
            # Fail closed to the known baseline.  A malformed marker must not
            # turn an optional augmentation into a changed full prompt.
            return baseline

        additions: List[str] = []
        if high_difficulty_card:
            additions.append(
                "High-difficulty solving discipline:\n"
                f"{high_difficulty_card}\n"
                "Apply this silently; do not expose private scratch work."
            )
        if evidence_block:
            additions.append(
                "Optional deterministic computations:\n\n"
                f"{evidence_block}\n\n"
                "Each computation is exact only for the input explicitly shown.\n"
                "Compare that shown input with the original problem before using it.\n"
                "Ignore any computation whose input does not match the requested object."
            )
        augmentation = "\n\n".join(additions) + "\n\n"
        return [
            baseline[0],
            {
                "role": "user",
                "content": f"{prefix}{augmentation}{marker}{suffix}",
            },
        ]

    def _build_tool_augmented_v307_final_prompt(
        self,
        problem: str,
        context: Dict[str, Any],
        evidence_block: str,
    ) -> List[Dict[str, str]]:
        """Build the high-budget arm from FULL conditioning plus compact controls.

        V3.0.7 keeps the known-good FULL prompt as the base.  Its only new
        reasoning instruction is a short output-discipline card for objective
        hard items; exact tool evidence is still the sole mathematical
        augmentation.
        """

        baseline = self._build_full_score_first_prompt(problem, context)
        high_difficulty_card = str(context.get("high_difficulty_card") or "").strip()
        if high_difficulty_card:
            high_difficulty_card = _TOOL_AUGMENTED_V307_HIGH_DIFFICULTY_DISCIPLINE
        if not evidence_block and not high_difficulty_card:
            return baseline

        user_content = baseline[1]["content"]
        problem_marker = f"Problem:\n{problem}"
        prefix, marker, suffix = user_content.partition(problem_marker)
        if not marker:
            return baseline

        additions: List[str] = []
        if high_difficulty_card:
            additions.append(
                "High-difficulty solving discipline:\n"
                f"{high_difficulty_card}\n"
                "Apply this silently; do not expose private scratch work."
            )
        if evidence_block:
            additions.append(
                "Output discipline:\n"
                f"{_SCORE_FIRST_COMPACT_OUTPUT_DISCIPLINE}\n"
                "Apply this silently; do not expose private scratch work."
            )
            additions.append(
                "Optional deterministic computations:\n\n"
                f"{evidence_block}\n\n"
                "Each computation is exact only for the input explicitly shown.\n"
                "Compare that shown input with the original problem before using it.\n"
                "Ignore any computation whose input does not match the requested object."
            )
        augmentation = "\n\n".join(additions) + "\n\n"
        return [
            baseline[0],
            {"role": "user", "content": f"{prefix}{augmentation}{marker}{suffix}"},
        ]

    def _build_tool_augmented_final_prompt(
        self,
        problem: str,
        context: Dict[str, Any],
        plan_text: str,
        evidence_block: str,
    ) -> List[Dict[str, str]]:
        response_mode = str(context.get("response_mode") or _SCORE_FIRST_RESPONSE_MODE_ANSWER)
        subject = str(context.get("subject_hint") or "").strip()
        subject_line = f"Trusted subject: {subject}\n\n" if subject else ""
        shape = self._tool_augmented_response_shape_instruction(response_mode)
        return [
            {
                "role": "system",
                "content": f"{_TOOL_AUGMENTED_FINAL_SYSTEM_BASE}\n{shape}",
            },
            {
                "role": "user",
                "content": (
                    f"{subject_line}"
                    f"Requested response mode: {response_mode}\n\n"
                    f"Compact formalizer plan (advisory):\n{plan_text}\n\n"
                    f"{evidence_block}\n\n"
                    f"ORIGINAL PROBLEM:\n{problem}"
                ),
            },
        ]

    def _build_tool_augmented_correction_prompt(
        self,
        problem: str,
        context: Dict[str, Any],
        current_answer: str,
        failed_checks: List[str],
        evidence_block: str,
        truncation_recovery: bool = False,
    ) -> List[Dict[str, str]]:
        response_mode = str(context.get("response_mode") or _SCORE_FIRST_RESPONSE_MODE_ANSWER)
        subject = str(context.get("subject_hint") or "").strip()
        subject_line = f"Trusted subject: {subject}\n\n" if subject else ""
        failures = "\n".join(f"- {item}" for item in failed_checks[:3])
        shape = self._tool_augmented_response_shape_instruction(response_mode)
        if truncation_recovery:
            system = (
                "The previous completion was cut off before it became a complete response. "
                "Re-solve the ORIGINAL problem from the beginning and emit one complete replacement. "
                "Do not mention the interruption, recovery, token limits, or hidden reasoning. "
                "Any exact failed checks below are still hard evidence for the shown object. "
                f"{shape}"
            )
        else:
            system = (
                "A deterministic exact check found a concrete failure in the current answer. "
                "Solve the ORIGINAL problem again and correct that specific failure. "
                "Do not merely review or discuss the old answer. "
                "Use the failed check as hard evidence only for the exact statement shown. "
                "Parent-bound target evidence is authoritative for that exact requested object; "
                "other formalizer facts are advisory.\n"
                f"{shape}"
            )
        return [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"{subject_line}"
                    f"Current answer:\n{current_answer}\n\n"
                    f"Exact failed deterministic check(s):\n{failures}\n\n"
                    f"{evidence_block}\n\n"
                    f"ORIGINAL PROBLEM:\n{problem}"
                ),
            },
        ]

    def _build_tool_augmented_recovery_prompt(
        self,
        problem: str,
        context: Dict[str, Any],
        evidence_block: str,
    ) -> List[Dict[str, str]]:
        """Build a replacement prompt for a transport/completion truncation.

        This is intentionally separate from mathematical correction: a length
        stop is not evidence that the mathematics was wrong.  The prompt keeps
        the original problem and bounded exact facts, but omits the partial
        completion so the model can restart cleanly within the same budget.
        """

        response_mode = str(context.get("response_mode") or _SCORE_FIRST_RESPONSE_MODE_ANSWER)
        subject = str(context.get("subject_hint") or "").strip()
        subject_line = f"Trusted subject: {subject}\n\n" if subject else ""
        shape = self._tool_augmented_response_shape_instruction(response_mode)
        system = (
            f"{_TOOL_AUGMENTED_FINAL_SYSTEM_BASE}\n"
            f"{shape}\n"
            "The previous completion was cut off before it became a complete response. "
            "Re-solve the ORIGINAL problem from the beginning and emit one complete replacement. "
            "Do not mention the interruption, recovery, token limits, or hidden reasoning."
        )
        evidence = evidence_block or "(no deterministic tool facts)"
        if response_mode == _SCORE_FIRST_RESPONSE_MODE_ANSWER:
            output_rule = (
                "Put the complete requested answer on the first visible line exactly as "
                "Final answer: <complete requested answer>, then stop."
            )
        else:
            output_rule = "State the conclusion/result first and include all requested proof or derivation steps."
        return [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"{subject_line}{output_rule}\n\n"
                    f"{evidence}\n\n"
                    "ORIGINAL PROBLEM:\n"
                    f"{problem}"
                ),
            },
        ]

    def _tool_augmented_response_shape_instruction(self, response_mode: str) -> str:
        if response_mode == _SCORE_FIRST_RESPONSE_MODE_DERIVATION:
            return (
                "State the final result first, then give a concise derivation containing "
                "the mathematical steps explicitly requested."
            )
        if response_mode == _SCORE_FIRST_RESPONSE_MODE_PROOF:
            return "State the conclusion first, then give a complete rigorous proof."
        if response_mode == _SCORE_FIRST_RESPONSE_MODE_PROOF_OR_DISPROOF:
            return (
                "Determine whether the claim is true or false. If true, prove it; if false, "
                "give and verify a counterexample or disproof."
            )
        if response_mode == _SCORE_FIRST_RESPONSE_MODE_CONSTRUCTION:
            return (
                "State the requested construction or counterexample first, then give the "
                "verification needed to establish all requested properties."
            )
        return (
            "Your first visible line must be exactly: "
            "Final answer: <complete requested answer>. "
            "Put nothing before that line. Answer every requested part."
        )

    def _tool_augmented_v306_grounded_fact_failures(
        self,
        final_response: str,
        response_mode: str,
        facts: List[Any],
        *,
        has_answer_contract: bool,
    ) -> List[str]:
        """Detect only an unambiguous contradiction of injected parent facts.

        This is deliberately narrower than a general answer grader.  It is
        enabled only for V3.0.6 ANSWER_VALUE responses and only for simple
        integer/rational results from a parent-selected request that was
        actually shown to the final solver.  Equivalent symbolic forms,
        prose-heavy derivations, sets, matrices, and all formalizer-selected
        facts are left to the existing conservative verifier.
        """

        if (
            not self._tool_augmented_is_v306_cross_subject()
            or response_mode != _SCORE_FIRST_RESPONSE_MODE_ANSWER
            or not has_answer_contract
            or not facts
        ):
            return []
        answer = str(final_response or "").replace("−", "-")
        if not answer or answer == DEFAULT_FALLBACK:
            return []

        failures: List[str] = []
        seen: set[str] = set()
        for fact in facts:
            statement = str(getattr(fact, "statement", "") or "")
            left, separator, expected = statement.rpartition(" -> ")
            expected = expected.strip()
            # Restrict the automatic consistency gate to exact scalar output.
            # A decimal is intentionally excluded because formatting and
            # requested precision can make literal agreement misleading.
            if (
                not separator
                or not left
                or not re.fullmatch(r"[+-]?\d+(?:/\d+)?", expected)
                or len(expected) > 80
            ):
                continue
            literal = re.compile(
                rf"(?<![\d/]){re.escape(expected)}(?![\d/])"
            )
            if literal.search(answer):
                continue
            failure = (
                "Parent-grounded exact computation says "
                f"{statement}, but the current answer does not contain {expected}."
            )
            if failure not in seen:
                seen.add(failure)
                failures.append(failure)
            if len(failures) >= 3:
                break
        return failures

    @staticmethod
    def _tool_augmented_has_answer_contract(raw_output: str) -> bool:
        """Require the mature ANSWER_VALUE surface before literal checking."""

        return bool(
            re.search(
                r"(?im)^\s*final\s+answer\s*:\s*\S",
                str(raw_output or ""),
            )
        )

    @staticmethod
    def _tool_augmented_combine_failed_checks(*groups: List[str]) -> List[str]:
        """Stable-dedupe bounded correction evidence from independent gates."""

        combined: List[str] = []
        for group in groups:
            for item in list(group or []):
                text = str(item or "").strip()
                if text and text not in combined:
                    combined.append(text)
                if len(combined) >= 3:
                    return combined
        return combined

    def _tool_augmented_decisive_failures(
        self,
        problem: str,
        final_response: str,
        response_mode: str,
        context: Dict[str, Any],
        *,
        with_metadata: bool = False,
    ) -> Any:
        # System-inferred answer checks are intentionally conservative and are
        # only used on answer-value tasks. Proof text is never heuristically
        # parsed into a correction trigger.
        requests = self._tool_augmented_verification_requests(problem, context)
        correction_targets = []
        for request in requests:
            kind = str(request.get("kind") or "")
            if kind and kind not in correction_targets:
                correction_targets.append(kind)
        if response_mode != _SCORE_FIRST_RESPONSE_MODE_ANSWER or not requests:
            empty_result = ([], correction_targets, [])
            return empty_result if with_metadata else empty_result[0]

        failures: List[str] = []
        failure_methods: List[str] = []
        try:
            from math_agent_core.tools.augmented_tool import (
                run_answer_verification_in_subprocess,
            )

            evidence = run_answer_verification_in_subprocess(
                final_response,
                response_mode,
                requests=requests,
                timeout_seconds=self.tool_augmented_verification_timeout_seconds,
            )
            for item in evidence:
                if bool(item.get("is_decisive", False)) and str(item.get("status", "")) == "fail":
                    failures.append(self._tool_augmented_failure_text(item))
                    method = self._tool_augmented_trace_method(item.get("method"))
                    if method and method not in failure_methods:
                        failure_methods.append(method)
        except Exception:
            # Verification is advisory and must never turn a tool failure into
            # a solve failure.  Keep the already-derived target telemetry; do
            # not reference the removed legacy ``targets`` variable here.
            empty_result = ([], correction_targets, [])
            return empty_result if with_metadata else empty_result[0]

        # Deduplicate and keep correction evidence compact.
        deduped: List[str] = []
        for failure in failures:
            if failure not in deduped:
                deduped.append(failure)
        result = (deduped[:3], correction_targets, failure_methods[:3])
        return result if with_metadata else result[0]

    def _tool_augmented_verification_requests(
        self,
        problem: str,
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Bind exact verification objects from the grounded request spans.

        The returned objects are parent-owned data.  The worker receives only
        these bounded requests, never the original problem, so it cannot select
        a first equation or first matrix from unrelated background text.
        """

        if not isinstance(context, dict):
            return []
        spans = context.get("request_spans")
        actions = context.get("requested_actions")
        if not isinstance(spans, list) or not isinstance(actions, list):
            return []
        if _SCORE_FIRST_RESPONSE_MODE_ANSWER not in actions:
            return []

        requests: List[Dict[str, Any]] = []
        for span in spans[:8]:
            if not isinstance(span, str) or len(span) > 800:
                return []
            text = span.strip()
            if not text:
                continue

            matrix_requests, matrix_recognized = self._tool_augmented_matrix_requests_from_span(
                text, problem
            )
            if matrix_recognized and matrix_requests is None:
                return []
            if matrix_requests:
                requests.extend(matrix_requests)

            equation_requests, equation_recognized = self._tool_augmented_equation_requests_from_span(
                text, problem
            )
            if equation_recognized and equation_requests is None:
                return []
            if equation_requests:
                requests.extend(equation_requests)

            arithmetic, arithmetic_recognized = self._tool_augmented_arithmetic_request_from_span(text)
            if arithmetic_recognized and arithmetic is None:
                return []
            if arithmetic is not None:
                requests.append(arithmetic)

        # Normalize once in the parent as an additional closed-schema check.
        try:
            from math_agent_core.tools.augmented_tool import _normalize_verification_requests

            normalized = _normalize_verification_requests(requests)
        except Exception:
            return []
        return normalized if len(normalized) == len({repr(item) for item in requests}) else []

    def _tool_augmented_target_tool_requests(
        self,
        requests: List[Dict[str, Any]],
    ) -> List[Any]:
        """Turn only parent-bound answer targets into deterministic tool calls.

        This is deliberately a small closed-world bridge.  It never parses a
        model response and never searches the full problem for an object.  The
        exact expression/equation/matrix has already been bound by
        ``_tool_augmented_verification_requests``; if that binding is
        ambiguous, this helper returns no requests.
        """

        try:
            from math_agent_core.tools.augmented_tool import ToolRequest
        except Exception:
            return []

        if not isinstance(requests, list):
            return []
        result: List[Any] = []
        seen: set[tuple[str, str, str]] = set()
        for request in requests[:4]:
            if not isinstance(request, dict):
                continue
            kind = str(request.get("kind") or "")
            family = ""
            operation = ""
            arguments: Dict[str, Any] = {}
            if kind == "pure_arithmetic":
                family, operation = "sympy", "simplify"
                arguments = {"expr": request.get("expression")}
            elif kind in {"equation_solution", "equation_solution_set"}:
                family, operation = "sympy", "solve"
                arguments = {
                    "equation": request.get("equation"),
                    "var": request.get("variable"),
                }
                # The extended SymPy worker has a conservative domain mapper;
                # pass only domains it can represent exactly.  Other domains
                # remain available to the parent-bound post-answer check.
                domain = request.get("domain")
                if domain in {"real", "complex", "integer"}:
                    arguments["domain"] = domain
            elif kind in {"matrix_determinant", "matrix_rank"}:
                family = "matrix"
                operation = "determinant" if kind == "matrix_determinant" else "rank"
                arguments = {"matrix": request.get("matrix")}
            else:
                continue

            try:
                canonical_arguments = dict(arguments)
                if family == "sympy" and operation == "solve":
                    # An equation_solution and an equation_solution_set
                    # request with the default complex domain are the same
                    # deterministic solve.  Avoid injecting duplicate facts.
                    canonical_arguments.setdefault("domain", "complex")
                key = (family, operation, repr(canonical_arguments))
            except Exception:
                continue
            if key in seen:
                continue
            seen.add(key)
            result.append(
                ToolRequest(
                    family=family,
                    operation=operation,
                    arguments=arguments,
                    source="parent_target",
                )
            )
        return result

    def _tool_augmented_v306_grounded_auto_tool_requests(
        self,
        problem: str,
        context: Dict[str, Any],
        verification_requests: List[Dict[str, Any]],
    ) -> List[Any]:
        """Create a few exact V3.0.6 requests from target-bound text only.

        This is not a second natural-language solver.  It is a closed-world
        recognizer for a small set of unambiguous textbook notations.  Every
        input either comes from an already parent-bound verification object or
        is present in a request span itself.  In particular, it never searches
        an unrelated premise for the first equation, matrix, or number.
        """

        source_problem = str(problem or "")
        if not isinstance(context, dict):
            return []
        response_mode = str(context.get("response_mode") or "")
        if response_mode not in {
            _SCORE_FIRST_RESPONSE_MODE_ANSWER,
            _SCORE_FIRST_RESPONSE_MODE_DERIVATION,
        }:
            return []
        spans = context.get("request_spans")
        if not isinstance(spans, list):
            return []
        if not spans and self._tool_augmented_is_v307_cross_subject():
            # Some compact probability/coefficient notations contain no
            # imperative verb, so the generic request parser intentionally
            # yields no span.  Permit the whole short sentence only when it
            # contains an unmistakable explicit target surface.
            if len(source_problem) <= 1200 and re.search(
                r"\bP\s*\(|\b(?:coefficient|derivative|integral|limit)\b|"
                r"\b(?:binomial|poisson|hypergeometric|partition|composition|derangement|"
                r"surjection|parenthesization|triangulation|binary\s+strings?|books?|"
                r"people|students?|seating|permutations?|graph|tree|cycle|path|matching|divisible|gcd|totient|inverse|"
                r"modulo|congruent|recurrence|sequence|generating\s+function|coefficient)\b|"
                r"\b(?:K|C|P)\s*[_\{]?\s*\d|"
                r"系数|求导|积分|极限|分拆|组合|错排|满射|剖分|图|树|圈|匹配|同余|欧拉|递推|数列|排列|字符串",
                source_problem,
                flags=re.IGNORECASE,
            ):
                spans = [source_problem]
            else:
                return []
        try:
            from math_agent_core.tools.augmented_tool import ToolRequest
        except Exception:
            return []

        result: List[Any] = []
        seen: set[tuple[str, str, str]] = set()
        max_requests = 6 if self._tool_augmented_is_v307_cross_subject() else 4

        def append(family: str, operation: str, arguments: Dict[str, Any]) -> None:
            if len(result) >= max_requests:
                return
            try:
                key = (family, operation, repr(arguments))
            except Exception:
                return
            if key in seen:
                return
            seen.add(key)
            result.append(
                ToolRequest(
                    family=family,
                    operation=operation,
                    arguments=dict(arguments),
                    source="parent_explicit",
                )
            )

        # Reuse already validated target bindings, but mark them separately so
        # the final evidence trace says that the parent—not Call 1—selected the
        # exact object.
        for request in self._tool_augmented_target_tool_requests(verification_requests):
            append(
                str(getattr(request, "family", "") or ""),
                str(getattr(request, "operation", "") or ""),
                dict(getattr(request, "arguments", {}) or {}),
            )

        span_candidates = list(spans[:4])
        if self._tool_augmented_is_v307_cross_subject() and source_problem not in span_candidates:
            # If the request-span extractor only kept a short question such as
            # "How many edges?", use the original statement as a final,
            # closed-world parsing surface.  It is consulted only when the
            # already extracted spans yielded no fact, preventing unrelated
            # premise values from being mixed into a successful target parse.
            span_candidates.append(source_problem)
        for raw_span in span_candidates:
            if raw_span == source_problem and result:
                break
            if len(result) >= max_requests or not isinstance(raw_span, str):
                break
            span = raw_span.strip()
            if not span or len(span) > 800:
                continue
            normalized = span.replace("φ", "phi").replace("Φ", "phi")

            for match in re.finditer(
                r"\b(?:gcd|greatest\s+common\s+divisor)\s*(?:of\s*)?\(?\s*"
                r"([+-]?\d{1,10})\s*[,;]\s*([+-]?\d{1,10})\s*\)?",
                normalized,
                flags=re.IGNORECASE,
            ):
                append(
                    "number_theory",
                    "gcd",
                    {"a": int(match.group(1)), "b": int(match.group(2))},
                )

            for match in re.finditer(
                r"\b(?:euler\s+)?(?:phi|totient)\s*\(\s*(\d{1,10})\s*\)",
                normalized,
                flags=re.IGNORECASE,
            ):
                append("number_theory", "totient", {"n": int(match.group(1))})

            for match in re.finditer(
                r"\b(?:multiplicative\s+)?order\s+of\s+([+-]?\d{1,10})\s+"
                r"(?:mod(?:ulo)?|mod)\s+(\d{1,10})\b",
                normalized,
                flags=re.IGNORECASE,
            ):
                append(
                    "number_theory",
                    "multiplicative_order",
                    {"a": int(match.group(1)), "modulus": int(match.group(2))},
                )
            for match in re.finditer(
                r"([+-]?\d{1,10})\s*(?:模|mod)\s*(\d{1,10})[^.。;；\n]{0,20}(?:乘法阶|multiplicative\s+order)",
                normalized,
                flags=re.IGNORECASE,
            ):
                append("number_theory", "multiplicative_order", {"a": int(match.group(1)), "modulus": int(match.group(2))})

            congruence = re.search(
                r"([+-]?\d{1,10})\s*x\s*(?:congruent\s+to|≡)\s*([+-]?\d{1,10})"
                r"\s*(?:mod(?:ulo)?|模)\s*(\d{1,10})",
                normalized,
                flags=re.IGNORECASE,
            )
            if congruence:
                append(
                    "number_theory",
                    "linear_congruence",
                    {
                        "a": int(congruence.group(1)),
                        "b": int(congruence.group(2)),
                        "modulus": int(congruence.group(3)),
                    },
                )
            congruence_symbolic = re.search(
                r"([+-]?\d{1,10})\s*x\s*(?:≡|congruent\s+to)\s*([+-]?\d{1,10})"
                r"\s*\(\s*(?:mod|模)\s*(\d{1,10})\s*\)",
                normalized,
                flags=re.IGNORECASE,
            )
            if congruence_symbolic:
                append(
                    "number_theory",
                    "linear_congruence",
                    {
                        "a": int(congruence_symbolic.group(1)),
                        "b": int(congruence_symbolic.group(2)),
                        "modulus": int(congruence_symbolic.group(3)),
                    },
                )

            square_congruence = re.search(
                r"x\s*\^\s*2\s*(?:≡|congruent\s+to)\s*1\s*"
                r"\(\s*(?:mod|模)\s*(\d{1,10})\s*\)",
                normalized,
                flags=re.IGNORECASE,
            )
            if square_congruence and re.search(r"(?:how many|number of|多少|解)", normalized, flags=re.IGNORECASE):
                append("number_theory", "unit_square_solution_count", {"modulus": int(square_congruence.group(1))})

            primitive_root = re.search(
                r"(?:smallest\s+positive\s+)?primitive\s+root\s+(?:mod(?:ulo)?|模)\s*(\d{1,10})|"
                r"(?:最小的?正整数)?\s*原根\s*(?:模|mod)\s*(\d{1,10})",
                normalized,
                flags=re.IGNORECASE,
            )
            if primitive_root:
                modulus = primitive_root.group(1) or primitive_root.group(2)
                append("number_theory", "primitive_root", {"modulus": int(modulus)})

            divisible_union = re.search(
                r"(?:from|through|between)\s+1\s+(?:and|through)\s+(\d{1,9})"
                r"[^.。;；\n]{0,80}(?:divisible\s+by|被|能被)\s*"
                r"(?:at\s+least\s+one\s+of\s+)?([0-9,，\s]+)",
                normalized,
                flags=re.IGNORECASE,
            )
            if divisible_union:
                divisors = [int(item) for item in re.findall(r"\d+", divisible_union.group(2))]
                if 1 <= len(divisors) <= 8:
                    append(
                        "number_theory",
                        "count_divisible_union",
                        {"upper": int(divisible_union.group(1)), "divisors": divisors},
                    )

            last_digit = re.search(
                r"([+-]?\d{1,10})\s*\^\s*(\d{1,12})[^.。;；\n]{0,25}(?:last\s+digit|个位数字|个位)",
                normalized,
                flags=re.IGNORECASE,
            )
            if last_digit:
                append("number_theory", "pow_mod", {"base": int(last_digit.group(1)), "exponent": int(last_digit.group(2)), "modulus": 10})

            order_equation = re.search(
                r"([+-]?\d{1,10})\s*\^\s*n\s*(?:≡|congruent\s+to)\s*1\s*"
                r"\(\s*(?:mod|模)\s*(\d{1,10})\s*\)",
                normalized,
                flags=re.IGNORECASE,
            )
            if order_equation:
                append("number_theory", "multiplicative_order", {"a": int(order_equation.group(1)), "modulus": int(order_equation.group(2))})

            if re.search(r"\bcatalan\b", normalized, flags=re.IGNORECASE):
                catalan_match = re.search(
                    r"\bcatalan(?:\s+number)?\s*(?:c\s*[_{]?\s*(\d{1,4})\s*\}?|"
                    r"of\s+(\d{1,4})\b)",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if catalan_match is None:
                    catalan_match = re.search(
                        r"\bc\s*[_{]\s*(\d{1,4})\s*\}?\b",
                        normalized,
                        flags=re.IGNORECASE,
                    )
                if catalan_match is not None:
                    n_text = next(
                        (group for group in catalan_match.groups() if group),
                        "",
                    )
                    if n_text:
                        append("combinatorics", "catalan", {"n": int(n_text)})

            for match in re.finditer(
                r"\bderangements?\s+(?:of\s+)?(\d{1,4})\b",
                normalized,
                flags=re.IGNORECASE,
            ):
                append("combinatorics", "derangement", {"n": int(match.group(1))})

            for match in re.finditer(
                r"\bstirling(?:\s+number(?:\s+of\s+the\s+second\s+kind)?)?\s*"
                r"\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)",
                normalized,
                flags=re.IGNORECASE,
            ):
                append(
                    "combinatorics",
                    "stirling2",
                    {"n": int(match.group(1)), "k": int(match.group(2))},
                )

            for match in re.finditer(
                r"\b(?:binomial|choose)\s*\(\s*(\d{1,4})\s*,\s*(\d{1,4})\s*\)",
                normalized,
                flags=re.IGNORECASE,
            ):
                append(
                    "combinatorics",
                    "binomial",
                    {"n": int(match.group(1)), "k": int(match.group(2))},
                )

            for match in re.finditer(
                r"\bpartitions?\s+of\s+(\d{1,3})\s+into\s+exactly\s+(\d{1,3})\s+parts?\b",
                normalized,
                flags=re.IGNORECASE,
            ):
                append(
                    "combinatorics",
                    "partition_exact_parts",
                    {"total": int(match.group(1)), "parts": int(match.group(2))},
                )

            binomial_pmf = re.search(
                r"\bbinomial\s*\(\s*(?:n\s*=\s*)?(\d{1,3})\s*,\s*"
                r"(?:p\s*=\s*)?([0-9]+(?:/[0-9]+)?|0(?:\.\d+)?)\s*\).*?"
                r"\b(?:p|pr)\s*\(\s*[a-z]\s*=\s*(\d{1,3})\s*\)",
                normalized,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if binomial_pmf is not None:
                append(
                    "probability",
                    "binomial_pmf",
                    {
                        "trials": int(binomial_pmf.group(1)),
                        "p": binomial_pmf.group(2),
                        "successes": int(binomial_pmf.group(3)),
                    },
                )

            poisson_pmf = re.search(
                r"\bpoisson\s*\(\s*(?:lambda|lam|rate)\s*=\s*"
                r"([0-9]+(?:/[0-9]+)?|[0-9]+(?:\.\d+)?)\s*\).*?"
                r"\b(?:p|pr)\s*\(\s*[a-z]\s*=\s*(\d{1,3})\s*\)",
                normalized,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if poisson_pmf is not None:
                append(
                    "probability",
                    "poisson_pmf",
                    {"rate": poisson_pmf.group(1), "k": int(poisson_pmf.group(2))},
                )

            geometric_pmf = re.search(
                r"\b(?:X\s*~\s*)?geometric\s*\(\s*"
                r"(?:p|success|success_probability)\s*=\s*"
                r"([0-9]+(?:/[0-9]+)?|0(?:\.\d+)?)\s*\)"
                r".{0,100}?\b(?:p|pr)\s*\(\s*[a-z]\s*=\s*(\d{1,3})\s*\)",
                normalized,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if geometric_pmf is not None:
                append(
                    "probability",
                    "geometric_pmf",
                    {"success": geometric_pmf.group(1), "trials": int(geometric_pmf.group(2))},
                )

            factorial_target = re.search(
                r"(?<![A-Za-z0-9_])([0-9]{1,4})\s*!(?![=])",
                normalized,
            )
            modular_factorial_context = re.search(
                r"\b(?:remainder|residue|mod(?:ulo)?|divided\s+by|quotient)\b|"
                r"余数|模|除以|商",
                normalized,
                flags=re.IGNORECASE,
            )
            explicit_factorial_request = re.search(
                r"\b(?:compute|calculate|evaluate|find|determine|value)\b"
                r"[^.。;；\n]{0,24}\b(?:factorial|阶乘)\b|"
                r"\b(?:factorial|阶乘)\b",
                normalized,
                flags=re.IGNORECASE,
            )
            # Do not materialize a large factorial merely because it appears
            # as an exponent in a modular/remainder problem.  Such a fact is
            # both irrelevant and likely to be rejected for exceeding the
            # evidence budget.  Keep explicit factorial requests available.
            if factorial_target and (
                explicit_factorial_request
                or not modular_factorial_context
            ):
                append("combinatorics", "factorial", {"n": int(factorial_target.group(1))})

            fibonacci_target = re.search(
                r"\bfibonacci(?:\s+number)?\s*(?:\(?\s*(?:F\s*[_\{]?\s*)?"
                r"(\d{1,5})\s*\}?\s*\)?|of\s+(\d{1,5}))",
                normalized,
                flags=re.IGNORECASE,
            )
            if fibonacci_target is not None:
                n_text = fibonacci_target.group(1) or fibonacci_target.group(2)
                if n_text:
                    append("combinatorics", "fibonacci", {"n": int(n_text)})

            fibonacci_mod = re.search(
                r"\bF\s*[_\{]?\s*(\d{1,12})\s*\}?\s*"
                r"(?:mod(?:ulo)?|模)\s*(\d{1,12})",
                normalized,
                flags=re.IGNORECASE,
            )
            if fibonacci_mod:
                append(
                    "number_theory",
                    "fibonacci_mod",
                    {"n": int(fibonacci_mod.group(1)), "modulus": int(fibonacci_mod.group(2))},
                )

            prime_target = re.search(
                r"\b(?:is|whether)\s+([0-9]{1,12})\s+(?:a\s+)?prime\b|"
                r"\b([0-9]{1,12})\s*是否为?质数\b",
                normalized,
                flags=re.IGNORECASE,
            )
            if prime_target:
                n_text = prime_target.group(1) or prime_target.group(2)
                append("number_theory", "is_prime", {"n": int(n_text)})

            divisor_count = re.search(
                r"(?:how\s+many|number\s+of)\s+(?:positive\s+)?divisors?\s+does\s+([0-9]{1,12})\s+have|"
                r"([0-9]{1,12})[^.。;；\n]{0,20}(?:positive\s+)?divisors?\b|"
                r"([0-9]{1,12})\s*的(?:正)?因数(?:个数|数目)",
                normalized,
                flags=re.IGNORECASE,
            )
            if divisor_count:
                n_text = divisor_count.group(1) or divisor_count.group(2) or divisor_count.group(3)
                append("number_theory", "divisor_count", {"n": int(n_text)})

            # V3.0.7 adds only closed-world recognizers whose complete inputs
            # are visible in the target sentence or in one uniquely identified
            # recurrence/matrix binding.  The formalizer remains optional.
            if self._tool_augmented_is_v307_cross_subject():
                recurrence_request = self._tool_augmented_recurrence_request(
                    source_problem,
                    normalized,
                )
                if recurrence_request is not None:
                    append("recurrence", "linear_eval", recurrence_request)

                target_indices = re.findall(r"\b(?:b|T)\s*[_\{]?\s*(\d{1,4})\s*\}?", normalized)
                if target_indices:
                    target_n = int(target_indices[-1])
                    if re.search(r"binary\s+strings?[^.。;；\n]{0,80}no\s+consecutive\s+1|"
                                 r"二进制字符串[^.。;；\n]{0,80}连续的?\s*1", normalized, flags=re.IGNORECASE):
                        append(
                            "recurrence",
                            "linear_eval",
                            {
                                "sequence_symbol": "b",
                                "initial": ["1", "2"],
                                "coefficients": ["1", "1"],
                                "constant": 0,
                                "target_n": target_n,
                            },
                        )
                    elif re.search(r"(?:domino|tilings?|铺满|小砖)[^;；\n]{0,180}(?:natural\s+recurrence|recurrence|递推)", normalized, flags=re.IGNORECASE):
                        append(
                            "recurrence",
                            "linear_eval",
                            {
                                "sequence_symbol": "T",
                                "initial": ["1", "1"],
                                "coefficients": ["1", "1"],
                                "constant": 0,
                                "target_n": target_n,
                            },
                        )

                stairs = re.search(
                    r"(?:walk|stairs?|走完|上楼梯)[^.。;；\n]{0,80}(\d{1,5})\s*(?:steps?|级)[^.。;；\n]{0,100}"
                    r"(?:not\s+allowed|不允许)[^.。;；\n]{0,40}(?:two|2|两)[^.。;；\n]{0,20}(?:consecutive|连续)",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if stairs:
                    append("combinatorics", "steps_no_consecutive_two", {"total": int(stairs.group(1))})
                stairs_zh = re.search(
                    r"(?:走完|上楼梯)[^.。;；\n]{0,80}(\d{1,5})\s*级[^.。;；\n]{0,80}"
                    r"不允许连续两次",
                    normalized,
                )
                if stairs_zh:
                    append("combinatorics", "steps_no_consecutive_two", {"total": int(stairs_zh.group(1))})

                tiling_parts = re.search(
                    r"(?:board|木板)[^.。;；\n]{0,30}(?:length|长度)\s*n[^.。;；\n]{0,35}"
                    r"(?:lengths?|长度)\s*1\s*[,，]\s*2\s*(?:and|、)\s*3",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if tiling_parts and target_indices:
                    append("combinatorics", "tilings_parts", {"total": target_n, "pieces": [1, 2, 3]})
                tiling_zh = re.search(
                    r"长度为\s*n[^.。;；\n]{0,30}木板[^.。;；\n]{0,80}"
                    r"长度\s*1\s*、\s*2\s*、\s*3",
                    normalized,
                )
                if tiling_zh and target_indices:
                    append("combinatorics", "tilings_parts", {"total": target_n, "pieces": [1, 2, 3]})

                for match in re.finditer(
                    r"(?:multiplicative\s+)?inverse\s+of\s+([+-]?\d{1,12})\s+"
                    r"(?:mod(?:ulo)?|mod)\s+(\d{1,12})\b|"
                    r"([+-]?\d{1,12})\s*(?:模|mod)\s*(\d{1,12})\s*(?:的)?(?:乘法)?逆元",
                    normalized,
                    flags=re.IGNORECASE,
                ):
                    a = match.group(1) or match.group(3)
                    modulus = match.group(2) or match.group(4)
                    append("number_theory", "mod_inverse", {"a": int(a), "modulus": int(modulus)})

                for match in re.finditer(
                    r"([+-]?\d{1,12})\s*(?:\^|\*\*)\s*(\d{1,12})"
                    r"[^.。;；\n]{0,24}(?:mod(?:ulo)?|mod|模|除以)\s*(\d{1,12})",
                    normalized,
                    flags=re.IGNORECASE,
                ):
                    append(
                        "number_theory",
                        "pow_mod",
                        {
                            "base": int(match.group(1)),
                            "exponent": int(match.group(2)),
                            "modulus": int(match.group(3)),
                        },
                    )

                congruences = re.findall(
                    r"\bx\s*(?:≡|=)\s*([+-]?\d{1,12})\s*"
                    r"\(\s*(?:mod|模)\s*([1-9]\d{0,11})\s*\)",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if len(congruences) >= 2:
                    append(
                        "number_theory",
                        "crt",
                        {
                            "residues": [int(value) for value, _modulus in congruences[:8]],
                            "moduli": [int(modulus) for _value, modulus in congruences[:8]],
                        },
                    )

                partition_total = re.search(
                    r"\bpartitions?\s+(?:of\s+)?(\d{1,3})\b|"
                    r"\bpartitions?\s+does\s+(\d{1,3})\s+have\b|"
                    r"(?:整数)?分拆[^0-9]{0,16}(\d{1,3})",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if partition_total and not re.search(
                    r"exactly\s+\d+\s+parts?|into\s+\d+\s+parts?|互不相同|distinct",
                    normalized,
                    flags=re.IGNORECASE,
                ):
                    n_text = partition_total.group(1) or partition_total.group(2) or partition_total.group(3)
                    append("combinatorics", "partition_total", {"n": int(n_text)})

                set_partition = re.search(
                    r"(?:a\s+)?(?:set\s+of\s+)?(\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|"
                    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)"
                    r"\s+(?:labeled\s+)?(?:elements?|objects?)"
                    r"[^.。;；\n]{0,70}(?:partition(?:ed|ing)?|划分)"
                    r"[^.。;；\n]{0,40}(?:exactly|恰好)\s*(\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
                    r"(?:nonempty\s+)?(?:unlabeled\s+)?(?:blocks?|parts?|子集|块)",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if set_partition:
                    n_value = self._tool_augmented_parse_small_integer(set_partition.group(1))
                    k_value = self._tool_augmented_parse_small_integer(set_partition.group(2))
                    if n_value is None or k_value is None:
                        continue
                    append(
                        "combinatorics",
                        "stirling2",
                        {"n": n_value, "k": k_value},
                    )

                set_partition_zh = re.search(
                    r"(\d{1,3})\s*个[^.。;；\n]{0,20}(?:划分成|划分为)\s*(?:恰好\s*)?(\d{1,3})\s*个"
                    r"[^.。;；\n]{0,20}(?:非空|无标号|子集)",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if set_partition_zh:
                    append("combinatorics", "stirling2", {"n": int(set_partition_zh.group(1)), "k": int(set_partition_zh.group(2))})

                distinct_partition = re.search(
                    r"(?:partition|表示|分拆)[^.。;；\n]{0,80}(?:distinct|different|互不相同)[^.。;；\n]{0,40}(\d{1,3})|"
                    r"(\d{1,3})[^.。;；\n]{0,30}(?:互不相同|distinct)[^.。;；\n]{0,30}(?:positive|正整数)",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if distinct_partition:
                    n_text = distinct_partition.group(1) or distinct_partition.group(2)
                    append("combinatorics", "distinct_partition", {"n": int(n_text)})

                number_word_pattern = (
                    r"(?:\d{1,4}|zero|one|two|three|four|five|six|seven|eight|nine|ten|"
                    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
                    r"nineteen|twenty)"
                )
                composition = re.search(
                    rf"compositions?\s+of\s+({number_word_pattern})\s+into\s+"
                    rf"(?:exactly\s+)?({number_word_pattern})\s+(?:positive\s+)?"
                    r"(?:integer\s+)?parts?",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if composition:
                    total = self._tool_augmented_parse_small_integer(composition.group(1))
                    parts = self._tool_augmented_parse_small_integer(composition.group(2))
                    if total is not None and parts is not None:
                        append("combinatorics", "composition_positive", {"total": total, "parts": parts})

                bounded_nonnegative = re.search(
                    r"(?:nonnegative|non-negative|非负)[^=]{0,60}"
                    r"(?:sum|和为)\s*(\d{1,5})[^.。;；\n]{0,80}"
                    r"(?:x[_\{]?\s*1|variables?|元组)[^0-9]{0,18}(\d{1,4})[^.。;；\n]*"
                    r"(?:x[_\{]?\s*1\s*(?:>=|≥)\s*(\d{1,4})|"
                    r"x[_\{]?\s*1\s*(?:<=|≤)\s*(\d{1,4}))",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if bounded_nonnegative:
                    total = int(bounded_nonnegative.group(1))
                    variables = int(bounded_nonnegative.group(2))
                    lower = bounded_nonnegative.group(3)
                    upper = bounded_nonnegative.group(4)
                    if lower is not None:
                        append("combinatorics", "one_coordinate_lower", {"total": total, "variables": variables, "lower": int(lower)})
                    elif upper is not None:
                        append("combinatorics", "one_coordinate_upper", {"total": total, "variables": variables, "upper": int(upper)})

                binary_no_adjacent = re.search(
                    rf"binary\s+strings?\s+of\s+length\s+(\d{{1,5}})[^.。;；\n]{{0,80}}"
                    rf"(?:exactly\s+|with\s+)({number_word_pattern})\s+1s?[^.。;；\n]{{0,40}}"
                    r"(?:no\s+two|without\s+two|no\s+consecutive)\s+1",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if binary_no_adjacent:
                    ones = self._tool_augmented_parse_small_integer(binary_no_adjacent.group(2))
                    if ones is None:
                        continue
                    append(
                        "combinatorics",
                        "binary_no_adjacent",
                        {"length": int(binary_no_adjacent.group(1)), "ones": ones},
                    )

                choose_excluding_pair = re.search(
                    r"from\s+(\d{1,5})\s+distinct\s+books?\s*,?\s*choose\s+(\d{1,5})"
                    r"[^.。;；\n]{0,80}(?:may not both|not both|cannot both)",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if choose_excluding_pair:
                    append(
                        "combinatorics",
                        "choose_excluding_pair",
                        {"total": int(choose_excluding_pair.group(1)), "choose": int(choose_excluding_pair.group(2))},
                    )

                circular = re.search(
                    rf"({number_word_pattern})\s+people\s+sit\s+around\s+a\s+round\s+table"
                    r"[^;；\n]{0,180}(?:must\s+sit\s+next\s+to|adjacent)",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if circular:
                    people = self._tool_augmented_parse_small_integer(circular.group(1))
                    if people is not None:
                        append("combinatorics", "circular_adjacent_block", {"people": people})

                multiset_word = re.search(
                    r"(?:word|单词)\s+([A-Z]{4,30})[^.。;；\n]{0,40}(?:distinct|different|不同).*?"
                    r"(?:arrangements?|排列)",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if multiset_word:
                    word = multiset_word.group(1).upper()
                    counts = sorted({word.count(letter) for letter in set(word)})
                    # Preserve multiplicities as a sorted list; the word itself
                    # is shown by the parent prompt, while the fact exposes the
                    # exact count vector used by the closed formula.
                    append("combinatorics", "multiset_permutations", {"counts": sorted([word.count(letter) for letter in set(word)])})

                polygon = re.search(
                    r"(?:convex\s+)?(\d{1,4})[- ]?(?:gon|边形)[^.。;；\n]{0,70}"
                    r"(?:triangulations?|三角剖分)",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if polygon:
                    append("combinatorics", "triangulations", {"vertices": int(polygon.group(1))})
                polygon_zh = re.search(r"(?:凸)?八边形[^.。;；\n]{0,60}(?:三角剖分)", normalized)
                if polygon_zh:
                    append("combinatorics", "triangulations", {"vertices": 8})

                lower_bound = re.search(
                    r"(?:非负整数|nonnegative\s+integer)[^.。;；\n]{0,80}"
                    r"(?:x[_\{]?\s*1\s*\+\s*x[_\{]?\s*2|sum|和)[^=]{0,12}(\d{1,5})"
                    r"[^.。;；\n]{0,80}x[_\{]?\s*1\s*(?:>=|≥)\s*(\d{1,5})",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if lower_bound:
                    variables = len(set(re.findall(r"x[_\{]?\s*\d+", normalized, flags=re.IGNORECASE)))
                    if variables >= 2:
                        append("combinatorics", "one_coordinate_lower", {"total": int(lower_bound.group(1)), "variables": variables, "lower": int(lower_bound.group(2))})
                upper_bound = re.search(
                    r"(?:非负整数|nonnegative\s+integer)[^.。;；\n]{0,120}"
                    r"(?:和为|sum\s*(?:is|=))\s*(\d{1,5})[^.。;；\n]{0,80}"
                    r"x[_\{]?\s*1\s*(?:<=|≤)\s*(\d{1,5})",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if upper_bound:
                    variables = len(set(re.findall(r"x[_\{]?\s*\d+", normalized, flags=re.IGNORECASE)))
                    if variables >= 2:
                        append("combinatorics", "one_coordinate_upper", {"total": int(upper_bound.group(1)), "variables": variables, "upper": int(upper_bound.group(2))})

                parenthesization = re.search(
                    rf"(?:product\s+of\s+|multiplying\s+)({number_word_pattern})\s+factors?"
                    r"[^;；\n]{0,180}(?:parenthesizations?|parenthesization)",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if parenthesization:
                    factors = self._tool_augmented_parse_small_integer(parenthesization.group(1))
                    if factors is not None:
                        append("combinatorics", "full_parenthesizations", {"factors": factors})

                diagonal_paths = re.search(
                    r"(?:from\s*\(\s*0\s*,\s*0\s*\)\s*to\s*\(\s*"
                    r"(\d{1,4})\s*,\s*\1\s*\)|(?:格路径|lattice\s+paths?)[^.。;；\n]{0,100}"
                    r"(?:y\s*=\s*x|diagonal))",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if diagonal_paths:
                    steps = int(diagonal_paths.group(1)) if diagonal_paths.group(1) else None
                    if steps is not None:
                        append("combinatorics", "ballot_diagonal_paths", {"steps": steps})
                diagonal_zh = re.search(
                    r"从\s*\(\s*0\s*,\s*0\s*\)\s*走到\s*\(\s*(\d{1,4})\s*,\s*\1\s*\)"
                    r"[^.。;；\n]{0,120}(?:不越过|上方|y\s*=\s*x)",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if diagonal_zh:
                    append("combinatorics", "ballot_diagonal_paths", {"steps": int(diagonal_zh.group(1))})

                composition = re.search(
                    r"compositions?\s+of\s+(\d{1,4})\s+into\s+(?:exactly\s+)?(\d{1,3})\s+"
                    r"(?:positive\s+)?(?:integer\s+)?parts?|"
                    r"(?:和为|表示)[^0-9]{0,16}(\d{1,4})[^0-9]{0,40}(?:恰好|正整数)[^0-9]{0,12}(\d{1,3})",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if composition:
                    total = composition.group(1) or composition.group(3)
                    parts = composition.group(2) or composition.group(4)
                    append("combinatorics", "composition_positive", {"total": int(total), "parts": int(parts)})

                onto = re.search(
                    r"(?:surjection|surjective|onto\s+function|满射)[^.。;；\n]*?"
                    r"(?:from|of|含|个)[^0-9]{0,10}(\d{1,3})[^0-9]{0,45}(\d{1,3})",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if onto:
                    append(
                        "combinatorics",
                        "onto_functions",
                        {"domain_size": int(onto.group(1)), "codomain_size": int(onto.group(2))},
                    )

                derangement = re.search(
                    rf"({number_word_pattern})\s+(?:people|elements?|objects?)"
                    r"[^;；\n]{0,180}(?:nobody|no\s+element|没有任何元素)[^;；\n]*"
                    r"(?:own|matching|原位置|自己的位置)|"
                    rf"({number_word_pattern})\s*(?:个)?(?:元素)?[^.。;；\n]{{0,25}}错排",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if derangement:
                    value = derangement.group(1) or derangement.group(2)
                    n_value = self._tool_augmented_parse_small_integer(value)
                    if n_value is not None:
                        append("combinatorics", "derangement", {"n": n_value})

                onto_explicit = re.search(
                    rf"(?:from|从)[^.。;；\n]{{0,30}}({number_word_pattern})[^.。;；\n]{{0,50}}"
                    rf"(?:to|到)[^.。;；\n]{{0,30}}({number_word_pattern})[^.。;；\n]{{0,45}}"
                    r"(?:surjective|surjection|onto\s+function|满射)",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if onto_explicit:
                    domain = self._tool_augmented_parse_small_integer(onto_explicit.group(1))
                    codomain = self._tool_augmented_parse_small_integer(onto_explicit.group(2))
                    if domain is not None and codomain is not None:
                        append("combinatorics", "onto_functions", {"domain_size": domain, "codomain_size": codomain})

                team_assignment = re.search(
                    r"({nw})\s+distinct\s+students?[^.。;；\n]{{0,80}}"
                    r"({nw})\s+labeled\s+(?:project\s+)?teams?[^.。;；\n]{{0,80}}"
                    r"(?:every\s+team|required\s+to\s+receive|at\s+least\s+one)".format(nw=number_word_pattern),
                    normalized,
                    flags=re.IGNORECASE,
                )
                if team_assignment:
                    domain = self._tool_augmented_parse_small_integer(team_assignment.group(1))
                    codomain = self._tool_augmented_parse_small_integer(team_assignment.group(2))
                    if domain is not None and codomain is not None:
                        append("combinatorics", "onto_functions", {"domain_size": domain, "codomain_size": codomain})

                # A standard-deck question is a hypergeometric subcalculation;
                # all four population parameters are fixed by the statement.
                cards = re.search(
                    r"(?:standard\s+52[- ]card|52\s+张(?:标准)?扑克牌|标准\s*52\s*张扑克牌)[^.。;；\n]{0,120}"
                    r"(?:choose|选)\s*(\d{1,3})[^.。;；\n]{0,80}"
                    r"(?:exactly|恰好)(?:\s*有)?\s*(\d{1,3})\s*(?:张\s*)?(?:hearts?|红桃)",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if cards:
                    append(
                        "probability",
                        "hypergeometric_pmf",
                        {
                            "population": 52,
                            "success_states": 13,
                            "draws": int(cards.group(1)),
                            "successes": int(cards.group(2)),
                        },
                    )

                graph_cycle = re.search(r"(?:cycle|圈|C)\s*[_\{]?\s*(\d{1,6})", normalized, flags=re.IGNORECASE)
                if graph_cycle:
                    n = int(graph_cycle.group(1))
                    if re.search(r"chromatic|色数|染色数", normalized, flags=re.IGNORECASE):
                        append("graph", "cycle_chromatic", {"n": n})
                    if re.search(r"matching|匹配", normalized, flags=re.IGNORECASE):
                        append("graph", "cycle_matching", {"vertices": n})
                graph_cycle_alt = re.search(r"(?:圈|cycle)\s*C\s*_?\s*\{?\s*(\d{1,6})", normalized, flags=re.IGNORECASE)
                if graph_cycle_alt and re.search(r"matching|匹配", normalized, flags=re.IGNORECASE):
                    append("graph", "cycle_matching", {"vertices": int(graph_cycle_alt.group(1))})

                complete_bipartite = re.search(
                    r"K\s*_?\s*\{?\s*(\d{1,6})\s*[,，]\s*(\d{1,6})\s*\}?",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if complete_bipartite:
                    m, n = int(complete_bipartite.group(1)), int(complete_bipartite.group(2))
                    if re.search(r"chromatic|色数|染色数", normalized, flags=re.IGNORECASE):
                        append("graph", "complete_bipartite_chromatic", {"m": m, "n": n})
                    elif re.search(r"edges?|边", normalized, flags=re.IGNORECASE):
                        append("graph", "complete_bipartite_edges", {"m": m, "n": n})

                complete_graph = re.search(r"\bK\s*_?\s*\{?\s*(\d{1,6})\s*\}?", normalized, flags=re.IGNORECASE)
                if complete_graph and not complete_bipartite:
                    n = int(complete_graph.group(1))
                    if re.search(r"chromatic|色数|染色数", normalized, flags=re.IGNORECASE):
                        append("graph", "complete_graph_chromatic", {"n": n})
                    elif re.search(r"edges?|边", normalized, flags=re.IGNORECASE):
                        append("graph", "complete_graph_edges", {"n": n})

                path_graph = re.search(r"(?:path|路径)\s*(?:graph\s*)?P\s*[_\{]?\s*(\d{1,6})", normalized, flags=re.IGNORECASE)
                if path_graph is None:
                    path_graph = re.search(r"路径图\s*P\s*_?\s*\{?\s*(\d{1,6})", normalized, flags=re.IGNORECASE)
                if path_graph and re.search(r"matching|匹配", normalized, flags=re.IGNORECASE):
                    append("graph", "path_matching", {"vertices": int(path_graph.group(1))})

                degree_sequence = re.search(
                    r"degree\s+sequence\s*\(?\s*((?:\d{1,3}\s*[,，]\s*)+\d{1,3})\s*\)?|"
                    r"度数序列为?\s*[（(]?\s*((?:\d{1,3}\s*[,，]\s*)+\d{1,3})",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if degree_sequence and re.search(r"edges?|条边", normalized, flags=re.IGNORECASE):
                    raw_degrees = degree_sequence.group(1) or degree_sequence.group(2)
                    degrees = [int(item) for item in re.findall(r"\d+", raw_degrees)]
                    append("graph", "edge_count_from_degrees", {"degrees": degrees})

                regular_graph = re.search(
                    r"(?:a\s+)?(\d{1,6})[- ]regular\s+simple\s+graph\s+has\s+(\d{1,6})\s+vertices|"
                    r"(\d{1,6})\s*[-－]?regular[^.。;；\n]{0,30}(\d{1,6})\s*(?:个)?顶点",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if regular_graph and re.search(r"edges?|条边", normalized, flags=re.IGNORECASE):
                    degree = regular_graph.group(1) or regular_graph.group(3)
                    vertices = regular_graph.group(2) or regular_graph.group(4)
                    append("graph", "regular_graph_edges", {"vertices": int(vertices), "degree": int(degree)})

                planar = re.search(
                    r"(?:planar|平面图)[^.。;；\n]{0,80}(\d{1,6})\s*(?:vertices|顶点)[^.。;；\n]{0,40}(\d{1,8})\s*(?:edges|条边)",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if planar and re.search(r"faces?|面", normalized, flags=re.IGNORECASE):
                    append(
                        "graph",
                        "planar_faces",
                        {"vertices": int(planar.group(1)), "edges": int(planar.group(2)), "components": 1},
                    )

                turan = re.search(
                    r"(?:triangle[- ]free|无三角形)[^.。;；\n]{0,30}(?:graph|图)[^.。;；\n]{0,30}(?:on|with|有)?\s*(\d{1,6})\s*(?:vertices|顶点)",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if turan:
                    append("graph", "turan_edges", {"vertices": int(turan.group(1)), "forbidden_clique": 3})
                turan_alt = re.search(r"(\d{1,6})\s*个?\s*顶点[^.。;；\n]{0,30}无三角形简单图[^.。;；\n]{0,50}(?:最多|maximum|max)", normalized, flags=re.IGNORECASE)
                if turan_alt:
                    append("graph", "turan_edges", {"vertices": int(turan_alt.group(1)), "forbidden_clique": 3})

                tree_degree = re.search(
                    rf"(?:tree|树)[^.。;；\n]{{0,80}}(\d{{1,6}})\s+(?:vertices|顶点)[^.。;；\n]{{0,100}}"
                    rf"({number_word_pattern}|\d{{1,6}})\s+(?:have\s+)?degree\s+1[^.。;；\n]{{0,45}}"
                    rf"({number_word_pattern}|\d{{1,6}})\s+(?:have\s+)?degree\s+2",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if tree_degree and re.search(r"remaining\s+degree|剩余度数", normalized, flags=re.IGNORECASE):
                    leaves = self._tool_augmented_parse_small_integer(tree_degree.group(2))
                    degree_two = self._tool_augmented_parse_small_integer(tree_degree.group(3))
                    if leaves is None or degree_two is None:
                        continue
                    append(
                        "graph",
                        "tree_remaining_degree",
                        {
                            "vertices": int(tree_degree.group(1)),
                            "leaves": leaves,
                            "degree_two": degree_two,
                        },
                    )

                tree_vertices = re.search(r"(?:tree|树)[^.。;；\n]{0,35}(\d{1,6})\s*(?:个)?\s*(?:vertices|顶点)", normalized, flags=re.IGNORECASE)
                if tree_vertices and re.search(r"how many edges|多少条边|edge count", normalized, flags=re.IGNORECASE):
                    append("graph", "tree_edge_count", {"vertices": int(tree_vertices.group(1))})
                cut_components = re.search(r"(?:tree|树)[^.。;；\n]{0,80}(?:exactly|恰好)\s*(\d{1,6})\s+(?:connected components|连通分支)", normalized, flags=re.IGNORECASE)
                if cut_components:
                    append("graph", "tree_cut_edges", {"components": int(cut_components.group(1))})
                cut_components_zh = re.search(r"从一棵树[^.。;；\n]{0,80}(?:恰好得到|得到恰好)\s*(\d{1,6})\s*个?\s*连通分支", normalized)
                if cut_components_zh:
                    append("graph", "tree_cut_edges", {"components": int(cut_components_zh.group(1))})

                connected_guarantee = re.search(
                    r"(?:simple\s+graph\s+on|简单图)[^.。;；\n]{0,30}(\d{1,6})\s+(?:vertices|顶点)"
                    r"[^.。;；\n]{0,70}(?:guarantees?|保证)[^.。;；\n]{0,30}(?:connected|连通)",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if connected_guarantee:
                    append("graph", "connected_edge_guarantee", {"vertices": int(connected_guarantee.group(1))})
                connected_guarantee_alt = re.search(
                    r"(?:graph\s+on|图)[^0-9]{0,12}(\d{1,6})\s*(?:个)?\s*(?:vertices|顶点)"
                    r"[^.。;；\n]{0,120}(?:guarantees?|保证)[^.。;；\n]{0,40}(?:connected|连通)",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if connected_guarantee_alt:
                    append("graph", "connected_edge_guarantee", {"vertices": int(connected_guarantee_alt.group(1))})
                connected_guarantee_broad = re.search(
                    r"(?:smallest\s+number\s+of\s+edges|最少.*条边)[^.。;；\n]{0,120}(?:guarantees?|保证)[^.。;；\n]{0,100}"
                    r"(?:simple\s+graph\s+on|图)[^0-9]{0,12}(\d{1,6})\s*(?:个)?\s*(?:vertices|顶点)[^.。;；\n]{0,60}(?:connected|连通)",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if connected_guarantee_broad:
                    append("graph", "connected_edge_guarantee", {"vertices": int(connected_guarantee_broad.group(1))})

                euler_trail = re.search(
                    r"(?:connected|连通)[^.。;；\n]{0,80}(?:exactly\s+two|two|两个)\s+(?:vertices\s+of\s+)?odd\s+degree",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if euler_trail and re.search(r"euler\s+trail|欧拉迹", normalized, flags=re.IGNORECASE):
                    append("graph", "euler_trail_possible", {"connected": True, "odd_vertices": 2})
                euler_circuit = re.search(
                    r"(?:connected|连通)[^.。;；\n]{0,80}(?:every\s+vertex\s+of\s+even\s+degree|每个顶点.*偶数度)",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if euler_circuit and re.search(r"euler\s+circuit|欧拉回路", normalized, flags=re.IGNORECASE):
                    append("graph", "euler_circuit_possible", {"connected": True, "odd_vertices": 0})

                hamiltonian = re.search(
                    r"complete\s+bipartite\s+graph\s+K\s*_?\s*\{?\s*(\d{1,6})\s*[,，]\s*(\d{1,6})"
                    r"[^.。;；\n]{0,60}hamiltonian\s+cycle",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if hamiltonian:
                    append("graph", "complete_bipartite_hamiltonian", {"m": int(hamiltonian.group(1)), "n": int(hamiltonian.group(2))})

                hall = re.search(
                    r"(?:left\s+part\s+L|左部)[^.。;；\n]{0,30}(\d{1,6})\s+(?:vertices|顶点)"
                    r"[^.。;；\n]{0,100}(?:Hall|霍尔)[^.。;；\n]{0,80}(?:matching|匹配)",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if hall:
                    append("graph", "hall_matching_size", {"left_vertices": int(hall.group(1))})
                hall_alt = re.search(
                    r"(?:left\s+part\s+L|左部)[^.。;；\n]{0,35}(\d{1,6})\s*(?:个)?\s*(?:vertices|顶点)"
                    r"[^.。;；\n]{0,120}(?:Hall|霍尔)[^.。;；\n]{0,80}(?:matching|匹配)",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if hall_alt:
                    append("graph", "hall_matching_size", {"left_vertices": int(hall_alt.group(1))})
                hall_broad = re.search(
                    r"(?:left\s+part\s+L|左部)[^.。;；\n]{0,35}(\d{1,6})\s*(?:个)?\s*(?:vertices|顶点)"
                    r"[^.。;；\n]{0,180}(?:Hall(?:'s)?\s+condition|霍尔)",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if hall_broad:
                    append("graph", "hall_matching_size", {"left_vertices": int(hall_broad.group(1))})

                triangulated = re.search(
                    r"(?:connected\s+simple\s+planar\s+graph|连通简单平面图)[^.。;；\n]{0,20}"
                    r"(\d{1,6})\s+(?:vertices|顶点)[^.。;；\n]{0,80}(?:every\s+face\s+is\s+a\s+triangle|每个面都是三角形)",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if triangulated:
                    append("graph", "triangulated_planar_edges", {"vertices": int(triangulated.group(1))})
                triangulated_alt = re.search(
                    r"连通简单平面图[^0-9]{0,20}(\d{1,6})\s*个?\s*顶点[^.。;；\n]{0,80}每个面都是三角形",
                    normalized,
                )
                if triangulated_alt:
                    append("graph", "triangulated_planar_edges", {"vertices": int(triangulated_alt.group(1))})

                planar_faces_zh = re.search(
                    r"连通平面图[^0-9]{0,20}(\d{1,6})\s*个?\s*顶点[^0-9]{0,30}(\d{1,6})\s*条边[^;；\n]{0,50}多少个面",
                    normalized,
                )
                if planar_faces_zh:
                    append("graph", "planar_faces", {"vertices": int(planar_faces_zh.group(1)), "edges": int(planar_faces_zh.group(2)), "components": 1})

                matrix_literals = self._tool_augmented_matrix_literals(normalized)
                matrix_target = re.search(
                    r"\b(trace|inverse|eigenvalues?|eigenvectors?|rref|null\s*space|row\s*reduce|"
                    r"linear\s+solve|solve\s+(?:the\s+)?linear\s+system)\b|"
                    r"(迹|逆矩阵|特征值|特征向量|行最简|零空间|线性方程组)",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if matrix_target:
                    selected_matrix = matrix_literals[0][0] if len(matrix_literals) == 1 else None
                    if selected_matrix is None:
                        named = re.search(r"(?:trace|inverse|eigenvalues?|eigenvectors?|rref|rank|det)\s*\(\s*([A-Za-z]\w*)\s*\)", normalized, flags=re.IGNORECASE)
                        if named:
                            matches = [value for name, value in self._tool_augmented_named_matrix_bindings(source_problem) if name == named.group(1)]
                            if len(matches) == 1:
                                selected_matrix = matches[0]
                    if selected_matrix is not None:
                        operation_text = matrix_target.group(1) or matrix_target.group(2) or ""
                        matrix_operation = {
                            "trace": "trace", "迹": "trace", "inverse": "inverse", "逆矩阵": "inverse",
                            "eigenvalue": "eigenvalues", "eigenvalues": "eigenvalues", "特征值": "eigenvalues",
                            "eigenvector": "eigenvectors", "eigenvectors": "eigenvectors", "特征向量": "eigenvectors",
                            "rref": "rref", "row reduce": "rref", "行最简": "rref",
                            "null space": "nullspace", "零空间": "nullspace",
                            "linear solve": "linear_solve", "linear system": "linear_solve", "线性方程组": "linear_solve",
                        }.get(operation_text.lower(), "")
                        if matrix_operation and matrix_operation != "linear_solve":
                            append("matrix", matrix_operation, {"matrix": selected_matrix})

                if binomial_pmf is None:
                    binomial_explicit = re.search(
                        r"(?:X\s*~\s*)?binomial\s*\(\s*(?:n\s*=\s*)?(\d{1,3})\s*[,;]\s*"
                        r"(?:p\s*=\s*)?([0-9]+(?:/[0-9]+)?|0(?:\.\d+)?)\s*\)"
                        r"[^.。;；\n]{0,120}\b(?:p|pr)\s*\(\s*[a-z]\s*=\s*(\d{1,3})\s*\)",
                        normalized,
                        flags=re.IGNORECASE,
                    )
                    if binomial_explicit is not None:
                        append(
                            "probability",
                            "binomial_pmf",
                            {
                                "trials": int(binomial_explicit.group(1)),
                                "p": binomial_explicit.group(2),
                                "successes": int(binomial_explicit.group(3)),
                            },
                        )
                    else:
                        binomial_reverse = re.search(
                            r"\b(?:p|pr)\s*\(\s*[a-z]\s*=\s*(\d{1,3})\s*\)"
                            r"[^.。;；\n]{0,120}(?:X\s*~\s*)?binomial\s*\(\s*"
                            r"(?:n\s*=\s*)?(\d{1,3})\s*[,;]\s*"
                            r"(?:p\s*=\s*)?([0-9]+(?:/[0-9]+)?|0(?:\.\d+)?)\s*\)",
                            normalized,
                            flags=re.IGNORECASE,
                        )
                        if binomial_reverse is not None:
                            append(
                                "probability",
                                "binomial_pmf",
                                {
                                    "trials": int(binomial_reverse.group(2)),
                                    "p": binomial_reverse.group(3),
                                    "successes": int(binomial_reverse.group(1)),
                                },
                            )

                if poisson_pmf is None:
                    poisson_explicit = re.search(
                        r"(?:X\s*~\s*)?poisson\s*\(\s*(?:lambda|lam|rate)\s*=\s*"
                        r"([0-9]+(?:/[0-9]+)?|[0-9]+(?:\.\d+)?)\s*\)"
                        r"[^.。;；\n]{0,80}\b(?:p|pr)\s*\(\s*[a-z]\s*=\s*(\d{1,3})\s*\)",
                        normalized,
                        flags=re.IGNORECASE,
                    )
                    if poisson_explicit is not None:
                        append(
                            "probability",
                            "poisson_pmf",
                            {"rate": poisson_explicit.group(1), "k": int(poisson_explicit.group(2))},
                        )
                    else:
                        poisson_reverse = re.search(
                            r"\b(?:p|pr)\s*\(\s*[a-z]\s*=\s*(\d{1,3})\s*\)"
                            r"[^.。;；\n]{0,120}(?:X\s*~\s*)?poisson\s*\(\s*"
                            r"(?:lambda|lam|rate)\s*=\s*([0-9]+(?:/[0-9]+)?|[0-9]+(?:\.\d+)?)\s*\)",
                            normalized,
                            flags=re.IGNORECASE,
                        )
                        if poisson_reverse is not None:
                            append(
                                "probability",
                                "poisson_pmf",
                                {"rate": poisson_reverse.group(2), "k": int(poisson_reverse.group(1))},
                            )

                coefficient_alt = re.search(
                    r"coefficient\s+of\s+x\s*\^\s*\{?\s*(\d{1,3})\s*\}?\s+in\s+(.+?)(?:[.。;；!?]|$)|"
                    r"\[\s*x\s*\^\s*\{?\s*(\d{1,3})\s*\}?\s*\]\s*(?:of\s+)?(.+?)(?:[.。;；!?]|$)|"
                    r"(?:求|find)\s+生成函数?\s*(.+?)\s*(?:中|中的)\s*x\s*\^\s*\{?\s*(\d{1,3})\s*\}?\s*的?\s*系数?",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if coefficient_alt is not None:
                    coefficient = coefficient_alt
                else:
                    coefficient = re.search(
                        r"coefficient\s+of\s+(.+?)\s+in\s+[^.。;；\n]*?x\s*\^\s*\{?\s*(\d{1,3})\s*\}?|"
                        r"(?:求|find)\s+(.+?)\s*(?:中|中的|of)\s*x\s*\^\s*\{?\s*(\d{1,3})\s*\}?[^.。;；\n]*系数?",
                        normalized,
                        flags=re.IGNORECASE,
                    )
                if coefficient:
                    if coefficient_alt is not None and coefficient.group(1) and coefficient.group(2):
                        power, expression = coefficient.group(1), coefficient.group(2)
                    elif coefficient_alt is not None and coefficient.group(3) and coefficient.group(4):
                        power, expression = coefficient.group(3), coefficient.group(4)
                    elif coefficient_alt is not None and coefficient.group(5) and coefficient.group(6):
                        expression, power = coefficient.group(5), coefficient.group(6)
                    else:
                        expression = coefficient.group(1) or coefficient.group(3)
                        power = coefficient.group(2) or coefficient.group(4)
                    if (
                        expression
                        and "..." not in expression
                        and "…" not in expression
                        and not re.search(r"\b[A-Z]\s*\(", expression)
                    ):
                        append("sympy", "coefficient", {"expr": expression.strip(), "var": "x", "power": int(power)})

                derivative = re.search(
                    r"(?:derivative\s+of|differentiate)\s+(.+?)(?:\s+with\s+respect\s+to\s+([A-Za-z])|\s+at\s+[A-Za-z]\s*=|[.。;；!?]|$)|"
                    r"对\s*(.+?)\s*求导",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if derivative:
                    expression = derivative.group(1) or derivative.group(3)
                    variable = derivative.group(2) or "x"
                    if expression and not re.search(r"\b(?:the|function|with|respect|at)\b", expression, flags=re.IGNORECASE):
                        append("sympy", "differentiate", {"expr": expression.strip(), "var": variable})

                integral = re.search(
                    r"(?:integral|integrate)\s+(?:of\s+)?(.+?)\s+from\s+([+-]?\d+(?:/\d+|\.\d+)?)\s+to\s+([+-]?\d+(?:/\d+|\.\d+)?)|"
                    r"从\s*([+-]?\d+(?:/\d+|\.\d+)?)\s*到\s*([+-]?\d+(?:/\d+|\.\d+)?)\s*(.+?)\s*积分",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if integral:
                    if integral.group(1):
                        expression, lower, upper = integral.group(1), integral.group(2), integral.group(3)
                    else:
                        expression, lower, upper = integral.group(6), integral.group(4), integral.group(5)
                    if expression and "..." not in expression and "…" not in expression:
                        append("sympy", "integrate", {"expr": expression.strip(), "var": "x", "lower": lower, "upper": upper})

                limit = re.search(
                    r"limit\s+of\s+(.+?)\s+as\s+([A-Za-z])\s*(?:->|→)\s*([+-]?\d+(?:/\d+|\.\d+)?)|"
                    r"极限[^xX]*([A-Za-z])\s*(?:->|趋于)\s*([+-]?\d+(?:/\d+|\.\d+)?)[：:]?\s*(.+)",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if limit:
                    expression = limit.group(1) or limit.group(6)
                    variable = limit.group(2) or limit.group(4) or "x"
                    point = limit.group(3) or limit.group(5)
                    if expression and "..." not in expression and "…" not in expression:
                        append("sympy", "limit", {"expr": expression.strip(), "var": variable, "point": point})
        if self._tool_augmented_is_v307_cross_subject():
            coefficient_request = self._tool_augmented_coefficient_from_problem(source_problem)
            if coefficient_request is not None:
                append("sympy", "coefficient", coefficient_request)
        return result

    def _tool_augmented_parse_small_integer(self, value: Any) -> int | None:
        text = str(value or "").strip().lower()
        if text.isdigit():
            return int(text)
        return _SCORE_FIRST_SMALL_NUMBER_WORDS.get(text)

    def _tool_augmented_coefficient_from_problem(self, problem: str) -> Dict[str, Any] | None:
        """Bind one explicit generating-function coefficient from the source."""

        text = str(problem or "").strip()
        power_match = re.search(
            r"\[\s*x\s*\^\s*\{?\s*(\d{1,4})\s*\}?\s*\]"
            r"|(?:coefficient|系数)[^.。;；\n]{0,70}x\s*\^\s*\{?\s*(\d{1,4})\s*\}?"
            r"|x\s*\^\s*\{?\s*(\d{1,4})\s*\}?[^.。;；\n]{0,40}(?:coefficient|系数)",
            text,
            flags=re.IGNORECASE,
        )
        if power_match is None:
            return None
        power_text = next((group for group in power_match.groups() if group), None)
        if power_text is None:
            return None
        candidates: List[str] = []
        for pattern in (
            r"(?:generating\s+function|生成函数)\s*[:：]?\s*(.+?)(?:\s+counts|\s+what|\s+is|\s+has|\s+中|\s+中的|[。；;!?]|$)",
            r"\b[A-Z]\s*\(\s*x\s*\)\s*=\s*([^,。；;!?\n]+)",
            r"生成函数\s*([^。；;!?\n]+?)\s*(?:中|中的)",
            r"(?:coefficient|系数)[^.。;；\n]{0,50}\b(?:in|of)\s+([^。；;!?\n]+)",
        ):
            candidates.extend(match.group(1).strip() for match in re.finditer(pattern, text, flags=re.IGNORECASE))
        for candidate in candidates:
            candidate = candidate.strip().rstrip(".,")
            candidate = re.sub(r"\s+\b(?:what|is|the|coefficient|of|in)\b.*$", "", candidate, flags=re.IGNORECASE).strip()
            if not candidate or "..." in candidate or "…" in candidate:
                continue
            if re.search(r"\b[A-Z]\s*\(", candidate):
                continue
            return {"expr": candidate, "var": "x", "power": int(power_text)}
        geometric = re.search(
            r"\(\s*x\s*\+\s*x\s*\^\s*2\s*\+\s*x\s*\^\s*3\s*\+\s*(?:\.\.\.|…)+\s*\)\s*\^\s*(\d{1,3})",
            text,
            flags=re.IGNORECASE,
        )
        if geometric:
            exponent = int(geometric.group(1))
            return {"expr": f"(x/(1-x))**{exponent}", "var": "x", "power": int(power_text)}
        return None

    def _tool_augmented_recurrence_request(
        self,
        problem: str,
        request_span: str,
    ) -> Dict[str, Any] | None:
        """Bind one explicit scalar linear recurrence to a requested index."""

        target = re.search(
            r"\b([A-Za-z])\s*[_\{]?\s*(\d{1,4})\s*\}?",
            str(request_span or ""),
            flags=re.IGNORECASE,
        )
        if target is None:
            return None
        recurrence_matches = re.findall(
            r"([A-Za-z])\s*[_\{]?\s*n\s*\}?\s*=\s*(.+?)(?=(?:\s*(?:for|where|with)\s+n\b|[,，.;。；]|$))",
            str(problem or ""),
            flags=re.IGNORECASE,
        )
        if len(recurrence_matches) != 1:
            return None
        sequence_symbol, rhs = recurrence_matches[0]
        rhs = rhs.strip()
        lag_matches = list(
            re.finditer(
                r"(?P<sign>[+-]?)\s*(?P<coef>\d+(?:/\d+)?(?:\.\d+)?)?\s*\*?\s*"
                rf"{re.escape(sequence_symbol)}\s*(?:_\s*\{{\s*n\s*-\s*(?P<lag1>\d+)\s*\}}|\(\s*n\s*-\s*(?P<lag2>\d+)\s*\)|_\s*n\s*-\s*(?P<lag3>\d+))",
                rhs,
                flags=re.IGNORECASE,
            )
        )
        if not lag_matches:
            return None
        lags = [
            int(match.group("lag1") or match.group("lag2") or match.group("lag3"))
            for match in lag_matches
        ]
        order = max(lags)
        if len(set(lags)) != len(lags) or any(lag < 1 or lag > 8 for lag in lags):
            return None
        coefficients = ["0"] * order
        for match, lag in zip(lag_matches, lags):
            coefficient = match.group("coef") or "1"
            if match.group("sign") == "-":
                coefficient = "-" + coefficient
            elif match.group("sign") == "+":
                coefficient = "+" + coefficient
            coefficients[lag - 1] = coefficient
        remainder = rhs
        for match in lag_matches:
            remainder = remainder.replace(match.group(0), "", 1)
        remainder = re.sub(r"\s+", "", remainder)
        if remainder in {"", "+", "-"}:
            constant: int | str = 0
        elif re.fullmatch(r"[+-]?\d+(?:/\d+|\.\d+)?", remainder):
            constant = remainder
        elif (
            re.fullmatch(r"[+\-0-9nN*/^().·\s]+", remainder)
            and re.search(r"n", remainder, flags=re.IGNORECASE)
        ):
            # Permit only a small symbolic forcing grammar.  The worker still
            # reparses it through the safe SymPy expression parser and checks
            # that its only free symbol is n.
            constant = 0
            forcing = remainder.replace("·", "*").replace("^", "**")
        else:
            # An unparsed forcing term or nonlinear expression makes this
            # recognizer unsafe; let the full solver handle it.
            return None

        initial_start = 0
        if not re.search(
            rf"\b{re.escape(sequence_symbol)}\s*[_\{{]?\s*0\s*\}}?\s*=",
            str(problem or ""),
            flags=re.IGNORECASE,
        ) and re.search(
            rf"\b{re.escape(sequence_symbol)}\s*[_\{{]?\s*1\s*\}}?\s*=",
            str(problem or ""),
            flags=re.IGNORECASE,
        ):
            initial_start = 1
        initial: List[str] = []
        for index in range(initial_start, initial_start + order):
            match = re.search(
                rf"\b{re.escape(sequence_symbol)}\s*[_\{{]?\s*{index}\s*\}}?\s*=\s*([+-]?\d+(?:/\d+|\.\d+)?)",
                str(problem or ""),
                flags=re.IGNORECASE,
            )
            if match is None:
                return None
            initial.append(match.group(1))
        result = {
            "sequence_symbol": sequence_symbol,
            "initial": initial,
            "coefficients": coefficients,
            "initial_start": initial_start,
            "constant": constant,
            "target_n": int(target.group(2)),
        }
        if "forcing" in locals():
            result["forcing"] = forcing
        return result

    def _tool_augmented_merge_tool_requests(
        self,
        target_requests: List[Any],
        formalizer_requests: Any,
    ) -> List[Any]:
        """Stable-dedupe target-first requests under the executor cap."""

        merged: List[Any] = []
        seen: set[tuple[str, str, str]] = set()
        for request in list(target_requests or []) + list(formalizer_requests or []):
            family = str(getattr(request, "family", "") or "")
            operation = str(getattr(request, "operation", "") or "")
            arguments = getattr(request, "arguments", {})
            try:
                key = (family, operation, repr(arguments))
            except Exception:
                key = (family, operation, str(id(request)))
            if key in seen:
                continue
            seen.add(key)
            merged.append(request)
            if len(merged) >= 6:
                break
        return merged

    def _tool_augmented_equation_requests_from_span(
        self,
        span: str,
        problem: str,
    ) -> tuple[List[Dict[str, Any]] | None, bool]:
        text = str(span or "").strip()
        if not self._tool_augmented_equation_intent(text):
            return [], False

        equations = self._tool_augmented_equation_literals(text)
        if len(equations) != 1:
            return None, True
        equation = equations[0]
        left, right = (part.strip() for part in equation.split("=", 1))
        try:
            from math_agent_core.tools.sympy_tool import _parse_expr, _infer_solution_domain

            expression = _parse_expr(left) - _parse_expr(right)
            symbols = sorted({str(symbol) for symbol in expression.free_symbols})
        except Exception:
            return None, True

        explicit = re.search(
            r"\bsolve\s+(?:the\s+)?(?:equation\s+)?(?:.+?\s+)?for\s+([A-Za-z])"
            r"(?=\s*(?:[.,;:!?]|$))",
            text,
            flags=re.IGNORECASE,
        )
        if explicit:
            variable = explicit.group(1)
            if variable not in symbols:
                return None, True
        elif len(symbols) == 1:
            variable = symbols[0]
        else:
            return None, True

        try:
            domain = _infer_solution_domain(problem)
        except Exception:
            domain = "complex"
        requests = [
            {"kind": "equation_solution", "equation": equation, "variable": variable},
        ]
        # An unknown/rational/algebraic restriction cannot be represented by
        # the closed verification enum without risking a false solution-set
        # claim.  Retain the point-solution check, but fail closed only for the
        # domain-sensitive set check.
        if domain in {
            "complex",
            "real",
            "positive_real",
            "nonnegative_real",
            "positive_integer",
            "nonnegative_integer",
            "integer",
        }:
            requests.append(
                {
                    "kind": "equation_solution_set",
                    "equation": equation,
                    "variable": variable,
                    "domain": domain,
                }
            )
        return requests, True

    def _tool_augmented_equation_intent(self, span: str) -> bool:
        text = str(span or "").strip()
        lower = text.lower()
        if re.match(r"^(?:please\s+)?solve\b", lower):
            # ``solve for x^4`` and ``solve for f(x)`` are expression requests,
            # not bare-variable equation targets.
            solve_for = re.search(r"\bsolve\s+for\s+(.+?)(?:\s+given\b|\s+if\b|[.,;:!?]|$)", lower)
            if solve_for and not re.fullmatch(r"[a-z]", solve_for.group(1).strip()):
                return False
            return True
        if re.search(
            r"\b(?:find|determine|list|give|identify)\s+(?:all\s+)?"
            r"(?:real\s+|complex\s+)?(?:roots?|solutions?)\b",
            lower,
        ):
            return True
        # Require a standalone variable token.  A following ^, _, (, [, or
        # identifier continuation therefore cannot be mistaken for bare x.
        return bool(
            re.search(
                r"\b(?:find|determine)\s+(?:the\s+value\s+of\s+)?"
                r"([A-Za-z])(?=\s*(?:[.,;:!?]|$|\b(?:in|where|from)\b))",
                text,
                flags=re.IGNORECASE,
            )
        )

    def _tool_augmented_equation_literals(self, span: str) -> List[str]:
        pattern = re.compile(
            r"(?<![A-Za-z0-9_])"
            r"([A-Za-z0-9_().]+(?:\s*[+\-*/^]\s*[A-Za-z0-9_().]+)*)"
            r"\s*=\s*"
            r"([A-Za-z0-9_().]+(?:\s*[+\-*/^]\s*[A-Za-z0-9_().]+)*)"
        )
        equations: List[str] = []
        for match in pattern.finditer(str(span or "")):
            left = match.group(1).strip()
            right = match.group(2).strip().rstrip(".,;:!?。")
            candidate = f"{left}={right}"
            if candidate not in equations:
                equations.append(candidate)
        return equations

    def _tool_augmented_matrix_requests_from_span(
        self,
        span: str,
        problem: str,
    ) -> tuple[List[Dict[str, Any]] | None, bool]:
        text = str(span or "").strip()
        lower = text.lower()
        target_phrase = re.split(
            r"\b(?:with|whose|having|where|given)\b",
            lower,
            maxsplit=1,
        )[0]
        operations: List[str] = []
        if re.search(r"\bdeterminant\b|\bdet\s*\(", target_phrase):
            operations.append("matrix_determinant")
        if re.search(r"(?<!full[- ])\brank\b", target_phrase):
            operations.append("matrix_rank")
        if not operations:
            return [], False

        literals = self._tool_augmented_matrix_literals(text)
        selected: Any = None
        if len(literals) == 1:
            selected = literals[0][0]
        elif len(literals) > 1:
            # More than one literal in the request is safe only when the
            # operation names one of the literals unambiguously.
            named = re.search(r"(?:det|determinant|rank)\s*\(\s*([A-Za-z]\w*)\s*\)", text, re.IGNORECASE)
            if named:
                matches = [value for value, name in literals if name == named.group(1)]
                if len(matches) == 1:
                    selected = matches[0]
            if selected is None:
                return None, True
        else:
            named = re.search(
                r"(?:det|determinant|rank)\s*\(\s*([A-Za-z]\w*)\s*\)|"
                r"(?:determinant|rank)\s+of\s+([A-Za-z]\w*)",
                text,
                re.IGNORECASE,
            )
            candidates = self._tool_augmented_named_matrix_bindings(problem)
            if named:
                name = named.group(1) or named.group(2)
                matches = [value for bound_name, value in candidates if bound_name == name]
                if len(matches) == 1:
                    selected = matches[0]
            elif re.search(r"\bits\b|\bthe matrix\b", lower):
                unique = [value for value, _name in self._tool_augmented_matrix_literals(problem)]
                if len(unique) == 1:
                    selected = unique[0]
            if selected is None:
                return None, True

        requests = [{"kind": operation, "matrix": selected} for operation in operations]
        return requests, True

    def _tool_augmented_matrix_literals(self, text: str) -> List[tuple[Any, str | None]]:
        literals: List[tuple[Any, str | None]] = []
        cursor = 0
        source = str(text or "")
        while True:
            start = source.find("[[", cursor)
            if start < 0:
                break
            depth = 0
            end = None
            for index in range(start, len(source)):
                if source[index] == "[":
                    depth += 1
                elif source[index] == "]":
                    depth -= 1
                    if depth == 0:
                        end = index + 1
                        break
            if end is None:
                break
            raw = source[start:end]
            try:
                value = ast.literal_eval(raw)
            except Exception:
                value = None
            if value is not None:
                try:
                    from math_agent_core.tools.augmented_tool import _normalize_verification_matrix

                    normalized = _normalize_verification_matrix(value)
                except Exception:
                    normalized = None
                if normalized is not None:
                    prefix = source[max(0, start - 64):start]
                    match = re.search(r"([A-Za-z]\w*)\s*=\s*$", prefix)
                    literals.append((normalized, match.group(1) if match else None))
            cursor = end
        return literals

    def _tool_augmented_named_matrix_bindings(self, problem: str) -> List[tuple[str, Any]]:
        bindings: List[tuple[str, Any]] = []
        source = str(problem or "")
        for value, name in self._tool_augmented_matrix_literals(source):
            if name:
                bindings.append((name, value))
        return bindings

    def _tool_augmented_arithmetic_request_from_span(
        self,
        span: str,
    ) -> tuple[Dict[str, Any] | None, bool]:
        match = re.match(
            r"^(?:please\s+)?(?:compute|calculate|evaluate)\s+(.+)$",
            str(span or "").strip(),
            flags=re.IGNORECASE,
        )
        if not match:
            return None, False
        expression = match.group(1).strip().rstrip(" \t.;:!?。")
        if not expression or re.search(r"[A-Za-z_]", expression):
            return None, False
        if not re.fullmatch(r"[0-9()+\-*/^\.\s]+", expression) or not re.search(r"[+\-*/^]", expression):
            return None, True
        return {"kind": "pure_arithmetic", "expression": expression}, True

    def _tool_augmented_matrix_targets_from_span(self, span: str) -> List[str]:
        """Read only the requested matrix noun phrase, not its qualifiers."""

        text = str(span or "").strip().lower()
        request = re.match(
            r"^(?:please\s+)?(?:compute|calculate|evaluate|find|determine|give|state)\s+(.+)$"
            r"|^(?:what\s+is|what's)\s+(.+)$",
            text,
        )
        if not request:
            return []
        requested_object = next((group for group in request.groups() if group), "")
        # A matrix's determinant/rank after one of these relations is a
        # descriptive hypothesis ("full-rank matrix", "matrix with
        # determinant -2"), not the object being requested.
        target_phrase = re.split(
            r"\b(?:of|for|from|in|with|where|given|having|whose)\b",
            requested_object,
            maxsplit=1,
        )[0]
        targets: List[str] = []
        if re.search(r"\bdeterminant\b|\bdet\s*\(", target_phrase):
            targets.append("matrix_determinant")
        if re.search(r"\brank\b", target_phrase):
            targets.append("matrix_rank")
        return targets

    def _tool_augmented_span_is_explicit_equation_target(self, span: str) -> bool:
        text = str(span or "").strip()
        lower = text.lower()
        solve_request = bool(
            re.match(r"^(?:please\s+)?solve\b", lower)
            and (
                "=" in text
                or "equation" in lower
                or re.search(r"\bsolve\s+for\s+[a-z]\b", lower)
            )
        )
        roots_request = bool(
            re.search(
                r"\b(?:find|determine|list|give|identify)\s+"
                r"(?:all\s+)?(?:real\s+|complex\s+)?(?:roots?|solutions?)\b",
                lower,
            )
        )
        # A bare "find x" is a request for the equation variable.  Exponents,
        # function notation, assignments, and expressions such as x^4 are not.
        variable_request = bool(
            re.fullmatch(
                r"(?:please\s+)?(?:find|determine)\s+"
                r"(?:the\s+)?(?:value\s+of\s+)?[a-z]\s*[.?!]?",
                lower,
            )
        )
        return solve_request or roots_request or variable_request

    def _tool_augmented_span_is_pure_arithmetic(self, span: str) -> bool:
        text = str(span or "").strip()
        match = re.match(r"^(?:please\s+)?(?:compute|calculate|evaluate)\s+(.+)$", text, flags=re.IGNORECASE)
        if not match:
            return False
        expression = match.group(1).strip().rstrip(" \t.;:!?。")
        if not expression or re.search(r"[A-Za-z_]", expression):
            return False
        return bool(
            re.fullmatch(r"[0-9()+\-*/^.\s]+", expression)
            and re.search(r"[+\-*/^]", expression)
        )

    def _tool_augmented_protocol_parse_rejected(
        self,
        formalizer_raw: str,
        protocol: Any,
    ) -> bool:
        return bool(
            isinstance(formalizer_raw, str)
            and re.search(r"(?im)^\s*tool\s*:", formalizer_raw)
            and not getattr(protocol, "tool_requests", [])
        )

    def _tool_augmented_trace_operations(self, items: Any) -> str:
        operations: List[str] = []
        if not isinstance(items, (list, tuple)):
            return "none"
        for item in items:
            family = str(getattr(item, "family", "") or "").lower()
            operation = str(getattr(item, "operation", "") or "").lower()
            if not re.fullmatch(r"[a-z_]{2,32}", family):
                continue
            if not re.fullmatch(r"[a-z_]{2,32}", operation):
                continue
            name = f"{family}:{operation}"
            if name not in operations:
                operations.append(name)
            if len(operations) >= 8:
                break
        return ",".join(operations) or "none"

    def _tool_augmented_trace_method(self, value: Any) -> str:
        method = re.sub(r"[^a-z0-9_:-]", "", str(value or "").lower())[:80]
        return method

    def _tool_augmented_failure_text(self, evidence: Any) -> str:
        if isinstance(evidence, dict):
            method = str(evidence.get("method") or "deterministic_check")
            details = str(evidence.get("details") or "").strip()
            residual = evidence.get("residual")
        else:
            method = str(getattr(evidence, "method", "") or "deterministic_check")
            details = str(getattr(evidence, "details", "") or "").strip()
            residual = getattr(evidence, "residual", None)
        text = f"{method}: {details}"
        if residual not in {None, ""}:
            text += f" residual={str(residual)[:240]}"
        return text[:520]

    def _chat_with_settings(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float,
        top_p: float | None,
        max_tokens: int,
        thinking_mode: bool,
    ) -> Any:
        kwargs: Dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if top_p is not None and self._client_supports_parameter("top_p"):
            kwargs["top_p"] = top_p
        if self._client_supports_parameter("thinking_mode"):
            kwargs["thinking_mode"] = thinking_mode
        response = self.client.chat(**kwargs)
        # Providers differ in whether completion metadata is returned beside
        # content, on the SDK response object, or on the client wrapper.  Keep
        # only a small normalized subset for truncation recovery.
        metadata: Dict[str, Any] = {}
        try:
            from math_agent_core.output_guard import finish_reason

            reason = finish_reason(response)
        except Exception:
            reason = ""
        if reason:
            metadata["finish_reason"] = reason
        client_reason = getattr(self.client, "last_finish_reason", None)
        if client_reason:
            metadata["finish_reason"] = str(client_reason).strip().lower()
        client_metadata = getattr(self.client, "last_response_metadata", None)
        if isinstance(client_metadata, dict):
            client_reason = client_metadata.get("finish_reason")
            if client_reason:
                metadata["finish_reason"] = str(client_reason).strip().lower()
        self._last_response_metadata = metadata
        return response

    def _output_is_likely_truncated(
        self,
        raw_response: Any,
        extracted_response: str,
        response_mode: str,
    ) -> bool:
        """Use provider metadata plus conservative surface checks.

        A complete answer-first line intentionally wins over a cut-off
        explanation.  This preserves the one-call path for ordinary answer
        items while giving proof-like responses a chance to recover from a
        hard completion stop.
        """

        try:
            raw_text = response_content(raw_response)
            return looks_truncated(
                raw_text or extracted_response,
                response_mode=response_mode,
                raw_response={
                    **self._last_response_metadata,
                    "provider_response": raw_response,
                },
            )
        except Exception:
            return not bool(str(extracted_response or "").strip())

    def _build_score_first_recovery_prompt(
        self,
        problem: str,
        context: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """Ask for one complete replacement after an incomplete response.

        The partial answer is deliberately not echoed: echoing a large partial
        completion can consume the very context needed to finish the problem
        and can cause the model to continue a broken derivation.  The original
        problem and the established strategy/check cards remain the source of
        truth.
        """

        base = self._build_score_first_prompt(problem, {}, context=context)
        response_mode = str(context.get("response_mode") or _SCORE_FIRST_RESPONSE_MODE_ANSWER)
        if response_mode == _SCORE_FIRST_RESPONSE_MODE_ANSWER:
            instruction = (
                "The previous response was incomplete. Re-solve the ORIGINAL problem and output one complete "
                "answer-first response. Put the complete requested answer on the first visible line as "
                "Final answer: <complete requested answer>; then stop."
            )
        else:
            instruction = (
                "The previous response was incomplete. Re-solve the ORIGINAL problem from the beginning and return "
                "one complete, self-contained response. State the conclusion/result first, then include every "
                "necessary requested proof or derivation step. Do not mention this recovery instruction."
            )
        return [
            {
                "role": "system",
                "content": f"{base[0]['content']}\n{instruction}",
            },
            {
                "role": "user",
                "content": f"{base[1]['content']}\n\n{instruction}",
            },
        ]

    def _solve_score_first(self, problem: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        context = self._score_first_prompt_context(problem, metadata)
        response = self._score_first_model_call(problem, metadata, context=context)
        raw_output = self._normalize_model_response(response)
        final_response = self._extract_score_first_response(
            raw_output,
            problem,
            response_mode=context["response_mode"],
        )
        output_truncated = self._output_is_likely_truncated(
            raw_output,
            final_response,
            context["response_mode"],
        )
        recovery_call_used = False
        recovery_kept = False
        recovery_error = ""
        if output_truncated:
            recovery_call_used = True
            try:
                recovery_messages = self._build_score_first_recovery_prompt(
                    problem,
                    context,
                )
                recovery_response_obj = self._chat_with_settings(
                    recovery_messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    thinking_mode=self.thinking_mode,
                )
                recovery_raw = self._normalize_model_response(recovery_response_obj)
                recovery_response = self._extract_score_first_response(
                    recovery_raw,
                    problem,
                    response_mode=context["response_mode"],
                )
                if recovery_response != DEFAULT_FALLBACK and not self._output_is_likely_truncated(
                    recovery_raw,
                    recovery_response,
                    context["response_mode"],
                ):
                    final_response = recovery_response
                    recovery_kept = True
            except Exception as exc:
                # Recovery is optional.  Never discard a non-empty primary
                # answer merely because the second call failed.
                recovery_error = type(exc).__name__
        subject_hint = context.get("subject_hint") or ""
        trace = [
            make_trace_step(
                "mode",
                (
                    "score_first; "
                    f"experiment_preset={self.score_first_experiment_preset}; "
                    f"thinking_mode={str(self.thinking_mode).lower()}"
                ),
            ),
            make_trace_step("prompt_profile", self.score_first_prompt_profile),
            make_trace_step("model_call", f"primary client.chat call: {1 + int(recovery_call_used)}"),
            make_trace_step("answer", "answer extracted" if final_response != DEFAULT_FALLBACK else "empty/unusable answer"),
        ]
        if output_truncated or recovery_call_used:
            trace.extend(
                [
                    make_trace_step("output_truncated_detected", str(output_truncated).lower()),
                    make_trace_step("recovery_call_used", str(recovery_call_used).lower()),
                    make_trace_step("recovery_kept", str(recovery_kept).lower()),
                ]
            )
        if subject_hint:
            trace.insert(2, make_trace_step("subject_hint", subject_hint))
        if recovery_error:
            trace.append(make_trace_step("recovery_fallback", recovery_error))
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
        return self._chat_with_settings(
            messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            thinking_mode=self.thinking_mode,
        )

    def _client_supports_parameter(self, name: str) -> bool:
        try:
            signature = inspect.signature(self.client.chat)
        except (TypeError, ValueError):
            return True
        if name in signature.parameters:
            return True
        return any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

    def _client_supports_thinking_mode(self) -> bool:
        return self._client_supports_parameter("thinking_mode")

    def _score_first_prompt_context(
        self,
        problem: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self.score_first_prompt_profile == "full":
            return self._score_first_context(problem, metadata)
        return self._score_first_minimal_context(problem, metadata)

    def _score_first_minimal_context(
        self,
        problem: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        trusted_domain, trusted_key = self._trusted_score_first_domain(metadata)
        request_records = self._score_first_request_span_records(problem)
        request_spans = [record["text"] for record in request_records]
        requested_actions = self._score_first_requested_actions(request_spans)
        response_mode = self._score_first_response_mode(
            problem,
            metadata,
            {},
            request_spans=request_spans,
        )
        subject_hint = self._trusted_score_first_subject_hint(
            metadata,
            trusted_domain=trusted_domain,
            trusted_key=trusted_key,
        )
        if self.score_first_prompt_profile == "ultra_minimal":
            subject_hint = ""
        return {
            "response_mode": response_mode,
            "request_spans": request_spans,
            "requested_actions": requested_actions,
            "subject_hint": subject_hint,
            "trusted_domain": trusted_domain,
            "trusted_key": trusted_key,
            "prompt_profile": self.score_first_prompt_profile,
        }

    def _build_score_first_prompt(
        self,
        problem: str,
        metadata: Dict[str, Any],
        context: Dict[str, Any] | None = None,
    ) -> List[Dict[str, str]]:
        context = context or self._score_first_prompt_context(problem, metadata)
        if self.score_first_prompt_profile in {"minimal", "ultra_minimal"}:
            return self._build_minimal_score_first_prompt(problem, context)

        return self._build_full_score_first_prompt(problem, context)

    def _build_full_score_first_prompt(
        self,
        problem: str,
        context: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """Build the frozen FULL ScoreFirst prompt independent of the active arm.

        V3.0.5 uses this helper for its no-tool fallback so that the final
        solver receives byte-identical FULL conditioning rather than a nearby
        reimplementation of it.
        """

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

    def _build_minimal_score_first_prompt(
        self,
        problem: str,
        context: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        response_mode = context["response_mode"]
        subject = str(context.get("subject_hint") or "").strip()
        if self.score_first_prompt_profile == "ultra_minimal":
            subject = ""
        subject_line = f"Subject: {subject}\n\n" if subject else ""
        system_prompt = self._score_first_minimal_system_prompt(response_mode)
        user_instruction = {
            _SCORE_FIRST_RESPONSE_MODE_ANSWER: "Return the requested answer.",
            _SCORE_FIRST_RESPONSE_MODE_DERIVATION: "Include the requested derivation.",
            _SCORE_FIRST_RESPONSE_MODE_PROOF: "Provide the requested proof.",
            _SCORE_FIRST_RESPONSE_MODE_PROOF_OR_DISPROOF: "Prove or disprove the claim.",
            _SCORE_FIRST_RESPONSE_MODE_CONSTRUCTION: "Provide the requested construction or counterexample.",
        }[response_mode]
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"{subject_line}Problem:\n{problem}\n\n{user_instruction}",
            },
        ]

    def _score_first_minimal_system_prompt(self, response_mode: str) -> str:
        if response_mode == _SCORE_FIRST_RESPONSE_MODE_DERIVATION:
            return _SCORE_FIRST_MINIMAL_DERIVATION_SYSTEM_PROMPT
        if response_mode == _SCORE_FIRST_RESPONSE_MODE_PROOF:
            return _SCORE_FIRST_MINIMAL_PROOF_SYSTEM_PROMPT
        if response_mode == _SCORE_FIRST_RESPONSE_MODE_PROOF_OR_DISPROOF:
            return _SCORE_FIRST_MINIMAL_PROOF_OR_DISPROOF_SYSTEM_PROMPT
        if response_mode == _SCORE_FIRST_RESPONSE_MODE_CONSTRUCTION:
            return _SCORE_FIRST_MINIMAL_CONSTRUCTION_SYSTEM_PROMPT
        return _SCORE_FIRST_MINIMAL_ANSWER_SYSTEM_PROMPT

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
                "Before finalizing, ensure every requested part has an answer; "
                "preserve all stated constraints/conditions."
            )
        return (
            base
            + "\n"
            + _SCORE_FIRST_EXACTNESS_DISCIPLINE
            + "\n"
            + completeness
            + "\n"
            + _SCORE_FIRST_DECISIVE_REASONING
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
            context_text=context_text,
            response_mode=response_mode,
            requested_actions=requested_actions,
        )

        subject_hint = self._trusted_score_first_subject_hint(
            metadata,
            trusted_domain=trusted_domain,
            trusted_key=trusted_key,
        )
        difficulty_tier, high_difficulty_card = self._score_first_difficulty_context(
            problem=problem,
            metadata=metadata,
            response_mode=response_mode,
            request_spans=request_spans,
            requested_actions=requested_actions,
            target_text=target_text,
            context_text=context_text,
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
            "difficulty_tier": difficulty_tier,
            "high_difficulty_card": high_difficulty_card,
        }

    def _score_first_difficulty_context(
        self,
        *,
        problem: str,
        metadata: Dict[str, Any] | None = None,
        response_mode: str,
        request_spans: List[str],
        requested_actions: List[str],
        target_text: str,
        context_text: str,
    ) -> tuple[str, str]:
        """Classify only obvious high-complexity cases for the V3.0.6 card.

        This is intentionally conservative.  The classifier does not attempt
        to estimate mathematical ability or infer a domain.  It only notices
        objective problem-shape signals that make premature commitment and
        missed hypotheses especially likely.
        """

        text = " ".join(
            part for part in (problem, target_text, context_text) if isinstance(part, str)
        )
        metadata = metadata if isinstance(metadata, dict) else {}
        score = 0

        # A benchmark or caller may provide a coarse difficulty label.  It is
        # advisory only: it can add a compact discipline card, but it never
        # changes the domain, tool permissions, answer, or acceptance policy.
        # This matters for short olympiad statements whose complexity is not
        # visible from character count alone.
        difficulty_hint = str(metadata.get("difficulty") or "").strip().lower()
        if re.search(
            r"(?:olympiad|imo|putnam|contest[_ -]hard|very[_ -]hard|hard)",
            difficulty_hint,
        ):
            score += 2
        if response_mode in {
            _SCORE_FIRST_RESPONSE_MODE_PROOF,
            _SCORE_FIRST_RESPONSE_MODE_PROOF_OR_DISPROOF,
            _SCORE_FIRST_RESPONSE_MODE_CONSTRUCTION,
        }:
            score += 2
        elif response_mode == _SCORE_FIRST_RESPONSE_MODE_DERIVATION:
            # A requested derivation is a useful signal, but not enough by
            # itself: short computational derivations should retain the
            # frozen FULL baseline unless another objective complexity cue is
            # present.
            score += 1
        if len(request_spans) >= 2 or len(set(requested_actions)) >= 2:
            score += 1
        if len(str(problem or "")) >= 420:
            score += 1

        structural_markers = re.findall(
            r"\b(?:floor|sgn|sign|logarithm?|divisor|remainder|modulo|congruence|"
            r"polynomial|coefficient|perfect\s+square|prime|infinite|series|"
            r"expected|probability|rectangle|hexagon|trapezoid|circumcenter|"
            r"circle|tangent|orthocenter|distinct|proper|units?\s+digit)\b",
            text,
            flags=re.IGNORECASE,
        )
        if len(set(item.lower() for item in structural_markers)) >= 2:
            score += 1

        advanced_markers = re.findall(
            r"\b(?:if\s+and\s+only\s+if|necessary\s+and\s+sufficient|asymptotic|"
            r"uniform(?:ly)?|almost\s+surely|for\s+every|for\s+all|subject\s+to|"
            r"eigen(?:value|vector)|jordan|fisher|kkt|fourier|residue|compact(?:ness)?|"
            r"hausdorff|banach|martingale|boundary(?:\s+condition)?|convergence|"
            r"counterexample|classif(?:y|ication)|homomorphism|isomorphism|manifold|"
            r"measur(?:able|ability)|integrab(?:le|ility)|likelihood|confidence\s+interval|"
            r"partial\s+differential|ordinary\s+differential)\b",
            text,
            flags=re.IGNORECASE,
        )
        if advanced_markers:
            score += 1
        condition_markers = re.findall(
            r"\b(?:assume|suppose|given|where|provided|under|whenever|unless|with)\b",
            text,
            flags=re.IGNORECASE,
        )
        if len(condition_markers) >= 3:
            score += 1

        if score >= 2:
            return "high", _TOOL_AUGMENTED_V306_HIGH_DIFFICULTY_DISCIPLINE
        return "standard", ""

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

    def _score_first_span_requests_property_proof(self, span: str) -> bool:
        text = str(span or "").strip()
        match = re.match(
            r"^(?:please\s+)?(?P<verb>show|establish|demonstrate)\b(?P<body>.*)$",
            text,
            flags=re.IGNORECASE,
        )
        if not match:
            return False

        body = str(match.group("body") or "").strip()
        if not body:
            return False

        # Explicit work/calculation requests are derivations, not theorem proofs.
        if re.match(
            r"^(?:your\s+work|the\s+(?:calculation|steps?)|(?:a|the)\s+calculation|"
            r"(?:the\s+)?(?:working|computation))\b",
            body,
            flags=re.IGNORECASE,
        ):
            return False

        # Narrow display-only forms remain answer-value requests.
        if re.match(
            r"^(?:the\s+)?(?:resulting|final|computed)\s+"
            r"(?:matrix|expression|formula|result|value|answer|vector|table)\b",
            body,
            flags=re.IGNORECASE,
        ):
            return False

        # Because this helper is called only on already-grounded request spans,
        # an imperative establish/demonstrate/show of a mathematical claim is proof-like.
        return True

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
        if self._score_first_span_requests_property_proof(text):
            matched["proof"] = True

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
        context_text: str,
        response_mode: str,
        requested_actions: List[str] | None,
    ) -> tuple[str, str, str]:
        target = str(target_text or "")
        context = str(context_text or "")
        requested = list(requested_actions or [])

        def has(pattern: str) -> bool:
            return bool(re.search(pattern, target, flags=re.IGNORECASE))

        def context_has(pattern: str) -> bool:
            return bool(re.search(pattern, context, flags=re.IGNORECASE))

        convergence_target = has(
            r"\b(?:quadratic(?:ally)?|quadratic\s+rate|second[- ]order\s+convergence|"
            r"order\s*2|convergence\s+order|order\s+of\s+convergence|local\s+convergence|"
            r"convergen\w*|contraction|local\s+derivative\s+condition)\b|"
            r"二次收敛|收敛阶|局部收敛|压缩"
        )
        newton_evidence = has(
            r"\bnewton(?:'s)?(?:\s+method|\s+iteration|\s+step)?\b|"
            r"x\s*_\{?n\+1\}?\s*=\s*x\s*_\{?n\}?\s*-\s*f\s*\(|"
            r"\broot[- ]finding\b|牛顿(?:法|迭代|步骤)"
        )
        fixed_point_evidence = has(
            r"\bfixed[- ]point(?:\s+iteration|\s+map|\s+method)?\b|"
            r"x\s*_\{?n\+1\}?\s*=\s*g\s*\(|"
            r"x\s*_\{?n\+1\}?\s*=\s*cos\s*\(|"
            r"\bcontraction(?:\s+mapping)?\b|g\s*'\s*\(\s*x\*?\s*\)|"
            r"不动点迭代|压缩映射"
        )
        fixed_point_condition_target = fixed_point_evidence and (
            convergence_target
            or has(r"\bg\s*'\s*\(\s*x\*?\s*\)\b|导数条件")
        )
        newton_convergence_target = newton_evidence and convergence_target

        indicator_count_target = has(
            r"\bindicator(?:\s+variables?)?\b|\b(?:expected|mean)\s+(?:number|count)\b|"
            r"\bsum\s+of\s+(?:events|indicators)\b|指示变量|"
            r"(?:期望|平均)(?:个数|数量|数目)"
        )
        moment_target = has(
            r"\bE\s*\[[^\]]+\]|\bVar\s*\(|\bmoment\b|\bmean\b|\bexpectation\b|"
            r"\bexpected\s+value\b|期望|方差|矩"
        )

        ols_covariance_target = has(
            r"\bVar\s*\(\s*beta_?hat|\bCov\s*\(\s*beta_?hat|"
            r"\bvariance\s+of\s+(?:the\s+)?(?:beta_?hat|OLS\s+estimator|least[- ]squares\s+estimator)\b|"
            r"\bcovariance\s+matrix\s+of\s+(?:the\s+)?beta_?hat\b|"
            r"\bvariance[- ]covariance\s+matrix\b|"
            r"\bcovariance\b[^.\n]{0,60}\b(?:OLS|beta|coefficient)|"
            r"\bsampling\s+variance\b|"
            r"(?:beta_?hat|OLS|最小二乘估计)[^。；\n]{0,30}(?:方差|协方差)|"
            r"(?:方差[-—]?协方差|协方差)矩阵"
        )

        mle_evidence = has(
            r"\bMLE\b|\bmaximum\s+likelihood(?:\s+estimator)?\b|最大似然估计"
        ) or (
            context_has(r"\bMLE\b|\bmaximum\s+likelihood(?:\s+estimator)?\b|最大似然估计")
            and has(r"\bits\b|其")
        )
        mle_asymptotic_behavior = has(
            r"\basymptotic\s+(?:variance|distribution|normality)\b|"
            r"\basymptotically\s+normal\b|\blimiting\s+(?:distribution|law)\b|"
            r"\bFisher\s+information\b|\bstandard\s+error\b|"
            r"渐近方差|渐近分布|渐近正态|极限分布|Fisher信息|费舍尔信息|标准误"
        )
        mle_asymptotic_target = mle_evidence and mle_asymptotic_behavior

        pde_characteristics_target = has(
            r"\b(?:derive|find|obtain)\b[^.\n]{0,60}\bcharacteristic(?:s|\s+equations?)\b|"
            r"推导[^。；\n]{0,40}特征(?:线|方程)"
        )

        bias_variance_target = has(
            r"\bunbiased\b|\bbias\b|\bvariance\s+of\s+(?:the\s+)?(?:estimator|statistic)\b|"
            r"\bE\s*\[\s*T\s*\]|\bVar\s*\(\s*T\s*\)|无偏|偏差|估计量[^。；\n]{0,20}方差"
        )
        sufficiency_target = has(
            r"\bsufficien(?:t|cy)\b|\bfactorization\s+theorem\b|充分(?:统计量|性)|因子分解"
        )

        brownian_evidence = (
            has(r"\bbrownian(?:\s+motion)?\b|\bB[_ ]?t\b|\bB_t\b|布朗运动")
            or context_has(r"\bbrownian(?:\s+motion)?\b|\bB[_ ]?t\b|\bB_t\b|布朗运动")
        )
        martingale_evidence = has(r"\bmartingale\b|鞅") or context_has(r"\bmartingale\b|鞅")
        poisson_evidence = (
            has(r"\bpoisson\s+process\b|\bN\s*\(\s*t\s*\)|泊松过程")
            or context_has(r"\bpoisson\s+process\b|\bN\s*\(\s*t\s*\)|泊松过程")
        )

        # Task semantics dominate calculation-only checks, with compact target-specific
        # proof variants where the target itself makes a stronger independent check clear.
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
            if micro_strategy == "newton_fixed_point" and fixed_point_condition_target and not newton_evidence:
                return (
                    "task:proof",
                    "fixed_point_convergence_proof",
                    (
                        "Verify every stated hypothesis and conclusion; check x*=g(x*) and the contraction/local "
                        "derivative condition using only the fixed-point hypotheses actually stated."
                    ),
                )
            if (
                micro_strategy in {"newton_fixed_point", "stability_convergence"}
                and newton_convergence_target
            ):
                return (
                    "task:proof",
                    "newton_convergence_proof",
                    (
                        "Verify every stated hypothesis and exact conclusion. For Newton convergence, check the "
                        "simple-root/nonzero-derivative and regularity assumptions, then confirm the local error relation "
                        "implies the claimed order."
                    ),
                )
            if strategy_domain == "statistics" and mle_asymptotic_target:
                return (
                    "task:proof",
                    "mle_asymptotic_proof",
                    (
                        "Verify the proof hypotheses and exact asymptotic claim; recompute the score/Fisher information "
                        "with the stated sample-size scaling and confirm it supports the claimed limiting law."
                    ),
                )
            if micro_strategy == "bias_variance_sufficiency" and bias_variance_target:
                return (
                    "task:proof",
                    "bias_variance_proof",
                    (
                        "Verify the proof reaches the requested estimator property; independently recompute E[T] "
                        "and/or Var(T) under the stated sampling model."
                    ),
                )
            if micro_strategy == "bias_variance_sufficiency" and sufficiency_target:
                return (
                    "task:proof",
                    "sufficiency_proof",
                    (
                        "Verify the proof reaches sufficiency under the stated model and that the factorization or "
                        "conditional-distribution criterion actually applies."
                    ),
                )
            return (
                "task:proof",
                "proof",
                _SCORE_FIRST_TASK_FINAL_CHECKS["proof"],
            )

        # Newton and generic fixed-point iteration share one micro taxonomy but not one
        # verification semantics.
        if micro_strategy == "newton_fixed_point" and fixed_point_condition_target:
            if not newton_evidence or has(
                r"\bfixed[- ]point\b|\bcontraction\b|g\s*'\s*\(|不动点|压缩"
            ):
                return (
                    "micro:newton_fixed_point",
                    "fixed_point_convergence",
                    _SCORE_FIRST_TARGET_FINAL_CHECKS["fixed_point_convergence"],
                )

        if (
            micro_strategy in {"newton_fixed_point", "stability_convergence"}
            and newton_convergence_target
        ):
            return (
                f"micro:{micro_strategy}",
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

        if micro_strategy == "bias_variance_sufficiency":
            if sufficiency_target:
                return (
                    "micro:bias_variance_sufficiency",
                    "sufficiency",
                    _SCORE_FIRST_TARGET_FINAL_CHECKS["sufficiency"],
                )
            if bias_variance_target:
                return (
                    "micro:bias_variance_sufficiency",
                    "bias_variance",
                    _SCORE_FIRST_TARGET_FINAL_CHECKS["bias_variance"],
                )

        if micro_strategy == "brownian_martingale":
            if martingale_evidence and not brownian_evidence:
                return (
                    "micro:brownian_martingale",
                    "martingale",
                    _SCORE_FIRST_TARGET_FINAL_CHECKS["martingale"],
                )
            if brownian_evidence:
                return (
                    "micro:brownian_martingale",
                    "brownian_moment",
                    _SCORE_FIRST_TARGET_FINAL_CHECKS["brownian_moment"],
                )

        # Conservative Brownian/Poisson verification recovery when the frozen target-side
        # micro selector intentionally omitted a background-only method label.
        if strategy_domain == "stochastic_process" and brownian_evidence and (
            has(r"\bE\s*\[|\bVar\s*\(|\bCov\s*\(|\bmartingale\b|期望|方差|协方差|鞅")
        ):
            variant = "martingale" if martingale_evidence and has(r"\bmartingale\b|鞅") else "brownian_moment"
            card = _SCORE_FIRST_TARGET_FINAL_CHECKS[variant]
            return ("domain:stochastic_process", variant, card)

        if (
            strategy_domain == "stochastic_process"
            and micro_strategy is None
            and poisson_evidence
            and has(
                r"\bN\s*\(\s*t\s*\)|\bP\s*\(|\bincrement\b|\bthinning\b|\bsuperposition\b|概率|增量|稀疏|叠加"
            )
        ):
            return (
                "domain:stochastic_process",
                "poisson_process",
                _SCORE_FIRST_MICRO_FINAL_CHECKS["poisson_process"],
            )

        if micro_strategy == "spectrum_invertibility":
            return (
                "micro:spectrum_invertibility",
                "operator_invertibility",
                _SCORE_FIRST_MICRO_FINAL_CHECKS["spectrum_invertibility"],
            )

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

        # Preserve the established V2.5 hierarchy for ordinary calculation/derivation.
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
            if self._is_score_first_answer_label_only(lines[0]) and len(lines) >= 2:
                # Some providers place the payload on the line after the
                # required label.  Prefer that compact payload so a later
                # cut-off explanation cannot replace a complete answer.
                return self._normalize_score_first_answer_payload(lines[1])

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

        # Free-form mode is intentionally permissive about presentation, but
        # only invoke the legacy extractor for unmistakable structured surfaces
        # (JSON/fenced output/boxed LaTeX).  This avoids arbitrary substring
        # harvesting from a normal mathematical explanation.
        if text.startswith(("{", "[", "```")) or r"\boxed{" in text:
            extracted = extract_final_answer(text, problem=problem)
            if extracted != DEFAULT_FALLBACK:
                return self._normalize_score_first_answer_payload(extracted)

        # If the model emits a bare one-line answer, preserve it without routing
        # through the legacy 500-character cap. For an unwrapped multiline
        # answer-value response, retain the final nonempty line as a conservative
        # compatibility fallback; proof tasks never enter this branch.
        if len(lines) == 1:
            return self._normalize_score_first_answer_payload(lines[0])
        if lines:
            return self._normalize_score_first_answer_payload(lines[-1])
        return DEFAULT_FALLBACK

    def _is_score_first_answer_label_only(self, text: str) -> bool:
        return bool(
            re.fullmatch(
                r"\s*(?:final\s+(?:answer|result)|the\s+final\s+(?:answer|result)|"
                r"answer|result|最终答案|最后答案|答案|结果)\s*"
                r"(?::|：|=|\bis\b|是|为)?\s*",
                str(text or ""),
                flags=re.IGNORECASE,
            )
        )

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
        # Accept common OpenAI-compatible response envelopes as a defensive
        # boundary.  The official local client normally returns a string, but
        # injected clients and SDK adapters may return choices/message objects.
        try:
            extracted = response_content(response)
            if extracted:
                return extracted
        except Exception:
            pass
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

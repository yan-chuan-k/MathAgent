from __future__ import annotations

from typing import Dict, List


STRATEGIES: Dict[str, List[str]] = {
    "calculus": ["direct_derivative", "substitution", "convexity", "inequality"],
    "real_analysis": ["definition_check", "sequence_criterion", "epsilon_delta", "counterexample_search"],
    "linear_algebra": ["direct_matrix_computation", "rank_nullity", "eigen_structure", "basis_change"],
    "abstract_algebra": ["homomorphism_kernel", "quotient_structure", "group_action", "counterexample_search"],
    "complex_analysis": ["residue_theorem", "laurent_expansion", "contour_deformation", "rouche_argument"],
    "ode": ["characteristic_equation", "integrating_factor", "variation_of_parameters", "substitution_check"],
    "pde": ["separation_of_variables", "characteristics", "green_function", "energy_method"],
    "probability": ["direct_distribution", "conditioning", "generating_function", "normalization_check"],
    "statistics": ["likelihood", "sampling_distribution", "moment_calculation", "test_statistic"],
    "optimization": ["kkt_conditions", "duality", "convexity", "boundary_check"],
    "topology": ["definition_unfolding", "cover_argument", "quotient_map", "counterexample_search"],
    "discrete_math": ["inclusion_exclusion", "recurrence", "bijection", "generating_function"],
    "numerical_analysis": ["error_expansion", "stability_check", "fixed_point_iteration", "order_conditions"],
    "measure_integration": ["dominant_convergence", "tonelli_fubini", "simple_function_approx", "counterexample_search"],
    "differential_geometry": ["fundamental_forms", "connection_curvature", "frame_computation", "coordinate_invariant"],
    "functional_analysis": ["operator_norm", "compactness_argument", "duality", "counterexample_search"],
    "stochastic_process": ["markov_property", "generator_method", "martingale_stopping", "covariance_function"],
    "linear_regression": ["normal_equations", "projection_matrix", "sampling_distribution", "residual_analysis"],
    "advanced_math": ["definition_first", "direct_computation", "theorem_hypotheses", "counterexample_search"],
    "unknown": ["definition_first", "direct_computation", "independent_check", "counterexample_search"],
}


def strategies_for_domain(domain: str) -> List[str]:
    return list(STRATEGIES.get(domain, STRATEGIES["unknown"]))


def choose_strategy_budget(task_type: str, max_candidates: int = 3) -> int:
    if max_candidates <= 1:
        return 1
    if task_type in {"proof", "construction", "counterexample"}:
        return min(max_candidates, 3)
    if task_type in {"calculation", "derivation"}:
        return min(max_candidates, 3)
    return min(max_candidates, 2)

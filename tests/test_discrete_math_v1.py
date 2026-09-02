import pytest

from math_agent_core.prompts import DISCRETE_CHECK_GUIDE, DISCRETE_SUBTYPE_GUIDE
from math_agent_core.router import classify_problem
from math_agent_core.search.strategy_pool import strategies_for_domain
from math_agent_core.verifiers.discrete_math import _extract_final_integer, run_discrete_math_verification


def _result(checks=None):
    return {"problem_type": "discrete_math", "requested_checks": checks or []}


def _system_item(problem, answer, method=None):
    evidence = run_discrete_math_verification(problem, answer, _result())
    if method is None:
        assert evidence, (problem, answer)
        return evidence[0]
    return next(item for item in evidence if item.method == method)


@pytest.mark.parametrize(
    ("problem", "subtype"),
    [
        ("How many length-10 binary strings contain exactly four 1s?", "combinatorial_counting"),
        ("In how many ways can 8 students be divided into two labeled teams?", "combinatorial_counting"),
        ("Use inclusion-exclusion to count the valid arrangements.", "combinatorial_counting"),
        ("How many subsets of size 3 can be chosen from 9 objects?", "combinatorial_counting"),
        ("组合计数：从10个学生中选择4个有多少种方法？", "combinatorial_counting"),
        ("Let a_n=3a_{n-1}-2a_{n-2}; solve the recurrence from the initial data.", "recurrence"),
        ("Find a closed form for the recurrence b_n=b_{n-1}+2b_{n-2}.", "recurrence"),
        ("递推关系 a_n=a_{n-1}+a_{n-2}，求通项公式。", "recurrence"),
        ("递归定义 b_n=2b_{n-1}+1，求 b_n。", "recurrence"),
        ("For the recurrence c_n=4c_{n-1}-4c_{n-2}, determine c_n.", "recurrence"),
        ("Find the coefficient of x^12 in the ordinary generating function F(x).", "generating_function"),
        ("Using an ordinary generating function, derive the requested coefficient.", "generating_function"),
        ("用生成函数求数列的第10项。", "generating_function"),
        ("求母函数 F(x) 中 x^8 的系数。", "generating_function"),
        ("Count integer partitions using a generating function.", "generating_function"),
        ("How many edges can a simple graph on n vertices have?", "graph_theory"),
        ("Let T be a tree and prove the edge formula.", "graph_theory"),
        ("Find a maximum matching in the graph G.", "graph_theory"),
        ("给定简单图 G，求其顶点度数与边数的关系。", "graph_theory"),
        ("Count the spanning trees of the graph G.", "graph_theory"),
        ("Solve 7x ≡ 3 (mod 20).", "number_theory_modular"),
        ("Compute 3^100 modulo 17.", "number_theory_modular"),
        ("Use the Chinese remainder theorem to solve the congruences.", "number_theory_modular"),
        ("解同余方程 5x ≡ 1 (模 12)。", "number_theory_modular"),
        ("Check gcd(a,m) before finding the modular inverse.", "number_theory_modular"),
    ],
)
def test_discrete_v1_routing_and_subtype(problem, subtype):
    route = classify_problem(problem, {})
    assert route["primary_domain"] == "discrete_math", route
    assert route["discrete_subtype"] == subtype, route


def test_mixed_counting_word_prefers_graph_structure():
    route = classify_problem("Count the spanning trees of the graph G.", {})
    assert route["discrete_subtype"] == "graph_theory"


@pytest.mark.parametrize(
    ("subtype", "expected_markers"),
    [
        ("combinatorial_counting", {"counting_case_split", "symmetry_overcount_audit", "small_case_enumeration"}),
        ("recurrence", {"recurrence_unroll", "closed_form_substitution"}),
        ("generating_function", {"ordinary_generating_function", "index_shift_initial_term_audit"}),
        ("graph_theory", {"graph_invariant", "handshake_tree_invariants"}),
        ("number_theory_modular", {"modular_reduction", "gcd_inverse_check", "crt_compatibility"}),
    ],
)
def test_each_discrete_subtype_has_specialized_strategy_pool(subtype, expected_markers):
    strategies = set(strategies_for_domain("discrete_math", subtype))
    assert expected_markers <= strategies


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("C_10=16796", 16796),
        ("Therefore the answer is 210.", 210),
        ("There are C(10,4)=210 such strings.", 210),
        ("A tree with 12 vertices has 11 edges.", 11),
        ("$\\boxed{132}$", 132),
        ("The possible answers are 10 or 12.", None),
    ],
)
def test_discrete_final_integer_extraction(answer, expected):
    assert _extract_final_integer(answer) == expected


@pytest.mark.parametrize(
    ("problem", "answer", "method"),
    [
        ("How many binary strings of length 8 with exactly 3 ones?", "56", "small_case_enumeration"),
        ("How many binary strings of length 5 with no consecutive 1s?", "13", "small_case_enumeration"),
        ("How many binary strings of length 8 containing exactly three 1s and no adjacent 1s?", "20", "small_case_enumeration"),
        ("Choose 4 objects from 10.", "210", "small_case_enumeration"),
        ("How many subsets of size 5 can be chosen from 12 distinct objects?", "792", "small_case_enumeration"),
        ("How many ordered selections of 3 from 7 without repetition?", "210", "small_case_enumeration"),
        ("How many nonnegative integer triples satisfy x+y+z=5?", "21", "small_case_enumeration"),
        ("How many positive integer triples satisfy x+y+z=5?", "6", "small_case_enumeration"),
        ("Compute 3^100 modulo 17.", "13", "modular_check"),
        ("Solve 7x ≡ 3 (mod 20).", "9", "modular_check"),
        ("A tree with 12 vertices has how many edges?", "11", "graph_invariant_check"),
        ("How many edges does K_8 have?", "28", "graph_invariant_check"),
        ("How many edges does K_{3,5} have?", "15", "graph_invariant_check"),
        ("A graph has degree sequence [3,3,2,2,2,2]. How many edges does it have?", "7", "graph_invariant_check"),
        ("What is the maximum number of edges in a simple graph on 6 vertices?", "15", "graph_invariant_check"),
        ("Find the modular inverse of 3 modulo 11.", "4", "modular_check"),
    ],
)
def test_system_inferred_discrete_checks_pass_and_are_decisive(problem, answer, method):
    item = _system_item(problem, answer, method)
    assert item.status == "pass"
    assert item.is_decisive is True
    assert item.claim_scope == "full_answer"


def test_multi_solution_linear_congruence_pass_is_not_decisive_for_completeness():
    item = _system_item("Solve 6x ≡ 8 (mod 14).", "6", "modular_check")
    assert item.status == "pass"
    assert item.is_decisive is False
    assert item.claim_scope == "subclaim"


@pytest.mark.parametrize(
    ("problem", "wrong_answer", "method"),
    [
        ("Choose 4 objects from 10.", "209", "small_case_enumeration"),
        ("How many subsets of size 5 can be chosen from 12 distinct objects?", "791", "small_case_enumeration"),
        ("How many ordered selections of 3 from 7 without repetition?", "209", "small_case_enumeration"),
        ("How many binary strings of length 8 with exactly 3 ones?", "55", "small_case_enumeration"),
        ("How many binary strings of length 5 with no consecutive 1s?", "12", "small_case_enumeration"),
        ("How many nonnegative integer triples satisfy x+y+z=5?", "20", "small_case_enumeration"),
        ("Solve 7x ≡ 3 (mod 20).", "8", "modular_check"),
        ("A tree with 12 vertices has how many edges?", "12", "graph_invariant_check"),
        ("How many edges does K_8 have?", "27", "graph_invariant_check"),
        ("How many edges does K_{3,5} have?", "14", "graph_invariant_check"),
        ("A graph has degree sequence [3,3,2,2,2,2]. How many edges does it have?", "6", "graph_invariant_check"),
        ("What is the maximum number of edges in a simple graph on 6 vertices?", "14", "graph_invariant_check"),
        ("Find the modular inverse of 3 modulo 11.", "5", "modular_check"),
    ],
)
def test_system_inferred_discrete_checks_fail_wrong_answers(problem, wrong_answer, method):
    item = _system_item(problem, wrong_answer, method)
    assert item.status == "fail"
    assert item.is_decisive is True


def test_requested_recurrence_check_is_usable_and_remains_nondecisive():
    check = {
        "tool": "recurrence_check",
        "arguments": {
            "initial_values": [0, 1],
            "coefficients": [1, 1],
            "claimed_values": [1, 2, 3, 5, 8],
        },
    }
    evidence = run_discrete_math_verification("Fibonacci recurrence.", "8", _result([check]))
    item = next(entry for entry in evidence if entry.method == "recurrence_check")
    assert item.status == "pass"
    assert item.is_decisive is False
    assert item.claim_scope == "subclaim"


def test_requested_recurrence_check_fails_wrong_term():
    check = {
        "tool": "recurrence_check",
        "arguments": {
            "initial_values": [0, 1],
            "coefficients": [1, 1],
            "claimed_values": [1, 2, 3, 6],
        },
    }
    evidence = run_discrete_math_verification("Fibonacci recurrence.", "6", _result([check]))
    item = next(entry for entry in evidence if entry.method == "recurrence_check")
    assert item.status == "fail"
    assert item.is_decisive is False


@pytest.mark.parametrize(
    ("problem", "answer"),
    [
        ("Count some selections of 5 objects.", "10"),
        ("Find a graph satisfying the conditions.", "7"),
        ("Solve the recurrence.", "13"),
        ("Select 3 objects; order or repetition may or may not matter.", "20"),
        ("Find an integer satisfying a modular condition.", "4"),
    ],
)
def test_ambiguous_prose_does_not_create_decisive_system_check(problem, answer):
    evidence = run_discrete_math_verification(problem, answer, _result())
    assert not any(item.is_decisive and item.status == "pass" for item in evidence)


def test_requested_new_small_case_kinds_are_supported_but_nondecisive():
    checks = [
        {"tool": "small_case_enumeration", "arguments": {"kind": "subsets", "n": 10, "k": 4, "expected": 210}},
        {"tool": "small_case_enumeration", "arguments": {"kind": "permutations", "n": 7, "k": 3, "expected": 210}},
    ]
    evidence = run_discrete_math_verification("Check two finite counts.", "210", _result(checks))
    items = [item for item in evidence if item.method == "small_case_enumeration"]
    assert [item.status for item in items] == ["pass", "pass"]
    assert all(not item.is_decisive and item.claim_scope == "subclaim" for item in items)


@pytest.mark.parametrize(
    ("subtype", "required_phrases"),
    [
        ("combinatorial_counting", ("order matters", "repetition", "symmetry", "stars-and-bars")),
        ("recurrence", ("indexing convention", "initial conditions", "substitute", "2-3 valid indices")),
        ("generating_function", ("ordinary or exponential", "index shifts", "initial terms", "coefficient")),
        ("graph_theory", ("handshake lemma", "n-1 edge count", "theorem hypotheses")),
        ("number_theory_modular", ("gcd(a,m)", "CRT", "Euler/Fermat", "all residue classes")),
    ],
)
def test_discrete_subtype_prompt_guides_preserve_v1_error_checks(subtype, required_phrases):
    guide = DISCRETE_SUBTYPE_GUIDE[subtype]
    for phrase in required_phrases:
        assert phrase in guide


def test_discrete_check_guide_encourages_recurrence_checks_without_making_them_decisive():
    assert 'actually populate this check' in DISCRETE_CHECK_GUIDE
    assert 'Candidate-requested checks are auxiliary subclaim checks and are nondecisive' in DISCRETE_CHECK_GUIDE

from math_agent_core.verifiers.discrete_math import run_discrete_math_verification


def _result(checks=None):
    return {"problem_type": "discrete_math", "requested_checks": checks or []}


def test_requested_modular_check_is_exact_but_nondecisive():
    evidence = run_discrete_math_verification(
        "Find the residue.",
        "2",
        _result([{"tool": "modular_check", "arguments": {"base": 2, "exponent": 100, "modulus": 7, "expected": 2}}]),
    )
    item = next(x for x in evidence if x.method == "modular_check")
    assert item.status == "pass"
    assert item.is_decisive is False


def test_system_inferred_modular_power_is_decisive_full_answer():
    evidence = run_discrete_math_verification("Compute 2^100 mod 7.", "2", _result())
    item = next(x for x in evidence if x.method == "modular_check")
    assert item.status == "pass"
    assert item.is_decisive is True
    assert item.claim_scope == "full_answer"


def test_recurrence_check_detects_generated_terms():
    check = {
        "tool": "recurrence_check",
        "arguments": {"initial_values": [0, 1], "coefficients": [1, 1], "claimed_values": [1, 2, 3, 5]},
    }
    evidence = run_discrete_math_verification("Check Fibonacci terms.", "5", _result([check]))
    item = next(x for x in evidence if x.method == "recurrence_check")
    assert item.status == "pass"


def test_recurrence_check_fails_wrong_term():
    check = {
        "tool": "recurrence_check",
        "arguments": {"initial_values": [0, 1], "coefficients": [1, 1], "claimed_values": [1, 2, 4]},
    }
    evidence = run_discrete_math_verification("Check Fibonacci terms.", "4", _result([check]))
    item = next(x for x in evidence if x.method == "recurrence_check")
    assert item.status == "fail"


def test_small_case_binary_enumeration():
    check = {
        "tool": "small_case_enumeration",
        "arguments": {"kind": "binary_strings", "length": 5, "ones": 2, "expected": 10},
    }
    evidence = run_discrete_math_verification("Count binary strings.", "10", _result([check]))
    item = next(x for x in evidence if x.method == "small_case_enumeration")
    assert item.status == "pass"


def test_system_infers_binary_string_count():
    evidence = run_discrete_math_verification("How many binary strings of length 5 with exactly 2 ones?", "10", _result())
    item = next(x for x in evidence if x.method == "small_case_enumeration")
    assert item.status == "pass"
    assert item.is_decisive is True

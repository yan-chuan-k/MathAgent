from math_agent_core.search.candidate_compare import compare_candidate_answers


def _candidate(answer):
    return {"final_answer": {"answer": answer}}


def test_candidate_exact_agreement():
    result = compare_candidate_answers(_candidate("Answer: 72"), _candidate("72"))
    assert result["agreement"] is True
    assert result["agreement_type"] == "normalized_exact"


def test_candidate_numeric_agreement():
    result = compare_candidate_answers(_candidate("1/2"), _candidate("0.5"))
    assert result["agreement"] is True
    assert result["agreement_type"] == "numeric"


def test_candidate_symbolic_agreement():
    result = compare_candidate_answers(_candidate("x^2-1"), _candidate("(x-1)(x+1)"))
    assert result["agreement"] is True
    assert result["agreement_type"] == "symbolic"


def test_candidate_disagreement():
    result = compare_candidate_answers(_candidate("1"), _candidate("2"))
    assert result["agreement"] is False
    assert result["agreement_type"] == "conflict"

import pytest

from math_agent_core.search.strategy_pool import choose_strategy_budget


@pytest.mark.parametrize("task_type", ["calculation", "choice"])
def test_highly_verifiable_tasks_use_one_candidate(task_type):
    assert choose_strategy_budget(task_type, max_candidates=3, verifiability="high") == 1


@pytest.mark.parametrize("task_type", ["proof", "construction", "counterexample"])
def test_open_ended_tasks_use_two_candidates(task_type):
    assert choose_strategy_budget(task_type, max_candidates=3, verifiability="low") == 2


def test_medium_calculation_uses_two_candidates_within_cap():
    assert choose_strategy_budget("calculation", max_candidates=3, verifiability="medium") == 2
    assert choose_strategy_budget("calculation", max_candidates=1, verifiability="medium") == 1


def test_unknown_low_verifiability_task_can_use_three_candidates():
    assert choose_strategy_budget("unknown", max_candidates=3, verifiability="low") == 3
    assert choose_strategy_budget("unknown", max_candidates=2, verifiability="low") == 2


def test_legacy_two_argument_call_remains_supported():
    assert choose_strategy_budget("calculation", 3) == 2


def test_budget_normalizes_labels_and_unknown_verifiability():
    assert choose_strategy_budget(" CALCULATION ", 3, "HIGH") == 1
    assert choose_strategy_budget("calculation", 3, "unsupported") == 2

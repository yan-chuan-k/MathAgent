"""Evaluation helpers with lazy grader imports.

``candidate_compare`` imports ``evaluation.answer_equivalence`` while the
grader imports ``search.candidate_compare``.  Eagerly importing both packages
made an otherwise valid ``import math_agent_core.search.strategy_pool`` fail
with a circular-import error.  Keep the tiny normalization API eager and load
the benchmark grader only when it is requested.
"""

from .answer_equivalence import answer_cluster_key, answers_equivalent, normalize_answer_for_comparison

__all__ = [
    "answer_cluster_key",
    "answers_equivalent",
    "normalize_answer_for_comparison",
    "grade_primary_answer",
    "grade_required_claims",
    "grade_full_problem",
]


def __getattr__(name):
    if name in {"grade_primary_answer", "grade_required_claims", "grade_full_problem"}:
        from . import grader

        return getattr(grader, name)
    raise AttributeError(name)

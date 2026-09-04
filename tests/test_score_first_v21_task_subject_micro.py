from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pytest

from user_agent import (
    ReasoningAgent,
    _SCORE_FIRST_DISCRETE_SUBTYPE_STRATEGIES,
    _SCORE_FIRST_DOMAIN_STRATEGIES,
    _SCORE_FIRST_HUMAN_DOMAIN_LABELS,
    _SCORE_FIRST_MICRO_STRATEGIES,
    _SCORE_FIRST_RESPONSE_MODE_ANSWER,
    _SCORE_FIRST_RESPONSE_MODE_CONSTRUCTION,
    _SCORE_FIRST_RESPONSE_MODE_DERIVATION,
    _SCORE_FIRST_RESPONSE_MODE_PROOF,
)


ROOT = Path(__file__).resolve().parents[1]
ROUTING_FIXTURE = ROOT / "sample_data" / "score_recovery_v2_synthetic_hard.jsonl"
STRESS_FIXTURE = ROOT / "sample_data" / "score_recovery_v21_hidden_style_stress.jsonl"
OLD_HARD = ROOT / "sample_data" / "hard_diagnostics.jsonl"


class RecordingClient:
    def __init__(self, response: str = "Final answer: 42"):
        self.response = response
        self.calls: List[Dict[str, Any]] = []

    def chat(self, messages, temperature=0.1, max_tokens=8192, thinking_mode=True):
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "thinking_mode": thinking_mode,
            }
        )
        return self.response


def _solve(problem: str, response: str, metadata=None):
    client = RecordingClient(response)
    agent = ReasoningAgent(client=client)
    result = agent.solve(problem, metadata or {})
    return result, client, agent


def _prompt(problem: str, metadata):
    result, client, agent = _solve(problem, "Final answer: 42", metadata)
    assert result["final_response"]
    assert len(client.calls) == 1
    return "\n".join(item["content"] for item in client.calls[0]["messages"]), agent


@pytest.mark.parametrize(
    ("problem", "metadata", "expected_mode"),
    [
        ("Compute 6*7.", {"task_type": "calculation"}, _SCORE_FIRST_RESPONSE_MODE_ANSWER),
        ("Which of the following values equals 6*7?", {"task_type": "choice"}, _SCORE_FIRST_RESPONSE_MODE_ANSWER),
        ("Find the requested value.", {}, _SCORE_FIRST_RESPONSE_MODE_ANSWER),
        ("Derive the OLS covariance formula.", {"task_type": "calculation"}, _SCORE_FIRST_RESPONSE_MODE_DERIVATION),
        ("Deduce an expression for the recurrence solution.", {}, _SCORE_FIRST_RESPONSE_MODE_DERIVATION),
        ("推导最小二乘估计量的协方差公式。", {}, _SCORE_FIRST_RESPONSE_MODE_DERIVATION),
        ("Prove that a continuous image of a compact space is compact.", {"task_type": "calculation"}, _SCORE_FIRST_RESPONSE_MODE_PROOF),
        ("证明连续映射下紧致集的像仍紧致。", {"task_type": "derivation"}, _SCORE_FIRST_RESPONSE_MODE_PROOF),
        ("Construct a continuous function with the stated properties.", {"task_type": "calculation"}, _SCORE_FIRST_RESPONSE_MODE_CONSTRUCTION),
        ("Give a counterexample to the converse.", {"task_type": "proof"}, _SCORE_FIRST_RESPONSE_MODE_CONSTRUCTION),
        ("构造一个满足这些性质的反例。", {}, _SCORE_FIRST_RESPONSE_MODE_CONSTRUCTION),
    ],
)
def test_task_mode_precedence_and_selection(problem, metadata, expected_mode):
    agent = ReasoningAgent(client=RecordingClient())
    context = agent._score_first_context(problem, metadata)
    assert context["response_mode"] == expected_mode


def test_trusted_metadata_task_type_is_used_when_problem_has_no_explicit_marker():
    agent = ReasoningAgent(client=RecordingClient())
    context = agent._score_first_context(
        "Obtain the requested covariance formula.",
        {"task_type": "derivation", "subject": "Linear Regression"},
    )
    assert context["response_mode"] == _SCORE_FIRST_RESPONSE_MODE_DERIVATION


def test_answer_value_mode_keeps_one_line_contract():
    prompt, _ = _prompt("Compute 6*7.", {"subject": "Advanced Math"})
    assert "Output exactly ONE visible line" in prompt
    assert "Final answer: <complete requested answer>" in prompt
    assert "Then stop." in prompt
    assert "Do not provide visible explanation or derivation." in prompt


def test_derivation_mode_allows_and_preserves_visible_derivation():
    response = (
        "Final result: Var(beta_hat)=sigma^2(X^T X)^(-1)\n"
        "Derivation: beta_hat=beta+(X^T X)^(-1)X^T epsilon, so covariance follows."
    )
    result, client, _ = _solve(
        "Derive the OLS estimator covariance formula.",
        response,
        {"subject": "Linear Regression"},
    )
    prompt = client.calls[0]["messages"][0]["content"]
    assert "Give the final result first." in prompt
    assert "concise derivation" in prompt
    assert "Output exactly ONE visible line" not in prompt
    assert result["final_response"] == response
    assert len(client.calls) == 1


def test_proof_mode_preserves_complete_response():
    response = (
        "Conclusion: the image is compact.\n"
        "Proof: let {U_i} cover f(K). Then {f^{-1}(U_i)} covers K; "
        "take a finite subcover and map it forward."
    )
    result, client, _ = _solve(
        "Prove that a continuous image of a compact space is compact.",
        response,
        {"subject": "Topology"},
    )
    assert result["final_response"] == response
    assert "State the conclusion first" in client.calls[0]["messages"][0]["content"]
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    ("problem", "response"),
    [
        (
            "Construct a continuous function with the requested endpoint values.",
            "Construction: f(x)=x.\nVerification: it is continuous and has the required endpoint values.",
        ),
        (
            "Give a counterexample to the converse statement.",
            "Counterexample: f_n(x)=x^n on [0,1].\nVerification: pointwise convergence is not uniform.",
        ),
    ],
)
def test_construction_counterexample_mode_preserves_object_and_verification(problem, response):
    result, client, _ = _solve(problem, response, {"subject": "Real Analysis"})
    prompt = client.calls[0]["messages"][0]["content"]
    assert "State the constructed object or counterexample first." in prompt
    assert result["final_response"] == response
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("discrete_math", "discrete_math"),
        ("Discrete Math", "discrete_math"),
        ("Discrete Mathematics", "discrete_math"),
        ("Numerical Analysis", "numerical_analysis"),
        ("Measure Theory", "measure_integration"),
        ("Measure and Integration", "measure_integration"),
        ("Measure Theory and Integration", "measure_integration"),
        ("Differential Geometry", "differential_geometry"),
        ("Probability", "probability"),
        ("Probability Theory", "probability"),
        ("Abstract Algebra", "abstract_algebra"),
        ("Stochastic Process", "stochastic_process"),
        ("Stochastic Processes", "stochastic_process"),
        ("Complex Analysis", "complex_analysis"),
        ("ODE", "ode"),
        ("Ordinary Differential Equation", "ode"),
        ("Ordinary Differential Equations", "ode"),
        ("Statistics", "statistics"),
        ("Functional Analysis", "functional_analysis"),
        ("Linear Regression", "linear_regression"),
        ("PDE", "pde"),
        ("Partial Differential Equation", "pde"),
        ("Partial Differential Equations", "pde"),
        ("Advanced Mathematics", "advanced_math"),
        ("Advanced Math", "advanced_math"),
        ("Linear Algebra", "linear_algebra"),
        ("Optimization", "optimization"),
        ("Real Analysis", "real_analysis"),
        ("Topology", "topology"),
        ("离散数学", "discrete_math"),
        ("数值分析", "numerical_analysis"),
        ("测度论", "measure_integration"),
        ("微分几何", "differential_geometry"),
        ("概率论", "probability"),
        ("抽象代数", "abstract_algebra"),
        ("随机过程", "stochastic_process"),
        ("复分析", "complex_analysis"),
        ("常微分方程", "ode"),
        ("统计学", "statistics"),
        ("泛函分析", "functional_analysis"),
        ("线性回归", "linear_regression"),
        ("偏微分方程", "pde"),
        ("高等代数", "linear_algebra"),
        ("运筹学", "optimization"),
        ("数学分析", "real_analysis"),
        ("拓扑学", "topology"),
    ],
)
def test_subject_label_canonicalization(label, expected):
    agent = ReasoningAgent(client=RecordingClient())
    assert agent._canonical_score_first_domain_label(label) == expected


def test_subject_type_category_precedence_is_deterministic():
    agent = ReasoningAgent(client=RecordingClient())
    trusted, key = agent._trusted_score_first_domain(
        {
            "subject": "Numerical Analysis",
            "type": "Probability Theory",
            "category": "Topology",
        }
    )
    assert (trusted, key) == ("numerical_analysis", "subject")

    trusted, key = agent._trusted_score_first_domain(
        {"subject": "untrusted label", "type": "Differential Geometry", "category": "Topology"}
    )
    assert (trusted, key) == ("differential_geometry", "type")

    trusted, key = agent._trusted_score_first_domain(
        {"subject": "untrusted", "type": "also untrusted", "category": "Abstract Algebra"}
    )
    assert (trusted, key) == ("abstract_algebra", "category")


@pytest.mark.parametrize(
    ("subject", "problem", "expected_domain"),
    [
        ("Numerical Analysis", "Find the next Newton iterate.", "numerical_analysis"),
        ("Differential Geometry", "Compute the requested curvature.", "differential_geometry"),
        ("Abstract Algebra", "Find the kernel of the map.", "abstract_algebra"),
        ("Ordinary Differential Equations", "Solve the initial value problem.", "ode"),
    ],
)
def test_human_subject_selects_expected_domain_strategy(subject, problem, expected_domain):
    prompt, agent = _prompt(problem, {"subject": subject})
    context = agent._score_first_context(problem, {"subject": subject})
    assert context["strategy_domain"] == expected_domain
    assert _SCORE_FIRST_DOMAIN_STRATEGIES[expected_domain] in prompt


@pytest.mark.parametrize(
    ("domain", "problem", "expected_micro"),
    [
        ("numerical_analysis", "Perform two Newton iterations for f(x)=x^2-2.", "newton_fixed_point"),
        ("measure_integration", "Decide whether dominated convergence permits interchanging this limit and integral.", "limit_integral"),
        ("differential_geometry", "Compute the Gaussian curvature of the parametrized surface.", "curvature"),
        ("probability", "Given that B occurred, compute the conditional probability of A.", "conditioning_bayes"),
        ("abstract_algebra", "Compute the kernel and image of this homomorphism.", "homomorphism_quotient"),
        ("stochastic_process", "Find the stationary distribution of this Markov chain transition matrix.", "markov_stationary"),
        ("complex_analysis", "Evaluate the contour integral by residues.", "residue_contour"),
    ],
)
def test_high_value_domain_selects_exactly_one_micro_strategy(domain, problem, expected_micro):
    agent = ReasoningAgent(client=RecordingClient())
    context = agent._score_first_context(problem, {"subject": domain})
    assert context["micro_strategy"] == expected_micro
    assert context["micro_card"]

    prompt, _ = _prompt(problem, {"subject": domain})
    assert "Method hint:" in prompt
    chosen_card = context["micro_card"]
    assert chosen_card in prompt

    all_cards = [
        card
        for _, entries in _SCORE_FIRST_MICRO_STRATEGIES.items()
        for _, _, _, card in entries
    ]
    assert sum(card in prompt for card in all_cards) == 1


def test_no_micro_card_for_discrete_math_beyond_single_subtype_card():
    problem = "Solve 7x ≡ 3 (mod 20)."
    agent = ReasoningAgent(client=RecordingClient())
    context = agent._score_first_context(problem, {"subject": "Discrete Mathematics"})
    assert context["strategy_domain"] == "discrete_math"
    assert context["discrete_subtype"] == "number_theory_modular"
    assert context["micro_card"] is None
    prompt, _ = _prompt(problem, {"subject": "Discrete Mathematics"})
    assert "Subtype hint:" in prompt
    assert "Method hint:" not in prompt


def test_total_strategy_block_stays_under_120_words():
    agent = ReasoningAgent(client=RecordingClient())

    for domain, entries in _SCORE_FIRST_MICRO_STRATEGIES.items():
        for name, _, _, card in entries:
            context = {
                "strategy_domain": domain,
                "discrete_subtype": None,
                "micro_card": card,
            }
            block = agent._score_first_strategy_block(context)
            assert len(block.split()) <= 120, (domain, name, len(block.split()), block)

    for subtype, card in _SCORE_FIRST_DISCRETE_SUBTYPE_STRATEGIES.items():
        context = {
            "strategy_domain": "discrete_math",
            "discrete_subtype": subtype,
            "micro_card": None,
        }
        block = agent._score_first_strategy_block(context)
        assert len(block.split()) <= 120, (subtype, len(block.split()), block)


def test_110_case_subject_robustness_views_and_conservative_no_subject_precision():
    rows = [json.loads(line) for line in ROUTING_FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]
    agent = ReasoningAgent(client=RecordingClient())

    canonical_correct = 0
    human_correct = 0
    no_subject_specialized = 0
    no_subject_specialized_correct = 0
    generic_fallback = 0

    for row in rows:
        expected = row["expected_domain"]

        canonical = agent._score_first_context(
            row["problem"],
            {"subject": expected, "task_type": row["task_type"]},
        )
        canonical_correct += int(canonical["strategy_domain"] == expected)

        human = agent._score_first_context(
            row["problem"],
            {"subject": _SCORE_FIRST_HUMAN_DOMAIN_LABELS[expected], "task_type": row["task_type"]},
        )
        human_correct += int(human["strategy_domain"] == expected)

        no_subject = agent._score_first_context(
            row["problem"],
            {"task_type": row["task_type"]},
        )
        if no_subject["strategy_is_specialized"]:
            no_subject_specialized += 1
            no_subject_specialized_correct += int(no_subject["strategy_domain"] == expected)
        else:
            generic_fallback += 1

    assert canonical_correct == 110
    assert human_correct == 110
    assert no_subject_specialized == 11
    assert no_subject_specialized_correct == 11
    assert generic_fallback == 99


def test_low_confidence_no_subject_problem_uses_general_strategy():
    agent = ReasoningAgent(client=RecordingClient())
    context = agent._score_first_context("Determine the requested mathematical object.", {})
    assert context["strategy_domain"] == "advanced_math"
    assert context["strategy_is_specialized"] is False
    assert context["domain_source"] == "general_fallback"


def test_no_subject_discrete_subtype_requires_high_confidence_discrete_route():
    agent = ReasoningAgent(client=RecordingClient())

    low = agent._score_first_context(
        "A random permutation is observed; compute the requested probability.",
        {},
    )
    if low["strategy_domain"] != "discrete_math":
        assert low["discrete_subtype"] is None

    high = agent._score_first_context(
        "Use the Chinese remainder theorem to solve x ≡ 2 (mod 5), x ≡ 3 (mod 7).",
        {},
    )
    assert high["strategy_domain"] == "discrete_math"
    assert high["discrete_subtype"] == "number_theory_modular"


def test_v21_preserves_one_call_budget_for_100_mixed_mode_successes():
    responses = {
        0: "Final answer: 42",
        1: "Final result: 42\nDerivation: 6*7=42.",
        2: "Conclusion: true.\nProof: direct verification.",
        3: "Construction: x=0.\nVerification: it has the required property.",
    }
    client = RecordingClient()
    agent = ReasoningAgent(client=client)
    problems = [
        "Compute 6*7.",
        "Derive the product 6*7.",
        "Prove that 6*7=42.",
        "Construct an integer x satisfying x=0.",
    ]
    for index in range(100):
        mode = index % 4
        client.response = responses[mode]
        result = agent.solve(problems[mode], {"subject": "Advanced Math"})
        assert result["final_response"]
    assert len(client.calls) == 100
    assert all(call["temperature"] == 0.1 for call in client.calls)
    assert all(call["max_tokens"] == 8192 for call in client.calls)
    assert all(call["thinking_mode"] is True for call in client.calls)


def test_hidden_style_stress_suite_structure_language_features_and_no_overlap():
    rows = [json.loads(line) for line in STRESS_FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 72
    assert Counter(row["language"] for row in rows) == {"en": 36, "zh": 36}

    domains = Counter(row["expected_domain"] for row in rows)
    assert len(domains) == 18
    assert domains["discrete_math"] == 15
    assert domains["numerical_analysis"] == 8
    assert domains["measure_integration"] == 7

    task_types = Counter(row["task_type"] for row in rows)
    assert task_types["proof"] > 0
    assert task_types["derivation"] > 0
    assert task_types["construction"] + task_types["counterexample"] > 0
    assert task_types["choice"] > 0
    assert task_types["calculation"] > 0

    features = {row["stress_feature"] for row in rows}
    for required in {
        "multi_step_calculation",
        "theorem_selection_trap",
        "missing_hypothesis_trap",
        "cross_concept",
        "multi_part_answer",
        "proof",
        "derivation",
        "construction_counterexample",
    }:
        assert required in features

    old_problems = set()
    for fixture in (OLD_HARD, ROUTING_FIXTURE):
        old_problems.update(
            json.loads(line)["problem"].strip()
            for line in fixture.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    assert all(row["problem"].strip() not in old_problems for row in rows)


def test_hidden_style_stress_ground_truth_is_grading_only():
    rows = [json.loads(line) for line in STRESS_FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]
    forbidden = {
        "grading",
        "expected_domain",
        "expected_subtype",
        "answer",
        "expected_answer",
        "answer_hint",
        "expected_micro",
    }
    for row in rows:
        metadata = {
            key: value
            for key, value in row.items()
            if key not in forbidden | {"problem"}
        }
        assert forbidden.isdisjoint(metadata)


def test_v21_prompts_still_have_no_internal_json_or_orchestration_contract():
    prompt, _ = _prompt(
        "Derive the OLS covariance formula.",
        {"subject": "Linear Regression", "task_type": "derivation"},
    )
    for forbidden in (
        "OUTPUT_CONTRACT",
        "Return exactly one valid JSON object",
        "requested_checks",
        "reasoning_plan",
        "candidate B",
        "AcceptancePolicy",
    ):
        assert forbidden not in prompt

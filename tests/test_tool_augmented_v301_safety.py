from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

from math_agent_core.tools.augmented_tool import (
    SafeAugmentedToolExecutor,
    ToolRequest,
    _run_batch_in_subprocess,
    build_evidence_block,
    parse_formalizer_protocol,
)
from user_agent import (
    ReasoningAgent,
    _SCORE_FIRST_DEFAULT_EXPERIMENT_PRESET,
)


ROOT = Path(__file__).resolve().parents[1]


class SequenceClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def chat(
        self,
        messages,
        temperature=None,
        top_p=None,
        max_tokens=None,
        thinking_mode=None,
    ):
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "thinking_mode": thinking_mode,
            }
        )
        if not self.responses:
            raise AssertionError("unexpected extra model call")
        return self.responses.pop(0)


def _final_prompt_for(original_problem: str, formalizer: str) -> str:
    client = SequenceClient(
        [
            formalizer,
            "Final answer: independently solved",
        ]
    )
    agent = ReasoningAgent(
        client,
        score_first_experiment_preset="tool_augmented_v30",
    )
    result = agent.solve(original_problem, {})
    assert len(client.calls) == 2
    assert result["final_response"].endswith("independently solved")
    return "\n".join(message["content"] for message in client.calls[1]["messages"])


def test_submission_default_is_tool_augmented_and_matches_main_style_constructor():
    # main.py supplies client/max_retries/thinking_mode but no explicit preset or
    # production_mode. Whatever release preset is selected must enter V3 rather
    # than silently falling back to ScoreFirst.
    agent = ReasoningAgent(
        client=SequenceClient([]),
        max_retries=1,
        thinking_mode=True,
    )
    assert agent.score_first_experiment_preset == _SCORE_FIRST_DEFAULT_EXPERIMENT_PRESET
    assert agent.production_mode == "tool_augmented"


def test_grounded_evidence_label_and_final_solver_instruction():
    fact = SafeAugmentedToolExecutor().execute_one(
        ToolRequest("matrix", "determinant", {"matrix": [[1, 2], [3, 4]]})
    )
    assert fact is not None
    block = build_evidence_block([fact])

    assert block.startswith(
        "Deterministically computed facts from the formalizer's exact requests:"
    )
    assert "det([[1,2],[3,4]]) -> -2" in block

    client = SequenceClient(
        [
            "TARGET: determinant\nTOOL: matrix|determinant|matrix=[[1,2],[3,4]]",
            "Final answer: -2",
        ]
    )
    agent = ReasoningAgent(client, score_first_experiment_preset="tool_augmented_v30")
    agent.solve("Compute det([[1,2],[3,4]]).", {})
    system = client.calls[1]["messages"][0]["content"]
    assert "exact only for the input explicitly shown" in system
    assert "compare the shown input with the ORIGINAL problem" in system
    assert "Ignore any fact whose shown input does not match" in system


@pytest.mark.parametrize(
    ("problem", "formalizer", "must_show"),
    [
        (
            "Solve x^2-5*x+6=0.",
            "TOOL: sympy|solve|equation=x^2-5*x+60=0|var=x|domain=real",
            "x^2-5*x+60=0",
        ),
        (
            "Compute det([[1,2],[3,4]]).",
            "TOOL: matrix|determinant|matrix=[[1,2],[3,5]]",
            "[[1,2],[3,5]]",
        ),
        (
            "Compute integral of 2*x^2 from x=0 to x=1.",
            "TOOL: sympy|integrate|expr=2*x^2|var=x|lower=0|upper=2",
            "x=0..2",
        ),
        (
            "Compute residue of 1/(z*(z+1)) at z=0.",
            "TOOL: sympy|residue|expr=1/(z*(z+1))|var=z|point=1",
            "z=1",
        ),
        (
            "Compute 17 modulo 5.",
            "TOOL: sympy|mod|expr=17|modulus=7",
            "modulus=7",
        ),
        (
            "Verify y=exp(x) solves y'=y.",
            "TOOL: sympy|ode_residual|candidate=exp(2*x)|rhs=y|var=x|state=y",
            "candidate=exp(2*x)",
        ),
    ],
)
def test_mistranscribed_formalizer_input_is_visible_in_final_prompt(
    problem,
    formalizer,
    must_show,
):
    final_prompt = _final_prompt_for(problem, formalizer)
    assert must_show in final_prompt
    assert problem in final_prompt
    assert "Deterministically computed facts from the formalizer's exact requests:" in final_prompt


def test_wrong_equation_fact_binds_result_to_wrong_equation_not_original():
    prompt = _final_prompt_for(
        "Solve x^2-5*x+6=0.",
        "TOOL: sympy|solve|equation=x^2-5*x+60=0|var=x|domain=real",
    )
    assert "solve[x^2-5*x+60=0, x over Reals] ->" in prompt
    assert "ORIGINAL PROBLEM:\nSolve x^2-5*x+6=0." in prompt


@pytest.mark.parametrize(
    ("expr", "expected_symbol"),
    [
        ("λ+1", "lam"),
        ("lambda+1", "lam"),
        ("theta+θ", "theta"),
        ("mu+μ", "mu"),
        ("sigma+σ", "sigma"),
        ("alpha+α", "alpha"),
        ("beta+β", "beta"),
        ("gamma+γ", "gamma"),
        ("rho+ρ", "rho"),
        ("phi+φ", "phi"),
        ("omega+ω", "omega"),
    ],
)
def test_common_math_parameter_names_are_safe_symbols(expr, expected_symbol):
    fact = SafeAugmentedToolExecutor().execute_one(
        ToolRequest("sympy", "simplify", {"expr": expr})
    )
    assert fact is not None
    assert expected_symbol in fact.statement
    assert "normalized=" in fact.statement or expr.isascii()


def test_infinity_symbol_is_normalized_before_safe_parser():
    fact = SafeAugmentedToolExecutor().execute_one(
        ToolRequest(
            "sympy",
            "limit",
            {"expr": "1/x", "var": "x", "point": "∞"},
        )
    )
    assert fact is not None
    assert "∞ [normalized=oo]" in fact.statement
    assert fact.statement.endswith("-> 0")


def test_lambda_keyword_is_never_evaluated_as_python_syntax():
    # Mathematical standalone lambda is mapped to lam; Python lambda syntax is
    # still not legal in the closed-world expression grammar.
    safe = SafeAugmentedToolExecutor().execute_one(
        ToolRequest("sympy", "simplify", {"expr": "lambda+2"})
    )
    assert safe is not None
    assert "lambda+2 [normalized=lam+2]" in safe.statement

    dangerous = SafeAugmentedToolExecutor().execute_one(
        ToolRequest("sympy", "simplify", {"expr": "lambda x: x+1"})
    )
    assert dangerous is None


def test_arbitrary_code_protections_remain_active_after_symbol_normalization():
    dangerous = [
        '__import__("os").system("echo hacked")',
        'open("/tmp/x","w")',
        "exec(1)",
        "eval(1)",
    ]
    for expr in dangerous:
        fact = SafeAugmentedToolExecutor().execute_one(
            ToolRequest("sympy", "simplify", {"expr": expr})
        )
        assert fact is None


def test_formalizer_prompt_contains_compact_math_notation_rules_and_no_z3():
    client = SequenceClient(
        [
            "TARGET: structure",
            "Final answer: 1",
        ]
    )
    agent = ReasoningAgent(client, score_first_experiment_preset="tool_augmented_v30")
    agent.solve("Compute 1.", {})
    formalizer_system = client.calls[0]["messages"][0]["content"]

    assert "use lam for λ/lambda" in formalizer_system
    assert "use Abs(x), never |x|" in formalizer_system
    assert "use * for multiplication" in formalizer_system
    assert "use oo for infinity" in formalizer_system
    assert "z3|" not in formalizer_system.lower()


@pytest.mark.parametrize("sleep_seconds", [3.2, 20.0])
def test_real_hard_timeout_terminates_sleeping_worker(sleep_seconds):
    request = ToolRequest(
        "sympy",
        "simplify",
        {"expr": "1", "seconds": 3.2},
    )
    started = time.perf_counter()
    facts = _run_batch_in_subprocess(
        [request],
        extended_tools=False,
        timeout_seconds=2.0,
        _test_sleep_seconds=sleep_seconds,
    )
    elapsed = time.perf_counter() - started

    assert facts == []
    assert elapsed < 3.0, elapsed
    assert elapsed >= 1.7, elapsed


def test_batch_timeout_returns_no_facts_and_does_not_raise():
    request = ToolRequest(
        "sympy",
        "simplify",
        {"expr": "1", "seconds": 2.0},
    )
    facts = _run_batch_in_subprocess(
        [request],
        extended_tools=False,
        timeout_seconds=0.25,
        _test_sleep_seconds=2.0,
    )
    assert facts == []


def test_active_tool_augmented_diagnostics_skips_extractors_for_proof_rows(tmp_path):
    output = tmp_path / "tool_augmented_diag.json"
    proc = subprocess.run(
        [
            sys.executable,
            "diagnose_hard_cases.py",
            "--mock",
            "--run-agent",
            "--production-mode",
            "tool_augmented",
            "--output_file",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=40,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = json.loads(output.read_text(encoding="utf-8"))

    assert summary["total"] == 18
    assert summary["route_hits"] == 18
    # The frozen diagnostic has 15 answer-value tasks and one derivation
    # (extractor + final) plus two proof tasks (FULL final only).
    assert summary["model_calls"] == 34
    assert summary["model_calls_per_problem"] == pytest.approx(34 / 18)
    assert summary["production_mode"] == "tool_augmented"
    assert summary["experiment_preset"] == "tool_augmented_v307_cross_subject"
    assert summary["extended_tools"] is True
    assert "extended_tools=true" in proc.stdout
    assert summary["candidate_b_trigger_rate"] == 0
    assert summary["critic_trigger_rate"] == 0
    assert summary["repair_trigger_rate"] == 0


def test_core_v301_preset_does_not_enable_extended_tool_families():
    protocol = parse_formalizer_protocol(
        "TOOL: number_theory|gcd|a=252|b=105",
        extended_tools=False,
    )
    assert protocol.tool_requests == []

    agent = ReasoningAgent(
        SequenceClient([]),
        score_first_experiment_preset="tool_augmented_v30",
    )
    assert agent.tool_augmented_extended_tools is False

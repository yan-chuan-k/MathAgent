from __future__ import annotations

import inspect
import sys
import threading
import time
from typing import Any, Dict, List

import pytest

import math_agent_core.tools.augmented_tool as augmented_tool
import math_agent_core.tools.sympy_tool as legacy_sympy
import math_agent_core.verifiers.linear_algebra as legacy_matrix
from math_agent_core.tools.augmented_tool import (
    SafeAugmentedToolExecutor,
    ToolRequest,
    run_answer_verification_in_subprocess,
)
from user_agent import ReasoningAgent


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


def _trace(result):
    return {item["step"]: item["content"] for item in result["trace"]}


def _legacy_v31_agent(client: Any, **kwargs: Any) -> ReasoningAgent:
    """Keep V3.0.2 isolation regressions on the legacy tool pipeline."""

    return ReasoningAgent(
        client,
        score_first_experiment_preset="tool_augmented_v31_extended",
        **kwargs,
    )


def test_active_executor_uses_only_fixed_popen_command_without_fork():
    source = inspect.getsource(augmented_tool)
    assert "multiprocessing" not in source
    assert "get_context(" not in source
    assert "os.fork(" not in source
    assert augmented_tool._WORKER_COMMAND == (
        sys.executable,
        "-m",
        "math_agent_core.tools.augmented_worker",
    )


def test_threaded_parent_stress_finishes_20_batches_and_reaps_workers(monkeypatch):
    real_popen = augmented_tool.subprocess.Popen
    workers = []

    def tracked_popen(*args, **kwargs):
        assert args[0] == [
            sys.executable,
            "-m",
            "math_agent_core.tools.augmented_worker",
        ]
        assert kwargs.get("shell") is False
        process = real_popen(*args, **kwargs)
        workers.append(process)
        return process

    monkeypatch.setattr(augmented_tool.subprocess, "Popen", tracked_popen)
    stop = threading.Event()

    def harmless_background_thread():
        while not stop.wait(0.01):
            pass

    threads = [
        threading.Thread(target=harmless_background_thread, daemon=True)
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()

    requests = [
        ToolRequest("sympy", "simplify", {"expr": "(x+1)^2-(x^2+2*x+1)"}),
        ToolRequest("matrix", "determinant", {"matrix": [[1, 2], [3, 4]]}),
        ToolRequest(
            "sympy",
            "solve",
            {"equation": "x^2-5*x+6=0", "var": "x", "domain": "real"},
        ),
    ]
    completed = 0
    try:
        assert all(thread.is_alive() for thread in threads)
        executor = SafeAugmentedToolExecutor(timeout_seconds=7.0)
        for _ in range(20):
            statements = [fact.statement for fact in executor.execute(requests)]
            assert len(statements) == 3
            assert any("-> 0" in statement for statement in statements)
            assert any("det([[1,2],[3,4]]) -> -2" in statement for statement in statements)
            assert any("-> {2, 3}" in statement for statement in statements)
            completed += 1
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=1.0)

    assert completed == 20
    assert len(workers) == 20
    assert all(process.poll() is not None for process in workers)
    assert all(not thread.is_alive() for thread in threads)


def test_isolated_answer_verification_returns_only_compact_schema():
    evidence = run_answer_verification_in_subprocess(
        "3",
        "ANSWER_VALUE",
        requests=[{"kind": "pure_arithmetic", "expression": "1+1"}],
    )
    assert evidence
    assert set(evidence[0]) == {
        "status",
        "method",
        "details",
        "residual",
        "is_decisive",
        "claim_scope",
    }
    assert evidence[0]["status"] == "fail"
    assert evidence[0]["is_decisive"] is True


def test_active_v3_never_calls_legacy_verifiers_in_parent(monkeypatch):
    parent_calls = []

    def forbidden_parent_call(*args, **kwargs):
        parent_calls.append((args, kwargs))
        raise AssertionError("legacy verifier executed in parent")

    monkeypatch.setattr(legacy_sympy, "run_sympy_verification", forbidden_parent_call)
    monkeypatch.setattr(
        legacy_matrix,
        "run_system_inferred_matrix_verification",
        forbidden_parent_call,
    )
    client = SequenceClient(
        [
            "TARGET: exact arithmetic",
            "Final answer: 2",
        ]
    )
    result = _legacy_v31_agent(client).solve(
        "Compute 1+1.",
        {},
    )

    assert result["final_response"] == "2"
    assert len(client.calls) == 2
    assert parent_calls == []
    assert _trace(result)["correction_call_used"] == "false"


def test_verification_timeout_returns_call_2_answer_without_correction(monkeypatch):
    real_verify = run_answer_verification_in_subprocess

    def slow_isolated_verification(final_response, response_mode, *, requests, timeout_seconds):
        return real_verify(
            final_response,
            response_mode,
            requests=requests,
            timeout_seconds=timeout_seconds,
            _test_sleep_seconds=20.0,
        )

    monkeypatch.setattr(
        augmented_tool,
        "run_answer_verification_in_subprocess",
        slow_isolated_verification,
    )
    client = SequenceClient(
        [
            "TARGET: exact arithmetic",
            "Final answer: 3",
        ]
    )
    agent = _legacy_v31_agent(
        client,
        tool_augmented_verification_timeout_seconds=2.0,
    )
    started = time.perf_counter()
    result = agent.solve("Compute 1+1.", {})
    elapsed = time.perf_counter() - started

    assert elapsed < 3.0, elapsed
    assert result["final_response"] == "3"
    assert len(client.calls) == 2
    assert _trace(result)["decisive_check_failure"] == "false"
    assert _trace(result)["correction_call_used"] == "false"


@pytest.mark.parametrize(
    "evidence",
    [
        [],
        [
            {
                "status": "inconclusive",
                "method": "timeout",
                "details": "no result",
                "residual": None,
                "is_decisive": False,
                "claim_scope": "subclaim",
            }
        ],
        [
            {
                "status": "fail",
                "method": "auxiliary",
                "details": "supporting check",
                "residual": "1",
                "is_decisive": False,
                "claim_scope": "subclaim",
            }
        ],
        [
            {
                "status": "pass",
                "method": "exact",
                "details": "passed",
                "residual": "0",
                "is_decisive": True,
                "claim_scope": "full_answer",
            }
        ],
    ],
)
def test_only_decisive_failure_can_trigger_correction(monkeypatch, evidence):
    monkeypatch.setattr(
        augmented_tool,
        "run_answer_verification_in_subprocess",
        lambda *args, **kwargs: evidence,
    )
    client = SequenceClient(["TARGET: compute", "Final answer: 3"])
    result = _legacy_v31_agent(client).solve(
        "Compute 1+1.",
        {},
    )
    assert result["final_response"] == "3"
    assert len(client.calls) == 2
    assert _trace(result)["correction_call_used"] == "false"

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from intern_s1_client import InternS1Client
from math_agent_core.clients import MockClient
from math_agent_core.evaluation import answers_equivalent
from math_agent_core.router import classify_problem
from user_agent import ReasoningAgent


BASE_DIR = Path(__file__).resolve().parent


class CountingClient:
    def __init__(self, client: Any):
        self.client = client
        self.model = getattr(client, "model", "unknown")
        self.total_calls = 0
        self.calls_by_role: Dict[str, int] = {}

    def chat(self, *args, **kwargs):
        role = _detect_call_role(kwargs.get("messages", args[0] if args else []))
        self.total_calls += 1
        self.calls_by_role[role] = self.calls_by_role.get(role, 0) + 1
        return self.client.chat(*args, **kwargs)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def build_client(use_mock: bool, thinking_mode: bool = True) -> Any:
    if use_mock:
        return MockClient()
    model = os.getenv("INTERN_MODEL", "intern-s2-preview-397b")
    base_url = os.getenv("INTERN_API_BASE", "https://chat.intern-ai.org.cn/api/v1/")
    return InternS1Client(model=model, base_url=base_url, thinking_mode=thinking_mode)


def evaluate(
    input_file: Path,
    output_file: Path,
    use_mock: bool,
    run_agent: bool,
    thinking_mode: bool,
    include_ids: set[str] | None = None,
) -> Dict[str, Any]:
    items = load_jsonl(input_file)
    if include_ids:
        items = [item for item in items if str(item.get("idx", "")) in include_ids]
    raw_client = build_client(use_mock=use_mock, thinking_mode=thinking_mode) if run_agent else None
    client = CountingClient(raw_client) if raw_client is not None else None
    agent = ReasoningAgent(client=client, thinking_mode=thinking_mode) if client is not None else None
    rows = []
    route_hits = 0
    valid_outputs = 0
    answer_evaluated = 0
    answer_correct = 0

    for item in items:
        problem = str(item.get("problem", ""))
        metadata = {
            key: value
            for key, value in item.items()
            if key not in {"problem", "answer", "expected_answer", "answer_hint"}
        }
        route = classify_problem(problem, metadata)
        expected = str(item.get("expected_domain", ""))
        route_ok = route.get("primary_domain") == expected or expected in route.get("domain_candidates", [])
        if route_ok:
            route_hits += 1

        result = None
        final_response = ""
        calls_before = client.total_calls if client is not None else 0
        if agent is not None:
            result = agent.solve(problem, metadata)
            final_response = str(result.get("final_response", "")).strip() if isinstance(result, dict) else ""
            try:
                json.dumps(result, ensure_ascii=False)
                if final_response:
                    valid_outputs += 1
            except TypeError:
                pass
        model_calls = (client.total_calls - calls_before) if client is not None else 0
        expected_answer = item.get("expected_answer", item.get("answer"))
        answer_ok = None
        if agent is not None and expected_answer is not None:
            answer_evaluated += 1
            answer_ok = answers_equivalent(final_response, str(expected_answer))
            if answer_ok:
                answer_correct += 1

        rows.append(
            {
                "idx": item.get("idx"),
                "subject": item.get("subject"),
                "expected_domain": expected,
                "route_primary": route.get("primary_domain"),
                "route_candidates": route.get("domain_candidates"),
                "route_ok": route_ok,
                "final_response": final_response,
                "expected_answer": expected_answer,
                "answer_ok": answer_ok,
                "model_calls": model_calls,
                "answer_hint": item.get("answer_hint"),
                "agent_result": result,
            }
        )

    summary = {
        "input_file": str(input_file),
        "total": len(items),
        "route_hits": route_hits,
        "route_accuracy": route_hits / len(items) if items else 0.0,
        "run_agent": run_agent,
        "use_mock": use_mock,
        "valid_outputs": valid_outputs,
        "answer_evaluated": answer_evaluated,
        "answer_correct": answer_correct,
        "answer_accuracy": answer_correct / answer_evaluated if answer_evaluated else None,
        "model_calls": client.total_calls if client is not None else 0,
        "model_calls_per_problem": client.total_calls / len(items) if client is not None and items else 0.0,
        "model_calls_by_role": dict(client.calls_by_role) if client is not None else {},
        "rows": rows,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    _safe_print(
        f"total={summary['total']} "
        f"route_hits={summary['route_hits']} "
        f"route_accuracy={summary['route_accuracy']:.3f} "
        f"run_agent={summary['run_agent']} "
        f"use_mock={summary['use_mock']} "
        f"answer_accuracy={summary['answer_accuracy']} "
        f"model_calls={summary['model_calls']} "
        f"calls_per_problem={summary['model_calls_per_problem']:.3f}"
    )
    for row in summary["rows"]:
        marker = "OK" if row["route_ok"] else "MISS"
        _safe_print(
            f"{marker} {row['idx']}: expected={row['expected_domain']} "
            f"primary={row['route_primary']} candidates={row['route_candidates']} "
            f"calls={row['model_calls']} final={row['final_response'][:80]}"
        )


def _detect_call_role(messages: Any) -> str:
    if not isinstance(messages, list):
        return "unknown"
    system_text = " ".join(
        str(message.get("content", ""))
        for message in messages
        if isinstance(message, dict) and message.get("role") == "system"
    ).lower()
    if "mathematical critic" in system_text:
        return "critic"
    if "mathematical planning agent" in system_text:
        return "planner"
    if "final answer formatter" in system_text:
        return "finalizer"
    return "solver"


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        print(text.encode(encoding, errors="backslashreplace").decode(encoding, errors="replace"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose routing and output quality on hard math cases.")
    parser.add_argument("--input_file", default="sample_data/hard_diagnostics.jsonl")
    parser.add_argument("--output_file", default="sample_outputs/hard_diagnostics_summary.json")
    parser.add_argument("--mock", action="store_true", help="Use MockClient for offline pipeline checks.")
    parser.add_argument("--run-agent", action="store_true", help="Call ReasoningAgent in addition to route checks.")
    parser.add_argument("--no-thinking-mode", action="store_true")
    parser.add_argument("--idx", action="append", help="Run only the case with this idx. Can be repeated.")
    args = parser.parse_args()

    summary = evaluate(
        input_file=BASE_DIR / args.input_file,
        output_file=BASE_DIR / args.output_file,
        use_mock=args.mock,
        run_agent=args.run_agent,
        thinking_mode=not args.no_thinking_mode,
        include_ids=set(args.idx) if args.idx else None,
    )
    print_summary(summary)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from intern_s1_client import InternS1Client
from math_agent_core.clients import MockClient
from math_agent_core.evaluation import grade_full_problem
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
                item = json.loads(line)
                items.append(item)
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
    agent = ReasoningAgent(client=client, thinking_mode=thinking_mode, production_mode="orchestrated") if client is not None else None
    rows = []
    route_hits = 0
    valid_outputs = 0
    answer_evaluated = 0
    answer_correct = 0
    primary_correct = 0
    primary_evaluated = 0
    claims_correct = 0
    claims_evaluated = 0
    full_correct = 0
    full_evaluated = 0
    candidate_agreements = 0
    candidate_conflicts = 0
    candidate_comparisons = 0
    critic_triggers = 0
    candidate_b_triggers = 0
    repair_triggers = 0
    targeted_repairs = 0
    expected_answer_count = 0
    grader_unresolved_count = 0
    required_claim_unresolved_count = 0

    for item in items:
        problem = str(item.get("problem", ""))
        metadata = {
            key: value
            for key, value in item.items()
            if key not in {"problem", "answer", "expected_answer", "grading", "answer_hint"}
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
            run_log = getattr(agent.orchestrator, "last_log", {}) if agent.orchestrator is not None else {}
            comparison = run_log.get("route", {}).get("candidate_comparison", {}) if isinstance(run_log, dict) else {}
            if isinstance(comparison, dict) and comparison.get("agreement") is not None:
                candidate_comparisons += 1
                if comparison.get("agreement"):
                    candidate_agreements += 1
                else:
                    candidate_conflicts += 1
            metrics = run_log.get("route", {}) if isinstance(run_log, dict) else {}
            critic_triggers += int(metrics.get("critic_triggered", 0) or 0)
            candidate_b_triggers += int(metrics.get("candidate_b_triggered", 0) or 0)
            repair_triggers += int(metrics.get("repair_triggered", 0) or 0)
            targeted_repairs += int(metrics.get("targeted_repair_triggered", 0) or 0)
        model_calls = (client.total_calls - calls_before) if client is not None else 0
        # Ground truth is grading-only and is never included in solver metadata.
        expected_answer = item.get("grading", item.get("expected_answer"))
        if expected_answer is None:
            expected_answer = item.get("answer")
        expected_answer_count += int(expected_answer is not None)
        answer_ok = None
        if agent is not None and expected_answer is not None:
            answer_evaluated += 1
            grading = grade_full_problem(final_response, expected_answer)
            primary = grading["primary"]
            answer_ok = primary["correct"]
            primary_evaluated += int(answer_ok is not None)
            primary_correct += int(answer_ok is True)
            claim_items = grading["required_claims"]["claims"]
            claims_evaluated += sum(item["correct"] is not None for item in claim_items)
            claims_correct += sum(item["correct"] is True for item in claim_items)
            required_claim_unresolved_count += sum(item["correct"] is None for item in claim_items)
            full_evaluated += 1
            grader_unresolved_count += int(grading["correct"] is None)
            full_correct += int(grading["correct"] is True)
            if answer_ok is True:
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
                "grading": grading if agent is not None and expected_answer is not None else None,
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
        "primary_answer_accuracy": primary_correct / primary_evaluated if primary_evaluated else None,
        "required_claim_accuracy": claims_correct / claims_evaluated if claims_evaluated else None,
        "full_problem_accuracy": full_correct / full_evaluated if full_evaluated else None,
        "strict_accuracy": full_correct / answer_evaluated if answer_evaluated else None,
        "accuracy_on_gradable_cases": full_correct / (full_evaluated - grader_unresolved_count) if (full_evaluated - grader_unresolved_count) else None,
        "grader_unresolved_count": grader_unresolved_count,
        "required_claim_unresolved_count": required_claim_unresolved_count,
        "grader_unresolved_rate": grader_unresolved_count / full_evaluated if full_evaluated else 0.0,
        "expected_answer_coverage": expected_answer_count / len(items) if items else 0.0,
        "model_calls": client.total_calls if client is not None else 0,
        "model_calls_per_problem": client.total_calls / len(items) if client is not None and items else 0.0,
        "model_calls_by_role": dict(client.calls_by_role) if client is not None else {},
        "candidate_agreement_rate": candidate_agreements / candidate_comparisons if candidate_comparisons else 0.0,
        "candidate_conflict_rate": candidate_conflicts / candidate_comparisons if candidate_comparisons else 0.0,
        "critic_trigger_rate": critic_triggers / len(items) if items else 0.0,
        "candidate_b_trigger_rate": candidate_b_triggers / len(items) if items else 0.0,
        "targeted_repair_rate": targeted_repairs / len(items) if items else 0.0,
        "repair_trigger_rate": repair_triggers / len(items) if items else 0.0,
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
        f"calls_per_problem={summary['model_calls_per_problem']:.3f} "
        f"candidate_agreement_rate={summary['candidate_agreement_rate']:.3f} "
        f"candidate_conflict_rate={summary['candidate_conflict_rate']:.3f} "
        f"candidate_b_trigger_rate={summary['candidate_b_trigger_rate']:.3f} "
        f"critic_trigger_rate={summary['critic_trigger_rate']:.3f} "
        f"repair_trigger_rate={summary['repair_trigger_rate']:.3f} "
        f"targeted_repair_rate={summary['targeted_repair_rate']:.3f} "
        f"grader_unresolved_count={summary['grader_unresolved_count']} "
        f"required_claim_unresolved_count={summary['required_claim_unresolved_count']} "
        f"grader_unresolved_rate={summary['grader_unresolved_rate']:.3f}"
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

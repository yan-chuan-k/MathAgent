"""Run the public hard-problem stress set without leaking its oracle.

This is intentionally a separate benchmark harness.  The problem statement and
non-oracle provenance go to the agent; expected answers stay in this evaluator
and are used only after a live response returns.  ``--mock`` is useful for
pipeline checks, but its answers are explicitly not decision-grade.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

from intern_s1_client import InternS1Client
from math_agent_core.clients import MockClient
from math_agent_core.evaluation.grader import grade_full_problem
from math_agent_core.router import classify_problem
from user_agent import ReasoningAgent


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "sample_data" / "web_hard_stress_v1.jsonl"

_ORACLE_FIELDS = frozenset(
    {
        "expected_domain",
        "expected_subtype",
        "grading",
        "expected_answer",
        "answer",
        "answer_hint",
        "manual_failure_category",
        "manual_failure_notes",
        "gold_summary",
    }
)
_SAFE_METADATA_FIELDS = (
    "idx",
    "source_label",
    "subject",
    "task_type",
    "language",
    "difficulty",
    "concept_tags",
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def safe_solver_metadata(item: Dict[str, Any]) -> Dict[str, Any]:
    """Return caller metadata while proving that oracle fields are absent."""

    metadata = {key: item.get(key) for key in _SAFE_METADATA_FIELDS if key in item}
    if _ORACLE_FIELDS.intersection(metadata):
        raise AssertionError("web stress oracle leaked into solver metadata")
    return metadata


class CountingClient:
    def __init__(self, client: Any) -> None:
        self.client = client
        self.model = getattr(client, "model", "unknown")
        self.total_calls = 0

    def chat(self, *args: Any, **kwargs: Any) -> Any:
        self.total_calls += 1
        call_kwargs = dict(kwargs)
        try:
            signature = inspect.signature(self.client.chat)
            if not any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            ):
                call_kwargs = {
                    key: value
                    for key, value in call_kwargs.items()
                    if key in signature.parameters
                }
        except (TypeError, ValueError):
            pass
        return self.client.chat(*args, **call_kwargs)


def _build_client(*, use_mock: bool, thinking_mode: bool) -> Any:
    if use_mock:
        return MockClient()
    if not os.getenv("INTERN_API_KEY"):
        raise RuntimeError(
            "INTERN_API_KEY is not configured; live model accuracy cannot be measured."
        )

    def env_int(name: str, default: int, minimum: int) -> int:
        raw = os.getenv(name, str(default)).strip()
        try:
            return max(minimum, int(raw))
        except ValueError as exc:
            raise RuntimeError(f"{name} must be an integer") from exc

    return InternS1Client(
        model=os.getenv("INTERN_MODEL", "intern-s2-preview-397b"),
        base_url=os.getenv("INTERN_API_BASE", "https://chat.intern-ai.org.cn/api/v1/"),
        thinking_mode=thinking_mode,
        timeout=env_int("INTERN_API_TIMEOUT", 180, 10),
        retry=env_int("INTERN_API_RETRY", 3, 1),
    )


def _trace_map(result: Any) -> Dict[str, str]:
    trace = result.get("trace") if isinstance(result, dict) else None
    if not isinstance(trace, list):
        return {}
    return {
        str(item.get("step")): str(item.get("content"))
        for item in trace
        if isinstance(item, dict) and item.get("step")
    }


_TRANSPORT_ERROR_MARKERS = (
    "timed out",
    "timeout",
    "connection error",
    "connection refused",
    "network error",
    "temporary failure",
    "rate limit",
    "429",
    "502",
    "503",
    "504",
    "server error",
)


def _classify_solver_error(error: str) -> str:
    """Classify a trace error without treating it as a wrong answer."""

    normalized = str(error or "").strip().lower()
    if not normalized:
        return ""
    if any(marker in normalized for marker in _TRANSPORT_ERROR_MARKERS):
        return "transport_error"
    return "solver_error"


def evaluate(
    items: Iterable[Dict[str, Any]],
    *,
    run_agent: bool,
    use_mock: bool,
    thinking_mode: bool,
) -> Dict[str, Any]:
    rows = list(items)
    route_hits = 0
    route_subtype_hits = 0
    client: CountingClient | None = None
    agent: ReasoningAgent | None = None
    blocked_reason = ""

    if run_agent:
        try:
            client = CountingClient(_build_client(use_mock=use_mock, thinking_mode=thinking_mode))
            agent = ReasoningAgent(client=client, thinking_mode=thinking_mode)
        except Exception as exc:
            blocked_reason = f"{type(exc).__name__}: {str(exc)[:300]}"

    evaluated_numeric = 0
    correct_numeric = 0
    unresolved_numeric = 0
    manual_review = 0
    completed_items = 0
    transport_error_items = 0
    solver_error_items = 0
    rows_out: List[Dict[str, Any]] = []

    for item in rows:
        problem = str(item.get("problem") or "")
        metadata = safe_solver_metadata(item)
        route = classify_problem(problem, metadata)
        domain_ok = route.get("primary_domain") == item.get("expected_domain")
        subtype_ok = (
            not item.get("expected_subtype")
            or route.get("discrete_subtype") == item.get("expected_subtype")
        )
        route_hits += int(domain_ok)
        route_subtype_hits += int(subtype_ok)

        final_response = ""
        grading: Dict[str, Any] | None = None
        trace: Dict[str, str] = {}
        model_calls = 0
        evaluation_status = "not_run"
        error_kind = ""
        model_error = ""
        if agent is not None and client is not None:
            before = client.total_calls
            result = agent.solve(problem, metadata)
            model_calls = client.total_calls - before
            final_response = str(result.get("final_response") or "").strip() if isinstance(result, dict) else ""
            trace = _trace_map(result)
            model_error = str(trace.get("error") or "").strip()
            error_kind = _classify_solver_error(model_error)
            if error_kind:
                evaluation_status = error_kind
                if error_kind == "transport_error":
                    transport_error_items += 1
                else:
                    solver_error_items += 1
            else:
                evaluation_status = "completed"
                completed_items += 1
                grading_spec = item.get("grading") if isinstance(item.get("grading"), dict) else {}
                if grading_spec.get("primary_type") == "numeric":
                    evaluated_numeric += 1
                    grading = grade_full_problem(final_response, grading_spec)
                    if grading.get("correct") is True:
                        correct_numeric += 1
                    elif grading.get("correct") is None:
                        unresolved_numeric += 1
                else:
                    manual_review += 1

        rows_out.append(
            {
                "idx": item.get("idx"),
                "source_label": item.get("source_label"),
                "source": item.get("source"),
                "route_primary": route.get("primary_domain"),
                "route_subtype": route.get("discrete_subtype"),
                "route_domain_match": domain_ok,
                "route_subtype_match": subtype_ok,
                "final_response": final_response,
                "grading": grading,
                "model_calls": model_calls,
                "evaluation_status": evaluation_status,
                "error_kind": error_kind,
                "model_error": model_error,
                "output_truncated_detected": trace.get("output_truncated_detected") == "true",
                "recovery_kept": trace.get("recovery_kept") == "true",
                "correction_kept": trace.get("correction_kept") == "true",
                "trace": trace,
            }
        )

    total = len(rows)
    live_run = bool(agent is not None and not use_mock)
    unmeasured_items = total - completed_items if run_agent and not blocked_reason else total
    numeric_total = sum(
        1
        for item in rows
        if isinstance(item.get("grading"), dict)
        and item["grading"].get("primary_type") == "numeric"
    )
    numeric_unmeasured = max(0, numeric_total - evaluated_numeric)
    if blocked_reason:
        mode = "live_model_blocked"
    elif run_agent and use_mock:
        mode = "mock_pipeline_only"
    elif live_run and unmeasured_items:
        mode = "live_model_partial"
    elif live_run:
        mode = "live_model_decision_grade"
    else:
        mode = "route_only"

    return {
        "benchmark_name": "web_hard_stress_v1",
        "benchmark_sha256": hashlib.sha256(
            DEFAULT_INPUT.read_bytes()
        ).hexdigest() if DEFAULT_INPUT.exists() else "",
        "evaluation_mode": mode,
        "decision_grade": bool(live_run and unmeasured_items == 0),
        "blocked_reason": blocked_reason,
        "run_agent_requested": run_agent,
        "use_mock": use_mock,
        "total": total,
        "route_domain_hits": route_hits,
        "route_domain_accuracy": route_hits / total if total else 0.0,
        "route_subtype_hits": route_subtype_hits,
        "route_subtype_accuracy": route_subtype_hits / total if total else 0.0,
        "numeric_evaluated": evaluated_numeric,
        "numeric_total": numeric_total,
        "numeric_unmeasured": numeric_unmeasured,
        "numeric_correct": correct_numeric,
        "numeric_accuracy": correct_numeric / evaluated_numeric if evaluated_numeric else None,
        "numeric_grader_unresolved": unresolved_numeric,
        "manual_review_items": manual_review,
        "model_completed_items": completed_items,
        "transport_error_items": transport_error_items,
        "solver_error_items": solver_error_items,
        "unmeasured_items": unmeasured_items,
        "rows": rows_out,
    }


def render_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        "# Web Hard Stress Evaluation",
        "",
        "This report uses public competition sources. Gold answers are retained in the evaluator and are not sent to the solver.",
        "",
        f"- Mode: `{summary['evaluation_mode']}`",
        f"- Decision-grade live model run: `{summary['decision_grade']}`",
        f"- Cases: `{summary['total']}`",
        f"- Numeric cases total / evaluated / unmeasured: `{summary.get('numeric_total', 0)} / {summary['numeric_evaluated']} / {summary.get('numeric_unmeasured', 0)}`",
        f"- Numeric correct: `{summary['numeric_correct']}`",
        f"- Manual proof review cases: `{summary['manual_review_items']}`",
        f"- Model-completed cases: `{summary.get('model_completed_items', 0)}`",
        f"- Transport-error cases: `{summary.get('transport_error_items', 0)}`",
        f"- Unmeasured cases: `{summary.get('unmeasured_items', 0)}`",
    ]
    if summary.get("blocked_reason"):
        lines.extend(["", f"**Live run blocked:** {summary['blocked_reason']}"])
    lines.extend(
        [
            "",
            "## Sources",
            "",
            "- [HMMT February 2025 Algebra and Number Theory problems](https://hmmt-archive.s3.amazonaws.com/tournaments/2025/feb/algnt/problems.pdf) and [official solutions](https://hmmt-archive.s3.amazonaws.com/tournaments/2025/feb/algnt/solutions.pdf)",
            "- [HMMT February 2025 Combinatorics problems](https://hmmt-archive.s3.amazonaws.com/tournaments/2025/feb/comb/problems.pdf) and [official solutions](https://hmmt-archive.s3.amazonaws.com/tournaments/2025/feb/comb/solutions.pdf)",
            "- [HMMT February 2025 Geometry problems](https://hmmt-archive.s3.amazonaws.com/tournaments/2025/feb/geo/problems.pdf) and [official solutions](https://hmmt-archive.s3.amazonaws.com/tournaments/2025/feb/geo/solutions.pdf)",
            "- [IMO 2025 official problems](https://www.imo-official.org/assets/documents/problems/2025/2025_eng.pdf)",
            "",
            "## Case results",
            "",
            "| Case | Route | Model answer / state | Status | Grading | Truncation |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in summary.get("rows", []):
        answer = row.get("final_response") or "(not run)"
        answer = str(answer).replace("\n", " ")[:180]
        grading = row.get("grading")
        grade_state = "manual/none"
        if isinstance(grading, dict):
            grade_state = str(grading.get("correct"))
        status = str(row.get("evaluation_status") or "not_run")
        if status != "completed":
            grade_state = "not measured"
        lines.append(
            f"| {row.get('idx')} | {row.get('route_primary')} / {row.get('route_subtype') or '-'} | {answer} | {status} | {grade_state} | {row.get('output_truncated_detected')} -> recovery {row.get('recovery_kept')} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Public cross-subject hard math stress evaluator")
    parser.add_argument("--input_file", default=str(DEFAULT_INPUT))
    parser.add_argument("--output_json", default="sample_outputs/web_hard_stress_v1.json")
    parser.add_argument("--output_md", default="sample_outputs/web_hard_stress_v1.md")
    parser.add_argument("--run-agent", action="store_true", help="Attempt a real model run")
    parser.add_argument("--mock", action="store_true", help="Pipeline-only mock run; never decision-grade")
    parser.add_argument("--no-thinking-mode", action="store_true")
    parser.add_argument("--idx", action="append")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    input_path = Path(args.input_file)
    items = load_jsonl(input_path)
    if args.idx:
        selected = set(args.idx)
        items = [item for item in items if str(item.get("idx")) in selected]
    if args.limit is not None:
        items = items[: max(0, args.limit)]

    summary = evaluate(
        items,
        run_agent=args.run_agent,
        use_mock=args.mock,
        thinking_mode=not args.no_thinking_mode,
    )
    # Preserve the exact selected-set hash in addition to the default fixture
    # hash so a partial run is still auditable.
    summary["input_file"] = str(input_path)
    summary["selected_ids"] = [str(item.get("idx")) for item in items]
    summary["selected_sha256"] = hashlib.sha256(
        input_path.read_bytes()
    ).hexdigest()

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in (
        "evaluation_mode", "decision_grade", "total", "numeric_evaluated",
        "numeric_total", "numeric_unmeasured", "numeric_correct",
        "numeric_grader_unresolved", "model_completed_items",
        "transport_error_items", "solver_error_items", "unmeasured_items",
        "blocked_reason",
    )}, ensure_ascii=False, indent=2))
    return 0 if not summary.get("blocked_reason") else 2


if __name__ == "__main__":
    raise SystemExit(main())

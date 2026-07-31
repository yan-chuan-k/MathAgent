import argparse
import json
import time
from pathlib import Path

from intern_s1_client import InternS1Client
from math_agent_core import MathAgentOrchestrator
from math_agent_core.clients import MockClient


BASE_DIR = Path(__file__).resolve().parent


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run math problems from JSONL.")
    parser.add_argument("--input", default="problems.jsonl")
    parser.add_argument("--output", default="results.jsonl")
    parser.add_argument("--invalid-output", default="invalid_results.jsonl")
    parser.add_argument("--summary", default="validation_summary.json")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--backend", choices=["simple", "lagent"], default="simple")
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mock", action="store_true", help="Use MockClient instead of Intern-S1 API.")
    return parser


def resolve_path(path):
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = BASE_DIR / resolved
    return resolved


def read_jsonl(file_path):
    problems = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                problems.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"line {line_number} input JSON parse failed: {exc}")
    return problems


def append_jsonl(file_path, data):
    with open(file_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(data, ensure_ascii=False) + "\n")


def save_json(file_path, data):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def read_completed_ids(output_path):
    completed = set()
    if not output_path.exists():
        return completed
    with open(output_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                completed.add(json.loads(line).get("problem_id"))
            except json.JSONDecodeError:
                continue
    return completed


def build_client(use_mock):
    if use_mock:
        return MockClient()
    return InternS1Client(model="intern-s1")


def main():
    args = build_arg_parser().parse_args()
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    invalid_output_path = resolve_path(args.invalid_output)
    summary_path = resolve_path(args.summary)
    log_dir = resolve_path(args.log_dir)
    log_dir.mkdir(exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"problem file not found: {input_path}")

    problems = read_jsonl(input_path)
    completed_ids = read_completed_ids(output_path) if args.resume else set()
    if not args.resume:
        output_path.write_text("", encoding="utf-8")
        invalid_output_path.write_text("", encoding="utf-8")

    agent = MathAgentOrchestrator(
        client=build_client(args.mock),
        max_retries=args.max_retries,
        enable_repair=True,
        enable_tool_verify=True,
        backend=args.backend,
        schema_path=BASE_DIR / "result_schema.json",
    )

    total = len(problems)
    valid_count = 0
    invalid_count = 0
    failed_count = 0
    skipped_count = 0

    print(f"loaded {total} problems; starting batch")

    for index, problem in enumerate(problems, start=1):
        problem_id = str(problem.get("problem_id") or f"UNKNOWN_{index}")
        if args.resume and problem_id in completed_ids:
            skipped_count += 1
            print(f"[{index}/{total}] skip completed: {problem_id}")
            continue

        print(f"[{index}/{total}] solving: {problem_id}")
        try:
            result = agent.solve(problem)
            schema_valid = result.get("_meta", {}).get("schema_valid", False)
            if schema_valid:
                valid_count += 1
                append_jsonl(output_path, result)
                save_json(log_dir / f"{problem_id}.json", agent.last_log)
                print(f"completed and schema-valid: {problem_id}")
            else:
                invalid_count += 1
                append_jsonl(invalid_output_path, result)
                save_json(log_dir / f"{problem_id}_invalid.json", agent.last_log)
                print(f"completed but schema-invalid: {problem_id}")
                print(f"reason: {result.get('_meta', {}).get('schema_error')}")
        except Exception as exc:
            failed_count += 1
            error_result = {
                "problem_id": problem_id,
                "status": "failed",
                "error": str(exc),
            }
            append_jsonl(invalid_output_path, error_result)
            save_json(log_dir / f"{problem_id}_error.json", error_result)
            print(f"failed: {problem_id}")
            print(exc)

        if args.sleep > 0:
            time.sleep(args.sleep)

    summary = {
        "total": total,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "failed_count": failed_count,
        "skipped_count": skipped_count,
        "valid_rate": valid_count / total if total > 0 else 0,
    }
    save_json(summary_path, summary)

    print("batch finished")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

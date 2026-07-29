import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from intern_s1_client import InternS1Client
from math_agent_core import MathAgentOrchestrator
from math_agent_core.clients import MockClient
from user_agent import ReasoningAgent


BASE_DIR = Path(__file__).resolve().parent


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run math agent locally.")
    parser.add_argument("--input_file", help="Official baseline JSONL input file.")
    parser.add_argument("--output_dir", help="Official baseline output directory.")
    parser.add_argument("--input", default="input.json", help="Legacy single-problem input JSON file.")
    parser.add_argument("--output", default="result.json", help="Legacy single-problem output JSON file.")
    parser.add_argument("--backend", choices=["simple", "lagent"], default="simple")
    parser.add_argument("--mock", action="store_true", help="Use MockClient instead of local Intern-S API.")
    parser.add_argument("--max-retries", type=int, default=1)
    return parser


def resolve_path(path):
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = BASE_DIR / resolved
    return resolved


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_jsonl(path) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                items.append(
                    {
                        "idx": f"line_{line_number}",
                        "problem": "",
                        "_load_error": f"JSONDecodeError: {exc}",
                    }
                )
    return items


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def build_client(use_mock):
    if use_mock:
        return MockClient()
    model = os.getenv("INTERN_MODEL", "intern-s1")
    base_url = os.getenv("INTERN_API_BASE", "https://chat.intern-ai.org.cn/api/v1/")
    return InternS1Client(model=model, base_url=base_url)


def run_baseline(args) -> None:
    input_path = resolve_path(args.input_file)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    items = load_jsonl(input_path)
    max_workers = max(1, int(os.getenv("LOCAL_MAX_CONCURRENCY", "8")))
    max_workers = min(max_workers, max(1, len(items)))

    def run_one(item: Dict[str, Any]) -> Dict[str, Any]:
        idx = item.get("idx", item.get("id", item.get("problem_id", "unknown")))
        output_path = output_dir / f"{idx}.json"
        if output_path.exists() and output_path.stat().st_size > 0:
            return {"idx": idx, "status": "skipped", "path": str(output_path)}
        if item.get("_load_error"):
            result = {
                "idx": idx,
                "status": "error",
                "final_response": "",
                "error": {"type": "JSONDecodeError", "message": item["_load_error"]},
                "trace": [],
            }
            save_json(output_path, result)
            return {"idx": idx, "status": "error", "path": str(output_path)}

        problem_text = str(item.get("problem", item.get("problem_text", "")))
        metadata = {key: value for key, value in item.items() if key not in {"problem", "problem_text", "answer"}}

        try:
            agent = ReasoningAgent(client=build_client(args.mock), max_retries=args.max_retries)
            solved = agent.solve(problem_text, metadata)
            result = {
                "idx": idx,
                "status": "success",
                "final_response": str(solved.get("final_response", "")).strip(),
                "trace": solved.get("trace", []),
            }
        except Exception as exc:
            result = {
                "idx": idx,
                "status": "error",
                "final_response": "",
                "error": {"type": type(exc).__name__, "message": str(exc)[:300]},
                "trace": [],
            }
        save_json(output_path, result)
        return {"idx": idx, "status": result["status"], "path": str(output_path)}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_one, item) for item in items]
        for future in as_completed(futures):
            info = future.result()
            print(f"{info['status']}: idx={info['idx']} path={info['path']}")


def run_legacy_single(args) -> None:
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    problem = load_json(input_path)
    agent = MathAgentOrchestrator(
        client=build_client(args.mock),
        max_retries=args.max_retries,
        enable_repair=True,
        enable_tool_verify=True,
        backend=args.backend,
        schema_path=BASE_DIR / "result_schema.json",
    )

    result = agent.solve(problem)
    save_json(output_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(
        f"completed problem_id={result.get('problem_id')} "
        f"schema_valid={result.get('_meta', {}).get('schema_valid')} "
        f"output={output_path}"
    )


def main():
    args = build_arg_parser().parse_args()
    if args.input_file or args.output_dir:
        if not args.input_file or not args.output_dir:
            raise ValueError("--input_file and --output_dir must be provided together")
        run_baseline(args)
        return
    run_legacy_single(args)


if __name__ == "__main__":
    main()

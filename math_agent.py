from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from intern_s1_client import InternS1Client
from math_agent_core.clients import MockClient
from user_agent import ReasoningAgent


DEFAULT_MODEL = "intern-s2-preview-397b"


class MathAgent:
    """Legacy compatibility wrapper around the official ReasoningAgent."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        client: Any = None,
        mock: bool = False,
        model: str = DEFAULT_MODEL,
    ):
        if client is not None:
            self.client = client
        elif mock:
            self.client = MockClient()
        else:
            self.client = InternS1Client(api_key=api_key, model=model)
        self.agent = ReasoningAgent(client=self.client)

    def solve_math_problem(self, problem_id: str, problem_text: str) -> Dict[str, Any]:
        result = self.agent.solve(problem_text, {"problem_id": problem_id})
        return {
            "problem_id": problem_id,
            "problem_text": problem_text,
            "final_response": result.get("final_response", "无法确定"),
            "trace": result.get("trace", []),
            "metadata": {
                "model": getattr(self.client, "model", DEFAULT_MODEL),
                "agent_type": "legacy_wrapper",
                "version": "v2.0",
            },
        }


def _read_problem_from_file(path: str) -> Dict[str, Any]:
    file_path = Path(path)
    data = json.loads(file_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data
    raise ValueError("input file must contain one JSON object")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the legacy MathAgent wrapper.")
    parser.add_argument("--problem", help="Problem text to solve.")
    parser.add_argument("--problem-id", default="MANUAL_001")
    parser.add_argument("--input", help="JSON file with problem/problem_text and optional id.")
    parser.add_argument("--mock", action="store_true", help="Use offline MockClient.")
    parser.add_argument("--model", default=os.getenv("INTERN_MODEL", DEFAULT_MODEL))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.input:
        item = _read_problem_from_file(args.input)
        problem_text = str(item.get("problem") or item.get("problem_text") or "")
        problem_id = str(item.get("problem_id") or item.get("idx") or args.problem_id)
    else:
        problem_text = args.problem or ""
        problem_id = args.problem_id

    if not problem_text.strip():
        print(
            "math_agent.py is importable and ready. "
            "Pass --problem \"1+1=?\" --mock for an offline smoke run, "
            "or configure INTERN_API_KEY for real API calls."
        )
        return

    agent = MathAgent(mock=args.mock, model=args.model)
    result = agent.solve_math_problem(problem_id, problem_text)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

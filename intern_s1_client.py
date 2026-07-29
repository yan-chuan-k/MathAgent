import json
import os
import time
from typing import Any, Dict, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class InternS1Client:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "intern-s2-preview-397b",
        base_url: str = "https://chat.intern-ai.org.cn/api/v1/",
        temperature: float = 0.1,
        max_tokens: int = 8192,
        thinking_mode: bool = True,
        timeout: int = 120,
        retry: int = 3,
    ):
        if load_dotenv is not None:
            load_dotenv()

        self.api_key = api_key or os.getenv("INTERN_API_KEY")
        if not self.api_key:
            raise ValueError("INTERN_API_KEY is required when not using --mock")
        if OpenAI is None:
            raise ImportError("openai package is required when not using --mock")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking_mode = thinking_mode
        self.retry = retry
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def chat(
        self,
        messages,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        thinking_mode: Optional[bool] = None,
    ) -> str:
        last_error = None
        for attempt in range(1, self.retry + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature if temperature is None else temperature,
                    max_tokens=self.max_tokens if max_tokens is None else max_tokens,
                    extra_body={
                        "thinking_mode": self.thinking_mode if thinking_mode is None else thinking_mode
                    },
                )
                content = response.choices[0].message.content
                if not content:
                    raise RuntimeError("Intern-S1 returned empty content")
                return content
            except Exception as exc:
                last_error = exc
                if attempt >= self.retry:
                    break
                time.sleep(min(2 ** (attempt - 1), 8))
        raise RuntimeError(f"Intern-S1 request failed after {self.retry} attempts: {last_error}")

    def solve_math_problem(self, problem_text: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a rigorous math agent. Plan the solution, solve step by step, "
                    "and give a verifiable final answer."
                ),
            },
            {"role": "user", "content": problem_text},
        ]
        return self.chat(messages)

    def solve_math_problem_json(self, problem_id: str, problem_text: str) -> Dict[str, Any]:
        prompt = {
            "problem_id": problem_id,
            "problem_text": problem_text,
            "output_requirement": "Return strict JSON only, without Markdown fences.",
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a math agent. Return one parseable JSON object only. "
                    "Do not output Markdown or text outside JSON."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]
        content = self.chat(messages, temperature=0.1, max_tokens=8192, thinking_mode=True)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {
                "problem_id": problem_id,
                "raw_output": content,
                "json_parse_error": True,
            }

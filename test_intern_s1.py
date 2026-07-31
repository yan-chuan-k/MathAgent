import json
import os

import pytest

from intern_s1_client import InternS1Client
from math_agent_core import MathAgentOrchestrator
from math_agent_core.clients import MockClient


def test_mock_orchestrator_returns_schema_valid_result():
    agent = MathAgentOrchestrator(client=MockClient())
    result = agent.solve(
        {
            "problem_id": "LA_001",
            "problem_text": "A=[[1,2],[3,4]], find det(A).",
        }
    )

    assert result["problem_id"] == "LA_001"
    assert result["_meta"]["schema_valid"] is True
    assert result["problem_type"] == "linear_algebra"
    assert result["final_answer"]["answer"]


@pytest.mark.skipif(
    not os.getenv("INTERN_API_KEY"),
    reason="INTERN_API_KEY is not configured",
)
def test_intern_s1_connectivity():
    pytest.importorskip("openai")
    client = InternS1Client(model="intern-s1", max_tokens=512)
    content = client.chat(
        [
            {"role": "system", "content": "Answer briefly."},
            {"role": "user", "content": "What is 1+1?"},
        ],
        temperature=0.2,
        max_tokens=512,
        thinking_mode=True,
    )

    assert isinstance(content, str)
    assert content.strip()


if __name__ == "__main__":
    result = MathAgentOrchestrator(client=MockClient()).solve(
        {
            "problem_id": "LA_001",
            "problem_text": "A=[[1,2],[3,4]], find det(A).",
        }
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

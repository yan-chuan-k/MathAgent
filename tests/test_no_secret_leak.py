import json

from user_agent import ReasoningAgent


class MockClient:
    def chat(self, messages, temperature=0.2, max_tokens=4096):
        return "final answer: 2"


def test_output_does_not_contain_secret_markers():
    agent = ReasoningAgent(client=MockClient())
    result = agent.solve("1+1=?", {"idx": 0})
    serialized = json.dumps(result, ensure_ascii=False)

    assert "sk-" not in serialized
    assert "INTERN_API_KEY" not in serialized
    assert "api_key" not in serialized

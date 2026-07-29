import json

from user_agent import ReasoningAgent


class MockClient:
    def chat(self, messages, temperature=0.2, max_tokens=4096):
        return "2"


def test_user_agent_entry_returns_final_response():
    agent = ReasoningAgent(client=MockClient())
    result = agent.solve("1+1=?", {"idx": 0})

    assert isinstance(result, dict)
    assert isinstance(result.get("final_response"), str)
    assert result["final_response"].strip()


def test_user_agent_result_is_json_serializable():
    agent = ReasoningAgent(client=MockClient())
    result = agent.solve("1+1=?", {"idx": 0})

    json.dumps(result, ensure_ascii=False)
